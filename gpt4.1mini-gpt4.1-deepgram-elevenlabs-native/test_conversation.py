"""Conversation simulator — tests dual-LLM voice agent via direct WebSocket.

Connects to the agent's /ws endpoint, simulates a Plivo caller by sending
TTS-generated audio (via ElevenLabs), records the full two-party conversation,
and saves it as a playable WAV file.

Conversation script exercises:
  1. Simple greeting (mini responds directly)
  2. Product interest + company details (conversational turn)
  3. CRM lookup request (triggers delegate_to_reasoning → HubSpot)
  4. Barge-in mid-response (tests interruption handling)
  5. BANT details (budget, timeline — may trigger lead scoring)
  6. Meeting request (triggers delegate_to_reasoning → Cal.com)
  7. Goodbye (conversational close)

Usage:
    # Start the inbound server first, then:
    uv run python test_conversation.py
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import wave
from pathlib import Path

import httpx
import numpy as np
import websockets
from dotenv import load_dotenv
from scipy import signal as scipy_signal

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

AGENT_WS_URL = os.getenv("TEST_WS_URL", "ws://localhost:8000/ws")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
# Use a distinct voice for the caller (Adam) vs agent (Rachel)
CALLER_VOICE_ID = os.getenv("TEST_CALLER_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
CALLER_MODEL_ID = os.getenv("TEST_CALLER_MODEL_ID", "eleven_flash_v2_5")

OUTPUT_DIR = Path(__file__).parent / "test_recordings"
PLIVO_SAMPLE_RATE = 8000
ELEVENLABS_SAMPLE_RATE = 24000
PLIVO_CHUNK_SIZE = 160  # 20ms at 8kHz μ-law
CHUNK_INTERVAL = 0.02  # 20ms between chunks (real-time pacing)
SILENCE_TIMEOUT = 2.0  # seconds of no agent audio → agent done speaking
BARGE_IN_DELAY = 1.5  # seconds into agent response before barge-in

# Conversation script: (label, utterance, is_barge_in)
CONVERSATION_SCRIPT: list[tuple[str, str, bool]] = [
    (
        "greeting",
        "Hi, I'm interested in learning about TechFlow's workflow automation platform.",
        False,
    ),
    (
        "company_details",
        "We're a fifty person marketing agency struggling with project tracking across teams.",
        False,
    ),
    (
        "crm_lookup",
        "Could you look up my account? My email is sarah at brightwave dot io.",
        False,
    ),
    (
        "barge_in",
        "Actually, never mind the lookup. Let me tell you more about what we need.",
        True,  # Send this while agent is still responding
    ),
    (
        "bant_details",
        "Our budget is around ten thousand per quarter, and we need something by next month. "
        "I'm the VP of operations so I make the final call.",
        False,
    ),
    (
        "schedule_meeting",
        "That sounds great. Can you set up a demo for me next Tuesday afternoon?",
        False,
    ),
    (
        "goodbye",
        "Perfect, thanks so much. Have a great day!",
        False,
    ),
]


# =============================================================================
# Audio Utilities (standalone — no dependency on agent's utils.py)
# =============================================================================

# μ-law decode table (ITU-T G.711)
_ULAW_DECODE_TABLE = np.array(
    [
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
        -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
        -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
        -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
        -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
        -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
        -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
        -876, -844, -812, -780, -748, -716, -684, -652,
        -620, -588, -556, -524, -492, -460, -428, -396,
        -372, -356, -340, -324, -308, -292, -276, -260,
        -244, -228, -212, -196, -180, -164, -148, -132,
        -120, -112, -104, -96, -88, -80, -72, -64,
        -56, -48, -40, -32, -24, -16, -8, 0,
        32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
        23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
        15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
        11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
        7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
        5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
        3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
        2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
        1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
        1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
        876, 844, 812, 780, 748, 716, 684, 652,
        620, 588, 556, 524, 492, 460, 428, 396,
        372, 356, 340, 324, 308, 292, 276, 260,
        244, 228, 212, 196, 180, 164, 148, 132,
        120, 112, 104, 96, 88, 80, 72, 64,
        56, 48, 40, 32, 24, 16, 8, 0,
    ],
    dtype=np.int16,
)


def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    samples = np.frombuffer(ulaw_data, dtype=np.uint8)
    return _ULAW_DECODE_TABLE[samples].tobytes()


def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    BIAS = 0x84
    CLIP = 32635
    pcm = np.frombuffer(pcm_data, dtype=np.int16).astype(np.int32)
    sign = (pcm >> 8) & 0x80
    pcm = np.where(sign != 0, -pcm, pcm)
    pcm = np.clip(pcm, 0, CLIP) + BIAS
    segment = np.floor(np.log2(pcm >> 7)).astype(np.int32)
    segment = np.clip(segment, 0, 7)
    ulaw = sign | ((segment << 4) | ((pcm >> (segment + 3)) & 0x0F))
    ulaw = ~ulaw & 0xFF
    return ulaw.astype(np.uint8).tobytes()


def resample(audio: bytes, in_rate: int, out_rate: int) -> bytes:
    if in_rate == out_rate:
        return audio
    samples = np.frombuffer(audio, dtype=np.int16)
    if len(samples) == 0:
        return audio
    new_len = int(len(samples) * out_rate / in_rate)
    resampled = scipy_signal.resample(samples.astype(np.float64), new_len)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def pcm24k_to_ulaw8k(pcm_24k: bytes) -> bytes:
    """ElevenLabs PCM 24kHz → μ-law 8kHz (Plivo format)."""
    pcm_8k = resample(pcm_24k, ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
    return pcm_to_ulaw(pcm_8k)


def generate_silence_ulaw(duration_ms: int) -> bytes:
    """Generate μ-law silence (used for gaps between turns)."""
    n_bytes = int(PLIVO_SAMPLE_RATE * duration_ms / 1000)
    # μ-law silence = 0xFF
    return bytes([0xFF] * n_bytes)


# =============================================================================
# TTS — Generate Caller Audio
# =============================================================================


async def synthesize_caller_audio(text: str) -> bytes:
    """Generate caller utterance via ElevenLabs, return as μ-law 8kHz bytes."""
    print(f"  [tts] Synthesizing: '{text[:60]}...'")
    t0 = time.monotonic()

    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech"
        f"/{CALLER_VOICE_ID}/stream"
        f"?output_format=pcm_24000"
    )
    pcm_chunks: list[bytes] = []
    async with httpx.AsyncClient(timeout=30.0) as client, client.stream(
        "POST",
        url,
        headers={
            "xi-api-key": ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
        },
        json={"text": text, "model_id": CALLER_MODEL_ID},
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes(chunk_size=4800):
            pcm_chunks.append(chunk)

    pcm_24k = b"".join(pcm_chunks)
    ulaw_8k = pcm24k_to_ulaw8k(pcm_24k)
    elapsed = (time.monotonic() - t0) * 1000
    duration = len(ulaw_8k) / PLIVO_SAMPLE_RATE
    print(f"  [tts] Done: {duration:.1f}s audio in {elapsed:.0f}ms")
    return ulaw_8k


# =============================================================================
# Conversation Recorder
# =============================================================================


class ConversationRecorder:
    """Records both sides of the conversation as PCM 8kHz mono."""

    def __init__(self):
        self._pcm_frames: list[bytes] = []

    def add_caller_audio(self, ulaw_data: bytes) -> None:
        """Record caller audio (what the simulated person says)."""
        self._pcm_frames.append(ulaw_to_pcm(ulaw_data))

    def add_agent_audio(self, ulaw_data: bytes) -> None:
        """Record agent audio (what the voice agent says)."""
        self._pcm_frames.append(ulaw_to_pcm(ulaw_data))

    def add_silence(self, duration_ms: int) -> None:
        """Insert silence gap between turns."""
        n_samples = int(PLIVO_SAMPLE_RATE * duration_ms / 1000)
        self._pcm_frames.append(b"\x00\x00" * n_samples)

    def save(self, path: Path) -> float:
        """Save recorded conversation as WAV. Returns duration in seconds."""
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = b"".join(self._pcm_frames)
        duration = len(pcm) / (PLIVO_SAMPLE_RATE * 2)  # 2 bytes per sample

        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(PLIVO_SAMPLE_RATE)
            wf.writeframes(pcm)

        print(f"\n  Recording saved: {path} ({duration:.1f}s)")
        return duration


# =============================================================================
# WebSocket Conversation Driver
# =============================================================================


async def run_conversation() -> Path:
    """Connect to agent, run conversation script, return path to recording."""
    recorder = ConversationRecorder()

    # Phase 1: Pre-generate all caller utterances
    print("\n=== Phase 1: Generating caller audio ===\n")
    caller_audio: list[bytes] = []
    for label, text, _ in CONVERSATION_SCRIPT:
        print(f"  [{label}]")
        audio = await synthesize_caller_audio(text)
        caller_audio.append(audio)

    # Phase 2: Connect and run conversation
    print("\n=== Phase 2: Running conversation ===\n")

    # Build body param (simulating Plivo's call metadata)
    call_data = {
        "call_uuid": f"test-{int(time.time())}",
        "from": "14155551234",
        "to": "13305263709",
    }
    body_b64 = base64.b64encode(json.dumps(call_data).encode()).decode()
    ws_url = f"{AGENT_WS_URL}?body={body_b64}"

    print(f"  Connecting to {AGENT_WS_URL}")
    async with websockets.connect(
        ws_url, max_size=10 * 1024 * 1024, close_timeout=5,
    ) as ws:
        # Send Plivo start event
        start_event = {
            "event": "start",
            "start": {
                "callId": call_data["call_uuid"],
                "streamId": f"stream-{int(time.time())}",
            },
        }
        await ws.send(json.dumps(start_event))
        print(f"  Sent start event (callId={call_data['call_uuid']})")

        # Wait for agent greeting
        print("\n  --- Waiting for agent greeting ---")
        agent_audio = await collect_agent_response(ws, recorder, timeout=15.0)
        print(f"  Agent greeting: {len(agent_audio)} bytes ({len(agent_audio)/8000:.1f}s)")

        # Run through conversation script
        for i, (label, text, is_barge_in) in enumerate(CONVERSATION_SCRIPT):
            recorder.add_silence(500)  # 500ms gap between turns
            print(f"\n  --- Turn {i + 1}: [{label}] ---")
            print(f"  Caller: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")

            if is_barge_in:
                # For barge-in: send caller audio while agent may still be
                # responding from the previous turn. The agent should detect
                # speech_started, cancel its response, and listen.
                print("  (barge-in test)")
                recorder.add_caller_audio(caller_audio[i])
                await send_caller_audio(ws, caller_audio[i])

                # Now wait for agent to respond to the barge-in content
                agent_audio = await collect_agent_response(
                    ws, recorder, timeout=20.0,
                )
            else:
                # Normal turn: send caller audio, then wait for response
                recorder.add_caller_audio(caller_audio[i])
                await send_caller_audio(ws, caller_audio[i])

                # Wait longer for delegated turns (LLM reasoning + TTS)
                agent_audio = await collect_agent_response(
                    ws, recorder, timeout=25.0,
                )

            print(f"  Agent response: {len(agent_audio)} bytes ({len(agent_audio)/8000:.1f}s)")

        # Send stop event
        recorder.add_silence(500)
        try:
            await ws.send(json.dumps({"event": "stop"}))
            print("\n  Sent stop event — conversation complete")
        except websockets.ConnectionClosed:
            print("\n  WebSocket already closed — conversation complete")

    # Phase 3: Save recording
    print("\n=== Phase 3: Saving recording ===")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"conversation_{timestamp}.wav"
    recorder.save(output_path)

    return output_path


async def send_caller_audio(ws, ulaw_audio: bytes, send_trailing_silence: bool = True) -> None:
    """Send caller audio through WebSocket in 160-byte chunks at real-time pace.

    Also sends trailing silence (600ms) so VAD detects speech_ended. Without
    this silence gap, back-to-back audio is treated as one continuous utterance.
    """
    offset = 0
    chunks_sent = 0
    try:
        while offset < len(ulaw_audio):
            chunk = ulaw_audio[offset : offset + PLIVO_CHUNK_SIZE]
            if len(chunk) < PLIVO_CHUNK_SIZE:
                chunk = chunk + bytes([0xFF] * (PLIVO_CHUNK_SIZE - len(chunk)))

            media_event = {
                "event": "media",
                "media": {"payload": base64.b64encode(chunk).decode()},
            }
            await ws.send(json.dumps(media_event))
            chunks_sent += 1
            offset += PLIVO_CHUNK_SIZE
            await asyncio.sleep(CHUNK_INTERVAL)

        # Send trailing silence so VAD detects end-of-speech.
        # 600ms of silence > VAD_MIN_SILENCE_MS (300ms), ensuring speech_ended fires.
        if send_trailing_silence:
            silence = generate_silence_ulaw(600)
            sil_offset = 0
            while sil_offset < len(silence):
                chunk = silence[sil_offset : sil_offset + PLIVO_CHUNK_SIZE]
                if len(chunk) < PLIVO_CHUNK_SIZE:
                    chunk = chunk + bytes([0xFF] * (PLIVO_CHUNK_SIZE - len(chunk)))
                media_event = {
                    "event": "media",
                    "media": {"payload": base64.b64encode(chunk).decode()},
                }
                await ws.send(json.dumps(media_event))
                sil_offset += PLIVO_CHUNK_SIZE
                await asyncio.sleep(CHUNK_INTERVAL)

    except websockets.ConnectionClosed:
        print(f"  (connection closed after {chunks_sent} chunks)")
        return

    print(f"  Sent {chunks_sent} audio chunks ({len(ulaw_audio)/8000:.1f}s + 600ms silence)")


async def collect_agent_response(
    ws,
    recorder: ConversationRecorder,
    timeout: float = 10.0,
) -> bytes:
    """Collect agent audio until silence timeout. Returns all μ-law bytes."""
    agent_bytes = bytearray()
    last_audio_time = time.monotonic()
    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        silence = time.monotonic() - last_audio_time

        # If we've received some audio and there's been a silence gap, we're done
        if len(agent_bytes) > 0 and silence > SILENCE_TIMEOUT:
            break

        # Absolute timeout (agent might never respond)
        if elapsed > timeout:
            if len(agent_bytes) == 0:
                print("  (timeout: no agent response)")
            break

        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
            message = json.loads(raw)
            event = message.get("event")

            if event == "playAudio":
                payload = message.get("media", {}).get("payload", "")
                if payload:
                    audio = base64.b64decode(payload)
                    agent_bytes.extend(audio)
                    recorder.add_agent_audio(audio)
                    last_audio_time = time.monotonic()

            elif event == "clearAudio":
                print("  (agent sent clearAudio — barge-in acknowledged)")

        except asyncio.TimeoutError:
            continue
        except websockets.ConnectionClosed:
            print("  (WebSocket closed)")
            break

    return bytes(agent_bytes)


# =============================================================================
# Main
# =============================================================================


async def main() -> None:
    print("=" * 60)
    print("  Dual-LLM Voice Agent — Conversation Simulator")
    print("=" * 60)

    if not ELEVENLABS_API_KEY:
        print("\nERROR: ELEVENLABS_API_KEY not set in .env")
        return

    t0 = time.monotonic()
    output_path = await run_conversation()
    elapsed = time.monotonic() - t0

    print(f"\n  Total test time: {elapsed:.1f}s")
    print("\n  To play the recording:")
    print(f"    afplay {output_path}")
    print(f"    # or: open {output_path}")

    # Auto-play on macOS
    print("\n  Playing recording...")
    proc = await asyncio.create_subprocess_exec("afplay", str(output_path))
    await proc.wait()


if __name__ == "__main__":
    asyncio.run(main())
