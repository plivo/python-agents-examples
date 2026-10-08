"""
End-to-end live tests for the GPT-4o + Modulate Velma-2 + Cartesia Sonic 3 Pipecat agent.

These tests:
1. Start the inbound server as a subprocess (logs to a file)
2. Fetch the <Stream> URL from a Plivo-signed /answer webhook
3. Connect to /ws, simulating Plivo: start event, then μ-law silence every 20ms
4. Receive the agent's opening line (LLM-generated; inbound has no scripted greeting)
5. Transcribe it locally with faster-whisper and check it reads like a greeting

No phone call is placed and the server gets no Plivo account or number, so it cannot
touch a real Plivo number.

Requirements:
    - OPENAI_API_KEY, MODULATE_API_KEY and CARTESIA_API_KEY in .env
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (PATH, FFMPEG_DIR, or a parent directory)
    - Port 18005 available (used by the test server)

Usage:
    uv run pytest tests/test_e2e_live.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import struct
import tempfile
import time
import uuid
import wave

import pytest
import websockets
from dotenv import load_dotenv

from tests.helpers import (
    LIVE_API_KEYS,
    TEST_AUTH_TOKEN,
    ensure_ffmpeg_on_path,
    local_server_env,
    log_tail,
    missing_env,
    server_log_path,
    signed_webhook,
    start_server,
    stop_server,
    stream_signature_headers,
    stream_url_from_xml,
    ulaw_to_pcm,
)

load_dotenv()
ensure_ffmpeg_on_path()

TEST_PORT = 18005
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"
LOG_PATH = server_log_path("modulate_e2e_live_server")

# The opening line is the LLM's reply to "Hello, I'm calling for help." under
# inbound/system_prompt.md, so the wording varies; these are what a greeting contains.
GREETING_WORDS = ["hello", "hi", "help", "assist", "how can", "what can", "welcome"]

pytestmark = pytest.mark.skipif(
    bool(missing_env(*LIVE_API_KEYS)),
    reason=f"not configured: {', '.join(missing_env(*LIVE_API_KEYS))}",
)


# =============================================================================
# Helpers
# =============================================================================


def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def transcribe_ulaw(ulaw_audio: bytes) -> str:
    """Transcribe μ-law 8kHz audio locally using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(pcm16_to_wav(ulaw_to_pcm(ulaw_audio)))
        tmp_path = f.name
    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


def compute_audio_rms(ulaw_audio: bytes) -> float:
    """Compute RMS of μ-law audio."""
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return (sum(s**2 for s in samples) / max(len(samples), 1)) ** 0.5


async def collect_audio_from_ws(ws, timeout: float = 25.0, min_bytes: int = 3000) -> bytes:
    """Stream silence like Plivo does and return the agent's concatenated μ-law audio."""
    audio_chunks: list[bytes] = []
    total_bytes = 0
    start = time.time()
    last_audio = start
    silence = base64.b64encode(b"\xff" * 160).decode()

    while time.time() - start < timeout:
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))
        try:
            data = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.02))
            if data.get("event") == "playAudio":
                chunk = base64.b64decode(data["media"]["payload"])
                audio_chunks.append(chunk)
                total_bytes += len(chunk)
                last_audio = time.time()
        except TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed:
            break
        if total_bytes > min_bytes and (time.time() - last_audio) > 2.5:
            break

    return b"".join(audio_chunks)


def stream_url(call_uuid: str) -> str:
    """The /ws URL from a Plivo-signed answer webhook."""
    form = {"CallUUID": call_uuid, "From": "+15551234567", "To": "+16572338892"}
    resp = signed_webhook("POST", f"{TEST_HTTP_URL}/answer", TEST_AUTH_TOKEN, form)
    assert resp.status_code == 200, resp.text
    return stream_url_from_xml(resp.text)


async def opening_line_audio(call_uuid: str) -> bytes:
    url = stream_url(call_uuid)
    async with websockets.connect(
        url, additional_headers=stream_signature_headers(url, TEST_AUTH_TOKEN), close_timeout=3
    ) as ws:
        start_event = {
            "event": "start",
            "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
        }
        await ws.send(json.dumps(start_event))
        return await collect_audio_from_ws(ws, min_bytes=2000)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_process():
    """Start the inbound server on TEST_PORT (SIGTERM -> wait(5) -> SIGKILL on teardown)."""
    proc = start_server("inbound.server", TEST_PORT, LOG_PATH, local_server_env(TEST_PORT))
    print(f"\n[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


# =============================================================================
# Tests
# =============================================================================


class TestE2ELive:
    """End-to-end tests that start the server and simulate a Plivo call."""

    async def test_agent_greeting(self, server_process):
        """The agent opens the call with spoken audio that reads like a greeting."""
        ulaw_audio = await opening_line_audio("test-e2e")
        assert len(ulaw_audio) > 2000, (
            f"Opening line too short: {len(ulaw_audio)} bytes\n{log_tail(LOG_PATH)}"
        )

        rms = compute_audio_rms(ulaw_audio)
        print(f"\n[Greeting] Audio: {len(ulaw_audio)} bytes, RMS: {rms:.1f}")
        assert rms > 500, f"Audio RMS {rms:.1f} too low: likely silence"

        transcript = transcribe_ulaw(ulaw_audio)
        print(f"[Greeting transcript]: {transcript}")
        assert len(transcript) > 5, "Greeting transcript is too short"
        assert any(w in transcript.lower() for w in GREETING_WORDS), (
            f"Opening line does not read like a greeting: {transcript}"
        )

    async def test_audio_is_not_silence(self, server_process):
        """The received audio is speech: energy above the silence floor, longer than 0.5s."""
        ulaw_audio = await opening_line_audio("test-e2e-2")
        assert ulaw_audio, f"No audio received\n{log_tail(LOG_PATH)}"

        rms = compute_audio_rms(ulaw_audio)
        duration_s = len(ulaw_audio) / 8000
        print(f"\n[Audio quality] RMS: {rms:.1f}, duration: {duration_s:.2f}s")
        assert rms > 500, f"Audio RMS {rms:.1f} too low: likely silence"
        assert duration_s > 0.5, f"Audio too short: {duration_s:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
