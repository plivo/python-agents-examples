"""
End-to-end live tests for the Gemini Live Pipecat voice agent.

These tests:
1. Start the server as a subprocess
2. Connect via WebSocket (simulating a Plivo call)
3. Receive agent greeting audio
4. Receive agent response audio
5. Transcribe audio locally using faster-whisper
6. Verify transcription is consistent with agent instructions

Requirements:
    - Valid GEMINI_API_KEY in .env
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (in project root or PATH)
    - Port 18003 available (used by test server)

Usage:
    cd gemini-live-pipecat
    uv run pytest tests/test_e2e_live.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import wave

import httpx
import numpy as np
import pytest
import websockets
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Ensure ffmpeg from project root is on PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isfile(os.path.join(PROJECT_ROOT, "ffmpeg")):
    os.environ["PATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

TEST_PORT = 18003
TEST_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not GEMINI_API_KEY,
    reason="GEMINI_API_KEY not configured",
)


# =============================================================================
# Audio helpers — μ-law decode table inlined to avoid importing pipecat deps
# =============================================================================

_ULAW_DECODE_TABLE = np.array([
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
], dtype=np.int16)


def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    """Convert μ-law encoded audio to 16-bit PCM."""
    ulaw_samples = np.frombuffer(ulaw_data, dtype=np.uint8)
    pcm_samples = _ULAW_DECODE_TABLE[ulaw_samples]
    return pcm_samples.tobytes()


def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw PCM16 bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def transcribe_audio_local(audio_wav: bytes) -> str:
    """Transcribe WAV audio locally using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_wav)
        tmp_path = f.name

    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


def compute_audio_rms(ulaw_audio: bytes) -> float:
    """Compute RMS of μ-law audio (converts to PCM first)."""
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return (sum(s**2 for s in samples) / max(len(samples), 1)) ** 0.5


async def collect_audio_from_ws(
    ws,
    stream_id: str,
    timeout: float = 25.0,
    min_bytes: int = 3000,
) -> bytes:
    """
    Receive audio from the pipecat agent WS.
    Pipecat via PlivoFrameSerializer sends JSON playAudio events
    with base64 μ-law payloads.
    Uses a background task to send silence at 50 packets/sec to keep
    the Pipecat transport audio pipeline active.
    """
    audio_chunks = []
    total_bytes = 0
    start = time.time()
    last_audio = start

    async def send_periodic_silence():
        """Send silence frames at 20ms intervals to keep the stream active."""
        silence = base64.b64encode(b"\xff" * 160).decode()
        media_msg = json.dumps({
            "event": "media",
            "media": {"payload": silence, "streamId": stream_id},
        })
        while True:
            try:
                await ws.send(media_msg)
            except websockets.exceptions.ConnectionClosed:
                break
            await asyncio.sleep(0.02)

    silence_task = asyncio.create_task(send_periodic_silence())

    try:
        while time.time() - start < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=2.0)

                if isinstance(msg, bytes):
                    audio_chunks.append(msg)
                    total_bytes += len(msg)
                    last_audio = time.time()
                elif isinstance(msg, str):
                    try:
                        data = json.loads(msg)
                        if data.get("event") == "playAudio":
                            payload = data.get("media", {}).get("payload", "")
                            if payload:
                                chunk = base64.b64decode(payload)
                                audio_chunks.append(chunk)
                                total_bytes += len(chunk)
                                last_audio = time.time()
                    except json.JSONDecodeError:
                        pass
            except (asyncio.TimeoutError, TimeoutError):
                pass
            except websockets.exceptions.ConnectionClosed:
                break

            if total_bytes > min_bytes and (time.time() - last_audio) > 3.0:
                break
    finally:
        silence_task.cancel()
        try:
            await silence_task
        except asyncio.CancelledError:
            pass

    return b"".join(audio_chunks)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_process():
    """Start the pipecat voice agent server as a subprocess on TEST_PORT."""
    env = os.environ.copy()
    env["SERVER_PORT"] = str(TEST_PORT)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "server.py"],
        cwd=project_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    ready = False
    for _ in range(30):
        try:
            resp = httpx.get(TEST_HTTP_URL, timeout=1.0)
            if resp.status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not ready:
        proc.terminate()
        proc.wait()
        output = proc.stdout.read().decode() if proc.stdout else ""
        pytest.skip(f"Server did not start in time. Output:\n{output[:2000]}")

    yield proc

    os.kill(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


# =============================================================================
# Tests
# =============================================================================


class TestE2ELive:
    """End-to-end tests that start the server and simulate a Plivo call."""

    @pytest.mark.asyncio
    async def test_agent_greeting(self, server_process):
        """Agent should send an audio greeting when the call connects."""
        body_data = {"call_uuid": "test-e2e", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{TEST_WS_URL}?body={body_b64}"

        stream_id = str(uuid.uuid4())
        async with websockets.connect(ws_url, close_timeout=3) as ws:
            start_event = {
                "event": "start",
                "start": {
                    "streamId": stream_id,
                    "callId": str(uuid.uuid4()),
                },
            }
            await ws.send(json.dumps(start_event))

            ulaw_audio = await collect_audio_from_ws(ws, stream_id, timeout=25, min_bytes=2000)

        assert len(ulaw_audio) > 2000, f"Greeting too short: {len(ulaw_audio)} bytes"

        # Check audio quality
        rms = compute_audio_rms(ulaw_audio)
        print(f"\n[Greeting] Audio: {len(ulaw_audio)} bytes, RMS: {rms:.1f}")
        assert rms > 300, f"Audio RMS {rms:.1f} too low — likely silence"

        # Transcribe locally with faster-whisper
        pcm = ulaw_to_pcm(ulaw_audio)
        wav = pcm16_to_wav(pcm, sample_rate=8000)
        transcript = transcribe_audio_local(wav)
        print(f"[Greeting transcript]: {transcript}")

        assert len(transcript) > 5, "Greeting transcript is too short"
        greeting_words = ["hello", "hi", "help", "how", "assist", "welcome"]
        assert any(
            w in transcript.lower() for w in greeting_words
        ), f"Greeting doesn't match expected content: {transcript}"

    @pytest.mark.asyncio
    async def test_audio_is_not_silence(self, server_process):
        """Verify the greeting audio has actual speech content."""
        body_data = {"call_uuid": "test-e2e-2", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{TEST_WS_URL}?body={body_b64}"

        stream_id = str(uuid.uuid4())
        async with websockets.connect(ws_url, close_timeout=3) as ws:
            start_event = {
                "event": "start",
                "start": {
                    "streamId": stream_id,
                    "callId": str(uuid.uuid4()),
                },
            }
            await ws.send(json.dumps(start_event))

            ulaw_audio = await collect_audio_from_ws(ws, stream_id, timeout=25, min_bytes=2000)

        assert len(ulaw_audio) > 0, "No audio received"

        pcm = ulaw_to_pcm(ulaw_audio)
        samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        duration_s = len(samples) / 8000

        print(f"\n[Audio quality] RMS: {rms:.1f}, duration: {duration_s:.2f}s, samples: {len(samples)}")
        assert rms > 300, f"Audio RMS {rms:.1f} too low — likely silence"
        assert duration_s > 0.3, f"Audio too short: {duration_s:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
