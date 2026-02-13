"""
End-to-end live tests for the Grok Native voice agent.

These tests:
1. Start the server as a subprocess
2. Connect via WebSocket (simulating a Plivo call)
3. Receive agent greeting audio
4. Send a text prompt to the agent
5. Receive agent response audio
6. Transcribe audio locally using faster-whisper
7. Verify transcription is consistent with agent instructions

Requirements:
    - Valid XAI_API_KEY in .env
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (in project root or PATH)
    - Port 18001 available (used by test server)

Usage:
    cd grok-voice-native
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
import pytest
import websockets
from dotenv import load_dotenv

from utils import ulaw_to_pcm

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Ensure ffmpeg from project root is on PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isfile(os.path.join(PROJECT_ROOT, "ffmpeg")):
    os.environ["PATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

TEST_PORT = 18001
TEST_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not XAI_API_KEY,
    reason="XAI_API_KEY not configured",
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


async def collect_audio_from_ws(
    ws,
    timeout: float = 20.0,
    min_bytes: int = 3000,
) -> bytes:
    """Receive playAudio events from agent, return concatenated μ-law bytes."""
    audio_chunks = []
    total_bytes = 0
    start = time.time()
    last_audio = start

    while time.time() - start < timeout:
        silence = base64.b64encode(b"\xff" * 160).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))

        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=0.1)
            data = json.loads(msg)
            if data.get("event") == "playAudio":
                payload = data.get("media", {}).get("payload", "")
                if payload:
                    chunk = base64.b64decode(payload)
                    audio_chunks.append(chunk)
                    total_bytes += len(chunk)
                    last_audio = time.time()
        except (asyncio.TimeoutError, TimeoutError):
            pass
        except websockets.exceptions.ConnectionClosed:
            break

        if total_bytes > min_bytes and (time.time() - last_audio) > 2.5:
            break

    return b"".join(audio_chunks)


def compute_audio_rms(ulaw_audio: bytes) -> float:
    """Compute RMS of μ-law audio."""
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return (sum(s**2 for s in samples) / max(len(samples), 1)) ** 0.5


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_process():
    """Start the voice agent server as a subprocess on TEST_PORT."""
    env = os.environ.copy()
    env["SERVER_PORT"] = str(TEST_PORT)

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-m", "inbound.server"],
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
    proc.wait(timeout=5)


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

        async with websockets.connect(ws_url, close_timeout=3) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))
            ulaw_audio = await collect_audio_from_ws(ws, timeout=20, min_bytes=2000)

        assert len(ulaw_audio) > 2000, f"Greeting too short: {len(ulaw_audio)} bytes"

        rms = compute_audio_rms(ulaw_audio)
        print(f"\n[Greeting] Audio: {len(ulaw_audio)} bytes, RMS: {rms:.1f}")
        assert rms > 500, f"Audio RMS {rms:.1f} too low — likely silence"

        # Transcribe locally with Whisper
        pcm = ulaw_to_pcm(ulaw_audio)
        wav = pcm16_to_wav(pcm, sample_rate=8000)
        transcript = transcribe_audio_local(wav)
        print(f"[Greeting transcript]: {transcript}")

        assert len(transcript) > 5, "Greeting transcript is too short"
        greeting_words = ["hello", "hi", "welcome", "help", "how", "assist", "alex", "techflow"]
        assert any(w in transcript.lower() for w in greeting_words), (
            f"Greeting doesn't match expected content: {transcript}"
        )

    @pytest.mark.asyncio
    async def test_agent_responds_to_text(self, server_process):
        """Agent should respond when sent a text message about products."""
        body_data = {"call_uuid": "test-e2e-2", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{TEST_WS_URL}?body={body_b64}"

        async with websockets.connect(ws_url, close_timeout=3) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))

            greeting = await collect_audio_from_ws(ws, timeout=20, min_bytes=2000)
            assert len(greeting) > 0, "No greeting received"
            print(f"\n[Greeting] {len(greeting)} bytes received")

            # Inject a text question (native agent supports 'text' events)
            await ws.send(
                json.dumps(
                    {
                        "event": "text",
                        "text": "What plans do you offer and how much do they cost?",
                    }
                )
            )

            response_audio = await collect_audio_from_ws(ws, timeout=25, min_bytes=3000)

        assert len(response_audio) > 3000, f"Response too short: {len(response_audio)} bytes"

        rms = compute_audio_rms(response_audio)
        print(f"[Response] Audio: {len(response_audio)} bytes, RMS: {rms:.1f}")
        assert rms > 500, f"Response RMS {rms:.1f} too low — likely silence"

        # Transcribe locally with Whisper
        pcm = ulaw_to_pcm(response_audio)
        wav = pcm16_to_wav(pcm, sample_rate=8000)
        transcript = transcribe_audio_local(wav)
        print(f"[Product response transcript]: {transcript}")

        assert len(transcript) > 10, "Product response transcript is too short"
        product_words = [
            "pro",
            "team",
            "enterprise",
            "twelve",
            "twenty",
            "dollar",
            "month",
            "plan",
            "price",
            "cost",
            "12",
            "25",
        ]
        matches = [w for w in product_words if w in transcript.lower()]
        assert len(matches) >= 2, (
            f"Response doesn't discuss products enough. "
            f"Matches: {matches}, transcript: {transcript}"
        )

    @pytest.mark.asyncio
    async def test_audio_is_not_silence(self, server_process):
        """Verify the received audio has actual speech content (not silence)."""
        body_data = {"call_uuid": "test-e2e-3", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{TEST_WS_URL}?body={body_b64}"

        async with websockets.connect(ws_url, close_timeout=3) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))
            ulaw_audio = await collect_audio_from_ws(ws, timeout=20, min_bytes=2000)

        assert len(ulaw_audio) > 0, "No audio received"

        pcm = ulaw_to_pcm(ulaw_audio)
        samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        duration_s = len(samples) / 8000

        print(
            f"\n[Audio quality] RMS: {rms:.1f}, duration: {duration_s:.2f}s, "
            f"samples: {len(samples)}"
        )
        assert rms > 500, f"Audio RMS {rms:.1f} too low — likely silence"
        assert duration_s > 0.5, f"Audio too short: {duration_s:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
