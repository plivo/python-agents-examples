"""End-to-end live tests for the xAI realtime  voice agent."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import signal
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

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
TEST_PORT = 18001
TEST_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(not XAI_API_KEY, reason="XAI_API_KEY not configured")


def pcmu_to_pcm16(audio_data: bytes) -> bytes:
    import audioop

    return audioop.ulaw2lin(audio_data, 2)


def pcm16_to_wav(pcm_data: bytes, sample_rate: int = 8000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buf.getvalue()


def transcribe_audio_local(audio_wav: bytes) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as file_handle:
        file_handle.write(audio_wav)
        tmp_path = file_handle.name

    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        os.unlink(tmp_path)


async def collect_audio_from_ws(ws, timeout: float = 20.0, min_bytes: int = 3000) -> bytes:
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


@pytest.fixture(scope="module")
def server_process():
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
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


class TestE2ELive:
    @pytest.mark.asyncio
    async def test_agent_greeting(self, server_process):
        body_data = {"call_uuid": "test-e2e", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{TEST_WS_URL}?body={body_b64}"

        async with websockets.connect(ws_url, close_timeout=3) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))
            pcmu_audio = await collect_audio_from_ws(ws, timeout=20, min_bytes=2000)

        assert len(pcmu_audio) > 2000, f"Greeting too short: {len(pcmu_audio)} bytes"

        pcm = pcmu_to_pcm16(pcmu_audio)
        wav = pcm16_to_wav(pcm, sample_rate=8000)
        transcript = transcribe_audio_local(wav)

        assert len(transcript) > 5, "Greeting transcript is too short"
        greeting_words = ["hello", "hi", "help", "assist", "techflow"]
        assert any(word in transcript.lower() for word in greeting_words)
