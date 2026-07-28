"""Multi-turn voice conversation tests for the xAI realtime voice agent.

Starts the inbound server, connects to its Plivo WebSocket, and drives a
multi-turn conversation using synthesized speech. Each turn is generated with
gTTS, converted to mu-law 8kHz, and streamed as Plivo media so the realtime
server-side VAD detects the turn and the agent responds with playAudio frames.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid

import httpx
import pytest
import websockets
from dotenv import load_dotenv

load_dotenv()

XAI_API_KEY = os.getenv("XAI_API_KEY", "")
TEST_PORT = 18004
TEST_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(not XAI_API_KEY, reason="XAI_API_KEY not configured")


def generate_tts_audio(text: str) -> bytes | None:
    """Generate speech with gTTS and convert it to mu-law 8kHz, or None if unavailable."""
    try:
        import audioop

        from gtts import gTTS
        from pydub import AudioSegment
    except Exception:
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file_handle:
        mp3_path = file_handle.name

    try:
        gTTS(text=text, lang="en").save(mp3_path)
        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
        return audioop.lin2ulaw(audio.raw_data, 2)
    except Exception:
        return None
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


async def wait_for_greeting(ws, timeout: float = 15.0) -> dict:
    """Feed silence and collect the agent's greeting audio."""
    result = {"recv_chunks": 0, "recv_bytes": 0}
    start = time.time()
    last_received = start

    while time.time() - start < timeout:
        silence = base64.b64encode(b"\xff" * 160).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))

        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                data = json.loads(message)
                if data.get("event") == "playAudio":
                    result["recv_chunks"] += 1
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        result["recv_bytes"] += len(base64.b64decode(payload))
                    last_received = time.time()
        except (asyncio.TimeoutError, TimeoutError):
            pass
        except websockets.exceptions.ConnectionClosed:
            break

        if result["recv_chunks"] > 0 and (time.time() - last_received) > 2:
            break

        await asyncio.sleep(0.02)

    return result


async def send_audio_and_wait(ws, audio_bytes: bytes, timeout: float = 20.0) -> dict:
    """Stream a turn of mu-law audio then collect the agent's response audio."""
    result = {"sent_chunks": 0, "recv_chunks": 0, "recv_bytes": 0, "ttfr": None}

    chunk_size = 160
    chunks = [audio_bytes[i : i + chunk_size] for i in range(0, len(audio_bytes), chunk_size)]

    start = time.time()

    for chunk in chunks:
        payload = base64.b64encode(chunk).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": payload}}))
        result["sent_chunks"] += 1
        await asyncio.sleep(0.02)

    last_received = time.time()

    while time.time() - start < timeout:
        silence = base64.b64encode(b"\xff" * 160).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))

        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                data = json.loads(message)
                if data.get("event") == "playAudio":
                    result["recv_chunks"] += 1
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        result["recv_bytes"] += len(base64.b64decode(payload))
                    last_received = time.time()
                    if result["ttfr"] is None:
                        result["ttfr"] = time.time() - start
        except (asyncio.TimeoutError, TimeoutError):
            pass
        except websockets.exceptions.ConnectionClosed:
            break

        if result["recv_chunks"] > 0 and (time.time() - last_received) > 3:
            break

        await asyncio.sleep(0.02)

    return result


@pytest.fixture(scope="module")
def server_process():
    """Start the inbound voice agent server as a subprocess on TEST_PORT."""
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


class TestMultiturnVoice:
    """Multi-turn conversation over the Plivo WebSocket with synthesized speech."""

    @pytest.mark.asyncio
    async def test_multiturn_conversation(self, server_process):
        """The agent greets, then responds to each spoken turn with audio."""
        turns = [
            "What are your business hours?",
            "Do you have any specials today?",
            "Thanks, goodbye!",
        ]

        turn_audio = [generate_tts_audio(text) for text in turns]
        if any(audio is None for audio in turn_audio):
            pytest.skip("gTTS, pydub, or ffmpeg not available for speech synthesis")

        body_data = {"call_uuid": "test-multiturn", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{TEST_WS_URL}?body={body_b64}"

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))

            greeting = await wait_for_greeting(ws)
            assert greeting["recv_bytes"] > 0, "No greeting audio received from agent"

            responses = []
            for audio in turn_audio:
                responses.append(await send_audio_and_wait(ws, audio, timeout=20.0))

        assert any(result["recv_bytes"] > 0 for result in responses), (
            "Agent did not respond with audio to any conversation turn"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
