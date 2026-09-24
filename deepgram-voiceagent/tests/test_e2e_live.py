"""
End-to-end live tests for the Deepgram Voice Agent API example.

These tests:
1. Start the inbound server as a subprocess (JSON logs to a file)
2. Connect via WebSocket, simulating Plivo: start event, μ-law silence every 20ms,
   and playedStream replies to checkpoints once the audio would have finished playing
3. Receive the agent greeting audio
4. Send text turns (Plivo ``text`` events -> Deepgram InjectUserMessage)
5. Transcribe the agent audio locally with faster-whisper and check its content
6. Ask the agent to end the call and verify the server closes the WebSocket

Every test runs twice: ``[inline]`` (Settings carries the agent block) and ``[saved]``
(a reusable agent config is created for the module with the README's create body, the
server runs with DEEPGRAM_INBOUND_AGENT_ID set, and the config is always deleted afterwards).

Requirements:
    - Valid DEEPGRAM_API_KEY in .env
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (PATH, FFMPEG_DIR, or a parent directory)
    - Port 18001 available (used by test server)

Usage:
    uv run pytest tests/test_e2e_live.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
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
    create_agent_config,
    delete_agent_config,
    ensure_ffmpeg_on_path,
    log_messages,
    read_log_events,
    server_log_path,
    start_server,
    stop_server,
)
from utils import ulaw_to_pcm

load_dotenv()
ensure_ffmpeg_on_path()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

TEST_PORT = 18001
TEST_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
LOG_PATH = server_log_path("e2e_live_server")

pytestmark = pytest.mark.skipif(not DEEPGRAM_API_KEY, reason="DEEPGRAM_API_KEY not configured")


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


class SimulatedPlivo:
    """Plays Plivo's side of the bidirectional stream against the local server."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.stream_id = str(uuid.uuid4())
        self.call_id = str(uuid.uuid4())
        self.audio = bytearray()  # agent audio for the current phase
        self.checkpoints: list[str] = []
        self.played: list[str] = []  # checkpoints acked with playedStream
        self.clear_audio = 0
        self.closed = asyncio.Event()
        self._play_start: float | None = None
        self._play_chunks = 0
        self._last_audio = 0.0
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        start = {
            "event": "start",
            "start": {"callId": self.call_id, "streamId": self.stream_id},
        }
        await self.ws.send(json.dumps(start))
        self._tasks = [asyncio.create_task(self._silence()), asyncio.create_task(self._recv())]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _silence(self) -> None:
        payload = base64.b64encode(b"\xff" * 160).decode()
        with contextlib.suppress(Exception):
            while not self.closed.is_set():
                await self.ws.send(json.dumps({"event": "media", "media": {"payload": payload}}))
                await asyncio.sleep(0.02)

    async def _recv(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                event = msg.get("event")
                if event == "playAudio":
                    assert msg["media"]["contentType"] == "audio/x-mulaw"
                    chunk = base64.b64decode(msg["media"]["payload"])
                    assert len(chunk) == 160
                    self.audio.extend(chunk)
                    now = time.monotonic()
                    self._last_audio = now
                    if self._play_start is None:
                        self._play_start = now
                        self._play_chunks = 0
                    self._play_chunks += 1
                elif event == "checkpoint":
                    self.checkpoints.append(msg["name"])
                    end = (self._play_start or time.monotonic()) + self._play_chunks * 0.02
                    self._play_start = None
                    asyncio.get_running_loop().create_task(self._played(msg["name"], end))
                elif event == "clearAudio":
                    self.clear_audio += 1
                    self._play_start = None
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.closed.set()

    async def _played(self, name: str, end: float) -> None:
        await asyncio.sleep(max(0.0, end - time.monotonic()))
        with contextlib.suppress(Exception):
            played = {"event": "playedStream", "name": name, "streamId": self.stream_id}
            await self.ws.send(json.dumps(played))
            self.played.append(name)

    async def send_text(self, text: str) -> None:
        await self.ws.send(json.dumps({"event": "text", "text": text}))

    async def collect_response(self, timeout: float = 30.0, quiet_s: float = 1.0) -> bytes:
        """Return agent audio once its playback finished (playedStream sent) and went quiet.

        Waiting for the simulated playback matters: a text turn sent while the agent is
        still "playing" is (correctly) handled as a barge-in.
        """
        self.audio.clear()
        seen = len(self.played)
        start = time.monotonic()
        while time.monotonic() - start < timeout and not self.closed.is_set():
            await asyncio.sleep(0.1)
            done = len(self.played) > seen and self.audio
            if done and time.monotonic() - self._last_audio > quiet_s:
                break
        return bytes(self.audio)


@contextlib.asynccontextmanager
async def plivo_call(call_uuid: str):
    body = base64.b64encode(
        json.dumps({"call_uuid": call_uuid, "from": "+15551234567", "to": "+16572338892"}).encode()
    ).decode()
    async with websockets.connect(f"{TEST_WS_URL}?body={body}", close_timeout=3) as ws:
        call = SimulatedPlivo(ws)
        await call.start()
        try:
            yield call
        finally:
            await call.stop()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module", params=["inline", "saved"])
def server_process(request):
    """Start the inbound server on TEST_PORT (SIGTERM -> wait(5) -> SIGKILL on teardown).

    ``saved`` creates a reusable agent config first and always deletes it afterwards.
    """
    # No Plivo credentials: end_call must not try a REST hangup of a fake call
    env = {"PLIVO_AUTH_ID": "", "PLIVO_AUTH_TOKEN": "", "PUBLIC_URL": ""}
    env["DEEPGRAM_INBOUND_AGENT_ID"] = ""
    if request.param == "saved":
        env["DEEPGRAM_INBOUND_AGENT_ID"] = create_agent_config("inbound")
        print(f"\n[saved config] created {env['DEEPGRAM_INBOUND_AGENT_ID']}")
    try:
        proc = start_server("inbound.server", TEST_PORT, LOG_PATH, env)
        print(f"\n[server] mode={request.param} logs: {LOG_PATH}")
        yield request.param, env["DEEPGRAM_INBOUND_AGENT_ID"]
        stop_server(proc)
    finally:
        if env["DEEPGRAM_INBOUND_AGENT_ID"]:
            delete_agent_config(env["DEEPGRAM_INBOUND_AGENT_ID"])
            print(f"\n[saved config] deleted {env['DEEPGRAM_INBOUND_AGENT_ID']}")


# =============================================================================
# Tests
# =============================================================================


class TestE2ELive:
    """End-to-end tests that start the server and simulate a Plivo call."""

    async def test_agent_greeting(self, server_process):
        """Agent speaks the greeting when the call connects."""
        async with plivo_call("test-e2e-greeting") as call:
            greeting = await call.collect_response(timeout=20)

        assert len(greeting) > 8000, f"Greeting too short: {len(greeting)} bytes"
        rms = compute_audio_rms(greeting)
        print(f"\n[Greeting] {len(greeting)} bytes ({len(greeting) / 8000:.1f}s), RMS {rms:.0f}")
        assert rms > 500, f"Audio RMS {rms:.1f} too low — likely silence"
        assert call.checkpoints, "No checkpoint was sent after the greeting audio"

        transcript = transcribe_ulaw(greeting)
        print(f"[Greeting transcript]: {transcript}")
        words = ["alex", "techflow", "deepgram", "help", "hi"]
        assert any(w in transcript.lower() for w in words), transcript

    async def test_agent_responds_to_text(self, server_process):
        """A text turn about plans gets a spoken answer about products and pricing."""
        async with plivo_call("test-e2e-plans") as call:
            greeting = await call.collect_response(timeout=20)
            assert greeting, "No greeting received"
            await call.send_text("What plans do you offer and how much do they cost?")
            response = await call.collect_response(timeout=30)

        assert len(response) > 8000, f"Response too short: {len(response)} bytes"
        rms = compute_audio_rms(response)
        print(f"\n[Response] {len(response)} bytes ({len(response) / 8000:.1f}s), RMS {rms:.0f}")
        assert rms > 500, f"Response RMS {rms:.1f} too low — likely silence"

        transcript = transcribe_ulaw(response)
        print(f"[Product response transcript]: {transcript}")
        product_words = [
            "pro",
            "team",
            "enterprise",
            "starter",
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
        assert len(matches) >= 2, f"Not about products. Matches: {matches}, got: {transcript}"

        # The server emits turn_complete when it processes playedStream; poll its log
        # instead of assuming a fixed flush delay.
        turns: list[dict] = []
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            turns = [
                e
                for e in read_log_events(LOG_PATH, "turn_complete")
                if e["call_id"] == call.call_id
            ]
            if len(turns) >= 2:
                break
            await asyncio.sleep(0.2)
        print(
            f"[turn_complete] {[(e['turn'], e['barge_in'], e['total_latency_ms']) for e in turns]}"
        )
        assert [(e["turn"], e["barge_in"]) for e in turns] == [(1, False), (2, False)]
        assert turns[1]["user_text"] == "What plans do you offer and how much do they cost?"
        assert turns[1]["total_latency_ms"], "Deepgram LatencyReport not merged into turn_complete"

    async def test_audio_is_not_silence(self, server_process):
        """The greeting has real speech energy and duration."""
        async with plivo_call("test-e2e-quality") as call:
            audio = await call.collect_response(timeout=20)

        assert audio, "No audio received"
        rms = compute_audio_rms(audio)
        duration_s = len(audio) / 8000
        print(f"\n[Audio quality] RMS: {rms:.1f}, duration: {duration_s:.2f}s")
        assert rms > 500, f"Audio RMS {rms:.1f} too low — likely silence"
        assert duration_s > 0.5, f"Audio too short: {duration_s:.2f}s"

    async def test_end_call_closes_websocket(self, server_process):
        """Saying goodbye makes the agent call end_call; the server closes the stream."""
        async with plivo_call("test-e2e-goodbye") as call:
            await call.collect_response(timeout=20)
            t0 = time.monotonic()
            await call.send_text("That's all, thank you, goodbye.")
            try:
                await asyncio.wait_for(call.closed.wait(), timeout=30)
            except (TimeoutError, asyncio.TimeoutError):
                pytest.fail("Server did not close the WebSocket within 30s of goodbye")
            elapsed = time.monotonic() - t0
            goodbye_audio = bytes(call.audio)

        print(f"\n[Goodbye] server closed the stream {elapsed:.1f}s after goodbye")
        assert goodbye_audio, "Agent hung up without saying goodbye"
        await asyncio.sleep(0.5)  # let the server flush its logs
        messages = log_messages(LOG_PATH)
        assert any("calling end_call" in m for m in messages), "end_call was never invoked"
        assert any("goodbye played -- hanging up" in m for m in messages)
        sessions = read_log_events(LOG_PATH, "session_end")
        assert sessions, "No session_end event logged"

    async def test_settings_mode_logged(self, server_process):
        """The session logs which Settings mode it used, and session_end carries it."""
        mode, config_id = server_process
        async with plivo_call("test-e2e-mode") as call:
            await call.collect_response(timeout=20)
        expected = f"settings: saved agent config {config_id}" if config_id else "settings: inline"
        startup = (
            f"Deepgram agent: reusable config {config_id}"
            if config_id
            else "Deepgram agent: inline (listen="
        )
        deadline = time.monotonic() + 5.0
        ends: list[dict] = []
        while time.monotonic() < deadline and not ends:
            await asyncio.sleep(0.2)
            ends = [
                e for e in read_log_events(LOG_PATH, "session_end") if e["call_id"] == call.call_id
            ]
        print(f"\n[{mode}] session_end agent_config={ends and ends[0].get('agent_config')}")
        messages = log_messages(LOG_PATH)
        assert any(m.startswith(startup) for m in messages), startup
        assert any(expected in m for m in messages), expected
        assert ends, "No session_end for this call"
        assert ends[0]["agent_config"] == (config_id or "inline")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
