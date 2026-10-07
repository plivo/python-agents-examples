"""
Multi-turn voice conversation + barge-in tests against a local inbound server.

The test plays Plivo's side of the bidirectional stream: it fetches the <Stream> URL from
a Plivo-signed /answer webhook, sends the start event, then μ-law 8kHz audio in 20ms
frames. User turns are real speech synthesised with gTTS, so they pass through Silero VAD
and Modulate Velma-2 exactly as caller audio does. No phone call is placed and the server
gets no Plivo account or number.

Tests:
1. Multi-turn: the opening line, then three spoken user turns; each must be answered
   with speech (playAudio with energy above the silence floor).
2. Barge-in: speak over an answer while it is still streaming. The server must send
   Plivo a clearAudio event (CLAUDE.md "WebSocket Protocol" step 5) and then answer the
   interrupting turn.

Requirements (the tests skip, with the reason, when any is missing):
    - OPENAI_API_KEY, MODULATE_API_KEY and CARTESIA_API_KEY in .env
    - gTTS and pydub (not dev dependencies: `uv pip install gTTS pydub`; pydub also
      needs `audioop-lts` on Python 3.13+), network access for gTTS
    - ffmpeg binary available (PATH, FFMPEG_DIR, or a parent directory)
    - Port 18004 available

Usage:
    uv run pytest tests/test_multiturn_voice.py -v -s
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import shutil
import struct
import tempfile
import time
import uuid

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
    stream_url_from_xml,
)
from utils import pcm_to_ulaw, ulaw_to_pcm

load_dotenv()
ensure_ffmpeg_on_path()

TEST_PORT = 18004
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"
LOG_PATH = server_log_path("modulate_multiturn_server")

FRAME_BYTES = 160  # 20ms of μ-law at 8kHz
FRAME_SECS = 0.02
SILENCE_FRAME = base64.b64encode(b"\xff" * FRAME_BYTES).decode()

pytestmark = pytest.mark.skipif(
    bool(missing_env(*LIVE_API_KEYS)),
    reason=f"not configured: {', '.join(missing_env(*LIVE_API_KEYS))}",
)


# =============================================================================
# Helpers
# =============================================================================


def rms_of_ulaw(ulaw_audio: bytes) -> float:
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5


class SimulatedCaller:
    """Plays Plivo's side of the stream: frames out every 20ms, events in."""

    def __init__(self, ws) -> None:
        self.ws = ws
        self.audio = bytearray()  # every playAudio payload so far
        self.clear_audio_at: list[float] = []  # arrival times of clearAudio events
        self.last_audio_at = 0.0
        self.closed = False
        self._speech: list[str] = []  # queued user frames (base64), sent ahead of silence
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        start = {
            "event": "start",
            "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
        }
        await self.ws.send(json.dumps(start))
        self._tasks = [
            asyncio.create_task(self._send_frames(), name="caller_tx"),
            asyncio.create_task(self._receive(), name="caller_rx"),
        ]

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

    async def _send_frames(self) -> None:
        """One frame every 20ms, as Plivo sends them: queued speech, otherwise silence."""
        next_send = time.monotonic()
        while True:
            payload = self._speech.pop(0) if self._speech else SILENCE_FRAME
            await self.ws.send(json.dumps({"event": "media", "media": {"payload": payload}}))
            next_send += FRAME_SECS
            await asyncio.sleep(max(0.0, next_send - time.monotonic()))

    async def _receive(self) -> None:
        try:
            async for message in self.ws:
                data = json.loads(message)
                if data.get("event") == "playAudio":
                    self.audio.extend(base64.b64decode(data["media"]["payload"]))
                    self.last_audio_at = time.monotonic()
                elif data.get("event") == "clearAudio":
                    self.clear_audio_at.append(time.monotonic())
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.closed = True

    def say(self, ulaw_audio: bytes) -> None:
        """Queue a spoken user turn; it goes out in real time, 20ms per frame."""
        for i in range(0, len(ulaw_audio), FRAME_BYTES):
            frame = ulaw_audio[i : i + FRAME_BYTES].ljust(FRAME_BYTES, b"\xff")
            self._speech.append(base64.b64encode(frame).decode())

    async def wait_until_said(self) -> None:
        while self._speech and not self.closed:
            await asyncio.sleep(FRAME_SECS)

    async def wait_for_audio(self, offset: int, min_bytes: int, timeout: float) -> bool:
        """True once more than ``min_bytes`` of agent audio arrived after ``offset``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.closed:
            if len(self.audio) - offset >= min_bytes:
                return True
            await asyncio.sleep(0.05)
        return len(self.audio) - offset >= min_bytes

    async def wait_until_quiet(self, quiet_secs: float = 2.5, timeout: float = 40.0) -> None:
        """Wait until no agent audio has arrived for ``quiet_secs``."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.closed:
            if self.last_audio_at and time.monotonic() - self.last_audio_at >= quiet_secs:
                return
            await asyncio.sleep(0.05)


def stream_url(call_uuid: str) -> str:
    """The /ws URL from a Plivo-signed answer webhook."""
    form = {"CallUUID": call_uuid, "From": "+15551234567", "To": "+16572338892"}
    resp = signed_webhook("POST", f"{TEST_HTTP_URL}/answer", TEST_AUTH_TOKEN, form)
    assert resp.status_code == 200, resp.text
    return stream_url_from_xml(resp.text)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def synthesize():
    """text -> μ-law 8kHz speech via gTTS; skips the module when it cannot be produced."""
    gtts = pytest.importorskip("gtts", reason="gTTS not installed (uv pip install gTTS pydub)")
    pydub = pytest.importorskip(
        "pydub", reason="pydub not importable (uv pip install pydub; audioop-lts on 3.13+)"
    )
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg not found (PATH, FFMPEG_DIR or a parent directory)")

    def synth(text: str) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        try:
            try:
                gtts.gTTS(text=text, lang="en").save(mp3_path)
            except Exception as e:  # gTTS needs network access to Google
                pytest.skip(f"gTTS could not synthesise speech: {type(e).__name__}: {e}")
            audio = pydub.AudioSegment.from_mp3(mp3_path)
            audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            ulaw = pcm_to_ulaw(audio.raw_data)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(mp3_path)
        assert len(ulaw) > 4000, f"Synthesised speech too short for '{text}'"
        return ulaw

    return synth


@pytest.fixture(scope="module")
def server_process(synthesize):
    """Inbound server on TEST_PORT (SIGTERM -> wait(5) -> SIGKILL on teardown).

    Depends on ``synthesize`` so no server is started when the speech deps are missing.
    """
    proc = start_server("inbound.server", TEST_PORT, LOG_PATH, local_server_env(TEST_PORT))
    print(f"\n[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


# =============================================================================
# Tests
# =============================================================================


class TestMultiturnVoice:
    """Multi-turn conversation and barge-in over the simulated Plivo stream."""

    async def test_multiturn_conversation(self, server_process, synthesize):
        """Opening line, then three spoken turns, each answered with speech."""
        turns = [
            "What can you help me with?",
            "What is the capital city of France?",
            "Thank you, that is all. Goodbye.",
        ]
        speech = [synthesize(text) for text in turns]
        responses: list[bytes] = []

        async with websockets.connect(stream_url("multiturn"), close_timeout=3) as ws:
            caller = SimulatedCaller(ws)
            await caller.start()
            try:
                assert await caller.wait_for_audio(0, 1600, timeout=25), (
                    f"No opening line\n{log_tail(LOG_PATH)}"
                )
                await caller.wait_until_quiet()

                for text, audio in zip(turns, speech, strict=True):
                    offset = len(caller.audio)
                    print(f"[Turn] user: '{text}' ({len(audio) / 8000:.1f}s)")
                    caller.say(audio)
                    await caller.wait_until_said()
                    answered = await caller.wait_for_audio(offset, 1600, timeout=30)
                    assert answered, f"No answer to '{text}'\n{log_tail(LOG_PATH)}"
                    await caller.wait_until_quiet()
                    responses.append(bytes(caller.audio[offset:]))
                    print(f"[Turn] agent: {len(responses[-1]) / 8000:.1f}s of audio")
            finally:
                await caller.stop()

        assert len(responses) == len(turns)
        for text, response in zip(turns, responses, strict=True):
            assert rms_of_ulaw(response) > 500, f"Answer to '{text}' is silence"

    async def test_barge_in(self, server_process, synthesize):
        """Speaking over an answer makes the server send clearAudio, then answer again."""
        question = synthesize("Can you explain, step by step, how a phone call gets connected?")
        interruption = synthesize("Wait, stop. What is two plus two?")

        async with websockets.connect(stream_url("barge-in"), close_timeout=3) as ws:
            caller = SimulatedCaller(ws)
            await caller.start()
            try:
                assert await caller.wait_for_audio(0, 1600, timeout=25), (
                    f"No opening line\n{log_tail(LOG_PATH)}"
                )
                await caller.wait_until_quiet()

                offset = len(caller.audio)
                caller.say(question)
                await caller.wait_until_said()
                assert await caller.wait_for_audio(offset, 1600, timeout=30), (
                    f"The agent never started answering\n{log_tail(LOG_PATH)}"
                )

                # Interrupt while the answer is still streaming
                interrupted_at = time.monotonic()
                assert interrupted_at - caller.last_audio_at < 1.0, (
                    "The answer had already finished streaming; nothing left to interrupt"
                )
                cleared_before = len(caller.clear_audio_at)
                caller.say(interruption)
                await caller.wait_until_said()

                deadline = time.monotonic() + 10
                while time.monotonic() < deadline and len(caller.clear_audio_at) == cleared_before:
                    await asyncio.sleep(0.05)
                new_clears = caller.clear_audio_at[cleared_before:]
                assert new_clears, (
                    f"No clearAudio after speaking over the agent\n{log_tail(LOG_PATH)}"
                )
                print(f"[Barge-in] clearAudio {new_clears[0] - interrupted_at:.2f}s after speech")

                # The interrupting turn gets its own answer
                after_clear = len(caller.audio)
                assert await caller.wait_for_audio(after_clear, 1600, timeout=30), (
                    f"No answer to the interrupting turn\n{log_tail(LOG_PATH)}"
                )
            finally:
                await caller.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
