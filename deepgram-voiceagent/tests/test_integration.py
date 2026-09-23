"""
Integration tests for the Deepgram Voice Agent API + Plivo example.

Test Levels:
1. Unit Tests (offline) - audio conversion, phone normalization, Settings builder,
   function dispatch, Deepgram event handling with fake WebSockets, CallManager,
   server routes via FastAPI TestClient
2. Local Integration - start the inbound server, drive the Plivo WebSocket protocol
3. Deepgram Agent Integration - connect directly to the Deepgram Voice Agent API
4. Plivo Integration - validate Plivo credentials and phone number

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local or Deepgram or Plivo"
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import math
import os
import struct
import time
import uuid
from typing import Any

import httpx
import plivo
import pytest
import websockets
from dotenv import load_dotenv
from loguru import logger

from tests.helpers import server_log_path, start_server, stop_server
from utils import (
    deepgram_to_plivo,
    normalize_phone_number,
    pcm_to_ulaw,
    plivo_to_deepgram,
    resample_audio,
    ulaw_to_pcm,
)

load_dotenv()

# Configuration from environment
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

TEST_PORT = 18001
LOCAL_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"

STREAM_ID = "stream-0001"
CALL_ID = "call-1234-5678-90ab-cdef00000001"


# =============================================================================
# Fakes + fixtures for offline unit tests
# =============================================================================


class FakePlivoWS:
    """Stands in for the FastAPI WebSocket connected to Plivo."""

    def __init__(self, auto_played_stream: bool = False) -> None:
        self.incoming: asyncio.Queue[Any] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.auto_played_stream = auto_played_stream

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if isinstance(item, BaseException):
            raise item
        return item

    async def send_text(self, text: str) -> None:
        msg = json.loads(text)
        self.sent.append(msg)
        if self.auto_played_stream and msg.get("event") == "checkpoint":
            # Plivo acks the checkpoint once playback finishes, then the caller hangs up
            self.push({"event": "playedStream", "name": msg["name"], "streamId": STREAM_ID})
            self.push({"event": "stop"})

    def push(self, message: dict[str, Any]) -> None:
        self.incoming.put_nowait(json.dumps(message))

    def events(self, name: str) -> list[dict[str, Any]]:
        return [m for m in self.sent if m.get("event") == name]


class FakeDeepgramWS:
    """Stands in for the Deepgram Voice Agent WebSocket (recv, async-iter, send, close)."""

    def __init__(self, script: list[Any] | None = None) -> None:
        self.sent: list[Any] = []
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self.closed = False
        for item in script or []:
            self.feed(item)

    def feed(self, item: Any) -> None:
        """Queue a server message: dict -> JSON text, bytes -> binary audio, None -> EOF."""
        if isinstance(item, dict):
            item = json.dumps(item)
        self._queue.put_nowait(item)

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def recv(self) -> Any:
        item = await self._queue.get()
        if item is None:
            raise websockets.exceptions.ConnectionClosedOK(None, None)
        return item

    def __aiter__(self) -> FakeDeepgramWS:
        return self

    async def __anext__(self) -> Any:
        item = await self._queue.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def close(self) -> None:
        self.closed = True

    async def __aenter__(self) -> FakeDeepgramWS:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.closed = True

    def sent_json(self) -> list[dict[str, Any]]:
        return [json.loads(s) for s in self.sent if isinstance(s, str)]

    def sent_types(self) -> list[str]:
        return [m.get("type", "") for m in self.sent_json()]


@pytest.fixture
def captured_events():
    """Collect loguru ``extra`` dicts that carry a structured ``event`` key."""
    events: list[dict[str, Any]] = []

    def sink(message):
        extra = message.record["extra"]
        if "event" in extra:
            events.append(dict(extra))

    sink_id = logger.add(sink, level="DEBUG")
    yield events
    logger.remove(sink_id)


@pytest.fixture
def captured_messages():
    """Collect every log message text."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    yield messages
    logger.remove(sink_id)


def make_agent(
    stream_id: str = STREAM_ID,
    settings_applied: bool = True,
    hangup_callback: Any = None,
    **kwargs: Any,
):
    """Build an inbound agent wired to fake Plivo + Deepgram sockets."""
    from inbound.agent import DeepgramVoiceAgent

    plivo_ws = kwargs.pop("plivo_ws", None) or FakePlivoWS()
    dg_ws = kwargs.pop("dg_ws", None) or FakeDeepgramWS()
    agent = DeepgramVoiceAgent(
        websocket=plivo_ws,
        call_id=CALL_ID,
        from_number=kwargs.pop("from_number", "+15551234567"),
        to_number=kwargs.pop("to_number", "+16572338892"),
        stream_id=stream_id,
        hangup_callback=hangup_callback,
        **kwargs,
    )
    agent._dg_ws = dg_ws
    agent._running = True
    if settings_applied:
        agent._settings_applied.set()
    return agent, plivo_ws, dg_ws


class HangupRecorder:
    """Async hangup callback that counts invocations."""

    def __init__(self, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    async def __call__(self) -> None:
        self.calls += 1
        if self.fail:
            raise RuntimeError("REST hangup failed")


def of_event(events: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [e for e in events if e["event"] == name]


async def run_send_loop_until(agent, predicate, timeout: float = 2.0) -> None:
    """Run _send_to_plivo until predicate() is true, then stop it."""
    task = asyncio.create_task(agent._send_to_plivo())
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    agent._running = False
    await asyncio.wait_for(task, timeout=1.0)


def rms_of_ulaw(ulaw_audio: bytes) -> float:
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return (sum(s**2 for s in samples) / max(len(samples), 1)) ** 0.5


# =============================================================================
# UNIT TESTS - Audio conversion
# =============================================================================


class TestUnitAudioConversion:
    """Unit tests for audio format conversion."""

    def test_ulaw_to_pcm_conversion(self):
        pcm_audio = ulaw_to_pcm(b"\xff" * 160)
        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert sum(abs(s) for s in samples) / len(samples) < 100  # near silence

    def test_pcm_to_ulaw_conversion(self):
        assert len(pcm_to_ulaw(b"\x00" * 320)) == 160

    def test_audio_roundtrip(self):
        samples = [int(16000 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(160)]
        pcm_original = struct.pack(f"{len(samples)}h", *samples)
        restored = struct.unpack("160h", ulaw_to_pcm(pcm_to_ulaw(pcm_original)))

        corr = sum(o * r for o, r in zip(samples, restored, strict=True))
        energy = (sum(o * o for o in samples) * sum(r * r for r in restored)) ** 0.5
        assert corr / energy > 0.9, "Audio quality degraded too much"

    def test_plivo_to_deepgram_passthrough(self):
        """Deepgram is configured for μ-law 8kHz input, so Plivo bytes pass through."""
        data = bytes(range(160))
        assert plivo_to_deepgram(data) == data

    def test_deepgram_to_plivo_passthrough(self):
        """Deepgram emits raw μ-law 8kHz (container none), which Plivo plays as-is."""
        data = bytes(range(256)) * 2
        assert deepgram_to_plivo(data) == data

    def test_resample_identity(self):
        pcm = struct.pack("4h", 1, 2, 3, 4)
        assert resample_audio(pcm, 8000, 8000) == pcm

    def test_resample_doubles_length(self):
        pcm = b"\x00\x01" * 160
        assert len(resample_audio(pcm, 8000, 16000)) == len(pcm) * 2


# =============================================================================
# UNIT TESTS - Phone normalization
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for utils.normalize_phone_number (E.164 digits, no '+')."""

    def test_normalize_e164_format(self):
        assert normalize_phone_number("+16572338892") == "16572338892"

    def test_normalize_with_spaces(self):
        assert normalize_phone_number("+1 657-233-8892") == "16572338892"

    def test_normalize_local_format(self):
        assert normalize_phone_number("(657) 233-8892", "US") == "16572338892"

    def test_normalize_empty(self):
        assert normalize_phone_number("") == ""

    def test_normalize_unparseable_keeps_digits(self):
        assert normalize_phone_number("abc") == ""


# =============================================================================
# UNIT TESTS - Settings builder
# =============================================================================


def _find_keys(obj: Any, key: str) -> list[Any]:
    found = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                found.append(v)
            found.extend(_find_keys(v, key))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_keys(v, key))
    return found


class TestUnitSettingsBuilder:
    """Unit tests for DeepgramVoiceAgent._build_settings()."""

    def test_type_and_tags(self):
        from inbound.agent import EXAMPLE_NAME

        agent, _, _ = make_agent()
        settings = agent._build_settings()
        assert settings["type"] == "Settings"
        assert "plivo" in settings["tags"]
        assert EXAMPLE_NAME in settings["tags"]

    def test_audio_is_mulaw_8k_both_ways(self):
        agent, _, _ = make_agent()
        audio = agent._build_settings()["audio"]
        assert audio["input"] == {"encoding": "mulaw", "sample_rate": 8000}
        assert audio["output"] == {"encoding": "mulaw", "sample_rate": 8000, "container": "none"}

    def test_listen_is_flux_v2_with_eot(self):
        from inbound import agent as agent_mod

        agent, _, _ = make_agent()
        provider = agent._build_settings()["agent"]["listen"]["provider"]
        assert provider == {
            "type": "deepgram",
            "model": "flux-general-en",
            "version": "v2",
            "eot_threshold": agent_mod.DEEPGRAM_LISTEN_EOT_THRESHOLD,
            "eot_timeout_ms": agent_mod.DEEPGRAM_LISTEN_EOT_TIMEOUT_MS,
        }

    def test_listen_non_flux_omits_v2_and_eot(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_LISTEN_MODEL", "nova-3")
        monkeypatch.setattr(agent_mod, "DEEPGRAM_LISTEN_LANGUAGE", "")
        agent, _, _ = make_agent()
        provider = agent._build_settings()["agent"]["listen"]["provider"]
        assert provider == {"type": "deepgram", "model": "nova-3"}

    def test_listen_non_flux_language_passed_verbatim(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_LISTEN_MODEL", "nova-3")
        monkeypatch.setattr(agent_mod, "DEEPGRAM_LISTEN_LANGUAGE", "multi")
        agent, _, _ = make_agent()
        provider = agent._build_settings()["agent"]["listen"]["provider"]
        assert provider == {"type": "deepgram", "model": "nova-3", "language": "multi"}

    def test_think_open_ai_with_prompt_context(self):
        from inbound import agent as agent_mod

        agent, _, _ = make_agent(from_number="+15550001111")
        think = agent._build_settings()["agent"]["think"]
        assert think["provider"]["type"] == "open_ai"
        assert think["provider"]["model"] == agent_mod.DEEPGRAM_THINK_MODEL
        assert think["provider"]["temperature"] == agent_mod.DEEPGRAM_THINK_TEMPERATURE
        assert "+15550001111" in think["prompt"]
        assert "Current Call Context" in think["prompt"]

    def test_think_provider_passed_verbatim(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_THINK_PROVIDER", "anthropic")
        monkeypatch.setattr(agent_mod, "DEEPGRAM_THINK_MODEL", "claude-haiku-4-5")
        agent, _, _ = make_agent()
        provider = agent._build_settings()["agent"]["think"]["provider"]
        assert provider["type"] == "anthropic"
        assert provider["model"] == "claude-haiku-4-5"

    def test_prompt_without_caller_has_no_context(self):
        agent, _, _ = make_agent(from_number="")
        assert "Current Call Context" not in agent._build_settings()["agent"]["think"]["prompt"]

    def test_no_language_fields_anywhere(self):
        """A language field anywhere makes Deepgram reject Settings with a fatal Error."""
        agent, _, _ = make_agent()
        assert _find_keys(agent._build_settings(), "language") == []

    def test_functions_are_client_side(self):
        agent, _, _ = make_agent()
        functions = agent._build_settings()["agent"]["think"]["functions"]
        names = [f["name"] for f in functions]
        assert names == [
            "check_order_status",
            "send_sms",
            "schedule_callback",
            "transfer_call",
            "end_call",
        ]
        for fn in functions:
            assert "endpoint" not in fn, f"{fn['name']} must be client-side"
            assert fn["parameters"]["type"] == "object"
            assert fn["description"]
        deferred = {f["name"] for f in functions if f.get("defer_until_eot")}
        assert deferred == {"end_call", "transfer_call"}

    def test_speak_and_greeting(self):
        from inbound import agent as agent_mod

        agent, _, _ = make_agent(initial_message="Hello there from a test.")
        settings = agent._build_settings()
        assert settings["agent"]["speak"]["provider"] == {
            "type": "deepgram",
            "model": agent_mod.DEEPGRAM_SPEAK_MODEL,
        }
        assert settings["agent"]["greeting"] == "Hello there from a test."

    def test_speak_flux_tts_uses_v2(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_SPEAK_MODEL", "flux-alexis-en")
        agent, _, _ = make_agent()
        provider = agent._build_settings()["agent"]["speak"]["provider"]
        assert provider == {"type": "deepgram", "model": "flux-alexis-en", "version": "v2"}

    def test_default_greeting_is_literal_text(self):
        from inbound.agent import DEFAULT_GREETING

        agent, _, _ = make_agent()
        assert agent._build_settings()["agent"]["greeting"] == DEFAULT_GREETING
        assert "TechFlow" in DEFAULT_GREETING

    def test_settings_json_serializable(self):
        agent, _, _ = make_agent()
        assert json.loads(json.dumps(agent._build_settings()))["type"] == "Settings"


# =============================================================================
# UNIT TESTS - Function dispatch
# =============================================================================


class TestUnitFunctionDispatch:
    """Unit tests for _handle_function_call() and FunctionCallRequest handling."""

    async def test_check_order_status(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call(
            "check_order_status", json.dumps({"order_number": "TF-123456"})
        )
        assert result["order_number"] == "TF-123456"
        assert result["status"] in ("shipped", "processing", "delivered")

    async def test_check_order_status_needs_identifier(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call("check_order_status", "{}")
        assert result["status"] == "error"

    async def test_send_sms(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call(
            "send_sms", {"phone_number": "+15551234567", "message": "hi"}
        )
        assert result["status"] == "sent"
        assert result["confirmation_id"].startswith("SMS")

    async def test_schedule_callback(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call(
            "schedule_callback",
            json.dumps({"phone_number": "+15551234567", "reason": "billing", "department": "x"}),
        )
        assert result["status"] == "scheduled"
        assert result["scheduled_time"] == "within 2 business hours"

    async def test_transfer_call(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call(
            "transfer_call", json.dumps({"department": "sales", "reason": "pricing"})
        )
        assert result["status"] == "transferring"
        assert result["department"] == "sales"

    async def test_unknown_function_returns_error(self):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call("does_not_exist", "{}")
        assert "Unknown function" in result["error"]

    @pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", "", None])
    async def test_bad_arguments_tolerated(self, arguments):
        agent, _, _ = make_agent()
        result = await agent._handle_function_call("transfer_call", arguments)
        assert result["status"] == "transferring"
        assert result["department"] == "support"

    async def test_end_call_sets_hangup_pending_without_stopping(self):
        hangup = HangupRecorder()
        agent, _, _ = make_agent(hangup_callback=hangup)
        result = await agent._handle_function_call("end_call", json.dumps({"reason": "done"}))
        assert result["status"] == "call_ending"
        assert agent._hangup_pending is True
        assert agent._hangup_deadline is not None
        assert agent._running is True  # goodbye must still play
        assert hangup.calls == 0

    async def test_function_call_request_sends_response(self):
        agent, _, dg = make_agent()
        keep = await agent._handle_deepgram_event(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {
                        "id": "fn-1",
                        "name": "check_order_status",
                        "arguments": json.dumps({"order_number": "TF-999999"}),
                        "client_side": True,
                    }
                ],
            }
        )
        assert keep is True
        responses = [m for m in dg.sent_json() if m["type"] == "FunctionCallResponse"]
        assert len(responses) == 1
        response = responses[0]
        assert response["id"] == "fn-1"
        assert response["name"] == "check_order_status"
        assert isinstance(response["content"], str)
        assert json.loads(response["content"])["order_number"] == "TF-999999"

    async def test_server_side_function_is_skipped(self):
        agent, _, dg = make_agent()
        await agent._handle_deepgram_event(
            {
                "type": "FunctionCallRequest",
                "functions": [
                    {"id": "x", "name": "remote", "arguments": "{}", "client_side": False}
                ],
            }
        )
        assert dg.sent == []

    async def test_function_timeout_returns_error(self, monkeypatch):
        from inbound import agent as agent_mod

        async def slow(*_args, **_kwargs):
            await asyncio.sleep(5)

        agent, _, dg = make_agent()
        monkeypatch.setattr(agent, "_handle_function_call", slow)
        real_wait_for = asyncio.wait_for

        async def fast_wait_for(coro, timeout):
            return await real_wait_for(coro, timeout=0.05)

        monkeypatch.setattr(agent_mod.asyncio, "wait_for", fast_wait_for)
        await agent._on_function_call_request(
            {"functions": [{"id": "t", "name": "slow", "arguments": "{}", "client_side": True}]}
        )
        response = dg.sent_json()[-1]
        assert json.loads(response["content"]) == {"error": "function timed out"}
        assert agent._error_count == 1


# =============================================================================
# UNIT TESTS - Deepgram event handling (fake Plivo + Deepgram sockets)
# =============================================================================


class TestUnitDeepgramEventHandling:
    """Unit tests for the handshake, the 3 streaming tasks and Deepgram server events."""

    # -- handshake --

    async def test_handshake_sends_settings_after_welcome(self):
        agent, _, dg = make_agent(settings_applied=False)
        dg.feed({"type": "Welcome", "request_id": "req-1"})
        dg.feed({"type": "Welcome", "request_id": "req-1"})  # duplicate: Settings sent once
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        assert dg.sent_types() == ["Settings"]
        assert agent._settings_applied.is_set()
        assert agent._request_id == "req-1"

    async def test_handshake_error_raises(self):
        agent, _, dg = make_agent(settings_applied=False)
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "Error", "code": "INVALID_SETTINGS", "description": "bad"})
        with pytest.raises(RuntimeError, match="INVALID_SETTINGS"):
            await agent._handshake(dg)
        assert not agent._settings_applied.is_set()

    async def test_handshake_timeout_raises(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "SETTINGS_TIMEOUT_S", 0.2)
        agent, _, dg = make_agent(settings_applied=False)
        dg.feed({"type": "Welcome", "request_id": "r"})
        with pytest.raises(RuntimeError, match="timed out"):
            await agent._handshake(dg)

    async def test_handshake_ignores_binary(self):
        agent, _, dg = make_agent(settings_applied=False)
        dg.feed(b"\x00" * 10)
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        assert agent._send_queue.empty()

    async def test_media_buffered_until_settings_then_flushed_in_order(self):
        agent, plivo_ws, dg = make_agent(settings_applied=False)
        chunk_a, chunk_b = b"\x01" * 160, b"\x02" * 160
        for chunk in (chunk_a, chunk_b):
            plivo_ws.push(
                {"event": "media", "media": {"payload": base64.b64encode(chunk).decode()}}
            )
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()

        assert dg.sent == [], "nothing may be sent to Deepgram before SettingsApplied"
        assert bytes(agent._pre_settings_audio) == chunk_a + chunk_b

        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        assert json.loads(dg.sent[0])["type"] == "Settings"
        assert dg.sent[1] == chunk_a + chunk_b
        assert len(agent._pre_settings_audio) == 0

    async def test_pre_settings_buffer_is_capped(self):
        from inbound.agent import PRE_SETTINGS_BUFFER_MAX

        agent, plivo_ws, _ = make_agent(settings_applied=False)
        chunks = [bytes([i % 256]) * 160 for i in range(120)]  # 19200B > 16000B cap
        for chunk in chunks:
            plivo_ws.push(
                {"event": "media", "media": {"payload": base64.b64encode(chunk).decode()}}
            )
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        buffered = bytes(agent._pre_settings_audio)
        assert len(buffered) == PRE_SETTINGS_BUFFER_MAX
        assert buffered.endswith(b"".join(chunks[-5:])), "oldest audio is dropped, not newest"

    async def test_media_forwarded_after_settings(self):
        agent, plivo_ws, dg = make_agent()
        plivo_ws.push(
            {"event": "media", "media": {"payload": base64.b64encode(b"\x07" * 160).decode()}}
        )
        plivo_ws.push({"event": "media", "media": {"payload": ""}})  # empty payload ignored
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert dg.sent == [b"\x07" * 160]
        assert agent._plivo_rx_bytes == 160

    async def test_malformed_plivo_frames_do_not_end_call(self):
        """A non-JSON frame or bad base64 payload is skipped; the stream keeps going."""
        agent, plivo_ws, dg = make_agent()
        plivo_ws.incoming.put_nowait("{not json")
        plivo_ws.push({"event": "media", "media": {"payload": "!!!not-base64"}})
        plivo_ws.push(
            {"event": "media", "media": {"payload": base64.b64encode(b"\x07" * 160).decode()}}
        )
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert dg.sent == [b"\x07" * 160]
        assert agent._error_count == 0

    async def test_text_before_settings_is_injected_after(self):
        agent, plivo_ws, dg = make_agent(settings_applied=False)
        plivo_ws.push({"event": "text", "text": "hello agent"})
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert dg.sent == []
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        assert dg.sent_json()[1] == {"type": "InjectUserMessage", "content": "hello agent"}

    async def test_text_after_settings_is_injected(self, captured_events):
        agent, plivo_ws, dg = make_agent()
        plivo_ws.push({"event": "text", "text": "What plans do you offer?"})
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert dg.sent_json() == [
            {"type": "InjectUserMessage", "content": "What plans do you offer?"}
        ]
        user_texts = of_event(captured_events, "user_text")
        assert [(e["turn"], e["text"]) for e in user_texts] == [(1, "What plans do you offer?")]

    async def test_binary_audio_is_queued(self):
        agent, _, _ = make_agent()
        agent._on_agent_audio(b"\x10" * 400)
        assert agent._is_playing is True
        assert agent._send_queue.qsize() == 1
        assert agent._dg_rx_audio_bytes == 400

    async def test_send_to_plivo_chunks_and_checkpoint_after_audio(self):
        from inbound.agent import PLIVO_CHUNK_SIZE, _Checkpoint

        agent, plivo_ws, _ = make_agent()
        agent._send_queue.put_nowait(b"\x10" * 400)
        agent._send_queue.put_nowait(_Checkpoint("turn_1_1"))
        await run_send_loop_until(agent, lambda: plivo_ws.events("checkpoint"))

        assert [m["event"] for m in plivo_ws.sent] == [
            "playAudio",
            "playAudio",
            "playAudio",
            "checkpoint",
        ]
        payloads = [base64.b64decode(m["media"]["payload"]) for m in plivo_ws.events("playAudio")]
        assert all(len(p) == PLIVO_CHUNK_SIZE for p in payloads)
        assert payloads[2] == b"\x10" * 80 + b"\xff" * 80, "leftover padded with μ-law silence"
        for m in plivo_ws.events("playAudio"):
            assert m["media"]["contentType"] == "audio/x-mulaw"
            assert m["media"]["sampleRate"] == 8000
        assert plivo_ws.sent[-1] == {
            "event": "checkpoint",
            "streamId": STREAM_ID,
            "name": "turn_1_1",
        }
        assert agent._checkpoint_sent_time is not None

    async def test_no_checkpoint_without_stream_id(self):
        from inbound.agent import _Checkpoint

        agent, plivo_ws, _ = make_agent(stream_id="")
        agent._send_queue.put_nowait(b"\x10" * 200)
        agent._send_queue.put_nowait(_Checkpoint("cp"))
        await run_send_loop_until(agent, lambda: len(plivo_ws.events("playAudio")) >= 2)
        assert plivo_ws.events("checkpoint") == []
        assert len(plivo_ws.events("playAudio")) == 2

    async def test_barge_in_clears_queue_and_plivo(self, captured_events, captured_messages):
        agent, plivo_ws, _ = make_agent()
        agent._on_conversation_text("assistant", "Hi, this is Alex from TechFlow.")
        for _ in range(3):
            agent._on_agent_audio(b"\x10" * 320)
        agent._tx_buffer.extend(b"\x10" * 50)

        assert await agent._handle_deepgram_event({"type": "UserStartedSpeaking"}) is True

        assert agent._send_queue.empty()
        assert len(agent._tx_buffer) == 0
        assert plivo_ws.events("clearAudio") == [{"event": "clearAudio", "streamId": STREAM_ID}]
        assert agent._barge_in_count == 1
        assert agent._is_playing is False
        assert agent._drop_agent_audio is True
        assert agent._pending_checkpoint is None
        turn_completes = of_event(captured_events, "turn_complete")
        assert len(turn_completes) == 1
        assert turn_completes[0]["barge_in"] is True
        assert turn_completes[0]["turn"] == 1
        assert any("barge-in: cleared=3" in m for m in captured_messages)

    async def test_user_started_speaking_while_idle_not_counted(self, captured_events):
        agent, _, _ = make_agent()
        await agent._handle_deepgram_event({"type": "UserStartedSpeaking"})
        assert agent._barge_in_count == 0
        assert agent._drop_agent_audio is False
        assert of_event(captured_events, "turn_complete") == []

    async def test_late_audio_dropped_after_barge_in(self):
        agent, _, _ = make_agent()
        agent._on_agent_audio(b"\x10" * 160)
        await agent._on_user_started_speaking()
        agent._on_agent_audio(b"\x11" * 160)
        assert agent._send_queue.empty()
        # ConversationText(assistant) does NOT reopen the gate (tail of the cancelled reply)
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "assistant", "content": "late sentence"}
        )
        agent._on_agent_audio(b"\x12" * 160)
        assert agent._send_queue.empty()
        assert agent._is_playing is False

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "ConversationText", "role": "user", "content": "new question"},
            {"type": "EndOfTurn", "trigger": "eot"},
            {"type": "AgentThinking", "content": "thinking"},
            {"type": "AgentStartedSpeaking", "total_latency": 1.0},
            {"type": "FunctionCallRequest", "functions": []},
        ],
        ids=lambda e: e["type"] + (f"-{e['role']}" if "role" in e else ""),
    )
    async def test_drop_gate_cleared_by_next_response_events(self, event):
        agent, _, _ = make_agent()
        agent._on_agent_audio(b"\x10" * 160)
        await agent._on_user_started_speaking()
        assert agent._drop_agent_audio is True
        await agent._handle_deepgram_event(event)
        assert agent._drop_agent_audio is False
        agent._on_agent_audio(b"\x13" * 160)
        assert agent._send_queue.qsize() == 1

    async def test_agent_audio_done_after_barge_in_queues_nothing(self):
        agent, _, _ = make_agent()
        agent._on_agent_audio(b"\x10" * 160)
        await agent._on_user_started_speaking()
        await agent._handle_deepgram_event({"type": "AgentAudioDone"})
        assert agent._send_queue.empty()
        assert agent._pending_checkpoint is None

    async def test_played_stream_emits_turn_complete_once(self, captured_events):
        from inbound.agent import _Checkpoint

        agent, plivo_ws, _ = make_agent()
        agent._on_conversation_text("assistant", "Hi there.")
        agent._on_agent_audio(b"\x10" * 320)
        await agent._handle_deepgram_event({"type": "AgentAudioDone"})
        checkpoint = agent._pending_checkpoint
        assert isinstance(checkpoint, _Checkpoint)
        assert checkpoint.name == "turn_1_1"
        assert agent._send_queue.qsize() == 2  # audio then checkpoint sentinel

        plivo_ws.push({"event": "playedStream", "name": "stale_name"})
        plivo_ws.push({"event": "playedStream", "name": checkpoint.name})
        plivo_ws.push({"event": "playedStream", "name": checkpoint.name})
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()

        turn_completes = of_event(captured_events, "turn_complete")
        assert len(turn_completes) == 1
        assert turn_completes[0]["barge_in"] is False
        assert turn_completes[0]["agent_text"] == "Hi there."
        assert agent._is_playing is False
        assert agent._pending_checkpoint is None

    async def test_stale_played_stream_ignored(self, captured_events):
        agent, _, _ = make_agent()
        agent._on_agent_audio(b"\x10" * 160)
        await agent._on_agent_audio_done()
        await agent._on_played_stream("turn_9_9")
        assert of_event(captured_events, "turn_complete") == []
        assert agent._pending_checkpoint is not None
        assert agent._is_playing is True

    async def test_cleared_audio_event_stops_playing(self):
        agent, plivo_ws, _ = make_agent()
        agent._is_playing = True
        plivo_ws.push({"event": "clearedAudio"})
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert agent._is_playing is False

    async def test_conversation_text_emits_user_and_agent_text(self, captured_events):
        agent, _, _ = make_agent()
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "assistant", "content": "Greeting."}
        )
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "user", "content": "What plans?"}
        )
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "assistant", "content": "We have three."}
        )
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "assistant", "content": "Pro is popular."}
        )
        agent_texts = of_event(captured_events, "agent_text")
        user_texts = of_event(captured_events, "user_text")
        assert [(e["turn"], e["text"]) for e in agent_texts] == [
            (1, "Greeting."),
            (2, "We have three."),
            (2, "Pro is popular."),
        ]
        assert [(e["turn"], e["text"]) for e in user_texts] == [(2, "What plans?")]
        assert all(e["call_id"] == CALL_ID for e in agent_texts + user_texts)
        assert agent._turn_agent_text == "We have three. Pro is popular."

    async def test_injected_text_not_counted_twice(self, captured_events):
        agent, _, dg = make_agent()
        await agent._inject_user_text("Check my order")
        # Deepgram echoes the injected text back as ConversationText(user)
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "user", "content": "Check my order"}
        )
        assert agent._turn_count == 1
        assert len(of_event(captured_events, "user_text")) == 1
        # A genuinely new spoken turn still counts
        await agent._handle_deepgram_event(
            {"type": "ConversationText", "role": "user", "content": "Something else"}
        )
        assert agent._turn_count == 2
        assert dg.sent_types() == ["InjectUserMessage"]

    async def test_inject_during_playback_is_local_barge_in(self, captured_events):
        agent, plivo_ws, dg = make_agent()
        agent._on_conversation_text("assistant", "Long answer...")
        agent._on_agent_audio(b"\x10" * 1600)
        await agent._inject_user_text("Actually, stop")
        turn_completes = of_event(captured_events, "turn_complete")
        assert len(turn_completes) == 1
        assert turn_completes[0]["barge_in"] is True
        assert turn_completes[0]["turn"] == 1, "barge-in belongs to the interrupted turn"
        assert agent._turn_count == 2
        assert plivo_ws.events("clearAudio")
        assert agent._barge_in_count == 1
        assert dg.sent_types() == ["InjectUserMessage"]

    async def test_latency_report_merged_into_turn_complete(self, captured_events):
        agent, _, _ = make_agent(stream_id="")
        agent._start_user_turn("What plans do you offer?")
        for report in (
            {"type": "LatencyReport", "stt_latency": 0.05},
            {"type": "LatencyReport", "ttt_token_latency": 0.653},
            {"type": "LatencyReport", "ttt_text_latency": 0.8},
            {"type": "LatencyReport", "tts_latency": 0.119},
            {"type": "LatencyReport", "total_latency": 1.122},
            {"type": "LatencyReport", "flag": True, "note": "ignored"},
        ):
            await agent._handle_deepgram_event(report)
        agent._on_agent_audio(b"\x10" * 160)
        await agent._handle_deepgram_event({"type": "AgentAudioDone"})

        (turn_complete,) = of_event(captured_events, "turn_complete")
        assert turn_complete["total_latency_ms"] == 1122
        assert turn_complete["tts_latency_ms"] == 119
        assert turn_complete["ttt_latency_ms"] == 800
        assert turn_complete["latency_report"]["stt_latency_ms"] == 50
        assert turn_complete["latency_report"]["ttt_token_latency_ms"] == 653
        assert "flag" not in turn_complete["latency_report"]

    async def test_agent_started_speaking_latency_in_ms(self, captured_events):
        agent, _, _ = make_agent(stream_id="")
        agent._start_user_turn("hi")
        await agent._handle_deepgram_event(
            {
                "type": "AgentStartedSpeaking",
                "total_latency": 1.5,
                "tts_latency": 0.2,
                "ttt_latency": 0.9,
            }
        )
        agent._on_agent_audio(b"\x10" * 160)
        await agent._on_agent_audio_done()
        (turn_complete,) = of_event(captured_events, "turn_complete")
        assert turn_complete["total_latency_ms"] == 1500
        assert turn_complete["tts_latency_ms"] == 200
        assert turn_complete["ttt_latency_ms"] == 900

    async def test_latency_resets_each_user_turn(self):
        agent, _, _ = make_agent()
        await agent._handle_deepgram_event({"type": "LatencyReport", "total_latency": 1.0})
        agent._start_user_turn("next")
        assert agent._turn_latency["total_latency_ms"] is None
        assert agent._turn_latency_report == {}

    async def test_ttfs_measured_from_user_turn_to_first_chunk(self):
        agent, plivo_ws, _ = make_agent()
        agent._start_user_turn("hello")
        agent._send_queue.put_nowait(b"\x10" * 320)
        await run_send_loop_until(agent, lambda: len(plivo_ws.events("playAudio")) >= 2)
        assert len(agent._ttfs_samples) == 1
        assert agent._speech_end_time is None

    async def test_end_call_hangs_up_after_played_stream(self):
        hangup = HangupRecorder()
        agent, _, _ = make_agent(hangup_callback=hangup)
        await agent._handle_function_call("end_call", json.dumps({"reason": "done"}))
        agent._on_agent_audio(b"\x10" * 800)  # goodbye audio
        await agent._on_agent_audio_done()
        assert hangup.calls == 0, "must not hang up before the goodbye has played"
        assert agent._running is True

        await agent._on_played_stream(agent._pending_checkpoint.name)
        assert hangup.calls == 1
        assert agent._running is False
        assert agent._hangup_done is True

    async def test_end_call_without_stream_id_hangs_up_on_audio_done(self, captured_events):
        hangup = HangupRecorder()
        agent, _, _ = make_agent(stream_id="", hangup_callback=hangup)
        await agent._handle_function_call("end_call", "{}")
        agent._on_agent_audio(b"\x10" * 800)
        await agent._on_agent_audio_done()
        assert hangup.calls == 1
        assert agent._running is False
        assert len(of_event(captured_events, "turn_complete")) == 1

    async def test_hangup_deadline_fallback(self):
        hangup = HangupRecorder()
        agent, _, _ = make_agent(hangup_callback=hangup)
        await agent._handle_function_call("end_call", "{}")
        agent._on_agent_audio(b"\x10" * 800)
        await agent._on_agent_audio_done()  # checkpoint queued, playedStream never comes
        await agent._idle_housekeeping()
        assert hangup.calls == 0
        agent._hangup_deadline = time.monotonic() - 0.1
        await agent._idle_housekeeping()
        assert hangup.calls == 1
        assert agent._running is False

    async def test_hangup_is_idempotent_and_survives_callback_error(self):
        hangup = HangupRecorder(fail=True)
        agent, _, _ = make_agent(hangup_callback=hangup)
        await agent._finish_hangup()
        await agent._finish_hangup()
        assert hangup.calls == 1
        assert agent._error_count == 1
        assert agent._running is False

    async def test_keepalive_sent_when_idle(self):
        agent, _, dg = make_agent()
        agent._last_dg_send = time.monotonic()
        await agent._idle_housekeeping()
        assert dg.sent == []
        agent._last_dg_send = time.monotonic() - 10
        await agent._idle_housekeeping()
        assert dg.sent_json() == [{"type": "KeepAlive"}]

    async def test_no_keepalive_before_settings_applied(self):
        agent, _, dg = make_agent(settings_applied=False)
        agent._last_dg_send = time.monotonic() - 10
        await agent._idle_housekeeping()
        assert dg.sent == []

    async def test_error_event_ends_session(self):
        agent, _, _ = make_agent()
        keep = await agent._handle_deepgram_event(
            {"type": "Error", "code": "BAD", "description": "boom"}
        )
        assert keep is False
        assert agent._error_count == 1

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "Warning", "code": "W1", "description": "careful"},
            {"type": "InjectionRefused", "message": "busy"},
            {"type": "History", "role": "user", "content": "x"},
            {"type": "FunctionCallCancelled", "id": "x"},
            {"type": "SomethingNew", "value": 1},
        ],
        ids=lambda e: e["type"],
    )
    async def test_non_fatal_events(self, event):
        agent, _, _ = make_agent()
        assert await agent._handle_deepgram_event(event) is True
        assert agent._error_count == 0

    async def test_receive_from_deepgram_full_flow(self):
        agent, _, dg = make_agent(settings_applied=False)
        for item in (
            {"type": "Welcome", "request_id": "req-9"},
            {"type": "SettingsApplied"},
            {"type": "ConversationText", "role": "assistant", "content": "Hello!"},
            b"\x10" * 480,
            "not json",
            {"type": "AgentAudioDone"},
            {"type": "Error", "code": "X", "description": "stop here"},
            b"\x20" * 160,  # never reached
        ):
            dg.feed(item)
        await asyncio.wait_for(agent._receive_from_deepgram(dg), timeout=2)
        assert agent._settings_applied.is_set()
        assert agent._send_queue.qsize() == 2  # audio + checkpoint
        assert agent._turn_count == 1

    async def test_receive_from_deepgram_stops_on_eof(self):
        agent, _, dg = make_agent(settings_applied=False)
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        dg.feed(None)
        await asyncio.wait_for(agent._receive_from_deepgram(dg), timeout=2)

    async def test_plivo_tx_disconnect_is_not_an_error(self):
        """Caller hangs up mid-playback: send_text raises WebSocketDisconnect (empty str)."""
        from starlette.websockets import WebSocketDisconnect

        agent, plivo_ws, _ = make_agent()

        async def closed_send_text(_text):
            raise WebSocketDisconnect(code=1006)

        plivo_ws.send_text = closed_send_text
        agent._send_queue.put_nowait(b"\x10" * 160)
        await asyncio.wait_for(agent._send_to_plivo(), timeout=1)
        assert agent._error_count == 0

    async def test_plivo_tx_unexpected_error_propagates(self):
        agent, plivo_ws, _ = make_agent()

        async def broken_send_text(_text):
            raise ValueError("boom")

        plivo_ws.send_text = broken_send_text
        agent._send_queue.put_nowait(b"\x10" * 160)
        with pytest.raises(ValueError):
            await asyncio.wait_for(agent._send_to_plivo(), timeout=1)

    async def test_plivo_rx_disconnect_is_not_an_error(self):
        from starlette.websockets import WebSocketDisconnect

        agent, plivo_ws, _ = make_agent()
        plivo_ws.incoming.put_nowait(WebSocketDisconnect(code=1000))
        await agent._receive_from_plivo()
        assert agent._error_count == 0

    async def test_plivo_rx_unexpected_error_counted(self):
        agent, plivo_ws, _ = make_agent()
        plivo_ws.incoming.put_nowait(RuntimeError("socket exploded"))
        await agent._receive_from_plivo()
        assert agent._error_count == 1

    async def test_session_end_fields(self, captured_events):
        agent, _, _ = make_agent()
        agent._request_id = "req-5"
        agent._ttfs_samples = [1000.0, 1200.0]
        agent._emit_session_end()
        (session_end,) = of_event(captured_events, "session_end")
        for key in (
            "call_id",
            "duration_s",
            "turns",
            "barge_ins",
            "errors",
            "ttfs_avg_ms",
            "ttfs_samples",
            "rx_bytes",
            "tx_chunks",
            "deepgram_request_id",
        ):
            assert key in session_end, key
        assert session_end["ttfs_avg_ms"] == 1100
        assert session_end["deepgram_request_id"] == "req-5"

    async def test_parent_call_id_used_for_events(self, captured_events):
        agent, _, _ = make_agent(parent_call_id="parent-uuid")
        agent._on_conversation_text("assistant", "hi")
        assert of_event(captured_events, "agent_text")[0]["call_id"] == "parent-uuid"


# =============================================================================
# UNIT TESTS - Outbound CallManager
# =============================================================================


class TestUnitCallManager:
    """Unit tests for outbound call records, prompts and greetings."""

    def test_literal_greeting_with_opening_reason(self):
        from outbound.agent import CallManager

        record = CallManager().create_call(
            phone_number="+15551234567", opening_reason="you requested a demo"
        )
        assert record.initial_message.startswith("Hi, this is Alex from TechFlow.")
        assert "you requested a demo" in record.initial_message
        assert record.status == "initiating"

    def test_default_greeting_without_reason(self):
        from outbound.agent import DEFAULT_OUTBOUND_GREETING, CallManager

        record = CallManager().create_call(phone_number="+15551234567")
        assert record.initial_message == DEFAULT_OUTBOUND_GREETING

    def test_prompt_substitution(self):
        from outbound.agent import CallManager

        record = CallManager().create_call(
            phone_number="+15551234567",
            opening_reason="your trial ends soon",
            objective="book a renewal call",
            context="customer since 2024",
        )
        assert "{{" not in record.system_prompt
        assert "your trial ends soon" in record.system_prompt
        assert "book a renewal call" in record.system_prompt

    def test_outbound_settings_use_record(self):
        from outbound.agent import CallManager, DeepgramVoiceAgent

        record = CallManager().create_call(phone_number="+1555", opening_reason="a demo")
        agent = DeepgramVoiceAgent(
            websocket=FakePlivoWS(),
            call_id="c",
            system_prompt=record.system_prompt,
            initial_message=record.initial_message,
        )
        settings = agent._build_settings()
        assert settings["agent"]["greeting"] == record.initial_message
        assert settings["agent"]["think"]["prompt"].startswith(record.system_prompt)
        assert _find_keys(settings, "language") == []

    def test_status_and_active_calls(self):
        from outbound.agent import CallManager

        manager = CallManager()
        a = manager.create_call(phone_number="+1")
        b = manager.create_call(phone_number="+2")
        manager.update_status(a.call_id, "connected", plivo_call_uuid="uuid-a", bogus="x")
        manager.update_status(b.call_id, "completed", outcome="success")
        assert manager.get_call(a.call_id).plivo_call_uuid == "uuid-a"
        assert not hasattr(manager.get_call(a.call_id), "bogus")
        assert [r.call_id for r in manager.get_active_calls()] == [a.call_id]
        assert manager.update_status("missing", "completed") is None
        assert manager.get_call("missing") is None

    def test_calls_by_campaign(self):
        from outbound.agent import CallManager

        manager = CallManager()
        manager.create_call(phone_number="+1", campaign_id="c1")
        manager.create_call(phone_number="+2", campaign_id="c1")
        manager.create_call(phone_number="+3", campaign_id="c2")
        assert len(manager.get_calls_by_campaign("c1")) == 2
        assert len(manager.get_calls_by_campaign("c3")) == 0

    @pytest.mark.parametrize(
        ("cause", "duration", "expected"),
        [
            ("NO_ANSWER", 0, "no_answer"),
            ("ORIGINATOR_CANCEL", 0, "no_answer"),
            ("USER_BUSY", 0, "busy"),
            ("CALL_REJECTED", 0, "busy"),
            ("UNALLOCATED_NUMBER", 0, "failed"),
            ("NORMAL_CLEARING", 42, "success"),
            ("", 0, "success"),
            ("SOMETHING_ELSE", 0, "failed"),
            ("something_else", 5, "success"),
        ],
    )
    def test_determine_outcome(self, cause, duration, expected):
        from outbound.agent import determine_outcome

        assert determine_outcome(cause, duration) == expected

    def test_reset(self):
        from outbound.agent import CallManager

        manager = CallManager()
        record = manager.create_call(phone_number="+1")
        manager.reset()
        assert manager.get_call(record.call_id) is None


# =============================================================================
# UNIT TESTS - Server routes (FastAPI TestClient, no network)
# =============================================================================


class TestUnitServerRoutes:
    """Offline tests for the inbound/outbound FastAPI routes."""

    def test_inbound_answer_returns_stream_xml(self, monkeypatch):
        from fastapi.testclient import TestClient

        from inbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.ngrok.app")
        client = TestClient(server.app)
        resp = client.post(
            "/answer",
            data={"CallUUID": "uuid-1", "From": "+15551234567", "To": "+16572338892"},
        )
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        xml = resp.text
        assert "<Stream" in xml
        assert 'bidirectional="true"' in xml
        assert 'keepCallAlive="true"' in xml
        assert "audio/x-mulaw;rate=8000" in xml
        assert "wss://example.ngrok.app/ws?body=" in xml
        body = xml.split("body=")[1].split("<")[0]
        meta = json.loads(base64.b64decode(body))
        assert meta["call_uuid"] == "uuid-1"
        assert meta["from"] == "+15551234567"

    def test_inbound_health(self):
        from fastapi.testclient import TestClient

        from inbound import server

        assert TestClient(server.app).get("/").json()["status"] == "ok"

    async def test_rest_hangup_skipped_without_credentials(self, monkeypatch):
        from inbound import server

        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "")
        await server._hangup_call("uuid-1")  # must not raise

    async def test_rest_hangup_calls_plivo(self, monkeypatch):
        from inbound import server

        deleted: list[str] = []

        class FakeCalls:
            def delete(self, call_uuid):
                deleted.append(call_uuid)

        class FakeRestClient:
            def __init__(self, auth_id, auth_token):
                self.calls = FakeCalls()

        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "id")
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "token")
        monkeypatch.setattr(server.plivo, "RestClient", FakeRestClient)
        await server._hangup_call("uuid-42")
        assert deleted == ["uuid-42"]

    def test_outbound_answer_marks_connected(self, monkeypatch):
        from fastapi.testclient import TestClient

        from outbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://example.ngrok.app")
        record = server.call_manager.create_call(phone_number="+15551234567")
        client = TestClient(server.app)
        resp = client.post(
            f"/outbound/answer?call_id={record.call_id}",
            data={"CallUUID": "a-leg-uuid", "To": "+15551234567"},
        )
        assert "<Stream" in resp.text
        assert "audio/x-mulaw;rate=8000" in resp.text
        meta = json.loads(base64.b64decode(resp.text.split("body=")[1].split("<")[0]))
        assert meta["is_outbound"] is True
        assert meta["call_id"] == record.call_id

        status = client.get(f"/outbound/status/{record.call_id}").json()
        assert status["status"] == "connected"
        assert status["plivo_call_uuid"] == "a-leg-uuid"

    def test_outbound_call_requires_phone_number(self):
        from fastapi.testclient import TestClient

        from outbound import server

        resp = TestClient(server.app).post("/outbound/call")
        assert resp.json() == {"error": "phone_number is required"}

    def test_outbound_status_unknown(self):
        from fastapi.testclient import TestClient

        from outbound import server

        assert TestClient(server.app).get("/outbound/status/nope").json() == {
            "error": "Call not found"
        }


# =============================================================================
# UNIT TESTS - Saved agent configurations (opt-in DEEPGRAM_{INBOUND,OUTBOUND}_AGENT_ID)
# =============================================================================

SAVED_UUID = "11111111-2222-3333-4444-555555555555"


def make_outbound_agent(**kwargs: Any):
    """Build an outbound agent wired to fake Plivo + Deepgram sockets."""
    from outbound.agent import DeepgramVoiceAgent

    dg_ws = FakeDeepgramWS()
    agent = DeepgramVoiceAgent(
        websocket=FakePlivoWS(),
        call_id=CALL_ID,
        from_number=kwargs.pop("from_number", "+15551234567"),
        stream_id=STREAM_ID,
        **kwargs,
    )
    agent._dg_ws = dg_ws
    agent._running = True
    return agent, dg_ws


class TestUnitQuickTunnel:
    """``--tunnel`` helpers (utils.start_quick_tunnel & friends) with a fake cloudflared."""

    SAMPLE_LINE = (
        "2026-09-23T10:00:00Z INF |  https://quiet-river-demo-42.trycloudflare.com"
        "                                  |"
    )

    @staticmethod
    def _fake_cloudflared(tmp_path, body: str):
        script = tmp_path / "cloudflared"
        script.write_text(f"#!/bin/sh\n{body}\n")
        script.chmod(0o755)
        return str(script)

    def test_parse_tunnel_url(self):
        from utils import parse_tunnel_url

        assert parse_tunnel_url(self.SAMPLE_LINE) == (
            "https://quiet-river-demo-42.trycloudflare.com"
        )
        assert parse_tunnel_url("INF Registered tunnel connection connIndex=0") == ""

    def test_missing_cloudflared_raises_with_install_link(self, monkeypatch):
        import utils

        monkeypatch.setattr(utils.shutil, "which", lambda _name: None)
        with pytest.raises(utils.TunnelError, match="cloudflared not found on PATH") as exc:
            utils.start_quick_tunnel(8000)
        assert utils.CLOUDFLARED_INSTALL_URL in str(exc.value)

    def test_start_returns_url_and_stop_terminates(self, monkeypatch, tmp_path):
        import utils

        fake = self._fake_cloudflared(
            tmp_path,
            'echo "starting" >&2\n'
            'echo "INF |  https://abc-def-1.trycloudflare.com  |" >&2\n'
            "exec sleep 30",
        )
        monkeypatch.setattr(utils.shutil, "which", lambda _name: fake)
        url, proc = utils.start_quick_tunnel(8123, timeout_s=5)
        try:
            assert url == "https://abc-def-1.trycloudflare.com"
            assert proc.poll() is None
            assert proc.args[-2:] == ["--url", "http://localhost:8123"]
        finally:
            utils.stop_tunnel(proc)
        assert proc.poll() is not None
        utils.stop_tunnel(proc)  # idempotent

    def test_no_url_times_out_and_cleans_up(self, monkeypatch, tmp_path):
        import utils

        fake = self._fake_cloudflared(tmp_path, "exec sleep 30")
        monkeypatch.setattr(utils.shutil, "which", lambda _name: fake)
        spawned = []
        real_popen = utils.subprocess.Popen

        def tracking_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            spawned.append(proc)
            return proc

        monkeypatch.setattr(utils.subprocess, "Popen", tracking_popen)
        with pytest.raises(utils.TunnelError, match="did not report a public URL"):
            utils.start_quick_tunnel(8000, timeout_s=0.5)
        assert spawned and spawned[0].poll() is not None

    def test_process_exiting_early_raises(self, monkeypatch, tmp_path):
        import utils

        fake = self._fake_cloudflared(tmp_path, 'echo "ERR failed to request tunnel" >&2')
        monkeypatch.setattr(utils.shutil, "which", lambda _name: fake)
        with pytest.raises(utils.TunnelError):
            utils.start_quick_tunnel(8000, timeout_s=5)

    def test_wait_until_reachable_gives_up(self):
        from utils import wait_until_reachable

        assert not wait_until_reachable("http://127.0.0.1:9/", timeout_s=0.3, interval_s=0.1)

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    def test_start_tunnel_sets_public_url(self, monkeypatch, module):
        import importlib

        server = importlib.import_module(module)
        monkeypatch.setattr(server, "PUBLIC_URL", "")
        monkeypatch.setattr(
            server, "start_quick_tunnel", lambda port: ("https://t-1.trycloudflare.com", None)
        )
        registered = []
        monkeypatch.setattr(server.atexit, "register", lambda *a: registered.append(a))
        server._start_tunnel()
        assert server.PUBLIC_URL == "https://t-1.trycloudflare.com"
        assert registered and registered[0][0] is server.stop_tunnel

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    def test_start_tunnel_failure_exits(self, monkeypatch, module):
        import importlib

        import utils

        server = importlib.import_module(module)

        def boom(port):
            raise utils.TunnelError("cloudflared not found on PATH")

        monkeypatch.setattr(server, "start_quick_tunnel", boom)
        with pytest.raises(SystemExit):
            server._start_tunnel()

    async def test_answer_uses_tunnel_url_for_stream(self, monkeypatch):
        """After --tunnel sets PUBLIC_URL, /answer streams to the tunnel's wss:// URL."""
        from inbound import server

        monkeypatch.setattr(server, "PUBLIC_URL", "https://t-2.trycloudflare.com")
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/answer", data={"CallUUID": "c-1", "From": "+15551230000", "To": "+15557654321"}
            )
        assert resp.status_code == 200
        assert "wss://t-2.trycloudflare.com/ws?body=" in resp.text


class TestUnitSavedAgentConfig:
    """Saved-mode Settings, build_agent_config() and the saved-mode handshake."""

    def test_saved_settings_reference_uuid_only(self):
        agent, _, _ = make_agent(agent_config_id=SAVED_UUID)
        settings = agent._build_settings()
        inline = make_agent()[0]._build_settings()
        assert settings == {
            "type": "Settings",
            "tags": inline["tags"],
            "audio": inline["audio"],
            "agent": SAVED_UUID,
        }
        assert json.loads(json.dumps(settings))["agent"] == SAVED_UUID

    def test_agent_config_id_defaults_to_env(self, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_INBOUND_AGENT_ID", SAVED_UUID)
        agent, _, _ = make_agent()
        assert agent._build_settings()["agent"] == SAVED_UUID
        assert agent._settings_mode() == f"saved agent config {SAVED_UUID}"
        monkeypatch.setattr(agent_mod, "DEEPGRAM_INBOUND_AGENT_ID", "")
        assert make_agent()[0]._settings_mode() == "inline"

    @pytest.mark.parametrize("direction", ["inbound", "outbound"])
    def test_build_agent_config_is_static(self, direction):
        import importlib

        agent_mod = importlib.import_module(f"{direction}.agent")
        config = agent_mod.build_agent_config()
        assert set(config) == {"listen", "think", "speak"}, "no greeting in a saved config"
        assert config["think"]["functions"] == agent_mod.FUNCTION_DEFINITIONS
        prompt = config["think"]["prompt"]
        assert "Current Call Context" not in prompt
        assert "{{" not in prompt
        assert _find_keys(config, "language") == []
        json.dumps(config)

    def test_inbound_config_prompt_is_system_prompt(self):
        from inbound.agent import SYSTEM_PROMPT, build_agent_config

        assert build_agent_config()["think"]["prompt"] == SYSTEM_PROMPT

    def test_outbound_config_prompt_points_at_this_call(self):
        from outbound.agent import build_agent_config

        prompt = build_agent_config()["think"]["prompt"]
        assert prompt.count('"This Call"') >= 3  # opening reason, objective, context

    def test_inline_settings_are_agent_config_plus_greeting_and_context(self):
        from inbound.agent import SYSTEM_PROMPT, build_agent_config

        agent, _, _ = make_agent(initial_message="Hello from a test.")
        context = agent._build_call_context()
        assert "+15551234567" in context
        assert agent._build_settings()["agent"] == {
            **build_agent_config(prompt=SYSTEM_PROMPT + context),
            "greeting": "Hello from a test.",
        }

    def test_outbound_inline_settings_are_agent_config_plus_greeting_and_context(self):
        from outbound.agent import CallManager, build_agent_config

        record = CallManager().create_call(phone_number="+1555", opening_reason="a demo")
        agent, _ = make_outbound_agent(
            system_prompt=record.system_prompt,
            initial_message=record.initial_message,
            opening_reason="a demo",
        )
        assert agent._build_settings()["agent"] == {
            **build_agent_config(prompt=record.system_prompt + agent._build_call_context()),
            "greeting": record.initial_message,
        }

    async def test_saved_handshake_sends_context_then_greeting_then_flushes(self):
        agent, plivo_ws, dg = make_agent(
            settings_applied=False, agent_config_id=SAVED_UUID, initial_message="Hi there."
        )
        chunk = b"\x01" * 160
        plivo_ws.push({"event": "media", "media": {"payload": base64.b64encode(chunk).decode()}})
        plivo_ws.push({"event": "text", "text": "hello agent"})
        plivo_ws.push({"event": "stop"})
        await agent._receive_from_plivo()
        assert dg.sent == []

        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)

        settings, update, greeting = (json.loads(m) for m in dg.sent[:3])
        assert settings == {**settings, "type": "Settings", "agent": SAVED_UUID}
        assert update["type"] == "UpdatePrompt"
        assert "+15551234567" in update["prompt"]
        assert "Current Call Context" in update["prompt"]
        assert greeting == {"type": "InjectAgentMessage", "message": "Hi there."}
        assert dg.sent[3] == chunk, "buffered audio is flushed after the greeting"
        assert json.loads(dg.sent[4]) == {"type": "InjectUserMessage", "content": "hello agent"}
        assert agent._settings_applied.is_set()

    async def test_inline_handshake_sends_no_personalization(self):
        agent, _, dg = make_agent(settings_applied=False)
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        assert dg.sent_types() == ["Settings"]

    async def test_saved_handshake_without_caller_skips_update_prompt(self):
        agent, _, dg = make_agent(
            settings_applied=False, agent_config_id=SAVED_UUID, from_number=""
        )
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        assert dg.sent_types() == ["Settings", "InjectAgentMessage"]

    async def test_saved_greeting_counts_as_turn_1(self, captured_events):
        """InjectAgentMessage comes back as ConversationText(assistant), like agent.greeting."""
        agent, _, dg = make_agent(stream_id="", settings_applied=False, agent_config_id=SAVED_UUID)
        for item in (
            {"type": "Welcome", "request_id": "req-s"},
            {"type": "SettingsApplied"},
            {"type": "PromptUpdated"},
            {"type": "ConversationText", "role": "assistant", "content": "Hi there."},
            b"\x10" * 480,
            {"type": "AgentAudioDone"},
            None,
        ):
            dg.feed(item)
        await asyncio.wait_for(agent._receive_from_deepgram(dg), timeout=2)
        assert [(e["turn"], e["text"]) for e in of_event(captured_events, "agent_text")] == [
            (1, "Hi there.")
        ]
        (turn,) = of_event(captured_events, "turn_complete")
        assert (turn["turn"], turn["barge_in"], turn["agent_text"]) == (1, False, "Hi there.")

    @pytest.mark.parametrize(("config_id", "expected"), [(SAVED_UUID, SAVED_UUID), ("", "inline")])
    def test_session_end_reports_agent_config(self, captured_events, config_id, expected):
        agent, _, _ = make_agent(agent_config_id=config_id)
        agent._emit_session_end()
        (session_end,) = of_event(captured_events, "session_end")
        assert session_end["agent_config"] == expected

    async def test_outbound_saved_mode_appends_campaign_details(self):
        agent, dg = make_outbound_agent(
            agent_config_id=SAVED_UUID,
            initial_message="Hi, this is Alex from TechFlow.",
            opening_reason="your trial ends soon",
            objective="book a renewal call",
            context="customer since 2024",
        )
        agent._settings_applied.clear()
        dg.feed({"type": "Welcome", "request_id": "r"})
        dg.feed({"type": "SettingsApplied"})
        await agent._handshake(dg)
        settings, update, greeting = dg.sent_json()
        assert settings["agent"] == SAVED_UUID
        prompt = update["prompt"]
        assert "## This Call" in prompt
        for text in (
            "your trial ends soon",
            "book a renewal call",
            "customer since 2024",
            '"Hi, this is Alex from TechFlow."',
            "+15551234567",
        ):
            assert text in prompt, text
        assert greeting == {
            "type": "InjectAgentMessage",
            "message": "Hi, this is Alex from TechFlow.",
        }

    def test_outbound_ws_passes_campaign_to_agent(self, monkeypatch):
        from fastapi.testclient import TestClient

        from outbound import server

        seen: dict[str, Any] = {}

        async def fake_run_agent(**kwargs):
            seen.update(kwargs)

        monkeypatch.setattr(server, "run_agent", fake_run_agent)
        record = server.call_manager.create_call(
            phone_number="+1555", opening_reason="a demo", objective="book", context="ctx"
        )
        body = base64.b64encode(
            json.dumps({"call_uuid": "u", "is_outbound": True, "call_id": record.call_id}).encode()
        ).decode()
        with TestClient(server.app).websocket_connect(f"/ws?body={body}") as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"callId": "u", "streamId": "s"}}))
        assert (seen["opening_reason"], seen["objective"], seen["context"]) == (
            "a demo",
            "book",
            "ctx",
        )
        assert seen["initial_message"] == record.initial_message


class _FakeHTTPResponse:
    def __init__(self, payload: Any) -> None:
        self._raw = json.dumps(payload).encode() if payload is not None else b""

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def fake_deepgram_rest(monkeypatch):
    """Replace urllib.request.urlopen: route -> JSON payload (or an exception to raise)."""
    import urllib.request

    requests: list[Any] = []
    routes: dict[tuple[str, str], Any] = {}

    def urlopen(request, timeout=None):
        requests.append(request)
        path = request.full_url.split("/v1", 1)[1]
        result = routes[(request.get_method(), path)]
        if isinstance(result, BaseException):
            raise result
        return _FakeHTTPResponse(result)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return routes, requests


def _http_error(code: int, payload: dict[str, Any]):
    import io
    import urllib.error

    return urllib.error.HTTPError(
        "https://api.deepgram.com/v1/x", code, "err", {}, io.BytesIO(json.dumps(payload).encode())
    )


class TestUnitSavedConfigPublish:
    """publish/list/delete + project-id resolution against a mocked REST API."""

    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        from inbound import agent as inbound_mod
        from outbound import agent as outbound_mod

        for mod in (inbound_mod, outbound_mod):
            monkeypatch.setattr(mod, "DEEPGRAM_API_KEY", "test-key")
            monkeypatch.setattr(mod, "DEEPGRAM_PROJECT_ID", "proj-1")

    @pytest.mark.parametrize("id_field", ["agent_uuid", "agent_id"])
    def test_publish_payload(self, fake_deepgram_rest, id_field):
        from inbound.agent import EXAMPLE_NAME, build_agent_config, publish_agent_config

        routes, requests = fake_deepgram_rest
        routes[("POST", "/projects/proj-1/agents")] = {id_field: "new-uuid"}
        assert publish_agent_config() == "new-uuid"

        (request,) = requests
        assert request.get_header("Authorization") == "Token test-key"
        body = json.loads(request.data)
        assert isinstance(body["config"], str), "config must be a JSON string"
        assert json.loads(body["config"]) == build_agent_config()
        assert body["metadata"] == {"example": EXAMPLE_NAME, "direction": "inbound"}

    def test_outbound_publish_payload(self, fake_deepgram_rest):
        from outbound.agent import build_agent_config, publish_agent_config

        routes, requests = fake_deepgram_rest
        routes[("POST", "/projects/proj-1/agents")] = {"agent_uuid": "out-uuid"}
        assert publish_agent_config() == "out-uuid"
        body = json.loads(requests[0].data)
        config = json.loads(body["config"])
        assert config == build_agent_config()
        assert "greeting" not in config
        assert "{{" not in config["think"]["prompt"]
        assert body["metadata"]["direction"] == "outbound"

    def test_publish_without_uuid_in_response_fails(self, fake_deepgram_rest):
        from inbound.agent import DeepgramAPIError, publish_agent_config

        routes, _ = fake_deepgram_rest
        routes[("POST", "/projects/proj-1/agents")] = {"something": "else"}
        with pytest.raises(DeepgramAPIError, match="agent_uuid"):
            publish_agent_config()

    def test_403_names_the_missing_scope(self, fake_deepgram_rest):
        from inbound.agent import DeepgramAPIError, publish_agent_config

        routes, _ = fake_deepgram_rest
        routes[("POST", "/projects/proj-1/agents")] = _http_error(
            403, {"err_code": "INSUFFICIENT_PERMISSIONS", "err_msg": "Missing scope"}
        )
        with pytest.raises(DeepgramAPIError) as excinfo:
            publish_agent_config()
        message = str(excinfo.value)
        assert "HTTP 403" in message
        assert "INSUFFICIENT_PERMISSIONS" in message
        assert "agent:write" in message

    def test_project_id_from_env_needs_no_lookup(self, fake_deepgram_rest):
        from inbound.agent import resolve_project_id

        assert resolve_project_id() == "proj-1"
        assert fake_deepgram_rest[1] == []

    def test_single_project_is_used(self, fake_deepgram_rest, monkeypatch):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_PROJECT_ID", "")
        routes, _ = fake_deepgram_rest
        routes[("GET", "/projects")] = {"projects": [{"project_id": "only", "name": "x"}]}
        assert agent_mod.resolve_project_id() == "only"

    @pytest.mark.parametrize(
        ("projects", "match"),
        [
            ([{"project_id": "a"}, {"project_id": "b"}], "set DEEPGRAM_PROJECT_ID"),
            ([], "no Deepgram projects"),
        ],
    )
    def test_ambiguous_or_missing_project_errors(
        self, fake_deepgram_rest, monkeypatch, projects, match
    ):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_PROJECT_ID", "")
        routes, _ = fake_deepgram_rest
        routes[("GET", "/projects")] = {"projects": projects}
        with pytest.raises(agent_mod.DeepgramAPIError, match=match):
            agent_mod.resolve_project_id()

    def test_list_and_delete(self, fake_deepgram_rest):
        from inbound.agent import delete_agent_config, list_agent_configs

        routes, requests = fake_deepgram_rest
        routes[("GET", "/projects/proj-1/agents")] = [{"agent_uuid": "a", "metadata": {}}]
        routes[("DELETE", "/projects/proj-1/agents/a")] = {}
        assert list_agent_configs() == [{"agent_uuid": "a", "metadata": {}}]
        delete_agent_config("a")
        assert requests[-1].get_method() == "DELETE"

    def test_cli_publish_prints_env_line(self, fake_deepgram_rest, capsys):
        from outbound.agent import main

        routes, _ = fake_deepgram_rest
        routes[("POST", "/projects/proj-1/agents")] = {"agent_uuid": "cli-uuid"}
        assert main(["--publish"]) == 0
        assert "DEEPGRAM_OUTBOUND_AGENT_ID=cli-uuid" in capsys.readouterr().out

    def test_cli_reports_api_errors(self, fake_deepgram_rest, capsys):
        from inbound.agent import main

        routes, _ = fake_deepgram_rest
        routes[("DELETE", "/projects/proj-1/agents/x")] = _http_error(404, {"err_msg": "nope"})
        assert main(["--delete", "x"]) == 1
        assert "HTTP 404" in capsys.readouterr().err

    def test_cli_requires_api_key(self, monkeypatch, capsys):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod, "DEEPGRAM_API_KEY", "")
        assert agent_mod.main(["--list"]) == 2
        assert "DEEPGRAM_API_KEY" in capsys.readouterr().err


# =============================================================================
# LOCAL INTEGRATION TESTS (real server subprocess + real Deepgram)
# =============================================================================


@pytest.mark.skipif(not DEEPGRAM_API_KEY, reason="DEEPGRAM_API_KEY not configured")
class TestLocalIntegration:
    """Integration tests using a local WebSocket connection to the inbound server."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the inbound server as a subprocess (SIGTERM -> wait(5) -> SIGKILL)."""
        log_path = server_log_path("integration_local_server")
        # No Plivo credentials: the local server must never touch the Plivo REST API
        proc = start_server(
            "inbound.server",
            TEST_PORT,
            log_path,
            {"PLIVO_AUTH_ID": "", "PLIVO_AUTH_TOKEN": "", "PUBLIC_URL": ""},
        )
        yield proc
        stop_server(proc)

    async def test_local_health_check(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    async def test_local_answer_webhook(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/answer",
                params={"CallUUID": "test123", "From": "+15551234567", "To": "+16572338892"},
            )
        assert response.status_code == 200
        assert "application/xml" in response.headers["content-type"]
        assert "<Stream" in response.text
        assert "bidirectional" in response.text
        assert "audio/x-mulaw" in response.text

    async def test_local_websocket_connection(self, server_process):
        """A Plivo start event produces playAudio (the greeting) within 15s."""
        body = base64.b64encode(json.dumps({"call_uuid": "test123"}).encode()).decode()
        async with websockets.connect(f"{LOCAL_WS_URL}?body={body}", close_timeout=2) as ws:
            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
                    }
                )
            )
            audio_received = False
            try:
                async with asyncio.timeout(15):
                    while True:
                        data = json.loads(await ws.recv())
                        if data.get("event") == "playAudio":
                            audio_received = True
                            break
            except (TimeoutError, websockets.exceptions.ConnectionClosed):
                pass
        assert audio_received, "No audio received from server"

    async def test_local_audio_quality(self, server_process):
        """Greeting audio has speech energy (RMS > 500) while we stream silence."""
        body = base64.b64encode(json.dumps({"call_uuid": "test456"}).encode()).decode()
        audio_chunks: list[bytes] = []
        silence = base64.b64encode(b"\xff" * 160).decode()

        async with websockets.connect(f"{LOCAL_WS_URL}?body={body}", close_timeout=2) as ws:
            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
                    }
                )
            )
            start_time = time.time()
            while time.time() - start_time < 15 and len(audio_chunks) < 100:
                try:
                    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.02))
                    if data.get("event") == "playAudio":
                        audio_chunks.append(base64.b64decode(data["media"]["payload"]))
                except (TimeoutError, asyncio.TimeoutError):
                    await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))
                except websockets.exceptions.ConnectionClosed:
                    break

        assert audio_chunks, "No audio chunks received"
        assert all(len(c) == 160 for c in audio_chunks)
        rms = rms_of_ulaw(b"".join(audio_chunks))
        assert rms > 500, f"Audio RMS {rms} too low - may be silence"


# =============================================================================
# DEEPGRAM VOICE AGENT API INTEGRATION (direct connection)
# =============================================================================


@pytest.mark.skipif(not DEEPGRAM_API_KEY, reason="DEEPGRAM_API_KEY not configured")
class TestDeepgramAgentIntegration:
    """Talk to the Deepgram Voice Agent API directly with this example's Settings."""

    async def _connect(self):
        from inbound.agent import DEEPGRAM_AGENT_URL

        return websockets.connect(
            DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            max_size=None,
        )

    async def _handshake(self, dg) -> None:
        agent, _, _ = make_agent()
        welcome = json.loads(await asyncio.wait_for(dg.recv(), timeout=10))
        assert welcome["type"] == "Welcome"
        assert welcome.get("request_id")
        await dg.send(json.dumps(agent._build_settings()))
        async with asyncio.timeout(10):
            while True:
                msg = await dg.recv()
                if isinstance(msg, bytes):
                    continue
                evt = json.loads(msg)
                assert evt["type"] != "Error", f"Deepgram rejected Settings: {evt}"
                if evt["type"] == "SettingsApplied":
                    return

    async def test_settings_applied(self):
        async with await self._connect() as dg:
            await self._handshake(dg)

    async def test_greeting_audio(self):
        audio = bytearray()
        events: list[str] = []
        async with await self._connect() as dg:
            await self._handshake(dg)
            async with asyncio.timeout(20):
                while True:
                    msg = await dg.recv()
                    if isinstance(msg, bytes):
                        audio.extend(msg)
                        continue
                    evt = json.loads(msg)
                    events.append(evt["type"])
                    assert evt["type"] != "Error", evt
                    if evt["type"] == "AgentAudioDone":
                        break
        assert len(audio) >= 8000, f"Greeting too short: {len(audio)} bytes ({events})"
        rms = rms_of_ulaw(bytes(audio))
        assert rms > 500, f"Greeting RMS {rms:.0f} too low"
        assert "ConversationText" in events


# =============================================================================
# DEEPGRAM SAVED AGENT CONFIGURATION INTEGRATION (publish -> use -> delete)
# =============================================================================

_SPOKEN_DIGITS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}


def spoken_digits(text: str) -> str:
    """Digits in ``text``, counting spelled-out digits ("five five five" -> "555")."""
    out = []
    for token in "".join(c if c.isalnum() else " " for c in text.lower()).split():
        if token.isdigit():
            out.append(token)
        elif token in _SPOKEN_DIGITS:
            out.append(_SPOKEN_DIGITS[token])
    return "".join(out)


@pytest.fixture(scope="module")
def saved_config_id():
    """Publish a real inbound saved agent config for this module; ALWAYS delete it."""
    from inbound.agent import DeepgramAPIError, delete_agent_config, publish_agent_config

    try:
        config_id = publish_agent_config()
    except DeepgramAPIError as e:
        pytest.skip(f"cannot publish a saved agent config: {e}")
    print(f"\n[saved config] published {config_id}")
    try:
        yield config_id
    finally:
        delete_agent_config(config_id)
        print(f"\n[saved config] deleted {config_id}")


class TestUnitSavedConfigStartupCheck:
    """check_saved_agent_config(): server startup verification of the agent settings mode."""

    AGENT_ID = "11111111-2222-3333-4444-555555555555"

    @pytest.fixture(autouse=True)
    def _key(self, monkeypatch):
        from inbound import agent as inbound_mod
        from outbound import agent as outbound_mod

        for mod in (inbound_mod, outbound_mod):
            monkeypatch.setattr(mod, "DEEPGRAM_API_KEY", "test-key")
            monkeypatch.setattr(mod, "DEEPGRAM_PROJECT_ID", "proj-1")
        for name in inbound_mod._PUBLISHED_ENV_FIELDS:
            monkeypatch.delenv(name, raising=False)

    def _route(self, routes, config):
        routes[("GET", f"/projects/proj-1/agents/{self.AGENT_ID}")] = {
            "agent_uuid": self.AGENT_ID,
            "config": json.dumps(config),
            "metadata": {},
        }

    def test_inline_mode_makes_no_api_call(self, fake_deepgram_rest, captured_messages):
        from inbound.agent import check_saved_agent_config

        _routes, requests = fake_deepgram_rest
        assert check_saved_agent_config(agent_id="") is True
        assert requests == []
        assert any("Deepgram agent settings: inline" in m for m in captured_messages)

    def test_saved_config_matching_code_passes_without_warnings(
        self, fake_deepgram_rest, captured_messages
    ):
        from inbound.agent import build_agent_config, check_saved_agent_config

        routes, _ = fake_deepgram_rest
        self._route(routes, build_agent_config())
        assert check_saved_agent_config(agent_id=self.AGENT_ID) is True
        assert any(f"saved config {self.AGENT_ID}" in m for m in captured_messages)
        assert not any("Re-publish" in m for m in captured_messages)

    def test_explicit_model_env_var_that_differs_is_reported(self, monkeypatch):
        from inbound import agent as agent_mod

        published = agent_mod.build_agent_config()
        monkeypatch.setenv("DEEPGRAM_THINK_MODEL", "gpt-5.4-mini")
        monkeypatch.setattr(agent_mod, "DEEPGRAM_THINK_MODEL", "gpt-5.4-mini")
        (warning,) = agent_mod.saved_config_drift(published)
        assert "DEEPGRAM_THINK_MODEL=gpt-5.4-mini is ignored" in warning
        assert "'gpt-4.1-mini'" in warning

    def test_unset_model_env_vars_are_not_compared(self, monkeypatch):
        """A deployment that sets only the agent id must not get model warnings."""
        from inbound import agent as agent_mod

        published = agent_mod.build_agent_config()
        published["think"]["provider"]["model"] = "claude-haiku-4-5"
        assert agent_mod.saved_config_drift(published) == []

    def test_prompt_and_function_drift_reported(self):
        from inbound import agent as agent_mod

        published = agent_mod.build_agent_config()
        published["think"]["prompt"] = "an older prompt"
        published["think"]["functions"] = published["think"]["functions"][:-1]
        warnings = agent_mod.saved_config_drift(published)
        assert any("prompt differs" in w for w in warnings)
        assert any("only in code: ['end_call']" in w for w in warnings)

    @pytest.mark.parametrize("status", [400, 404])
    def test_unknown_agent_id_fails_startup(self, fake_deepgram_rest, captured_messages, status):
        from inbound.agent import check_saved_agent_config

        routes, _ = fake_deepgram_rest
        routes[("GET", f"/projects/proj-1/agents/{self.AGENT_ID}")] = _http_error(
            status, {"category": "NOT_FOUND", "message": "agent not found"}
        )
        assert check_saved_agent_config(agent_id=self.AGENT_ID) is False
        assert any("DEEPGRAM_INBOUND_AGENT_ID" in m and "--publish" in m for m in captured_messages)

    def test_unverifiable_agent_id_warns_but_starts(self, fake_deepgram_rest, captured_messages):
        from inbound.agent import check_saved_agent_config

        routes, _ = fake_deepgram_rest
        routes[("GET", f"/projects/proj-1/agents/{self.AGENT_ID}")] = _http_error(
            403, {"category": "INSUFFICIENT_PERMISSIONS", "message": "no agent:read"}
        )
        assert check_saved_agent_config(agent_id=self.AGENT_ID) is True
        assert any("Using it unverified" in m for m in captured_messages)

    def test_outbound_check_uses_outbound_env_var(self, fake_deepgram_rest, captured_messages):
        from outbound.agent import check_saved_agent_config

        routes, _ = fake_deepgram_rest
        routes[("GET", f"/projects/proj-1/agents/{self.AGENT_ID}")] = _http_error(404, {})
        assert check_saved_agent_config(agent_id=self.AGENT_ID) is False
        assert any("DEEPGRAM_OUTBOUND_AGENT_ID" in m for m in captured_messages)

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    def test_server_refuses_to_start_when_check_fails(self, monkeypatch, module):
        import importlib

        server = importlib.import_module(module)
        started = []
        monkeypatch.setattr(server, "check_saved_agent_config", lambda: False)
        monkeypatch.setattr(server, "PUBLIC_URL", "", raising=False)
        monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: started.append(True))
        with pytest.raises(SystemExit):
            server.main()
        assert started == []


@pytest.mark.skipif(not DEEPGRAM_API_KEY, reason="DEEPGRAM_API_KEY not configured")
class TestDeepgramSavedConfigIntegration:
    """Publish a real saved agent config, connect with ``agent: <uuid>``, always delete it."""

    CALLER = "+14155550123"

    @contextlib.asynccontextmanager
    async def _session(self, config_id: str):
        """Connect in saved mode and run the agent's own handshake (UpdatePrompt + greeting)."""
        from inbound.agent import DEEPGRAM_AGENT_URL, DeepgramVoiceAgent

        agent = DeepgramVoiceAgent(
            websocket=FakePlivoWS(),
            call_id=CALL_ID,
            from_number=self.CALLER,
            agent_config_id=config_id,
        )
        async with websockets.connect(
            DEEPGRAM_AGENT_URL,
            additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
            max_size=None,
        ) as dg:
            agent._dg_ws = dg
            await agent._handshake(dg)
            yield agent, dg

    async def _until_audio_done(self, agent, dg, timeout: float = 30) -> dict[str, Any]:
        """Collect one agent response; answer FunctionCallRequests via the agent's handler."""
        result: dict[str, Any] = {"audio": bytearray(), "types": [], "text": [], "functions": []}
        async with asyncio.timeout(timeout):
            while True:
                msg = await dg.recv()
                if isinstance(msg, bytes):
                    result["audio"].extend(msg)
                    continue
                evt = json.loads(msg)
                result["types"].append(evt["type"])
                assert evt["type"] != "Error", evt
                if evt["type"] == "ConversationText" and evt.get("role") == "assistant":
                    result["text"].append(evt["content"])
                elif evt["type"] == "FunctionCallRequest":
                    result["functions"] += [f["name"] for f in evt["functions"]]
                    await agent._on_function_call_request(evt)
                elif evt["type"] == "AgentAudioDone":
                    return result

    async def test_saved_config_is_listed(self, saved_config_id):
        from inbound.agent import EXAMPLE_NAME, list_agent_configs

        configs = {c.get("agent_uuid"): c for c in list_agent_configs()}
        assert saved_config_id in configs
        assert configs[saved_config_id]["metadata"] == {
            "example": EXAMPLE_NAME,
            "direction": "inbound",
        }

    async def test_greeting_and_injected_caller_context(self, saved_config_id):
        from inbound.agent import DEFAULT_GREETING

        async with self._session(saved_config_id) as (agent, dg):
            greeting = await self._until_audio_done(agent, dg)
            assert "PromptUpdated" in greeting["types"]
            assert " ".join(greeting["text"]) == DEFAULT_GREETING
            assert len(greeting["audio"]) >= 8000, f"greeting too short: {greeting['types']}"
            assert rms_of_ulaw(bytes(greeting["audio"])) > 500

            await agent._inject_user_text("What phone number am I calling from?")
            answer = await self._until_audio_done(agent, dg)
        spoken = " ".join(answer["text"])
        print(f"\n[saved config] caller-number answer: {spoken}")
        assert "4155550123" in spoken_digits(spoken), spoken

    async def test_client_side_function_call(self, saved_config_id):
        async with self._session(saved_config_id) as (agent, dg):
            await self._until_audio_done(agent, dg)  # greeting
            await agent._inject_user_text("Please check the status of my order TF-123456.")
            response = await self._until_audio_done(agent, dg)
            if not response["functions"]:  # the function result may arrive after a filler
                response = await self._until_audio_done(agent, dg)
        spoken = " ".join(response["text"])
        print(f"\n[saved config] functions={response['functions']} response: {spoken}")
        assert "check_order_status" in response["functions"]
        assert spoken


# =============================================================================
# PLIVO INTEGRATION TESTS
# =============================================================================


class TestPlivoIntegration:
    """Integration tests for the Plivo API."""

    @pytest.fixture
    def plivo_configured(self):
        if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
            pytest.skip("Plivo credentials not configured")

    def test_plivo_credentials_valid(self, plivo_configured):
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        assert client.account.get() is not None

    def test_plivo_phone_number_exists(self, plivo_configured):
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        try:
            number = client.numbers.get(number=normalize_phone_number(PLIVO_PHONE_NUMBER))
            assert number is not None
        except plivo.exceptions.ResourceNotFoundError:
            pytest.fail(f"Phone number {PLIVO_PHONE_NUMBER} not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
