"""Tests for observability sinks (stderr JSON, file JSONL, Redis Streams) and agent events."""

from __future__ import annotations

import asyncio
import io
import json
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from loguru import logger
from opentelemetry.sdk.trace import SpanProcessor  # dev dependency

from tests.test_integration import CALL_ID, STREAM_ID, FakeDeepgramWS, FakePlivoWS


class TestUnitLogFormatJson:
    """Verify LOG_FORMAT=json produces serialized JSON on stderr."""

    def test_json_format_produces_valid_json(self, capsys):
        # Remove all sinks, add a JSON stderr sink like server.py does
        logger.remove()
        logger.add(
            __import__("sys").stderr,
            serialize=True,
            level="DEBUG",
        )

        logger.bind(call_id="abc123", stage="stt", elapsed_s=1.5).info("STT result")

        captured = capsys.readouterr()
        record = json.loads(captured.err.strip())

        assert "STT result" in record["text"]
        assert record["record"]["extra"]["call_id"] == "abc123"
        assert record["record"]["extra"]["stage"] == "stt"
        assert record["record"]["extra"]["elapsed_s"] == 1.5
        assert record["record"]["level"]["name"] == "INFO"

        # Restore default sink
        logger.remove()
        logger.add(__import__("sys").stderr)


class TestUnitLogFile:
    """Verify LOG_FILE writes structured JSONL."""

    def test_file_sink_writes_jsonl(self, tmp_path):
        log_file = tmp_path / "test.jsonl"

        logger.remove()
        logger.add(
            str(log_file),
            serialize=True,
            rotation="100 MB",
            retention="7 days",
            level="DEBUG",
        )

        logger.bind(call_id="def456", stage="llm").info("LLM response")
        logger.bind(call_id="def456", stage="tts").info("TTS started")
        logger.complete()

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["record"]["extra"]["call_id"] == "def456"
        assert first["record"]["extra"]["stage"] == "llm"

        second = json.loads(lines[1])
        assert second["record"]["extra"]["stage"] == "tts"

        # Restore default sink
        logger.remove()
        logger.add(__import__("sys").stderr)


class TestUnitRedisSink:
    """Verify Redis Streams sink calls XADD with correct fields."""

    def test_redis_sink_publishes_all_events(self):
        mock_redis = MagicMock()
        stream_key = "voice-agent:events"
        captured_calls = []

        def fake_xadd(key, fields, maxlen=None, approximate=False):
            captured_calls.append({"key": key, "fields": dict(fields)})

        mock_redis.xadd = fake_xadd

        # Build the sink exactly as server.py does
        import contextlib

        def redis_sink(message):
            record = message.record
            fields = {
                "ts": record["time"].isoformat(),
                "level": record["level"].name,
                "msg": str(record["message"]),
            }
            for k, v in record["extra"].items():
                fields[k] = str(v)
            with contextlib.suppress(Exception):
                mock_redis.xadd(stream_key, fields, maxlen=10000, approximate=True)

        logger.remove()
        sink_id = logger.add(redis_sink, level="DEBUG")

        # Emit structured event
        logger.bind(call_id="ghi789", stage="stt", elapsed_s=2.1).info("Transcription done")
        # Emit plain event (no extras)
        logger.debug("Generic debug message")

        logger.remove(sink_id)

        # Both events should reach Redis
        assert len(captured_calls) == 2

        # Check structured event fields
        first = captured_calls[0]
        assert first["key"] == stream_key
        assert first["fields"]["call_id"] == "ghi789"
        assert first["fields"]["stage"] == "stt"
        assert first["fields"]["elapsed_s"] == "2.1"
        assert first["fields"]["level"] == "INFO"
        assert "Transcription done" in first["fields"]["msg"]

        # Check plain event
        second = captured_calls[1]
        assert second["fields"]["level"] == "DEBUG"
        assert "Generic debug" in second["fields"]["msg"]

        # Restore default sink
        logger.add(__import__("sys").stderr)

    def test_redis_sink_survives_xadd_failure(self):
        """Sink should not raise even if XADD throws."""
        mock_redis = MagicMock()
        mock_redis.xadd.side_effect = ConnectionError("Redis down")

        import contextlib

        def redis_sink(message):
            record = message.record
            fields = {"ts": record["time"].isoformat(), "msg": str(record["message"])}
            with contextlib.suppress(Exception):
                mock_redis.xadd("test", fields)

        logger.remove()
        sink_id = logger.add(redis_sink, level="DEBUG")

        # Should not raise
        logger.info("This should not crash")

        logger.remove(sink_id)
        logger.add(__import__("sys").stderr)

        # Verify XADD was attempted
        mock_redis.xadd.assert_called_once()


class TestUnitAllSinksTogether:
    """Verify all three sinks can run simultaneously."""

    def test_all_sinks_receive_same_event(self, tmp_path, capsys):
        log_file = tmp_path / "all_sinks.jsonl"
        redis_events = []

        def redis_sink(message):
            record = message.record
            redis_events.append(
                {
                    "level": record["level"].name,
                    "msg": str(record["message"]),
                    **{k: str(v) for k, v in record["extra"].items()},
                }
            )

        logger.remove()
        # stderr JSON sink
        logger.add(__import__("sys").stderr, serialize=True, level="DEBUG")
        # File sink
        logger.add(str(log_file), serialize=True, level="DEBUG")
        # Redis sink
        logger.add(redis_sink, level="DEBUG")

        logger.bind(call_id="jkl012", stage="metrics", event="session_end").info("Call completed")
        logger.complete()

        # Verify stderr
        captured = capsys.readouterr()
        stderr_record = json.loads(captured.err.strip())
        assert stderr_record["record"]["extra"]["call_id"] == "jkl012"

        # Verify file
        file_record = json.loads(log_file.read_text().strip())
        assert file_record["record"]["extra"]["call_id"] == "jkl012"

        # Verify Redis
        assert len(redis_events) == 1
        assert redis_events[0]["call_id"] == "jkl012"
        assert redis_events[0]["event"] == "session_end"

        # Restore
        logger.remove()
        logger.add(__import__("sys").stderr)


# =============================================================================
# Agent structured events (full session with fake Plivo + Deepgram sockets)
# =============================================================================

GREETING = "Hi, this is Alex from TechFlow."


def _greeting_script(extra: list | None = None) -> list:
    return [
        {"type": "Welcome", "request_id": "req-obs-1"},
        {"type": "SettingsApplied"},
        {"type": "ConversationText", "role": "assistant", "content": GREETING},
        b"\x10" * 800,
        {"type": "AgentAudioDone"},
        *(extra or []),
    ]


class TestUnitAgentEvents:
    """Drive DeepgramVoiceAgent.run() end-to-end and check the structured event stream."""

    @pytest.fixture
    def events(self):
        captured: list[dict] = []

        def sink(message):
            extra = message.record["extra"]
            if "event" in extra:
                captured.append(dict(extra))

        sink_id = logger.add(sink, level="DEBUG")
        yield captured
        logger.remove(sink_id)

    @staticmethod
    def _patch_connect(monkeypatch, dg):
        from inbound import agent as agent_mod

        monkeypatch.setattr(agent_mod.websockets, "connect", lambda *_a, **_k: dg)

    @staticmethod
    def _agent(plivo_ws, **kwargs):
        from inbound.agent import DeepgramVoiceAgent

        return DeepgramVoiceAgent(
            websocket=plivo_ws,
            call_id=CALL_ID,
            from_number="+15551234567",
            to_number="+16572338892",
            stream_id=STREAM_ID,
            sip_headers={"SIP-X-Test": "1"},
            **kwargs,
        )

    async def test_greeting_session_emits_all_events(self, monkeypatch, events):
        dg = FakeDeepgramWS(_greeting_script())
        self._patch_connect(monkeypatch, dg)
        plivo_ws = FakePlivoWS(auto_played_stream=True)
        agent = self._agent(plivo_ws)

        await asyncio.wait_for(agent.run(), timeout=5)

        names = [e["event"] for e in events]
        for required in ("call_answered", "agent_text", "turn_complete", "session_end"):
            assert required in names, f"missing {required}: {names}"
        assert names[0] == "call_answered"
        assert names[-1] == "session_end"
        for event in events:
            assert event["call_id"] == CALL_ID, event

        answered = next(e for e in events if e["event"] == "call_answered")
        assert answered["stream_id"] == STREAM_ID
        assert answered["sip_headers"] == {"SIP-X-Test": "1"}
        agent_text = next(e for e in events if e["event"] == "agent_text")
        assert agent_text == {**agent_text, "turn": 1, "text": GREETING}
        turn_complete = next(e for e in events if e["event"] == "turn_complete")
        assert turn_complete["barge_in"] is False
        assert turn_complete["agent_text"] == GREETING
        assert turn_complete["plivo_tx_chunks"] == 5
        session_end = events[-1]
        assert session_end["turns"] == 1
        assert session_end["barge_ins"] == 0
        assert session_end["errors"] == 0
        assert session_end["tx_chunks"] == 5
        assert session_end["deepgram_request_id"] == "req-obs-1"

        assert json.loads(dg.sent[0])["type"] == "Settings"
        assert [m["event"] for m in plivo_ws.sent] == ["playAudio"] * 5 + ["checkpoint"]

    async def test_end_call_session_hangs_up_via_callback(self, monkeypatch, events):
        hangups: list[str] = []

        async def hangup_callback() -> None:
            hangups.append("hangup")

        end_call = {
            "type": "FunctionCallRequest",
            "functions": [{"id": "f1", "name": "end_call", "arguments": "{}", "client_side": True}],
        }
        dg = FakeDeepgramWS(
            [
                {"type": "Welcome", "request_id": "r"},
                {"type": "SettingsApplied"},
                end_call,
                {"type": "ConversationText", "role": "assistant", "content": "Goodbye!"},
                b"\x10" * 320,
                {"type": "AgentAudioDone"},
            ]
        )
        self._patch_connect(monkeypatch, dg)
        plivo_ws = FakePlivoWS()
        plivo_ws.auto_played_stream = False

        async def ack_checkpoints():
            while not plivo_ws.events("checkpoint"):
                await asyncio.sleep(0.01)
            plivo_ws.push(
                {"event": "playedStream", "name": plivo_ws.events("checkpoint")[0]["name"]}
            )

        agent = self._agent(plivo_ws, hangup_callback=hangup_callback)
        acker = asyncio.create_task(ack_checkpoints())
        # No Plivo "stop" is ever sent: the session must end on its own after the hangup
        await asyncio.wait_for(agent.run(), timeout=5)
        await acker

        assert hangups == ["hangup"]
        responses = [json.loads(m) for m in dg.sent if isinstance(m, str)]
        assert any(
            r["type"] == "FunctionCallResponse" and r["name"] == "end_call" for r in responses
        )
        assert events[-1]["event"] == "session_end"

    async def test_connect_failure_still_emits_session_end(self, monkeypatch, events):
        from inbound import agent as agent_mod

        def boom(*_args, **_kwargs):
            raise OSError("connection refused")

        monkeypatch.setattr(agent_mod.websockets, "connect", boom)
        await asyncio.wait_for(self._agent(FakePlivoWS()).run(), timeout=5)
        names = [e["event"] for e in events]
        assert names == ["call_answered", "session_end"]
        assert events[-1]["errors"] == 1

    async def test_events_serialize_to_json(self, monkeypatch):
        """With LOG_FORMAT=json every structured event is a parseable JSON line."""
        buffer = io.StringIO()
        sink_id = logger.add(buffer, serialize=True, level="DEBUG")
        try:
            dg = FakeDeepgramWS(_greeting_script())
            self._patch_connect(monkeypatch, dg)
            await asyncio.wait_for(self._agent(FakePlivoWS(auto_played_stream=True)).run(), 5)
        finally:
            logger.remove(sink_id)
        records = [json.loads(line)["record"] for line in buffer.getvalue().splitlines()]
        events = {r["extra"]["event"] for r in records if "event" in r["extra"]}
        assert {"call_answered", "agent_text", "turn_complete", "session_end"} <= events
        stages = {r["extra"].get("stage") for r in records}
        assert {"session", "deepgram", "turn"} <= stages


# =============================================================================
# OpenTelemetry span tree (in-memory exporter, fake Plivo + Deepgram sockets)
# =============================================================================

AGENT_MODULES = ("inbound.agent", "outbound.agent")
ORDER_QUESTION = "Where is my order TF-123456?"
ORDER_ANSWER = "Your order has shipped."


class _TrackingProcessor(SpanProcessor):
    """SpanProcessor that records every started span, to prove none is left open."""

    def __init__(self) -> None:
        self.started: list = []

    def on_start(self, span, parent_context=None) -> None:
        self.started.append(span)


class _Spans:
    def __init__(self, exporter, tracking) -> None:
        self.exporter = exporter
        self.tracking = tracking

    @property
    def finished(self) -> list:
        return list(self.exporter.get_finished_spans())

    def named(self, name: str) -> list:
        return [s for s in self.finished if s.name == name]

    def one(self, name: str):
        spans = self.named(name)
        assert len(spans) == 1, f"{name}: {[s.name for s in self.finished]}"
        return spans[0]

    def turn(self, number: int):
        return next(s for s in self.named("turn") if s.attributes["turn"] == number)

    def children(self, parent) -> list:
        return [
            s
            for s in self.finished
            if s.parent is not None and s.parent.span_id == parent.context.span_id
        ]

    def assert_none_open(self) -> None:
        open_spans = [s.name for s in self.tracking.started if s.end_time is None]
        assert open_spans == [], f"spans never ended: {open_spans}"


async def _wait_for(predicate, timeout: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        assert asyncio.get_running_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


def _function_call(name: str, arguments: str) -> dict:
    return {
        "type": "FunctionCallRequest",
        "functions": [{"id": "fc-1", "name": name, "arguments": arguments, "client_side": True}],
    }


class TestUnitOtelSpans:
    """Span tree from Deepgram signals: session -> tasks, turn -> stt/llm/tts/tool/playback."""

    @pytest.fixture(params=AGENT_MODULES)
    def agent_mod(self, request):
        import importlib

        return importlib.import_module(request.param)

    @pytest.fixture
    def spans(self, monkeypatch, agent_mod):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        tracking = _TrackingProcessor()
        provider = TracerProvider()
        provider.add_span_processor(tracking)
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        monkeypatch.setattr(agent_mod, "_tracer", provider.get_tracer("voice-agent"))
        yield _Spans(exporter, tracking)
        provider.shutdown()

    @staticmethod
    def _session(monkeypatch, agent_mod, **kwargs):
        dg = FakeDeepgramWS()
        monkeypatch.setattr(agent_mod.websockets, "connect", lambda *_a, **_k: dg)
        plivo_ws = FakePlivoWS()
        agent = agent_mod.DeepgramVoiceAgent(
            websocket=plivo_ws,
            call_id=CALL_ID,
            from_number="+15551234567",
            stream_id=STREAM_ID,
            agent_config_id=kwargs.pop("agent_config_id", ""),
            **kwargs,
        )
        return agent, dg, plivo_ws

    @staticmethod
    async def _ack_checkpoint(plivo_ws, index: int) -> None:
        await _wait_for(lambda: len(plivo_ws.events("checkpoint")) > index)
        name = plivo_ws.events("checkpoint")[index]["name"]
        plivo_ws.push({"event": "playedStream", "name": name, "streamId": STREAM_ID})

    async def _greeting(self, dg, plivo_ws, spans) -> None:
        for item in _greeting_script():
            dg.feed(item)
        await self._ack_checkpoint(plivo_ws, 0)
        await _wait_for(lambda: spans.named("turn"))

    async def test_greeting_user_turn_tool_and_playback_tree(self, monkeypatch, agent_mod, spans):
        agent, dg, plivo_ws = self._session(monkeypatch, agent_mod)
        run = asyncio.create_task(agent.run())

        await self._greeting(dg, plivo_ws, spans)
        for item in (
            {"type": "LatencyReport", "stt_latency": 0.12},
            {"type": "EndOfTurn", "trigger": "eot_threshold"},
            {"type": "ConversationText", "role": "user", "content": ORDER_QUESTION},
            _function_call("check_order_status", '{"order_number": "TF-123456"}'),
            {"type": "LatencyReport", "ttt_tool_latency": 0.4},
            {"type": "Warning", "code": "SLOW_THINK_REQUEST", "description": "slow"},
            {"type": "LatencyReport", "ttt_token_latency": 0.3},
            {"type": "LatencyReport", "ttt_text_latency": 0.5},
            {"type": "LatencyReport", "tts_latency": 0.2},
            {"type": "LatencyReport", "total_latency": 1.1},
            {"type": "ConversationText", "role": "assistant", "content": ORDER_ANSWER},
            b"\x10" * 480,
            {"type": "AgentAudioDone"},
        ):
            dg.feed(item)
        await self._ack_checkpoint(plivo_ws, 1)
        await _wait_for(lambda: len(spans.named("turn")) == 2)
        plivo_ws.push({"event": "stop"})
        await asyncio.wait_for(run, timeout=5)

        spans.assert_none_open()
        session = spans.one("session")
        assert session.parent is None
        assert session.attributes["call_id"] == CALL_ID
        assert session.attributes["gen_ai.system"] == "deepgram"
        assert session.attributes["gen_ai.request.model"] == agent_mod.DEEPGRAM_THINK_MODEL
        assert session.attributes["deepgram.think.provider"] == agent_mod.DEEPGRAM_THINK_PROVIDER
        assert session.attributes["deepgram.listen.model"] == agent_mod.DEEPGRAM_LISTEN_MODEL
        assert session.attributes["deepgram.speak.model"] == agent_mod.DEEPGRAM_SPEAK_MODEL
        assert session.attributes["deepgram.agent_config"] == "inline"
        assert session.attributes["deepgram.request_id"] == "req-obs-1"
        assert session.status.is_ok

        top = sorted(s.name for s in spans.children(session))
        assert top == ["deepgram_rx", "plivo_rx", "plivo_tx", "turn", "turn"]

        greeting = spans.turn(1)
        assert greeting.attributes["turn.source"] == "greeting"
        assert greeting.attributes["agent_text"] == GREETING
        assert greeting.attributes["user_text"] == ""
        assert greeting.attributes["barge_in"] is False
        assert greeting.attributes["turn.completed"] is True
        assert greeting.attributes["call_id"] == CALL_ID
        assert [s.name for s in spans.children(greeting)] == ["playback"]

        turn = spans.turn(2)
        assert turn.attributes["user_text"] == ORDER_QUESTION
        assert turn.attributes["agent_text"] == ORDER_ANSWER
        assert turn.attributes["eot.trigger"] == "eot_threshold"
        assert turn.attributes["turn.source"] == "audio"
        assert turn.attributes["barge_in"] is False
        assert turn.attributes["deepgram.total_latency_ms"] == 1100
        assert [e.name for e in turn.events] == ["deepgram.warning"]
        assert turn.events[0].attributes["code"] == "SLOW_THINK_REQUEST"
        children = sorted(s.name for s in spans.children(turn))
        assert children == ["llm", "llm", "playback", "stt", "tool.check_order_status", "tts"]

        durations_ms = {
            (s.name, s.attributes["deepgram.metric"]): (s.end_time - s.start_time) / 1e6
            for s in spans.children(turn)
            if s.name in ("stt", "llm", "tts")
        }
        # ttt_token_latency is not used when ttt_text_latency was reported
        assert durations_ms == {
            ("stt", "stt_latency"): 120,
            ("llm", "ttt_tool_latency"): 400,
            ("llm", "ttt_text_latency"): 500,
            ("tts", "tts_latency"): 200,
        }
        for span in spans.named("llm"):
            assert span.attributes["deepgram.timing"] == "reported_duration"
            assert span.attributes["gen_ai.request.model"] == agent_mod.DEEPGRAM_THINK_MODEL

        tool = spans.one("tool.check_order_status")
        assert tool.attributes["tool.name"] == "check_order_status"
        assert "TF-123456" in tool.attributes["tool.arguments"]
        assert tool.attributes["tool.status"] in ("shipped", "processing", "delivered")
        assert "TF-123456" in tool.attributes["tool.result"]

        playback = next(s for s in spans.children(turn) if s.name == "playback")
        assert playback.attributes["interrupted"] is False
        assert playback.attributes["plivo.tx_chunks"] == 3
        assert "playback_ms" in playback.attributes
        for task in ("plivo_rx", "deepgram_rx", "plivo_tx"):
            assert spans.one(task).status.is_ok

    async def test_barge_in_event_and_interrupted_playback(self, monkeypatch, agent_mod, spans):
        agent, dg, plivo_ws = self._session(monkeypatch, agent_mod)
        run = asyncio.create_task(agent.run())
        for item in _greeting_script()[:4]:  # greeting audio, no AgentAudioDone yet
            dg.feed(item)
        await _wait_for(lambda: plivo_ws.events("playAudio"))
        dg.feed({"type": "UserStartedSpeaking"})
        await _wait_for(lambda: spans.named("turn"))
        plivo_ws.push({"event": "stop"})
        await asyncio.wait_for(run, timeout=5)

        spans.assert_none_open()
        turn = spans.one("turn")
        assert turn.attributes["barge_in"] is True
        assert [e.name for e in turn.events] == ["barge_in"]
        playback = spans.one("playback")
        assert playback.parent.span_id == turn.context.span_id
        assert playback.attributes["interrupted"] is True

    async def test_turn_ends_on_mid_turn_teardown(self, monkeypatch, agent_mod, spans):
        agent, dg, plivo_ws = self._session(monkeypatch, agent_mod)
        run = asyncio.create_task(agent.run())
        await self._greeting(dg, plivo_ws, spans)
        plivo_ws.push({"event": "text", "text": ORDER_QUESTION})
        await _wait_for(lambda: agent._turn_count == 2)
        dg.feed({"type": "EndOfTurn", "trigger": "manual"})
        await _wait_for(lambda: agent._trace.turn_awaiting_eot is False)
        plivo_ws.push({"event": "stop"})  # caller hangs up before any response
        await asyncio.wait_for(run, timeout=5)

        spans.assert_none_open()
        turn = spans.turn(2)
        assert turn.attributes["turn.completed"] is False
        assert turn.attributes["turn.ended_by"] == "session_end"
        assert turn.attributes["turn.source"] == "text"
        assert turn.attributes["eot.trigger"] == "manual"
        assert turn.attributes["user_text"] == ORDER_QUESTION
        assert turn.parent.span_id == spans.one("session").context.span_id

    async def test_deepgram_error_marks_session(self, monkeypatch, agent_mod, spans):
        agent, dg, plivo_ws = self._session(monkeypatch, agent_mod)
        run = asyncio.create_task(agent.run())
        await self._greeting(dg, plivo_ws, spans)
        dg.feed({"type": "Error", "code": "FAILED_TO_THINK", "description": "llm down"})
        await asyncio.wait_for(run, timeout=5)

        spans.assert_none_open()
        session = spans.one("session")
        assert not session.status.is_ok
        error = next(e for e in session.events if e.name == "deepgram.error")
        assert error.attributes["code"] == "FAILED_TO_THINK"

    async def test_reusable_config_model_labels(self, monkeypatch, agent_mod, spans):
        agent, *_ = self._session(monkeypatch, agent_mod, agent_config_id="uuid-1")
        session, llm = agent._trace_attributes()
        assert session["deepgram.agent_config"] == "uuid-1"
        assert session["gen_ai.request.model"] == agent_mod.SAVED_CONFIG_MODEL
        assert session["deepgram.listen.model"] == agent_mod.SAVED_CONFIG_MODEL

        models = {
            "listen_model": "nova-3",
            "think_provider": "anthropic",
            "think_model": "claude-haiku-4-5",
            "speak_model": "aura-2-thalia-en",
        }
        agent, *_ = self._session(
            monkeypatch, agent_mod, agent_config_id="uuid-1", saved_agent_models=models
        )
        session, llm = agent._trace_attributes()
        assert session["deepgram.listen.model"] == "nova-3"
        assert session["deepgram.speak.model"] == "aura-2-thalia-en"
        assert llm == {
            "gen_ai.system": "deepgram",
            "gen_ai.request.model": "claude-haiku-4-5",
            "deepgram.think.provider": "anthropic",
        }


class TestUnitOtelMissing:
    """Without opentelemetry installed the agent runs unchanged and creates no spans."""

    @pytest.mark.parametrize("direction", ["inbound", "outbound"])
    async def test_session_runs_without_opentelemetry(self, monkeypatch, direction):
        import importlib.util
        import sys
        from pathlib import Path

        monkeypatch.setitem(sys.modules, "opentelemetry", None)  # import -> ImportError
        path = Path(__file__).parent.parent / direction / "agent.py"
        spec = importlib.util.spec_from_file_location(f"{direction}_agent_no_otel", path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, mod)  # dataclasses look it up
        spec.loader.exec_module(mod)
        assert mod._tracer is None and mod._otel_trace is None

        events: list[str] = []
        sink_id = logger.add(
            lambda m: events.append(m.record["extra"].get("event", "")), level="DEBUG"
        )
        dg = FakeDeepgramWS(
            [
                *_greeting_script()[:4],
                {"type": "UserStartedSpeaking"},  # barge-in
                {"type": "EndOfTurn", "trigger": "eot_threshold"},
                {"type": "ConversationText", "role": "user", "content": ORDER_QUESTION},
                {"type": "LatencyReport", "stt_latency": 0.1},
                _function_call("check_order_status", '{"order_number": "TF-1"}'),
                {"type": "LatencyReport", "ttt_text_latency": 0.5},
                {"type": "Warning", "code": "SLOW_THINK_REQUEST", "description": "slow"},
                {"type": "ConversationText", "role": "assistant", "content": ORDER_ANSWER},
                b"\x10" * 320,
                {"type": "AgentAudioDone"},
            ]
        )
        monkeypatch.setattr(mod.websockets, "connect", lambda *_a, **_k: dg)
        plivo_ws = FakePlivoWS(auto_played_stream=True)
        agent = mod.DeepgramVoiceAgent(
            websocket=plivo_ws, call_id=CALL_ID, stream_id=STREAM_ID, agent_config_id=""
        )
        try:
            await asyncio.wait_for(agent.run(), timeout=5)
        finally:
            logger.remove(sink_id)

        assert agent._trace.enabled is False
        assert agent._trace.turn is None and agent._trace.session is None
        assert agent._error_count == 0
        assert events.count("turn_complete") == 2
        assert "session_end" in events
        assert json.loads(dg.sent[-1])["type"] == "FunctionCallResponse"


class TestUnitSavedConfigModels:
    """server.py resolves a reusable config's models from the startup lookup it already does."""

    BODY: ClassVar[dict] = {
        "agent_uuid": "u",
        "config": json.dumps(
            {
                "listen": {"provider": {"type": "deepgram", "model": "flux-general-en"}},
                "think": {"provider": {"type": "open_ai", "model": "gpt-4.1-mini"}},
                "speak": [{"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}}],
            }
        ),
    }

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    def test_verify_stores_models(self, monkeypatch, module):
        import importlib

        server = importlib.import_module(module)
        monkeypatch.setattr(server, "_saved_agent_models", {})
        monkeypatch.setattr(
            server,
            "_deepgram_get",
            lambda p: {"projects": [{"project_id": "p1"}]} if p == "/projects" else self.BODY,
        )
        assert server.verify_deepgram_agent_id("u") is True
        assert server._saved_agent_models == {
            "listen_model": "flux-general-en",
            "think_provider": "open_ai",
            "think_model": "gpt-4.1-mini",
            "speak_model": "aura-2-thalia-en",
        }

    @pytest.mark.parametrize("module", ["inbound.server", "outbound.server"])
    @pytest.mark.parametrize("body", [{}, {"config": "not json"}, {"config": "[]"}])
    def test_unparseable_config_gives_no_models(self, module, body):
        import importlib

        assert importlib.import_module(module)._saved_config_models(body) == {}
