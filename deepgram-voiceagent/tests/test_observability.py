"""Tests for observability sinks (stderr JSON, file JSONL, Redis Streams) and agent events."""

from __future__ import annotations

import asyncio
import io
import json
from unittest.mock import MagicMock

import pytest
from loguru import logger

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
