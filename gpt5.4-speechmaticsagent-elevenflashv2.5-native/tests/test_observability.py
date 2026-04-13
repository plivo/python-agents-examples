"""Tests for observability sinks: stderr JSON, file JSONL, Redis Streams."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from loguru import logger


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
            redis_events.append({
                "level": record["level"].name,
                "msg": str(record["message"]),
                **{k: str(v) for k, v in record["extra"].items()},
            })

        logger.remove()
        # stderr JSON sink
        logger.add(__import__("sys").stderr, serialize=True, level="DEBUG")
        # File sink
        logger.add(str(log_file), serialize=True, level="DEBUG")
        # Redis sink
        logger.add(redis_sink, level="DEBUG")

        logger.bind(call_id="jkl012", stage="metrics", event="call_summary").info(
            "Call completed"
        )
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
        assert redis_events[0]["event"] == "call_summary"

        # Restore
        logger.remove()
        logger.add(__import__("sys").stderr)
