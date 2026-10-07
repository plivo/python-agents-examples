"""
Integration tests for the GPT-4o + Modulate Velma-2 + Cartesia Sonic 3 Pipecat voice agent.

Test Levels:
1. Unit Tests (offline) - audio conversion, phone normalization, ModulateSTTService event
   handling, the Tavily search tool with a stubbed client, prompts and the outbound
   greeting, server routes and Plivo webhook authentication via FastAPI TestClient
2. Local Integration - start the inbound server, drive the Plivo WebSocket protocol
   (needs OPENAI_API_KEY, MODULATE_API_KEY and CARTESIA_API_KEY)
3. OpenAI Integration - test the OpenAI API connection
4. Plivo Integration - validate Plivo credentials and phone number

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local or OpenAI or Plivo"
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import inspect
import json
import math
import os
import struct
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote

import httpx
import plivo
import pytest
import websockets
from dotenv import load_dotenv
from loguru import logger

from tests.helpers import (
    LIVE_API_KEYS,
    TEST_AUTH_TOKEN,
    local_server_env,
    missing_env,
    plivo_signature_headers,
    server_log_path,
    start_server,
    stop_server,
    stream_body,
    stream_url_from_xml,
)
from utils import (
    cartesia_to_plivo,
    normalize_phone_number,
    pcm_to_ulaw,
    plivo_to_cartesia,
    resample_audio,
    ulaw_to_pcm,
)

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

TEST_PORT = 18001
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"

AGENT_MODULES = ["inbound.agent", "outbound.agent"]
SERVER_MODULES = ["inbound.server", "outbound.server"]
PUBLIC = "https://agent.example.com"
FORM = {"CallUUID": "c-1", "From": "+15551230000", "To": "+15557654321"}


# =============================================================================
# Fixtures + helpers for offline unit tests
# =============================================================================


@pytest.fixture(autouse=True)
def plivo_test_auth_token(monkeypatch):
    """Both servers check webhook signatures with TEST_AUTH_TOKEN (never a real credential)."""
    for module in SERVER_MODULES:
        monkeypatch.setattr(importlib.import_module(module), "PLIVO_AUTH_TOKEN", TEST_AUTH_TOKEN)


@pytest.fixture(params=AGENT_MODULES)
def agent_mod(request):
    """Each agent module in turn: the STT service and the tool are duplicated in both."""
    return importlib.import_module(request.param)


@pytest.fixture
def captured_messages():
    """Collect every log message text."""
    messages: list[str] = []
    sink_id = logger.add(lambda m: messages.append(m.record["message"]), level="DEBUG")
    yield messages
    logger.remove(sink_id)


def signed(method: str, public_url: str, path: str, data: dict | None = None) -> dict[str, str]:
    """Headers Plivo sends for ``path`` (query included) under ``public_url``."""
    url = public_url.rstrip("/") + path
    return plivo_signature_headers(method, url, TEST_AUTH_TOKEN, data)


def ws_path(xml: str) -> str:
    """Path + query of the <Stream> URL (what Plivo opens on this server)."""
    url = stream_url_from_xml(xml)
    return url[url.index("/ws") :]


def _server(module: str, monkeypatch, public_url: str = PUBLIC):
    server = importlib.import_module(module)
    monkeypatch.setattr(server, "PUBLIC_URL", public_url)
    return server


def _answer_path(module: str) -> str:
    return "/answer" if module == "inbound.server" else "/outbound/answer?greeting=a%20demo"


def rms_of_ulaw(ulaw_audio: bytes) -> float:
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"{len(pcm) // 2}h", pcm)
    return (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5


# =============================================================================
# UNIT TESTS - audio conversion, phone normalization
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

        correlation = sum(o * r for o, r in zip(samples, restored, strict=True))
        orig_energy = sum(o * o for o in samples)
        rest_energy = sum(r * r for r in restored)
        assert orig_energy > 0 and rest_energy > 0
        assert correlation / (orig_energy * rest_energy) ** 0.5 > 0.9

    def test_resample_changes_length_by_ratio(self):
        pcm_8k = struct.pack("160h", *([1000] * 160))
        assert len(resample_audio(pcm_8k, 8000, 24000)) == 960
        assert resample_audio(pcm_8k, 8000, 8000) == pcm_8k

    def test_plivo_cartesia_helpers(self):
        """20ms of Plivo μ-law (160 bytes) is 480 PCM16 samples at 24kHz, and back."""
        pcm_24k = plivo_to_cartesia(b"\xff" * 160)
        assert len(pcm_24k) == 960
        assert len(cartesia_to_plivo(pcm_24k)) == 160


class TestUnitPhoneNormalization:
    """Unit tests for utils.normalize_phone_number (E.164 digits, no '+')."""

    def test_normalize_e164_format(self):
        assert normalize_phone_number("+16572338892") == "16572338892"

    def test_normalize_with_spaces(self):
        assert normalize_phone_number("+1 657-233-8892") == "16572338892"

    def test_normalize_local_format(self):
        assert normalize_phone_number("(657) 233-8892", "US") == "16572338892"

    def test_normalize_other_region(self):
        assert normalize_phone_number("98804 65079", "IN") == "919880465079"

    def test_normalize_empty(self):
        assert normalize_phone_number("") == ""

    def test_normalize_unparseable_keeps_digits(self):
        assert normalize_phone_number("abc") == ""


# =============================================================================
# UNIT TESTS - ModulateSTTService event handling
# =============================================================================

CLIP = {
    "text": "I would like to check my order",
    "language": "en",
    "emotion": "neutral",
    "accent": "american",
    "deepfake_score": 0.02,
    "speaker_label": "speaker_0",
}


def make_stt(agent_mod, **kwargs: Any):
    """A ModulateSTTService whose push_frame records frames instead of sending them on."""
    service = agent_mod.ModulateSTTService(api_key="test-modulate-key", **kwargs)
    pushed: list[Any] = []

    async def record(frame, *args: Any, **kw: Any) -> None:
        pushed.append(frame)

    service.push_frame = record
    return service, pushed


class TestUnitModulateSTTEvents:
    """ModulateSTTService._handle_event: which Velma events become frames, and which do not."""

    async def test_partial_clip_pushes_interim_transcript(self, agent_mod):
        from pipecat.frames.frames import InterimTranscriptionFrame

        stt, pushed = make_stt(agent_mod)
        await stt._handle_event({"type": "partial_clip", "partial_clip": {"text": "I would"}})
        assert len(pushed) == 1
        assert type(pushed[0]) is InterimTranscriptionFrame
        assert pushed[0].text == "I would"

    @pytest.mark.parametrize(
        "event",
        [
            {"type": "partial_clip", "partial_clip": {"text": ""}},
            {"type": "partial_clip", "partial_clip": None},
            {"type": "partial_clip"},
        ],
    )
    async def test_empty_partial_clip_pushes_nothing(self, agent_mod, event):
        stt, pushed = make_stt(agent_mod)
        await stt._handle_event(event)
        assert pushed == []

    async def test_clip_pushes_final_transcript_with_extras(self, agent_mod):
        from pipecat.frames.frames import TranscriptionFrame
        from pipecat.transcriptions.language import Language

        stt, pushed = make_stt(agent_mod)
        await stt._handle_event({"type": "clip", "clip": CLIP})
        assert len(pushed) == 1
        frame = pushed[0]
        assert type(frame) is TranscriptionFrame
        assert frame.text == CLIP["text"]
        assert frame.language == Language.EN
        # emotion, accent, synthetic-voice score and speaker label ride on ``result``
        assert frame.result == CLIP

    @pytest.mark.parametrize("language", [None, "", "not-a-language"])
    async def test_clip_with_unknown_language_still_pushes(self, agent_mod, language):
        stt, pushed = make_stt(agent_mod)
        await stt._handle_event({"type": "clip", "clip": {"text": "hello", "language": language}})
        assert [f.text for f in pushed] == ["hello"]
        assert pushed[0].language is None

    @pytest.mark.parametrize(
        "event",
        [{"type": "clip", "clip": {"text": ""}}, {"type": "clip", "clip": None}, {"type": "clip"}],
    )
    async def test_empty_clip_pushes_nothing(self, agent_mod, event):
        stt, pushed = make_stt(agent_mod)
        await stt._handle_event(event)
        assert pushed == []

    async def test_clip_log_omits_transcript_text(self, agent_mod, captured_messages):
        stt, _pushed = make_stt(agent_mod)
        await stt._handle_event({"type": "clip", "clip": CLIP})
        velma = [m for m in captured_messages if m.startswith("[Velma] clip")]
        assert velma and f"{len(CLIP['text'])} chars" in velma[0]
        assert CLIP["text"] not in "\n".join(captured_messages)

    @pytest.mark.parametrize(
        "event",
        [
            # a refinement of a clip already transcribed: a second frame would repeat the words
            {"type": "clip_update", "clip_update": {**CLIP, "emotion": "frustrated"}},
            {"type": "clip_update"},
            {
                "type": "behavior_detection",
                "detection": {"detected": True, "behavior_name": "vishing", "confidence": 0.99},
            },
            {"type": "behavior_detection", "detection": {"detected": False}},
            {"type": "behavior_detection"},
            {"type": "error", "error": "bad config"},
            {"type": "something_new", "text": "ignored"},
            {},
        ],
    )
    async def test_ignored_events_push_nothing(self, agent_mod, event):
        stt, pushed = make_stt(agent_mod)
        await stt._handle_event(event)
        assert pushed == []

    async def test_clip_update_and_behavior_are_logged(self, agent_mod, captured_messages):
        stt, _pushed = make_stt(agent_mod)
        await stt._handle_event(
            {"type": "clip_update", "clip_update": {"emotion": "frustrated", "accent": "irish"}}
        )
        await stt._handle_event(
            {
                "type": "behavior_detection",
                "detection": {"detected": True, "behavior_name": "vishing", "confidence": 0.99},
            }
        )
        await stt._handle_event({"type": "error", "error": "bad config"})
        assert "[Velma] refined: frustrated irish" in captured_messages
        assert "[Velma] behaviour vishing @ 0.99" in captured_messages
        assert "Modulate error: bad config" in captured_messages

    async def test_undetected_behavior_is_not_logged(self, agent_mod, captured_messages):
        stt, _pushed = make_stt(agent_mod)
        await stt._handle_event(
            {"type": "behavior_detection", "detection": {"detected": False, "behavior_name": "x"}}
        )
        assert not [m for m in captured_messages if "behaviour" in m]

    async def test_configuration_frame(self, agent_mod):
        """The one JSON text frame sent before any audio."""
        default, _ = make_stt(agent_mod)
        assert json.loads(default._config) == {
            "behaviors": [],
            "produce_topics": False,
            "produce_summary": False,
        }
        custom, _ = make_stt(
            agent_mod, behaviors=["preset:vishing"], produce_topics=True, produce_summary=True
        )
        assert json.loads(custom._config) == {
            "behaviors": ["preset:vishing"],
            "produce_topics": True,
            "produce_summary": True,
        }

    async def test_run_stt_drops_audio_while_disconnected(self, agent_mod):
        """No socket: the frame is dropped and nothing connects from run_stt."""
        stt, pushed = make_stt(agent_mod)
        assert [frame async for frame in stt.run_stt(b"\x00" * 320)] == [None]
        assert pushed == []

    def test_constants(self, agent_mod):
        assert agent_mod.MODULATE_STT_URL == "wss://platform.modulate.ai/api/velma-2-streaming"
        assert agent_mod.MODULATE_STT_MODEL == "velma-2"


class TestUnitInboundOutboundParity:
    """inbound/agent.py and outbound/agent.py carry identical copies of the service and tool."""

    async def test_same_frames_for_the_same_events(self):
        events = [
            {"type": "partial_clip", "partial_clip": {"text": "I would"}},
            {"type": "clip", "clip": CLIP},
            {"type": "clip_update", "clip_update": CLIP},
            {"type": "behavior_detection", "detection": {"detected": True}},
            {"type": "clip", "clip": {"text": ""}},
        ]
        seen = []
        for name in AGENT_MODULES:
            stt, pushed = make_stt(importlib.import_module(name))
            for event in events:
                await stt._handle_event(event)
            seen.append([(type(f).__name__, f.text, f.language, f.result) for f in pushed])
        assert seen[0] == seen[1]
        assert [kind for kind, *_ in seen[0]] == ["InterimTranscriptionFrame", "TranscriptionFrame"]

    @pytest.mark.parametrize("name", ["ModulateSTTService", "search_the_web"])
    def test_source_is_identical(self, name):
        inbound, outbound = (importlib.import_module(m) for m in AGENT_MODULES)
        assert inspect.getsource(getattr(inbound, name)) == inspect.getsource(
            getattr(outbound, name)
        )

    @pytest.mark.parametrize(
        "name",
        [
            "MODULATE_STT_URL",
            "MODULATE_STT_MODEL",
            "MODULATE_AUTH_CLOSE_CODE",
            "TAVILY_TIMEOUT_SECS",
            "TAVILY_SEARCH_DEPTH",
            "LLM_MODEL",
            "TTS_MODEL",
            "TTS_VOICE",
        ],
    )
    def test_constants_match(self, name):
        inbound, outbound = (importlib.import_module(m) for m in AGENT_MODULES)
        assert getattr(inbound, name) == getattr(outbound, name)

    def test_tool_schema_matches(self):
        inbound, outbound = (importlib.import_module(m) for m in AGENT_MODULES)
        for schema in (inbound.SEARCH_SCHEMA, outbound.SEARCH_SCHEMA):
            assert schema.name == "search_the_web"
            assert schema.required == ["query"]
            assert set(schema.properties) == {"query"}
        assert inbound.SEARCH_SCHEMA.description == outbound.SEARCH_SCHEMA.description


# =============================================================================
# UNIT TESTS - Tavily search tool (stubbed tavily.AsyncTavilyClient)
# =============================================================================


class FakeTavily:
    """Stands in for tavily.AsyncTavilyClient; ``behaviour`` decides what search() does."""

    instances: list[FakeTavily]
    behaviour: Any = None

    def __init__(self, api_key: str | None = None, **kwargs: Any) -> None:
        self.api_key = api_key
        self.searches: list[dict[str, Any]] = []
        type(self).instances.append(self)

    async def search(self, **kwargs: Any) -> dict:
        self.searches.append(kwargs)
        behaviour = type(self).behaviour
        if isinstance(behaviour, Exception):
            raise behaviour
        if behaviour == "hang":
            await asyncio.sleep(30)
        return behaviour


@pytest.fixture
def tavily_stub(monkeypatch, agent_mod):
    """Stub the lazily imported client and give the agent module a (fake) API key."""
    import tavily

    FakeTavily.instances = []
    FakeTavily.behaviour = {"answer": "", "results": []}
    monkeypatch.setattr(tavily, "AsyncTavilyClient", FakeTavily)
    monkeypatch.setattr(agent_mod, "TAVILY_API_KEY", "tvly-test")
    return FakeTavily


async def call_tool(agent_mod, arguments: dict[str, Any]) -> list[Any]:
    """Run search_the_web the way Pipecat does and return what it handed the LLM."""
    results: list[Any] = []

    async def result_callback(result: Any, **kwargs: Any) -> None:
        results.append(result)

    params = SimpleNamespace(arguments=arguments, result_callback=result_callback)
    await agent_mod.search_the_web(params)
    return results


class TestUnitTavilySearchTool:
    """search_the_web: every path gives the LLM one short sentence and never raises."""

    async def test_no_api_key(self, monkeypatch, agent_mod, tavily_stub):
        monkeypatch.setattr(agent_mod, "TAVILY_API_KEY", "")
        results = await call_tool(agent_mod, {"query": "plivo pricing"})
        assert results == [{"result": "Web search is not configured. Say you are not certain."}]
        assert tavily_stub.instances == []  # no client, no request

    @pytest.mark.parametrize("arguments", [{}, {"query": ""}, {"query": "   "}, {"query": None}])
    async def test_empty_query(self, agent_mod, tavily_stub, arguments):
        results = await call_tool(agent_mod, arguments)
        assert results == [{"result": "No search query was given. Ask what to look up."}]
        assert tavily_stub.instances == []

    async def test_client_raises(self, agent_mod, tavily_stub, captured_messages):
        tavily_stub.behaviour = RuntimeError("401 for key tvly-test")
        results = await call_tool(agent_mod, {"query": "plivo pricing"})
        assert results == [
            {"result": "The search failed. Tell the caller you could not look that up."}
        ]
        # only the exception type is logged, never its text
        assert "Tavily search failed: RuntimeError" in captured_messages
        assert "tvly-test" not in "\n".join(captured_messages)

    async def test_client_hangs_past_the_timeout(self, monkeypatch, agent_mod, tavily_stub):
        monkeypatch.setattr(agent_mod, "TAVILY_TIMEOUT_SECS", 0.05)
        tavily_stub.behaviour = "hang"
        started = time.monotonic()
        results = await call_tool(agent_mod, {"query": "plivo pricing"})
        assert time.monotonic() - started < 2.0
        assert results == [
            {"result": "The search took too long. Tell the caller you could not look that up."}
        ]

    @pytest.mark.parametrize(
        "response",
        [
            {"answer": "", "results": [{"title": "A"}]},
            {"answer": "   ", "results": []},
            {"answer": None},
            {},
        ],
    )
    async def test_empty_answer(self, agent_mod, tavily_stub, response):
        tavily_stub.behaviour = response
        results = await call_tool(agent_mod, {"query": "plivo pricing"})
        assert results == [{"result": "No reliable sources found for that."}]

    async def test_answer_with_sources(self, agent_mod, tavily_stub):
        tavily_stub.behaviour = {
            "answer": " Plivo charges per minute. ",
            "results": [{"title": "Plivo Pricing"}, {"title": "Docs"}, {"title": "Third"}],
        }
        results = await call_tool(agent_mod, {"query": "  plivo pricing  "})
        # the first two result titles are named; the third is not
        assert results == [{"result": "Plivo charges per minute.\nSources: Plivo Pricing, Docs."}]

    @pytest.mark.parametrize("results_field", [[], None, [{"title": ""}, {"url": "x"}]])
    async def test_answer_without_named_sources(self, agent_mod, tavily_stub, results_field):
        tavily_stub.behaviour = {"answer": "Yes.", "results": results_field}
        assert await call_tool(agent_mod, {"query": "q"}) == [{"result": "Yes."}]

    async def test_search_arguments(self, agent_mod, tavily_stub):
        """What ships: the configured depth, an advanced answer, 5 results, a bounded request."""
        tavily_stub.behaviour = {"answer": "Yes.", "results": []}
        await call_tool(agent_mod, {"query": "  plivo pricing  "})
        (client,) = tavily_stub.instances
        assert client.api_key == "tvly-test"
        assert client.searches == [
            {
                "query": "plivo pricing",
                "search_depth": agent_mod.TAVILY_SEARCH_DEPTH,
                "include_answer": "advanced",
                "max_results": 5,
                "timeout": agent_mod.TAVILY_TIMEOUT_SECS,
            }
        ]

    async def test_query_is_not_logged(self, agent_mod, tavily_stub, captured_messages):
        tavily_stub.behaviour = {"answer": "Yes.", "results": [{"title": "T"}]}
        await call_tool(agent_mod, {"query": "my account number is 99887766"})
        assert "99887766" not in "\n".join(captured_messages)

    def test_defaults(self, agent_mod):
        assert agent_mod.TAVILY_TIMEOUT_SECS == 5.0


# =============================================================================
# UNIT TESTS - prompts, greeting, run_agent wiring
# =============================================================================


class _FakeTask:
    """Stands in for PipelineTask: records what run_agent queues, runs nothing."""

    created: list[_FakeTask]

    def __init__(self, pipeline: Any, **kwargs: Any) -> None:
        self.pipeline = pipeline
        self.queued: list[Any] = []
        type(self).created.append(self)

    async def queue_frames(self, frames: Any) -> None:
        self.queued.extend(frames)

    async def cancel(self, *args: Any, **kwargs: Any) -> None:
        return None


class _FakeRunner:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def run(self, task: Any) -> None:
        return None


class _FakeWebSocket:
    """Enough of a FastAPI WebSocket for the transport's constructor; never used for I/O."""

    client_state = None
    application_state = None


async def run_agent_offline(monkeypatch, module_name: str, **kwargs: Any) -> tuple[Any, list]:
    """Assemble the real pipeline but never run it: no socket is opened to any service.

    Returns (the LLMContext passed to the aggregators, the frames queued on the task).
    """
    mod = importlib.import_module(module_name)
    for key in ("OPENAI_API_KEY", "MODULATE_API_KEY", "CARTESIA_API_KEY"):
        monkeypatch.setattr(mod, key, "test-key")
    _FakeTask.created = []
    contexts: list[Any] = []
    real_pair = mod.LLMContextAggregatorPair

    def recording_pair(context: Any, *args: Any, **kw: Any):
        contexts.append(context)
        return real_pair(context, *args, **kw)

    monkeypatch.setattr(mod, "LLMContextAggregatorPair", recording_pair)
    monkeypatch.setattr(mod, "PipelineTask", _FakeTask)
    monkeypatch.setattr(mod, "PipelineRunner", _FakeRunner)
    await mod.run_agent(websocket=_FakeWebSocket(), call_id="c-1", stream_id="s-1", **kwargs)
    (task,) = _FakeTask.created
    (context,) = contexts
    return context, task.queued


def _messages(context: Any) -> list[dict]:
    return [m for m in context.get_messages() if isinstance(m, dict)]


class TestUnitOutboundGreeting:
    """The answer_url ``greeting`` is spoken verbatim; DEFAULT_GREETING applies when absent."""

    GREETING = "Hi, this is a courtesy call about your appointment tomorrow. Is now a good time?"

    @pytest.mark.parametrize("greeting", ["", "   ", GREETING, f"  {GREETING}  "])
    async def test_greeting_goes_straight_to_tts(self, monkeypatch, greeting):
        from pipecat.frames.frames import TTSSpeakFrame

        from outbound import agent as agent_mod

        context, queued = await run_agent_offline(monkeypatch, "outbound.agent", greeting=greeting)
        expected = greeting.strip() or agent_mod.DEFAULT_GREETING
        assert len(queued) == 1
        assert type(queued[0]) is TTSSpeakFrame
        assert queued[0].text == expected
        # recorded once as an assistant turn so the LLM does not introduce itself again
        assert _messages(context) == [
            {"role": "system", "content": agent_mod.SYSTEM_PROMPT},
            {"role": "assistant", "content": expected},
        ]

    async def test_default_greeting_when_argument_omitted(self, monkeypatch):
        from outbound import agent as agent_mod

        _context, queued = await run_agent_offline(monkeypatch, "outbound.agent")
        assert queued[0].text == agent_mod.DEFAULT_GREETING
        assert agent_mod.DEFAULT_GREETING.strip()

    async def test_inbound_opens_with_an_llm_turn(self, monkeypatch):
        """Inbound has no scripted greeting: a stand-in caller turn makes the LLM open."""
        from pipecat.frames.frames import LLMContextFrame

        from inbound import agent as agent_mod

        context, queued = await run_agent_offline(monkeypatch, "inbound.agent")
        assert len(queued) == 1
        assert type(queued[0]) is LLMContextFrame
        assert _messages(queued[0].context) == [
            {"role": "system", "content": agent_mod.SYSTEM_PROMPT},
            {"role": "user", "content": agent_mod.OPENING_USER_MESSAGE},
        ]
        # the stand-in turn is not part of the conversation context
        assert _messages(context) == [{"role": "system", "content": agent_mod.SYSTEM_PROMPT}]

    @pytest.mark.parametrize("module", AGENT_MODULES)
    def test_run_agent_signature(self, module):
        params = list(inspect.signature(importlib.import_module(module).run_agent).parameters)
        expected = ["websocket", "call_id", "stream_id", "from_number", "to_number"]
        if module == "outbound.agent":
            expected.append("greeting")
        assert params == expected


class TestUnitSystemPromptSource:
    """system_prompt.md is the only prompt source; a SYSTEM_PROMPT env var is ignored."""

    @pytest.mark.parametrize("direction", ["inbound", "outbound"])
    def test_env_var_does_not_override_file(self, monkeypatch, direction):
        import importlib.util
        import sys

        monkeypatch.setenv("SYSTEM_PROMPT", "You are a pirate.")
        path = Path(__file__).parent.parent / direction / "agent.py"
        spec = importlib.util.spec_from_file_location(f"{direction}_agent_env_prompt", path)
        mod = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, mod)
        spec.loader.exec_module(mod)
        expected = (path.parent / "system_prompt.md").read_text().strip()
        assert expected == mod.SYSTEM_PROMPT
        assert "pirate" not in mod.SYSTEM_PROMPT

    @pytest.mark.parametrize("module", AGENT_MODULES)
    def test_prompt_mentions_the_tool_and_has_no_placeholders(self, module):
        prompt = importlib.import_module(module).SYSTEM_PROMPT
        assert "search_the_web" in prompt
        assert "{{" not in prompt and "}}" not in prompt

    def test_outbound_prompt_does_not_reintroduce(self):
        from outbound.agent import SYSTEM_PROMPT

        assert "already been spoken" in SYSTEM_PROMPT


# =============================================================================
# UNIT TESTS - server routes (FastAPI TestClient, no network)
# =============================================================================


class _RunAgentRecorder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)


def _stream_attributes(xml: str) -> dict[str, str]:
    from xml.etree import ElementTree

    stream = ElementTree.fromstring(xml).find("Stream")
    assert stream is not None, xml
    return dict(stream.attrib)


class TestUnitServerRoutes:
    """Offline tests for the inbound/outbound FastAPI routes."""

    def test_inbound_answer_returns_stream_xml(self, monkeypatch):
        from fastapi.testclient import TestClient

        server = _server("inbound.server", monkeypatch, "https://example.ngrok.app")
        form = {"CallUUID": "uuid-1", "From": "+15551234567", "To": "+16572338892"}
        resp = TestClient(server.app).post(
            "/answer",
            data=form,
            headers=signed("POST", "https://example.ngrok.app", "/answer", form),
        )
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        assert _stream_attributes(resp.text) == {
            "bidirectional": "true",
            "keepCallAlive": "true",
            "contentType": "audio/x-mulaw;rate=8000",
        }
        url = stream_url_from_xml(resp.text)
        assert url.startswith("wss://example.ngrok.app/ws?body=")
        # ``body`` is percent-encoded base64: decode both layers
        raw_body = url.split("body=", 1)[1]
        assert "+" not in raw_body and "/" not in raw_body
        assert json.loads(base64.b64decode(unquote(raw_body))) == {
            "call_uuid": "uuid-1",
            "from": "+15551234567",
            "to": "+16572338892",
        }

    @pytest.mark.parametrize("greeting", ["", "Hi, quick question about your order?"])
    def test_outbound_answer_returns_stream_xml(self, monkeypatch, greeting):
        from urllib.parse import quote

        from fastapi.testclient import TestClient

        server = _server("outbound.server", monkeypatch)
        path = "/outbound/answer" + (f"?greeting={quote(greeting, safe='')}" if greeting else "")
        resp = TestClient(server.app).post(
            path, data=FORM, headers=signed("POST", PUBLIC, path, FORM)
        )
        assert resp.status_code == 200
        assert "application/xml" in resp.headers["content-type"]
        assert _stream_attributes(resp.text) == {
            "bidirectional": "true",
            "keepCallAlive": "true",
            "contentType": "audio/x-mulaw;rate=8000",
        }
        assert stream_url_from_xml(resp.text).startswith("wss://agent.example.com/ws?body=")
        assert stream_body(resp.text) == {
            "call_uuid": "c-1",
            "from": FORM["From"],
            "to": FORM["To"],
            "greeting": greeting,
        }

    def test_outbound_answer_get_reads_plivo_fields_from_query(self, monkeypatch):
        from fastapi.testclient import TestClient

        server = _server("outbound.server", monkeypatch)
        path = "/outbound/answer?CallUUID=c-9&From=%2B15551230000&To=%2B15557654321&greeting=hi"
        resp = TestClient(server.app).get(path, headers=signed("GET", PUBLIC, path))
        assert resp.status_code == 200
        assert stream_body(resp.text) == {
            "call_uuid": "c-9",
            "from": "+15551230000",
            "to": "+15557654321",
            "greeting": "hi",
        }

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_stream_url_percent_encodes_body(self, monkeypatch, module):
        """A raw '+' in base64 would reach /ws as a space."""
        server = _server(module, monkeypatch)
        data = {"greeting": "~~~???>>>"}  # base64 of this JSON contains '+' and '/'
        assert {"+", "/"} & set(base64.b64encode(json.dumps(data).encode()).decode())
        url = server.stream_url(data)
        raw_body = url.split("body=", 1)[1]
        assert "+" not in raw_body and "/" not in raw_body
        assert json.loads(base64.b64decode(unquote(raw_body))) == data
        assert url.startswith("wss://agent.example.com/ws?body=")

    @pytest.mark.parametrize(
        ("public_url", "ws_base"),
        [
            ("https://a.example.com/", "wss://a.example.com"),
            ("http://localhost:8000", "ws://localhost:8000"),
        ],
    )
    def test_stream_url_scheme(self, monkeypatch, public_url, ws_base):
        server = _server("inbound.server", monkeypatch, public_url)
        assert server.stream_url({}).startswith(f"{ws_base}/ws?body=")

    @pytest.mark.parametrize(
        ("module", "service"),
        [
            ("inbound.server", "gpt4o-modulatevelma2-cartesiasonic3-pipecat-inbound"),
            ("outbound.server", "gpt4o-modulatevelma2-cartesiasonic3-pipecat-outbound"),
        ],
    )
    def test_health(self, monkeypatch, module, service):
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch)
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "")
        assert TestClient(server.app).get("/").json() == {
            "status": "ok",
            "service": service,
            "phone_number": "not configured",
        }
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "+1 415-555-0123")
        assert TestClient(server.app).get("/").json()["phone_number"] == "+14155550123"

    def test_inbound_fallback_and_hold_xml(self, monkeypatch):
        from fastapi.testclient import TestClient

        client = TestClient(_server("inbound.server", monkeypatch).app)
        fallback = client.post(
            "/fallback", data=FORM, headers=signed("POST", PUBLIC, "/fallback", FORM)
        )
        assert "<Speak" in fallback.text and "<Hangup" in fallback.text
        hold = client.post("/hold", data=FORM, headers=signed("POST", PUBLIC, "/hold", FORM))
        assert "<Wait" in hold.text

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("POST", "/outbound/call"),
            ("GET", "/outbound/status/abc"),
            ("POST", "/outbound/hangup/abc"),
            ("GET", "/outbound/campaign/abc"),
            ("GET", "/hold"),
            ("POST", "/hold"),
            ("POST", "/answer"),
        ],
    )
    def test_outbound_has_no_dial_or_tracking_routes(self, monkeypatch, method, path):
        """Calls are placed with Plivo's Make Call API; the server keeps no call state."""
        from fastapi.testclient import TestClient

        client = TestClient(_server("outbound.server", monkeypatch).app)
        resp = client.request(method, path, headers=signed(method, PUBLIC, path))
        assert resp.status_code == 404

    def test_outbound_route_table(self):
        from outbound import server

        routes = {(m, r.path) for r in server.app.routes for m in getattr(r, "methods", None) or ()}
        plivo_routes = {(m, p) for m, p in routes if p.startswith("/outbound")}
        assert plivo_routes == {
            ("GET", "/outbound/answer"),
            ("POST", "/outbound/answer"),
            ("POST", "/outbound/hangup"),
        }
        for name in ("CallManager", "OutboundCallRecord", "determine_outcome"):
            assert not hasattr(server, name)

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_ws_with_issued_stream_url_runs_agent(self, monkeypatch, module):
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch)
        recorder = _RunAgentRecorder()
        monkeypatch.setattr(server, "run_agent", recorder)
        client = TestClient(server.app)
        path = _answer_path(module)
        answer = client.post(path, data=FORM, headers=signed("POST", PUBLIC, path, FORM))
        with client.websocket_connect(ws_path(answer.text)) as ws:
            ws.send_text(
                json.dumps({"event": "start", "start": {"callId": "c-1", "streamId": "s-1"}})
            )
        assert len(recorder.calls) == 1
        call = recorder.calls[0]
        expected = {
            "call_id": "c-1",
            "stream_id": "s-1",
            "from_number": FORM["From"],
            "to_number": FORM["To"],
        }
        if module == "outbound.server":
            expected["greeting"] = "a demo"
        assert {k: v for k, v in call.items() if k != "websocket"} == expected

    def test_outbound_ws_without_greeting_passes_empty_string(self, monkeypatch):
        """No ``greeting`` on the answer_url: the agent module's default applies."""
        from fastapi.testclient import TestClient

        server = _server("outbound.server", monkeypatch)
        recorder = _RunAgentRecorder()
        monkeypatch.setattr(server, "run_agent", recorder)
        client = TestClient(server.app)
        path = "/outbound/answer"
        answer = client.post(path, data=FORM, headers=signed("POST", PUBLIC, path, FORM))
        with client.websocket_connect(ws_path(answer.text)) as ws:
            ws.send_text(json.dumps({"event": "start", "start": {"callId": "c-1"}}))
        assert recorder.calls[0]["greeting"] == ""
        assert recorder.calls[0]["stream_id"] == ""

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_ws_requires_start_event_first(self, monkeypatch, module):
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch)
        recorder = _RunAgentRecorder()
        monkeypatch.setattr(server, "run_agent", recorder)
        with TestClient(server.app).websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"event": "media", "media": {"payload": ""}}))
        assert recorder.calls == []

    def test_inbound_auto_config_skipped_without_settings(self, monkeypatch, captured_messages):
        """Missing settings: no Plivo REST client is ever built."""
        from inbound import server

        def no_client(*args: Any, **kwargs: Any):
            raise AssertionError("Plivo REST client must not be created")

        monkeypatch.setattr(server.plivo, "RestClient", no_client)
        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "")
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "")
        monkeypatch.setattr(server, "PUBLIC_URL", "")
        assert server.configure_plivo_webhooks() is False
        skipped = [m for m in captured_messages if m.startswith("Skipping Plivo auto-config")]
        assert skipped
        assert all(n in skipped[0] for n in ("PLIVO_AUTH_ID", "PLIVO_PHONE_NUMBER", "PUBLIC_URL"))

    def test_inbound_auto_config_points_number_at_answer_and_hangup(self, monkeypatch):
        from inbound import server

        calls: list[tuple[str, dict]] = []

        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                self.applications = SimpleNamespace(
                    list=lambda **kw: {"objects": [{"app_name": "Other", "app_id": "1"}]},
                    create=lambda **kw: calls.append(("create", kw)) or {"app_id": "42"},
                    update=lambda **kw: calls.append(("update", kw)),
                )
                self.numbers = SimpleNamespace(update=lambda **kw: calls.append(("number", kw)))

        monkeypatch.setattr(server.plivo, "RestClient", FakeClient)
        monkeypatch.setattr(server, "PLIVO_AUTH_ID", "MATEST")
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "+1 415-555-0123")
        monkeypatch.setattr(server, "PUBLIC_URL", "https://agent.example.com/")
        assert server.configure_plivo_webhooks() is True
        assert calls == [
            (
                "create",
                {
                    "app_name": server.PLIVO_APP_NAME,
                    "answer_url": "https://agent.example.com/answer",
                    "answer_method": "POST",
                    "hangup_url": "https://agent.example.com/hangup",
                    "hangup_method": "POST",
                },
            ),
            ("number", {"number": "14155550123", "app_id": "42"}),
        ]


# =============================================================================
# UNIT TESTS - webhook authentication (Plivo V3 signatures, always on)
# =============================================================================

# Every Plivo webhook route: (module, method, path)
WEBHOOK_ROUTES = [
    ("inbound.server", "POST", "/answer"),
    ("inbound.server", "GET", "/answer?CallUUID=c-1&From=%2B15551230000"),
    ("inbound.server", "POST", "/hangup"),
    ("inbound.server", "POST", "/fallback"),
    ("inbound.server", "GET", "/hold"),
    ("inbound.server", "POST", "/hold"),
    ("outbound.server", "POST", "/outbound/answer"),
    ("outbound.server", "POST", "/outbound/answer?greeting=a%20demo"),
    ("outbound.server", "GET", "/outbound/answer?CallUUID=c-1&greeting=hi"),
    ("outbound.server", "POST", "/outbound/hangup"),
]


class TestUnitWebhookAuth:
    """Plivo V3 signature checks on every HTTP webhook of both servers."""

    def test_every_route_but_health_and_ws_is_signature_checked(self):
        """A new webhook route cannot be added without the dependency."""
        for module in SERVER_MODULES:
            server = importlib.import_module(module)
            for route in server.app.routes:
                if not hasattr(route, "methods") or route.path in ("/", "/ws"):
                    continue
                if route.path in ("/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"):
                    continue
                calls = [d.call for d in route.dependant.dependencies]
                assert server.verify_plivo_signature in calls, f"{module} {route.path}"

    @pytest.mark.parametrize(("module", "method", "path"), WEBHOOK_ROUTES)
    def test_every_webhook_accepts_signed_and_rejects_unsigned(
        self, monkeypatch, module, method, path
    ):
        from fastapi.testclient import TestClient

        client = TestClient(_server(module, monkeypatch).app)
        data = FORM if method == "POST" else None
        ok = client.request(method, path, data=data, headers=signed(method, PUBLIC, path, data))
        assert ok.status_code == 200, ok.text
        unsigned = client.request(method, path, data=data)
        assert unsigned.status_code == 403
        assert unsigned.json() == {"detail": "Invalid Plivo signature"}

    @pytest.mark.parametrize(("module", "method", "path"), WEBHOOK_ROUTES)
    def test_every_webhook_rejects_wrong_token(self, monkeypatch, module, method, path):
        from fastapi.testclient import TestClient

        client = TestClient(_server(module, monkeypatch).app)
        data = FORM if method == "POST" else None
        headers = plivo_signature_headers(method, PUBLIC + path, "some-other-token", data)
        assert client.request(method, path, data=data, headers=headers).status_code == 403

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_valid_post_logs_verification(self, monkeypatch, captured_messages, module):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        resp = client.post(path, data=FORM, headers=signed("POST", PUBLIC, path, FORM))
        assert resp.status_code == 200
        suffix = " + query string" if "?" in path else ""
        assert f"Plivo signature verified: POST {path.split('?')[0]}{suffix}" in captured_messages

    @pytest.mark.parametrize("module", SERVER_MODULES)
    @pytest.mark.parametrize("drop", ["X-Plivo-Signature-V3", "X-Plivo-Signature-V3-Nonce", "both"])
    def test_missing_headers_rejected(self, monkeypatch, captured_messages, module, drop):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        headers = signed("POST", PUBLIC, path, FORM)
        for name in list(headers):
            if drop in (name, "both"):
                del headers[name]
        assert client.post(path, data=FORM, headers=headers).status_code == 403
        rejected = f"Rejected Plivo webhook POST {path.split('?')[0]}: missing"
        assert any(rejected in m for m in captured_messages)

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_wrong_signature_rejected_and_not_logged(self, monkeypatch, captured_messages, module):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        headers = plivo_signature_headers("POST", PUBLIC + path, "some-other-token", FORM)
        assert client.post(path, data=FORM, headers=headers).status_code == 403
        rejected = [m for m in captured_messages if "Rejected Plivo webhook" in m]
        assert rejected and "signature mismatch" in rejected[0]
        joined = "\n".join(captured_messages)
        assert headers["X-Plivo-Signature-V3"] not in joined
        assert TEST_AUTH_TOKEN not in joined

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_tampered_form_field_rejected(self, monkeypatch, module):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        headers = signed("POST", PUBLIC, path, FORM)
        tampered = {**FORM, "To": "+19995550000"}
        assert client.post(path, data=tampered, headers=headers).status_code == 403
        extra = {**FORM, "Extra": "x"}
        assert client.post(path, data=extra, headers=headers).status_code == 403

    @pytest.mark.parametrize("method", ["POST", "GET"])
    def test_tampered_outbound_query_string_rejected(self, monkeypatch, method):
        """The answer_url greeting is covered by the signature."""
        from fastapi.testclient import TestClient

        client = TestClient(_server("outbound.server", monkeypatch).app)
        data = FORM if method == "POST" else None
        signed_path = "/outbound/answer?greeting=a%20demo&x=1"
        headers = signed(method, PUBLIC, signed_path, data)
        for sent in (
            "/outbound/answer?greeting=free%20money&x=1",
            "/outbound/answer?greeting=a%20demo&x=1&y=injected",
            "/outbound/answer?greeting=a%20demo",
        ):
            assert client.request(method, sent, data=data, headers=headers).status_code == 403
        ok = client.request(method, signed_path, data=data, headers=headers)
        assert ok.status_code == 200

    def test_tampered_inbound_get_query_rejected(self, monkeypatch):
        from fastapi.testclient import TestClient

        client = TestClient(_server("inbound.server", monkeypatch).app)
        signed_path = "/answer?CallUUID=c-1&From=%2B15551230000"
        headers = signed("GET", PUBLIC, signed_path)
        assert (
            client.get("/answer?CallUUID=c-1&From=%2B19995550000", headers=headers).status_code
            == 403
        )
        assert client.get(signed_path, headers=headers).status_code == 200

    def test_post_with_query_base_string_rule(self):
        """SDK rule: URL + '?' + sorted decoded query + '.' + sorted form name/value pairs."""
        from plivo.utils.signature_v3 import construct_post_url

        base = construct_post_url(
            f"{PUBLIC}/outbound/answer?x=c&greeting=a%20demo", {"To": "1", "From": "2"}
        )
        assert base.decode() == f"{PUBLIC}/outbound/answer?greeting=a demo&x=c.From2To1"

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_signature_checked_against_public_url_not_request_url(self, monkeypatch, module):
        """Behind a tunnel the server sees http://localhost (here http://testserver)."""
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch).app)
        seen_by_server = plivo_signature_headers(
            "POST", f"http://testserver{path}", TEST_AUTH_TOKEN, FORM
        )
        assert client.post(path, data=FORM, headers=seen_by_server).status_code == 403
        as_plivo = signed("POST", PUBLIC, path, FORM)
        assert client.post(path, data=FORM, headers=as_plivo).status_code == 200

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_public_request_url_reconstruction(self, monkeypatch, module):
        from starlette.requests import Request

        server = _server(module, monkeypatch, "https://agent.example.com/")
        scope = {
            "type": "http",
            "method": "POST",
            "scheme": "http",
            "server": ("127.0.0.1", 8000),
            "path": "/outbound/answer",
            "query_string": b"greeting=a%20demo&x=1",
            "headers": [(b"host", b"localhost:8000")],
        }
        assert server.public_request_url(Request(scope)) == (
            "https://agent.example.com/outbound/answer?greeting=a%20demo&x=1"
        )
        scope["query_string"] = b""
        assert server.public_request_url(Request(scope)) == (
            "https://agent.example.com/outbound/answer"
        )

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_public_url_is_read_at_request_time(self, monkeypatch, module):
        """PUBLIC_URL set after import (a tunnel started at runtime) is the one that counts."""
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch, "https://old.example.com")
        path = _answer_path(module)
        client = TestClient(server.app)
        monkeypatch.setattr(server, "PUBLIC_URL", "https://new.example.com")
        new = signed("POST", "https://new.example.com", path, FORM)
        resp = client.post(path, data=FORM, headers=new)
        assert resp.status_code == 200
        assert stream_url_from_xml(resp.text).startswith("wss://new.example.com/ws?")
        old = signed("POST", "https://old.example.com", path, FORM)
        assert client.post(path, data=FORM, headers=old).status_code == 403

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_no_public_url_rejects(self, monkeypatch, captured_messages, module):
        from fastapi.testclient import TestClient

        path = _answer_path(module)
        client = TestClient(_server(module, monkeypatch, "").app)
        headers = signed("POST", PUBLIC, path, FORM)
        assert client.post(path, data=FORM, headers=headers).status_code == 403
        assert any("PUBLIC_URL not set" in m for m in captured_messages)

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_empty_token_rejects_every_request(self, monkeypatch, module):
        """Even a request signed with the empty token is refused."""
        from fastapi.testclient import TestClient

        server = _server(module, monkeypatch)
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "")
        path = _answer_path(module)
        headers = plivo_signature_headers("POST", PUBLIC + path, "", FORM)
        assert TestClient(server.app).post(path, data=FORM, headers=headers).status_code == 403

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_refuses_to_start_without_auth_token(self, monkeypatch, captured_messages, module):
        import sys

        server = _server(module, monkeypatch)
        monkeypatch.setattr(server, "PLIVO_AUTH_TOKEN", "")
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "")  # never reach Plivo auto-config
        monkeypatch.setattr(sys, "argv", [module])
        started = []
        monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: started.append(True))
        with pytest.raises(SystemExit) as exc:
            server.main()
        assert exc.value.code == 1
        assert started == []
        assert any("PLIVO_AUTH_TOKEN is empty" in m for m in captured_messages)

    @pytest.mark.parametrize("module", SERVER_MODULES)
    def test_starts_with_auth_token(self, monkeypatch, captured_messages, module):
        server = _server(module, monkeypatch)
        monkeypatch.setattr(server, "PLIVO_PHONE_NUMBER", "")  # never reach Plivo auto-config
        started: list[dict] = []
        monkeypatch.setattr(server.uvicorn, "run", lambda *a, **k: started.append(k))
        server.main()
        assert [k["port"] for k in started] == [server.SERVER_PORT]
        assert "Webhook auth: Plivo V3 signatures on webhooks" in captured_messages
        if module == "outbound.server":
            hint = [m for m in captured_messages if m.startswith("Place calls with Plivo's Make")]
            assert hint and f"answer_url={PUBLIC}/outbound/answer?greeting=" in hint[0]


# =============================================================================
# LOCAL INTEGRATION TESTS (real LLM/STT/TTS, no phone call)
# =============================================================================


@pytest.mark.skipif(
    bool(missing_env(*LIVE_API_KEYS)),
    reason=f"not configured: {', '.join(missing_env(*LIVE_API_KEYS))}",
)
class TestLocalIntegration:
    """Integration tests using a local WebSocket connection to the inbound server."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the inbound server as a subprocess (SIGTERM -> wait(5) -> SIGKILL).

        The env carries no Plivo account or number, so the server cannot reconfigure a
        real number; webhook auth is keyed with a test token and requests are signed.
        """
        log_path = server_log_path("modulate_integration_local_server")
        proc = start_server("inbound.server", TEST_PORT, log_path, local_server_env(TEST_PORT))
        yield proc
        stop_server(proc)

    async def test_local_health_check(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.json()["phone_number"] == "not configured"

    @staticmethod
    async def _answer(call_uuid: str, signed_request: bool = True) -> httpx.Response:
        form = {"CallUUID": call_uuid, "From": "+15551234567", "To": "+16572338892"}
        headers = signed("POST", LOCAL_HTTP_URL, "/answer", form) if signed_request else {}
        async with httpx.AsyncClient() as client:
            return await client.post(f"{LOCAL_HTTP_URL}/answer", data=form, headers=headers)

    async def _stream_url(self, call_uuid: str) -> str:
        """The /ws URL from a Plivo-signed answer webhook."""
        return stream_url_from_xml((await self._answer(call_uuid)).text)

    async def test_local_unsigned_answer_rejected(self, server_process):
        assert (await self._answer("test-unsigned", signed_request=False)).status_code == 403

    async def test_local_answer_webhook(self, server_process):
        response = await self._answer("test123")
        assert response.status_code == 200
        assert "application/xml" in response.headers["content-type"]
        assert _stream_attributes(response.text)["contentType"] == "audio/x-mulaw;rate=8000"
        assert stream_url_from_xml(response.text).startswith(f"ws://localhost:{TEST_PORT}/ws?body=")
        assert stream_body(response.text)["call_uuid"] == "test123"

    async def test_local_websocket_connection(self, server_process):
        """A Plivo start event produces playAudio (the LLM's opening line) within 20s."""
        async with websockets.connect(await self._stream_url("test123"), close_timeout=2) as ws:
            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
                    }
                )
            )
            play_audio = None
            deadline = time.time() + 20
            try:
                while time.time() < deadline:
                    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=20))
                    if data.get("event") == "playAudio":
                        play_audio = data
                        break
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass
        assert play_audio, "No audio received from server"
        assert play_audio["media"]["contentType"] == "audio/x-mulaw"
        assert play_audio["media"]["sampleRate"] == 8000

    async def test_local_audio_quality(self, server_process):
        """The opening line has speech energy (RMS > 500) while we stream silence."""
        stream_url = await self._stream_url("test456")
        audio_chunks: list[bytes] = []
        silence = base64.b64encode(b"\xff" * 160).decode()

        async with websockets.connect(stream_url, close_timeout=2) as ws:
            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
                    }
                )
            )
            start_time = time.time()
            while time.time() - start_time < 20 and len(audio_chunks) < 50:
                try:
                    data = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.02))
                    if data.get("event") == "playAudio":
                        audio_chunks.append(base64.b64decode(data["media"]["payload"]))
                except asyncio.TimeoutError:
                    await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))
                except websockets.exceptions.ConnectionClosed:
                    break

        assert audio_chunks, "No audio chunks received"
        rms = rms_of_ulaw(b"".join(audio_chunks))
        assert rms > 500, f"Audio RMS {rms} too low - may be silence"


# =============================================================================
# OPENAI INTEGRATION TESTS
# =============================================================================


class TestOpenAIIntegration:
    """Integration tests for the OpenAI API."""

    @pytest.fixture
    def openai_configured(self):
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

    async def test_openai_chat_completion(self, openai_configured):
        """The configured key can reach the Chat Completions API."""
        import openai

        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello briefly."}],
            max_tokens=50,
        )
        assert response.choices[0].message.content


# =============================================================================
# PLIVO INTEGRATION TESTS
# =============================================================================


class TestPlivoIntegration:
    """Integration tests for the Plivo API (read-only)."""

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
