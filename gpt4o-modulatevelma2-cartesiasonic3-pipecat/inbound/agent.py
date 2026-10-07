"""Inbound voice agent: Pipecat pipeline for incoming calls.

Modulate Velma-2 STT, GPT-4o LLM with a Tavily web-search tool, and Cartesia
Sonic TTS, orchestrated by Pipecat over Plivo telephony. The Modulate STT
service is defined in this file (and duplicated in outbound/agent.py, since each
direction is self-contained).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlencode

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMContextFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.utils.time import time_now_iso8601
from pipecat.workers.runner import WorkerRunner
from websockets.asyncio.client import connect as websocket_connect
from websockets.exceptions import ConnectionClosed
from websockets.protocol import State

if TYPE_CHECKING:
    from fastapi import WebSocket
    from pipecat.services.llm_service import FunctionCallParams

load_dotenv()

# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODULATE_API_KEY = os.getenv("MODULATE_API_KEY", "")
CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
TTS_MODEL = os.getenv("TTS_MODEL", "sonic-3.6")
TTS_VOICE = os.getenv("TTS_VOICE", "71a7ad14-091c-4e8e-a314-022ece01c121")

# Tavily is slow enough at its default depth that the caller hears silence.
# Measured 2026-10-05: search_depth "advanced" 5.46s median, "fast" 1.27s.
TAVILY_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "fast")
# Hard ceiling on one search. Past this the tool tells the LLM it could not look
# the answer up, rather than leaving the caller in silence.
TAVILY_TIMEOUT_SECS = 5.0

# =============================================================================
# Prompts
# =============================================================================

# Loaded only from the sibling file: no env override (edit or mount the file).
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()

# Inbound has no scripted greeting: this stand-in caller turn makes the LLM open
# the call. It is not added to the conversation context.
OPENING_USER_MESSAGE = "Hello, I'm calling for help."

# =============================================================================
# Modulate Velma-2 STT
# =============================================================================

MODULATE_STT_URL = "wss://platform.modulate.ai/api/velma-2-streaming"
MODULATE_STT_MODEL = "velma-2"
MODULATE_AUTH_CLOSE_CODE = 4001  # the server closes with this code on a bad API key
MODULATE_CLOSE_TIMEOUT_SECS = 2.0  # cap on the closing handshake at call teardown


class ModulateSTTService(WebsocketSTTService):
    """Modulate Velma-2 streaming as a Pipecat STT service.

    Velma returns more than a transcript on a single socket: each finalised clip
    carries an optional emotion, accent and synthetic-voice score. The transcript
    drives the pipeline; the extra signals are exposed on the TranscriptionFrame's
    ``result`` for anything downstream that wants them.

    Protocol (https://docs.modulate.ai/api-reference/velma/streaming):
      - API key goes in the query string, not a header. A bad key closes with 4001.
      - Exactly one configuration text frame must precede any audio.
      - Audio is raw binary; we send s16le at the pipeline's own sample rate.
      - ``partial_clip`` -> interim transcript, ``clip`` -> final.

    Connection ownership: the socket is opened in ``start()`` and from then on
    only the base class's receive task reconnects it (bounded retries with
    backoff, then a terminal ErrorFrame). ``run_stt`` never connects; it drops
    audio while the socket is down, so there is exactly one connector.

    Note: ``behavior_detection`` events are documented for this endpoint but were
    not observed on the streaming socket during testing (2026-10-05). Behaviours
    come back reliably from the batch endpoint. The handler is kept because it is
    correct per the docs and costs nothing.
    """

    def __init__(
        self,
        *,
        api_key: str,
        behaviors: list[str] | None = None,
        produce_topics: bool = False,
        produce_summary: bool = False,
        sample_rate: int | None = None,
        **kwargs,
    ) -> None:
        # Velma-2 auto-detects language and exposes no model selector, so
        # language is explicitly None rather than left NOT_GIVEN; the base
        # class asserts every settings field has a real value.
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=MODULATE_STT_MODEL, language=None),
            **kwargs,
        )
        self._api_key = api_key
        self._config = json.dumps(
            {
                "behaviors": behaviors or [],
                "produce_topics": produce_topics,
                "produce_summary": produce_summary,
            }
        )
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    # -- lifecycle ------------------------------------------------------------

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self._connect()

    # No stop()/cancel() overrides: WebsocketSTTService calls _disconnect() from
    # its own stop(), cancel() and cleanup() on Pipecat 1.x.

    async def _connect(self) -> None:
        await super()._connect()
        try:
            await self._connect_websocket()
        except Exception as e:
            # Surface the failure instead of swallowing it. The receive task
            # below then finds no socket and runs the base reconnect loop.
            await self.push_error(error_msg=str(e), exception=e)
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        """Open the socket and send the configuration frame. Raises on failure."""
        if self._websocket and self._websocket.state is State.OPEN:
            return
        query = urlencode(
            {
                "api_key": self._api_key,
                "audio_format": "s16le",
                "sample_rate": self.sample_rate,
                "num_channels": 1,
            }
        )
        try:
            ws = await websocket_connect(
                f"{MODULATE_STT_URL}?{query}",
                max_size=None,
                close_timeout=MODULATE_CLOSE_TIMEOUT_SECS,
            )
            try:
                await ws.send(self._config)
            except Exception:
                with contextlib.suppress(Exception):
                    await ws.close()
                raise
        except Exception as e:
            # The URL carries the API key and websockets puts the URL in some
            # error messages, so neither the URL nor the exception text is ever
            # logged or re-raised: only the exception type and HTTP status.
            status = getattr(getattr(e, "response", None), "status_code", None)
            detail = type(e).__name__ + (f", HTTP {status}" if status else "")
            raise ConnectionError(f"Modulate Velma-2 connect failed ({detail})") from None
        # Published only after the configuration frame is sent, so run_stt can
        # never put audio ahead of it.
        self._websocket = ws
        logger.info(f"Modulate Velma-2 connected at {self.sample_rate}Hz")

    async def _disconnect_websocket(self) -> None:
        ws = self._websocket
        self._websocket = None
        if ws is None:
            return
        try:
            if ws.state is State.OPEN:
                await ws.send("")  # end-of-stream signal
            await ws.close()
        except Exception as e:
            logger.debug(f"Modulate close: {type(e).__name__}")

    # -- audio in / transcripts out -------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Forward caller audio. Transcripts arrive on the receive task."""
        ws = self._websocket
        if ws is None or ws.state is not State.OPEN:
            # Down or mid-reconnect: drop the frame. Reconnecting belongs to the
            # receive task alone; connecting here would race it.
            yield None
            return
        try:
            await ws.send(audio)
        except Exception as e:
            # The receive task sees the same failure and reconnects.
            logger.debug(f"Modulate send failed: {type(e).__name__}")
        yield None

    async def _receive_messages(self) -> None:
        ws = self._websocket
        if ws is None:
            raise ConnectionError("Modulate Velma-2 is not connected")
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    logger.warning(f"Non-JSON message from Modulate ({len(message)} chars)")
                    continue
                await self._handle_event(event)
        except ConnectionClosed as e:
            if e.rcvd is not None and e.rcvd.code == MODULATE_AUTH_CLOSE_CODE:
                # Retrying cannot fix a rejected key; let the base class report
                # the close as an error instead of reconnecting.
                logger.error("Modulate rejected MODULATE_API_KEY (close 4001); not reconnecting")
                self._reconnect_on_error = False
            raise

    @staticmethod
    def _language(code: str | None) -> Language | None:
        if not code:
            return None
        try:
            return Language(code)
        except (ValueError, KeyError):
            return None

    async def _handle_event(self, event: dict) -> None:
        kind = event.get("type")

        if kind == "partial_clip":
            text = (event.get("partial_clip") or {}).get("text", "")
            if text:
                await self.push_frame(
                    InterimTranscriptionFrame(text, self._user_id, time_now_iso8601())
                )

        elif kind == "clip":
            clip = event.get("clip") or {}
            text = clip.get("text", "")
            if not text:
                return
            # Emotion, accent, deepfake score and speaker label are personal
            # data about the caller: DEBUG only, and never the transcript text.
            extras = {
                k: clip.get(k)
                for k in ("emotion", "accent", "deepfake_score", "speaker_label")
                if clip.get(k) is not None
            }
            logger.debug(f"[Velma] clip, {len(text)} chars, {extras}")
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._language(clip.get("language")),
                    result=clip,
                )
            )

        elif kind == "clip_update":
            # A refinement of a clip already transcribed. Log the better emotion
            # and accent, but do not push another frame or the LLM sees the same
            # words twice.
            clip = event.get("clip_update") or {}
            logger.debug(f"[Velma] refined: {clip.get('emotion')} {clip.get('accent')}")

        elif kind == "behavior_detection":
            detection = event.get("detection") or {}
            if detection.get("detected"):
                logger.debug(
                    f"[Velma] behaviour {detection.get('behavior_name')} "
                    f"@ {detection.get('confidence')}"
                )

        elif kind == "error":
            logger.error(f"Modulate error: {event.get('error')}")


# =============================================================================
# Tools
# =============================================================================

SEARCH_SCHEMA = FunctionSchema(
    name="search_the_web",
    description=(
        "Search the live web for current facts, pricing, product details, or "
        "anything you are not certain of. Use this instead of guessing."
    ),
    properties={
        "query": {
            "type": "string",
            "description": "A focused search query, not the caller's whole sentence.",
        }
    },
    required=["query"],
)


async def search_the_web(params: FunctionCallParams) -> None:
    """Answer from a live web search so the agent does not invent facts.

    Every path hands the LLM a short sentence it can speak; nothing here raises
    and nothing waits longer than TAVILY_TIMEOUT_SECS.
    """
    query = str(params.arguments.get("query") or "").strip()

    if not TAVILY_API_KEY:
        await params.result_callback(
            {"result": "Web search is not configured. Say you are not certain."}
        )
        return
    if not query:
        await params.result_callback({"result": "No search query was given. Ask what to look up."})
        return

    try:
        from tavily import AsyncTavilyClient

        client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
        # The SDK's own timeout bounds the HTTP request; wait_for bounds the
        # whole call so a caller never sits in silence past the limit.
        response = await asyncio.wait_for(
            client.search(
                query=query,
                search_depth=TAVILY_SEARCH_DEPTH,
                include_answer="advanced",
                max_results=5,
                timeout=TAVILY_TIMEOUT_SECS,
            ),
            timeout=TAVILY_TIMEOUT_SECS,
        )
    except asyncio.TimeoutError:
        logger.warning(f"Tavily search timed out after {TAVILY_TIMEOUT_SECS}s")
        await params.result_callback(
            {"result": "The search took too long. Tell the caller you could not look that up."}
        )
        return
    except Exception as e:
        logger.error(f"Tavily search failed: {type(e).__name__}")
        await params.result_callback(
            {"result": "The search failed. Tell the caller you could not look that up."}
        )
        return

    answer = (response.get("answer") or "").strip()
    sources = [r.get("title", "") for r in (response.get("results") or [])[:2]]
    # The query is the caller's words; log sizes only.
    logger.debug(f"[Tavily] {len(answer)} chars, {len(sources)} sources")

    if not answer:
        await params.result_callback({"result": "No reliable sources found for that."})
        return

    named = ", ".join(s for s in sources if s)
    await params.result_callback({"result": answer + (f"\nSources: {named}." if named else "")})


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    stream_id: str,
    from_number: str = "",
    to_number: str = "",
) -> None:
    """Run a Pipecat voice agent pipeline for an incoming call."""
    logger.info(f"Starting Pipecat pipeline for call {call_id}")

    # PlivoFrameSerializer wraps outgoing audio in Plivo playAudio events
    # (contentType audio/x-mulaw, 8kHz) and turns interruptions into clearAudio.
    # auto_hang_up needs Plivo REST credentials, which belong to server.py; the
    # pipeline only ends when Plivo closes the stream, so there is no call left
    # to hang up.
    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        params=PlivoFrameSerializer.InputParams(auto_hang_up=False),
    )

    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    stt = ModulateSTTService(api_key=MODULATE_API_KEY)

    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        settings=OpenAILLMService.Settings(model=LLM_MODEL),
    )
    # Backstop behind the handler's own timeout, in case the handler itself hangs.
    llm.register_function("search_the_web", search_the_web, timeout_secs=TAVILY_TIMEOUT_SECS + 2.0)

    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        settings=CartesiaTTSService.Settings(voice=TTS_VOICE, model=TTS_MODEL),
    )

    tools = ToolsSchema(standard_tools=[SEARCH_SCHEMA])
    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=tools,
    )

    # VAD is Pipecat's Silero analyzer on the user aggregator; it drives turn
    # taking and barge-in (interruptions are on by default).
    context_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            context_aggregator.assistant(),
            transport.output(),
        ]
    )

    latency_observer = UserBotLatencyObserver()

    @latency_observer.event_handler("on_latency_measured")
    async def _on_latency(observer, latency: float):
        logger.info(f"[Latency] user stopped -> bot started: {latency:.2f}s")

    # The two log observers print transcripts and LLM text at DEBUG only.
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            TranscriptionLogObserver(),
            LLMLogObserver(),
            latency_observer,
        ],
    )

    opening_context = LLMContext(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": OPENING_USER_MESSAGE},
        ],
        tools=tools,
    )
    await worker.queue_frames([LLMContextFrame(context=opening_context)])

    # WorkerRunner's default is handle_sigterm=False, which is what running
    # inside uvicorn needs: uvicorn keeps its own SIGTERM handler.
    runner = WorkerRunner()

    try:
        await runner.add_workers(worker)
        await runner.run()
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        with contextlib.suppress(Exception):
            await worker.cancel()
        logger.info(f"Pipeline ended for call {call_id}")
