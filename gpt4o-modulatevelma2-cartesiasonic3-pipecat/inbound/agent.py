"""Inbound voice agent — Pipecat pipeline for incoming calls.

Uses Modulate Velma-2 STT, GPT-4o LLM with a Tavily web-search tool, and
Cartesia Sonic TTS with Pipecat framework orchestration and Plivo telephony.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMContextFrame
from pipecat.observers.loggers.llm_log_observer import LLMLogObserver
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import ModulateSTTService

if TYPE_CHECKING:
    from fastapi import WebSocket

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

# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)


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


async def search_the_web(params) -> None:
    """Answer from a live web search so the agent does not invent facts."""
    query = params.arguments.get("query", "")

    if not TAVILY_API_KEY:
        await params.result_callback(
            {"result": "Web search is not configured. Say you are not certain."}
        )
        return

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        # The SDK is synchronous; keep it off the loop carrying live audio.
        response = await asyncio.to_thread(
            client.search,
            query=query,
            search_depth=TAVILY_SEARCH_DEPTH,
            include_answer="advanced",
            max_results=5,
        )
    except Exception as e:
        logger.error(f"Tavily search failed: {e}")
        await params.result_callback(
            {"result": "The search failed. Tell the caller you could not look that up."}
        )
        return

    answer = (response.get("answer") or "").strip()
    sources = [r.get("title", "") for r in (response.get("results") or [])[:2]]
    logger.info(f"[Tavily] {query!r} -> {len(answer)} chars, {len(sources)} sources")

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
    system_prompt: str | None = None,
    initial_message: str = "Hello, I'm calling for help.",
    plivo_auth_id: str = "",
    plivo_auth_token: str = "",
) -> None:
    """Run a Pipecat voice agent pipeline for an incoming call."""
    prompt = system_prompt or SYSTEM_PROMPT
    logger.info(f"Starting Pipecat pipeline for call {call_id}")

    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=plivo_auth_id,
        auth_token=plivo_auth_token,
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

    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model=LLM_MODEL)
    llm.register_function("search_the_web", search_the_web)

    tts = CartesiaTTSService(
        api_key=CARTESIA_API_KEY,
        voice_id=TTS_VOICE,
        model=TTS_MODEL,
    )

    context = LLMContext(
        messages=[{"role": "system", "content": prompt}],
        tools=ToolsSchema(standard_tools=[SEARCH_SCHEMA]),
    )
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

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[
            TranscriptionLogObserver(),
            LLMLogObserver(),
            latency_observer,
        ],
    )

    initial_context = LLMContext(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": initial_message},
        ],
        tools=ToolsSchema(standard_tools=[SEARCH_SCHEMA]),
    )
    await task.queue_frames([LLMContextFrame(context=initial_context)])

    runner = PipelineRunner()

    try:
        await runner.run(task)
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
    finally:
        with contextlib.suppress(Exception):
            await task.cancel()
        logger.info(f"Pipeline ended for call {call_id}")
