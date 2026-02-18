"""Inbound voice agent — Pipecat pipeline for incoming calls.

Uses Deepgram STT, GPT-4o-mini LLM, and GPT-4o-mini-TTS with Pipecat
framework orchestration and Plivo telephony.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from loguru import logger
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
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")

# =============================================================================
# Prompts
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)

TTS_INSTRUCTIONS = (Path(__file__).parent / "tts_instructions.md").read_text().strip()


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

    stt = DeepgramSTTService(
        api_key=DEEPGRAM_API_KEY,
    )

    llm = OpenAILLMService(
        api_key=OPENAI_API_KEY,
        model=LLM_MODEL,
    )

    tts = OpenAITTSService(
        api_key=OPENAI_API_KEY,
        voice=TTS_VOICE,
        model=TTS_MODEL,
        params=OpenAITTSService.InputParams(
            instructions=TTS_INSTRUCTIONS,
        ),
    )

    context = LLMContext(
        messages=[{"role": "system", "content": prompt}],
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
