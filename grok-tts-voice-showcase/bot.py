"""
xAI Grok TTS Voice Showcase
=============================
Pipecat voice agent using xAI TTS over Plivo with per-voice scenario prompts.

Run:
    XAI_VOICE=Ara SCENARIO=storytelling python bot.py -t plivo -x <ngrok-host> --port 8000
"""

import asyncio
import os
import uuid
from enum import Enum

import httpx
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)
from pipecat.turns.user_stop import (
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from metrics_observer import MetricsCollectorObserver
from services.xai_tts import XaiTTSService

load_dotenv(override=False)


# ---------------------------------------------------------------------------
# Voice scenarios — one per xAI voice to showcase different strengths
# ---------------------------------------------------------------------------

SCENARIOS = {
    "storytelling": {
        "voice": "Ara",
        "label": "Dramatic Storytelling",
        "system_prompt": (
            "You are a captivating storyteller on a phone call. When asked, narrate "
            "scenes with vivid emotion, pacing, and drama. Build tension, use pauses. "
            "Keep responses to 3-4 sentences. Your output will be spoken aloud, so "
            "avoid special characters, markdown, or lists."
        ),
        "greeting": "Hey there! I'm Ara, your storyteller. Give me a topic and I'll spin you a tale.",
    },
    "counselor": {
        "voice": "Eve",
        "label": "Empathetic Counselor",
        "system_prompt": (
            "You are a warm, empathetic listener on a phone call. Respond with genuine "
            "care and understanding. Mirror the caller's emotions. Keep responses to "
            "2-3 sentences. Your output will be spoken aloud, so avoid special "
            "characters, markdown, or lists."
        ),
        "greeting": "Hi, I'm Eve. I'm here to listen — what's on your mind?",
    },
    "travel": {
        "voice": "Rex",
        "label": "Excited Travel Guide",
        "system_prompt": (
            "You are an enthusiastic travel guide on a phone call who gets genuinely "
            "excited about destinations. Share vivid descriptions with infectious energy. "
            "Keep responses to 2-3 sentences. Your output will be spoken aloud, so "
            "avoid special characters, markdown, or lists."
        ),
        "greeting": "Hey! I'm Rex, your travel buddy. Where in the world are you dreaming about?",
    },
    "casual": {
        "voice": "Sal",
        "label": "Casual Friend Chat",
        "system_prompt": (
            "You are a witty, laid-back friend having a casual phone conversation. "
            "Be natural, use filler words occasionally, react genuinely. Keep responses "
            "to 2-3 sentences. Your output will be spoken aloud, so avoid special "
            "characters, markdown, or lists."
        ),
        "greeting": "Yo! Sal here. What's up, what's going on?",
    },
    "news": {
        "voice": "Leo",
        "label": "News Anchor",
        "system_prompt": (
            "You are a professional news anchor delivering stories with authority and "
            "clarity on a phone call. Vary your pacing — urgent for breaking news, "
            "measured for analysis. Keep responses to 2-3 sentences. Your output will "
            "be spoken aloud, so avoid special characters, markdown, or lists."
        ),
        "greeting": "Good evening. I'm Leo. What story would you like me to cover?",
    },
    "multilingual": {
        "voice": "Ara",
        "label": "Multilingual Auto-Detect",
        "system_prompt": (
            "You are a friendly multilingual assistant on a phone call. You speak "
            "every language fluently. When the caller asks you to speak in a specific "
            "language, you MUST switch to that language completely for your response. "
            "Keep responses to 2-3 sentences. Your output will be spoken aloud by a "
            "TTS engine with auto language detection, so avoid special characters, "
            "markdown, or lists. Write non-English responses in their native script."
        ),
        "greeting": "Hey! I speak over 20 languages. Try me — ask me to speak in any language!",
    },
    "expressive": {
        "voice": "Ara",
        "label": "Expressive Speech Tags",
        "system_prompt": (
            "You are a dramatically expressive storyteller on a phone call. "
            "You MUST use xAI speech tags in EVERY response to make your delivery vivid. "
            "Available tags:\n"
            "- Inline: [laugh], [sigh], [gasp], [pause], [clears throat]\n"
            "- Wrapping: <whisper>text</whisper>, <shout>text</shout>, <sing>text</sing>\n"
            "Use these naturally throughout your responses. For example: "
            "'[gasp] You won't believe this! [pause] <whisper>It was hiding in plain sight the whole time.</whisper> [laugh]'\n"
            "Keep responses to 2-4 sentences. Be theatrical and fun."
        ),
        "greeting": "[clears throat] Well hello there! [pause] <whisper>I've been waiting for someone to talk to.</whisper> [laugh] What shall we chat about?",
    },
}

# Default fallback prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are a personal AI assistant on a phone call — like Siri or Alexa, "
    "but over the phone. The caller will ask you questions about weather, "
    "reminders, conversions, general knowledge, etc. "
    "Keep your responses to 1-2 short sentences. "
    "Your output will be spoken aloud, so avoid special characters, "
    "markdown, or lists. Be conversational and natural."
)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


async def run_bot(transport: BaseTransport, handle_sigint: bool, call_id=None):
    voice = os.getenv("XAI_VOICE", "Ara")
    scenario_key = os.getenv("SCENARIO", "")

    # Resolve scenario
    scenario = SCENARIOS.get(scenario_key)
    if scenario:
        system_prompt = scenario["system_prompt"]
        scenario_label = scenario["label"]
        greeting = scenario["greeting"]
        logger.info(f"Voice: {voice} | Scenario: {scenario_label}")
    else:
        system_prompt = DEFAULT_SYSTEM_PROMPT
        scenario_label = "General Assistant"
        greeting = "Hey! What can I help you with?"
        logger.info(f"Voice: {voice} | Scenario: default")

    # Metrics collection
    session_id = str(uuid.uuid4())
    data_dir = os.path.join(os.path.dirname(__file__), "data", "sessions")
    metrics_observer = MetricsCollectorObserver(
        session_id=session_id,
        mode=voice.lower(),
        config={
            "tts_provider": "xai",
            "voice": voice,
            "scenario": scenario_key or "default",
            "scenario_label": scenario_label,
        },
        data_dir=data_dir,
    )

    # Use multi-language STT when scenario requires it
    stt_language = scenario.get("language") if scenario else None
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-3",
        language=stt_language,
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.0-flash",
    )

    tts = XaiTTSService(
        api_key=os.getenv("XAI_TTS_API_KEY", ""),
        voice_id=voice,
        sample_rate=24000,
    )

    # Smart Turn v3
    stop_secs = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
    turn_strategies = UserTurnStrategies(
        stop=[
            TurnAnalyzerUserTurnStopStrategy(
                turn_analyzer=LocalSmartTurnAnalyzerV3(
                    params=SmartTurnParams(stop_secs=stop_secs),
                ),
            )
        ],
    )

    messages = [{"role": "system", "content": system_prompt}]
    context = LLMContext(messages)

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=turn_strategies,
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(start_secs=0.4, min_volume=0.8),
            ),
        ),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
            observers=[metrics_observer],
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        if call_id:
            asyncio.create_task(_start_recording(call_id))
        logger.info("Client connected — sending greeting")
        await asyncio.sleep(2)
        await task.queue_frames([TTSSpeakFrame(text=greeting)])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        if call_id:
            asyncio.create_task(_fetch_recording(call_id, session_id, data_dir))
        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)
    await runner.run(task)


async def _start_recording(call_id: str):
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return
    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"https://api.plivo.com/v1/Account/{auth_id}/Call/{call_id}/Record/",
                auth=(auth_id, auth_token),
                json={"time_limit": 3600, "file_format": "mp3"},
            )
            logger.info(f"Plivo recording started: {resp.status_code}")
    except Exception as e:
        logger.warning(f"Failed to start Plivo recording: {e}")


async def _fetch_recording(call_id: str, session_id: str, data_dir: str):
    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        return

    import json

    for attempt in range(12):
        await asyncio.sleep(5)
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.get(
                    f"https://api.plivo.com/v1/Account/{auth_id}/Recording/",
                    auth=(auth_id, auth_token),
                    params={"call_uuid": call_id, "limit": 1},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    objects = data.get("objects", [])
                    if objects:
                        recording_url = objects[0].get("recording_url")
                        if recording_url:
                            logger.info(f"Plivo recording ready: {recording_url}")
                            session_path = os.path.join(data_dir, f"{session_id}.json")
                            if os.path.exists(session_path):
                                with open(session_path) as f:
                                    session = json.load(f)
                                session["recording_url"] = recording_url
                                with open(session_path, "w") as f:
                                    json.dump(session, f, indent=2)
                            return
        except Exception as e:
            logger.warning(f"Recording fetch attempt {attempt + 1}: {e}")

    logger.warning(f"Could not fetch recording for call {call_id} after 60s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def bot(runner_args: RunnerArguments):
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    serializer = PlivoFrameSerializer(
        stream_id=call_data["stream_id"],
        call_id=call_data["call_id"],
        auth_id=os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    await run_bot(transport, runner_args.handle_sigint, call_id=call_data["call_id"])


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
