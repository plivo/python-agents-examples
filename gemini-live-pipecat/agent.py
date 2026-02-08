"""
Voice agent using Pipecat with Google Gemini Live API for speech-to-speech conversations.

This module provides the pipeline logic for the voice agent that:
- Uses Pipecat framework for orchestration
- Connects to Google's Gemini Live API for real-time speech processing
- Handles bidirectional audio streaming with Plivo telephony

Usage:
    from agent import run_agent

    # In your WebSocket handler:
    await run_agent(websocket, call_id, stream_id, auth_id, auth_token)
"""

from __future__ import annotations

import asyncio
import os

from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Puck")

# System prompt for the voice agent
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    """You are a helpful voice assistant. Keep your responses concise and natural
for voice conversation. Be friendly and professional. Ask clarifying questions
when needed. Your responses will be spoken aloud, so avoid special characters
and speak naturally.""",
)


async def run_agent(
    websocket,
    call_id: str,
    stream_id: str,
    auth_id: str = "",
    auth_token: str = "",
) -> PipelineTask:
    """Run the Pipecat pipeline for a voice agent session.

    Args:
        websocket: FastAPI WebSocket connection from Plivo
        call_id: Unique identifier for this call
        stream_id: Plivo stream identifier
        auth_id: Plivo auth ID for serializer
        auth_token: Plivo auth token for serializer

    Returns:
        The PipelineTask instance
    """
    # Create Plivo serializer with authentication
    serializer = PlivoFrameSerializer(
        stream_id=stream_id,
        call_id=call_id,
        auth_id=auth_id or os.getenv("PLIVO_AUTH_ID", ""),
        auth_token=auth_token or os.getenv("PLIVO_AUTH_TOKEN", ""),
    )

    # Create transport with Plivo serializer
    transport = FastAPIWebsocketTransport(
        websocket=websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            serializer=serializer,
        ),
    )

    # Initialize Gemini Live service for speech-to-speech
    llm = GeminiLiveLLMService(
        api_key=GEMINI_API_KEY,
        model=GEMINI_MODEL,
        voice_id=GEMINI_VOICE,
        system_instruction=SYSTEM_PROMPT,
    )

    # Create the pipeline
    # For Gemini Multimodal Live, audio flows directly through the LLM service
    # which handles both speech recognition and speech synthesis natively
    pipeline = Pipeline(
        [
            transport.input(),  # Receive audio from Plivo
            llm,  # Gemini Multimodal Live (speech-to-speech)
            transport.output(),  # Send audio to Plivo
        ]
    )

    logger.info("Creating pipeline task...")

    # Create and run the pipeline task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    # Queue greeting after Gemini session is established.
    # The greeting frame must be sent after the pipeline starts and
    # the Gemini Live session is connected, otherwise it gets dropped.
    async def send_greeting():
        await asyncio.sleep(1.5)  # Wait for Gemini session to connect
        logger.info("Sending greeting message to Gemini...")
        await task.queue_frames(
            [LLMMessagesAppendFrame([{"role": "user", "content": "Hello!"}], run_llm=True)]
        )

    asyncio.create_task(send_greeting())

    logger.info("Running pipeline...")

    # Run the pipeline
    runner = PipelineRunner()
    await runner.run(task)

    logger.info("Pipeline completed")
    return task
