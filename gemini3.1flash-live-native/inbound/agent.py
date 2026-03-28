"""Inbound voice agent — Gemini 3.1 Flash Live S2S engine.

Uses Gemini 3.1 Flash Live API for native speech-to-speech processing.
Audio flows bidirectionally: Plivo μ-law ↔ PCM ↔ Gemini Live API.

Loads the inbound system prompt and provides run_agent() for handling
inbound call WebSocket sessions.

Pipeline logging is controlled by the LOG_LEVEL env var:
  verbose — every pipeline event: per-packet stats, VAD frames, queue sizes
  normal  — key events: turn lifecycle, Gemini responses, audio timing (default)
  quiet   — errors and session start/end only
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import json
import os
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger

from utils import (
    GEMINI_INPUT_RATE,
    PLIVO_SAMPLE_RATE,
    SileroVADProcessor,
    gemini_to_plivo,
    resample_audio,
    ulaw_to_pcm,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# ---------------------------------------------------------------------------
# OTel tracing (optional — no-op when opentelemetry is not installed)
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace as _otel_trace

    _tracer = _otel_trace.get_tracer("voice-agent")
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    _tracer = None  # type: ignore[assignment]


def _traced(span_name: str):
    """Decorator that wraps an async method in an OTel span.

    Creates a span with call_id, records exceptions automatically,
    and ends the span on exit. No-op when opentelemetry is not installed.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if not _tracer:
                return await fn(self, *args, **kwargs)
            with _tracer.start_as_current_span(span_name, attributes={"call_id": self.call_id}):
                return await fn(self, *args, **kwargs)

        return wrapper

    return decorator


# Agent configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-live-preview")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Kore")
AUDIO_CHUNK_SIZE = 1024  # Bytes per chunk sent to Gemini

# Logging verbosity: "verbose", "normal" (default), "quiet"
LOG_LEVEL = os.getenv("LOG_LEVEL", "normal").lower()

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", SYSTEM_PROMPT)

# =============================================================================
# Tool Functions — replace these with your actual implementations
# =============================================================================


async def check_order_status(order_number: str | None, email: str | None) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    if not order_number and not email:
        return {"status": "error", "message": "Need order number or email"}

    statuses = [
        {
            "status": "shipped",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "shipping_carrier": "FedEx",
            "tracking_number": f"FX{random.randint(1000000000, 9999999999)}",
            "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d"),
            "items": "TechFlow Pro Annual Subscription",
        },
        {
            "status": "processing",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "message": "Order is being prepared and will ship within 24 hours",
            "items": "TechFlow Teams License (5 seats)",
        },
        {
            "status": "delivered",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "delivered_date": (datetime.now() - timedelta(days=1)).strftime("%B %d"),
            "signed_by": "Front Desk",
            "items": "TechFlow Enterprise Setup Kit",
        },
    ]
    return random.choice(statuses)


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS to customer. Replace with your actual implementation."""
    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "sent",
        "phone_number": phone_number,
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "confirmation_id": f"SMS{random.randint(100000, 999999)}",
    }


async def schedule_callback(
    phone_number: str, reason: str, preferred_time: str, department: str
) -> dict[str, Any]:
    """Schedule a callback. Replace with your actual implementation."""
    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "scheduled",
        "callback_id": f"CB{random.randint(100000, 999999)}",
        "phone_number": phone_number,
        "department": department,
        "scheduled_time": preferred_time or "within 2 business hours",
        "reason": reason,
    }


async def transfer_call(department: str, reason: str) -> dict[str, Any]:
    """Transfer call to human agent. Replace with your actual implementation."""
    return {
        "status": "transferring",
        "department": department,
        "reason": reason,
        "estimated_wait": "less than 2 minutes",
    }


# =============================================================================
# Gemini Voice Bot
# =============================================================================


class GeminiVoiceBot:
    """Manages a voice conversation session between Plivo and Gemini 3.1 Flash Live API.

    Uses Silero VAD (ONNX) for client-side voice activity detection and supports
    server-side VAD via Gemini's AutomaticActivityDetection for two-layer
    turn detection and barge-in interruption handling.

    Gemini 3.1 Flash Live API changes from 2.5:
    - Model: gemini-3.1-flash-live-preview
    - thinkingLevel replaces thinkingBudget (minimal/low/medium/high)
    - send_client_content restricted to initial context seeding only
    - Server events may contain multiple parts per event (audio + transcript)
    - turn_coverage defaults to TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO
    """

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = "Hello, I'm calling for help.",
        stream_id: str = "",
        parent_call_id: str = "",
        sip_headers: dict[str, str] | None = None,
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.parent_call_id = parent_call_id or call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.stream_id = stream_id
        self.sip_headers = sip_headers or {}

        self._running = False
        self._audio_buffer = bytearray()
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._tools = self._build_tools()

        # Silero VAD state
        self._vad = SileroVADProcessor(call_id=call_id)
        self._agent_speaking = False
        self._interruption_event = asyncio.Event()

        # Metrics
        self._turn_count = 0
        self._barge_in_count = 0
        self._error_count = 0
        self._session_start = time.monotonic()
        self._plivo_rx_bytes = 0
        self._plivo_tx_chunks = 0
        self._speech_end_time: float | None = None
        self._ttfs_samples: list[float] = []

    # — Structured logging with call ID, elapsed time, and pipeline stage —

    def _log(self, stage: str, msg: str) -> None:
        """Log at 'normal' level — key pipeline events."""
        if LOG_LEVEL == "quiet":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id, elapsed_s=elapsed, stage=stage).info(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _logv(self, stage: str, msg: str) -> None:
        """Log at 'verbose' level — detailed debugging info."""
        if LOG_LEVEL != "verbose":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id, elapsed_s=elapsed, stage=stage).debug(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _loge(self, stage: str, msg: str) -> None:
        """Log errors — always visible regardless of LOG_LEVEL."""
        self._error_count += 1
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id, elapsed_s=elapsed, stage=stage).error(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _emit_turn_complete(self, barge_in: bool = False) -> None:
        """Emit a structured turn_complete event with per-turn metrics."""
        self._turn_count += 1
        if barge_in:
            self._barge_in_count += 1

        logger.bind(
            event="turn_complete",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            barge_in=barge_in,
            plivo_rx_bytes=self._plivo_rx_bytes,
            plivo_tx_chunks=self._plivo_tx_chunks,
        ).info(
            f"[{self.call_id}] turn {self._turn_count} complete{' (barge-in)' if barge_in else ''}"
        )

    def _build_tools(self) -> list[types.Tool]:
        """Build Gemini function calling tools."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="check_order_status",
                        description="Look up the status of a customer's order.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "order_number": types.Schema(
                                    type=types.Type.STRING,
                                    description="Order number (usually starts with TF-)",
                                ),
                                "email": types.Schema(
                                    type=types.Type.STRING,
                                    description="Customer's email if order number unavailable",
                                ),
                            },
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="send_sms",
                        description="Send a text message to the customer's phone.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "phone_number": types.Schema(
                                    type=types.Type.STRING, description="Phone number"
                                ),
                                "message": types.Schema(
                                    type=types.Type.STRING, description="Message content"
                                ),
                            },
                            required=["phone_number", "message"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="schedule_callback",
                        description="Schedule a callback from a specialist.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "phone_number": types.Schema(
                                    type=types.Type.STRING, description="Phone number"
                                ),
                                "reason": types.Schema(
                                    type=types.Type.STRING,
                                    description="Why callback is needed",
                                ),
                                "preferred_time": types.Schema(
                                    type=types.Type.STRING, description="Preferred time"
                                ),
                                "department": types.Schema(
                                    type=types.Type.STRING, description="Department"
                                ),
                            },
                            required=["phone_number", "reason", "department"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="transfer_call",
                        description="Transfer call to human agent.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "department": types.Schema(
                                    type=types.Type.STRING, description="Department"
                                ),
                                "reason": types.Schema(
                                    type=types.Type.STRING, description="Transfer reason"
                                ),
                            },
                            required=["department", "reason"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="end_call",
                        description="End the call gracefully.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "reason": types.Schema(
                                    type=types.Type.STRING, description="Reason for ending"
                                ),
                                "resolution": types.Schema(
                                    type=types.Type.STRING,
                                    description="How issue was resolved",
                                ),
                            },
                        ),
                    ),
                ]
            )
        ]

    async def _handle_function_call(self, function_call: types.FunctionCall) -> str:
        """Route function calls to appropriate handlers."""
        name = function_call.name
        args = function_call.args or {}
        self._log("tool", f"calling {name}({args})")

        try:
            if name == "check_order_status":
                result = await check_order_status(
                    order_number=args.get("order_number"),
                    email=args.get("email"),
                )
            elif name == "send_sms":
                result = await send_sms(
                    phone_number=args.get("phone_number", ""),
                    message=args.get("message", ""),
                )
            elif name == "schedule_callback":
                result = await schedule_callback(
                    phone_number=args.get("phone_number", ""),
                    reason=args.get("reason", ""),
                    preferred_time=args.get("preferred_time", ""),
                    department=args.get("department", "general"),
                )
            elif name == "transfer_call":
                result = await transfer_call(
                    department=args.get("department", "support"),
                    reason=args.get("reason", "Customer requested transfer"),
                )
            elif name == "end_call":
                self._log("tool", f"end_call: {args.get('reason')}")
                self._running = False
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

            self._log("tool", f"{name} → {result.get('status', 'done')}")
            return json.dumps(result)

        except Exception as e:
            self._loge("tool", f"{name} ERROR: {e}")
            return json.dumps({"error": str(e)})

    def _build_session_config(self) -> types.LiveConnectConfig:
        """Build Gemini 3.1 Flash Live session configuration.

        Key differences from Gemini 2.5:
        - Uses thinkingLevel instead of thinkingBudget
        - Explicitly sets turn_coverage to TURN_INCLUDES_ONLY_ACTIVITY
          (3.1 defaults to TURN_INCLUDES_AUDIO_ACTIVITY_AND_ALL_VIDEO)
        """
        system_prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
            tools=self._tools,
            # Gemini 3.1: thinkingLevel replaces thinkingBudget — low for good balance
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            # Enable transcription for observability (user speech + model speech)
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=100,
                    silence_duration_ms=500,
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )

    def _trigger_interruption(self) -> None:
        """Handle barge-in: drain send queue, flush Plivo buffer, reset state."""
        self._log("barge-in", "Interruption triggered — clearing agent audio")
        self._agent_speaking = False

        # Drain queued audio
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Send clearAudio to Plivo immediately (don't wait for plivo_tx loop)
        if self.stream_id:
            clear_msg = json.dumps(
                {
                    "event": "clearAudio",
                    "stream_id": self.stream_id,
                }
            )
            with contextlib.suppress(Exception):
                asyncio.get_event_loop().create_task(self.websocket.send_text(clear_msg))

        self._interruption_event.set()

    @_traced("session")
    async def run(self) -> None:
        """Run the voice bot session."""
        self._log("session", f"Starting bot session for call {self.call_id}")
        self._running = True

        try:
            config = self._build_session_config()

            async with self._client.aio.live.connect(model=GEMINI_MODEL, config=config) as session:
                self._log("session", "Connected to Gemini 3.1 Flash Live API")

                # Gemini 3.1: send_realtime_input for initial text prompt
                await session.send_realtime_input(text=self.initial_message)

                await self._run_streaming_tasks(session)

        except Exception as e:
            self._loge("session", f"Bot session error: {e}")
        finally:
            self._running = False
            elapsed = round(time.monotonic() - self._session_start, 2)
            ttfs_avg = (
                round(sum(self._ttfs_samples) / len(self._ttfs_samples))
                if self._ttfs_samples
                else None
            )
            logger.bind(
                event="session_end",
                call_id=self.parent_call_id,
                duration_s=elapsed,
                turns=self._turn_count,
                barge_ins=self._barge_in_count,
                errors=self._error_count,
                ttfs_avg_ms=ttfs_avg,
            ).info(
                f"[{self.call_id}] Session ended: {elapsed}s, "
                f"{self._turn_count} turns, {self._barge_in_count} barge-ins, "
                f"{self._error_count} errors"
            )

    async def _run_streaming_tasks(self, session) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(session), name="plivo_rx"),
            asyncio.create_task(self._receive_from_gemini(session), name="gemini_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    self._loge("tasks", f"Task {task.get_name()} failed: {task.exception()}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    @_traced("plivo_rx")
    async def _receive_from_plivo(self, session) -> None:
        """Receive audio from Plivo WebSocket, run VAD, and forward to Gemini."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)
                        self._plivo_rx_bytes += len(mulaw_audio)

                        # Step 1: decode μ-law to 8kHz PCM
                        pcm_8k = ulaw_to_pcm(mulaw_audio)

                        # Step 2: run Silero VAD on 8kHz PCM (buffered)
                        speech_started, speech_ended = self._vad.process(pcm_8k)

                        if speech_started and self._agent_speaking:
                            self._log("vad", "Barge-in detected via VAD")
                            self._trigger_interruption()
                            self._emit_turn_complete(barge_in=True)

                        if speech_ended:
                            self._logv("vad", "User speech ended")
                            self._speech_end_time = time.monotonic()

                        # Step 3: resample to 16kHz for Gemini
                        pcm_16k = resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, GEMINI_INPUT_RATE)

                        self._audio_buffer.extend(pcm_16k)
                        if len(self._audio_buffer) >= AUDIO_CHUNK_SIZE:
                            chunk = bytes(self._audio_buffer[:AUDIO_CHUNK_SIZE])
                            self._audio_buffer = self._audio_buffer[AUDIO_CHUNK_SIZE:]
                            await session.send_realtime_input(
                                audio=types.Blob(data=chunk, mime_type="audio/pcm")
                            )

                elif event == "text":  # For testing - inject text as user input
                    text = message.get("text", "")
                    if text:
                        self._log("plivo_rx", f"Injecting text: {text[:50]}...")
                        try:
                            await session.send_realtime_input(text=text)
                        except Exception as e:
                            self._loge("plivo_rx", f"Failed to send text: {e}")

                elif event == "stop":
                    self._log("plivo_rx", "Plivo stop event received")
                    break

        except Exception as e:
            if "1000" not in str(e):
                self._loge("plivo_rx", f"ERROR: {e}")

    @_traced("gemini_rx")
    async def _receive_from_gemini(self, session) -> None:
        """Receive audio from Gemini and queue for Plivo.

        Gemini 3.1 change: a single server event can contain multiple content
        parts simultaneously (audio chunks + transcript). Process all parts
        in each event to avoid missing content.
        """
        try:
            while self._running:
                try:
                    async for response in session.receive():
                        if not self._running:
                            return

                        if response.server_content:
                            # Handle server-side interruption confirmation
                            if response.server_content.interrupted:
                                self._log("gemini_rx", "Gemini confirmed interruption")
                                self._trigger_interruption()

                            model_turn = response.server_content.model_turn
                            if model_turn and model_turn.parts:
                                # Gemini 3.1: process ALL parts — may include
                                # audio + transcript in a single event
                                for part in model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        # Mark agent as speaking from first audio chunk
                                        # until turn_complete (not when plivo_tx drains)
                                        self._agent_speaking = True

                                        plivo_audio = gemini_to_plivo(part.inline_data.data)
                                        await self._send_queue.put(plivo_audio)

                                        # Track TTFS (time-to-first-speech)
                                        if self._speech_end_time is not None:
                                            ttfs = (time.monotonic() - self._speech_end_time) * 1000
                                            self._ttfs_samples.append(ttfs)
                                            self._logv("metrics", f"TTFS: {ttfs:.0f}ms")
                                            self._speech_end_time = None

                                    if part.text:
                                        self._logv(
                                            "gemini_rx",
                                            f"Transcript: {part.text[:80]}",
                                        )

                                        # Emit agent_text in turn_complete events
                                        logger.bind(
                                            event="agent_text",
                                            call_id=self.parent_call_id,
                                            text=part.text[:200],
                                        ).debug(f"[{self.call_id}] agent: {part.text[:80]}")

                            # Gemini 3.1: transcription events from audio
                            if hasattr(response.server_content, "output_transcription"):
                                ot = response.server_content.output_transcription
                                if ot and hasattr(ot, "text") and ot.text:
                                    self._logv(
                                        "gemini_rx",
                                        f"Agent transcript: {ot.text[:80]}",
                                    )
                                    logger.bind(
                                        event="agent_text",
                                        call_id=self.parent_call_id,
                                        text=ot.text[:200],
                                    ).debug(f"[{self.call_id}] agent: {ot.text[:80]}")

                            if hasattr(response.server_content, "input_transcription"):
                                it = response.server_content.input_transcription
                                if it and hasattr(it, "text") and it.text:
                                    self._logv(
                                        "gemini_rx",
                                        f"User transcript: {it.text[:80]}",
                                    )
                                    logger.bind(
                                        event="user_text",
                                        call_id=self.parent_call_id,
                                        text=it.text[:200],
                                    ).debug(f"[{self.call_id}] user: {it.text[:80]}")

                            if response.server_content.turn_complete:
                                self._agent_speaking = False
                                self._log("gemini_rx", "Turn complete")
                                self._emit_turn_complete()

                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                result = await self._handle_function_call(fc)
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={"result": result},
                                        )
                                    ]
                                )

                except Exception as e:
                    if "cancelled" in str(e).lower():
                        raise
                    self._loge("gemini_rx", f"ERROR: {e}")
                    raise

        except asyncio.CancelledError:
            pass

    @_traced("plivo_tx")
    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160  # 20ms at 8kHz μ-law
        audio_buffer = bytearray()

        try:
            while self._running:
                # Check for interruption at loop start
                if self._interruption_event.is_set():
                    audio_buffer.clear()
                    self._interruption_event.clear()
                    # clearAudio already sent in _trigger_interruption()
                    self._logv("plivo_tx", "Send buffer cleared after interruption")
                    continue

                try:
                    audio = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                    audio_buffer.extend(audio)

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
                        # Check interruption between chunks
                        if self._interruption_event.is_set():
                            audio_buffer.clear()
                            break

                        chunk = bytes(audio_buffer[:PLIVO_CHUNK_SIZE])
                        audio_buffer = audio_buffer[PLIVO_CHUNK_SIZE:]

                        message = {
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": base64.b64encode(chunk).decode("utf-8"),
                            },
                        }
                        await self.websocket.send_text(json.dumps(message))
                        self._plivo_tx_chunks += 1

                except TimeoutError:
                    continue

        except asyncio.CancelledError:
            pass


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    from_number: str = "",
    to_number: str = "",
    system_prompt: str | None = None,
    initial_message: str = "Hello, I'm calling for help.",
    stream_id: str = "",
    parent_call_id: str = "",
    sip_headers: dict[str, str] | None = None,
) -> None:
    """Run a voice agent session for an incoming call."""
    agent = GeminiVoiceBot(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
        stream_id=stream_id,
        parent_call_id=parent_call_id,
        sip_headers=sip_headers,
    )
    await agent.run()
