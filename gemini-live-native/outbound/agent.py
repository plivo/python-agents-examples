"""Outbound voice agent — GeminiVoiceBot engine + call state management.

Loads the outbound system prompt and provides run_agent() for handling
outbound call WebSocket sessions, plus CallManager for tracking call lifecycle.

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import random
import threading
import uuid
from dataclasses import dataclass, field
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

# Agent configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Kore")
AUDIO_CHUNK_SIZE = 1024  # Bytes per chunk sent to Gemini

# =============================================================================
# System Prompt
# =============================================================================

_OUTBOUND_PROMPT_TEMPLATE = (Path(__file__).parent / "system_prompt.md").read_text().strip()


def build_outbound_prompt(
    opening_reason: str = "",
    objective: str = "",
    context: str = "",
) -> str:
    """Build a concrete outbound system prompt by substituting template variables."""
    prompt = _OUTBOUND_PROMPT_TEMPLATE
    prompt = prompt.replace("{{opening_reason}}", opening_reason)
    prompt = prompt.replace("{{objective}}", objective)
    prompt = prompt.replace("{{context}}", context)
    return prompt


# Default system prompt (no template substitution)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _OUTBOUND_PROMPT_TEMPLATE)

# =============================================================================
# Outbound Call Records
# =============================================================================


@dataclass
class OutboundCallRecord:
    """Tracks the state of a single outbound call."""

    call_id: str
    phone_number: str
    status: str = "initiating"  # initiating|ringing|connected|completed|failed|no_answer
    campaign_id: str = ""
    context: str = ""
    system_prompt: str = ""
    initial_message: str = ""
    opening_reason: str = ""
    objective: str = ""
    plivo_request_uuid: str = ""
    plivo_call_uuid: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration: int = 0
    hangup_cause: str = ""
    outcome: str = ""  # success|no_answer|busy|failed


def determine_outcome(hangup_cause: str, duration: int) -> str:
    """Map Plivo hangup cause and duration to a high-level outcome."""
    cause = hangup_cause.upper() if hangup_cause else ""

    if cause in ("NO_ANSWER", "ORIGINATOR_CANCEL"):
        return "no_answer"
    if cause in ("USER_BUSY", "CALL_REJECTED"):
        return "busy"
    if cause in (
        "UNALLOCATED_NUMBER",
        "INVALID_NUMBER_FORMAT",
        "NO_ROUTE_DESTINATION",
        "NETWORK_OUT_OF_ORDER",
        "SERVICE_UNAVAILABLE",
        "RECOVERY_ON_TIMER_EXPIRE",
        "BEARERCAPABILITY_NOTAVAIL",
    ):
        return "failed"

    if duration > 0 or cause in ("NORMAL_CLEARING", ""):
        return "success"

    return "failed"


class CallManager:
    """Thread-safe manager for outbound call records."""

    def __init__(self) -> None:
        self._calls: dict[str, OutboundCallRecord] = {}
        self._lock = threading.Lock()

    def create_call(
        self,
        phone_number: str,
        campaign_id: str = "",
        opening_reason: str = "",
        objective: str = "",
        context: str = "",
    ) -> OutboundCallRecord:
        """Create and register a new outbound call record."""
        call_id = str(uuid.uuid4())
        system_prompt = build_outbound_prompt(opening_reason, objective, context)

        if opening_reason:
            initial_message = (
                "The call has been answered. Begin with your outbound greeting now. "
                "State your name, company, and that you are reaching out regarding: "
                f"{opening_reason}. Then ask if now is a good time."
            )
        else:
            initial_message = (
                "The call has been answered. Begin with your outbound greeting now. "
                "State your name, company, and why you are calling. "
                "Then ask if now is a good time."
            )

        record = OutboundCallRecord(
            call_id=call_id,
            phone_number=phone_number,
            campaign_id=campaign_id,
            opening_reason=opening_reason,
            objective=objective,
            context=context,
            system_prompt=system_prompt,
            initial_message=initial_message,
        )

        with self._lock:
            self._calls[call_id] = record

        return record

    def get_call(self, call_id: str) -> OutboundCallRecord | None:
        """Look up a call by its ID."""
        with self._lock:
            return self._calls.get(call_id)

    def update_status(
        self, call_id: str, status: str, **kwargs: Any
    ) -> OutboundCallRecord | None:
        """Thread-safe status update with optional extra fields."""
        with self._lock:
            record = self._calls.get(call_id)
            if record is None:
                return None
            record.status = status
            for key, value in kwargs.items():
                if hasattr(record, key):
                    setattr(record, key, value)
            return record

    def get_active_calls(self) -> list[OutboundCallRecord]:
        """Return calls with status in (initiating, ringing, connected)."""
        with self._lock:
            return [
                r
                for r in self._calls.values()
                if r.status in ("initiating", "ringing", "connected")
            ]

    def get_calls_by_campaign(self, campaign_id: str) -> list[OutboundCallRecord]:
        """Return all calls for a given campaign."""
        with self._lock:
            return [r for r in self._calls.values() if r.campaign_id == campaign_id]

    def reset(self) -> None:
        """Clear all records (useful for testing)."""
        with self._lock:
            self._calls.clear()


# =============================================================================
# Tool Functions — replace these with your actual implementations
# =============================================================================


async def check_order_status(
    order_number: str | None, email: str | None
) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    logger.info(f"Checking order: number={order_number}, email={email}")

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
    logger.info(f"Sending SMS to {phone_number}: {message[:50]}...")

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
    logger.info(f"Scheduling callback: {phone_number}, {department}")

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
    logger.info(f"Transferring to {department}: {reason}")

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
    """Manages a voice conversation session between Plivo and Gemini Live API."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = "Hello, I'm calling for help.",
        stream_id: str = "",
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.stream_id = stream_id

        self._running = False
        self._audio_buffer = bytearray()
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._tools = self._build_tools()

        # Silero VAD state
        self._vad = SileroVADProcessor()
        self._agent_speaking = False
        self._interruption_event = asyncio.Event()

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
                                    description="Order number (starts with TF-)",
                                ),
                                "email": types.Schema(
                                    type=types.Type.STRING,
                                    description="Customer's email",
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
        logger.info(f"Function call: {name} with args: {args}")

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
                logger.info(f"Ending call: {args.get('reason')}")
                self._running = False
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in function {name}: {e}")
            return json.dumps({"error": str(e)})

    def _build_session_config(self) -> types.LiveConnectConfig:
        """Build Gemini Live session configuration."""
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
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=GEMINI_VOICE
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
            tools=self._tools,
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    disabled=False,
                    start_of_speech_sensitivity=(
                        types.StartSensitivity.START_SENSITIVITY_HIGH
                    ),
                    end_of_speech_sensitivity=(
                        types.EndSensitivity.END_SENSITIVITY_HIGH
                    ),
                    prefix_padding_ms=100,
                    silence_duration_ms=500,
                ),
                activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
                turn_coverage=types.TurnCoverage.TURN_INCLUDES_ONLY_ACTIVITY,
            ),
        )

    def _trigger_interruption(self) -> None:
        """Handle barge-in: drain send queue, signal send task."""
        logger.info("Interruption triggered — clearing agent audio")
        self._agent_speaking = False

        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self._interruption_event.set()

    async def run(self) -> None:
        """Run the voice bot session."""
        logger.info(f"Starting bot session for call {self.call_id}")
        self._running = True

        try:
            config = self._build_session_config()

            async with self._client.aio.live.connect(
                model=GEMINI_MODEL, config=config
            ) as session:
                logger.info("Connected to Gemini Live API")

                await session.send_client_content(
                    turns=types.Content(
                        role="user", parts=[types.Part(text=self.initial_message)]
                    ),
                    turn_complete=True,
                )

                await self._run_streaming_tasks(session)

        except Exception as e:
            logger.error(f"Bot session error: {e}")
        finally:
            self._running = False
            logger.info(f"Bot session ended for call {self.call_id}")

    async def _run_streaming_tasks(self, session) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(session), name="plivo_rx"),
            asyncio.create_task(
                self._receive_from_gemini(session), name="gemini_rx"
            ),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    logger.error(
                        f"Task {task.get_name()} failed: {task.exception()}"
                    )
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self, session) -> None:
        """Receive audio from Plivo, run VAD, and forward to Gemini."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)
                        pcm_8k = ulaw_to_pcm(mulaw_audio)

                        speech_started, speech_ended = self._vad.process(pcm_8k)

                        if speech_started and self._agent_speaking:
                            logger.info("Barge-in detected via VAD")
                            self._trigger_interruption()

                        if speech_ended:
                            logger.debug("VAD: user speech ended")

                        pcm_16k = resample_audio(
                            pcm_8k, PLIVO_SAMPLE_RATE, GEMINI_INPUT_RATE
                        )

                        self._audio_buffer.extend(pcm_16k)
                        if len(self._audio_buffer) >= AUDIO_CHUNK_SIZE:
                            chunk = bytes(self._audio_buffer[:AUDIO_CHUNK_SIZE])
                            self._audio_buffer = self._audio_buffer[AUDIO_CHUNK_SIZE:]
                            await session.send_realtime_input(
                                audio=types.Blob(
                                    data=chunk, mime_type="audio/pcm"
                                )
                            )

                elif event == "text":
                    text = message.get("text", "")
                    if text:
                        logger.info(f"Injecting text: {text[:50]}...")
                        await session.send_client_content(
                            turns=types.Content(
                                role="user", parts=[types.Part(text=text)]
                            ),
                            turn_complete=True,
                        )

                elif event == "stop":
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_gemini(self, session) -> None:
        """Receive audio from Gemini and queue for Plivo."""
        try:
            while self._running:
                try:
                    async for response in session.receive():
                        if not self._running:
                            return

                        if response.server_content:
                            if response.server_content.interrupted:
                                logger.info("Gemini confirmed interruption")
                                self._trigger_interruption()

                            model_turn = response.server_content.model_turn
                            if model_turn and model_turn.parts:
                                for part in model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        plivo_audio = gemini_to_plivo(
                                            part.inline_data.data
                                        )
                                        await self._send_queue.put(plivo_audio)

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
                    logger.error(f"Gemini receive error: {e}")
                    raise

        except asyncio.CancelledError:
            pass

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160  # 20ms at 8kHz μ-law
        audio_buffer = bytearray()

        try:
            while self._running:
                if self._interruption_event.is_set():
                    audio_buffer.clear()
                    self._interruption_event.clear()
                    self._agent_speaking = False

                    if self.stream_id:
                        clear_msg = json.dumps({
                            "event": "clearAudio",
                            "stream_id": self.stream_id,
                        })
                        with contextlib.suppress(Exception):
                            await self.websocket.send_text(clear_msg)

                    logger.debug("Send buffer cleared after interruption")
                    continue

                try:
                    audio = await asyncio.wait_for(
                        self._send_queue.get(), timeout=0.1
                    )
                    audio_buffer.extend(audio)
                    self._agent_speaking = True

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
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

                    if (
                        len(audio_buffer) < PLIVO_CHUNK_SIZE
                        and self._send_queue.empty()
                    ):
                        self._agent_speaking = False

                except TimeoutError:
                    if len(audio_buffer) < PLIVO_CHUNK_SIZE:
                        self._agent_speaking = False
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
) -> None:
    """Run a voice agent session for an outbound call."""
    agent = GeminiVoiceBot(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
        stream_id=stream_id,
    )
    await agent.run()
