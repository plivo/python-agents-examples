"""Outbound voice agent — GrokVoiceAgent engine + call state management.

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

from loguru import logger

from utils import (
    GROK_MODEL,
    GROK_VOICE,
    XAI_API_KEY,
    XAI_REALTIME_URL,
    SileroVADProcessor,
    grok_to_plivo,
    plivo_to_grok,
    plivo_to_vad,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

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
    """Map Plivo hangup cause and duration to a high-level outcome.

    See https://www.plivo.com/docs/voice/troubleshooting/hangup-causes/
    """
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

    # If the call was answered and had meaningful duration, consider it success
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
                "State your name, company, and why you are calling. Then ask if now is a good time."
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

    def update_status(self, call_id: str, status: str, **kwargs: Any) -> OutboundCallRecord | None:
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
                r for r in self._calls.values()
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


async def check_order_status(order_number: str | None, email: str | None) -> dict[str, Any]:
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
# Grok Voice Agent
# =============================================================================


class GrokVoiceAgent:
    """Manages a voice conversation session between Plivo and xAI Grok Realtime API."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = "Hello, I'm calling for help.",
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._grok_ws = None
        self._is_responding = False

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions for Grok function calling."""
        return [
            {
                "type": "function",
                "name": "check_order_status",
                "description": "Look up the status of a customer's order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_number": {
                            "type": "string",
                            "description": "Order number (usually starts with TF-)",
                        },
                        "email": {
                            "type": "string",
                            "description": "Customer's email if order number unavailable",
                        },
                    },
                },
            },
            {
                "type": "function",
                "name": "send_sms",
                "description": "Send a text message to the customer's phone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string", "description": "Phone number"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["phone_number", "message"],
                },
            },
            {
                "type": "function",
                "name": "schedule_callback",
                "description": "Schedule a callback from a specialist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string", "description": "Phone number"},
                        "reason": {"type": "string", "description": "Why callback is needed"},
                        "preferred_time": {"type": "string", "description": "Preferred time"},
                        "department": {"type": "string", "description": "Department"},
                    },
                    "required": ["phone_number", "reason", "department"],
                },
            },
            {
                "type": "function",
                "name": "transfer_call",
                "description": "Transfer call to human agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "description": "Department"},
                        "reason": {"type": "string", "description": "Transfer reason"},
                    },
                    "required": ["department", "reason"],
                },
            },
            {
                "type": "function",
                "name": "end_call",
                "description": "End the call gracefully.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason for ending"},
                        "resolution": {
                            "type": "string",
                            "description": "How issue was resolved",
                        },
                    },
                },
            },
        ]

    async def _handle_function_call(self, name: str, call_id: str, arguments: str) -> None:
        """Execute a function call and send the result back to Grok."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

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

        except Exception as e:
            logger.error(f"Error in function {name}: {e}")
            result = {"error": str(e)}

        await self._grok_send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        await self._grok_send(
            {
                "type": "response.create",
                "response": {"modalities": ["text", "audio"]},
            }
        )

    async def _grok_send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the Grok WebSocket."""
        if self._grok_ws:
            await self._grok_ws.send(json.dumps(message))

    def _build_session_config(self) -> dict[str, Any]:
        """Build Grok session configuration."""
        system_prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,
                "voice": GROK_VOICE,
                "turn_detection": None,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "tools": self._build_tools(),
            },
        }

    async def run(self) -> None:
        """Run the voice bot session."""
        import websockets

        logger.info(f"Starting Grok bot session for call {self.call_id}")
        self._running = True

        ws_url = f"{XAI_REALTIME_URL}?model={GROK_MODEL}"
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
        }

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                max_size=None,
            ) as grok_ws:
                self._grok_ws = grok_ws
                logger.info("Connected to xAI Grok Realtime API")

                await self._grok_send(self._build_session_config())

                while True:
                    msg = json.loads(await grok_ws.recv())
                    if msg.get("type") == "session.updated":
                        logger.info("Grok session configured")
                        break
                    elif msg.get("type") == "error":
                        logger.error(f"Grok session error: {msg}")
                        return

                await self._grok_send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": self.initial_message}],
                        },
                    }
                )
                await self._grok_send(
                    {
                        "type": "response.create",
                        "response": {"modalities": ["text", "audio"]},
                    }
                )

                await self._run_streaming_tasks(grok_ws)

        except Exception as e:
            logger.error(f"Bot session error: {e}")
        finally:
            self._running = False
            self._grok_ws = None
            logger.info(f"Bot session ended for call {self.call_id}")

    async def _run_streaming_tasks(self, grok_ws) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._receive_from_grok(grok_ws), name="grok_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, run VAD, and forward to Grok."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)

                        pcm_24k = plivo_to_grok(mulaw_audio)
                        audio_b64 = base64.b64encode(pcm_24k).decode("utf-8")
                        await self._grok_send(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }
                        )

                        vad_audio = plivo_to_vad(mulaw_audio)
                        speech_started, speech_ended = self._vad.process(vad_audio)

                        if speech_started and self._is_responding:
                            logger.info("Barge-in detected, cancelling response")
                            await self._grok_send({"type": "response.cancel"})
                            self._is_responding = False
                            while not self._send_queue.empty():
                                try:
                                    self._send_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break

                        if speech_ended:
                            logger.debug("VAD: committing audio buffer")
                            await self._grok_send({"type": "input_audio_buffer.commit"})
                            await self._grok_send(
                                {
                                    "type": "response.create",
                                    "response": {"modalities": ["text", "audio"]},
                                }
                            )
                            self._vad.reset()

                elif event == "text":
                    text = message.get("text", "")
                    if text:
                        logger.info(f"Injecting text: {text[:50]}...")
                        await self._grok_send(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": text}],
                                },
                            }
                        )
                        await self._grok_send(
                            {
                                "type": "response.create",
                                "response": {"modalities": ["text", "audio"]},
                            }
                        )

                elif event == "stop":
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_grok(self, grok_ws) -> None:
        """Receive events from Grok and queue audio for Plivo."""
        try:
            async for raw_message in grok_ws:
                if not self._running:
                    return

                message = json.loads(raw_message)
                event_type = message.get("type", "")

                if event_type == "response.output_audio.delta":
                    audio_b64 = message.get("delta", "")
                    if audio_b64:
                        pcm_24k = base64.b64decode(audio_b64)
                        plivo_audio = grok_to_plivo(pcm_24k)
                        await self._send_queue.put(plivo_audio)

                elif event_type == "response.created":
                    self._is_responding = True

                elif event_type == "response.done":
                    self._is_responding = False
                    logger.debug("Grok response complete")

                elif event_type == "response.function_call_arguments.done":
                    await self._handle_function_call(
                        name=message.get("name", ""),
                        call_id=message.get("call_id", ""),
                        arguments=message.get("arguments", ""),
                    )

                elif event_type == "response.output_audio_transcript.delta":
                    transcript = message.get("delta", "")
                    if transcript:
                        logger.debug(f"Agent: {transcript}")

                elif event_type == "error":
                    logger.error(f"Grok error: {message.get('error', message)}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "close" not in str(e).lower():
                logger.error(f"Grok receiver error: {e}")

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160
        audio_buffer = bytearray()

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                    audio_buffer.extend(audio)

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
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
) -> None:
    """Run a voice agent session for an outbound call."""
    agent = GrokVoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
    )
    await agent.run()
