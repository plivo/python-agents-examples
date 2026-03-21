"""Outbound voice agent — GPT-4.1 mini + Sarvam STT + ElevenLabs TTS + call state management.

Uses a pipeline architecture:
  Plivo audio → Sarvam STT → GPT-4.1 mini → ElevenLabs TTS → Plivo audio

Loads the outbound system prompt and provides run_agent() for handling
outbound call WebSocket sessions, plus CallManager for tracking call lifecycle.

Pipeline logging is controlled by the LOG_LEVEL env var:
  verbose — every pipeline event: per-packet stats, VAD frames, queue sizes, TTFB
  normal  — key events: turn lifecycle, STT results, LLM responses, TTS timing (default)
  quiet   — errors and session start/end only

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import json
import os
import random
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from utils import (
    SileroVADProcessor,
    elevenlabs_to_plivo,
    plivo_to_sarvam_streaming,
    plivo_to_vad,
)

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
    Methods can call ``_otel_trace.get_current_span()`` to set domain attributes.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if not _tracer:
                return await fn(self, *args, **kwargs)
            with _tracer.start_as_current_span(
                span_name, attributes={"call_id": self.call_id}
            ):
                return await fn(self, *args, **kwargs)

        return wrapper

    return decorator


class SarvamStreamingSTT:
    """Real-time speech-to-text using Sarvam WebSocket API.

    Streams audio continuously at 8kHz PCM and maintains the latest
    final transcript. Uses Sarvam's data events to accumulate transcript
    parts. Supports an optional on_transcript callback for event-driven
    turn processing.
    """

    def __init__(self, call_id: str = ""):
        self._call_id = call_id
        self._ws = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._receive_task: asyncio.Task | None = None
        self._transcript_parts: list[str] = []
        self.on_transcript: asyncio.Queue[str] | None = None

    @property
    def latest_transcript(self) -> str:
        """Get the latest accumulated transcript."""
        return " ".join(self._transcript_parts).strip()

    def clear_transcript(self) -> None:
        """Clear the transcript buffer for a new turn."""
        self._transcript_parts.clear()

    async def connect(self) -> None:
        """Connect to Sarvam streaming WebSocket."""
        self._session = aiohttp.ClientSession()
        self._running = True

        url = (
            f"wss://api.sarvam.ai/speech-to-text/ws"
            f"?language-code={SARVAM_STT_LANGUAGE}"
            f"&model=saaras:v3"
            f"&mode=transcribe"
            f"&sample_rate=8000"
            f"&input_audio_codec=pcm_s16le"
        )

        headers = {"Api-Subscription-Key": SARVAM_API_KEY}
        self._ws = await self._session.ws_connect(url, headers=headers)
        logger.bind(call_id=self._call_id).info("Connected to Sarvam streaming STT")
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Receive transcription results from Sarvam."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    msg_type = data.get("type", "")
                    if msg_type == "data":
                        transcript = data.get("data", {}).get("transcript", "")
                        if transcript.strip():
                            self._transcript_parts.append(transcript)
                            logger.bind(call_id=self._call_id).debug(f"Sarvam STT: '{transcript}'")
                            if self.on_transcript is not None:
                                self.on_transcript.put_nowait(transcript)
                    elif msg_type == "error":
                        logger.bind(call_id=self._call_id).error(f"Sarvam STT error: {data}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.bind(call_id=self._call_id).error(f"Sarvam WebSocket error: {msg.data}")
                    break
        except Exception as e:
            if self._running:
                logger.bind(call_id=self._call_id).error(f"Sarvam receive error: {e}")

    async def send_audio(self, pcm_8k: bytes) -> None:
        """Send PCM16 8kHz audio to Sarvam."""
        if self._ws and not self._ws.closed:
            payload = json.dumps({
                "audio": {
                    "data": base64.b64encode(pcm_8k).decode(),
                    "sample_rate": "8000",
                    "encoding": "audio/wav",
                }
            })
            await self._ws.send_str(payload)

    async def close(self) -> None:
        """Close the Sarvam connection."""
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()


# Agent configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL = os.getenv(
    "SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text"
)
SARVAM_STT_LANGUAGE = os.getenv("SARVAM_STT_LANGUAGE", "en-IN")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")  # Rachel
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")

# Logging verbosity: "verbose", "normal" (default), "quiet"
LOG_LEVEL = os.getenv("LOG_LEVEL", "normal").lower()

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
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Voice conversation session: Plivo + Sarvam STT + GPT-4.1 mini + ElevenLabs TTS."""

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
        self.sip_headers = sip_headers or {}

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor(call_id=call_id)
        self._is_playing = False  # True while Plivo is playing audio to caller
        self._barged_in = False  # True after first speech_started until speech_ended
        self._speech_ended_pending = False  # VAD ended but no transcript yet
        self._stt = SarvamStreamingSTT(call_id=call_id)
        self._transcript_queue: asyncio.Queue[str] = asyncio.Queue()
        self._stt.on_transcript = self._transcript_queue
        self._turn_lock = asyncio.Lock()
        self._conversation_history: list[dict[str, str]] = []
        self._current_tts_task: asyncio.Task | None = None
        self._current_turn_task: asyncio.Task | None = None
        self._stream_id = stream_id  # Plivo stream ID for checkpoint events
        self._checkpoint_counter = 0
        self._checkpoint_sent_time: float | None = None
        self._turn_count = 0
        self._barge_in_count = 0
        self._error_count = 0
        self._session_start = time.monotonic()
        self._plivo_rx_bytes = 0
        self._plivo_tx_chunks = 0
        self._speech_end_time: float | None = None
        self._ttfs_samples: list[float] = []

        # Per-turn metrics (reset at start of each _process_text_turn)
        self._turn_llm_ms: float | None = None
        self._turn_tts_total_ms: float | None = None
        self._turn_tts_ttfb_ms: float | None = None
        self._turn_tts_chunks: int = 0
        self._turn_tts_audio_s: float = 0.0
        self._turn_text: str = ""
        self._turn_start_time: float | None = None

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
        playback_ms = None
        if self._checkpoint_sent_time is not None:
            playback_ms = round((time.monotonic() - self._checkpoint_sent_time) * 1000)
            self._checkpoint_sent_time = None
        logger.bind(
            event="turn_complete",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            user_text=self._turn_text[:80] if self._turn_text else "",
            llm_ms=self._turn_llm_ms,
            tts_total_ms=self._turn_tts_total_ms,
            tts_ttfb_ms=self._turn_tts_ttfb_ms,
            tts_chunks=self._turn_tts_chunks,
            tts_audio_duration_s=self._turn_tts_audio_s,
            playback_ms=playback_ms,
            barge_in=barge_in,
        ).info(
            f"[{self.call_id}] turn {self._turn_count} complete"
            f"{' (barge-in)' if barge_in else ''}"
        )

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions for OpenAI function calling."""
        return [
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_callback",
                    "description": "Schedule a callback from a specialist.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "phone_number": {"type": "string", "description": "Phone number"},
                            "reason": {
                                "type": "string",
                                "description": "Why callback is needed",
                            },
                            "preferred_time": {
                                "type": "string",
                                "description": "Preferred time",
                            },
                            "department": {"type": "string", "description": "Department"},
                        },
                        "required": ["phone_number", "reason", "department"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
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
            },
            {
                "type": "function",
                "function": {
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
            },
        ]

    async def _handle_function_call(
        self, name: str, arguments: str
    ) -> dict[str, Any]:
        """Execute a function call and return the result."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

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
            return result

        except Exception as e:
            self._loge("tool", f"{name} ERROR: {e}")
            return {"error": str(e)}

    @_traced("llm")
    async def _generate_llm_response(self, user_text: str) -> str:
        """Send conversation to GPT-4.1 mini and return text response."""
        import httpx

        self._conversation_history.append({"role": "user", "content": user_text})

        messages = [
            {"role": "system", "content": self.system_prompt},
            *self._conversation_history,
        ]

        self._logv("llm", f"request ({len(messages)} messages, last: '{user_text[:60]}')")
        t0 = time.monotonic()

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENAI_MODEL,
                        "messages": messages,
                        "tools": self._build_tools(),
                        "max_tokens": 300,
                    },
                )
                response.raise_for_status()
                result = response.json()

            choice = result["choices"][0]
            message = choice["message"]
            latency = (time.monotonic() - t0) * 1000
            self._turn_llm_ms = round(latency)
            tokens = result.get("usage", {})

            if _otel_trace:
                span = _otel_trace.get_current_span()
                span.set_attribute("llm.latency_ms", latency)
                span.set_attribute("gen_ai.request.model", OPENAI_MODEL)
                span.set_attribute(
                    "gen_ai.usage.prompt_tokens", tokens.get("prompt_tokens", 0)
                )
                span.set_attribute(
                    "gen_ai.usage.completion_tokens", tokens.get("completion_tokens", 0)
                )

            # Handle tool calls
            if message.get("tool_calls"):
                tool_names = [tc["function"]["name"] for tc in message["tool_calls"]]
                self._log("llm", f"tool calls ({latency:.0f}ms): {tool_names}")
                self._conversation_history.append(message)
                for tool_call in message["tool_calls"]:
                    fn_name = tool_call["function"]["name"]
                    fn_args = tool_call["function"]["arguments"]
                    fn_result = await self._handle_function_call(fn_name, fn_args)

                    self._conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(fn_result),
                    })

                # Get follow-up response after tool calls
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    *self._conversation_history,
                ]
                self._logv("llm", "follow-up request after tool calls")
                t1 = time.monotonic()
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {OPENAI_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": OPENAI_MODEL,
                            "messages": messages,
                            "max_tokens": 300,
                        },
                    )
                    response.raise_for_status()
                    result = response.json()
                    message = result["choices"][0]["message"]
                    latency2 = (time.monotonic() - t1) * 1000
                    self._turn_llm_ms = round(latency + latency2)
                    self._logv("llm", f"follow-up response ({latency2:.0f}ms)")

            assistant_text = message.get("content", "")
            self._conversation_history.append({"role": "assistant", "content": assistant_text})
            self._log(
                "llm",
                f"response ({latency:.0f}ms, "
                f"{tokens.get('prompt_tokens', '?')}→{tokens.get('completion_tokens', '?')} tok): "
                f"'{assistant_text[:80]}'",
            )
            return assistant_text

        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            self._turn_llm_ms = round(latency)
            self._loge("llm", f"ERROR ({latency:.0f}ms): {e}")
            return "I'm sorry, I'm having trouble processing that right now. Could you repeat that?"

    @_traced("tts")
    async def _synthesize_with_elevenlabs(self, text: str) -> None:
        """Stream text through ElevenLabs WebSocket TTS and queue audio for Plivo.

        Uses the ElevenLabs text-to-speech WebSocket API for lower latency:
        text is sent in sentence chunks so audio generation starts before the
        full text is delivered, reducing time-to-first-byte.
        """
        if not text.strip():
            return

        self._logv("tts", f"requesting synthesis ({len(text)} chars)")
        t0 = time.monotonic()

        ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech"
            f"/{ELEVENLABS_VOICE_ID}/stream-input"
            f"?model_id={ELEVENLABS_MODEL_ID}"
            f"&output_format=pcm_24000"
        )

        try:
            async with aiohttp.ClientSession() as session, session.ws_connect(
                ws_url, timeout=30.0
            ) as ws:
                # Send BOS (beginning of stream) message with config
                bos_message = {
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                    },
                    "xi_api_key": ELEVENLABS_API_KEY,
                }
                await ws.send_json(bos_message)

                # Split text into sentence chunks for progressive synthesis
                import re

                sentences = re.split(r"(?<=[.!?])\s+", text.strip())
                for sentence in sentences:
                    if not sentence.strip():
                        continue
                    await ws.send_json({"text": sentence + " "})

                # Send EOS (end of stream) — flush remaining audio
                await ws.send_json({"text": ""})

                # Receive audio chunks until the server closes
                first_chunk_time = None
                chunk_count = 0
                total_bytes = 0
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        audio_b64 = data.get("audio")
                        if audio_b64:
                            pcm_24k = base64.b64decode(audio_b64)
                            if not self._running or not self._is_playing:
                                self._log(
                                    "tts",
                                    f"interrupted after {chunk_count} chunks "
                                    f"(running={self._running}, "
                                    f"responding={self._is_playing})",
                                )
                                break
                            if first_chunk_time is None:
                                first_chunk_time = time.monotonic()
                                ttfb = (first_chunk_time - t0) * 1000
                                self._logv(
                                    "tts", f"first chunk (TTFB: {ttfb:.0f}ms)"
                                )
                            plivo_audio = elevenlabs_to_plivo(pcm_24k)
                            await self._send_queue.put(plivo_audio)
                            chunk_count += 1
                            total_bytes += len(plivo_audio)
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.ERROR,
                    ):
                        break

                total_time = (time.monotonic() - t0) * 1000
                audio_duration = total_bytes / 8000  # μ-law 8kHz = 1 byte per sample
                ttfb_str = ""
                if first_chunk_time is not None:
                    ttfb_str = f"TTFB={(first_chunk_time - t0) * 1000:.0f}ms, "
                self._log(
                    "tts",
                    f"done: {chunk_count} chunks, "
                    f"{ttfb_str}{audio_duration:.1f}s audio in {total_time:.0f}ms",
                )

                # Store per-turn TTS metrics
                self._turn_tts_total_ms = round(total_time)
                self._turn_tts_chunks = chunk_count
                self._turn_tts_audio_s = round(audio_duration, 2)
                if first_chunk_time is not None:
                    self._turn_tts_ttfb_ms = round((first_chunk_time - t0) * 1000)

                if _otel_trace:
                    span = _otel_trace.get_current_span()
                    span.set_attribute("tts.total_ms", total_time)
                    span.set_attribute("tts.audio_duration_s", audio_duration)
                    span.set_attribute("tts.chunks", chunk_count)
                    if first_chunk_time is not None:
                        span.set_attribute(
                            "tts.ttfb_ms", (first_chunk_time - t0) * 1000
                        )

        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            self._loge("tts", f"ERROR ({latency:.0f}ms): {e}")

    def _build_system_prompt(self) -> str:
        """Build system prompt with call context."""
        system_prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return system_prompt

    async def run(self) -> None:
        """Run the voice bot session."""
        self._session_start = time.monotonic()
        self._running = True
        self.system_prompt = self._build_system_prompt()
        # Session start always logs (even in quiet mode)
        logger.info(
            f"[{self.call_id}] [  0.00s] [session] "
            f"started (from={self.from_number}, to={self.to_number}, log={LOG_LEVEL})"
        )
        logger.bind(
            event="call_answered",
            call_id=self.parent_call_id,
            leg_call_id=self.call_id,
            from_number=self.from_number,
            to_number=self.to_number,
            sip_headers=self.sip_headers,
        ).info(
            f"[{self.call_id}] [  0.00s] [session] "
            f"call answered (sip_headers={self.sip_headers})"
        )

        await self._stt.connect()

        # Generate initial greeting
        try:
            self._turn_count += 1
            self._log("turn", f"turn {self._turn_count}: generating greeting")
            greeting = await self._generate_llm_response(self.initial_message)
            if greeting:
                self._is_playing = True
                await self._synthesize_with_elevenlabs(greeting)
                await self._send_checkpoint()
                self._log("turn", f"turn {self._turn_count}: greeting queued for playback")
        except Exception as e:
            self._loge("session", f"greeting ERROR: {e}")

        # Run streaming tasks
        self._log("session", "starting streaming tasks (plivo_rx, plivo_tx)")
        try:
            await self._run_streaming_tasks()
        except Exception as e:
            self._loge("session", f"streaming ERROR: {e}")
        finally:
            self._running = False
            await self._stt.close()
            duration = round(time.monotonic() - self._session_start, 1)
            avg_ttfs = (
                round(sum(self._ttfs_samples) / len(self._ttfs_samples))
                if self._ttfs_samples
                else None
            )
            # Session end always logs (even in quiet mode)
            logger.bind(
                event="call_summary",
                call_id=self.parent_call_id,
                duration_s=duration,
                turns=self._turn_count,
                barge_ins=self._barge_in_count,
                ttfs_avg_ms=avg_ttfs,
                ttfs_samples=len(self._ttfs_samples),
                errors=self._error_count,
                rx_bytes=self._plivo_rx_bytes,
                tx_chunks=self._plivo_tx_chunks,
            ).info(
                f"[{self.call_id}] [{duration:7.1f}s] [session] "
                f"ended — {self._turn_count} turns, "
                f"{self._barge_in_count} barge-ins, "
                f"TTFS avg={avg_ttfs}ms, "
                f"rx={self._plivo_rx_bytes}B, tx={self._plivo_tx_chunks} chunks"
            )

    async def _run_streaming_tasks(self) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
            asyncio.create_task(self._watch_transcripts(), name="stt_watch"),
        ]

        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    self._loge("session", f"task {task.get_name()} failed: {task.exception()}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, stream to Sarvam STT, run VAD for turn detection."""
        media_count = 0
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
                        media_count += 1

                        if media_count == 1:
                            self._log("plivo_rx", "first audio packet received")
                        if media_count % 500 == 0:
                            self._logv("plivo_rx", f"{media_count} packets")

                        # Always forward audio to Sarvam STT — even
                        # during playback. Echo gets transcribed but is
                        # cleared on barge-in (clear_transcript). This
                        # ensures STT has processed user speech by the
                        # time speech_ended fires.
                        pcm_8k = plivo_to_sarvam_streaming(mulaw_audio)
                        await self._stt.send_audio(pcm_8k)

                        # VAD always runs — barge-in must work during
                        # playback. Echo triggers speech_started which
                        # stops playback; the empty STT transcript (no
                        # echo was sent to STT) causes the turn to be
                        # skipped. Real user speech after barge-in gets
                        # sent to STT and produces a real transcript.
                        vad_audio = plivo_to_vad(mulaw_audio)
                        speech_started, speech_ended = self._vad.process(
                            vad_audio
                        )

                        if speech_started and not self._barged_in:
                            self._barged_in = True
                            self._log("vad", "speech START detected")
                            interrupted = False
                            if (
                                self._current_turn_task
                                and not self._current_turn_task.done()
                            ):
                                self._current_turn_task.cancel()
                                self._log(
                                    "vad", "cancelled in-flight turn task"
                                )
                                interrupted = True
                            if (
                                self._current_tts_task
                                and not self._current_tts_task.done()
                            ):
                                self._current_tts_task.cancel()
                                interrupted = True
                            cleared = 0
                            while not self._send_queue.empty():
                                try:
                                    self._send_queue.get_nowait()
                                    cleared += 1
                                except asyncio.QueueEmpty:
                                    break
                            clear_event = {"event": "clearAudio"}
                            if self._stream_id:
                                clear_event["streamId"] = self._stream_id
                            await self.websocket.send_text(
                                json.dumps(clear_event)
                            )
                            self._log(
                                "vad",
                                f"barge-in: clearAudio sent, "
                                f"cancelled={interrupted}, "
                                f"cleared={cleared} chunks",
                            )
                            self._stt.clear_transcript()
                            self._is_playing = False
                            self._barge_in_count += 1
                            if interrupted:
                                self._emit_turn_complete(barge_in=True)

                        if speech_ended:
                            self._speech_end_time = time.monotonic()
                            transcript = self._stt.latest_transcript
                            if transcript.strip():
                                # STT already delivered — process now
                                self._log(
                                    "vad",
                                    "speech END — transcript ready",
                                )
                                self._commit_turn(transcript)
                            else:
                                # STT hasn't delivered yet — flag it.
                                # _watch_transcripts will trigger when
                                # Sarvam delivers.
                                self._speech_ended_pending = True
                                self._log(
                                    "vad",
                                    "speech END — waiting for STT",
                                )
                            self._vad.reset()
                            self._barged_in = False

                elif event == "playedStream":
                    # Plivo confirms all audio before checkpoint has played
                    name = message.get("name", "")
                    self._is_playing = False
                    playback_ms = None
                    if self._checkpoint_sent_time is not None:
                        playback_ms = round(
                            (time.monotonic() - self._checkpoint_sent_time) * 1000
                        )
                    self._log(
                        "plivo_rx",
                        f"playedStream: '{name}' — playback complete ({playback_ms}ms)",
                    )
                    self._emit_turn_complete(barge_in=False)

                elif event == "clearedAudio":
                    self._is_playing = False
                    self._logv("plivo_rx", "clearedAudio confirmed by Plivo")

                elif event == "text":
                    text = message.get("text", "")
                    if text:
                        self._turn_count += 1
                        self._log(
                            "plivo_rx",
                            f"text event (turn {self._turn_count}): '{text[:60]}'",
                        )
                        task = asyncio.create_task(
                            self._process_text_turn(text),
                            name=f"text_turn_{self._turn_count}",
                        )
                        task.add_done_callback(
                            lambda t: t.exception() if not t.cancelled() else None
                        )

                elif event == "stop":
                    self._log("plivo_rx", "received stop event — call ended")
                    break

        except Exception as e:
            if "1000" not in str(e):
                self._loge("plivo_rx", f"ERROR: {e}")
        finally:
            self._logv("plivo_rx", f"exiting — received {media_count} media packets")

    def _commit_turn(self, transcript: str) -> None:
        """Commit a turn for processing — creates the turn task."""
        self._turn_count += 1
        self._log("turn", f"turn {self._turn_count}: '{transcript[:80]}'")
        self._current_turn_task = asyncio.create_task(
            self._process_text_turn(transcript),
            name=f"turn_{self._turn_count}",
        )
        self._current_turn_task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() else None
        )
        self._stt.clear_transcript()
        self._speech_ended_pending = False

    async def _watch_transcripts(self) -> None:
        """Watch for STT transcripts that arrive after speech_ended.

        Convergence gate: if VAD fired speech_ended but STT hadn't
        delivered yet, this task picks up the transcript when it arrives
        and commits the turn.
        """
        try:
            while self._running:
                try:
                    await asyncio.wait_for(
                        self._transcript_queue.get(), timeout=0.2
                    )
                except TimeoutError:
                    continue
                # Transcript arrived — check if VAD is waiting
                if self._speech_ended_pending:
                    transcript = self._stt.latest_transcript
                    if transcript.strip():
                        self._log(
                            "stt",
                            "transcript arrived after speech_ended — "
                            "committing turn",
                        )
                        self._commit_turn(transcript)
        except asyncio.CancelledError:
            pass

    async def _process_text_turn(self, text: str) -> None:
        """Process a text-based turn: LLM → TTS."""
        async with self._turn_lock:
            # Reset per-turn metrics
            self._turn_llm_ms = None
            self._turn_tts_total_ms = None
            self._turn_tts_ttfb_ms = None
            self._turn_tts_chunks = 0
            self._turn_tts_audio_s = 0.0
            self._turn_text = text
            self._turn_start_time = time.monotonic()
            try:
                response_text = await self._generate_llm_response(text)
                if not response_text.strip():
                    self._logv("turn", "empty LLM response, skipping TTS")
                    return

                self._is_playing = True
                self._current_tts_task = asyncio.create_task(
                    self._synthesize_with_elevenlabs(response_text),
                    name="tts_synthesis",
                )
                await self._current_tts_task
                # Send checkpoint so Plivo notifies us when playback finishes
                await self._send_checkpoint()
                self._log("turn", "TTS done, audio queued for playback")

            except asyncio.CancelledError:
                self._is_playing = False
                self._log("turn", "turn cancelled (barge-in)")
            except Exception as e:
                self._is_playing = False
                self._loge("turn", f"text turn ERROR: {e}")

    async def _send_checkpoint(self) -> None:
        """Send a Plivo checkpoint event after queued audio.

        Plivo responds with playedStream when all audio before the checkpoint
        has been played to the caller. We use this to track _is_playing state.
        """
        if not self._stream_id:
            return
        self._checkpoint_counter += 1
        name = f"turn_{self._turn_count}_{self._checkpoint_counter}"
        checkpoint = {
            "event": "checkpoint",
            "streamId": self._stream_id,
            "name": name,
        }
        await self.websocket.send_text(json.dumps(checkpoint))
        self._checkpoint_sent_time = time.monotonic()
        self._logv("plivo_tx", f"checkpoint sent: {name}")

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
                        self._plivo_tx_chunks += 1
                        if self._plivo_tx_chunks == 1:
                            self._log("plivo_tx", "first audio chunk sent to Plivo")
                        if self._speech_end_time is not None:
                            ttfs = (time.monotonic() - self._speech_end_time) * 1000
                            self._ttfs_samples.append(ttfs)
                            self._log("metrics", f"TTFS: {ttfs:.0f}ms")
                            self._speech_end_time = None
                        if self._plivo_tx_chunks % 500 == 0:
                            q = self._send_queue.qsize()
                            self._logv(
                                "plivo_tx",
                                f"{self._plivo_tx_chunks} chunks sent, queue={q}",
                            )

                except TimeoutError:
                    continue

        except asyncio.CancelledError:
            pass
        finally:
            self._logv("plivo_tx", f"exiting — total {self._plivo_tx_chunks} chunks sent")


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
    """Run a voice agent session for an outbound call."""
    agent = VoiceAgent(
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
