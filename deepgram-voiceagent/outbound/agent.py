"""Outbound voice agent — Deepgram Voice Agent API (managed pipeline) + call state.

Native orchestration: one raw WebSocket to the Deepgram Voice Agent API, which
hosts the whole Listen -> Think -> Speak loop server-side:

  Plivo μ-law 8kHz --> Deepgram Voice Agent (wss://agent.deepgram.com/v1/agent/converse)
                         listen / think / speak models are .env config
                         (defaults: flux-general-en, open_ai gpt-4.1-mini,
                         aura-2-thalia-en), passed verbatim into Settings
  Plivo μ-law 8kHz <-- raw μ-law 8kHz binary frames (container "none")

Both directions are μ-law 8kHz, so audio is forwarded without transcoding.
There is no client-side VAD: turn-taking is Deepgram's (Flux end-of-turn by
default), and barge-in is driven by the server's ``UserStartedSpeaking`` event
(Deepgram cancels its own LLM/TTS; we drain our send queue and send
``clearAudio`` to Plivo).

Playback completion is tracked with Plivo checkpoints: a ``_Checkpoint``
sentinel travels through the send queue behind the last audio chunk, so the
checkpoint event is always sent after the audio it marks.

Provides run_agent() for outbound call WebSocket sessions plus CallManager for
tracking call lifecycle. The outbound greeting is literal text that the speak model
speaks verbatim when the callee answers (Deepgram ``agent.greeting``).

Pipeline logging is controlled by the LOG_LEVEL env var:
  verbose — every pipeline event: Deepgram events, packet counts, queue sizes
  normal  — key events: turn lifecycle, transcripts, latencies (default)
  quiet   — errors and session start/end only

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed

Two ways to define the agent, chosen by DEEPGRAM_OUTBOUND_AGENT_ID:
  inline (default, empty) — Settings.agent carries the full definition built here
  reusable config (a UUID) — Settings.agent is that UUID; the per-call campaign
      details + context (UpdatePrompt, "This Call") and greeting (InjectAgentMessage)
      follow SettingsApplied.
Creating a reusable config is a one-off REST call; see the README.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import functools
import json
import os
import random
import threading
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets
from dotenv import load_dotenv
from loguru import logger
from websockets.exceptions import ConnectionClosed

from utils import deepgram_to_plivo, plivo_to_deepgram

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
            with _tracer.start_as_current_span(span_name, attributes={"call_id": self.call_id[:8]}):
                return await fn(self, *args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Configuration
# =============================================================================

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_AGENT_URL = os.getenv("DEEPGRAM_AGENT_URL", "wss://agent.deepgram.com/v1/agent/converse")
# Env values are passed through verbatim to Deepgram's Settings schema
# (agent.listen / agent.think / agent.speak). Use the model ids and provider
# types exactly as Deepgram documents them; see the README "Model Configuration".
DEEPGRAM_LISTEN_MODEL = os.getenv("DEEPGRAM_LISTEN_MODEL", "flux-general-en")
# Flux-only end-of-turn params (sent only when the listen model is a Flux model)
DEEPGRAM_LISTEN_EOT_THRESHOLD = float(os.getenv("DEEPGRAM_LISTEN_EOT_THRESHOLD", "0.7"))
DEEPGRAM_LISTEN_EOT_TIMEOUT_MS = int(os.getenv("DEEPGRAM_LISTEN_EOT_TIMEOUT_MS", "5000"))
# Non-Flux (v1, e.g. nova-3) only; omitted when unset
DEEPGRAM_LISTEN_LANGUAGE = os.getenv("DEEPGRAM_LISTEN_LANGUAGE", "")
# think.provider.type: open_ai | anthropic | google | groq | nvidia | ...
DEEPGRAM_THINK_PROVIDER = os.getenv("DEEPGRAM_THINK_PROVIDER", "open_ai")
DEEPGRAM_THINK_MODEL = os.getenv("DEEPGRAM_THINK_MODEL", "gpt-4.1-mini")
DEEPGRAM_THINK_TEMPERATURE = float(os.getenv("DEEPGRAM_THINK_TEMPERATURE", "0.7"))
DEEPGRAM_SPEAK_MODEL = os.getenv("DEEPGRAM_SPEAK_MODEL", "aura-2-thalia-en")

# Reusable agent configuration UUID (created once via Deepgram's REST API, see the
# README). Empty = inline: Settings carries the full agent definition built below.
DEEPGRAM_OUTBOUND_AGENT_ID = os.getenv("DEEPGRAM_OUTBOUND_AGENT_ID", "").strip()

PLIVO_CHUNK_SIZE = 160  # 20ms of μ-law 8kHz mono
SETTINGS_TIMEOUT_S = 10.0  # Max wait for Welcome + SettingsApplied
KEEPALIVE_INTERVAL_S = 5.0  # Send KeepAlive if nothing sent to Deepgram for this long
END_CALL_GRACE_S = 15.0  # Max wait for goodbye playback before forcing hangup
PRE_SETTINGS_BUFFER_MAX = 16000  # 2s of μ-law 8kHz buffered before SettingsApplied
EXAMPLE_NAME = "deepgram-voiceagent"

# Logging verbosity: "verbose", "normal" (default), "quiet"
LOG_LEVEL = os.getenv("LOG_LEVEL", "normal").lower()

# =============================================================================
# System Prompt + Greeting
# =============================================================================

_OUTBOUND_PROMPT_TEMPLATE = (Path(__file__).parent / "system_prompt.md").read_text().strip()


def build_outbound_prompt(
    opening_reason: str = "",
    objective: str = "",
    context: str = "",
) -> str:
    """Build a concrete outbound system prompt by substituting template variables.

    Live calls pass the campaign fields. The README's reusable-config publish command
    passes pointers to the "This Call" section instead, so the saved prompt has no
    unfilled placeholders (see DeepgramVoiceAgent._build_prompt_update()).
    """
    prompt = _OUTBOUND_PROMPT_TEMPLATE
    prompt = prompt.replace("{{opening_reason}}", opening_reason)
    prompt = prompt.replace("{{objective}}", objective)
    prompt = prompt.replace("{{context}}", context)
    return prompt


# Deepgram speaks agent.greeting verbatim via TTS — literal text, not an LLM instruction.
DEFAULT_OUTBOUND_GREETING = (
    "Hi, this is Alex from TechFlow, following up on your recent interest in our "
    "products. Is now a good time for a quick chat?"
)

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

        # Deepgram speaks agent.greeting verbatim — this must be literal greeting text.
        if opening_reason:
            initial_message = (
                "Hi, this is Alex from TechFlow. "
                f"I'm reaching out because {opening_reason}. "
                "Is now a good time for a quick chat?"
            )
        else:
            initial_message = DEFAULT_OUTBOUND_GREETING

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
# Function Definitions (Deepgram Voice Agent format — client-side, no endpoint)
# =============================================================================

FUNCTION_DEFINITIONS: list[dict[str, Any]] = [
    {
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
        "defer_until_eot": True,
    },
    {
        "name": "end_call",
        "description": (
            "End the call gracefully. Say a short goodbye before calling this; "
            "the call hangs up after your goodbye finishes playing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for ending"},
                "resolution": {"type": "string", "description": "How issue was resolved"},
            },
        },
        "defer_until_eot": True,
    },
]


# =============================================================================
# Send-queue sentinel
# =============================================================================


def _is_plivo_disconnect(exc: BaseException) -> bool:
    """True if ``exc`` means Plivo closed the stream (caller hung up / REST hangup).

    Starlette raises ``WebSocketDisconnect`` (whose ``str()`` is empty) when sending on
    or receiving from a closed socket; that is a normal end of call, not an error.
    """
    return (
        type(exc).__name__ in ("WebSocketDisconnect", "ClientDisconnected")
        or isinstance(exc, (ConnectionClosed, ConnectionError))
        or "1000" in str(exc)
    )


def _is_flux(model: str) -> bool:
    """Deepgram Flux models (``flux-*``) are served by the v2 listen/speak APIs."""
    return model.startswith("flux-")


def _build_listen_provider() -> dict[str, Any]:
    """agent.listen.provider — Flux needs ``version: v2`` and takes EOT params."""
    provider: dict[str, Any] = {"type": "deepgram", "model": DEEPGRAM_LISTEN_MODEL}
    if _is_flux(DEEPGRAM_LISTEN_MODEL):
        provider["version"] = "v2"
        provider["eot_threshold"] = DEEPGRAM_LISTEN_EOT_THRESHOLD
        provider["eot_timeout_ms"] = DEEPGRAM_LISTEN_EOT_TIMEOUT_MS
    elif DEEPGRAM_LISTEN_LANGUAGE:
        provider["language"] = DEEPGRAM_LISTEN_LANGUAGE
    return provider


def _build_speak_provider() -> dict[str, Any]:
    """agent.speak.provider — Aura (v1, default) or Flux TTS (``version: v2``).

    Never add ``language`` here: Deepgram voices reject it and the session dies.
    """
    provider: dict[str, Any] = {"type": "deepgram", "model": DEEPGRAM_SPEAK_MODEL}
    if _is_flux(DEEPGRAM_SPEAK_MODEL):
        provider["version"] = "v2"
    return provider


@dataclass(frozen=True)
class _Checkpoint:
    """Marker placed in the send queue behind the last audio chunk of a response.

    When ``_send_to_plivo`` dequeues it, all preceding audio has been sent, so
    the Plivo ``checkpoint`` event is guaranteed to follow the audio it marks.
    """

    name: str


# =============================================================================
# Voice Agent
# =============================================================================


class DeepgramVoiceAgent:
    """Voice conversation session: Plivo <-> Deepgram Voice Agent API."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = DEFAULT_OUTBOUND_GREETING,
        stream_id: str = "",
        parent_call_id: str = "",
        sip_headers: dict[str, str] | None = None,
        hangup_callback: Callable[[], Awaitable[None]] | None = None,
        agent_config_id: str | None = None,
        opening_reason: str = "",
        objective: str = "",
        context: str = "",
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.parent_call_id = parent_call_id or call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.sip_headers = sip_headers or {}
        self.hangup_callback = hangup_callback
        self._stream_id = stream_id  # Plivo stream ID for checkpoint/clearAudio events
        # Per-call campaign details: inline mode has them rendered into system_prompt;
        # saved mode sends them via UpdatePrompt ("This Call")
        self.opening_reason = opening_reason
        self.objective = objective
        self.campaign_context = context
        # Reusable agent configuration UUID ("" = inline Settings)
        self.agent_config_id = (
            DEEPGRAM_OUTBOUND_AGENT_ID if agent_config_id is None else agent_config_id
        )

        # Connection + handshake
        self._running = False
        self._send_queue: asyncio.Queue[bytes | _Checkpoint] = asyncio.Queue()
        self._tx_buffer = bytearray()  # Partial (<160B) audio awaiting the next chunk
        self._dg_ws: Any = None
        self._settings_applied = asyncio.Event()
        self._pre_settings_audio = bytearray()
        self._pre_settings_texts: list[str] = []
        self._request_id: str | None = None

        # Playback tracking
        self._is_playing = False  # True from first agent audio until playedStream/barge-in
        self._agent_audio_done = False
        self._pending_checkpoint: _Checkpoint | None = None
        self._checkpoint_counter = 0
        self._checkpoint_sent_time: float | None = None
        # Set on barge-in; cleared at the next user EOT (ConversationText user / EndOfTurn)
        # or AgentThinking / FunctionCallRequest / AgentStartedSpeaking.
        # Not cleared by ConversationText(assistant): live, the first audio frame of a
        # response arrives before it, and a late sentence of the cancelled response may too.
        self._drop_agent_audio = False

        # Hangup + keepalive
        self._hangup_pending = False
        self._hangup_deadline: float | None = None
        self._hangup_done = False
        self._last_dg_send = time.monotonic()

        # Session metrics
        self._turn_count = 0
        self._barge_in_count = 0
        self._error_count = 0
        self._session_start = time.monotonic()
        self._plivo_rx_bytes = 0
        self._plivo_tx_chunks = 0
        self._dg_rx_audio_bytes = 0
        self._speech_end_time: float | None = None
        self._ttfs_samples: list[float] = []

        # Per-turn metrics
        self._turn_user_text = ""
        self._turn_agent_text = ""
        self._turn_latency: dict[str, int | None] = self._empty_latency()
        self._turn_latency_report: dict[str, Any] = {}
        self._last_injected_text: str | None = None

    @staticmethod
    def _empty_latency() -> dict[str, int | None]:
        return {"total_latency_ms": None, "tts_latency_ms": None, "ttt_latency_ms": None}

    # -- Structured logging with call ID, elapsed time, and pipeline stage --

    def _log(self, stage: str, msg: str) -> None:
        """Log at 'normal' level — key pipeline events."""
        if LOG_LEVEL == "quiet":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id[:8], elapsed_s=elapsed, stage=stage).info(
            f"[{self.call_id[:8]}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _logv(self, stage: str, msg: str) -> None:
        """Log at 'verbose' level — detailed debugging info."""
        if LOG_LEVEL != "verbose":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id[:8], elapsed_s=elapsed, stage=stage).debug(
            f"[{self.call_id[:8]}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _loge(self, stage: str, msg: str) -> None:
        """Log errors — always visible regardless of LOG_LEVEL."""
        self._error_count += 1
        elapsed = round(time.monotonic() - self._session_start, 2)
        logger.bind(call_id=self.call_id[:8], elapsed_s=elapsed, stage=stage).error(
            f"[{self.call_id[:8]}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    # -- Settings --

    def _build_call_context(self) -> str:
        """Per-call context appended to the prompt ("" when the caller is unknown)."""
        if not self.from_number:
            return ""
        call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
        return f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

    def _build_system_prompt(self) -> str:
        """Build system prompt with call context (inline mode)."""
        return self.system_prompt + self._build_call_context()

    def _build_prompt_update(self) -> str:
        """Saved mode: "This Call" campaign details + call context, appended via UpdatePrompt."""
        lines = ["## This Call", f'- Greeting you already spoke: "{self.initial_message}"']
        if self.opening_reason:
            lines.append(f"- Opening reason (why you are calling): {self.opening_reason}")
        if self.objective:
            lines.append(f"- Objective: {self.objective}")
        if self.campaign_context:
            lines.append(f"- Additional context: {self.campaign_context}")
        return "\n\n" + "\n".join(lines) + self._build_call_context()

    def _build_settings(self) -> dict[str, Any]:
        """Build the Deepgram Voice Agent Settings message.

        Reusable config: ``agent`` is the saved configuration's UUID (all-or-nothing: no
        inline agent fields may be mixed in). Inline: ``agent`` is the full definition —
        listen/think/speak providers, the per-call prompt, FUNCTION_DEFINITIONS and the
        greeting. The README's publish command stores this inline block, minus the
        greeting, as a reusable config.

        Never add ``agent.language`` or ``speak.provider.language`` — Deepgram
        rejects them with an Error and the session dies.
        """
        settings: dict[str, Any] = {
            "type": "Settings",
            "tags": ["plivo", EXAMPLE_NAME],
            "audio": {
                "input": {"encoding": "mulaw", "sample_rate": 8000},
                "output": {"encoding": "mulaw", "sample_rate": 8000, "container": "none"},
            },
        }
        if self.agent_config_id:
            settings["agent"] = self.agent_config_id
            return settings
        settings["agent"] = {
            "listen": {"provider": _build_listen_provider()},
            "think": {
                "provider": {
                    "type": DEEPGRAM_THINK_PROVIDER,
                    "model": DEEPGRAM_THINK_MODEL,
                    "temperature": DEEPGRAM_THINK_TEMPERATURE,
                },
                "prompt": self._build_system_prompt(),
                "functions": FUNCTION_DEFINITIONS,
            },
            "speak": {"provider": _build_speak_provider()},
            "greeting": self.initial_message,
        }
        return settings

    async def _personalize_saved_session(self, dg_ws: Any) -> None:
        """Saved mode, right after SettingsApplied: per-call context, then the greeting.

        UpdatePrompt appends to the saved prompt; InjectAgentMessage is spoken verbatim
        and flows back as ConversationText(assistant) like an inline greeting (turn 1).
        """
        prompt_update = self._build_prompt_update()
        if prompt_update:
            await dg_ws.send(json.dumps({"type": "UpdatePrompt", "prompt": prompt_update}))
        if self.initial_message:
            await dg_ws.send(
                json.dumps({"type": "InjectAgentMessage", "message": self.initial_message})
            )
        self._last_dg_send = time.monotonic()
        self._log(
            "deepgram",
            f"saved config: UpdatePrompt ({len(prompt_update)} chars), "
            f"InjectAgentMessage greeting ({len(self.initial_message)} chars)",
        )

    # -- Session lifecycle --

    @_traced("session")
    async def run(self) -> None:
        """Run the voice agent session."""
        self._session_start = time.monotonic()
        self._running = True
        # Session start always logs (even in quiet mode)
        logger.info(
            f"[{self.call_id[:8]}] [  0.00s] [session] "
            f"started (from={self.from_number}, to={self.to_number}, log={LOG_LEVEL}, "
            f"settings: {self._settings_mode()})"
        )
        logger.bind(
            event="call_answered",
            call_id=self.parent_call_id,
            leg_call_id=self.call_id,
            from_number=self.from_number,
            to_number=self.to_number,
            sip_headers=self.sip_headers,
            stream_id=self._stream_id,
        ).info(
            f"[{self.call_id[:8]}] [  0.00s] [session] "
            f"call answered (sip_headers={self.sip_headers})"
        )

        try:
            async with websockets.connect(
                DEEPGRAM_AGENT_URL,
                additional_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
                max_size=None,
                ping_interval=20,
            ) as dg_ws:
                self._dg_ws = dg_ws
                self._log("session", "connected to Deepgram Voice Agent")
                await self._run_streaming_tasks(dg_ws)
        except Exception as e:
            self._loge("session", f"ERROR: {e}")
        finally:
            self._running = False
            self._emit_session_end()

    def _settings_mode(self) -> str:
        if self.agent_config_id:
            return f"saved agent config {self.agent_config_id}"
        return "inline"

    async def _run_streaming_tasks(self, dg_ws: Any) -> None:
        """Run the three concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._receive_from_deepgram(dg_ws), name="deepgram_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]
        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if not task.cancelled() and task.exception():
                    self._loge("session", f"task {task.get_name()} failed: {task.exception()!r}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _handshake(self, dg_ws: Any) -> None:
        """Welcome -> Settings -> SettingsApplied, then flush buffered input.

        Nothing but Settings may be sent before SettingsApplied, and Settings is
        sent exactly once.
        """
        t0 = time.monotonic()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + SETTINGS_TIMEOUT_S
        settings_sent = False

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                raw = await asyncio.wait_for(dg_ws.recv(), timeout=remaining)
                if isinstance(raw, bytes):
                    self._logv("deepgram", f"ignoring {len(raw)}B binary during handshake")
                    continue
                evt = json.loads(raw)
                etype = evt.get("type")
                if etype == "Welcome":
                    self._request_id = evt.get("request_id")
                    self._log("deepgram", f"Welcome (request_id={self._request_id})")
                    if not settings_sent:
                        await dg_ws.send(json.dumps(self._build_settings()))
                        self._last_dg_send = time.monotonic()
                        settings_sent = True
                        self._logv("deepgram", "Settings sent")
                elif etype == "SettingsApplied":
                    break
                elif etype == "Error":
                    raise RuntimeError(
                        f"Deepgram Error during handshake: "
                        f"{evt.get('code')} {evt.get('description') or evt.get('message')}"
                    )
                else:
                    await self._handle_deepgram_event(evt)
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise RuntimeError(
                f"Deepgram handshake timed out after {SETTINGS_TIMEOUT_S:.0f}s "
                f"(settings_sent={settings_sent})"
            ) from e

        if self.agent_config_id:
            await self._personalize_saved_session(dg_ws)

        # Flush input buffered while waiting. Loop until empty with no await between
        # the final emptiness check and set(), so live input can't overtake buffered input.
        flushed_audio = 0
        flushed_texts = 0
        while self._pre_settings_audio or self._pre_settings_texts:
            audio = bytes(self._pre_settings_audio)
            self._pre_settings_audio.clear()
            texts = list(self._pre_settings_texts)
            self._pre_settings_texts.clear()
            if audio:
                await self._send_audio_to_deepgram(audio)
                flushed_audio += len(audio)
            for text in texts:
                await self._inject_user_text(text)
                flushed_texts += 1
        self._settings_applied.set()
        self._log(
            "deepgram",
            f"SettingsApplied in {(time.monotonic() - t0) * 1000:.0f}ms "
            f"(flushed {flushed_audio}B audio, {flushed_texts} texts)",
        )

    # -- plivo_rx --

    async def _receive_from_plivo(self) -> None:
        """Receive Plivo events: forward audio to Deepgram, handle text/checkpoint acks."""
        media_count = 0
        try:
            while self._running:
                data = await self.websocket.receive_text()
                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    self._logv("plivo_rx", f"ignoring non-JSON frame: {data[:60]!r}")
                    continue
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if not payload:
                        continue
                    try:
                        mulaw_audio = base64.b64decode(payload)
                    except (binascii.Error, ValueError):
                        self._logv("plivo_rx", "ignoring media frame with bad base64")
                        continue
                    self._plivo_rx_bytes += len(mulaw_audio)
                    media_count += 1
                    if media_count == 1:
                        self._log("plivo_rx", "first audio packet received")
                    if media_count % 500 == 0:
                        self._logv("plivo_rx", f"{media_count} packets")

                    dg_audio = plivo_to_deepgram(mulaw_audio)
                    if not self._settings_applied.is_set():
                        self._pre_settings_audio.extend(dg_audio)
                        overflow = len(self._pre_settings_audio) - PRE_SETTINGS_BUFFER_MAX
                        if overflow > 0:
                            del self._pre_settings_audio[:overflow]
                    else:
                        await self._send_audio_to_deepgram(dg_audio)

                elif event == "text":
                    text = message.get("text", "")
                    if text:
                        self._log("plivo_rx", f"text event: '{text[:60]}'")
                        if not self._settings_applied.is_set():
                            self._pre_settings_texts.append(text)
                        else:
                            await self._inject_user_text(text)

                elif event == "playedStream":
                    await self._on_played_stream(message.get("name", ""))

                elif event == "clearedAudio":
                    self._is_playing = False
                    self._logv("plivo_rx", "clearedAudio confirmed by Plivo")

                elif event == "stop":
                    self._log("plivo_rx", "received stop event -- call ended")
                    break

        except Exception as e:
            if _is_plivo_disconnect(e):
                self._log("plivo_rx", f"Plivo stream closed ({type(e).__name__})")
            elif self._running:
                self._loge("plivo_rx", f"ERROR: {e!r}")
        finally:
            self._logv("plivo_rx", f"exiting -- received {media_count} media packets")

    async def _send_audio_to_deepgram(self, audio: bytes) -> None:
        """Send a binary audio frame to Deepgram."""
        if self._dg_ws is None:
            return
        await self._dg_ws.send(audio)
        self._last_dg_send = time.monotonic()

    async def _inject_user_text(self, text: str) -> None:
        """Inject a text user turn (test harness / text channel) via InjectUserMessage.

        Injecting while the agent is playing is a barge-in; handle it locally first so the
        interrupted turn (not the new one) gets the barge-in turn_complete. Deepgram's own
        UserStartedSpeaking for the injection then arrives while idle and is a no-op.
        """
        if self._is_playing:
            await self._on_user_started_speaking()
        self._last_injected_text = text
        self._start_user_turn(text)
        if self._dg_ws is None:
            return
        await self._dg_ws.send(json.dumps({"type": "InjectUserMessage", "content": text}))
        self._last_dg_send = time.monotonic()

    # -- deepgram_rx --

    async def _receive_from_deepgram(self, dg_ws: Any) -> None:
        """Handshake, then dispatch Deepgram events and queue agent audio."""
        await self._handshake(dg_ws)
        try:
            async for msg in dg_ws:
                if isinstance(msg, bytes):
                    self._on_agent_audio(msg)
                    continue
                try:
                    evt = json.loads(msg)
                except json.JSONDecodeError:
                    self._logv("deepgram", f"non-JSON text frame: {msg[:80]!r}")
                    continue
                if not await self._handle_deepgram_event(evt):
                    return
        except ConnectionClosed as e:
            if self._running:
                self._log("deepgram", f"connection closed: {e}")

    async def _handle_deepgram_event(self, evt: dict[str, Any]) -> bool:
        """Handle one Deepgram server event. Returns False if the session must end."""
        etype = evt.get("type", "")

        if etype == "Welcome":
            self._request_id = evt.get("request_id")
        elif etype == "SettingsApplied":
            self._settings_applied.set()
        elif etype == "ConversationText":
            self._on_conversation_text(evt.get("role", ""), evt.get("content", ""))
        elif etype == "UserStartedSpeaking":
            await self._on_user_started_speaking()
        elif etype == "AgentThinking":
            self._drop_agent_audio = False
            self._logv("deepgram", f"AgentThinking: {str(evt.get('content', ''))[:60]}")
        elif etype == "FunctionCallRequest":
            self._drop_agent_audio = False
            await self._on_function_call_request(evt)
        elif etype == "FunctionCallCancelled":
            self._log("deepgram", f"FunctionCallCancelled: {evt}")
        elif etype == "AgentStartedSpeaking":
            self._drop_agent_audio = False
            for key in ("total_latency", "tts_latency", "ttt_latency"):
                value = evt.get(key)
                if isinstance(value, (int, float)):
                    self._turn_latency[f"{key}_ms"] = round(value * 1000)
            self._log(
                "deepgram",
                f"AgentStartedSpeaking (total={self._turn_latency['total_latency_ms']}ms, "
                f"ttt={self._turn_latency['ttt_latency_ms']}ms, "
                f"tts={self._turn_latency['tts_latency_ms']}ms)",
            )
        elif etype == "AgentAudioDone":
            await self._on_agent_audio_done()
        elif etype == "EndOfTurn":
            # Flux end-of-turn (or "manual" after InjectUserMessage): a new response follows
            self._drop_agent_audio = False
            self._logv("deepgram", f"EndOfTurn (trigger={evt.get('trigger')})")
        elif etype == "LatencyReport":
            self._on_latency_report(evt)
        elif etype == "PromptUpdated":
            self._log("deepgram", "PromptUpdated")
        elif etype == "History":
            self._logv("deepgram", f"History: {str(evt)[:120]}")
        elif etype == "InjectionRefused":
            logger.warning(
                f"[{self.call_id[:8]}] [deepgram] InjectionRefused: {evt.get('message', '')}"
            )
        elif etype == "Warning":
            logger.warning(
                f"[{self.call_id[:8]}] [deepgram] Warning: "
                f"{evt.get('code', '')} {evt.get('description') or evt.get('message', '')}"
            )
        elif etype == "Error":
            self._loge(
                "deepgram",
                f"Error: {evt.get('code', '')} {evt.get('description') or evt.get('message', '')}",
            )
            return False
        else:
            self._logv("deepgram", f"event {etype}: {str(evt)[:120]}")
        return True

    def _on_latency_report(self, evt: dict[str, Any]) -> None:
        """Merge a LatencyReport into the per-turn report (seconds -> ms).

        Observed live: Deepgram sends one key per message — ``stt_latency`` roughly
        every audio frame, then ``ttt_token_latency``/``ttt_text_latency``/
        ``ttt_tool_latency``, ``tts_latency`` and ``total_latency`` per response.
        """
        report: dict[str, Any] = {}
        for key, value in evt.items():
            if key == "type" or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key.endswith("latency"):
                report[f"{key}_ms"] = round(value * 1000)
            else:
                report[key] = value
        if not report:
            return
        self._turn_latency_report.update(report)
        for src, dst in (
            ("total_latency_ms", "total_latency_ms"),
            ("tts_latency_ms", "tts_latency_ms"),
            ("ttt_latency_ms", "ttt_latency_ms"),
            ("ttt_text_latency_ms", "ttt_latency_ms"),
        ):
            if src in report:
                self._turn_latency[dst] = report[src]
        if set(report) != {"stt_latency_ms"}:  # stt_latency arrives ~every frame
            self._logv("deepgram", f"LatencyReport: {report}")

    def _on_agent_audio(self, data: bytes) -> None:
        """Queue a binary agent-audio frame for Plivo (dropped after barge-in)."""
        if self._drop_agent_audio:
            self._logv("deepgram", f"dropping {len(data)}B late audio after barge-in")
            return
        if not self._is_playing:
            self._logv("deepgram", "first agent audio of response")
        self._is_playing = True
        self._agent_audio_done = False
        self._dg_rx_audio_bytes += len(data)
        self._send_queue.put_nowait(deepgram_to_plivo(data))

    async def _on_user_started_speaking(self) -> None:
        """Barge-in: Deepgram cancels its LLM/TTS; we clear local + Plivo playback."""
        cleared = 0
        while not self._send_queue.empty():
            try:
                self._send_queue.get_nowait()
                cleared += 1
            except asyncio.QueueEmpty:
                break
        self._tx_buffer.clear()
        clear_event: dict[str, Any] = {"event": "clearAudio"}
        if self._stream_id:
            clear_event["streamId"] = self._stream_id
        with contextlib.suppress(Exception):
            await self.websocket.send_text(json.dumps(clear_event))
        self._pending_checkpoint = None

        was_playing = self._is_playing
        if was_playing:
            self._barge_in_count += 1
            self._emit_turn_complete(barge_in=True)
            self._drop_agent_audio = True
        self._is_playing = False
        self._log(
            "barge_in" if was_playing else "deepgram",
            f"{'barge-in' if was_playing else 'UserStartedSpeaking (idle)'}: cleared={cleared}",
        )

    def _start_user_turn(self, text: str) -> None:
        """Begin a new user turn: bump the counter, reset per-turn state, emit user_text."""
        self._turn_count += 1
        self._turn_user_text = text
        self._turn_agent_text = ""
        self._turn_latency = self._empty_latency()
        self._turn_latency_report = {}
        self._speech_end_time = time.monotonic()
        self._log("turn", f"turn {self._turn_count}: user '{text[:80]}'")
        logger.bind(
            event="user_text",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            text=text,
        ).info(f"[{self.call_id[:8]}] user_text turn {self._turn_count}: '{text[:60]}'")

    def _on_conversation_text(self, role: str, content: str) -> None:
        """Handle ConversationText (final user transcript at EOT, or agent response text)."""
        if role == "user":
            # User EOT: audio after this belongs to the new response, not the cancelled one
            self._drop_agent_audio = False
            if self._last_injected_text is not None and content == self._last_injected_text:
                # Echo of an InjectUserMessage we already counted
                self._last_injected_text = None
                self._logv("deepgram", "ConversationText(user) echo of injected text")
                return
            self._start_user_turn(content)
        elif role == "assistant":
            if self._turn_count == 0:
                self._turn_count = 1  # greeting
            self._turn_agent_text = (
                f"{self._turn_agent_text} {content}".strip() if self._turn_agent_text else content
            )
            self._log("turn", f"turn {self._turn_count}: agent '{content[:80]}'")
            logger.bind(
                event="agent_text",
                call_id=self.parent_call_id,
                turn=self._turn_count,
                text=content,
            ).info(f"[{self.call_id[:8]}] agent_text turn {self._turn_count}: '{content[:60]}'")

    async def _on_function_call_request(self, evt: dict[str, Any]) -> None:
        """Run client-side functions and reply with FunctionCallResponse."""
        for fn in evt.get("functions", []) or []:
            if not fn.get("client_side", False):
                self._logv("tool", f"skipping server-side function {fn.get('name')}")
                continue
            fn_id = fn.get("id", "")
            name = fn.get("name", "")
            try:
                result = await asyncio.wait_for(
                    self._handle_function_call(name, fn.get("arguments", "")), timeout=10
                )
            except (TimeoutError, asyncio.TimeoutError):
                self._loge("tool", f"{name} timed out")
                result = {"error": "function timed out"}
            response = {
                "type": "FunctionCallResponse",
                "id": fn_id,
                "name": name,
                "content": json.dumps(result),
            }
            if self._dg_ws is not None:
                await self._dg_ws.send(json.dumps(response))
                self._last_dg_send = time.monotonic()

    async def _handle_function_call(self, name: str, arguments: Any) -> dict[str, Any]:
        """Execute a function call and return the result."""
        if isinstance(arguments, dict):
            args = arguments
        else:
            try:
                args = json.loads(arguments) if arguments else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
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
                self._log("tool", f"end_call: {args.get('reason')} -- hangup after goodbye")
                self._hangup_pending = True
                self._hangup_deadline = time.monotonic() + END_CALL_GRACE_S
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

            self._log("tool", f"{name} -> {result.get('status', 'done')}")
            return result

        except Exception as e:
            self._loge("tool", f"{name} ERROR: {e}")
            return {"error": str(e)}

    async def _on_agent_audio_done(self) -> None:
        """Deepgram finished sending a response's audio: queue a playback checkpoint."""
        self._agent_audio_done = True
        if self._drop_agent_audio and not self._is_playing:
            # Tail of a response cancelled by barge-in — nothing left to play
            self._logv("deepgram", "AgentAudioDone for interrupted response")
            if self._hangup_pending:
                await self._finish_hangup()
            return

        if not self._stream_id:
            # No Plivo stream (local tests): no checkpoint/playedStream round trip,
            # so treat end-of-audio as end-of-playback.
            await self._complete_playback(playback_ms=None)
            return

        self._checkpoint_counter += 1
        checkpoint = _Checkpoint(f"turn_{self._turn_count}_{self._checkpoint_counter}")
        self._pending_checkpoint = checkpoint
        self._send_queue.put_nowait(checkpoint)
        self._logv("deepgram", f"AgentAudioDone -> queued checkpoint {checkpoint.name}")

    async def _on_played_stream(self, name: str) -> None:
        """Plivo confirms all audio before the named checkpoint has played."""
        pending = self._pending_checkpoint
        if pending is None or name != pending.name:
            self._logv("plivo_rx", f"ignoring stale playedStream '{name}'")
            return
        playback_ms = None
        if self._checkpoint_sent_time is not None:
            playback_ms = round((time.monotonic() - self._checkpoint_sent_time) * 1000)
        self._pending_checkpoint = None
        self._log("plivo_rx", f"playedStream '{name}' -- playback complete ({playback_ms}ms)")
        await self._complete_playback(playback_ms=playback_ms)

    async def _complete_playback(self, playback_ms: int | None) -> None:
        """Common end-of-playback path: emit turn_complete, then hang up if pending."""
        if self._is_playing:
            self._is_playing = False
            self._emit_turn_complete(barge_in=False, playback_ms=playback_ms)
        self._checkpoint_sent_time = None
        if self._hangup_pending:
            await self._finish_hangup()

    async def _finish_hangup(self) -> None:
        """Hang up via the server-provided callback (idempotent), then stop the session."""
        if self._hangup_done:
            return
        self._hangup_done = True
        self._hangup_pending = False
        self._log("session", "goodbye played -- hanging up")
        if self.hangup_callback is not None:
            try:
                await self.hangup_callback()
            except Exception as e:
                self._loge("session", f"hangup callback ERROR: {e}")
        self._running = False

    # -- plivo_tx --

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo in 160-byte (20ms) chunks, plus checkpoints."""
        try:
            while self._running:
                try:
                    item = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                except (TimeoutError, asyncio.TimeoutError):
                    await self._idle_housekeeping()
                    continue

                if isinstance(item, _Checkpoint):
                    if self._tx_buffer:
                        pad = PLIVO_CHUNK_SIZE - len(self._tx_buffer)
                        self._tx_buffer.extend(b"\xff" * pad)  # μ-law silence
                        await self._send_play_audio(bytes(self._tx_buffer))
                        self._tx_buffer.clear()
                    if self._stream_id:
                        await self.websocket.send_text(
                            json.dumps(
                                {
                                    "event": "checkpoint",
                                    "streamId": self._stream_id,
                                    "name": item.name,
                                }
                            )
                        )
                        self._checkpoint_sent_time = time.monotonic()
                        self._logv("plivo_tx", f"checkpoint sent: {item.name}")
                    continue

                self._tx_buffer.extend(item)
                while len(self._tx_buffer) >= PLIVO_CHUNK_SIZE:
                    chunk = bytes(self._tx_buffer[:PLIVO_CHUNK_SIZE])
                    del self._tx_buffer[:PLIVO_CHUNK_SIZE]
                    await self._send_play_audio(chunk)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not _is_plivo_disconnect(e):
                raise
            self._log("plivo_tx", f"Plivo stream closed ({type(e).__name__})")
        finally:
            self._logv("plivo_tx", f"exiting -- total {self._plivo_tx_chunks} chunks sent")

    async def _send_play_audio(self, chunk: bytes) -> None:
        """Send one playAudio message to Plivo and update TTFS metrics."""
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
            self._logv(
                "plivo_tx",
                f"{self._plivo_tx_chunks} chunks sent, queue={self._send_queue.qsize()}",
            )

    async def _idle_housekeeping(self) -> None:
        """KeepAlive when Deepgram hasn't heard from us; enforce the hangup deadline."""
        now = time.monotonic()
        if (
            self._settings_applied.is_set()
            and self._dg_ws is not None
            and now - self._last_dg_send > KEEPALIVE_INTERVAL_S
        ):
            with contextlib.suppress(Exception):
                await self._dg_ws.send(json.dumps({"type": "KeepAlive"}))
                self._last_dg_send = now
                self._logv("deepgram", "KeepAlive sent")
        if (
            self._hangup_pending
            and self._hangup_deadline is not None
            and now >= self._hangup_deadline
        ):
            self._log("session", "hangup deadline reached -- forcing hangup")
            await self._finish_hangup()

    # -- Structured events --

    def _emit_turn_complete(self, barge_in: bool = False, playback_ms: int | None = None) -> None:
        """Emit a structured turn_complete event with per-turn metrics."""
        if playback_ms is None and self._checkpoint_sent_time is not None:
            playback_ms = round((time.monotonic() - self._checkpoint_sent_time) * 1000)
        logger.bind(
            event="turn_complete",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            barge_in=barge_in,
            user_text=self._turn_user_text or "",
            agent_text=self._turn_agent_text or "",
            plivo_rx_bytes=self._plivo_rx_bytes,
            plivo_tx_chunks=self._plivo_tx_chunks,
            playback_ms=playback_ms,
            total_latency_ms=self._turn_latency.get("total_latency_ms"),
            tts_latency_ms=self._turn_latency.get("tts_latency_ms"),
            ttt_latency_ms=self._turn_latency.get("ttt_latency_ms"),
            latency_report=dict(self._turn_latency_report),
        ).info(
            f"[{self.call_id[:8]}] turn {self._turn_count} complete"
            f"{' (barge-in)' if barge_in else ''}"
        )

    def _emit_session_end(self) -> None:
        """Emit the session_end summary event (always logged, even in quiet mode)."""
        duration = round(time.monotonic() - self._session_start, 1)
        avg_ttfs = (
            round(sum(self._ttfs_samples) / len(self._ttfs_samples)) if self._ttfs_samples else None
        )
        logger.bind(
            event="session_end",
            call_id=self.parent_call_id,
            duration_s=duration,
            turns=self._turn_count,
            barge_ins=self._barge_in_count,
            errors=self._error_count,
            ttfs_avg_ms=avg_ttfs,
            ttfs_samples=len(self._ttfs_samples),
            rx_bytes=self._plivo_rx_bytes,
            tx_chunks=self._plivo_tx_chunks,
            deepgram_request_id=self._request_id,
            agent_config=self.agent_config_id or "inline",
        ).info(
            f"[{self.call_id[:8]}] [{duration:7.1f}s] [session] "
            f"ended -- {self._turn_count} turns, "
            f"{self._barge_in_count} barge-ins, "
            f"TTFS avg={avg_ttfs}ms, "
            f"rx={self._plivo_rx_bytes}B, tx={self._plivo_tx_chunks} chunks"
        )


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    from_number: str = "",
    to_number: str = "",
    system_prompt: str | None = None,
    initial_message: str = DEFAULT_OUTBOUND_GREETING,
    stream_id: str = "",
    parent_call_id: str = "",
    sip_headers: dict[str, str] | None = None,
    hangup_callback: Callable[[], Awaitable[None]] | None = None,
    opening_reason: str = "",
    objective: str = "",
    context: str = "",
) -> None:
    """Run a voice agent session for an outbound call."""
    agent = DeepgramVoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
        stream_id=stream_id,
        parent_call_id=parent_call_id,
        sip_headers=sip_headers,
        hangup_callback=hangup_callback,
        opening_reason=opening_reason,
        objective=objective,
        context=context,
    )
    await agent.run()
