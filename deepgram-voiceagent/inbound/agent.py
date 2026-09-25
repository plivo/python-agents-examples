"""Inbound voice agent — Deepgram Voice Agent API (managed listen/think/speak pipeline).

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

Pipeline logging is controlled by the LOG_LEVEL env var:
  verbose — every pipeline event: Deepgram events, packet counts, queue sizes
  normal  — key events: turn lifecycle, transcripts, latencies (default)
  quiet   — errors and session start/end only

Two ways to define the agent, chosen by DEEPGRAM_INBOUND_AGENT_ID:
  inline (default, empty) — Settings.agent carries the full definition built here
  reusable config (a UUID) — Settings.agent is that UUID; the per-call context
      (UpdatePrompt) and greeting (InjectAgentMessage) follow SettingsApplied.
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
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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
    from opentelemetry.trace import Status as _Status
    from opentelemetry.trace import StatusCode as _StatusCode

    _tracer = _otel_trace.get_tracer("voice-agent")
except ImportError:
    _otel_trace = None  # type: ignore[assignment]
    _Status = _StatusCode = None  # type: ignore[assignment,misc]
    _tracer = None  # type: ignore[assignment]


def _traced(span_name: str):
    """Decorator that wraps an async method in an OTel span.

    Creates a span with call_id, records exceptions (a task cancelled at teardown is
    marked ``cancelled``, not an error), and ends the span on exit. No-op when
    opentelemetry is not installed.
    """

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self, *args, **kwargs):
            if not _tracer:
                return await fn(self, *args, **kwargs)
            with _tracer.start_as_current_span(
                span_name,
                attributes={"call_id": self.parent_call_id},
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                try:
                    return await fn(self, *args, **kwargs)
                except asyncio.CancelledError:
                    span.set_attribute("cancelled", True)
                    raise
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(_Status(_StatusCode.ERROR, str(exc)))
                    raise

        return wrapper

    return decorator


TRACE_TEXT_MAX = 512  # Truncation for transcript/tool text in span attributes
EOT_MATCH_WINDOW_NS = 5_000_000_000  # An EndOfTurn this recent starts the next user turn
STT_MATCH_WINDOW_NS = 2_000_000_000  # stt_latency this recent before EOT -> the turn's stt
SAVED_CONFIG_MODEL = "saved-config"  # Model label when a reusable config's models are unknown
# LatencyReport keys (seconds) that become child spans of the turn. One llm span per LLM
# pass: ttt_tool_latency (pass that emitted a tool call) and ttt_text_latency (pass that
# produced speech); ttt_token_latency is used only when ttt_text_latency is absent.
_LATENCY_SPAN_KEYS = ("ttt_text_latency", "ttt_tool_latency", "ttt_token_latency", "tts_latency")


def _truncate(value: Any, limit: int = TRACE_TEXT_MAX) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


class _CallTrace:
    """Per-call OTel spans below ``session``: turn -> stt / llm / tts / tool.<name> / playback.

    Spans are started with explicit parents instead of the current context, because one
    turn spans several tasks (plivo_rx injects text, deepgram_rx handles Deepgram events,
    plivo_tx sends the audio). Deepgram reports durations only (``LatencyReport``, in
    seconds), so stt/llm/tts spans are back-computed: end = report arrival, start = end -
    duration. Every method is a no-op when opentelemetry is not installed.
    """

    def __init__(self, call_id: str) -> None:
        self.tracer = _tracer
        self.call_id = call_id
        self.session: Any = None
        self.llm_attributes: dict[str, Any] = {}
        self.turn: Any = None
        self.turn_start_ns = 0
        self.turn_awaiting_eot = False  # user turn started without an EndOfTurn yet
        self.playback: Any = None
        self.playback_started = False
        self.playback_chunks0 = 0
        self.pending_eot: tuple[str, int] | None = None  # (trigger, time_ns)
        self.last_stt: tuple[float, int] | None = None  # (seconds, arrival_ns)
        self.samples: list[tuple[str, float, int]] = []  # (key, seconds, arrival_ns)

    @property
    def enabled(self) -> bool:
        return self.tracer is not None

    def _context(self, parent: Any) -> Any:
        return _otel_trace.set_span_in_context(parent) if parent is not None else None

    def _span(self, name: str, parent: Any, attributes: dict[str, Any], **kwargs: Any) -> Any:
        return self.tracer.start_span(
            name, context=self._context(parent), attributes=attributes, **kwargs
        )

    # -- session --

    def bind_session(self, attributes: dict[str, Any], llm_attributes: dict[str, Any]) -> None:
        """Attach to the ``session`` span that ``@_traced("session")`` made current."""
        if not self.enabled:
            return
        self.session = _otel_trace.get_current_span()
        self.session.set_attributes(attributes)
        self.llm_attributes = llm_attributes

    def set_session_attribute(self, key: str, value: Any) -> None:
        if self.enabled and self.session is not None and value is not None:
            self.session.set_attribute(key, value)

    def event(self, name: str, attributes: dict[str, Any]) -> None:
        """Span event on the current turn, falling back to the session."""
        target = self.turn if self.turn is not None else self.session
        if self.enabled and target is not None:
            target.add_event(name, attributes)

    def error(self, code: str, description: str) -> None:
        """Deepgram ``Error``: event on turn + session, session status ERROR."""
        if not self.enabled or self.session is None:
            return
        attributes = {"code": code, "description": description}
        if self.turn is not None:
            self.turn.add_event("deepgram.error", attributes)
        self.session.add_event("deepgram.error", attributes)
        self.session.set_status(_Status(_StatusCode.ERROR, f"{code} {description}".strip()))

    def exception(self, exc: BaseException) -> None:
        if self.enabled and self.session is not None:
            self.session.record_exception(exc)
            self.session.set_status(_Status(_StatusCode.ERROR, str(exc)))

    # -- turn --

    def start_turn(self, turn: int, user_text: str, source: str) -> None:
        """Open a ``turn`` span. ``source``: greeting | audio | text (InjectUserMessage)."""
        if not self.enabled:
            return
        if self.turn is not None:
            self._finish_turn({"turn.completed": False, "turn.ended_by": "next_turn"})
        now = time.time_ns()
        start = now
        attributes: dict[str, Any] = {
            "turn": turn,
            "call_id": self.call_id,
            "turn.source": source,
            "user_text": _truncate(user_text),
        }
        pending, self.pending_eot = self.pending_eot, None
        if source != "greeting" and pending and now - pending[1] <= EOT_MATCH_WINDOW_NS:
            attributes["eot.trigger"] = pending[0]
            start = pending[1]
        self.turn = self._span("turn", self.session, attributes, start_time=start)
        self.turn_start_ns = start
        self.turn_awaiting_eot = source != "greeting" and "eot.trigger" not in attributes
        self.playback = None
        self.playback_started = False
        self.samples = [s for s in self.samples if s[2] >= start]
        stt = self.last_stt
        if source == "audio" and stt and start - STT_MATCH_WINDOW_NS <= stt[1] <= now:
            self._reported_span("stt", "stt_latency", stt[0], stt[1])

    def on_end_of_turn(self, trigger: str) -> None:
        """Deepgram ``EndOfTurn``: label the open user turn, or start time of the next one."""
        if not self.enabled:
            return
        if self.turn is not None and self.turn_awaiting_eot:
            self.turn.set_attribute("eot.trigger", trigger)
            self.turn_awaiting_eot = False
        else:
            self.pending_eot = (trigger, time.time_ns())

    def on_latency(self, key: str, seconds: float) -> None:
        if not self.enabled:
            return
        if key == "stt_latency":  # arrives ~every audio frame; keep only the latest
            self.last_stt = (seconds, time.time_ns())
        elif key in _LATENCY_SPAN_KEYS:
            self.samples.append((key, seconds, time.time_ns()))

    def _reported_span(self, name: str, key: str, seconds: float, arrival_ns: int) -> None:
        attributes: dict[str, Any] = {
            "call_id": self.call_id,
            "deepgram.timing": "reported_duration",
            "deepgram.metric": key,
            "deepgram.latency_ms": round(seconds * 1000),
        }
        if name == "llm":
            attributes.update(self.llm_attributes)
        span = self._span(name, self.turn, attributes, start_time=arrival_ns - round(seconds * 1e9))
        span.end(end_time=arrival_ns)

    def end_turn(self, attributes: dict[str, Any], barge_in: bool, tx_chunks: int) -> None:
        """``turn_complete`` was emitted: close the playback span and the turn."""
        if not self.enabled or self.turn is None:
            return
        self._finish_turn(
            {**attributes, "turn.completed": True}, interrupted=barge_in, tx_chunks=tx_chunks
        )

    def _finish_turn(
        self, attributes: dict[str, Any], interrupted: bool = False, tx_chunks: int | None = None
    ) -> None:
        keys = {s[0] for s in self.samples}
        for key, seconds, arrival in self.samples:
            if key == "tts_latency":
                self._reported_span("tts", key, seconds, arrival)
            elif key != "ttt_token_latency" or "ttt_text_latency" not in keys:
                self._reported_span("llm", key, seconds, arrival)
        self.samples = []
        self.end_playback(interrupted, attributes.get("playback_ms"), tx_chunks)
        self.turn.set_attributes({k: v for k, v in attributes.items() if v is not None})
        self.turn.end()
        self.turn = None
        self.turn_awaiting_eot = False

    # -- playback --

    def on_play_audio(self, tx_chunks: int) -> None:
        """First playAudio of the turn opens ``playback`` (until playedStream / barge-in)."""
        if not self.enabled or self.turn is None or self.playback_started:
            return
        self.playback_started = True
        self.turn_awaiting_eot = False  # a later EndOfTurn belongs to the next turn
        self.playback_chunks0 = tx_chunks
        self.playback = self._span("playback", self.turn, {"call_id": self.call_id})

    def end_playback(
        self, interrupted: bool, playback_ms: int | None, tx_chunks: int | None
    ) -> None:
        if self.playback is None:
            return
        self.playback.set_attribute("interrupted", interrupted)
        if playback_ms is not None:
            self.playback.set_attribute("playback_ms", playback_ms)
        if tx_chunks is not None:
            self.playback.set_attribute("plivo.tx_chunks", tx_chunks - self.playback_chunks0)
        self.playback.end()
        self.playback = None

    # -- tools --

    def start_tool(self, name: str, fn_id: str, arguments: Any) -> Any:
        if not self.enabled:
            return None
        parent = self.turn if self.turn is not None else self.session
        return self._span(
            f"tool.{name}",
            parent,
            {
                "call_id": self.call_id,
                "tool.name": name,
                "tool.call_id": fn_id,
                "tool.arguments": _truncate(arguments),
            },
        )

    @staticmethod
    def end_tool(span: Any, status: str, result: Any = None) -> None:
        if span is None:
            return
        span.set_attribute("tool.status", status)
        if result is not None:
            span.set_attribute("tool.result", _truncate(result))
        if status in ("error", "timeout", "cancelled"):
            span.set_status(_Status(_StatusCode.ERROR, status))
        span.end()

    # -- teardown --

    def close(self) -> None:
        """Session teardown: end any open playback/turn so no span leaks."""
        if not self.enabled or self.turn is None:
            return
        self._finish_turn(
            {"turn.completed": False, "turn.ended_by": "session_end"}, interrupted=True
        )


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
DEEPGRAM_INBOUND_AGENT_ID = os.getenv("DEEPGRAM_INBOUND_AGENT_ID", "").strip()

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

# The only prompt source: edit the file, or mount another file over it (see the README).
SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().strip()

# Deepgram speaks agent.greeting verbatim via TTS — this is literal text, not an LLM instruction.
DEFAULT_GREETING = os.getenv(
    "AGENT_GREETING",
    "Hi, this is Alex from TechFlow. I'm built with the Deepgram Voice Agent API "
    "on Plivo. How can I help you today?",
)

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
# Helpers
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


# =============================================================================
# Send-queue sentinel
# =============================================================================


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
        initial_message: str = DEFAULT_GREETING,
        stream_id: str = "",
        parent_call_id: str = "",
        sip_headers: dict[str, str] | None = None,
        hangup_callback: Callable[[], Awaitable[None]] | None = None,
        agent_config_id: str | None = None,
        saved_agent_models: dict[str, str] | None = None,
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.parent_call_id = parent_call_id or call_id
        # Per-call logger: full IDs as structured fields, same keys as the call_answered event
        self._logger = logger.bind(call_id=self.parent_call_id, leg_call_id=self.call_id)
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.sip_headers = sip_headers or {}
        self.hangup_callback = hangup_callback
        self._stream_id = stream_id  # Plivo stream ID for checkpoint/clearAudio events
        # Reusable agent configuration UUID ("" = inline Settings)
        self.agent_config_id = (
            DEEPGRAM_INBOUND_AGENT_ID if agent_config_id is None else agent_config_id
        )
        # Reusable config's models, resolved by server.py at startup (trace attributes only)
        self.saved_agent_models = saved_agent_models or {}
        self._trace = _CallTrace(self.parent_call_id)

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
        self._logger.bind(elapsed_s=elapsed, stage=stage).info(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _logv(self, stage: str, msg: str) -> None:
        """Log at 'verbose' level — detailed debugging info."""
        if LOG_LEVEL != "verbose":
            return
        elapsed = round(time.monotonic() - self._session_start, 2)
        self._logger.bind(elapsed_s=elapsed, stage=stage).debug(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
        )

    def _loge(self, stage: str, msg: str) -> None:
        """Log errors — always visible regardless of LOG_LEVEL."""
        self._error_count += 1
        elapsed = round(time.monotonic() - self._session_start, 2)
        self._logger.bind(elapsed_s=elapsed, stage=stage).error(
            f"[{self.call_id}] [{elapsed:7.2f}s] [{stage}] {msg}"
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
        """Saved mode: text appended to the saved prompt via UpdatePrompt."""
        return self._build_call_context()

    def _build_settings(self) -> dict[str, Any]:
        """Build the Deepgram Voice Agent Settings message.

        Reusable config: ``agent`` is the saved configuration's UUID (all-or-nothing: no
        inline agent fields may be mixed in). Inline: ``agent`` is the full definition —
        listen/think/speak providers, the per-call prompt, FUNCTION_DEFINITIONS and the
        greeting. The README's create command stores this inline block, minus the
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
        self._logger.info(
            f"[{self.call_id}] [  0.00s] [session] "
            f"started (from={self.from_number}, to={self.to_number}, log={LOG_LEVEL}, "
            f"settings: {self._settings_mode()})"
        )
        self._logger.bind(
            event="call_answered",
            call_id=self.parent_call_id,
            leg_call_id=self.call_id,
            from_number=self.from_number,
            to_number=self.to_number,
            sip_headers=self.sip_headers,
            stream_id=self._stream_id,
        ).info(
            f"[{self.call_id}] [  0.00s] [session] call answered (sip_headers={self.sip_headers})"
        )
        self._trace.bind_session(*self._trace_attributes())

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
            self._trace.exception(e)
        finally:
            self._running = False
            self._trace.close()
            self._emit_session_end()

    def _trace_attributes(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """(session attributes, llm attributes). A reusable config's models come from
        server.py's startup lookup, else ``saved-config`` (no per-call network calls)."""
        if self.agent_config_id:
            models = self.saved_agent_models
            listen = models.get("listen_model", SAVED_CONFIG_MODEL)
            think_provider = models.get("think_provider", SAVED_CONFIG_MODEL)
            think_model = models.get("think_model", SAVED_CONFIG_MODEL)
            speak = models.get("speak_model", SAVED_CONFIG_MODEL)
        else:
            listen, speak = DEEPGRAM_LISTEN_MODEL, DEEPGRAM_SPEAK_MODEL
            think_provider, think_model = DEEPGRAM_THINK_PROVIDER, DEEPGRAM_THINK_MODEL
        llm = {
            "gen_ai.system": "deepgram",
            "gen_ai.request.model": think_model,
            "deepgram.think.provider": think_provider,
        }
        session = {
            **llm,
            "call_id": self.parent_call_id,
            "leg_call_id": self.call_id,
            "deepgram.listen.model": listen,
            "deepgram.speak.model": speak,
            "deepgram.agent_config": self.agent_config_id or "inline",
        }
        return session, llm

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
                    self._trace.set_session_attribute("deepgram.request_id", self._request_id)
                    self._log("deepgram", f"Welcome (request_id={self._request_id})")
                    if not settings_sent:
                        await dg_ws.send(json.dumps(self._build_settings()))
                        self._last_dg_send = time.monotonic()
                        settings_sent = True
                        self._logv("deepgram", "Settings sent")
                elif etype == "SettingsApplied":
                    break
                elif etype == "Error":
                    self._trace.error(
                        str(evt.get("code", "")),
                        str(evt.get("description") or evt.get("message", "")),
                    )
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

        if self._turn_count == 0 and self.initial_message:
            self._trace.start_turn(1, "", source="greeting")  # turn 1 = the greeting
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

    @_traced("plivo_rx")
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
        self._start_user_turn(text, source="text")
        if self._dg_ws is None:
            return
        await self._dg_ws.send(json.dumps({"type": "InjectUserMessage", "content": text}))
        self._last_dg_send = time.monotonic()

    # -- deepgram_rx --

    @_traced("deepgram_rx")
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
            self._trace.set_session_attribute("deepgram.request_id", self._request_id)
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
            self._trace.on_end_of_turn(str(evt.get("trigger") or "unknown"))
            self._logv("deepgram", f"EndOfTurn (trigger={evt.get('trigger')})")
        elif etype == "LatencyReport":
            self._on_latency_report(evt)
        elif etype == "PromptUpdated":
            self._log("deepgram", "PromptUpdated")
        elif etype == "History":
            self._logv("deepgram", f"History: {str(evt)[:120]}")
        elif etype == "InjectionRefused":
            self._logger.warning(
                f"[{self.call_id}] [deepgram] InjectionRefused: {evt.get('message', '')}"
            )
        elif etype == "Warning":
            self._logger.warning(
                f"[{self.call_id}] [deepgram] Warning: "
                f"{evt.get('code', '')} {evt.get('description') or evt.get('message', '')}"
            )
            self._trace.event(
                "deepgram.warning",
                {
                    "code": str(evt.get("code", "")),
                    "description": str(evt.get("description") or evt.get("message", "")),
                },
            )
        elif etype == "Error":
            self._loge(
                "deepgram",
                f"Error: {evt.get('code', '')} {evt.get('description') or evt.get('message', '')}",
            )
            self._trace.error(
                str(evt.get("code", "")), str(evt.get("description") or evt.get("message", ""))
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
                self._trace.on_latency(key, value)
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
            self._trace.event("barge_in", {"cleared_queue_items": cleared})
            self._emit_turn_complete(barge_in=True)
            self._drop_agent_audio = True
        self._is_playing = False
        self._log(
            "barge_in" if was_playing else "deepgram",
            f"{'barge-in' if was_playing else 'UserStartedSpeaking (idle)'}: cleared={cleared}",
        )

    def _start_user_turn(self, text: str, source: str = "audio") -> None:
        """Begin a new user turn: bump the counter, reset per-turn state, emit user_text.

        ``source`` (trace attribute): ``audio`` (ConversationText user) or ``text`` (injected).
        """
        self._turn_count += 1
        self._trace.start_turn(self._turn_count, text, source=source)
        self._turn_user_text = text
        self._turn_agent_text = ""
        self._turn_latency = self._empty_latency()
        self._turn_latency_report = {}
        self._speech_end_time = time.monotonic()
        self._log("turn", f"turn {self._turn_count}: user '{text[:80]}'")
        self._logger.bind(
            event="user_text",
            call_id=self.parent_call_id,
            turn=self._turn_count,
            text=text,
        ).info(f"[{self.call_id}] user_text turn {self._turn_count}: '{text[:60]}'")

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
            self._logger.bind(
                event="agent_text",
                call_id=self.parent_call_id,
                turn=self._turn_count,
                text=content,
            ).info(f"[{self.call_id}] agent_text turn {self._turn_count}: '{content[:60]}'")

    async def _on_function_call_request(self, evt: dict[str, Any]) -> None:
        """Run client-side functions and reply with FunctionCallResponse."""
        for fn in evt.get("functions", []) or []:
            if not fn.get("client_side", False):
                self._logv("tool", f"skipping server-side function {fn.get('name')}")
                continue
            fn_id = fn.get("id", "")
            name = fn.get("name", "")
            # tool.<name> span: FunctionCallRequest -> FunctionCallResponse sent
            span = self._trace.start_tool(name, fn_id, fn.get("arguments", ""))
            status = "cancelled"
            result: dict[str, Any] | None = None
            try:
                try:
                    result = await asyncio.wait_for(
                        self._handle_function_call(name, fn.get("arguments", "")), timeout=10
                    )
                    status = "error" if "error" in result else str(result.get("status", "ok"))
                except (TimeoutError, asyncio.TimeoutError):
                    self._loge("tool", f"{name} timed out")
                    result = {"error": "function timed out"}
                    status = "timeout"
                response = {
                    "type": "FunctionCallResponse",
                    "id": fn_id,
                    "name": name,
                    "content": json.dumps(result),
                }
                if self._dg_ws is not None:
                    await self._dg_ws.send(json.dumps(response))
                    self._last_dg_send = time.monotonic()
            finally:
                self._trace.end_tool(span, status, result)

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

    @_traced("plivo_tx")
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
        self._trace.on_play_audio(self._plivo_tx_chunks)
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
        self._logger.bind(
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
            f"[{self.call_id}] turn {self._turn_count} complete{' (barge-in)' if barge_in else ''}"
        )
        self._trace.end_turn(
            {
                "turn": self._turn_count,
                "user_text": _truncate(self._turn_user_text or ""),
                "agent_text": _truncate(self._turn_agent_text or ""),
                "barge_in": barge_in,
                "playback_ms": playback_ms,
                **{f"deepgram.{k}": v for k, v in self._turn_latency_report.items()},
            },
            barge_in=barge_in,
            tx_chunks=self._plivo_tx_chunks,
        )

    def _emit_session_end(self) -> None:
        """Emit the session_end summary event (always logged, even in quiet mode)."""
        duration = round(time.monotonic() - self._session_start, 1)
        avg_ttfs = (
            round(sum(self._ttfs_samples) / len(self._ttfs_samples)) if self._ttfs_samples else None
        )
        self._logger.bind(
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
            f"[{self.call_id}] [{duration:7.1f}s] [session] "
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
    initial_message: str = DEFAULT_GREETING,
    stream_id: str = "",
    parent_call_id: str = "",
    sip_headers: dict[str, str] | None = None,
    hangup_callback: Callable[[], Awaitable[None]] | None = None,
    saved_agent_models: dict[str, str] | None = None,
) -> None:
    """Run a voice agent session for an inbound call."""
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
        saved_agent_models=saved_agent_models,
    )
    await agent.run()
