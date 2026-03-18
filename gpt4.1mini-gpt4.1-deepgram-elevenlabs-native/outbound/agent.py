"""Outbound voice agent — Dual-LLM lead qualification with smart-turn detection.

Architecture:
  Plivo audio → Deepgram nova-3 (streaming STT) + Silero VAD + smart-turn-v3
  → GPT-4.1 mini (conversation) → [tool_calls?] → GPT-4.1 (reasoning + tools)
  → ElevenLabs flash v2.5 (streaming TTS) → Plivo audio

Self-contained outbound agent with OutboundCallRecord and CallManager for
tracking outbound call lifecycle. All components (SmartTurnProcessor,
DeepgramSTT, tool implementations) are duplicated here for isolation.

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
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiohttp
import numpy as np
from dotenv import load_dotenv
from loguru import logger

from utils import (
    SileroVADProcessor,
    elevenlabs_to_plivo,
    plivo_to_deepgram,
    plivo_to_vad,
)

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# =============================================================================
# Agent Configuration
# =============================================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CONVERSATION_MODEL = os.getenv("OPENAI_CONVERSATION_MODEL", "gpt-4.1-mini")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN", "")
CAL_COM_API_KEY = os.getenv("CAL_COM_API_KEY", "")
CAL_COM_EVENT_TYPE_ID = os.getenv("CAL_COM_EVENT_TYPE_ID", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

SMART_TURN_STOP_SECS = float(os.getenv("SMART_TURN_STOP_SECS", "3.0"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "normal").lower()

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
                span_name, attributes={"call_id": self.call_id[:8]}
            ):
                return await fn(self, *args, **kwargs)

        return wrapper

    return decorator


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


SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", _OUTBOUND_PROMPT_TEMPLATE)


# =============================================================================
# Smart-Turn Processor
# =============================================================================


class SmartTurnProcessor:
    """Semantic turn detection using smart-turn-v3 ONNX model.

    Wraps Silero VAD (for is_speech signal) + smart-turn-v3 ONNX model
    (Whisper Tiny encoder + linear classifier, ~12ms CPU inference).
    Runs model in thread executor to avoid blocking the event loop.
    """

    _ONNX_MODEL_NAME = "smart-turn-v3.2-cpu.onnx"
    _HF_REPO = "pipecat-ai/smart-turn-v3"
    _MAX_AUDIO_SECS = 8
    _SAMPLE_RATE = 16000

    def __init__(self, stop_secs: float = SMART_TURN_STOP_SECS):
        self._vad = SileroVADProcessor()
        self._audio_buffer: list[np.ndarray] = []
        self._speech_active = False
        self._silence_start: float | None = None
        self._stop_secs = stop_secs
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._session = None
        self._feature_extractor = None
        self._model_loaded = False

    def _ensure_model(self) -> None:
        """Lazy-load the smart-turn-v3 ONNX model."""
        if self._model_loaded:
            return
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from transformers import WhisperFeatureExtractor

            model_path = hf_hub_download(self._HF_REPO, self._ONNX_MODEL_NAME)
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(model_path, sess_options=so)
            self._feature_extractor = WhisperFeatureExtractor(
                chunk_length=self._MAX_AUDIO_SECS
            )
            self._model_loaded = True
            logger.info("Smart-turn-v3 ONNX model loaded")
        except Exception as e:
            logger.warning(
                f"Failed to load smart-turn-v3 model: {e}. "
                "Using VAD-only fallback."
            )
            self._model_loaded = True  # Don't retry

    def process_audio(
        self, audio_f32_16k: np.ndarray
    ) -> tuple[bool, bool]:
        """Process audio frame through VAD. Returns (speech_started, speech_ended)."""
        speech_started, speech_ended = self._vad.process(audio_f32_16k)

        if speech_started:
            self._speech_active = True
            self._silence_start = None

        if self._speech_active or self._vad.is_speaking:
            self._audio_buffer.append(audio_f32_16k)

        if speech_ended:
            self._silence_start = time.monotonic()

        return speech_started, speech_ended

    async def analyze_turn(self) -> bool:
        """Run smart-turn model on accumulated audio. Returns True if complete."""
        self._ensure_model()

        if not self._audio_buffer:
            return True

        if (
            self._silence_start
            and (time.monotonic() - self._silence_start) >= self._stop_secs
        ):
            logger.debug("Smart-turn: forced complete (silence timeout)")
            return True

        if self._session is None or self._feature_extractor is None:
            return True

        audio = np.concatenate(self._audio_buffer)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor, self._predict, audio
        )
        logger.debug(
            f"Smart-turn: prediction={result['prediction']} "
            f"probability={result['probability']:.3f}"
        )
        return result["prediction"] == 1

    def _predict(self, audio_array: np.ndarray) -> dict:
        """Run ONNX model inference (called in thread). ~12ms on CPU."""
        max_samples = self._MAX_AUDIO_SECS * self._SAMPLE_RATE

        if len(audio_array) > max_samples:
            audio_array = audio_array[-max_samples:]
        elif len(audio_array) < max_samples:
            padding = max_samples - len(audio_array)
            audio_array = np.pad(
                audio_array, (padding, 0), mode="constant", constant_values=0
            )

        inputs = self._feature_extractor(
            audio_array,
            sampling_rate=self._SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=max_samples,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)
        input_features = np.expand_dims(input_features, axis=0)

        outputs = self._session.run(None, {"input_features": input_features})
        probability = outputs[0][0].item()

        return {
            "prediction": 1 if probability > 0.5 else 0,
            "probability": probability,
        }

    def reset(self) -> None:
        """Reset state for a new turn."""
        self._audio_buffer.clear()
        self._speech_active = False
        self._silence_start = None
        self._vad.reset()

    @property
    def is_speaking(self) -> bool:
        return self._vad.is_speaking


# =============================================================================
# Deepgram STT Client
# =============================================================================


class DeepgramSTT:
    """Real-time speech-to-text using Deepgram WebSocket API."""

    def __init__(self):
        self._ws = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False
        self._receive_task: asyncio.Task | None = None
        self._latest_transcript = ""
        self._transcript_parts: list[str] = []

    @property
    def latest_transcript(self) -> str:
        return " ".join(self._transcript_parts).strip()

    def clear_transcript(self) -> None:
        self._transcript_parts.clear()
        self._latest_transcript = ""

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._running = True

        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?model={DEEPGRAM_MODEL}"
            f"&encoding=linear16"
            f"&sample_rate=8000"
            f"&channels=1"
            f"&punctuate=true"
            f"&interim_results=true"
        )

        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}
        self._ws = await self._session.ws_connect(url, headers=headers)
        logger.info("Connected to Deepgram STT")
        self._receive_task = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "Results":
                        is_final = data.get("is_final", False)
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get(
                                "transcript", ""
                            )
                            if transcript.strip() and is_final:
                                self._transcript_parts.append(transcript)
                                logger.debug(f"STT final: '{transcript}'")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Deepgram WebSocket error: {msg.data}")
                    break
        except Exception as e:
            if self._running:
                logger.error(f"Deepgram receive error: {e}")

    async def send_audio(self, pcm_audio: bytes) -> None:
        if self._ws and not self._ws.closed:
            await self._ws.send_bytes(pcm_audio)

    async def close(self) -> None:
        self._running = False
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._receive_task
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()


# =============================================================================
# Tool Implementations (Real Systems)
# =============================================================================


async def lookup_contact(email_or_phone: str) -> dict[str, Any]:
    """Search contacts in HubSpot CRM by email or phone."""
    import httpx

    if not HUBSPOT_ACCESS_TOKEN:
        return {"status": "error", "message": "HubSpot not configured"}

    filter_field = "email" if "@" in email_or_phone else "phone"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://api.hubapi.com/crm/v3/objects/contacts/search",
            headers={
                "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "filterGroups": [{
                    "filters": [{
                        "propertyName": filter_field,
                        "operator": "EQ",
                        "value": email_or_phone,
                    }]
                }],
                "properties": [
                    "email", "phone", "firstname", "lastname",
                    "company", "jobtitle", "lifecyclestage",
                ],
            },
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("results", [])
    if not results:
        return {
            "status": "not_found",
            "message": f"No contact found for {email_or_phone}",
        }

    contact = results[0]
    props = contact.get("properties", {})
    return {
        "status": "found",
        "contact_id": contact.get("id"),
        "email": props.get("email", ""),
        "phone": props.get("phone", ""),
        "name": (
            f"{props.get('firstname', '')} "
            f"{props.get('lastname', '')}"
        ).strip(),
        "company": props.get("company", ""),
        "job_title": props.get("jobtitle", ""),
        "lifecycle_stage": props.get("lifecyclestage", ""),
    }


async def create_or_update_contact(
    email: str, properties: dict[str, str]
) -> dict[str, Any]:
    """Create or update a contact in HubSpot CRM."""
    import httpx

    if not HUBSPOT_ACCESS_TOKEN:
        return {"status": "error", "message": "HubSpot not configured"}

    props = {"email": email, **properties}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://api.hubapi.com/crm/v3/objects/contacts",
            headers={
                "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={"properties": props},
        )

        if response.status_code == 409:
            existing = response.json()
            contact_id = existing.get("id", "")
            if not contact_id:
                search_result = await lookup_contact(email)
                contact_id = search_result.get("contact_id", "")

            if contact_id:
                response = await client.patch(
                    f"https://api.hubapi.com/crm/v3/objects/contacts/{contact_id}",
                    headers={
                        "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                        "Content-Type": "application/json",
                    },
                    json={"properties": props},
                )
                response.raise_for_status()
                return {"status": "updated", "contact_id": contact_id}

        response.raise_for_status()
        data = response.json()
        return {"status": "created", "contact_id": data.get("id", "")}


async def score_lead(
    contact_id: str,
    company_size: str,
    budget: str,
    timeline: str,
    authority: str,
    need: str,
) -> dict[str, Any]:
    """BANT scoring + create/update Deal in HubSpot."""
    import httpx

    if not HUBSPOT_ACCESS_TOKEN:
        return {"status": "error", "message": "HubSpot not configured"}

    score = 0
    reasons = []

    budget_lower = budget.lower()
    if any(w in budget_lower for w in ["allocated", "approved", "ready", "yes"]):
        score += 25
        reasons.append("budget_confirmed")
    elif any(w in budget_lower for w in ["exploring", "considering"]):
        score += 15
        reasons.append("budget_exploring")

    authority_lower = authority.lower()
    if any(
        w in authority_lower
        for w in ["decision", "final", "yes", "i decide", "owner", "ceo"]
    ):
        score += 25
        reasons.append("decision_maker")
    elif any(
        w in authority_lower for w in ["influencer", "evaluate", "recommend"]
    ):
        score += 15
        reasons.append("influencer")

    need_lower = need.lower()
    if any(
        w in need_lower
        for w in ["urgent", "critical", "pain", "problem", "struggling"]
    ):
        score += 25
        reasons.append("strong_need")
    elif any(w in need_lower for w in ["exploring", "curious", "interested"]):
        score += 15
        reasons.append("moderate_need")

    timeline_lower = timeline.lower()
    if any(
        w in timeline_lower for w in ["now", "asap", "this month", "immediate"]
    ):
        score += 25
        reasons.append("immediate_timeline")
    elif any(
        w in timeline_lower
        for w in ["quarter", "this quarter", "soon", "next month"]
    ):
        score += 20
        reasons.append("near_timeline")
    elif any(w in timeline_lower for w in ["year", "next year", "6 months"]):
        score += 10
        reasons.append("future_timeline")

    if score >= 70:
        deal_stage = "qualifiedtobuy"
        stage_label = "Qualified to Buy"
    elif score >= 50:
        deal_stage = "presentationscheduled"
        stage_label = "Presentation Scheduled"
    else:
        deal_stage = "appointmentscheduled"
        stage_label = "Appointment Scheduled"

    async with httpx.AsyncClient(timeout=10.0) as client:
        deal_response = await client.post(
            "https://api.hubapi.com/crm/v3/objects/deals",
            headers={
                "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "properties": {
                    "dealname": f"TechFlow Lead - Score {score}",
                    "dealstage": deal_stage,
                    "pipeline": "default",
                    "amount": (
                        budget if budget.replace(".", "").isdigit() else ""
                    ),
                    "description": (
                        f"BANT Score: {score}/100\n"
                        f"Budget: {budget}\nAuthority: {authority}\n"
                        f"Need: {need}\nTimeline: {timeline}\n"
                        f"Company Size: {company_size}\n"
                        f"Scoring: {', '.join(reasons)}"
                    ),
                },
            },
        )
        deal_response.raise_for_status()
        deal_data = deal_response.json()

        deal_id = deal_data.get("id", "")
        if deal_id and contact_id:
            await client.put(
                f"https://api.hubapi.com/crm/v3/objects/deals/{deal_id}"
                f"/associations/contacts/{contact_id}/deal_to_contact",
                headers={
                    "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                },
            )

    return {
        "status": "scored",
        "score": score,
        "max_score": 100,
        "stage": stage_label,
        "reasons": reasons,
        "deal_id": deal_id,
        "qualified": score >= 50,
    }


async def schedule_meeting(
    email: str, name: str, preferred_time: str
) -> dict[str, Any]:
    """Book a demo meeting via Cal.com API."""
    import httpx

    if not CAL_COM_API_KEY:
        return {"status": "error", "message": "Cal.com not configured"}

    event_type_id = CAL_COM_EVENT_TYPE_ID or "default"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            "https://api.cal.com/v2/bookings",
            headers={
                "Authorization": f"Bearer {CAL_COM_API_KEY}",
                "Content-Type": "application/json",
                "cal-api-version": "2024-08-13",
            },
            json={
                "eventTypeId": (
                    int(event_type_id) if event_type_id.isdigit() else 0
                ),
                "attendee": {
                    "name": name,
                    "email": email,
                    "timeZone": "America/New_York",
                },
                "start": preferred_time,
            },
        )
        response.raise_for_status()
        data = response.json()

    booking = data.get("data", {})
    return {
        "status": "booked",
        "booking_id": booking.get("uid", ""),
        "start_time": booking.get("start", preferred_time),
        "meeting_url": booking.get("meetingUrl", ""),
    }


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS via Plivo API."""
    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
        return {"status": "error", "message": "Plivo SMS not configured"}

    import plivo

    client = plivo.RestClient(
        auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN
    )
    try:
        response = client.messages.create(
            src=PLIVO_PHONE_NUMBER,
            dst=phone_number,
            text=message,
        )
        msg_uuid = ""
        if isinstance(response, dict):
            msg_uuid = response.get("message_uuid", [""])[0]
        elif hasattr(response, "message_uuid"):
            uuids = response.message_uuid
            msg_uuid = uuids[0] if isinstance(uuids, list) else str(uuids)
        return {
            "status": "sent",
            "message_uuid": msg_uuid,
            "phone_number": phone_number,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


async def notify_sales(lead_summary: dict[str, Any]) -> dict[str, Any]:
    """Post lead notification to Slack channel."""
    import httpx

    if not SLACK_WEBHOOK_URL:
        return {"status": "error", "message": "Slack webhook not configured"}

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "New Qualified Lead"},
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Name:*\n{lead_summary.get('name', 'Unknown')}",
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Company:*\n"
                        f"{lead_summary.get('company', 'Unknown')}"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Score:*\n{lead_summary.get('score', 'N/A')}/100"
                    ),
                },
                {
                    "type": "mrkdwn",
                    "text": (
                        f"*Email:*\n{lead_summary.get('email', 'N/A')}"
                    ),
                },
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Notes:*\n{lead_summary.get('notes', 'No notes')}"
                ),
            },
        },
    ]

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            SLACK_WEBHOOK_URL,
            json={"blocks": blocks, "text": "New Qualified Lead"},
        )
        response.raise_for_status()

    return {"status": "notified", "channel": "sales"}


# =============================================================================
# Outbound Call Records
# =============================================================================


@dataclass
class OutboundCallRecord:
    """Tracks the state of a single outbound call."""

    call_id: str
    phone_number: str
    status: str = "initiating"
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
    outcome: str = ""


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
        with self._lock:
            return self._calls.get(call_id)

    def update_status(
        self, call_id: str, status: str, **kwargs: Any
    ) -> OutboundCallRecord | None:
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
        with self._lock:
            return [
                r
                for r in self._calls.values()
                if r.status in ("initiating", "ringing", "connected")
            ]

    def get_calls_by_campaign(
        self, campaign_id: str
    ) -> list[OutboundCallRecord]:
        with self._lock:
            return [
                r
                for r in self._calls.values()
                if r.campaign_id == campaign_id
            ]

    def reset(self) -> None:
        with self._lock:
            self._calls.clear()


# =============================================================================
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Dual-LLM voice agent for outbound calls."""

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
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message
        self.parent_call_id = parent_call_id or call_id
        self.sip_headers = sip_headers or {}

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._smart_turn = SmartTurnProcessor()
        self._deepgram = DeepgramSTT()
        self._playback_end_time: float = 0.0  # estimated Plivo playback end
        self._conversation_history: list[dict[str, Any]] = []
        self._current_tts_task: asyncio.Task | None = None
        self._turn_count = 0
        self._turn_cooldown_until = 0.0  # suppress VAD after turn commit
        self._pending_incomplete = False  # smart-turn said "incomplete", awaiting timeout
        self._barge_in_pending = False  # waiting for sustained speech before committing
        self._barge_in_start: float = 0.0
        self._session_start = time.monotonic()
        self._plivo_rx_bytes = 0
        self._plivo_tx_chunks = 0
        self._barge_in_count = 0

        # Checkpoint-based playback tracking
        self._stream_id = stream_id
        self._checkpoint_counter = 0
        self._checkpoint_sent_time: float | None = None

        # Per-turn metrics
        self._turn_llm_ms: float | None = None
        self._turn_tts_total_ms: float | None = None
        self._turn_tts_ttfb_ms: float | None = None
        self._turn_tts_chunks: int = 0
        self._turn_tts_audio_s: float = 0.0
        self._turn_text: str = ""
        self._turn_start_time: float = 0.0

    @property
    def _is_responding(self) -> bool:
        """True while Plivo is estimated to still be playing agent audio."""
        return time.monotonic() < self._playback_end_time

    def _cancel_playback(self) -> None:
        """Cancel all pending playback (barge-in)."""
        self._playback_end_time = 0.0

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
            f"[{self.call_id[:8]}] turn {self._turn_count} complete"
            f"{' (barge-in)' if barge_in else ''}"
        )

    async def _send_checkpoint(self) -> None:
        """Send a checkpoint event to Plivo for playback tracking."""
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

    def _log(self, stage: str, msg: str) -> None:
        if LOG_LEVEL == "quiet":
            return
        elapsed = time.monotonic() - self._session_start
        logger.info(f"[{self.call_id[:8]}] [{elapsed:7.2f}s] [{stage}] {msg}")

    def _logv(self, stage: str, msg: str) -> None:
        if LOG_LEVEL != "verbose":
            return
        elapsed = time.monotonic() - self._session_start
        logger.debug(f"[{self.call_id[:8]}] [{elapsed:7.2f}s] [{stage}] {msg}")

    def _loge(self, stage: str, msg: str) -> None:
        elapsed = time.monotonic() - self._session_start
        logger.error(f"[{self.call_id[:8]}] [{elapsed:7.2f}s] [{stage}] {msg}")

    def _build_routing_tool(self) -> list[dict[str, Any]]:
        """Build the single delegation tool for GPT-4.1 mini (conversation model)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_reasoning",
                    "description": (
                        "Delegate a task to the reasoning system when the "
                        "conversation requires CRM lookups, data updates, lead "
                        "scoring, meeting booking, SMS sending, sales "
                        "notifications, or ending the call. Provide a clear "
                        "description of what needs to be done and all relevant "
                        "context. Always include a spoken_filler."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": (
                                    "Clear description of the task including "
                                    "all relevant data from the conversation."
                                ),
                            },
                            "spoken_filler": {
                                "type": "string",
                                "description": (
                                    "A natural 1-3 sentence phrase to speak "
                                    "to the caller while the task is being "
                                    "processed. Should acknowledge their "
                                    "request and optionally share something "
                                    "useful."
                                ),
                            },
                        },
                        "required": ["task", "spoken_filler"],
                    },
                },
            },
        ]

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions for GPT-4.1 (reasoning model)."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup_contact",
                    "description": "Search for an existing contact in HubSpot CRM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_or_phone": {
                                "type": "string",
                                "description": "Email or phone to search for",
                            },
                        },
                        "required": ["email_or_phone"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "create_or_update_contact",
                    "description": "Create or update a contact in HubSpot CRM.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "description": "Contact email"},
                            "firstname": {"type": "string", "description": "First name"},
                            "lastname": {"type": "string", "description": "Last name"},
                            "company": {"type": "string", "description": "Company name"},
                            "jobtitle": {"type": "string", "description": "Job title"},
                            "phone": {"type": "string", "description": "Phone number"},
                        },
                        "required": ["email"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "score_lead",
                    "description": "Score the lead using BANT criteria and create a deal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contact_id": {
                                "type": "string",
                                "description": "HubSpot contact ID",
                            },
                            "company_size": {
                                "type": "string",
                                "description": "Company size",
                            },
                            "budget": {"type": "string", "description": "Budget info"},
                            "timeline": {"type": "string", "description": "Timeline"},
                            "authority": {
                                "type": "string",
                                "description": "Decision authority",
                            },
                            "need": {"type": "string", "description": "Business need"},
                        },
                        "required": [
                            "contact_id", "budget", "timeline", "authority", "need",
                        ],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_meeting",
                    "description": "Book a demo meeting via Cal.com.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email": {"type": "string", "description": "Attendee email"},
                            "name": {"type": "string", "description": "Attendee name"},
                            "preferred_time": {
                                "type": "string",
                                "description": "ISO 8601 time",
                            },
                        },
                        "required": ["email", "name", "preferred_time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "send_sms",
                    "description": "Send an SMS message via Plivo.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "phone_number": {
                                "type": "string",
                                "description": "Phone number",
                            },
                            "message": {"type": "string", "description": "SMS content"},
                        },
                        "required": ["phone_number", "message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "notify_sales",
                    "description": "Post to Slack about a qualified lead.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Lead name"},
                            "company": {"type": "string", "description": "Company"},
                            "email": {"type": "string", "description": "Email"},
                            "score": {"type": "integer", "description": "Score 0-100"},
                            "notes": {"type": "string", "description": "Summary"},
                        },
                        "required": ["name", "company", "score"],
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
                            "outcome": {"type": "string", "description": "Call outcome"},
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

        self._log("tool", f"calling {name}({json.dumps(args)[:100]})")

        try:
            if name == "lookup_contact":
                result = await lookup_contact(
                    email_or_phone=args.get("email_or_phone", ""),
                )
            elif name == "create_or_update_contact":
                email = args.pop("email", "")
                result = await create_or_update_contact(
                    email=email, properties=args,
                )
            elif name == "score_lead":
                result = await score_lead(
                    contact_id=args.get("contact_id", ""),
                    company_size=args.get("company_size", ""),
                    budget=args.get("budget", ""),
                    timeline=args.get("timeline", ""),
                    authority=args.get("authority", ""),
                    need=args.get("need", ""),
                )
            elif name == "schedule_meeting":
                result = await schedule_meeting(
                    email=args.get("email", ""),
                    name=args.get("name", ""),
                    preferred_time=args.get("preferred_time", ""),
                )
            elif name == "send_sms":
                result = await send_sms(
                    phone_number=args.get("phone_number", ""),
                    message=args.get("message", ""),
                )
            elif name == "notify_sales":
                result = await notify_sales(lead_summary=args)
            elif name == "end_call":
                self._log("tool", f"end_call: {args.get('reason')}")
                self._running = False
                result = {
                    "status": "call_ending",
                    "reason": args.get("reason", ""),
                }
            else:
                result = {"error": f"Unknown function: {name}"}

            self._log("tool", f"{name} → {result.get('status', 'done')}")
            return result

        except Exception as e:
            self._loge("tool", f"{name} ERROR: {e}")
            return {"error": str(e)}

    async def _call_openai(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Make an OpenAI chat completion request."""
        import httpx

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
        }
        if tools:
            payload["tools"] = tools

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()
            return response.json()

    @_traced("llm")
    async def _generate_llm_response(self, user_text: str) -> str:
        """Dual-LLM: mini controls conversation, GPT-4.1 reasons + tools.

        Flow:
        1. Mini handles every turn with `delegate_to_reasoning`.
           Responds conversationally or delegates with spoken_filler.
        2. If delegated: filler streams via TTS, GPT-4.1 reasons and
           executes tools concurrently, returns structured JSON.
        3. Mini gets a second call with results and crafts the response.
        Mini owns the voice — heavy model never speaks directly.
        """
        self._conversation_history.append(
            {"role": "user", "content": user_text}
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            *self._conversation_history,
        ]

        self._logv(
            "llm",
            f"request to {OPENAI_CONVERSATION_MODEL} ({len(messages)} msgs)",
        )
        t0 = time.monotonic()

        try:
            # Step 1: GPT-4.1 mini with delegation tool only
            result = await self._call_openai(
                OPENAI_CONVERSATION_MODEL,
                messages,
                tools=self._build_routing_tool(),
            )
            message = result["choices"][0]["message"]
            latency = (time.monotonic() - t0) * 1000
            self._turn_llm_ms = round(latency)
            tokens = result.get("usage", {})

            if _otel_trace:
                span = _otel_trace.get_current_span()
                span.set_attribute("llm.latency_ms", latency)
                span.set_attribute("gen_ai.request.model", OPENAI_CONVERSATION_MODEL)
                span.set_attribute(
                    "gen_ai.usage.prompt_tokens", tokens.get("prompt_tokens", 0)
                )
                span.set_attribute(
                    "gen_ai.usage.completion_tokens", tokens.get("completion_tokens", 0)
                )

            if not message.get("tool_calls"):
                # Mini responded directly — fast conversational turn
                self._log(
                    "llm",
                    f"{OPENAI_CONVERSATION_MODEL} ({latency:.0f}ms, "
                    f"{tokens.get('prompt_tokens', '?')}"
                    f"→{tokens.get('completion_tokens', '?')} tok): "
                    f"'{message.get('content', '')[:80]}'",
                )
                assistant_text = message.get("content", "")
                self._conversation_history.append({
                    "role": "assistant", "content": assistant_text,
                })
                return assistant_text

            # Mini decided to delegate — extract task + filler
            tc = message["tool_calls"][0]
            task_args = json.loads(
                tc["function"]["arguments"]
            ) if tc["function"].get("arguments") else {}
            task_desc = task_args.get("task", "")
            spoken_filler = task_args.get(
                "spoken_filler", "One moment please.",
            )
            self._log(
                "llm",
                f"{OPENAI_CONVERSATION_MODEL} → delegate "
                f"({latency:.0f}ms): '{task_desc[:80]}'",
            )
            self._logv("llm", f"filler: '{spoken_filler[:60]}'")

            # Record delegation in conversation history
            self._conversation_history.append(message)
            self._conversation_history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({
                    "status": "delegated", "task": task_desc,
                }),
            })

            # Step 2: Stream filler TTS + run reasoning concurrently
            reasoning_task = asyncio.create_task(
                self._run_reasoning_loop(task_desc),
                name="reasoning",
            )

            # Stream filler while reasoning works
            await self._synthesize_with_elevenlabs(spoken_filler)

            # Step 3: Single patience fallback — 5s after filler
            if not reasoning_task.done():
                try:
                    await asyncio.wait_for(
                        asyncio.shield(reasoning_task), timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    if not reasoning_task.done():
                        patience = (
                            "Still pulling up your details, one moment."
                        )
                        self._log(
                            "llm", f"patience fallback: '{patience}'",
                        )
                        await self._synthesize_with_elevenlabs(patience)

            # Await reasoning result
            reasoning_result = await reasoning_task
            if isinstance(reasoning_result, BaseException):
                raise reasoning_result

            # Step 4: Mini crafts spoken response from results
            self._log(
                "llm",
                f"reasoning returned: "
                f"{json.dumps(reasoning_result)[:120]}",
            )
            self._conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "reasoning_result",
                    "type": "function",
                    "function": {
                        "name": "reasoning_result",
                        "arguments": json.dumps(reasoning_result),
                    },
                }],
            })
            self._conversation_history.append({
                "role": "tool",
                "tool_call_id": "reasoning_result",
                "content": json.dumps(reasoning_result),
            })

            followup_messages: list[dict[str, Any]] = [
                {"role": "system", "content": self.system_prompt},
                *self._conversation_history,
                {
                    "role": "system",
                    "content": (
                        "The reasoning system has completed the delegated "
                        "task. The results are in the tool response above. "
                        "Now craft a natural spoken response for the caller "
                        "based on these results. Keep it conversational and "
                        "concise (1-3 sentences). Do NOT mention the tools "
                        "or delegation — speak as if you did it."
                    ),
                },
            ]

            self._log(
                "llm",
                f"mini follow-up → {OPENAI_CONVERSATION_MODEL}",
            )
            t2 = time.monotonic()
            followup_result = await self._call_openai(
                OPENAI_CONVERSATION_MODEL, followup_messages, tools=None,
            )
            followup_msg = followup_result["choices"][0]["message"]
            latency2 = (time.monotonic() - t2) * 1000
            assistant_text = followup_msg.get("content", "")
            self._log(
                "llm",
                f"mini follow-up ({latency2:.0f}ms): "
                f"'{assistant_text[:80]}'",
            )

            self._conversation_history.append({
                "role": "assistant", "content": assistant_text,
            })
            return assistant_text

        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            self._loge("llm", f"ERROR ({latency:.0f}ms): {e}")
            return (
                "I'm sorry, I'm having trouble processing that right now. "
                "Could you repeat that?"
            )

    async def _run_reasoning_loop(
        self, task_desc: str,
    ) -> dict[str, Any]:
        """Run GPT-4.1 in a tool-execution loop, return structured JSON.

        The heavy model executes tools and returns a summary. Mini (the
        conversation model) crafts the spoken response.
        """
        reasoning_system = (
            "You are the reasoning and tool-execution system for a voice "
            "agent. You receive delegated tasks from the conversation "
            "model. Use the available tools to complete the task. When "
            "done, respond with a JSON object summarizing the results. "
            "Include:\n"
            '- "status": "success" or "error"\n'
            '- "actions_taken": list of tool calls and outcomes\n'
            '- "data": any relevant data retrieved\n'
            '- "summary": brief text summary of what happened\n\n'
            "IMPORTANT: Return ONLY valid JSON. Do NOT write spoken "
            "text.\n\n"
            f"Delegated task: {task_desc}"
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": reasoning_system},
            *self._conversation_history,
        ]

        max_rounds = 5
        for round_num in range(max_rounds):
            self._log(
                "llm",
                f"reasoning round {round_num + 1} "
                f"→ {OPENAI_REASONING_MODEL}",
            )
            t1 = time.monotonic()
            result = await self._call_openai(
                OPENAI_REASONING_MODEL,
                messages,
                tools=self._build_tools(),
            )
            message = result["choices"][0]["message"]
            latency = (time.monotonic() - t1) * 1000

            if not message.get("tool_calls"):
                self._log(
                    "llm",
                    f"{OPENAI_REASONING_MODEL} done ({latency:.0f}ms)",
                )
                raw = message.get("content", "{}")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    return {"status": "success", "summary": raw}

            tool_names = [
                tc["function"]["name"]
                for tc in message["tool_calls"]
            ]
            self._log(
                "llm",
                f"{OPENAI_REASONING_MODEL} → tools "
                f"({latency:.0f}ms): {tool_names}",
            )
            messages.append(message)

            for tool_call in message["tool_calls"]:
                fn_name = tool_call["function"]["name"]
                fn_args = tool_call["function"]["arguments"]
                fn_result = await self._handle_function_call(
                    fn_name, fn_args,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(fn_result),
                })

        # Exhausted max rounds
        self._log("llm", "reasoning max rounds, final response")
        result = await self._call_openai(
            OPENAI_REASONING_MODEL, messages, tools=None,
        )
        raw = result["choices"][0]["message"].get("content", "{}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"status": "success", "summary": raw}

    @_traced("tts")
    async def _synthesize_with_elevenlabs(self, text: str) -> None:
        """Stream text through ElevenLabs TTS and queue audio for Plivo."""
        import httpx

        if not text.strip():
            return

        self._logv("tts", f"requesting synthesis ({len(text)} chars)")
        t0 = time.monotonic()

        try:
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech"
                f"/{ELEVENLABS_VOICE_ID}/stream"
                f"?output_format=pcm_24000"
            )
            async with httpx.AsyncClient(timeout=30.0) as client, client.stream(
                "POST",
                url,
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": ELEVENLABS_MODEL_ID,
                },
            ) as response:
                response.raise_for_status()
                first_chunk_time = None
                chunk_count = 0
                total_bytes = 0
                async for chunk in response.aiter_bytes(chunk_size=4800):
                    if not self._running:
                        self._log(
                            "tts",
                            f"interrupted after {chunk_count} chunks (session ended)",
                        )
                        break
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                        ttfb = (first_chunk_time - t0) * 1000
                        self._logv("tts", f"first chunk (TTFB: {ttfb:.0f}ms)")
                    plivo_audio = elevenlabs_to_plivo(chunk)
                    await self._send_queue.put(plivo_audio)
                    chunk_count += 1
                    total_bytes += len(plivo_audio)

                total_time = (time.monotonic() - t0) * 1000
                audio_duration = total_bytes / 8000
                ttfb_str = ""
                if first_chunk_time is not None:
                    ttfb_str = (
                        f"TTFB={(first_chunk_time - t0) * 1000:.0f}ms, "
                    )
                self._log(
                    "tts",
                    f"done: {chunk_count} chunks, "
                    f"{ttfb_str}{audio_duration:.1f}s audio "
                    f"in {total_time:.0f}ms",
                )

                # Store per-turn TTS metrics
                self._turn_tts_total_ms = round(total_time)
                self._turn_tts_ttfb_ms = (
                    round((first_chunk_time - t0) * 1000) if first_chunk_time else None
                )
                self._turn_tts_chunks = chunk_count
                self._turn_tts_audio_s = round(audio_duration, 2)

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

You can use the caller's phone number for SMS or lookups without asking."""

        return system_prompt

    async def run(self) -> None:
        """Run the voice bot session."""
        self._session_start = time.monotonic()
        self._running = True
        self.system_prompt = self._build_system_prompt()
        logger.info(
            f"[{self.call_id[:8]}] [  0.00s] [session] "
            f"started (from={self.from_number}, to={self.to_number}, "
            f"log={LOG_LEVEL})"
        )
        logger.bind(
            event="call_answered",
            call_id=self.parent_call_id,
            leg_call_id=self.call_id,
            from_number=self.from_number,
            to_number=self.to_number,
            sip_headers=self.sip_headers,
        ).info(
            f"[{self.call_id[:8]}] [  0.00s] [session] "
            f"call answered (sip_headers={self.sip_headers})"
        )

        try:
            await self._deepgram.connect()
        except Exception as e:
            self._loge("session", f"Deepgram connect ERROR: {e}")

        try:
            self._turn_count += 1
            self._log("turn", f"turn {self._turn_count}: generating greeting")
            greeting = await self._generate_llm_response(self.initial_message)
            if greeting:
                await self._synthesize_with_elevenlabs(greeting)
                self._log(
                    "turn",
                    f"turn {self._turn_count}: greeting queued for playback",
                )
        except Exception as e:
            self._loge("session", f"greeting ERROR: {e}")

        self._log("session", "starting streaming tasks (plivo_rx, plivo_tx)")
        try:
            await self._run_streaming_tasks()
        except Exception as e:
            self._loge("session", f"streaming ERROR: {e}")
        finally:
            self._running = False
            await self._deepgram.close()
            duration = time.monotonic() - self._session_start
            logger.bind(
                event="call_summary",
                call_id=self.parent_call_id,
                leg_call_id=self.call_id,
                duration_s=round(duration, 1),
                turns=self._turn_count,
                barge_ins=self._barge_in_count,
                errors=0,
                rx_bytes=self._plivo_rx_bytes,
                tx_chunks=self._plivo_tx_chunks,
            ).info(
                f"[{self.call_id[:8]}] [{duration:7.1f}s] [session] "
                f"ended — {self._turn_count} turns, "
                f"{self._barge_in_count} barge-ins, "
                f"rx={self._plivo_rx_bytes}B, tx={self._plivo_tx_chunks} chunks"
            )

    async def _run_streaming_tasks(self) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(
                self._receive_from_plivo(), name="plivo_rx"
            ),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                if task.exception():
                    self._loge(
                        "session",
                        f"task {task.get_name()} failed: {task.exception()}",
                    )
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, stream to Deepgram, run smart-turn detection."""
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
                            self._logv(
                                "plivo_rx",
                                f"{media_count} packets, responding={self._is_responding}",
                            )

                        # Forward to Deepgram (continuous streaming)
                        pcm_audio = plivo_to_deepgram(mulaw_audio)
                        await self._deepgram.send_audio(pcm_audio)

                        # Run VAD + accumulate for smart-turn
                        vad_audio = plivo_to_vad(mulaw_audio)
                        # Skip VAD during post-turn cooldown (avoids ghost triggers
                        # from residual audio in the pipeline after turn commit)
                        if time.monotonic() < self._turn_cooldown_until:
                            continue

                        speech_started, speech_ended = self._smart_turn.process_audio(vad_audio)

                        # Barge-in
                        if speech_started:
                            self._pending_incomplete = False  # new speech cancels pending
                            self._log("vad", "speech START detected")
                            if self._is_responding:
                                # Track barge-in start — require sustained speech
                                # (500ms) before committing to barge-in. Phone echo
                                # and noise can sustain 200-400ms at borderline VAD
                                # probabilities (0.50-0.65). Real barge-in speech
                                # (user intentionally interrupting) is typically 500ms+.
                                self._barge_in_pending = True
                                self._barge_in_start = time.monotonic()
                                self._logv(
                                    "vad",
                                    "potential barge-in — waiting for sustained speech",
                                )

                        # Sustained barge-in confirmation: if pending barge-in
                        # and speech has continued for 500ms, commit it.
                        # If speech ended too quickly, it was noise — cancel.
                        if self._barge_in_pending:
                            if speech_ended:
                                # Speech ended before 500ms — false trigger
                                elapsed = time.monotonic() - self._barge_in_start
                                self._logv(
                                    "vad",
                                    f"barge-in cancelled — speech ended after "
                                    f"{elapsed * 1000:.0f}ms (too short)",
                                )
                                self._barge_in_pending = False
                            elif (
                                self._smart_turn.is_speaking
                                and (time.monotonic() - self._barge_in_start) >= 0.5
                            ):
                                # Sustained speech confirmed — commit barge-in
                                self._barge_in_pending = False
                                self._log(
                                    "vad",
                                    "barge-in confirmed (sustained speech >500ms)",
                                )
                                self._cancel_playback()
                                if (
                                    self._current_tts_task
                                    and not self._current_tts_task.done()
                                ):
                                    self._current_tts_task.cancel()
                                cleared = 0
                                while not self._send_queue.empty():
                                    try:
                                        self._send_queue.get_nowait()
                                        cleared += 1
                                    except asyncio.QueueEmpty:
                                        break
                                self._logv(
                                    "vad", f"cleared {cleared} queued audio chunks",
                                )
                                await self.websocket.send_text(
                                    json.dumps({"event": "clearAudio"})
                                )
                                self._logv("plivo_tx", "sent clearAudio event")
                                self._barge_in_count += 1
                                self._emit_turn_complete(barge_in=True)

                        # Turn detection: when VAD detects silence after speech
                        if speech_ended and not self._barge_in_pending:
                            self._log("vad", "speech END — running smart-turn analysis")
                            turn_complete = await self._smart_turn.analyze_turn()
                            if turn_complete:
                                self._pending_incomplete = False
                                transcript = self._deepgram.latest_transcript
                                if transcript.strip():
                                    self._turn_count += 1
                                    self._log(
                                        "smart_turn",
                                        f"turn {self._turn_count} complete: "
                                        f"'{transcript[:80]}'",
                                    )
                                    task = asyncio.create_task(
                                        self._process_text_turn(transcript),
                                        name=f"turn_{self._turn_count}",
                                    )
                                    task.add_done_callback(
                                        lambda t: t.exception()
                                        if not t.cancelled()
                                        else None
                                    )
                                    self._smart_turn.reset()
                                    self._deepgram.clear_transcript()
                                    # Cooldown: suppress VAD for 300ms to avoid
                                    # ghost triggers from residual pipeline audio
                                    self._turn_cooldown_until = time.monotonic() + 0.3
                                else:
                                    # Empty transcript — Deepgram hasn't finalized
                                    # yet. DON'T reset or clear: let the transcript
                                    # accumulate. The next speech_ended or silence
                                    # timeout will pick it up with the full text.
                                    self._log(
                                        "smart_turn",
                                        "complete but empty transcript — "
                                        "keeping buffer, waiting for Deepgram",
                                    )
                                    self._pending_incomplete = True
                            else:
                                self._log(
                                    "smart_turn",
                                    "incomplete turn, waiting for more speech",
                                )
                                self._pending_incomplete = True

                        # Silence timeout re-check: if smart-turn previously said
                        # "incomplete" and user has been silent long enough, force
                        # the turn complete. This prevents infinite silence when
                        # smart-turn misjudges a complete utterance as incomplete.
                        elif (
                            self._pending_incomplete
                            and self._smart_turn._silence_start is not None
                            and not self._smart_turn.is_speaking
                            and (time.monotonic() - self._smart_turn._silence_start)
                            >= self._smart_turn._stop_secs
                        ):
                            self._pending_incomplete = False
                            transcript = self._deepgram.latest_transcript
                            if transcript.strip():
                                self._turn_count += 1
                                self._log(
                                    "smart_turn",
                                    f"turn {self._turn_count} (silence timeout): "
                                    f"'{transcript[:80]}'",
                                )
                                task = asyncio.create_task(
                                    self._process_text_turn(transcript),
                                    name=f"turn_{self._turn_count}",
                                )
                                task.add_done_callback(
                                    lambda t: t.exception()
                                    if not t.cancelled()
                                    else None
                                )
                            else:
                                self._logv(
                                    "smart_turn",
                                    "silence timeout but empty transcript",
                                )
                            self._smart_turn.reset()
                            self._deepgram.clear_transcript()
                            self._turn_cooldown_until = time.monotonic() + 0.3

                elif event == "playedStream":
                    name = message.get("name", "")
                    self._cancel_playback()
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

                elif event == "stop":
                    self._log("plivo_rx", "received stop event — call ended")
                    break

        except Exception as e:
            if "1000" not in str(e):
                self._loge("plivo_rx", f"ERROR: {e}")
        finally:
            self._logv("plivo_rx", f"exiting — received {media_count} media packets")

    async def _process_text_turn(self, text: str) -> None:
        """Process a text-based turn: LLM → TTS."""
        # Reset per-turn metrics
        self._turn_llm_ms = None
        self._turn_tts_total_ms = None
        self._turn_tts_ttfb_ms = None
        self._turn_tts_chunks = None
        self._turn_tts_audio_s = None
        self._turn_text = text
        self._turn_start_time = time.monotonic()
        try:
            response_text = await self._generate_llm_response(text)
            if not response_text.strip():
                self._logv("turn", "empty LLM response, skipping TTS")
                return

            self._current_tts_task = asyncio.create_task(
                self._synthesize_with_elevenlabs(response_text),
                name="tts_synthesis",
            )
            await self._current_tts_task
            await self._send_checkpoint()

        except asyncio.CancelledError:
            self._log("turn", "TTS cancelled (barge-in)")
        except Exception as e:
            self._loge("turn", f"text turn ERROR: {e}")

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks.

        Tracks estimated playback end time: each 160-byte chunk = 20ms of audio.
        Plivo buffers and plays chunks, so _is_responding (property) checks
        whether estimated playback is still in progress.
        """
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

                        # Extend estimated playback end by 20ms per chunk
                        now = time.monotonic()
                        self._playback_end_time = max(
                            self._playback_end_time, now
                        ) + 0.02

                        if self._plivo_tx_chunks == 1:
                            self._log("plivo_tx", "first audio chunk sent to Plivo")
                        if self._plivo_tx_chunks % 500 == 0:
                            remaining = self._playback_end_time - now
                            self._logv(
                                "plivo_tx",
                                f"{self._plivo_tx_chunks} chunks sent, "
                                f"~{remaining:.1f}s playback remaining",
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
