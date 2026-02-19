"""Outbound voice agent — Vapi assistant config + call state management.

Uses Vapi's API to initiate outbound calls with:
- Deepgram Nova-3 for speech-to-text
- OpenAI GPT-4.1 for conversation intelligence
- ElevenLabs for text-to-speech
- Server-side VAD and turn detection

Status state machine:
    initiating -> ringing -> connected -> completed
                         |-> no_answer
                |-> failed
"""

from __future__ import annotations

import json
import os
import random
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Agent configuration
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4.1")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-3")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_flash_v2_5")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID", "")

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
    vapi_call_id: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: datetime | None = None
    ended_at: datetime | None = None
    duration: int = 0
    ended_reason: str = ""
    outcome: str = ""  # success|no_answer|busy|failed


def determine_outcome(ended_reason: str, duration: int) -> str:
    """Map Vapi ended reason and duration to a high-level outcome."""
    reason = ended_reason.lower() if ended_reason else ""

    if "no-answer" in reason or "customer-did-not-answer" in reason:
        return "no_answer"
    if "busy" in reason or "rejected" in reason:
        return "busy"
    if "error" in reason or "failed" in reason:
        return "failed"

    # If the call had meaningful duration, consider it success
    if duration > 0 or "customer" in reason or "assistant" in reason:
        return "success"

    return "failed"


class CallManager:
    """Thread-safe manager for outbound call records."""

    def __init__(self) -> None:
        self._calls: dict[str, OutboundCallRecord] = {}
        self._vapi_to_call: dict[str, str] = {}  # vapi_call_id -> call_id
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
                "Hi, this is Alex from TechFlow. I'm built with OpenAI GPT-4.1, "
                "Deepgram, ElevenLabs, and Vapi orchestration with Plivo telephony. "
                f"I'm reaching out because {opening_reason}. "
                "Is now a good time for a quick chat?"
            )
        else:
            initial_message = (
                "Hi, this is Alex from TechFlow. I'm built with OpenAI GPT-4.1, "
                "Deepgram, ElevenLabs, and Vapi orchestration with Plivo telephony. "
                "I'm reaching out about your recent sign-up. "
                "Is now a good time for a quick chat?"
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

    def get_call_by_vapi_id(self, vapi_call_id: str) -> OutboundCallRecord | None:
        """Look up a call by its Vapi call ID."""
        with self._lock:
            call_id = self._vapi_to_call.get(vapi_call_id)
            if call_id:
                return self._calls.get(call_id)
            return None

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
            # Maintain vapi_call_id reverse index
            if kwargs.get("vapi_call_id"):
                self._vapi_to_call[kwargs["vapi_call_id"]] = call_id
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
            self._vapi_to_call.clear()


# =============================================================================
# Tool Functions
# =============================================================================


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
# Webhook Handling
# =============================================================================

TOOL_HANDLERS = {
    "send_sms": send_sms,
    "schedule_callback": schedule_callback,
    "transfer_call": transfer_call,
}


async def handle_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute tool calls from Vapi and return results."""
    tool_call_list = message.get("toolCallList", [])
    results = []

    for tool_call in tool_call_list:
        function_info = tool_call.get("function", {})
        name = function_info.get("name", "")
        tool_call_id = tool_call.get("id", "")

        try:
            arguments = function_info.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            arguments = {}

        logger.info(f"Tool call: {name} with args: {arguments}")

        handler = TOOL_HANDLERS.get(name)
        if handler:
            result = await handler(**arguments)
        elif name == "end_call":
            logger.info(f"Ending call: {arguments.get('reason')}")
            result = {"status": "call_ending", "reason": arguments.get("reason", "")}
        else:
            result = {"error": f"Unknown function: {name}"}

        results.append({"toolCallId": tool_call_id, "result": json.dumps(result)})

    return results


def build_outbound_assistant_config(
    server_url: str,
    record: OutboundCallRecord,
) -> dict[str, Any]:
    """Build a transient Vapi assistant configuration for outbound calls."""
    tool_definitions = [
        {
            "type": "function",
            "function": {
                "name": "send_sms",
                "description": "Send a text message to the prospect's phone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string", "description": "Phone number"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["phone_number", "message"],
                },
            },
            "server": {"url": None},
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
                        "reason": {"type": "string", "description": "Why callback is needed"},
                        "preferred_time": {"type": "string", "description": "Preferred time"},
                        "department": {"type": "string", "description": "Department"},
                    },
                    "required": ["phone_number", "reason", "department"],
                },
            },
            "server": {"url": None},
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
            "server": {"url": None},
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
                    },
                },
            },
            "async": False,
            "server": {"url": None},
        },
    ]

    return {
        "firstMessage": record.initial_message,
        "transcriber": {
            "provider": "deepgram",
            "model": DEEPGRAM_MODEL,
            "language": "en",
        },
        "model": {
            "provider": "openai",
            "model": GPT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": record.system_prompt,
                }
            ],
            "temperature": 0.7,
            "tools": tool_definitions,
        },
        "voice": {
            "provider": "11labs",
            "voiceId": ELEVENLABS_VOICE_ID,
            "model": ELEVENLABS_MODEL,
            "stability": 0.5,
            "similarityBoost": 0.75,
        },
        "serverUrl": server_url,
        "recordingEnabled": True,
        "backgroundDenoisingEnabled": True,
        # VAD and turn detection configuration
        "stopSpeakingPlan": {
            "numWords": 2,
            "voiceSeconds": 0.2,
            "backoffSeconds": 1.0,
        },
        "startSpeakingPlan": {
            "waitSeconds": 0.4,
            "transcriptionEndpointingPlan": {
                "onPunctuationSeconds": 0.1,
                "onNoPunctuationSeconds": 1.5,
                "onNumberSeconds": 0.5,
            },
        },
    }


async def initiate_outbound_call(
    record: OutboundCallRecord,
    server_url: str,
) -> str:
    """Initiate an outbound call via Vapi API.

    Returns the Vapi call ID.
    """
    if not VAPI_PRIVATE_KEY:
        raise ValueError("VAPI_PRIVATE_KEY is required for outbound calls")

    if not VAPI_PHONE_NUMBER_ID:
        raise ValueError(
            "VAPI_PHONE_NUMBER_ID is required. Import your Plivo number into Vapi first."
        )

    from vapi import AsyncVapi

    client = AsyncVapi(token=VAPI_PRIVATE_KEY)

    assistant_config = build_outbound_assistant_config(server_url, record)

    call = await client.calls.create(
        assistant=assistant_config,
        phone_number_id=VAPI_PHONE_NUMBER_ID,
        customer={"number": record.phone_number, "numberE164CheckEnabled": True},
    )

    logger.info(f"Vapi outbound call created: {call.id}")
    return call.id
