"""Standalone FastAPI server for inbound calls via Vapi orchestrator.

Vapi handles the entire voice pipeline (Deepgram STT -> GPT-4.1 -> ElevenLabs TTS)
and connects to Plivo via SIP trunking. This server receives webhook events from
Vapi for dynamic assistant configuration, tool execution, and call lifecycle events.

Architecture:
    Caller -> Plivo (SIP) -> Vapi (orchestrator) -> This Server (webhooks)
"""

from __future__ import annotations

import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import Response
from loguru import logger

from inbound.agent import (
    GPT_MODEL,
    build_assistant_config,
    handle_tool_calls,
)
from utils import normalize_phone_number

load_dotenv()

# Server configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

app = FastAPI(
    title="GPT-4.1 Voice Agent via Vapi (Inbound)",
    description="Inbound voice agent using Vapi with GPT-4.1, Deepgram, ElevenLabs, and Plivo",
    version="0.1.0",
)


# =============================================================================
# Routes
# =============================================================================


@app.get("/test-answer")
async def test_answer() -> Response:
    """Answer endpoint for E2E test calls — keeps the call alive with <Wait>.

    Used by the test number's Plivo app to answer outbound calls from Vapi
    so the agent can speak its greeting and we can verify the audio pipeline.
    """
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Response>"
        '<Record maxLength="30" recordSession="true" redirect="false" />'
        "<Wait length=\"30\" />"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt4.1-deepgram-elevenlabs-vapi-inbound",
        "model": GPT_MODEL,
        "orchestrator": "vapi",
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> dict:
    """Main Vapi webhook endpoint.

    Handles all Vapi server events:
    - assistant-request: Return dynamic assistant config for inbound calls
    - tool-calls: Execute tools and return results
    - status-update: Log call status changes
    - end-of-call-report: Log call completion details
    - conversation-update: Real-time transcript updates
    """
    body = await request.json()
    message = body.get("message", {})
    message_type = message.get("type", "")

    if message_type == "assistant-request":
        # Vapi is asking for assistant configuration for an inbound call
        call = message.get("call", {})
        from_number = call.get("customer", {}).get("number", "")
        logger.info(f"Assistant request for inbound call from {from_number}")

        server_url = f"{PUBLIC_URL}/vapi/webhook" if PUBLIC_URL else ""
        assistant_config = build_assistant_config(
            server_url=server_url,
            from_number=from_number,
        )

        return {"assistant": assistant_config}

    if message_type == "tool-calls":
        logger.info("Processing tool calls from Vapi")
        results = await handle_tool_calls(message)
        return {"results": results}

    if message_type == "status-update":
        status = message.get("status", "")
        call = message.get("call", {})
        call_id = call.get("id", "")
        logger.info(f"Call status update: call_id={call_id}, status={status}")
        return {"ok": True}

    if message_type == "end-of-call-report":
        call = message.get("call", {})
        call_id = call.get("id", "")
        duration = message.get("durationSeconds", 0)
        ended_reason = message.get("endedReason", "")
        summary = message.get("summary", "")
        logger.info(
            f"Call ended: call_id={call_id}, duration={duration}s, "
            f"reason={ended_reason}, summary={summary[:100]}"
        )
        return {"ok": True}

    if message_type == "conversation-update":
        # Real-time transcript — useful for logging or analytics
        return {"ok": True}

    if message_type == "speech-update":
        return {"ok": True}

    if message_type == "hang":
        logger.info("Call hangup notification received")
        return {"ok": True}

    logger.debug(f"Unhandled Vapi event: {message_type}")
    return {"ok": True}


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the inbound server."""
    logger.info(f"Starting GPT-4.1 Vapi Inbound Voice Agent on port {SERVER_PORT}")

    if PUBLIC_URL:
        logger.info(f"Vapi webhook URL: {PUBLIC_URL}/vapi/webhook")
        logger.info(
            "Configure this URL as your Vapi phone number's Server URL, "
            "or set it as the assistant's serverUrl."
        )
    else:
        logger.warning(
            "PUBLIC_URL not set. Use ngrok or similar to expose this server, "
            "then set PUBLIC_URL and restart."
        )

    if PLIVO_PHONE_NUMBER:
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        logger.info(f"Plivo number: +{phone}")
        logger.info(
            "Ensure your Plivo number is connected to Vapi via SIP trunk. "
            "See README.md for setup instructions."
        )

    uvicorn.run("inbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
