"""Standalone FastAPI server for outbound calls via Vapi orchestrator.

Vapi handles the entire voice pipeline and connects to Plivo via SIP trunking.
This server manages outbound call initiation via Vapi API and handles webhook
events for tool execution and call lifecycle tracking.

Architecture:
    This Server -> Vapi API (initiate) -> Plivo (SIP) -> Callee
    Callee -> Plivo (SIP) -> Vapi (orchestrator) -> This Server (webhooks)
"""

from __future__ import annotations

import os
from datetime import datetime

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from loguru import logger

from outbound.agent import (
    GPT_MODEL,
    CallManager,
    determine_outcome,
    handle_tool_calls,
    initiate_outbound_call,
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
    title="GPT-4.1 Voice Agent via Vapi (Outbound)",
    description="Outbound voice agent using Vapi with GPT-4.1, Deepgram, ElevenLabs, and Plivo",
    version="0.1.0",
)

call_manager = CallManager()


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt4.1-deepgram-elevenlabs-vapi-outbound",
        "model": GPT_MODEL,
        "orchestrator": "vapi",
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.post("/outbound/call")
async def outbound_initiate(
    request: Request,
    phone_number: str = Query(default=""),
    campaign_id: str = Query(default=""),
    opening_reason: str = Query(default=""),
    objective: str = Query(default=""),
    context: str = Query(default=""),
) -> dict:
    """Initiate an outbound call via Vapi.

    Creates a call record, then uses the Vapi API to place a call.
    Vapi handles the SIP signaling through Plivo and manages the
    entire voice pipeline.
    """
    if not phone_number:
        return {"error": "phone_number is required"}

    record = call_manager.create_call(
        phone_number=phone_number,
        campaign_id=campaign_id,
        opening_reason=opening_reason,
        objective=objective,
        context=context,
    )

    try:
        server_url = f"{PUBLIC_URL}/vapi/webhook" if PUBLIC_URL else ""
        vapi_call_id = await initiate_outbound_call(record, server_url)

        call_manager.update_status(
            record.call_id,
            "ringing",
            vapi_call_id=vapi_call_id,
        )
        logger.info(
            f"Outbound call initiated: call_id={record.call_id}, "
            f"to={phone_number}, vapi_call_id={vapi_call_id}"
        )

        return {
            "call_id": record.call_id,
            "status": "ringing",
            "phone_number": phone_number,
            "vapi_call_id": vapi_call_id,
        }

    except Exception as e:
        logger.error(f"Failed to initiate outbound call: {e}")
        call_manager.update_status(record.call_id, "failed", outcome="failed")
        return {"error": str(e), "call_id": record.call_id}


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> dict:
    """Main Vapi webhook endpoint for outbound calls.

    Handles all Vapi server events:
    - tool-calls: Execute tools and return results
    - status-update: Track call status changes
    - end-of-call-report: Finalize call records
    - conversation-update: Real-time transcript updates
    """
    body = await request.json()
    message = body.get("message", {})
    message_type = message.get("type", "")

    if message_type == "tool-calls":
        logger.info("Processing tool calls from Vapi")
        results = await handle_tool_calls(message)
        return {"results": results}

    if message_type == "status-update":
        status = message.get("status", "")
        call = message.get("call", {})
        vapi_call_id = call.get("id", "")

        record = call_manager.get_call_by_vapi_id(vapi_call_id)
        if record:
            if status == "in-progress":
                call_manager.update_status(
                    record.call_id,
                    "connected",
                    connected_at=datetime.utcnow(),
                )
            logger.info(
                f"Call status update: call_id={record.call_id}, "
                f"vapi_status={status}"
            )
        else:
            logger.debug(f"Status update for unknown vapi_call_id={vapi_call_id}: {status}")

        return {"ok": True}

    if message_type == "end-of-call-report":
        call = message.get("call", {})
        vapi_call_id = call.get("id", "")
        duration = message.get("durationSeconds", 0)
        ended_reason = message.get("endedReason", "")
        summary = message.get("summary", "")

        record = call_manager.get_call_by_vapi_id(vapi_call_id)
        if record:
            outcome = determine_outcome(ended_reason, duration)
            call_manager.update_status(
                record.call_id,
                "completed",
                ended_at=datetime.utcnow(),
                duration=duration,
                ended_reason=ended_reason,
                outcome=outcome,
            )
            logger.info(
                f"Outbound call {record.call_id} completed: outcome={outcome}, "
                f"duration={duration}s, summary={summary[:100]}"
            )
        else:
            logger.debug(
                f"End-of-call report for unknown vapi_call_id={vapi_call_id}: "
                f"duration={duration}s"
            )

        return {"ok": True}

    if message_type == "conversation-update":
        return {"ok": True}

    if message_type == "speech-update":
        return {"ok": True}

    if message_type == "hang":
        logger.info("Call hangup notification received")
        return {"ok": True}

    logger.debug(f"Unhandled Vapi event: {message_type}")
    return {"ok": True}


@app.get("/outbound/status/{call_id}")
async def outbound_status(call_id: str) -> dict:
    """Get status and details for an outbound call."""
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}

    return {
        "call_id": record.call_id,
        "phone_number": record.phone_number,
        "status": record.status,
        "campaign_id": record.campaign_id,
        "opening_reason": record.opening_reason,
        "objective": record.objective,
        "outcome": record.outcome,
        "duration": record.duration,
        "vapi_call_id": record.vapi_call_id,
        "created_at": record.created_at.isoformat(),
        "connected_at": record.connected_at.isoformat() if record.connected_at else None,
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
    }


@app.get("/outbound/campaign/{campaign_id}")
async def outbound_campaign(campaign_id: str) -> dict:
    """Get all calls for a campaign."""
    records = call_manager.get_calls_by_campaign(campaign_id)
    return {
        "campaign_id": campaign_id,
        "total": len(records),
        "calls": [
            {
                "call_id": r.call_id,
                "phone_number": r.phone_number,
                "status": r.status,
                "outcome": r.outcome,
                "duration": r.duration,
            }
            for r in records
        ],
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the outbound server."""
    logger.info(f"Starting GPT-4.1 Vapi Outbound Voice Agent on port {SERVER_PORT}")

    if PUBLIC_URL:
        logger.info(f"Vapi webhook URL: {PUBLIC_URL}/vapi/webhook")
    else:
        logger.warning("PUBLIC_URL not set. Outbound calls require a public webhook URL.")

    uvicorn.run("outbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
