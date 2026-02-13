"""Standalone FastAPI server for outbound calls."""

from __future__ import annotations

import base64
import contextlib
import json
from datetime import datetime

import plivo
import uvicorn
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml

from outbound.agent import CallManager, determine_outcome, run_agent
from utils import (
    PLIVO_AUTH_ID,
    PLIVO_AUTH_TOKEN,
    PLIVO_FROM_NUMBER,
    PLIVO_PHONE_NUMBER,
    PUBLIC_URL,
    SERVER_PORT,
    normalize_phone_number,
)

app = FastAPI(
    title="Grok-Plivo Voice Agent (Outbound)",
    description="Outbound voice agent using xAI Grok with Plivo telephony",
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
        "service": "grok-plivo-voice-agent-outbound",
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
    """Initiate an outbound call.

    Creates a call record, then uses the Plivo API to place a call.
    When the callee answers, Plivo will hit /outbound/answer which starts
    the voice agent on the A-leg.
    """
    if not phone_number:
        return {"error": "phone_number is required"}

    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_FROM_NUMBER]):
        return {"error": "Plivo credentials or PLIVO_FROM_NUMBER not configured"}

    record = call_manager.create_call(
        phone_number=phone_number,
        campaign_id=campaign_id,
        opening_reason=opening_reason,
        objective=objective,
        context=context,
    )

    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        from_number = normalize_phone_number(PLIVO_FROM_NUMBER)
        to_number = normalize_phone_number(phone_number)

        answer_url = f"{PUBLIC_URL}/outbound/answer?call_id={record.call_id}"
        hangup_url = f"{PUBLIC_URL}/outbound/hangup"

        call_response = client.calls.create(
            from_=from_number,
            to_=to_number,
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=hangup_url,
            hangup_method="POST",
        )

        if isinstance(call_response, dict):
            request_uuid = call_response.get("request_uuid", "")
        else:
            request_uuid = getattr(call_response, "request_uuid", "")
        call_manager.update_status(
            record.call_id, "ringing",
            plivo_request_uuid=request_uuid,
        )
        logger.info(
            f"Outbound call initiated: call_id={record.call_id}, "
            f"to={to_number}, request_uuid={request_uuid}"
        )

        return {
            "call_id": record.call_id,
            "status": "ringing",
            "phone_number": phone_number,
            "plivo_request_uuid": request_uuid,
        }

    except Exception as e:
        logger.error(f"Failed to initiate outbound call: {e}")
        call_manager.update_status(record.call_id, "failed", outcome="failed")
        return {"error": str(e), "call_id": record.call_id}


@app.get("/outbound/answer")
@app.post("/outbound/answer")
async def outbound_answer_webhook(
    request: Request,
    call_id: str = Query(default=""),
    CallUUID: str = Query(default=""),
    From: str = Query(default=""),
    To: str = Query(default=""),
) -> Response:
    """Plivo webhook when the callee answers an outbound call.

    Returns <Stream> XML to start WebSocket audio streaming.
    The /ws endpoint detects this is an outbound call and loads
    the outbound prompt and initial message from CallManager.
    """
    call_uuid = CallUUID
    from_number = From
    to_number = To

    if request.method == "POST":
        try:
            form_data = await request.form()
            call_id = call_id or str(form_data.get("call_id", ""))
            call_uuid = call_uuid or str(form_data.get("CallUUID", ""))
            from_number = from_number or str(form_data.get("From", ""))
            to_number = to_number or str(form_data.get("To", ""))
        except Exception:
            pass

    logger.info(f"Outbound call answered: call_id={call_id}, CallUUID={call_uuid}, To={to_number}")

    # Update call record
    if call_id:
        call_manager.update_status(
            call_id, "connected",
            plivo_call_uuid=call_uuid,
            connected_at=datetime.utcnow(),
        )

    body_data = {
        "call_uuid": call_uuid,
        "from": from_number,
        "to": to_number,
        "is_outbound": True,
        "call_id": call_id,
    }
    body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()

    host = request.headers.get("host", f"localhost:{SERVER_PORT}")
    protocol = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{protocol}://{host}/ws?body={body_b64}"

    logger.info(f"Outbound WebSocket URL: {ws_url}")

    response = plivoxml.ResponseElement()
    stream = plivoxml.StreamElement(
        ws_url,
        bidirectional=True,
        keepCallAlive=True,
        contentType="audio/x-mulaw;rate=8000",
    )
    response.add(stream)

    return Response(content=response.to_string(), media_type="application/xml")


@app.post("/outbound/hangup")
async def outbound_hangup_webhook(request: Request) -> Response:
    """Plivo webhook when an outbound call ends."""
    try:
        form_data = await request.form()
        call_uuid = str(form_data.get("CallUUID", ""))
        duration = int(form_data.get("Duration", 0) or 0)
        hangup_cause = str(form_data.get("HangupCause", ""))

        logger.info(
            f"Outbound call ended: CallUUID={call_uuid}, "
            f"Duration={duration}s, HangupCause={hangup_cause}"
        )

        # Find and update the call record by plivo_call_uuid
        for record in call_manager.get_active_calls():
            if record.plivo_call_uuid == call_uuid or record.plivo_request_uuid == call_uuid:
                outcome = determine_outcome(hangup_cause, duration)
                call_manager.update_status(
                    record.call_id, "completed",
                    ended_at=datetime.utcnow(),
                    duration=duration,
                    hangup_cause=hangup_cause,
                    outcome=outcome,
                )
                logger.info(f"Outbound call {record.call_id} completed: outcome={outcome}")
                break
    except Exception as e:
        logger.warning(f"Error parsing outbound hangup webhook: {e}")

    return Response(content="OK", media_type="text/plain")


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
        "created_at": record.created_at.isoformat(),
        "connected_at": record.connected_at.isoformat() if record.connected_at else None,
        "ended_at": record.ended_at.isoformat() if record.ended_at else None,
    }


@app.post("/outbound/hangup/{call_id}")
async def outbound_hangup_call(call_id: str) -> dict:
    """Programmatically end an active outbound call."""
    record = call_manager.get_call(call_id)
    if not record:
        return {"error": "Call not found"}

    if record.status not in ("ringing", "connected"):
        return {"error": f"Call is not active (status: {record.status})"}

    if not record.plivo_call_uuid:
        return {"error": "No Plivo call UUID — call may not be connected yet"}

    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        client.calls.delete(record.plivo_call_uuid)
        call_manager.update_status(
            call_id, "completed",
            ended_at=datetime.utcnow(),
            outcome="success",
        )
        logger.info(f"Programmatically ended outbound call {call_id}")
        return {"call_id": call_id, "status": "completed"}
    except Exception as e:
        logger.error(f"Failed to end call {call_id}: {e}")
        return {"error": str(e)}


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


@app.get("/hold")
@app.post("/hold")
async def hold_webhook() -> Response:
    """Hold endpoint - keeps call alive silently (used for outbound A-leg)."""
    response = plivoxml.ResponseElement()
    response.add(plivoxml.WaitElement(length=120))
    return Response(content=response.to_string(), media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    body: str = Query(default=""),
) -> None:
    """WebSocket endpoint for bidirectional audio streaming with Plivo."""
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    call_data = {}
    if body:
        try:
            call_data = json.loads(base64.b64decode(body).decode())
            logger.info(f"Call metadata: {call_data}")
        except Exception as e:
            logger.warning(f"Failed to decode call metadata: {e}")

    try:
        start_data = await websocket.receive_text()
        start_message = json.loads(start_data)

        if start_message.get("event") != "start":
            logger.error(f"Expected start event, got: {start_message.get('event')}")
            await websocket.close()
            return

        start_info = start_message.get("start", {})
        call_id = start_info.get("callId", call_data.get("call_uuid", "unknown"))
        stream_id = start_info.get("streamId")
        logger.info(f"Plivo stream started: callId={call_id}, streamId={stream_id}")

        # Load outbound prompt and initial message from call record
        system_prompt = None
        initial_message = "Hello, I'm calling for help."
        if call_data.get("is_outbound"):
            outbound_call_id = call_data.get("call_id", "")
            record = call_manager.get_call(outbound_call_id)
            if record:
                system_prompt = record.system_prompt
                initial_message = record.initial_message
                logger.info(f"Outbound call detected: call_id={outbound_call_id}")
            else:
                logger.warning(f"Outbound call record not found: {outbound_call_id}")

        await run_agent(
            websocket=websocket,
            call_id=call_id,
            from_number=call_data.get("from", ""),
            to_number=call_data.get("to", ""),
            system_prompt=system_prompt,
            initial_message=initial_message,
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the outbound server."""
    logger.info(f"Starting Grok-Plivo Outbound Voice Agent on port {SERVER_PORT}")
    uvicorn.run("outbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
