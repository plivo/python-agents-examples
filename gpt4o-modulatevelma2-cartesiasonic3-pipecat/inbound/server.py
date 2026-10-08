"""Standalone FastAPI server for inbound calls."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from typing import NoReturn
from urllib.parse import quote

import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml
from plivo.utils import validate_v3_signature

from inbound.agent import run_agent
from utils import normalize_phone_number

load_dotenv()

# Server configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# Plivo application this server creates/updates and attaches PLIVO_PHONE_NUMBER to
PLIVO_APP_NAME = "GPT4o_ModulateVelma2_CartesiaSonic3_Pipecat"

app = FastAPI(
    title="GPT-4o Modulate Velma-2 Cartesia Sonic 3 Pipecat Voice Agent (Inbound)",
    description=(
        "Inbound voice agent using GPT-4o LLM, Modulate Velma-2 STT, "
        "Cartesia Sonic 3 TTS and Tavily web search, with Pipecat and Plivo telephony"
    ),
    version="0.1.0",
)


# =============================================================================
# Webhook authentication: Plivo V3 signatures (always on)
# =============================================================================


def check_webhook_auth_config() -> None:
    """Startup: refuse to run without PLIVO_AUTH_TOKEN (the signature-check key)."""
    if not PLIVO_AUTH_TOKEN:
        logger.error(
            "PLIVO_AUTH_TOKEN is empty: Plivo webhook signatures cannot be checked, so "
            "every Plivo request would be rejected. Set PLIVO_AUTH_TOKEN (a "
            "subaccount's token if the number belongs to a subaccount)."
        )
        raise SystemExit(1)
    logger.info("Webhook auth: Plivo V3 signatures on webhooks")


def public_request_url(request: Request) -> str:
    """The URL Plivo called and signed: PUBLIC_URL + request path + raw query string.

    ``request.url`` is what this process sees (``http://localhost:8000/...`` behind a
    tunnel or proxy), not what Plivo signed. PUBLIC_URL is read at call time.
    """
    url = PUBLIC_URL.rstrip("/") + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    return url


def _reject_webhook(request: Request, reason: str) -> NoReturn:
    logger.warning(f"Rejected Plivo webhook {request.method} {request.url.path}: {reason}")
    raise HTTPException(status_code=403, detail="Invalid Plivo signature")


async def verify_plivo_signature(request: Request) -> None:
    """FastAPI dependency: 403 unless the request carries a valid Plivo V3 signature.

    POST: the form fields are the signed params (the URL's query string is part of the
    signed URL). GET: Plivo's params are in the query string, so the URL carries them.
    """
    signature = request.headers.get("X-Plivo-Signature-V3", "")
    nonce = request.headers.get("X-Plivo-Signature-V3-Nonce", "")
    if not signature or not nonce:
        _reject_webhook(request, "missing X-Plivo-Signature-V3 / -Nonce header")
    if not (PLIVO_AUTH_TOKEN and PUBLIC_URL):
        _reject_webhook(request, "PLIVO_AUTH_TOKEN or PUBLIC_URL not set")
    params: dict = {}
    if request.method == "POST":
        form = await request.form()
        for key in form:
            values = [str(v) for v in form.getlist(key)]
            params[key] = values if len(values) > 1 else values[0]
    try:
        valid = validate_v3_signature(
            request.method, public_request_url(request), nonce, PLIVO_AUTH_TOKEN, signature, params
        )
    except Exception as e:  # malformed URL/headers fail the SDK's argument validation
        _reject_webhook(request, f"signature check error ({type(e).__name__})")
    if not valid:
        _reject_webhook(request, "signature mismatch (does PUBLIC_URL match the URL Plivo calls?)")
    signed_query = " + query string" if request.url.query else ""
    logger.debug(f"Plivo signature verified: {request.method} {request.url.path}{signed_query}")


PLIVO_SIGNED = [Depends(verify_plivo_signature)]


def stream_url(body_data: dict) -> str:
    """wss:// URL for <Stream> carrying the base64 call metadata.

    ``body`` is percent-encoded: a raw ``+`` in base64 would reach /ws as a space.
    """
    body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
    ws_base = PUBLIC_URL.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_base}/ws?body={quote(body_b64, safe='')}"


# =============================================================================
# Call control
# =============================================================================


async def _hangup_call(call_uuid: str) -> None:
    """Hang up a live call via the Plivo REST API. Never raises.

    /ws calls this when the agent is done with a call, whatever the reason, so a
    pipeline that ended on its own does not leave the caller on a silent line
    (the <Stream> has keepCallAlive). Usually the caller hung up first and Plivo
    answers "not found", which is not an error.
    """
    log = logger.bind(call_id=call_uuid)
    if not (PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN and call_uuid):
        log.info("Skipping REST hangup (no Plivo credentials)")
        return
    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        await asyncio.to_thread(client.calls.delete, call_uuid)
        log.info(f"Hung up call via REST: {call_uuid}")
    except plivo.exceptions.ResourceNotFoundError:
        log.debug(f"Call {call_uuid} already ended; nothing to hang up")
    except Exception as e:
        log.warning(f"REST hangup failed for call {call_uuid}: {type(e).__name__}")


# =============================================================================
# Plivo Webhook Configuration
# =============================================================================


def configure_plivo_webhooks() -> bool:
    """Point PLIVO_PHONE_NUMBER at this server's /answer and /hangup.

    Blocking Plivo REST calls: called from main() before the event loop starts.
    """
    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PUBLIC_URL]):
        missing = []
        if not PLIVO_AUTH_ID:
            missing.append("PLIVO_AUTH_ID")
        if not PLIVO_AUTH_TOKEN:
            missing.append("PLIVO_AUTH_TOKEN")
        if not PLIVO_PHONE_NUMBER:
            missing.append("PLIVO_PHONE_NUMBER")
        if not PUBLIC_URL:
            missing.append("PUBLIC_URL")
        logger.warning(f"Skipping Plivo auto-config. Missing: {', '.join(missing)}")
        return False

    phone_number = normalize_phone_number(PLIVO_PHONE_NUMBER)
    if not phone_number:
        logger.error(f"Invalid phone number format: {PLIVO_PHONE_NUMBER}")
        return False

    answer_url = f"{PUBLIC_URL.rstrip('/')}/answer"
    hangup_url = f"{PUBLIC_URL.rstrip('/')}/hangup"

    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        # app_name filters by prefix server-side, so pick the exact match
        apps = client.applications.list(app_name=PLIVO_APP_NAME)
        existing_app = next((a for a in apps["objects"] if a["app_name"] == PLIVO_APP_NAME), None)
        if existing_app:
            client.applications.update(
                app_id=existing_app["app_id"],
                answer_url=answer_url,
                answer_method="POST",
                hangup_url=hangup_url,
                hangup_method="POST",
            )
            app_id = existing_app["app_id"]
            logger.info(f"Updated Plivo application: {PLIVO_APP_NAME}")
        else:
            response = client.applications.create(
                app_name=PLIVO_APP_NAME,
                answer_url=answer_url,
                answer_method="POST",
                hangup_url=hangup_url,
                hangup_method="POST",
            )
            app_id = response["app_id"]
            logger.info(f"Created Plivo application: {PLIVO_APP_NAME}")

        client.numbers.update(number=phone_number, app_id=app_id)

        logger.info(f"Plivo webhooks configured for +{phone_number}")
        logger.info(f"  Answer URL: {answer_url}")
        logger.info(f"  Hangup URL: {hangup_url}")
        return True

    except plivo.exceptions.ValidationError as e:
        logger.error(f"Plivo validation error: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to configure Plivo: {e}")
        return False


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt4o-modulatevelma2-cartesiasonic3-pipecat-inbound",
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.get("/answer", dependencies=PLIVO_SIGNED)
@app.post("/answer", dependencies=PLIVO_SIGNED)
async def answer_webhook(
    request: Request,
    CallUUID: str = Query(default=""),
    From: str = Query(default=""),
    To: str = Query(default=""),
) -> Response:
    """Plivo answer webhook - returns XML to start WebSocket audio streaming."""
    call_uuid = CallUUID
    from_number = From
    to_number = To

    if request.method == "POST":
        try:
            form_data = await request.form()
            call_uuid = call_uuid or str(form_data.get("CallUUID", ""))
            from_number = from_number or str(form_data.get("From", ""))
            to_number = to_number or str(form_data.get("To", ""))
        except Exception as e:
            logger.warning(f"Could not parse answer webhook form: {e}")

    logger.bind(call_id=call_uuid).info(
        f"Incoming call: CallUUID={call_uuid}, From={from_number}, To={to_number}"
    )

    ws_url = stream_url({"call_uuid": call_uuid, "from": from_number, "to": to_number})
    logger.info(f"WebSocket URL: {ws_url.split('?')[0]}")

    response = plivoxml.ResponseElement()
    stream = plivoxml.StreamElement(
        ws_url,
        bidirectional=True,
        keepCallAlive=True,
        contentType="audio/x-mulaw;rate=8000",
    )
    response.add(stream)

    return Response(content=response.to_string(), media_type="application/xml")


@app.post("/hangup", dependencies=PLIVO_SIGNED)
async def hangup_webhook(request: Request) -> Response:
    """Plivo hangup webhook - called when a call ends."""
    try:
        form_data = await request.form()
        call_uuid = str(form_data.get("CallUUID", ""))
        logger.bind(call_id=call_uuid).info(
            f"Call ended: CallUUID={call_uuid}, "
            f"Duration={form_data.get('Duration')}s, "
            f"HangupCause={form_data.get('HangupCause')}"
        )
    except Exception as e:
        logger.warning(f"Error parsing hangup webhook: {e}")

    return Response(content="OK", media_type="text/plain")


@app.post("/fallback", dependencies=PLIVO_SIGNED)
async def fallback_webhook() -> Response:
    """Fallback webhook if primary answer webhook fails."""
    logger.warning("Fallback webhook triggered")

    response = plivoxml.ResponseElement()
    response.add(
        plivoxml.SpeakElement(
            "We're sorry, but we're experiencing technical difficulties. Please try again later.",
            voice="Polly.Joanna",
        )
    )
    response.add(plivoxml.HangupElement())

    return Response(content=response.to_string(), media_type="application/xml")


@app.get("/hold", dependencies=PLIVO_SIGNED)
@app.post("/hold", dependencies=PLIVO_SIGNED)
async def hold_webhook() -> Response:
    """Hold endpoint - keeps call alive silently (used during testing)."""
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

    call_data = {}
    call_id = "unknown"
    if body:
        try:
            call_data = json.loads(base64.b64decode(body).decode())
            call_id = call_data.get("call_uuid", "unknown")
            logger.bind(call_id=call_id).debug(f"Call metadata: {call_data}")
        except Exception as e:
            logger.warning(f"Failed to decode call metadata: {e}")

    try:
        start_data = await websocket.receive_text()
        start_message = json.loads(start_data)

        if start_message.get("event") != "start":
            logger.bind(call_id=call_id).error(
                f"Expected start event, got: {start_message.get('event')}"
            )
            await websocket.close()
            return

        start_info = start_message.get("start", {})
        call_id = start_info.get("callId", call_data.get("call_uuid", "unknown"))
        stream_id = start_info.get("streamId")
        logger.bind(call_id=call_id).info(
            f"Plivo stream started: callId={call_id}, streamId={stream_id}"
        )

        await run_agent(
            websocket=websocket,
            call_id=call_id,
            stream_id=stream_id or "",
            from_number=call_data.get("from", ""),
            to_number=call_data.get("to", ""),
        )

    except WebSocketDisconnect:
        logger.bind(call_id=call_id).info("WebSocket disconnected")
    except Exception as e:
        logger.bind(call_id=call_id).error(f"WebSocket error: {e}")
    finally:
        # The agent is done (it returned or raised) but keepCallAlive keeps the
        # call up after the stream closes, so hang it up here. "unknown" means
        # Plivo never told us which call this is.
        if call_id != "unknown":
            await _hangup_call(call_id)
        with contextlib.suppress(Exception):
            await websocket.close()


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    """Run the inbound server."""
    logger.info(
        f"Starting GPT-4o + Modulate Velma-2 + Cartesia Sonic 3 Pipecat inbound voice agent "
        f"on port {SERVER_PORT}"
    )
    check_webhook_auth_config()

    if not PUBLIC_URL:
        logger.warning(
            "PUBLIC_URL is not set: Plivo webhooks will be rejected (403) because their "
            "signatures are checked against PUBLIC_URL"
        )

    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo webhooks...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_plivo_webhooks():
            logger.info(f"Ready! Call +{phone} to talk to the agent")
        else:
            logger.warning("Plivo auto-configuration failed. Configure manually.")
    else:
        logger.info("To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER")

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
