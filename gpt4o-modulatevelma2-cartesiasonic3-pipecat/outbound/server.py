"""Standalone FastAPI server for outbound calls.

Calls are placed with Plivo's Make Call API directly (see the README), with
``answer_url`` pointing at this server's /outbound/answer. This server only answers
Plivo's webhooks and bridges audio: /outbound/answer reads the optional greeting from
its query string (``greeting``) and returns <Stream> XML; /ws runs the agent.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
from typing import NoReturn
from urllib.parse import quote, urlsplit

import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from fastapi.requests import HTTPConnection
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml
from plivo.utils import validate_v3_signature

from outbound.agent import run_agent
from utils import normalize_phone_number

load_dotenv()

# Server configuration. Own port (not SERVER_PORT) so inbound and outbound can run side by side.
SERVER_PORT = int(os.getenv("OUTBOUND_SERVER_PORT", "8001"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# Optional answer_url query param: the greeting spoken verbatim when the callee answers.
# Passed through as is; the agent module's default applies when absent.
GREETING_PARAM = "greeting"

app = FastAPI(
    title="GPT-4o Modulate Velma-2 Cartesia Sonic 3 Pipecat Voice Agent (Outbound)",
    description=(
        "Outbound voice agent using GPT-4o LLM, Modulate Velma-2 STT, "
        "Cartesia Sonic 3 TTS and Tavily web search, with Pipecat and Plivo telephony"
    ),
    version="0.1.0",
)


# =============================================================================
# Webhook and stream authentication: Plivo V3 signatures (always on)
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
    logger.info("Webhook auth: Plivo V3 signatures on webhooks and the /ws stream")


def public_request_url(conn: HTTPConnection) -> str:
    """The URL Plivo signed, rebuilt from PUBLIC_URL (read at call time).

    ``conn.url`` is what this process sees (``http://localhost:8001/...`` behind a tunnel
    or proxy), not what Plivo signed.
    Webhook: PUBLIC_URL + request path + raw query string.
    Stream (/ws): ``http://`` + PUBLIC_URL host + request path, no query string.
    """
    if isinstance(conn, WebSocket):
        return f"http://{urlsplit(PUBLIC_URL).netloc}{conn.url.path}"
    url = PUBLIC_URL.rstrip("/") + conn.url.path
    if conn.url.query:
        url += "?" + conn.url.query
    return url


def _reject_unsigned(conn: HTTPConnection, reason: str) -> NoReturn:
    """403 for a webhook; a stream is refused before accept (the client sees 403)."""
    if isinstance(conn, WebSocket):
        logger.warning(f"Rejected Plivo stream {conn.url.path}: {reason}")
        raise WebSocketException(code=1008, reason="Invalid Plivo signature")
    logger.warning(f"Rejected Plivo webhook {conn.scope['method']} {conn.url.path}: {reason}")
    raise HTTPException(status_code=403, detail="Invalid Plivo signature")


async def verify_plivo_signature(conn: HTTPConnection) -> None:
    """FastAPI dependency for webhooks and /ws: reject unless Plivo's V3 signature is valid.

    POST: the form fields are the signed params (the URL's query string is part of the
    signed URL). GET: Plivo's params are in the query string, so the URL carries them.
    Stream (/ws): a GET with no signed params.
    """
    signature = conn.headers.get("X-Plivo-Signature-V3", "")
    nonce = conn.headers.get("X-Plivo-Signature-V3-Nonce", "")
    if not signature or not nonce:
        _reject_unsigned(conn, "missing X-Plivo-Signature-V3 / -Nonce header")
    if not (PLIVO_AUTH_TOKEN and PUBLIC_URL):
        _reject_unsigned(conn, "PLIVO_AUTH_TOKEN or PUBLIC_URL not set")
    method = conn.scope.get("method", "GET")  # a WebSocket connects with GET
    params: dict = {}
    if isinstance(conn, Request) and method == "POST":
        form = await conn.form()
        for key in form:
            values = [str(v) for v in form.getlist(key)]
            params[key] = values if len(values) > 1 else values[0]
    try:
        valid = validate_v3_signature(
            method, public_request_url(conn), nonce, PLIVO_AUTH_TOKEN, signature, params
        )
    except Exception as e:  # malformed URL/headers fail the SDK's argument validation
        _reject_unsigned(conn, f"signature check error ({type(e).__name__})")
    if not valid:
        _reject_unsigned(conn, "signature mismatch (does PUBLIC_URL match the URL Plivo calls?)")
    if isinstance(conn, WebSocket):
        logger.debug(f"Plivo signature verified: stream {conn.url.path}")
        return
    signed_query = " + query string" if conn.url.query else ""
    logger.debug(f"Plivo signature verified: {method} {conn.url.path}{signed_query}")


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
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt4o-modulatevelma2-cartesiasonic3-pipecat-outbound",
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.get("/outbound/answer", dependencies=PLIVO_SIGNED)
@app.post("/outbound/answer", dependencies=PLIVO_SIGNED)
async def outbound_answer_webhook(
    request: Request,
    CallUUID: str = Query(default=""),
    From: str = Query(default=""),
    To: str = Query(default=""),
) -> Response:
    """Plivo answer webhook for a call placed with the Make Call API.

    The answer_url query string may carry the ``greeting``. It travels to /ws in the base64
    ``body`` of the <Stream> URL together with the Plivo call fields.
    """
    call_uuid = CallUUID
    from_number = From
    to_number = To
    greeting = request.query_params.get(GREETING_PARAM, "").strip()

    if request.method == "POST":
        try:
            form_data = await request.form()
            call_uuid = call_uuid or str(form_data.get("CallUUID", ""))
            from_number = from_number or str(form_data.get("From", ""))
            to_number = to_number or str(form_data.get("To", ""))
        except Exception as e:
            logger.warning(f"Could not parse answer webhook form: {e}")

    logger.bind(call_id=call_uuid).info(
        f"Outbound call answered: CallUUID={call_uuid}, To={to_number}, "
        f"greeting: {'from answer_url' if greeting else 'default'}"
    )

    body_data = {
        "call_uuid": call_uuid,
        "from": from_number,
        "to": to_number,
        GREETING_PARAM: greeting,
    }
    ws_url = stream_url(body_data)
    logger.info(f"Outbound WebSocket URL: {ws_url.split('?')[0]}")

    response = plivoxml.ResponseElement()
    stream = plivoxml.StreamElement(
        ws_url,
        bidirectional=True,
        keepCallAlive=True,
        contentType="audio/x-mulaw;rate=8000",
    )
    response.add(stream)

    return Response(content=response.to_string(), media_type="application/xml")


@app.post("/outbound/hangup", dependencies=PLIVO_SIGNED)
async def outbound_hangup_webhook(request: Request) -> Response:
    """Plivo hangup webhook - called when an outbound call ends."""
    try:
        form_data = await request.form()
        call_uuid = str(form_data.get("CallUUID", ""))
        logger.bind(call_id=call_uuid).info(
            f"Outbound call ended: CallUUID={call_uuid}, "
            f"Duration={form_data.get('Duration')}s, "
            f"HangupCause={form_data.get('HangupCause')}"
        )
    except Exception as e:
        logger.warning(f"Error parsing outbound hangup webhook: {e}")

    return Response(content="OK", media_type="text/plain")


@app.websocket("/ws", dependencies=PLIVO_SIGNED)
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
            greeting=str(call_data.get(GREETING_PARAM) or ""),
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
    """Run the outbound server."""
    logger.info(
        f"Starting GPT-4o + Modulate Velma-2 + Cartesia Sonic 3 Pipecat outbound voice agent "
        f"on port {SERVER_PORT}"
    )
    check_webhook_auth_config()

    if PUBLIC_URL:
        base = PUBLIC_URL.rstrip("/")
        logger.info(
            "Place calls with Plivo's Make Call API: "
            f"answer_url={base}/outbound/answer?{GREETING_PARAM}=<url-encoded greeting> "
            f"(greeting optional), hangup_url={base}/outbound/hangup"
        )
    else:
        logger.warning(
            "PUBLIC_URL is not set: Plivo cannot reach /outbound/answer, and signed webhooks "
            "are rejected (403) because signatures are checked against PUBLIC_URL"
        )

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
