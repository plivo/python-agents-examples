"""FastAPI server for Plivo webhook and WebSocket handling."""

from __future__ import annotations

import base64
import contextlib
import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import phonenumbers
import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml

import agent

load_dotenv()

# Configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Plivo configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# Public URL for webhooks (ngrok URL or production domain)
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# Store active pipeline tasks
active_tasks: dict[str, object] = {}


def normalize_phone_number(phone: str, default_region: str = DEFAULT_COUNTRY_CODE) -> str:
    """Normalize phone number to E.164 format (digits only, no leading +)."""
    if not phone:
        return ""

    try:
        parsed = phonenumbers.parse(phone, default_region)
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164.lstrip("+")
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Failed to parse phone number '{phone}': {e}")
        return "".join(c for c in phone if c.isdigit())


def configure_plivo_webhooks() -> bool:
    """Configure Plivo phone number with webhook URLs."""
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

    try:
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        app_name = "Gemini_Live_Pipecat_Agent"
        answer_url = f"{PUBLIC_URL}/answer"
        hangup_url = f"{PUBLIC_URL}/hangup"

        # Check if application already exists
        apps = client.applications.list()
        existing_app = None
        for app_obj in apps["objects"]:
            if app_obj["app_name"] == app_name:
                existing_app = app_obj
                break

        if existing_app:
            client.applications.update(
                app_id=existing_app["app_id"],
                answer_url=answer_url,
                answer_method="POST",
                hangup_url=hangup_url,
                hangup_method="POST",
            )
            app_id = existing_app["app_id"]
            logger.info(f"Updated Plivo application: {app_name}")
        else:
            response = client.applications.create(
                app_name=app_name,
                answer_url=answer_url,
                answer_method="POST",
                hangup_url=hangup_url,
                hangup_method="POST",
            )
            app_id = response["app_id"]
            logger.info(f"Created Plivo application: {app_name}")

        phone_number = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if not phone_number:
            logger.error(f"Invalid phone number format: {PLIVO_PHONE_NUMBER}")
            return False

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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI startup/shutdown."""
    logger.info("Starting Gemini Live Pipecat Voice Agent")
    yield
    logger.info("Shutting down...")
    for task in active_tasks.values():
        with contextlib.suppress(Exception):
            await task.cancel()


app = FastAPI(
    title="Gemini Live Pipecat Voice Agent",
    description="Voice agent using Pipecat with Google Gemini Live API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gemini-live-pipecat",
        "model": agent.GEMINI_MODEL,
        "phone_number": f"+{phone}" if phone else "not configured",
    }


@app.get("/answer")
@app.post("/answer")
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
        except Exception:
            pass

    logger.info(f"Incoming call: CallUUID={call_uuid}, From={from_number}, To={to_number}")

    body_data = {"call_uuid": call_uuid, "from": from_number, "to": to_number}
    body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()

    # Build WebSocket URL using request host (works with ngrok)
    host = request.headers.get("host", f"localhost:{SERVER_PORT}")
    protocol = "wss" if request.url.scheme == "https" else "ws"
    ws_url = f"{protocol}://{host}/ws?body={body_b64}"

    logger.info(f"WebSocket URL: {ws_url}")

    response = plivoxml.ResponseElement()
    stream = plivoxml.StreamElement(
        ws_url,
        bidirectional=True,
        keepCallAlive=True,
        contentType="audio/x-mulaw;rate=8000",
    )
    response.add(stream)

    return Response(content=response.to_string(), media_type="application/xml")


@app.post("/hangup")
async def hangup_webhook(request: Request) -> Response:
    """Plivo hangup webhook - called when a call ends."""
    try:
        form_data = await request.form()
        logger.info(
            f"Call ended: CallUUID={form_data.get('CallUUID')}, "
            f"Duration={form_data.get('Duration')}s, "
            f"HangupCause={form_data.get('HangupCause')}"
        )
    except Exception as e:
        logger.warning(f"Error parsing hangup webhook: {e}")

    return Response(content="OK", media_type="text/plain")


@app.post("/fallback")
async def fallback_webhook(request: Request) -> Response:
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


@app.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    body: str = Query(default=""),
) -> None:
    """WebSocket endpoint for bidirectional audio streaming with Plivo."""
    logger.info("WebSocket connection request received")

    await websocket.accept()
    logger.info("WebSocket connection accepted")

    call_data = {}
    if body:
        try:
            call_data = json.loads(base64.b64decode(body).decode())
            logger.info(f"Call metadata: {call_data}")
        except Exception as e:
            logger.warning(f"Failed to decode call metadata: {e}")

    call_id = None

    try:
        # Read the start message from Plivo
        logger.info("Waiting for start message from Plivo...")
        start_data = await websocket.receive_text()
        start_message = json.loads(start_data)

        logger.info(f"Received start message: {start_message}")

        if start_message.get("event") != "start":
            logger.error(f"Expected start event, got: {start_message.get('event')}")
            await websocket.close()
            return

        # Extract Plivo-specific IDs from the start event
        start_info = start_message.get("start", {})
        stream_id = start_info.get("streamId")
        call_id = start_info.get("callId", call_data.get("call_uuid", "unknown"))

        if not stream_id or not call_id:
            logger.error("Missing stream_id or call_id")
            await websocket.close()
            return

        task = await agent.run_agent(
            websocket=websocket,
            call_id=call_id,
            stream_id=stream_id,
            auth_id=PLIVO_AUTH_ID,
            auth_token=PLIVO_AUTH_TOKEN,
        )

        # Store the task
        active_tasks[call_id] = task

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        if call_id and call_id in active_tasks:
            del active_tasks[call_id]

        with contextlib.suppress(Exception):
            await websocket.close()


def main() -> None:
    """Run the server."""
    logger.info(f"Starting Gemini Live Pipecat Voice Agent on port {SERVER_PORT}")

    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo webhooks...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_plivo_webhooks():
            logger.info(f"Ready! Call +{phone} to test")
        else:
            logger.warning("Plivo auto-configuration failed. Configure manually.")
    else:
        logger.info("To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER")

    uvicorn.run("server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
