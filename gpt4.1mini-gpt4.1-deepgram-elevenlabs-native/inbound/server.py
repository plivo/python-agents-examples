"""Standalone FastAPI server for inbound calls."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys

import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml

from inbound.agent import run_agent
from utils import normalize_phone_number

load_dotenv()

# ---------------------------------------------------------------------------
# Loguru sink configuration (env-var driven)
# ---------------------------------------------------------------------------
_LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()
_LOG_FILE = os.getenv("LOG_FILE", "")

if _LOG_FORMAT == "json":
    logger.remove()
    logger.add(
        sys.stderr,
        serialize=True,
        level="DEBUG",
    )

if _LOG_FILE:
    logger.add(
        _LOG_FILE,
        serialize=True,
        rotation="100 MB",
        retention="7 days",
        level="DEBUG",
    )

# ---------------------------------------------------------------------------
# Redis Streams sink (optional — publishes structured events for real-time UIs)
# ---------------------------------------------------------------------------
_REDIS_EVENTS_URL = os.getenv("REDIS_EVENTS_URL", "")
_REDIS_STREAM_KEY = os.getenv("REDIS_STREAM_KEY", "voice-agent:events")

if _REDIS_EVENTS_URL:
    try:
        import redis as _redis_mod

        _redis_client = _redis_mod.Redis.from_url(_REDIS_EVENTS_URL, decode_responses=True)
        _redis_client.ping()

        def _redis_sink(message):
            record = message.record
            fields = {
                "ts": record["time"].isoformat(),
                "level": record["level"].name,
                "msg": str(record["message"]),
            }
            for k, v in record["extra"].items():
                fields[k] = str(v)
            with contextlib.suppress(Exception):
                _redis_client.xadd(_REDIS_STREAM_KEY, fields, maxlen=10000, approximate=True)

        logger.add(_redis_sink, level="DEBUG")
        logger.info("Redis Streams sink enabled — publishing to {}", _REDIS_STREAM_KEY)
    except ImportError:
        logger.warning("REDIS_EVENTS_URL set but 'redis' package not installed")
    except Exception as _redis_err:
        logger.warning("Redis Streams sink failed to connect: {}", _redis_err)

# ---------------------------------------------------------------------------
# OTel tracing setup (optional — install with `uv sync --extra observability`)
# ---------------------------------------------------------------------------
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider()
    if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        logger.info("OTel tracing enabled — exporting to OTLP endpoint")
    trace.set_tracer_provider(provider)
except ImportError:
    pass

try:
    from traceloop.sdk import Traceloop

    Traceloop.init(app_name="gpt4.1mini-gpt4.1-deepgram-elevenlabs-native")
    logger.info("OpenLLMetry (Traceloop) auto-instrumentation enabled")
except ImportError:
    pass

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()
    logger.info("httpx auto-instrumentation enabled")
except ImportError:
    pass

# Server configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

app = FastAPI(
    title="Dual-LLM Lead Qualification Voice Agent (Inbound)",
    description=(
        "Inbound voice agent: GPT-4.1 mini (conversation) + GPT-4.1 (reasoning) "
        "+ Deepgram STT + ElevenLabs TTS + Plivo"
    ),
    version="0.1.0",
)


# =============================================================================
# Plivo Webhook Configuration
# =============================================================================


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

        app_name = "DualLLM_Deepgram_ElevenLabs_Voice_Agent"
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


# =============================================================================
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "gpt4.1mini-gpt4.1-deepgram-elevenlabs-voice-agent-inbound",
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

    parent_call_uuid = ""
    sip_headers = {}
    if request.method == "POST":
        try:
            form_data = await request.form()
            call_uuid = call_uuid or str(form_data.get("CallUUID", ""))
            from_number = from_number or str(form_data.get("From", ""))
            to_number = to_number or str(form_data.get("To", ""))
            parent_call_uuid = str(form_data.get("ParentCallUUID", ""))
            for key in form_data:
                if key.startswith("SIP-") or key.startswith("sip-"):
                    sip_headers[key] = str(form_data.get(key, ""))
        except Exception:
            pass

    logger.info(f"Incoming call: CallUUID={call_uuid}, From={from_number}, To={to_number}")

    body_data = {
        "call_uuid": call_uuid,
        "from": from_number,
        "to": to_number,
        "parent_call_uuid": parent_call_uuid,
        "sip_headers": sip_headers,
    }
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


@app.get("/hold")
@app.post("/hold")
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

        await run_agent(
            websocket=websocket,
            call_id=call_id,
            from_number=call_data.get("from", ""),
            to_number=call_data.get("to", ""),
            stream_id=stream_id or "",
            parent_call_id=call_data.get("parent_call_uuid", ""),
            sip_headers=call_data.get("sip_headers", {}),
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
    """Run the inbound server."""
    logger.info(
        f"Starting Dual-LLM Lead Qualification Inbound Agent on port {SERVER_PORT}"
    )

    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo webhooks...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if configure_plivo_webhooks():
            logger.info(f"Ready! Call +{phone} to test")
        else:
            logger.warning("Plivo auto-configuration failed. Configure manually.")
    else:
        logger.info("To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER")

    uvicorn.run("inbound.server:app", host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
