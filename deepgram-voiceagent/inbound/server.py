"""Standalone FastAPI server for inbound calls."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
import contextlib
import functools
import hashlib
import hmac
import json
import os
import sys
import threading
import time
from collections.abc import AsyncIterator
from typing import NoReturn
from urllib.parse import quote

import httpx
import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml
from plivo.utils import validate_v3_signature

from inbound.agent import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_INBOUND_AGENT_ID,
    DEEPGRAM_LISTEN_MODEL,
    DEEPGRAM_SPEAK_MODEL,
    DEEPGRAM_THINK_MODEL,
    DEEPGRAM_THINK_PROVIDER,
    run_agent,
)
from utils import (
    TunnelError,
    normalize_phone_number,
    start_quick_tunnel,
    stop_tunnel,
)

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

    # Optional export destination only: the LLM runs inside Deepgram, so it adds no LLM spans.
    Traceloop.init(app_name="deepgram-voiceagent")
    logger.info("OpenLLMetry (Traceloop) auto-instrumentation enabled")
except ImportError:
    pass

# Server configuration
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# Webhook + /ws authentication: README "Webhook authentication". Always on.
# Lifetime of the /ws token issued by the answer webhook. Plivo opens the stream within
# seconds of receiving the answer XML; 5 minutes leaves slack for slow networks while a
# leaked stream URL stops working soon after.
WS_TOKEN_TTL_S = 300

# --tunnel: Plivo rejects answer URLs whose hostname it can't resolve yet ("Must be a
# valid url"); a fresh trycloudflare.com hostname took ~70s to be accepted in testing.
TUNNEL_URL_ACCEPT_TIMEOUT_S = 180.0
TUNNEL_URL_RETRY_INTERVAL_S = 5.0
_tunnel_proc = None  # cloudflared process started by --tunnel

# Startup "Ready!" line: needs the server accepting connections AND the Plivo number set up
READY_PROBE_TIMEOUT_S = 30.0
READY_PROBE_INTERVAL_S = 0.1


class ReadyLine:
    """Logs ``Ready! Call +N ...`` exactly once, when both conditions hold.

    The two conditions arrive in either order and from different threads: ``serving()``
    from the lifespan probe on the event loop, ``plivo_configured()`` from main() (fixed
    PUBLIC_URL) or the --tunnel background thread. Neither is called if its step fails,
    so a failed or skipped Plivo setup logs no Ready line.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._serving = False
        self._phone = ""
        self._logged = False

    def serving(self) -> None:
        with self._lock:
            self._serving = True
            self._log_if_ready()

    def plivo_configured(self, phone: str) -> None:
        with self._lock:
            self._phone = phone
            self._log_if_ready()

    def _log_if_ready(self) -> None:
        if self._serving and self._phone and not self._logged:
            self._logged = True
            logger.info(f"Ready! Call +{self._phone} to talk to the agent (Ctrl+C to stop)")


_ready = ReadyLine()


async def _wait_until_serving(port: int, timeout_s: float = READY_PROBE_TIMEOUT_S) -> bool:
    """True once a local TCP connect to ``port`` succeeds.

    The lifespan startup hook runs before uvicorn binds the port, so "Ready" must wait
    for a real connection rather than trust the hook.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            _reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except OSError:
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(READY_PROBE_INTERVAL_S)
            continue
        writer.close()
        with contextlib.suppress(OSError):
            await writer.wait_closed()
        return True


async def _mark_serving_when_up() -> None:
    if await _wait_until_serving(SERVER_PORT, READY_PROBE_TIMEOUT_S):
        _ready.serving()
    else:
        logger.warning(f"Server did not accept connections on port {SERVER_PORT}; no Ready line")


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Probe for "Ready!"; stop the --tunnel process on shutdown (Ctrl+C and SIGTERM alike)."""
    ready_task = asyncio.create_task(_mark_serving_when_up())
    try:
        yield
    finally:
        ready_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ready_task
        stop_tunnel(_tunnel_proc)


app = FastAPI(
    lifespan=_lifespan,
    title="Deepgram Voice Agent API + Plivo (Inbound)",
    description=(
        "Inbound voice agent: Deepgram Voice Agent API "
        "(managed listen/think/speak pipeline) + Plivo"
    ),
    version="0.1.0",
)


# =============================================================================
# Webhook authentication: Plivo V3 signatures + short-lived /ws tokens
# =============================================================================


def check_webhook_auth_config() -> None:
    """Startup: refuse to run without PLIVO_AUTH_TOKEN (the key for both checks)."""
    if not PLIVO_AUTH_TOKEN:
        logger.error(
            "PLIVO_AUTH_TOKEN is empty: Plivo webhook signatures and /ws tokens cannot be "
            "checked, so every Plivo request would be rejected. Set PLIVO_AUTH_TOKEN (a "
            "subaccount's token if the number belongs to a subaccount)."
        )
        raise SystemExit(1)
    logger.info("Webhook auth: Plivo V3 signatures on webhooks, signed tokens on /ws")


def public_request_url(request: Request) -> str:
    """The URL Plivo called and signed: PUBLIC_URL + request path + raw query string.

    ``request.url`` is what this process sees (``http://localhost:8000/...`` behind a
    tunnel or proxy), not what Plivo signed. PUBLIC_URL is read at call time, so the
    value set by --tunnel is used.
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


def _ws_token_mac(body: str, expires: int) -> str:
    message = f"deepgram-voiceagent/ws|{expires}|{body}".encode()
    return hmac.new(PLIVO_AUTH_TOKEN.encode(), message, hashlib.sha256).hexdigest()


def issue_ws_token(body: str, now: float | None = None) -> str:
    """``<expiry>.<hmac>``: binds the /ws ``body`` to an expiry, keyed with PLIVO_AUTH_TOKEN."""
    expires = int(time.time() if now is None else now) + WS_TOKEN_TTL_S
    return f"{expires}.{_ws_token_mac(body, expires)}"


def ws_token_error(body: str, token: str, now: float | None = None) -> str | None:
    """Why ``token`` does not authorize ``body`` (None if it does)."""
    if not token:
        return "missing token"
    expires_s, _, mac = token.partition(".")
    if not expires_s.isdigit() or not mac:
        return "malformed token"
    if not PLIVO_AUTH_TOKEN:
        return "PLIVO_AUTH_TOKEN not set"
    if not hmac.compare_digest(mac, _ws_token_mac(body, int(expires_s))):
        return "bad token (body or expiry tampered)"
    if int(expires_s) < (time.time() if now is None else now):
        return "expired token"
    return None


def stream_url(body_data: dict) -> str:
    """wss:// URL for <Stream>: base64 call metadata plus its /ws token."""
    body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
    ws_base = PUBLIC_URL.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_base}/ws?body={quote(body_b64, safe='')}&token={issue_ws_token(body_b64)}"


# =============================================================================
# REST hangup (used by the agent after its goodbye finishes playing)
# =============================================================================


async def _hangup_call(call_uuid: str) -> None:
    """Hang up a live call via the Plivo REST API (no-op without credentials)."""
    if not (PLIVO_AUTH_ID and PLIVO_AUTH_TOKEN and call_uuid):
        logger.bind(call_id=call_uuid).info("Skipping REST hangup (no Plivo credentials)")
        return
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    await asyncio.to_thread(client.calls.delete, call_uuid)
    logger.bind(call_id=call_uuid).info(f"Hung up call via REST: {call_uuid}")


# =============================================================================
# Plivo Webhook Configuration
# =============================================================================


def _is_invalid_url_error(e: Exception) -> bool:
    """Plivo's API rejects answer/hangup URLs whose hostname it can't resolve yet."""
    return isinstance(e, plivo.exceptions.ValidationError) and "valid url" in str(e).lower()


def configure_plivo_webhooks(wait_for_url_s: float = 0.0) -> bool:
    """Point PLIVO_PHONE_NUMBER at this server's /answer and /hangup.

    With ``wait_for_url_s`` > 0 (used by --tunnel), keep retrying while Plivo rejects a
    PUBLIC_URL it can't resolve yet, instead of failing on the first attempt.
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

    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    app_name = "Deepgram_VoiceAgent"
    answer_url = f"{PUBLIC_URL}/answer"
    hangup_url = f"{PUBLIC_URL}/hangup"
    deadline = time.monotonic() + wait_for_url_s
    waiting_logged = False

    while True:
        try:
            # app_name filters by prefix server-side, so pick the exact match
            apps = client.applications.list(app_name=app_name)
            existing_app = next((a for a in apps["objects"] if a["app_name"] == app_name), None)
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

            client.numbers.update(number=phone_number, app_id=app_id)

            logger.info(f"Plivo webhooks configured for +{phone_number}")
            logger.info(f"  Answer URL: {answer_url}")
            logger.info(f"  Hangup URL: {hangup_url}")
            return True

        except plivo.exceptions.ValidationError as e:
            if _is_invalid_url_error(e) and time.monotonic() < deadline:
                if not waiting_logged:
                    logger.info(
                        f"Waiting for Plivo to accept {PUBLIC_URL} (a new tunnel hostname can "
                        f"take a minute to resolve); retrying every "
                        f"{TUNNEL_URL_RETRY_INTERVAL_S:.0f}s..."
                    )
                    waiting_logged = True
                time.sleep(TUNNEL_URL_RETRY_INTERVAL_S)
                continue
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
        "service": "deepgram-voiceagent-inbound",
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

    parent_call_uuid = ""
    sip_headers = {}
    if request.method == "POST":
        try:
            form_data = await request.form()
            call_uuid = call_uuid or str(form_data.get("CallUUID", ""))
            from_number = from_number or str(form_data.get("From", ""))
            to_number = to_number or str(form_data.get("To", ""))
            parent_call_uuid = parent_call_uuid or str(form_data.get("ParentCallUUID", ""))
            for key in form_data:
                if key.startswith("SIP-") or key.startswith("sip-"):
                    sip_headers[key] = str(form_data.get(key, ""))
        except Exception:
            pass

    logger.bind(call_id=call_uuid).info(
        f"Incoming call: CallUUID={call_uuid}, From={from_number}, To={to_number}"
    )

    body_data = {
        "call_uuid": call_uuid,
        "from": from_number,
        "to": to_number,
        "parent_call_uuid": parent_call_uuid,
        "sip_headers": sip_headers,
    }
    ws_url = stream_url(body_data)
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
    token: str = Query(default=""),
) -> None:
    """WebSocket endpoint for bidirectional audio streaming with Plivo.

    ``token`` (issued by the signed answer webhook) must match ``body`` and be unexpired;
    otherwise the handshake is refused before any Deepgram connection.
    """
    error = ws_token_error(body, token)
    if error:
        logger.warning(f"Rejected /ws connection: {error}")
        await websocket.close(code=1008)  # before accept(): handshake refused (403)
        return
    await websocket.accept()

    call_data = {}
    call_id = "unknown"
    if body:
        try:
            call_data = json.loads(base64.b64decode(body).decode())
            call_id = call_data.get("call_uuid", "unknown")
            logger.bind(call_id=call_id).info(f"Call metadata: {call_data}")
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
            from_number=call_data.get("from", ""),
            to_number=call_data.get("to", ""),
            stream_id=stream_id or "",
            parent_call_id=call_data.get("parent_call_uuid", ""),
            sip_headers=call_data.get("sip_headers"),
            hangup_callback=functools.partial(_hangup_call, call_id),
            saved_agent_models=_saved_agent_models or None,
        )

    except WebSocketDisconnect:
        logger.bind(call_id=call_id).info("WebSocket disconnected")
    except Exception as e:
        logger.bind(call_id=call_id).error(f"WebSocket error: {e}")
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()


# =============================================================================
# Main
# =============================================================================


def _start_tunnel() -> None:
    """``--tunnel``: expose SERVER_PORT via a Cloudflare quick tunnel and use it as PUBLIC_URL."""
    global PUBLIC_URL, _tunnel_proc
    try:
        url, proc = start_quick_tunnel(SERVER_PORT)
    except TunnelError as e:
        logger.error(f"--tunnel: {e}")
        raise SystemExit(1) from e
    _tunnel_proc = proc
    atexit.register(stop_tunnel, proc)  # covers exits before uvicorn's lifespan starts
    PUBLIC_URL = url
    logger.info(f"Tunnel up: {url} -> http://localhost:{SERVER_PORT}")


def _configure_plivo_for_tunnel(phone: str) -> None:
    """Background: wait until Plivo accepts the tunnel URL, then point the number at it."""
    if configure_plivo_webhooks(wait_for_url_s=TUNNEL_URL_ACCEPT_TIMEOUT_S):
        _ready.plivo_configured(phone)
    else:
        logger.warning(
            f"Plivo did not accept {PUBLIC_URL} within {TUNNEL_URL_ACCEPT_TIMEOUT_S:.0f}s. "
            "Restart with --tunnel to get a new URL, or use a fixed PUBLIC_URL"
        )


DEEPGRAM_API_BASE = "https://api.deepgram.com/v1"


class AgentIdNotFound(RuntimeError):
    """The reusable agent config UUID does not exist in the API key's project."""


def _deepgram_get(path: str) -> dict:
    resp = httpx.get(
        f"{DEEPGRAM_API_BASE}{path}",
        headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# listen/think/speak models of the reusable config, parsed from the body that
# verify_deepgram_agent_id() fetches at startup (configs are immutable, so this cannot go
# stale). Passed to run_agent() for trace attributes only; {} if the check didn't succeed.
_saved_agent_models: dict[str, str] = {}


def _saved_config_models(body: dict) -> dict[str, str]:
    """Models from a GET .../agents/{uuid} body (``config`` is a JSON string); {} if unparseable."""

    def provider(config: dict, section: str) -> dict:
        value = config.get(section) or {}
        if isinstance(value, list):  # Deepgram accepts a list of fallback providers
            value = value[0] if value else {}
        return value.get("provider") or {}

    try:
        config = body.get("config")
        config = json.loads(config) if isinstance(config, str) else config
        models = {
            "listen_model": provider(config, "listen").get("model"),
            "think_provider": provider(config, "think").get("type"),
            "think_model": provider(config, "think").get("model"),
            "speak_model": provider(config, "speak").get("model"),
        }
    except (AttributeError, TypeError, ValueError):
        return {}
    return {key: str(value) for key, value in models.items() if value}


def verify_deepgram_agent_id(agent_id: str) -> bool:
    """Fail-fast startup check: does the reusable agent config UUID exist?

    A Deepgram API key belongs to one project, and agent configs are looked up per
    project, so: GET /projects (the key's project), then GET its /agents/{uuid}.
    Returns True if found. Raises AgentIdNotFound if it isn't (every call would fail at
    connect). Returns False if it could not be verified (network error, key without the
    agent:read scope); the caller starts anyway with a warning.
    """
    try:
        project_id = _deepgram_get("/projects")["projects"][0]["project_id"]
        body = _deepgram_get(f"/projects/{project_id}/agents/{agent_id}")
        _saved_agent_models.clear()
        _saved_agent_models.update(_saved_config_models(body))
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (400, 404):
            raise AgentIdNotFound(
                f"DEEPGRAM_INBOUND_AGENT_ID={agent_id} is not an agent config in the API key's "
                "Deepgram project. Every call would fail at connect. Create one (README: "
                f"'Choosing a path') or unset DEEPGRAM_INBOUND_AGENT_ID."
            ) from e
        error: Exception = e
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        error = e
    logger.warning(
        f"Could not verify DEEPGRAM_INBOUND_AGENT_ID={agent_id} ({error}); starting unverified"
    )
    return False


def describe_deepgram_agent() -> str:
    """One line naming the active agent path. Reads agent.py constants; no network calls."""
    if DEEPGRAM_INBOUND_AGENT_ID:
        return (
            f"Deepgram agent: reusable config {DEEPGRAM_INBOUND_AGENT_ID} (models, prompt and "
            "functions come from the saved config; DEEPGRAM_* model env vars are not used)"
        )
    return (
        f"Deepgram agent: inline (listen={DEEPGRAM_LISTEN_MODEL}, "
        f"think={DEEPGRAM_THINK_PROVIDER}/{DEEPGRAM_THINK_MODEL}, speak={DEEPGRAM_SPEAK_MODEL})"
    )


def main() -> None:
    """Run the inbound server."""
    parser = argparse.ArgumentParser(description="Deepgram Voice Agent inbound server")
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="expose the server via a free Cloudflare quick tunnel (requires cloudflared) "
        "and use it as PUBLIC_URL; no ngrok or PUBLIC_URL needed",
    )
    args = parser.parse_args()

    logger.info(f"Starting Deepgram Voice Agent Inbound Agent on port {SERVER_PORT}")
    check_webhook_auth_config()
    logger.info(describe_deepgram_agent())
    if DEEPGRAM_INBOUND_AGENT_ID:
        try:
            if verify_deepgram_agent_id(DEEPGRAM_INBOUND_AGENT_ID):
                logger.info(f"Verified reusable agent config {DEEPGRAM_INBOUND_AGENT_ID} exists")
        except AgentIdNotFound as e:
            logger.error(str(e))
            raise SystemExit(1) from e

    if args.tunnel:
        _start_tunnel()
    if not PUBLIC_URL:
        logger.warning(
            "PUBLIC_URL is not set: Plivo webhooks will be rejected (403) because their "
            "signatures are checked against PUBLIC_URL. Set PUBLIC_URL or use --tunnel"
        )

    if PLIVO_PHONE_NUMBER and PUBLIC_URL:
        logger.info("Configuring Plivo webhooks...")
        phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
        if args.tunnel:
            # Serve calls right away; the number is pointed here once Plivo accepts the URL
            threading.Thread(target=_configure_plivo_for_tunnel, args=(phone,), daemon=True).start()
        elif configure_plivo_webhooks():
            _ready.plivo_configured(phone)  # logged once uvicorn accepts connections
        else:
            logger.warning("Plivo auto-configuration failed. Configure manually.")
    else:
        logger.info("To enable auto-configuration, set PUBLIC_URL and PLIVO_PHONE_NUMBER")

    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
