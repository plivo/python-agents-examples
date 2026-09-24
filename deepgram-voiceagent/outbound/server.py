"""Standalone FastAPI server for outbound calls.

Calls are placed with Plivo's Make Call API directly (see the Ready line logged at
startup and the README). This server only answers Plivo's webhooks and bridges audio:
/outbound/answer reads the per-call context from its query string (opening_reason,
objective, context) and returns <Stream> XML; /ws runs the agent.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import base64
import contextlib
import functools
import json
import os
import sys
from collections.abc import AsyncIterator
from typing import NoReturn
from urllib.parse import quote, urlencode

import httpx
import plivo
import uvicorn
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from loguru import logger
from plivo import plivoxml
from plivo.utils import validate_v3_signature

from outbound.agent import (
    DEEPGRAM_API_KEY,
    DEEPGRAM_LISTEN_MODEL,
    DEEPGRAM_OUTBOUND_AGENT_ID,
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

# Server configuration. Own port (not SERVER_PORT) so inbound and outbound can run side by
# side, each with its own --tunnel.
SERVER_PORT = int(os.getenv("OUTBOUND_SERVER_PORT", "8001"))
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# Webhook authentication: README "Webhook authentication". Always on.

_tunnel_proc = None  # cloudflared process started by --tunnel

# Per-call context accepted on the answer_url query string (all optional)
CALL_DETAIL_PARAMS = ("opening_reason", "objective", "context")

# Startup "Ready!" line: logged once uvicorn accepts connections
READY_PROBE_TIMEOUT_S = 30.0
READY_PROBE_INTERVAL_S = 0.1
READY_EXAMPLE_OPENING_REASON = "you requested a demo"


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


def ready_message(tunnel: bool = False) -> str:
    """How to place a call with Plivo's Make Call API against this server's answer URL.

    Credentials appear only as $PLIVO_AUTH_ID / $PLIVO_AUTH_TOKEN shell references.
    """
    base = PUBLIC_URL.rstrip("/") or "<PUBLIC_URL>"
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    from_number = f"+{phone}" if phone else "<your Plivo number>"
    query = urlencode({"opening_reason": READY_EXAMPLE_OPENING_REASON}, quote_via=quote)
    lines = [
        "Ready! Place a call with Plivo's Make Call API; Plivo then requests this server's "
        "answer URL:",
        '  curl -X POST "https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/" \\',
        '    -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H "Content-Type: application/json" \\',
        f'    -d \'{{"from": "{from_number}", "to": "<E.164 number to call>",',
        f'         "answer_url": "{base}/outbound/answer?{query}",',
        f'         "hangup_url": "{base}/outbound/hangup",',
        '         "answer_method": "POST", "hangup_method": "POST"}\'',
        f"  Optional answer_url query params (URL-encoded): {', '.join(CALL_DETAIL_PARAMS)}.",
    ]
    if tunnel:
        lines.append(
            "  A new tunnel hostname can take a minute before Plivo accepts it; "
            'retry if the API answers "Must be a valid url".'
        )
    lines.append("(Ctrl+C to stop)")
    return "\n".join(lines)


async def _log_ready_when_serving() -> None:
    if await _wait_until_serving(SERVER_PORT, READY_PROBE_TIMEOUT_S):
        logger.info(ready_message(tunnel=_tunnel_proc is not None))
    else:
        logger.warning(f"Server did not accept connections on port {SERVER_PORT}; no Ready line")


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log "Ready!" once serving; stop the --tunnel process on shutdown (Ctrl+C / SIGTERM)."""
    ready_task = asyncio.create_task(_log_ready_when_serving())
    try:
        yield
    finally:
        ready_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ready_task
        stop_tunnel(_tunnel_proc)


app = FastAPI(
    lifespan=_lifespan,
    title="Deepgram Voice Agent API + Plivo (Outbound)",
    description=(
        "Outbound voice agent: Deepgram Voice Agent API "
        "(managed listen/think/speak pipeline) + Plivo"
    ),
    version="0.1.0",
)


# =============================================================================
# Webhook authentication: Plivo V3 signatures
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


def stream_url(body_data: dict) -> str:
    """wss:// URL for <Stream> carrying the base64 call metadata.

    ``body`` is percent-encoded: a raw ``+`` in base64 would reach /ws as a space.
    """
    body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
    ws_base = PUBLIC_URL.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    return f"{ws_base}/ws?body={quote(body_b64, safe='')}"


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
# Routes
# =============================================================================


@app.get("/")
async def health_check() -> dict:
    """Health check endpoint."""
    phone = normalize_phone_number(PLIVO_PHONE_NUMBER)
    return {
        "status": "ok",
        "service": "deepgram-voiceagent-outbound",
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

    The answer_url query string may carry the per-call context (``opening_reason``,
    ``objective``, ``context``). It travels to /ws in the base64 ``body`` of the <Stream>
    URL together with the Plivo call fields.
    """
    call_uuid = CallUUID
    from_number = From
    to_number = To
    details = {name: request.query_params.get(name, "").strip() for name in CALL_DETAIL_PARAMS}

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
        except Exception as e:
            logger.warning(f"Could not parse answer webhook form: {e}")

    provided = [name for name, value in details.items() if value]
    logger.bind(call_id=call_uuid).info(
        f"Outbound call answered: CallUUID={call_uuid}, To={to_number}, "
        f"call details: {', '.join(provided) or 'none (neutral defaults)'}"
    )

    body_data = {
        "call_uuid": call_uuid,
        "from": from_number,
        "to": to_number,
        "parent_call_uuid": parent_call_uuid,
        "sip_headers": sip_headers,
        **details,
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

        # Per-call context from the answer_url query string (renders prompt + greeting)
        details = {name: str(call_data.get(name) or "") for name in CALL_DETAIL_PARAMS}

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
            **details,
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
                f"DEEPGRAM_OUTBOUND_AGENT_ID={agent_id} is not an agent config in the API key's "
                "Deepgram project. Every call would fail at connect. Create one (README: "
                f"'Choosing a path') or unset DEEPGRAM_OUTBOUND_AGENT_ID."
            ) from e
        error: Exception = e
    except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
        error = e
    logger.warning(
        f"Could not verify DEEPGRAM_OUTBOUND_AGENT_ID={agent_id} ({error}); starting unverified"
    )
    return False


def describe_deepgram_agent() -> str:
    """One line naming the active agent path. Reads agent.py constants; no network calls."""
    if DEEPGRAM_OUTBOUND_AGENT_ID:
        return (
            f"Deepgram agent: reusable config {DEEPGRAM_OUTBOUND_AGENT_ID} (models, prompt and "
            "functions come from the saved config; DEEPGRAM_* model env vars are not used)"
        )
    return (
        f"Deepgram agent: inline (listen={DEEPGRAM_LISTEN_MODEL}, "
        f"think={DEEPGRAM_THINK_PROVIDER}/{DEEPGRAM_THINK_MODEL}, speak={DEEPGRAM_SPEAK_MODEL})"
    )


def main() -> None:
    """Run the outbound server."""
    parser = argparse.ArgumentParser(description="Deepgram Voice Agent outbound server")
    parser.add_argument(
        "--tunnel",
        action="store_true",
        help="expose the server via a free Cloudflare quick tunnel (requires cloudflared) "
        "and use it as PUBLIC_URL in the answer/hangup URLs you pass to Plivo",
    )
    args = parser.parse_args()

    logger.info(f"Starting Deepgram Voice Agent Outbound Agent on port {SERVER_PORT}")
    check_webhook_auth_config()
    logger.info(describe_deepgram_agent())
    if DEEPGRAM_OUTBOUND_AGENT_ID:
        try:
            if verify_deepgram_agent_id(DEEPGRAM_OUTBOUND_AGENT_ID):
                logger.info(f"Verified reusable agent config {DEEPGRAM_OUTBOUND_AGENT_ID} exists")
        except AgentIdNotFound as e:
            logger.error(str(e))
            raise SystemExit(1) from e

    if args.tunnel:
        _start_tunnel()
    if not PUBLIC_URL:
        logger.warning(
            "PUBLIC_URL is not set: Plivo cannot reach /outbound/answer, and signed webhooks "
            "are rejected (403) because signatures are checked against PUBLIC_URL. "
            "Set PUBLIC_URL or use --tunnel"
        )
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")


if __name__ == "__main__":
    main()
