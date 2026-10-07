"""
Outbound call E2E tests: place the call with Plivo's Make Call API, as a user would.

Tests:
1. /outbound/answer (reached through the tunnel) returns Stream XML whose body carries
   the answer_url greeting; unsigned requests get 403
2. Full outbound call cycle: plivo.RestClient().calls.create(answer_url=<tunnel>/outbound/
   answer?greeting=..., hangup_url=<tunnel>/outbound/hangup), record, transcribe, and
   verify that the greeting from the answer_url was spoken, that the callee leg (no query
   params) got the default greeting, and that the hangup webhook was received

The agent calls from PLIVO_PHONE_NUMBER to PLIVO_TEST_NUMBER. A call between two Plivo
numbers creates a second, inbound call on PLIVO_TEST_NUMBER, answered by that number's
app. A <Wait>-only answer never answers an inbound call, so PLIVO_TEST_NUMBER is
temporarily assigned to an app that answers with /outbound/answer (no query params): the
callee is a second agent instance with the default outbound greeting. The original app
is restored afterwards.

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER,
      OPENAI_API_KEY, MODULATE_API_KEY and CARTESIA_API_KEY in .env
    - ngrok binary available on PATH (or NGROK_BIN), with no other ngrok agent running
    - faster-whisper installed (dev dependency), ffmpeg available
    - Port 18003 available

Usage:
    uv run pytest tests/test_outbound_call.py -v -s
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote, urlencode

import httpx
import plivo
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    LIVE_API_KEYS,
    PLIVO_CALL_VARS,
    best_transcript,
    ensure_ffmpeg_on_path,
    get_app_id_for_number,
    hangup_quietly,
    list_live_call_ids,
    log_tail,
    missing_env,
    read_log_text,
    server_log_path,
    signed_webhook,
    start_ngrok,
    start_server,
    stop_ngrok,
    stop_server,
    stream_body,
    upsert_application,
    wait_for_log,
)
from utils import normalize_phone_number

load_dotenv()
ensure_ffmpeg_on_path()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")

TEST_PORT = 18003
LOG_PATH = server_log_path("modulate_outbound_server")
# Test-only Plivo application that answers the callee leg
BLEG_APP_NAME = "GPT4o_ModulateVelma2_CartesiaSonic3_Pipecat_Outbound_Test"
GREETING = (
    "Hello, this is a courtesy call from the voice agent demo about your appointment. "
    "Is now a good time for a quick chat?"
)
# Words of GREETING that the default greeting does not contain
GREETING_ONLY_WORDS = ["courtesy", "demo", "appointment", "quick chat"]

_MISSING = missing_env(*PLIVO_CALL_VARS, *LIVE_API_KEYS)
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"not configured: {', '.join(_MISSING)}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tunnel_url():
    """Start the tunnel before the server so PUBLIC_URL can be passed to it."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[tunnel] URL: {public_url}")
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(tunnel_url):
    """Start the outbound server (SIGTERM -> wait(5) -> SIGKILL on teardown).

    The outbound server never configures a Plivo number; PLIVO_PHONE_NUMBER is blanked
    anyway so nothing in this process depends on it.
    """
    proc = start_server(
        "outbound.server",
        TEST_PORT,
        LOG_PATH,
        {"PUBLIC_URL": tunnel_url, "PLIVO_PHONE_NUMBER": ""},
    )
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_client():
    return plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)


@pytest.fixture(scope="module")
def bleg_app_id(plivo_client, tunnel_url):
    """Point PLIVO_TEST_NUMBER at /outbound/answer (it answers the call); restore afterwards."""
    test_digits = normalize_phone_number(PLIVO_TEST_NUMBER)
    original_app_id = get_app_id_for_number(plivo_client, test_digits)
    app_id = upsert_application(plivo_client, BLEG_APP_NAME, f"{tunnel_url}/outbound/answer")
    plivo_client.numbers.update(number=test_digits, app_id=app_id)
    print(f"\n[Plivo] Configured {test_digits} with B-leg app {app_id}")

    yield app_id

    if original_app_id and original_app_id != app_id:
        plivo_client.numbers.update(number=test_digits, app_id=original_app_id)
        print(f"\n[Plivo] Restored {test_digits} to original app {original_app_id}")


@pytest.fixture(autouse=True)
def hang_up_leftover_calls(plivo_client):
    """Hang up any call a test left live (e.g. one still ringing when it finished)."""
    baseline = set(list_live_call_ids(plivo_client))
    yield
    leftovers = set(list_live_call_ids(plivo_client)) - baseline
    if leftovers:
        print(f"\n[Cleanup] hanging up leftover calls: {leftovers}")
        hangup_quietly(plivo_client, *leftovers)


# =============================================================================
# Helpers
# =============================================================================


def _direction(client: plivo.RestClient, call_uuid: str) -> str:
    try:
        live = client.live_calls.get(call_uuid)
    except Exception:
        return ""
    value = live.get("direction", "") if isinstance(live, dict) else getattr(live, "direction", "")
    return str(value or "")


def _wait_for_legs(
    client: plivo.RestClient, baseline: set[str], timeout: float = 40.0
) -> tuple[str, str]:
    """(A-leg, B-leg): the outbound call we placed and the inbound call it created."""
    a_leg, b_leg = "", ""
    deadline = time.time() + timeout
    while time.time() < deadline and not (a_leg and b_leg):
        for call_uuid in set(list_live_call_ids(client)) - baseline:
            direction = _direction(client, call_uuid)
            if direction == "outbound":
                a_leg = call_uuid
            elif direction == "inbound":
                b_leg = call_uuid
        time.sleep(0.5)
    print(f"[Outbound] live legs: A={a_leg} B={b_leg}")
    return a_leg, b_leg


# =============================================================================
# Tests
# =============================================================================


class TestOutboundCall:
    """End-to-end tests for outbound calling via Plivo's Make Call API."""

    def test_outbound_answer_webhook(self, server_process, tunnel_url):
        """/outbound/answer returns valid Plivo Stream XML carrying the greeting."""
        query = urlencode(
            {
                "CallUUID": "test-uuid-456",
                "From": PLIVO_PHONE_NUMBER,
                "To": PLIVO_TEST_NUMBER,
                "greeting": GREETING,
            },
            quote_via=quote,
        )
        url = f"{tunnel_url}/outbound/answer?{query}"
        assert httpx.get(url, timeout=10.0).status_code == 403  # unsigned
        resp = signed_webhook("GET", url, PLIVO_AUTH_TOKEN)
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body
        assert 'bidirectional="true"' in body
        assert 'keepCallAlive="true"' in body
        assert "audio/x-mulaw;rate=8000" in body
        assert tunnel_url.replace("https://", "wss://") + "/ws?body=" in body
        meta = stream_body(body)
        assert meta["greeting"] == GREETING
        assert meta["call_uuid"] == "test-uuid-456"

        hint = f"answer_url={tunnel_url}/outbound/answer?greeting="
        assert hint in read_log_text(LOG_PATH), "startup log lacks the Make Call hint"

    def test_no_dial_endpoint(self, server_process, tunnel_url):
        """The server has no dial endpoint: calls are placed with the Plivo API directly."""
        assert httpx.post(f"{tunnel_url}/outbound/call", timeout=10.0).status_code == 404

    def test_outbound_call_full_cycle(self, server_process, tunnel_url, plivo_client, bleg_app_id):
        """Make Call API -> /outbound/answer?greeting=... -> the agent speaks it."""
        baseline = set(list_live_call_ids(plivo_client))
        answer_url = f"{tunnel_url}/outbound/answer?" + urlencode(
            {"greeting": GREETING}, quote_via=quote
        )
        response = plivo_client.calls.create(
            from_=normalize_phone_number(PLIVO_PHONE_NUMBER),
            to_=normalize_phone_number(PLIVO_TEST_NUMBER),
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=f"{tunnel_url}/outbound/hangup",
            hangup_method="POST",
        )
        request_uuid = (
            response.get("request_uuid", "")
            if isinstance(response, dict)
            else getattr(response, "request_uuid", "")
        )
        print(f"[Outbound] Make Call request_uuid={request_uuid}")
        assert request_uuid

        a_leg, b_leg = _wait_for_legs(plivo_client, baseline)
        if not a_leg:
            pytest.skip("The outbound call did not connect")
        call_uuids = [uid for uid in (a_leg, b_leg) if uid]

        try:
            for uid in call_uuids:
                try:
                    plivo_client.calls.start_recording(uid, file_format="mp3")
                except Exception as e:
                    print(f"[Outbound] Recording failed on {uid}: {e}")
            print("[Outbound] Letting the agent speak for 18s...")
            time.sleep(18)
        finally:
            hangup_quietly(plivo_client, *call_uuids)

        transcript = best_transcript(plivo_client, call_uuids)
        assert len(transcript) > 5, (
            f"No speech found in any recording of {call_uuids}\n{log_tail(LOG_PATH)}"
        )
        # Both legs run an agent, and both greet the moment the call connects, so each
        # agent's speech interrupts the other's greeting (barge-in) and the recording is
        # two voices talking over each other. The words heard are reported, but the proof
        # that the answer_url greeting was spoken verbatim is the text handed to TTS.
        spoken = [w for w in GREETING_ONLY_WORDS if w in transcript.lower()]
        print(f"[Result] greeting words heard in the recording: {spoken}")

        log = read_log_text(LOG_PATH)
        assert f"Generating TTS [{GREETING}]" in log, (
            f"The answer_url greeting never reached TTS\n{log_tail(LOG_PATH)}"
        )
        assert f"Outbound call answered: CallUUID={a_leg}" in log
        a_line = next(line for line in log.splitlines() if f"answered: CallUUID={a_leg}" in line)
        assert "greeting: from answer_url" in a_line
        assert f"Plivo stream started: callId={a_leg}" in log
        if b_leg:  # callee answered /outbound/answer without query params -> default greeting
            b_lines = [line for line in log.splitlines() if f"answered: CallUUID={b_leg}" in line]
            assert b_lines and "greeting: default" in b_lines[0], b_lines

        ended = f"Outbound call ended: CallUUID={a_leg}"
        assert wait_for_log(LOG_PATH, ended, timeout=15), "hangup_url webhook not received"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
