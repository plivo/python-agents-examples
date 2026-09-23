"""
Outbound call E2E tests — verifies the outbound calling feature end-to-end.

Tests:
1. POST /outbound/call returns call_id and status tracking works
2. /outbound/answer returns valid Stream XML
3. Full outbound call cycle: place real call, record, transcribe, verify the greeting
   the speak model says verbatim (built from opening_reason)
4. Status lifecycle transitions (initiating -> ringing -> connected -> completed)
5. Programmatic hangup via POST /outbound/hangup/{call_id}
6. GET /outbound/campaign/{campaign_id}

The agent calls from PLIVO_PHONE_NUMBER to PLIVO_TEST_NUMBER. A call between two Plivo
numbers creates a second, inbound call on PLIVO_TEST_NUMBER, answered by that number's
app. A <Wait>-only answer (/hold) never answers an inbound call (Plivo ends both legs
with NO_USER_RESPONSE / Media Timeout), so PLIVO_TEST_NUMBER is temporarily assigned to
an app that answers with /outbound/answer: the callee is a second agent instance (with
the default outbound greeting). The original app is restored afterwards.

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER and
      DEEPGRAM_API_KEY in .env
    - ngrok binary available on PATH
    - faster-whisper installed (dev dependency), ffmpeg available
    - Port 18003 available

Usage:
    uv run pytest tests/test_outbound_call.py -v -s
"""

from __future__ import annotations

import os
import time

import httpx
import plivo
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    best_transcript,
    ensure_ffmpeg_on_path,
    get_app_id_for_number,
    hangup_quietly,
    list_live_call_ids,
    read_log_events,
    server_log_path,
    start_ngrok,
    start_server,
    stop_ngrok,
    stop_server,
    upsert_application,
)
from utils import normalize_phone_number

load_dotenv()
ensure_ffmpeg_on_path()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

TEST_PORT = 18003
LOG_PATH = server_log_path("outbound_server")
BLEG_APP_NAME = "Deepgram_VoiceAgent_Outbound_Test_Agent"
OPENING_REASON = "your recent demo request for TechFlow Teams"

pytestmark = pytest.mark.skipif(
    not all(
        [PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER, DEEPGRAM_API_KEY]
    ),
    reason="Plivo credentials, PLIVO_TEST_NUMBER, or DEEPGRAM_API_KEY not configured",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ngrok_tunnel():
    """Start ngrok before the server so PUBLIC_URL can be passed to it."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[ngrok] Tunnel URL: {public_url}")
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(ngrok_tunnel):
    """Start the outbound server (SIGTERM -> wait(5) -> SIGKILL on teardown)."""
    proc = start_server("outbound.server", TEST_PORT, LOG_PATH, {"PUBLIC_URL": ngrok_tunnel})
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_client():
    return plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)


@pytest.fixture(scope="module")
def bleg_app_id(plivo_client, ngrok_tunnel):
    """Point PLIVO_TEST_NUMBER at /outbound/answer (it answers the call); restore afterwards."""
    test_digits = normalize_phone_number(PLIVO_TEST_NUMBER)
    original_app_id = get_app_id_for_number(plivo_client, test_digits)
    app_id = upsert_application(plivo_client, BLEG_APP_NAME, f"{ngrok_tunnel}/outbound/answer")
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


def _initiate(public_url: str, **params: str) -> dict:
    resp = httpx.post(
        f"{public_url}/outbound/call",
        params={"phone_number": PLIVO_TEST_NUMBER, **params},
        timeout=30.0,
    )
    assert resp.status_code == 200
    data = resp.json()
    print(f"[Outbound] initiate: {data}")
    assert "call_id" in data, f"Expected call_id in response: {data}"
    assert "error" not in data, data
    return data


def _status(public_url: str, call_id: str) -> dict:
    return httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0).json()


def _wait_connected(public_url: str, call_id: str, timeout: float = 40.0) -> dict:
    status: dict = {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = _status(public_url, call_id)
        if status.get("status") == "connected" and status.get("plivo_call_uuid"):
            return status
        if status.get("status") in ("completed", "failed", "no_answer"):
            return status
        time.sleep(0.5)
    return status


# =============================================================================
# Tests
# =============================================================================


class TestOutboundCall:
    """End-to-end tests for outbound calling."""

    def test_initiate_outbound_call_api(self, server_process, ngrok_tunnel, bleg_app_id):
        """POST /outbound/call returns call_id and status tracking works."""
        data = _initiate(
            ngrok_tunnel,
            campaign_id="test-campaign-1",
            opening_reason=OPENING_REASON,
            objective="qualify interest and book a meeting with sales",
        )
        call_id = data["call_id"]
        assert data["status"] == "ringing"
        assert data["plivo_request_uuid"]

        status = _status(ngrok_tunnel, call_id)
        print(f"[Outbound] Status: {status}")
        assert status["call_id"] == call_id
        assert status["status"] in ("ringing", "connected", "completed", "failed", "no_answer")

        status = _wait_connected(ngrok_tunnel, call_id, timeout=30)
        hangup = httpx.post(f"{ngrok_tunnel}/outbound/hangup/{call_id}", timeout=10.0).json()
        print(f"[Outbound] Hangup response: {hangup}")

    def test_outbound_answer_webhook(self, server_process, ngrok_tunnel):
        """/outbound/answer returns valid Plivo Stream XML."""
        resp = httpx.get(
            f"{ngrok_tunnel}/outbound/answer",
            params={
                "call_id": "test-call-123",
                "CallUUID": "test-uuid-456",
                "From": PLIVO_PHONE_NUMBER,
                "To": PLIVO_TEST_NUMBER,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body
        assert "bidirectional" in body
        assert "audio/x-mulaw" in body
        assert ngrok_tunnel.replace("https://", "wss://") + "/ws?body=" in body

    def test_outbound_call_full_cycle(
        self, server_process, ngrok_tunnel, plivo_client, bleg_app_id
    ):
        """Place a real outbound call, record, transcribe, verify the outbound greeting."""
        baseline = set(list_live_call_ids(plivo_client))
        data = _initiate(
            ngrok_tunnel,
            campaign_id="test-full-cycle",
            opening_reason=OPENING_REASON,
            objective="qualify interest and book a meeting with sales",
        )
        call_id = data["call_id"]

        status = _wait_connected(ngrok_tunnel, call_id)
        a_leg_uuid = status.get("plivo_call_uuid", "")
        if status.get("status") != "connected" or not a_leg_uuid:
            pytest.skip(f"Call did not connect: {status}")
        print(f"[Outbound] Connected! A-leg UUID: {a_leg_uuid}")

        call_uuids = [a_leg_uuid]
        for uid in set(list_live_call_ids(plivo_client)) - baseline:
            if uid not in call_uuids:
                call_uuids.append(uid)
        print(f"[Outbound] Call legs: {call_uuids}")

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
        assert len(transcript) > 5, f"No speech found in any recording of {call_uuids}"
        lower = transcript.lower()
        identity = [w for w in ("alex", "techflow", "tech flow") if w in lower]
        reason = [w for w in ("demo", "teams", "reaching out", "request") if w in lower]
        # Both legs run an agent, so the recording can mix two greetings; the reason is
        # checked deterministically on the A-leg agent's own agent_text below.
        print(f"[Result] identity={identity} reason={reason}")
        assert identity, f"Greeting lacks the agent identity: '{transcript}'"

        time.sleep(1)
        a_leg = [e for e in read_log_events(LOG_PATH) if e.get("call_id") == a_leg_uuid]
        a_leg_texts = [e["text"] for e in a_leg if e["event"] == "agent_text"]
        print(f"[Log] A-leg agent_text: {a_leg_texts}")
        assert a_leg_texts, "The A-leg agent never spoke"
        assert OPENING_REASON in a_leg_texts[0], "Greeting lacks the opening reason"
        turns = [e for e in a_leg if e["event"] == "turn_complete"]
        print(
            f"[Log] turn_complete: {[(t['turn'], t['barge_in'], t['playback_ms']) for t in turns]}"
        )
        assert turns, "The A-leg agent never completed a turn"
        sessions = [e for e in a_leg if e["event"] == "session_end"]
        assert sessions and sessions[-1]["tx_chunks"] > 0, sessions

    def test_outbound_call_status_lifecycle(self, server_process, ngrok_tunnel, bleg_app_id):
        """Status transitions: initiating -> ringing -> connected -> completed."""
        data = _initiate(ngrok_tunnel, opening_reason="your recent free trial sign-up")
        call_id = data["call_id"]

        status = _status(ngrok_tunnel, call_id)
        print(f"\n[Lifecycle] Initial status: {status['status']}")
        assert status["status"] in ("ringing", "connected")

        status = _wait_connected(ngrok_tunnel, call_id)
        print(f"[Lifecycle] After connect wait: {status['status']}")
        assert status["status"] == "connected", status
        assert status["connected_at"]

        hangup = httpx.post(f"{ngrok_tunnel}/outbound/hangup/{call_id}", timeout=10.0).json()
        print(f"[Lifecycle] Hangup response: {hangup}")
        assert hangup.get("status") == "completed", hangup

        status = _status(ngrok_tunnel, call_id)
        print(f"[Lifecycle] Final status: {status['status']} outcome={status['outcome']}")
        assert status["status"] == "completed"

    def test_outbound_hangup_programmatic(
        self, server_process, ngrok_tunnel, plivo_client, bleg_app_id
    ):
        """POST /outbound/hangup/{call_id} ends an active call."""
        data = _initiate(ngrok_tunnel, opening_reason=OPENING_REASON)
        call_id = data["call_id"]

        status = _wait_connected(ngrok_tunnel, call_id)
        if status.get("status") != "connected":
            pytest.skip(f"Call did not connect in time: {status}")
        a_leg_uuid = status["plivo_call_uuid"]
        time.sleep(3)

        hangup = httpx.post(f"{ngrok_tunnel}/outbound/hangup/{call_id}", timeout=10.0).json()
        print(f"\n[Hangup] Response: {hangup}")
        assert hangup.get("status") == "completed", hangup

        time.sleep(3)
        status = _status(ngrok_tunnel, call_id)
        assert status["status"] in ("completed", "failed", "no_answer"), status
        assert a_leg_uuid not in list_live_call_ids(plivo_client), "Call still live after hangup"

    def test_outbound_campaign_endpoint(self, server_process, ngrok_tunnel, bleg_app_id):
        """GET /outbound/campaign/{campaign_id} returns calls for a campaign."""
        campaign_id = "test-campaign-endpoint"
        data = _initiate(ngrok_tunnel, campaign_id=campaign_id, opening_reason=OPENING_REASON)
        call_id = data["call_id"]

        _wait_connected(ngrok_tunnel, call_id, timeout=30)
        httpx.post(f"{ngrok_tunnel}/outbound/hangup/{call_id}", timeout=10.0)

        camp = httpx.get(f"{ngrok_tunnel}/outbound/campaign/{campaign_id}", timeout=10.0).json()
        print(f"\n[Campaign] Response: {camp}")
        assert camp["campaign_id"] == campaign_id
        assert camp["total"] >= 1
        assert call_id in [c["call_id"] for c in camp["calls"]]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
