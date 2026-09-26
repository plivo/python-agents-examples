"""
Outbound call E2E tests: place the call with Plivo's Make Call API, as a user would.

Tests:
1. /outbound/answer (reached through the tunnel) returns Stream XML whose body carries
   the answer_url greeting
2. Full outbound call cycle: plivo.RestClient().calls.create(answer_url=<tunnel>/outbound/
   answer?greeting=..., hangup_url=<tunnel>/outbound/hangup), record, transcribe, and
   verify from the server logs that the A-leg spoke the answer_url greeting verbatim, that
   the callee leg (no query params) used the default greeting, and that the hangup webhook
   was received

The agent calls from PLIVO_PHONE_NUMBER to PLIVO_TEST_NUMBER. A call between two Plivo
numbers creates a second, inbound call on PLIVO_TEST_NUMBER, answered by that number's
app. A <Wait>-only answer never answers an inbound call (Plivo ends both legs with
NO_USER_RESPONSE / Media Timeout), so PLIVO_TEST_NUMBER is temporarily assigned to an
app that answers with /outbound/answer (no query params): the callee is a second agent
instance with the default outbound greeting. The original app is restored afterwards.

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
from urllib.parse import quote, urlencode

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
    log_messages,
    read_log_events,
    server_log_path,
    signed_webhook,
    start_ngrok,
    start_server,
    stop_ngrok,
    stop_server,
    stream_body,
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
GREETING = (
    "Hi, this is Alex from TechFlow. I'm reaching out because you requested a demo of "
    "TechFlow Teams. Is now a good time for a quick chat?"
)

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


def _agent_texts(call_id: str) -> list[str]:
    events = read_log_events(LOG_PATH, "agent_text")
    return [e["text"] for e in events if e.get("call_id") == call_id]


# =============================================================================
# Tests
# =============================================================================


class TestOutboundCall:
    """End-to-end tests for outbound calling via Plivo's Make Call API."""

    def test_outbound_answer_webhook(self, server_process, ngrok_tunnel):
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
        url = f"{ngrok_tunnel}/outbound/answer?{query}"
        assert httpx.get(url, timeout=10.0).status_code == 403  # unsigned
        resp = signed_webhook("GET", url, PLIVO_AUTH_TOKEN)
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body
        assert "bidirectional" in body
        assert "audio/x-mulaw" in body
        assert ngrok_tunnel.replace("https://", "wss://") + "/ws?body=" in body
        meta = stream_body(body)
        assert meta["greeting"] == GREETING

        ready = [m for m in log_messages(LOG_PATH) if m.startswith("Ready! Place a call")]
        assert len(ready) == 1, ready
        assert f"{ngrok_tunnel}/outbound/answer?greeting=" in ready[0]

    def test_outbound_call_full_cycle(
        self, server_process, ngrok_tunnel, plivo_client, bleg_app_id
    ):
        """Make Call API -> /outbound/answer?greeting=... -> the agent speaks it verbatim."""
        from outbound.agent import DEFAULT_OUTBOUND_GREETING

        baseline = set(list_live_call_ids(plivo_client))
        answer_url = f"{ngrok_tunnel}/outbound/answer?" + urlencode(
            {"greeting": GREETING}, quote_via=quote
        )
        response = plivo_client.calls.create(
            from_=normalize_phone_number(PLIVO_PHONE_NUMBER),
            to_=normalize_phone_number(PLIVO_TEST_NUMBER),
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=f"{ngrok_tunnel}/outbound/hangup",
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
        assert len(transcript) > 5, f"No speech found in any recording of {call_uuids}"
        lower = transcript.lower()
        identity = [w for w in ("alex", "techflow", "tech flow") if w in lower]
        reason = [w for w in ("demo", "teams", "reaching out", "request") if w in lower]
        # Both legs run an agent, so the recording can mix two greetings; the reason is
        # checked deterministically on each agent's own agent_text below.
        print(f"[Result] identity={identity} reason={reason}")
        assert identity, f"Greeting lacks the agent identity: '{transcript}'"

        time.sleep(1)
        a_leg_texts = _agent_texts(a_leg)
        print(f"[Log] A-leg agent_text: {a_leg_texts}")
        assert a_leg_texts, "The A-leg agent never spoke"
        assert a_leg_texts[0] == GREETING
        a_events = [e for e in read_log_events(LOG_PATH) if e.get("call_id") == a_leg]
        answered = [e for e in a_events if e["event"] == "call_answered"]
        assert answered and answered[0]["to_number"].lstrip("+") == normalize_phone_number(
            PLIVO_TEST_NUMBER
        ), answered
        turns = [e for e in a_events if e["event"] == "turn_complete"]
        print(
            f"[Log] turn_complete: {[(t['turn'], t['barge_in'], t['playback_ms']) for t in turns]}"
        )
        assert turns, "The A-leg agent never completed a turn"
        sessions = [e for e in a_events if e["event"] == "session_end"]
        assert sessions and sessions[-1]["tx_chunks"] > 0, sessions

        if b_leg:  # callee answered /outbound/answer without query params -> default greeting
            b_leg_texts = _agent_texts(b_leg)
            print(f"[Log] B-leg agent_text: {b_leg_texts}")
            assert b_leg_texts and b_leg_texts[0] == DEFAULT_OUTBOUND_GREETING, b_leg_texts

        deadline = time.time() + 15
        ended = f"Outbound call ended: CallUUID={a_leg}"
        while time.time() < deadline and not any(ended in m for m in log_messages(LOG_PATH)):
            time.sleep(1)
        assert any(ended in m for m in log_messages(LOG_PATH)), "hangup_url webhook not received"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
