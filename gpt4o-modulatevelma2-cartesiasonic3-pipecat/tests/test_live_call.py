"""
Live call E2E test: places a real call through Plivo infrastructure.

This test:
1. Starts an ngrok tunnel to port 18002, then the inbound server with PUBLIC_URL set
2. Points PLIVO_PHONE_NUMBER at a test Plivo application (/answer on the tunnel) and
   restores the number's original application afterwards
3. Places a call from PLIVO_TEST_NUMBER to PLIVO_PHONE_NUMBER; the calling leg answers
   with /hold, so only the called number's leg runs the agent
4. Records both legs and lets the agent's opening line play
5. Hangs up, downloads the recordings and transcribes them with faster-whisper
6. Checks the transcript and that the server logged the Plivo stream and the hangup webhook

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER,
      OPENAI_API_KEY, MODULATE_API_KEY and CARTESIA_API_KEY in .env
      (PLIVO_TEST_NUMBER is a second Plivo number)
    - ngrok binary available on PATH (or NGROK_BIN), with no other ngrok agent running
    - faster-whisper installed (dev dependency), ffmpeg available
    - Port 18002 available

Usage:
    uv run pytest tests/test_live_call.py -v -s
"""

from __future__ import annotations

import os
import time

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
    log_tail,
    missing_env,
    place_call_and_wait,
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

TEST_PORT = 18002
LOG_PATH = server_log_path("modulate_live_call_server")
# Test-only Plivo application (not the one inbound.server auto-configures)
APP_NAME = "GPT4o_ModulateVelma2_CartesiaSonic3_Pipecat_Test"

# The opening line is LLM-generated under inbound/system_prompt.md; wording varies.
GREETING_WORDS = ["hello", "hi", "help", "assist", "how can", "what can", "welcome"]

_MISSING = missing_env(*PLIVO_CALL_VARS, *LIVE_API_KEYS)
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"not configured: {', '.join(_MISSING)}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tunnel_url():
    """Start ngrok first: the server needs PUBLIC_URL to verify signatures and build wss://."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[tunnel] URL: {public_url}")
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(tunnel_url):
    """Start the inbound server (SIGTERM -> wait(5) -> SIGKILL on teardown).

    PLIVO_PHONE_NUMBER is blanked so the server skips its own webhook auto-config;
    the plivo_configured fixture assigns (and later restores) the number instead.
    """
    proc = start_server(
        "inbound.server",
        TEST_PORT,
        LOG_PATH,
        {"PUBLIC_URL": tunnel_url, "PLIVO_PHONE_NUMBER": ""},
    )
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_configured(server_process, tunnel_url):
    """Point PLIVO_PHONE_NUMBER at a test app on the tunnel; restore it afterwards."""
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    phone_digits = normalize_phone_number(PLIVO_PHONE_NUMBER)
    original_app_id = get_app_id_for_number(client, phone_digits)

    app_id = upsert_application(
        client, APP_NAME, f"{tunnel_url}/answer", hangup_url=f"{tunnel_url}/hangup"
    )
    client.numbers.update(number=phone_digits, app_id=app_id)
    print(f"[Plivo] Assigned {phone_digits} to {APP_NAME} ({app_id})")

    yield {"client": client, "public_url": tunnel_url}

    if original_app_id and original_app_id != app_id:
        client.numbers.update(number=phone_digits, app_id=original_app_id)
        print(f"\n[Plivo] Restored {phone_digits} to app {original_app_id}")


def _start_recording(client, *call_uuids):
    for call_uuid in call_uuids:
        if call_uuid:
            try:
                client.calls.start_recording(call_uuid, file_format="mp3")
            except Exception as e:
                print(f"[Call] recording failed on {call_uuid}: {e}")


# =============================================================================
# Tests
# =============================================================================


class TestLiveCall:
    """End-to-end tests that place a real call through Plivo."""

    def test_tunnel_accessible(self, server_process, tunnel_url):
        resp = httpx.get(tunnel_url, timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_answer_webhook_via_ngrok(self, plivo_configured):
        """Signed like Plivo signs it: accepted through the tunnel; unsigned: 403."""
        public_url = plivo_configured["public_url"]
        form = {"CallUUID": "test-ngrok", "From": "+15551234567", "To": "+16572338892"}
        assert httpx.post(f"{public_url}/answer", data=form, timeout=10.0).status_code == 403
        resp = signed_webhook("POST", f"{public_url}/answer", PLIVO_AUTH_TOKEN, form)
        assert resp.status_code == 200
        assert "<Stream" in resp.text
        assert 'bidirectional="true"' in resp.text
        assert 'keepCallAlive="true"' in resp.text
        assert "audio/x-mulaw;rate=8000" in resp.text
        assert public_url.replace("https://", "wss://") + "/ws?body=" in resp.text
        assert stream_body(resp.text)["call_uuid"] == "test-ngrok"

    def test_live_call_greeting(self, plivo_configured):
        """Place a real call, record it, transcribe it, and verify the opening line."""
        client = plivo_configured["client"]
        outbound_uuid, agent_uuid = place_call_and_wait(
            client,
            from_digits=normalize_phone_number(PLIVO_TEST_NUMBER),
            to_digits=normalize_phone_number(PLIVO_PHONE_NUMBER),
            answer_url=f"{plivo_configured['public_url']}/hold",
        )
        assert outbound_uuid, "Call did not go live within 30s"

        try:
            _start_recording(client, outbound_uuid, agent_uuid)
            print("[Call] Letting the agent speak for 18s...")
            time.sleep(18)
        finally:
            hangup_quietly(client, outbound_uuid, agent_uuid)

        transcript = best_transcript(client, [outbound_uuid, agent_uuid])
        assert len(transcript) > 5, f"Transcript too short: '{transcript}'\n{log_tail(LOG_PATH)}"
        matches = [w for w in GREETING_WORDS if w in transcript.lower()]
        assert matches, f"No greeting. Expected one of {GREETING_WORDS}: '{transcript}'"

        # Plivo's own signed webhooks and stream reached the server
        assert wait_for_log(LOG_PATH, "Plivo stream started: callId="), log_tail(LOG_PATH)
        assert wait_for_log(LOG_PATH, "Call ended: CallUUID="), "hangup webhook not received"
        assert wait_for_log(LOG_PATH, "Pipeline ended for call"), "pipeline did not end on hangup"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
