"""
Live call E2E test — places a real call through Plivo infrastructure.

This test:
1. Starts an ngrok tunnel to port 18002, then the inbound server with PUBLIC_URL set
2. Points PLIVO_PHONE_NUMBER at a test Plivo application (/answer on the tunnel)
3. Places a call from PLIVO_TEST_NUMBER to PLIVO_PHONE_NUMBER; the calling leg answers
   with /hold, so only the called number's leg runs the Deepgram agent
4. Records both legs, lets the greeting play, optionally speaks a question into the call
5. Hangs up, downloads the recordings and transcribes them with faster-whisper
6. Checks the transcript and the server's structured JSON logs (turn_complete via
   real Plivo checkpoint/playedStream acks)

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER and
      DEEPGRAM_API_KEY in .env (PLIVO_TEST_NUMBER is a second Plivo number)
    - ngrok binary available on PATH
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
    best_transcript,
    ensure_ffmpeg_on_path,
    get_app_id_for_number,
    hangup_quietly,
    place_call_and_wait,
    read_log_events,
    server_log_path,
    signed_webhook,
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

TEST_PORT = 18002
LOG_PATH = server_log_path("live_call_server")
APP_NAME = "Deepgram_VoiceAgent_Test"

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
    """Start ngrok first: the server needs PUBLIC_URL to build the wss:// stream URL."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[ngrok] Tunnel URL: {public_url}")
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(ngrok_tunnel):
    """Start the inbound server (SIGTERM -> wait(5) -> SIGKILL on teardown).

    PLIVO_PHONE_NUMBER is blanked so the server skips its own webhook auto-config;
    the plivo_configured fixture assigns (and later restores) the number instead.
    """
    proc = start_server(
        "inbound.server",
        TEST_PORT,
        LOG_PATH,
        {"PUBLIC_URL": ngrok_tunnel, "PLIVO_PHONE_NUMBER": ""},
    )
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_configured(server_process, ngrok_tunnel):
    """Point PLIVO_PHONE_NUMBER at a test app on the tunnel; restore it afterwards."""
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    phone_digits = normalize_phone_number(PLIVO_PHONE_NUMBER)
    original_app_id = get_app_id_for_number(client, phone_digits)

    app_id = upsert_application(
        client, APP_NAME, f"{ngrok_tunnel}/answer", hangup_url=f"{ngrok_tunnel}/hangup"
    )
    client.numbers.update(number=phone_digits, app_id=app_id)
    print(f"[Plivo] Assigned {phone_digits} to {APP_NAME} ({app_id})")

    yield {"client": client, "public_url": ngrok_tunnel}

    if original_app_id and original_app_id != app_id:
        client.numbers.update(number=phone_digits, app_id=original_app_id)
        print(f"\n[Plivo] Restored {phone_digits} to app {original_app_id}")


def _place_call(client, public_url):
    return place_call_and_wait(
        client,
        from_digits=normalize_phone_number(PLIVO_TEST_NUMBER),
        to_digits=normalize_phone_number(PLIVO_PHONE_NUMBER),
        answer_url=f"{public_url}/hold",
    )


def _start_recording(client, *call_uuids):
    for call_uuid in call_uuids:
        if call_uuid:
            try:
                client.calls.start_recording(call_uuid, file_format="mp3")
            except Exception as e:
                print(f"[Call] recording failed on {call_uuid}: {e}")


def _events_for(agent_call_uuid: str, event: str) -> list[dict]:
    events = read_log_events(LOG_PATH, event)
    mine = [e for e in events if e.get("call_id") == agent_call_uuid]
    return mine or events  # fall back if Plivo didn't report which leg is inbound


GREETING_WORDS = ["alex", "techflow", "deepgram", "help", "hi", "hello"]


# =============================================================================
# Tests
# =============================================================================


class TestLiveCall:
    """End-to-end tests that place a real call through Plivo."""

    def test_ngrok_tunnel_accessible(self, server_process, ngrok_tunnel):
        resp = httpx.get(ngrok_tunnel, timeout=10.0)
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
        assert "bidirectional" in resp.text
        assert "audio/x-mulaw" in resp.text
        assert public_url.replace("https://", "wss://") + "/ws?body=" in resp.text

    def test_live_call_greeting(self, plivo_configured):
        """Place a real call, record it, transcribe it, and verify the greeting."""
        client = plivo_configured["client"]
        outbound_uuid, agent_uuid = _place_call(client, plivo_configured["public_url"])
        assert outbound_uuid, "Call did not go live within 30s"

        try:
            _start_recording(client, outbound_uuid, agent_uuid)
            print("[Call] Letting the agent greet for 18s...")
            time.sleep(18)
        finally:
            hangup_quietly(client, outbound_uuid, agent_uuid)

        transcript = best_transcript(client, [outbound_uuid, agent_uuid])
        assert len(transcript) > 5, f"Transcript too short: '{transcript}'"
        matches = [w for w in GREETING_WORDS if w in transcript.lower()]
        assert matches, f"Greeting not found. Expected one of {GREETING_WORDS}: '{transcript}'"

        # Real Plivo acked our checkpoint with playedStream -> greeting turn completed
        time.sleep(1)
        turns = _events_for(agent_uuid, "turn_complete")
        print(
            f"[Log] turn_complete: {[(t['turn'], t['barge_in'], t['playback_ms']) for t in turns]}"
        )
        assert any(t["turn"] == 1 and not t["barge_in"] for t in turns), (
            "No turn_complete for the greeting: playedStream never arrived from Plivo"
        )
        assert _events_for(agent_uuid, "session_end"), "Agent session did not end after hangup"

    def test_live_call_two_way_conversation(self, plivo_configured):
        """Ask the agent a question via Plivo TTS and verify the spoken answer."""
        client = plivo_configured["client"]
        outbound_uuid, agent_uuid = _place_call(client, plivo_configured["public_url"])
        assert outbound_uuid, "Call did not go live within 30s"

        try:
            _start_recording(client, outbound_uuid, agent_uuid)
            print("[Call] Waiting 16s for the greeting...")
            time.sleep(16)
            question = "What plans do you offer and how much do they cost?"
            print(f"[Call] Speaking into the call: '{question}'")
            client.calls.speak(outbound_uuid, text=question, language="en-US", legs="aleg")
            print("[Call] Waiting 30s for the agent's answer...")
            time.sleep(30)
        finally:
            hangup_quietly(client, outbound_uuid, agent_uuid)

        transcript = best_transcript(client, [outbound_uuid, agent_uuid])
        lower = transcript.lower()
        assert [w for w in GREETING_WORDS if w in lower], f"No greeting: '{transcript}'"
        product_words = [
            "pro",
            "team",
            "enterprise",
            "starter",
            "twelve",
            "twenty",
            "dollar",
            "month",
            "plan",
            "price",
            "cost",
            "12",
            "25",
        ]
        product_matches = [w for w in product_words if w in lower]
        assert len(product_matches) >= 2, f"No pricing answer ({product_matches}): '{transcript}'"

        time.sleep(1)
        user_texts = _events_for(agent_uuid, "user_text")
        print(f"[Log] user_text (Flux): {[u['text'] for u in user_texts]}")
        assert user_texts, "Flux produced no user transcript for the spoken question"
        turns = _events_for(agent_uuid, "turn_complete")
        for t in turns:
            print(
                f"[Log] turn {t['turn']}: barge_in={t['barge_in']} "
                f"total_latency_ms={t['total_latency_ms']} playback_ms={t['playback_ms']}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
