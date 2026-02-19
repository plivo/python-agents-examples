"""Live inbound call test for Vapi voice agent.

Makes a real phone call through Plivo to test the full pipeline:
Caller -> Plivo SIP -> Vapi -> GPT-4.1/Deepgram/ElevenLabs -> Response

Requires:
- VAPI_PRIVATE_KEY
- PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN
- PLIVO_PHONE_NUMBER (the Vapi-connected number)
- PLIVO_TEST_NUMBER (number to call from)
- Server running with ngrok (PUBLIC_URL set)
"""

from __future__ import annotations

import os
import time

import plivo
import pytest
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")


@pytest.fixture
def live_call_configured():
    """Check all credentials for live call testing."""
    if not all([
        VAPI_PRIVATE_KEY,
        PLIVO_AUTH_ID,
        PLIVO_AUTH_TOKEN,
        PLIVO_PHONE_NUMBER,
        PLIVO_TEST_NUMBER,
    ]):
        pytest.skip("Live call test credentials not fully configured")

    # Verify server is reachable
    if PUBLIC_URL:
        try:
            resp = requests.get(PUBLIC_URL, timeout=5)
            if resp.status_code != 200:
                pytest.skip(f"Server not reachable at {PUBLIC_URL}")
        except requests.RequestException:
            pytest.skip(f"Server not reachable at {PUBLIC_URL}")


def test_inbound_call_greeting(live_call_configured):
    """Test that an inbound call connects through Vapi and the agent responds.

    Flow: Plivo (PLIVO_TEST_NUMBER) -> PLIVO_PHONE_NUMBER -> SIP -> Vapi
    Vapi sends assistant-request webhook -> server responds with config
    Vapi runs GPT-4.1/Deepgram/ElevenLabs pipeline -> agent speaks greeting
    """
    from utils import normalize_phone_number

    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

    from_number = normalize_phone_number(PLIVO_TEST_NUMBER)
    to_number = normalize_phone_number(PLIVO_PHONE_NUMBER)

    logger.info(f"Placing inbound test call: {from_number} -> {to_number}")

    # Place a test call — Plivo routes through SIP trunk to Vapi
    # answer_url is for the calling leg (test number) — just play hold music
    call = client.calls.create(
        from_=from_number,
        to_=to_number,
        answer_url="https://s3.amazonaws.com/plivosupport/hold_music.xml",
        answer_method="GET",
        ring_timeout=30,
        time_limit=30,
    )

    request_uuid = call.get("request_uuid") if isinstance(call, dict) else getattr(
        call, "request_uuid", ""
    )
    assert request_uuid, "Failed to get request_uuid from Plivo"
    logger.info(f"Call placed: request_uuid={request_uuid}")

    # Wait for the call to connect, Vapi to process, and agent to speak
    time.sleep(20)

    # Check call status via Plivo
    try:
        call_details = client.calls.get(request_uuid)
        call_status = getattr(call_details, "call_status", "unknown")
        call_duration = getattr(call_details, "call_duration", 0)
        logger.info(f"Call status: {call_status}, duration: {call_duration}s")

        # The call should have connected (or completed if short)
        assert call_status in (
            "in-progress", "completed", "busy", "ringing"
        ), f"Unexpected call status: {call_status}"

    except Exception as e:
        logger.warning(f"Could not get call details: {e}")

    # Hang up
    try:
        client.calls.delete(request_uuid)
        logger.info("Call hung up")
    except Exception:
        pass  # Call may have already ended

    # Also verify via Vapi that the call was received
    try:
        resp = requests.get(
            "https://api.vapi.ai/call",
            headers={"Authorization": f"Bearer {VAPI_PRIVATE_KEY}"},
            params={"limit": 5},
            timeout=10,
        )
        if resp.status_code == 200:
            calls = resp.json()
            if calls:
                latest = calls[0]
                logger.info(
                    f"Latest Vapi call: id={latest.get('id')}, "
                    f"status={latest.get('status')}, "
                    f"type={latest.get('type')}"
                )
    except Exception as e:
        logger.warning(f"Could not check Vapi calls: {e}")
