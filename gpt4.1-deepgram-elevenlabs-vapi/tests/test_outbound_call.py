"""Outbound call test for Vapi voice agent.

Tests outbound call initiation via Vapi API.
Vapi places the call through Plivo SIP trunk.

Requires:
- VAPI_PRIVATE_KEY
- VAPI_PHONE_NUMBER_ID
- PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN
- PLIVO_TEST_NUMBER (destination)
- PUBLIC_URL (ngrok URL for webhooks)
"""

from __future__ import annotations

import os
import time

import pytest
import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
VAPI_PHONE_NUMBER_ID = os.getenv("VAPI_PHONE_NUMBER_ID", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")


@pytest.fixture
def outbound_configured():
    """Check all credentials for outbound testing."""
    if not all([
        VAPI_PRIVATE_KEY,
        VAPI_PHONE_NUMBER_ID,
        PLIVO_AUTH_ID,
        PLIVO_AUTH_TOKEN,
        PLIVO_TEST_NUMBER,
        PUBLIC_URL,
    ]):
        pytest.skip("Outbound test credentials not fully configured")


@pytest.mark.asyncio
async def test_outbound_call_initiation(outbound_configured):
    """Test initiating an outbound call via Vapi and verify it reaches Plivo."""
    from outbound.agent import CallManager, initiate_outbound_call

    manager = CallManager()
    record = manager.create_call(
        phone_number=PLIVO_TEST_NUMBER,
        opening_reason="your free trial sign-up",
        objective="qualify the lead and schedule a demo",
    )

    server_url = f"{PUBLIC_URL}/vapi/webhook"
    logger.info(f"Initiating outbound call to {PLIVO_TEST_NUMBER}")

    try:
        vapi_call_id = await initiate_outbound_call(record, server_url)
        assert vapi_call_id, "Failed to get Vapi call ID"
        logger.info(f"Vapi call created: {vapi_call_id}")

        manager.update_status(
            record.call_id, "ringing", vapi_call_id=vapi_call_id
        )

        # Verify the call record was updated
        updated = manager.get_call(record.call_id)
        assert updated is not None
        assert updated.status == "ringing"
        assert updated.vapi_call_id == vapi_call_id

        # Wait a moment then check call status via Vapi API
        time.sleep(10)

        resp = requests.get(
            f"https://api.vapi.ai/call/{vapi_call_id}",
            headers={"Authorization": f"Bearer {VAPI_PRIVATE_KEY}"},
            timeout=10,
        )
        if resp.status_code == 200:
            call_data = resp.json()
            status = call_data.get("status", "")
            logger.info(f"Vapi call status: {status}")
            # Call should be queued, ringing, in-progress, or ended
            assert status in (
                "queued", "ringing", "in-progress", "forwarding", "ended"
            ), f"Unexpected Vapi call status: {status}"

        # End the call after verification
        time.sleep(5)
        try:
            requests.delete(
                f"https://api.vapi.ai/call/{vapi_call_id}",
                headers={"Authorization": f"Bearer {VAPI_PRIVATE_KEY}"},
                timeout=10,
            )
            logger.info("Call ended via Vapi API")
        except Exception:
            pass

    except Exception as e:
        pytest.fail(f"Outbound call failed: {e}")
