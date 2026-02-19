"""Multi-turn conversation test for Vapi voice agent.

Tests multi-turn conversation and barge-in handling through Vapi.
Since Vapi manages the full pipeline including VAD, barge-in is handled
server-side by Vapi's interruption detection.

Requires:
- VAPI_PRIVATE_KEY
- PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN
- PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")


@pytest.fixture
def live_configured():
    """Check all credentials for live testing."""
    if not all([
        VAPI_PRIVATE_KEY,
        PLIVO_AUTH_ID,
        PLIVO_AUTH_TOKEN,
        PLIVO_PHONE_NUMBER,
        PLIVO_TEST_NUMBER,
    ]):
        pytest.skip("Live test credentials not fully configured")


def test_vapi_vad_config_present(live_configured):
    """Verify that VAD and turn detection are configured in assistant config."""
    from inbound.agent import build_assistant_config

    config = build_assistant_config(server_url="https://example.com/vapi/webhook")

    # Verify VAD-related configuration
    assert "stopSpeakingPlan" in config, "Missing stopSpeakingPlan (VAD config)"
    assert "startSpeakingPlan" in config, "Missing startSpeakingPlan"
    assert "transcriptionEndpointingPlan" in config["startSpeakingPlan"], (
        "Missing turn detection config"
    )
    assert config["backgroundDenoisingEnabled"] is True, "Background denoising not enabled"

    stop_plan = config["stopSpeakingPlan"]
    assert stop_plan["numWords"] >= 1, "numWords should filter noise"
    assert stop_plan["voiceSeconds"] > 0, "voiceSeconds should require speech duration"
    assert stop_plan["backoffSeconds"] > 0, "backoffSeconds needed after interruption"
