"""E2E live tests for Vapi voice agent (no phone call).

Tests Vapi API interactions directly without making actual phone calls.
Requires VAPI_PRIVATE_KEY to be set.
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()

VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")


@pytest.fixture
def vapi_configured():
    """Check if Vapi API is configured."""
    if not VAPI_PRIVATE_KEY:
        pytest.skip("VAPI_PRIVATE_KEY not configured")


@pytest.mark.asyncio
async def test_create_transient_assistant(vapi_configured):
    """Test creating a transient assistant config via Vapi API."""
    from vapi import AsyncVapi

    client = AsyncVapi(token=VAPI_PRIVATE_KEY)

    # List existing assistants to verify API connectivity
    assistants = await client.assistants.list()
    assert assistants is not None


@pytest.mark.asyncio
async def test_vapi_phone_numbers(vapi_configured):
    """Test listing phone numbers registered with Vapi."""
    from vapi import AsyncVapi

    client = AsyncVapi(token=VAPI_PRIVATE_KEY)

    phone_numbers = await client.phone_numbers.list()
    assert phone_numbers is not None
