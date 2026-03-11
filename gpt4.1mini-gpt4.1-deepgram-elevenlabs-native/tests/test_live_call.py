"""Live inbound call test.

Places a real phone call via Plivo, records the greeting,
and verifies the agent responds.

Run: uv run pytest tests/test_live_call.py -v
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires real Plivo credentials and phone numbers — run manually")
class TestLiveCall:
    """Real inbound call test."""

    @pytest.mark.asyncio
    async def test_inbound_call_greeting(self):
        """Test that calling the agent produces a greeting."""
        # TODO: Implement with real Plivo credentials
        pass
