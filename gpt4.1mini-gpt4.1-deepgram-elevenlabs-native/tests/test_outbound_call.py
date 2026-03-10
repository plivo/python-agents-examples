"""Outbound call test.

Places a real outbound phone call via Plivo and verifies the agent
delivers the greeting.

Run: uv run pytest tests/test_outbound_call.py -v
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires real Plivo credentials — run manually")
class TestOutboundCall:
    """Real outbound call test."""

    @pytest.mark.asyncio
    async def test_outbound_call_greeting(self):
        """Test that outbound call delivers greeting."""
        # TODO: Implement with real Plivo credentials
        pass
