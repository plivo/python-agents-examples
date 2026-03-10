"""Multi-turn voice conversation test.

Tests multi-turn conversation flow including barge-in.

Run: uv run pytest tests/test_multiturn_voice.py -v
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires real API credentials — run manually")
class TestMultiturnVoice:
    """Multi-turn conversation test."""

    @pytest.mark.asyncio
    async def test_multiturn_conversation(self):
        """Test multi-turn conversation with barge-in."""
        # TODO: Implement with real API credentials
        pass
