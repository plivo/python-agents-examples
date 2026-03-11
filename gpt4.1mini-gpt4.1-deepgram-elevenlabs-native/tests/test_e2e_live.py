"""E2E live tests with real APIs (no phone call).

These tests verify the full pipeline works with real API credentials
but without placing actual phone calls.

Run: uv run pytest tests/test_e2e_live.py -v
"""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires real API keys — run manually")
class TestE2ELive:
    """End-to-end tests with real API connections."""

    @pytest.mark.asyncio
    async def test_deepgram_stt_connection(self):
        """Test that Deepgram STT connects and accepts audio."""
        # TODO: Implement with real Deepgram API key
        pass

    @pytest.mark.asyncio
    async def test_elevenlabs_tts_synthesis(self):
        """Test that ElevenLabs TTS produces audio."""
        # TODO: Implement with real ElevenLabs API key
        pass

    @pytest.mark.asyncio
    async def test_openai_dual_llm_routing(self):
        """Test dual-LLM routing: mini for conversation, full for tools."""
        # TODO: Implement with real OpenAI API key
        pass

    @pytest.mark.asyncio
    async def test_smart_turn_model_inference(self):
        """Test smart-turn-v2 model loads and runs inference."""
        # TODO: Implement — requires downloading model weights
        pass
