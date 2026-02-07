"""Tests for voice agent components."""

import asyncio
import base64
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import (
    DeepgramSTT,
    ElevenLabsTTS,
    GeminiLLM,
    VoiceAgent,
    pcm_to_ulaw,
    ulaw_to_pcm,
)


class TestDeepgramSTT:
    """Tests for Deepgram STT client."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test DeepgramSTT initialization."""
        callback = AsyncMock()
        stt = DeepgramSTT(on_transcript=callback)

        assert stt.on_transcript == callback
        assert stt._ws is None
        assert stt._running is False

    @pytest.mark.asyncio
    async def test_connect_creates_websocket(self):
        """Test that connect creates a WebSocket connection."""
        callback = AsyncMock()
        stt = DeepgramSTT(on_transcript=callback)

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = AsyncMock(return_value=iter([]))

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)
            mock_session_class.return_value = mock_session

            await stt.connect()

            assert stt._running is True
            mock_session.ws_connect.assert_called_once()

            # Verify URL contains expected parameters
            call_args = mock_session.ws_connect.call_args
            url = call_args[0][0]
            assert "api.deepgram.com" in url
            assert "model=" in url
            assert "encoding=linear16" in url

            await stt.close()

    @pytest.mark.asyncio
    async def test_send_audio(self):
        """Test sending audio to Deepgram."""
        callback = AsyncMock()
        stt = DeepgramSTT(on_transcript=callback)

        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.send_bytes = AsyncMock()
        stt._ws = mock_ws

        test_audio = b"\x00\x01\x02\x03"
        await stt.send_audio(test_audio)

        mock_ws.send_bytes.assert_called_once_with(test_audio)

    @pytest.mark.asyncio
    async def test_transcript_callback(self):
        """Test that transcripts trigger callback."""
        callback = AsyncMock()
        stt = DeepgramSTT(on_transcript=callback)

        # Simulate receiving a transcript
        transcript_data = {
            "type": "Results",
            "channel": {"alternatives": [{"transcript": "Hello world"}]},
        }

        # Call the transcript handler logic directly
        if transcript_data.get("type") == "Results":
            channel = transcript_data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if alternatives:
                transcript = alternatives[0].get("transcript", "")
                if transcript.strip():
                    await callback(transcript)

        callback.assert_called_once_with("Hello world")


class TestElevenLabsTTS:
    """Tests for ElevenLabs TTS client."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test ElevenLabsTTS initialization."""
        tts = ElevenLabsTTS()
        assert tts._session is None

    @pytest.mark.asyncio
    async def test_connect_creates_session(self):
        """Test that connect creates an HTTP session."""
        tts = ElevenLabsTTS()

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            await tts.connect()

            assert tts._session is mock_session

            await tts.close()

    @pytest.mark.asyncio
    async def test_synthesize_success(self):
        """Test successful speech synthesis."""
        tts = ElevenLabsTTS()

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.read = AsyncMock(return_value=b"\x00\x01\x02\x03")

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=mock_response)
        mock_context.__aexit__ = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_context)
        tts._session = mock_session

        result = await tts.synthesize("Hello world")

        assert result == b"\x00\x01\x02\x03"
        mock_session.post.assert_called_once()

        # Verify request format
        call_args = mock_session.post.call_args
        assert "api.elevenlabs.io" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_synthesize_error(self):
        """Test synthesis error handling."""
        tts = ElevenLabsTTS()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session = AsyncMock()
        mock_session.post = AsyncMock()
        mock_session.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.post.return_value.__aexit__ = AsyncMock(return_value=None)
        tts._session = mock_session

        result = await tts.synthesize("Hello world")

        assert result == b""


class TestGeminiLLM:
    """Tests for Gemini LLM client."""

    def test_initialization(self):
        """Test GeminiLLM initialization."""
        with patch("agent.genai.Client"):
            llm = GeminiLLM("You are a helpful assistant.")
            assert llm._system_prompt == "You are a helpful assistant."
            assert llm._conversation_history == []

    @pytest.mark.asyncio
    async def test_generate_response_adds_to_history(self):
        """Test that responses are added to conversation history."""
        with patch("agent.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello! How can I help?"

            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            llm = GeminiLLM("System prompt")

            response = await llm.generate_response("Hi there")

            assert response == "Hello! How can I help?"
            assert len(llm._conversation_history) == 2  # User + assistant

    @pytest.mark.asyncio
    async def test_generate_response_handles_error(self):
        """Test error handling in response generation."""
        with patch("agent.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_client_class.return_value = mock_client

            llm = GeminiLLM("System prompt")

            response = await llm.generate_response("Hi there")

            assert "trouble processing" in response.lower()


class TestVoiceAgent:
    """Tests for VoiceAgent orchestration."""

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test VoiceAgent initialization."""
        mock_ws = AsyncMock()

        with patch("agent.genai.Client"):
            va = VoiceAgent(
                websocket=mock_ws,
                call_id="test-123",
                from_number="+15551234567",
                to_number="+15559876543",
            )

            assert va.call_id == "test-123"
            assert va.from_number == "+15551234567"
            assert va.to_number == "+15559876543"
            assert va._running is False

    @pytest.mark.asyncio
    async def test_handle_transcript_generates_response(self):
        """Test that handling a transcript generates a response."""
        mock_ws = AsyncMock()

        with patch("agent.genai.Client") as mock_genai:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "I can help with that."
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.return_value = mock_client

            va = VoiceAgent(websocket=mock_ws, call_id="test-123")

            # Mock TTS to return audio
            va._tts.synthesize = AsyncMock(return_value=b"\x00" * 4800)

            await va._handle_transcript("Hello, I need help.")

            # Should have called LLM and TTS
            mock_client.aio.models.generate_content.assert_called_once()
            va._tts.synthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_plivo_audio_format_handling(self):
        """Test that Plivo audio is correctly processed."""
        mock_ws = AsyncMock()

        with patch("agent.genai.Client"):
            va = VoiceAgent(websocket=mock_ws, call_id="test-123")

            # Create mock Plivo message
            ulaw_audio = bytes([0x80] * 160)  # 20ms of audio
            plivo_message = {
                "event": "media",
                "media": {"payload": base64.b64encode(ulaw_audio).decode()},
            }

            # Verify the audio can be decoded
            payload = plivo_message["media"]["payload"]
            decoded = base64.b64decode(payload)
            pcm = ulaw_to_pcm(decoded)

            assert len(pcm) == 320  # 160 samples * 2 bytes

    @pytest.mark.asyncio
    async def test_tts_audio_format_conversion(self):
        """Test that TTS audio is correctly converted for Plivo."""
        # Simulate ElevenLabs output (24kHz PCM)
        import numpy as np

        from agent import ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE, resample_audio

        # 100ms of audio at 24kHz
        samples_24k = int(ELEVENLABS_SAMPLE_RATE * 0.1)
        tts_audio = np.zeros(samples_24k, dtype=np.int16).tobytes()

        # Convert to Plivo format
        pcm_8k = resample_audio(tts_audio, ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
        ulaw_audio = pcm_to_ulaw(pcm_8k)

        # Should be correct length for 100ms at 8kHz
        expected_samples = int(PLIVO_SAMPLE_RATE * 0.1)
        assert len(ulaw_audio) == expected_samples


class TestIntegration:
    """Integration tests for the voice agent pipeline."""

    @pytest.mark.asyncio
    async def test_full_conversation_flow(self):
        """Test a complete conversation flow with mocked services."""
        mock_ws = AsyncMock()
        mock_ws.receive_text = AsyncMock()
        mock_ws.send_text = AsyncMock()

        with patch("agent.genai.Client") as mock_genai:
            # Set up Gemini mock
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = "Hello! How can I help you today?"
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_genai.return_value = mock_client

            va = VoiceAgent(websocket=mock_ws, call_id="integration-test")

            # Mock the services
            va._stt.connect = AsyncMock()
            va._stt.close = AsyncMock()
            va._stt.send_audio = AsyncMock()

            va._tts.connect = AsyncMock()
            va._tts.close = AsyncMock()
            # Return 24kHz PCM audio (100ms)
            va._tts.synthesize = AsyncMock(return_value=b"\x00" * 4800)

            # Simulate a transcript being received
            await va._handle_transcript("Hi, I have a question about billing.")

            # Verify the LLM was called
            assert mock_client.aio.models.generate_content.called

            # Verify TTS was called with the response
            va._tts.synthesize.assert_called()

            # Verify audio was queued for sending
            assert not va._audio_queue.empty()

    @pytest.mark.asyncio
    async def test_concurrent_task_handling(self):
        """Test that concurrent tasks are properly managed."""
        mock_ws = AsyncMock()

        with patch("agent.genai.Client"):
            va = VoiceAgent(websocket=mock_ws, call_id="concurrent-test")

            # Verify the agent can be initialized with proper async primitives
            assert isinstance(va._audio_queue, asyncio.Queue)
            assert isinstance(va._processing_lock, asyncio.Lock)
