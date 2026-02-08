"""
End-to-end integration tests with real APIs.

These tests make actual API calls to:
- Google Gemini (LLM)
- Deepgram (STT)
- ElevenLabs (TTS)

Requires valid API keys in .env file.
"""

import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Skip all tests if API keys are not configured
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

pytestmark = pytest.mark.skipif(
    not all([GEMINI_API_KEY, DEEPGRAM_API_KEY, ELEVENLABS_API_KEY]),
    reason="API keys not configured"
)


class TestGeminiLLMReal:
    """End-to-end tests for Gemini LLM with real API."""

    @pytest.mark.asyncio
    async def test_gemini_simple_response(self):
        """Test that Gemini generates a response."""
        from agent import GeminiLLM

        llm = GeminiLLM("You are a helpful assistant. Keep responses brief.")

        response = await llm.generate_response("What is 2 + 2?")

        assert response is not None
        assert len(response) > 0
        assert "4" in response or "four" in response.lower()
        print(f"\n[Gemini Response]: {response}")

    @pytest.mark.asyncio
    async def test_gemini_conversation_context(self):
        """Test that Gemini maintains conversation context."""
        from agent import GeminiLLM

        llm = GeminiLLM("You are a helpful assistant. Keep responses brief.")

        # First message
        response1 = await llm.generate_response("My name is Alice.")
        print(f"\n[Gemini Response 1]: {response1}")

        # Second message referencing first
        response2 = await llm.generate_response("What is my name?")
        print(f"[Gemini Response 2]: {response2}")

        assert "Alice" in response2 or "alice" in response2.lower()

    @pytest.mark.asyncio
    async def test_gemini_customer_service_prompt(self):
        """Test Gemini with the actual customer service prompt."""
        from agent import DEFAULT_SYSTEM_PROMPT, GeminiLLM

        llm = GeminiLLM(DEFAULT_SYSTEM_PROMPT)

        response = await llm.generate_response(
            "Hi, I'm interested in TechFlow Pro. How much does it cost?"
        )

        print(f"\n[Gemini Customer Service Response]: {response}")

        # Should mention the price (twelve dollars)
        assert response is not None
        assert len(response) > 0
        # Response should be conversational
        assert any(word in response.lower() for word in ["twelve", "12", "dollar", "month", "help"])


class TestDeepgramSTTReal:
    """End-to-end tests for Deepgram STT with real API."""

    @pytest.mark.asyncio
    async def test_deepgram_connection(self):
        """Test that we can connect to Deepgram."""
        from agent import DeepgramSTT

        transcript_received = asyncio.Event()
        received_text = []

        async def on_transcript(text):
            received_text.append(text)
            transcript_received.set()

        stt = DeepgramSTT(on_transcript=on_transcript)

        try:
            await stt.connect()
            assert stt._running is True
            assert stt._ws is not None
            print("\n[Deepgram]: Connected successfully")
        finally:
            await stt.close()

    @pytest.mark.asyncio
    async def test_deepgram_transcribe_audio(self):
        """Test transcription with synthesized audio containing speech pattern."""
        from agent import DeepgramSTT

        transcript_received = asyncio.Event()
        received_text = []

        async def on_transcript(text):
            print(f"[Deepgram Transcript]: {text}")
            received_text.append(text)
            transcript_received.set()

        stt = DeepgramSTT(on_transcript=on_transcript)

        try:
            await stt.connect()

            # Generate a tone that simulates speech-like audio
            # This won't produce meaningful transcription but tests the pipeline
            sample_rate = 8000
            duration = 2.0  # 2 seconds
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

            # Create audio with varying frequencies (speech-like modulation)
            audio = np.zeros_like(t)
            for freq in [200, 400, 800, 1200]:
                audio += np.sin(2 * np.pi * freq * t) * np.sin(2 * np.pi * 3 * t)

            audio = (audio / np.max(np.abs(audio)) * 16000).astype(np.int16)
            audio_bytes = audio.tobytes()

            # Send audio in chunks
            chunk_size = 640  # 40ms chunks
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                await stt.send_audio(chunk)
                await asyncio.sleep(0.02)

            # Wait briefly for any transcription (may be empty for synthetic audio)
            try:
                await asyncio.wait_for(transcript_received.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                print("[Deepgram]: No transcription (expected for synthetic audio)")

            print(f"[Deepgram]: Audio sent successfully, transcripts: {received_text}")

        finally:
            await stt.close()


class TestElevenLabsTTSReal:
    """End-to-end tests for ElevenLabs TTS with real API."""

    @pytest.mark.asyncio
    async def test_elevenlabs_synthesize_short_text(self):
        """Test synthesizing short text."""
        from agent import ElevenLabsTTS

        tts = ElevenLabsTTS()

        try:
            await tts.connect()

            audio = await tts.synthesize("Hello, how can I help you today?")

            assert audio is not None
            assert len(audio) > 0

            # ElevenLabs returns 24kHz 16-bit PCM
            # 1 second = 24000 samples * 2 bytes = 48000 bytes
            # A short phrase should be at least 0.5 seconds
            assert len(audio) > 24000  # At least 0.5 seconds

            print(f"\n[ElevenLabs]: Synthesized {len(audio)} bytes of audio")
            print(f"[ElevenLabs]: Duration ~{len(audio) / 48000:.2f} seconds")

        finally:
            await tts.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_synthesize_longer_text(self):
        """Test synthesizing longer text."""
        from agent import ElevenLabsTTS

        tts = ElevenLabsTTS()

        try:
            await tts.connect()

            text = (
                "Thank you for calling TechFlow support. "
                "I understand you have a question about our pricing. "
                "TechFlow Pro costs twelve dollars per month for individuals."
            )

            audio = await tts.synthesize(text)

            assert audio is not None
            assert len(audio) > 0

            # Longer text should produce more audio
            duration_seconds = len(audio) / 48000
            assert duration_seconds > 2.0  # At least 2 seconds for this text

            print(f"\n[ElevenLabs]: Synthesized {len(audio)} bytes of audio")
            print(f"[ElevenLabs]: Duration ~{duration_seconds:.2f} seconds")

        finally:
            await tts.close()

    @pytest.mark.asyncio
    async def test_elevenlabs_audio_format(self):
        """Test that ElevenLabs returns valid PCM audio."""
        from agent import ElevenLabsTTS

        tts = ElevenLabsTTS()

        try:
            await tts.connect()

            audio = await tts.synthesize("Test audio format.")

            # Parse as 16-bit PCM
            samples = np.frombuffer(audio, dtype=np.int16)

            # Should have valid sample values
            assert len(samples) > 0
            assert samples.min() >= -32768
            assert samples.max() <= 32767

            # Should have some dynamic range (not silence)
            assert np.std(samples) > 100

            print(f"\n[ElevenLabs]: Audio has {len(samples)} samples")
            print(f"[ElevenLabs]: Sample range: [{samples.min()}, {samples.max()}]")
            print(f"[ElevenLabs]: Standard deviation: {np.std(samples):.2f}")

        finally:
            await tts.close()


class TestFullPipelineReal:
    """End-to-end tests for the full voice agent pipeline."""

    @pytest.mark.asyncio
    async def test_llm_to_tts_pipeline(self):
        """Test generating a response and converting to speech."""
        from agent import (
            ELEVENLABS_SAMPLE_RATE,
            PLIVO_SAMPLE_RATE,
            ElevenLabsTTS,
            GeminiLLM,
            pcm_to_ulaw,
            resample_audio,
        )

        llm = GeminiLLM("You are a helpful assistant. Keep responses to one sentence.")
        tts = ElevenLabsTTS()

        try:
            await tts.connect()

            # Generate response
            response = await llm.generate_response("Say hello in a friendly way.")
            print(f"\n[Pipeline] LLM Response: {response}")

            assert response is not None
            assert len(response) > 0

            # Synthesize to speech
            tts_audio = await tts.synthesize(response)
            print(f"[Pipeline] TTS Audio: {len(tts_audio)} bytes")

            assert tts_audio is not None
            assert len(tts_audio) > 0

            # Convert to Plivo format
            pcm_8k = resample_audio(tts_audio, ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
            ulaw_audio = pcm_to_ulaw(pcm_8k)

            print(f"[Pipeline] Plivo Audio (u-law 8kHz): {len(ulaw_audio)} bytes")

            # Verify final format
            assert len(ulaw_audio) > 0
            # u-law is 1 byte per sample, 8kHz
            duration_seconds = len(ulaw_audio) / PLIVO_SAMPLE_RATE
            print(f"[Pipeline] Final duration: {duration_seconds:.2f} seconds")

            assert duration_seconds > 0.5  # At least half a second

        finally:
            await tts.close()

    @pytest.mark.asyncio
    async def test_audio_conversion_pipeline(self):
        """Test audio format conversions work correctly with real TTS output."""
        from agent import (
            ELEVENLABS_SAMPLE_RATE,
            PLIVO_SAMPLE_RATE,
            ElevenLabsTTS,
            pcm_to_ulaw,
            resample_audio,
            ulaw_to_pcm,
        )

        tts = ElevenLabsTTS()

        try:
            await tts.connect()

            # Get real TTS audio
            original_audio = await tts.synthesize("Testing audio conversion pipeline.")
            print(f"\n[Conversion] Original (24kHz PCM): {len(original_audio)} bytes")

            # Convert: 24kHz PCM -> 8kHz PCM
            pcm_8k = resample_audio(original_audio, ELEVENLABS_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
            print(f"[Conversion] Downsampled (8kHz PCM): {len(pcm_8k)} bytes")

            # Ratio should be close to 3:1
            ratio = len(original_audio) / len(pcm_8k)
            assert 2.9 < ratio < 3.1

            # Convert: 8kHz PCM -> 8kHz u-law
            ulaw = pcm_to_ulaw(pcm_8k)
            print(f"[Conversion] Encoded (8kHz u-law): {len(ulaw)} bytes")

            # u-law is 1 byte per sample vs 2 bytes for PCM
            assert len(ulaw) == len(pcm_8k) // 2

            # Convert back: u-law -> PCM (for verification)
            pcm_recovered = ulaw_to_pcm(ulaw)

            # Compare original downsampled with recovered
            orig_samples = np.frombuffer(pcm_8k, dtype=np.int16).astype(float)
            recovered_samples = np.frombuffer(pcm_recovered, dtype=np.int16).astype(float)

            correlation = np.corrcoef(orig_samples, recovered_samples)[0, 1]
            print(f"[Conversion] Roundtrip correlation: {correlation:.4f}")

            assert correlation > 0.99  # Should be nearly identical

        finally:
            await tts.close()

    @pytest.mark.asyncio
    async def test_conversation_flow(self):
        """Test a multi-turn conversation flow."""
        from agent import DEFAULT_SYSTEM_PROMPT, ElevenLabsTTS, GeminiLLM

        llm = GeminiLLM(DEFAULT_SYSTEM_PROMPT)
        tts = ElevenLabsTTS()

        try:
            await tts.connect()

            # Simulate a conversation
            conversation = [
                "Hi, I'm interested in your products.",
                "What's the difference between Pro and Teams?",
                "How much does Teams cost?",
            ]

            print("\n[Conversation Flow Test]")
            print("=" * 50)

            for i, user_input in enumerate(conversation, 1):
                print(f"\n[Turn {i}] User: {user_input}")

                # Get LLM response
                response = await llm.generate_response(user_input)
                print(f"[Turn {i}] Agent: {response}")

                # Synthesize response
                audio = await tts.synthesize(response)
                duration = len(audio) / 48000
                print(f"[Turn {i}] Audio: {duration:.2f}s")

                assert response is not None
                assert len(audio) > 0

            print("\n" + "=" * 50)
            print("[Conversation Flow] All turns completed successfully")

        finally:
            await tts.close()
