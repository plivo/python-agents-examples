"""
Integration tests for GPT 5.4 Mini + Speechmatics + ElevenLabs Voice Agent.

Test Levels:
1. Unit Tests - Test individual components (audio conversion, phone normalization)
2. Local Integration - Test WebSocket flow without external services
3. API Integration - Test OpenAI, Speechmatics, ElevenLabs API connections
4. Plivo Integration - Test Plivo API configuration

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import signal
import struct
import subprocess
import sys
import time
import uuid

import aiohttp
import httpx
import numpy as np
import phonenumbers
import plivo
import pytest
import websockets
from dotenv import load_dotenv

from utils import (
    normalize_phone_number,
    pcm_to_ulaw,
    plivo_to_speechmatics,
    plivo_to_vad,
    ulaw_to_pcm,
)

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SPEECHMATICS_API_KEY = os.getenv("SPEECHMATICS_API_KEY", "")
SPEECHMATICS_PROFILE = os.getenv("SPEECHMATICS_PROFILE", "adaptive")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

TEST_PORT = 18001
LOCAL_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"

# =============================================================================
# UNIT TESTS - Test individual components
# =============================================================================


class TestUnitAudioConversion:
    """Unit tests for audio format conversion."""

    def test_ulaw_to_pcm_conversion(self):
        """Test μ-law to PCM conversion produces expected output size and low amplitude silence."""
        ulaw_silence = b"\xff" * 160
        pcm_audio = ulaw_to_pcm(ulaw_silence)

        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        avg_amplitude = sum(abs(s) for s in samples) / len(samples)

        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert avg_amplitude < 100  # Should be near silence

    def test_pcm_to_ulaw_conversion(self):
        """Test PCM to μ-law conversion produces half-size output."""
        pcm_silence = b"\x00" * 320
        ulaw_audio = pcm_to_ulaw(pcm_silence)

        assert len(ulaw_audio) == 160  # Half the size

    def test_audio_roundtrip(self):
        """Test that audio survives roundtrip conversion with correlation > 0.9."""
        samples = []
        for i in range(160):
            sample = int(16000 * math.sin(2 * math.pi * 440 * i / 8000))
            samples.append(sample)
        pcm_original = struct.pack(f"{len(samples)}h", *samples)

        ulaw = pcm_to_ulaw(pcm_original)
        pcm_restored = ulaw_to_pcm(ulaw)

        original_samples = struct.unpack(f"{len(pcm_original) // 2}h", pcm_original)
        restored_samples = struct.unpack(f"{len(pcm_restored) // 2}h", pcm_restored)

        # Check correlation (should be > 0.9)
        correlation = sum(o * r for o, r in zip(original_samples, restored_samples, strict=True))
        orig_energy = sum(o * o for o in original_samples)
        rest_energy = sum(r * r for r in restored_samples)

        if orig_energy > 0 and rest_energy > 0:
            normalized_corr = correlation / (orig_energy * rest_energy) ** 0.5
            assert normalized_corr > 0.9, "Audio quality degraded too much"

    def test_plivo_to_speechmatics_no_resample(self):
        """plivo_to_speechmatics converts μ-law 8kHz to PCM16 8kHz (no resample)."""
        # 160 bytes μ-law = 20ms at 8kHz (one Plivo packet)
        mulaw_data = b"\xff" * 160
        result = plivo_to_speechmatics(mulaw_data)

        # Should be PCM16 at 8kHz — same number of samples, 2 bytes each
        assert len(result) == 320  # 160 samples * 2 bytes
        # Should match raw ulaw_to_pcm (no resampling)
        assert result == ulaw_to_pcm(mulaw_data)

    def test_plivo_to_vad_resamples_to_16k(self):
        """plivo_to_vad returns float32 numpy array at 16kHz (2x input samples)."""
        mulaw_data = b"\xff" * 160
        result = plivo_to_vad(mulaw_data)

        # Output should be a float32 numpy array
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        # 160 samples at 8kHz -> 320 samples at 16kHz (2x)
        assert len(result) == 320


# =============================================================================
# UNIT TESTS - Phone number normalization
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        """Test normalizing E.164 formatted numbers."""
        phone = "+16572338892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"

    def test_normalize_with_spaces(self):
        """Test normalizing numbers with spaces."""
        phone = "+1 657-233-8892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"

    def test_normalize_local_format(self):
        """Test normalizing local format numbers."""
        result = normalize_phone_number("(657) 233-8892", "US")
        assert result == "16572338892"


# =============================================================================
# LOCAL INTEGRATION TESTS
# =============================================================================


class TestLocalIntegration:
    """Integration tests using local WebSocket connection."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start the inbound server as a subprocess."""
        env = os.environ.copy()
        env["SERVER_PORT"] = str(TEST_PORT)

        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        proc = subprocess.Popen(
            [sys.executable, "-m", "inbound.server"],
            cwd=project_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        ready = False
        for _ in range(30):
            try:
                resp = httpx.get(LOCAL_HTTP_URL, timeout=1.0)
                if resp.status_code == 200:
                    ready = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not ready:
            proc.terminate()
            proc.wait()
            output = proc.stdout.read().decode() if proc.stdout else ""
            pytest.skip(f"Server did not start in time. Output:\n{output[:2000]}")

        yield proc

        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    @pytest.mark.asyncio
    async def test_local_health_check(self, server_process):
        """Test the health check endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_local_answer_webhook(self, server_process):
        """Test the answer webhook returns valid XML."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/answer",
                params={"CallUUID": "test123", "From": "+15551234567", "To": "+16572338892"},
            )
            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            assert "<Stream" in response.text
            assert "bidirectional" in response.text

    @pytest.mark.asyncio
    async def test_local_websocket_connection(self, server_process):
        """Test WebSocket connection and audio reception."""
        body_data = {"call_uuid": "test123", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{LOCAL_WS_URL}?body={body_b64}"

        async with websockets.connect(ws_url, close_timeout=2) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))

            audio_received = False
            try:
                async with asyncio.timeout(10):
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        if data.get("event") == "playAudio":
                            audio_received = True
                            break
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

            assert audio_received, "No audio received from server"

    @pytest.mark.asyncio
    async def test_local_audio_quality(self, server_process):
        """Test audio quality from the agent."""
        body_data = {"call_uuid": "test123", "from": "+15551234567", "to": "+16572338892"}
        body_b64 = base64.b64encode(json.dumps(body_data).encode()).decode()
        ws_url = f"{LOCAL_WS_URL}?body={body_b64}"

        audio_chunks = []

        async with websockets.connect(ws_url, close_timeout=2) as ws:
            start_event = {
                "event": "start",
                "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
            }
            await ws.send(json.dumps(start_event))

            start_time = time.time()
            while time.time() - start_time < 15:
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.5)
                    data = json.loads(message)
                    if data.get("event") == "playAudio":
                        payload = data.get("media", {}).get("payload", "")
                        if payload:
                            audio_chunks.append(base64.b64decode(payload))
                except asyncio.TimeoutError:
                    silence = base64.b64encode(b"\xff" * 160).decode()
                    await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))
                except websockets.exceptions.ConnectionClosed:
                    break

                if len(audio_chunks) >= 20:
                    break

        assert len(audio_chunks) > 0, "No audio chunks received"

        combined_audio = b"".join(audio_chunks)
        pcm_audio = ulaw_to_pcm(combined_audio)
        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)

        rms = (sum(s**2 for s in samples) / len(samples)) ** 0.5
        assert rms > 500, f"Audio RMS {rms} too low - may be silence"


# =============================================================================
# API INTEGRATION TESTS
# =============================================================================


class TestOpenAIIntegration:
    """Integration tests for OpenAI API."""

    @pytest.fixture
    def openai_configured(self):
        """Check if OpenAI API is configured."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_openai_streaming_chat_completion(self, openai_configured):
        """Test OpenAI streaming chat completion with GPT 5.4 Mini."""
        chunks = []
        async with (
            httpx.AsyncClient(timeout=30.0) as client,
            client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-5.4-mini",
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "max_completion_tokens": 50,
                    "stream": True,
                },
            ) as response,
        ):
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    if delta.get("content"):
                        chunks.append(delta["content"])

        assert len(chunks) > 0, "No streaming chunks received"
        full_text = "".join(chunks)
        assert len(full_text) > 0, "Empty response from GPT 5.4 Mini"


class TestSpeechmaticsIntegration:
    """Integration tests for Speechmatics real-time STT API."""

    @pytest.fixture
    def speechmatics_configured(self):
        """Check if Speechmatics API is configured."""
        if not SPEECHMATICS_API_KEY:
            pytest.skip("SPEECHMATICS_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_speechmatics_websocket_connection(self, speechmatics_configured):
        """Test connecting to Speechmatics WebSocket and completing a recognition session."""
        ws_url = f"wss://preview.rt.speechmatics.com/v2/agent/{SPEECHMATICS_PROFILE}"
        headers = {"Authorization": f"Bearer {SPEECHMATICS_API_KEY}"}

        start_recognition = {
            "message": "StartRecognition",
            "audio_format": {
                "type": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 8000,
            },
            "transcription_config": {
                "language": "en",
                "enable_partials": False,
            },
        }

        recognition_started = False
        end_of_transcript = False

        async with websockets.connect(
            ws_url, additional_headers=headers, close_timeout=10
        ) as ws:
            await ws.send(json.dumps(start_recognition))

            # Wait for RecognitionStarted
            async with asyncio.timeout(15):
                while True:
                    message = await ws.recv()
                    data = json.loads(message)
                    if data.get("message") == "RecognitionStarted":
                        recognition_started = True
                        break

            assert recognition_started, "Did not receive RecognitionStarted"

            # Send 1 second of silence PCM16 at 8kHz (8000 samples * 2 bytes = 16000 bytes)
            silence_pcm = b"\x00" * 16000
            await ws.send(silence_pcm)

            # Send EndOfStream
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": 1}))

            # Wait for EndOfTranscript
            try:
                async with asyncio.timeout(15):
                    while True:
                        message = await ws.recv()
                        data = json.loads(message)
                        if data.get("message") == "EndOfTranscript":
                            end_of_transcript = True
                            break
            except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                pass

        assert end_of_transcript, "Did not receive EndOfTranscript"


class TestElevenLabsIntegration:
    """Integration tests for ElevenLabs WebSocket TTS API."""

    @pytest.fixture
    def elevenlabs_configured(self):
        """Check if ElevenLabs API is configured."""
        if not ELEVENLABS_API_KEY:
            pytest.skip("ELEVENLABS_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_elevenlabs_websocket_tts(self, elevenlabs_configured):
        """Test ElevenLabs WebSocket text-to-speech streaming."""
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
        ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech"
            f"/{voice_id}/stream-input"
            f"?model_id={model_id}"
            f"&output_format=pcm_24000"
        )
        audio_chunks = []
        async with aiohttp.ClientSession() as session, session.ws_connect(
            ws_url, timeout=30.0
        ) as ws:
            # BOS message
            await ws.send_json({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": ELEVENLABS_API_KEY,
            })
            # Send text in two sentence chunks
            await ws.send_json({"text": "Hello, this is a test. "})
            await ws.send_json({"text": "How are you doing today? "})
            # EOS — flush
            await ws.send_json({"text": ""})

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        audio_chunks.append(base64.b64decode(audio_b64))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

        total_bytes = sum(len(c) for c in audio_chunks)
        assert len(audio_chunks) > 0, "No audio chunks received"
        assert total_bytes > 1000, f"TTS audio too short ({total_bytes} bytes)"

    @pytest.mark.asyncio
    async def test_elevenlabs_websocket_sentence_chunking(self, elevenlabs_configured):
        """Test that sentence-chunked input produces audio progressively."""
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
        ws_url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech"
            f"/{voice_id}/stream-input"
            f"?model_id={model_id}"
            f"&output_format=pcm_24000"
        )
        first_audio_time = None
        t0 = time.monotonic()
        total_bytes = 0

        async with aiohttp.ClientSession() as session, session.ws_connect(
            ws_url, timeout=30.0
        ) as ws:
            await ws.send_json({
                "text": " ",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.8},
                "xi_api_key": ELEVENLABS_API_KEY,
            })
            # Send three sentences with small delays to simulate LLM streaming
            for sentence in [
                "Welcome to our customer support line. ",
                "We are happy to help you today. ",
                "Please tell me how I can assist you. ",
            ]:
                await ws.send_json({"text": sentence})
                await asyncio.sleep(0.05)
            await ws.send_json({"text": ""})

            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    audio_b64 = data.get("audio")
                    if audio_b64:
                        if first_audio_time is None:
                            first_audio_time = time.monotonic()
                        total_bytes += len(base64.b64decode(audio_b64))
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break

        assert first_audio_time is not None, "No audio received"
        ttfb_ms = (first_audio_time - t0) * 1000
        # WebSocket TTFB should be reasonable (under 5s even on slow connections)
        assert ttfb_ms < 5000, f"TTFB too high: {ttfb_ms:.0f}ms"
        assert total_bytes > 2000, f"Audio too short for 3 sentences ({total_bytes} bytes)"


# =============================================================================
# PLIVO INTEGRATION TESTS
# =============================================================================


class TestPlivoIntegration:
    """Integration tests for Plivo API."""

    @pytest.fixture
    def plivo_configured(self):
        """Check if Plivo is configured."""
        if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
            pytest.skip("Plivo credentials not configured")

    def test_plivo_credentials_valid(self, plivo_configured):
        """Test that Plivo credentials are valid."""
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        account = client.account.get()
        assert account is not None

    def test_plivo_phone_number_exists(self, plivo_configured):
        """Test that the configured phone number exists."""
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        parsed = phonenumbers.parse(PLIVO_PHONE_NUMBER, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        number_digits = e164.lstrip("+")

        try:
            number = client.numbers.get(number=number_digits)
            assert number is not None
        except plivo.exceptions.ResourceNotFoundError:
            pytest.fail(f"Phone number {PLIVO_PHONE_NUMBER} not found")


# =============================================================================
# UNIT TESTS - SpeechmaticsSTT
# =============================================================================


class TestUnitSpeechmaticsSTT:
    """Unit tests for SpeechmaticsSTT message parsing."""

    def test_transcript_accumulation(self):
        """Verify transcript parts accumulate correctly."""
        from inbound.agent import SpeechmaticsSTT

        stt = SpeechmaticsSTT()
        stt._transcript_parts.append("Hello")
        stt._transcript_parts.append("how are you")
        assert stt.latest_transcript == "Hello how are you"

    def test_clear_transcript(self):
        """Verify clear resets state."""
        from inbound.agent import SpeechmaticsSTT

        stt = SpeechmaticsSTT()
        stt._transcript_parts.append("test")
        stt.clear_transcript()
        assert stt.latest_transcript == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
