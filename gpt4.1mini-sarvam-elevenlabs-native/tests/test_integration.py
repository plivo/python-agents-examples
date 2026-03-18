"""
Integration tests for GPT-4.1 Mini + Sarvam + ElevenLabs Voice Agent.

Test Levels:
1. Unit Tests - Test individual components (audio conversion, phone normalization)
2. Local Integration - Test WebSocket flow without external services
3. API Integration - Test OpenAI, Sarvam, ElevenLabs API connections
4. Plivo Integration - Test Plivo API configuration

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
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

import httpx
import plivo
import pytest
import websockets
from dotenv import load_dotenv

# Import audio functions from utils module
from utils import pcm_to_ulaw, ulaw_to_pcm

load_dotenv()

# Configuration from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
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
        """Test μ-law to PCM conversion."""
        ulaw_silence = b"\xff" * 160
        pcm_audio = ulaw_to_pcm(ulaw_silence)

        samples = struct.unpack(f"{len(pcm_audio) // 2}h", pcm_audio)
        avg_amplitude = sum(abs(s) for s in samples) / len(samples)

        assert len(pcm_audio) == 320  # 160 samples * 2 bytes
        assert avg_amplitude < 100  # Should be near silence

    def test_pcm_to_ulaw_conversion(self):
        """Test PCM to μ-law conversion."""
        pcm_silence = b"\x00" * 320
        ulaw_audio = pcm_to_ulaw(pcm_silence)

        assert len(ulaw_audio) == 160  # Half the size

    def test_audio_roundtrip(self):
        """Test that audio survives roundtrip conversion."""
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


    def test_plivo_to_sarvam_streaming_no_resample(self):
        """plivo_to_sarvam_streaming converts μ-law 8kHz to PCM16 8kHz (no resample)."""
        from utils import plivo_to_sarvam_streaming, ulaw_to_pcm

        # 160 bytes μ-law = 20ms at 8kHz (one Plivo packet)
        mulaw_data = b"\xff" * 160
        result = plivo_to_sarvam_streaming(mulaw_data)

        # Should be PCM16 at 8kHz — same number of samples, 2 bytes each
        assert len(result) == 320  # 160 samples * 2 bytes
        # Should match raw ulaw_to_pcm (no resampling)
        assert result == ulaw_to_pcm(mulaw_data)


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        """Test normalizing E.164 formatted numbers."""
        import phonenumbers

        phone = "+16572338892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"

    def test_normalize_with_spaces(self):
        """Test normalizing numbers with spaces."""
        import phonenumbers

        phone = "+1 657-233-8892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"

    def test_normalize_local_format(self):
        """Test normalizing local format numbers."""
        import phonenumbers

        phone = "(657) 233-8892"
        parsed = phonenumbers.parse(phone, "US")
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)

        assert e164 == "+16572338892"


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
    async def test_openai_chat_completion(self, openai_configured):
        """Test basic OpenAI chat completion."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4.1-mini",
                    "messages": [{"role": "user", "content": "Say hello briefly."}],
                    "max_tokens": 50,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["choices"]) > 0
            assert data["choices"][0]["message"]["content"]


class TestSarvamIntegration:
    """Integration tests for Sarvam.ai STT API."""

    @pytest.fixture
    def sarvam_configured(self):
        """Check if Sarvam API is configured."""
        if not SARVAM_API_KEY:
            pytest.skip("SARVAM_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_sarvam_stt(self, sarvam_configured):
        """Test basic Sarvam speech-to-text with a silent WAV."""
        import io
        import wave

        # Create a short WAV file with silence (Sarvam should return empty/short transcript)
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00" * 32000)  # 1 second of silence
        wav_bytes = wav_buf.getvalue()

        sarvam_url = os.getenv(
            "SARVAM_STT_URL", "https://api.sarvam.ai/speech-to-text"
        )
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                sarvam_url,
                headers={"api-subscription-key": SARVAM_API_KEY},
                files={"file": ("audio.wav", wav_bytes, "audio/wav")},
                data={
                    "language_code": "en-IN",
                    "model": "saarika:v2.5",
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "transcript" in data


class TestElevenLabsIntegration:
    """Integration tests for ElevenLabs API."""

    @pytest.fixture
    def elevenlabs_configured(self):
        """Check if ElevenLabs API is configured."""
        if not ELEVENLABS_API_KEY:
            pytest.skip("ELEVENLABS_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_elevenlabs_tts(self, elevenlabs_configured):
        """Test basic ElevenLabs text-to-speech."""
        voice_id = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            f"?output_format=pcm_24000"
        )
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                url,
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "text": "Hello, this is a test.",
                    "model_id": "eleven_flash_v2_5",
                },
            )
            assert response.status_code == 200
            assert len(response.content) > 1000, "TTS audio too short"


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
        import phonenumbers

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
# UNIT TESTS - SarvamStreamingSTT
# =============================================================================


class TestUnitSarvamStreamingSTT:
    """Unit tests for SarvamStreamingSTT message parsing."""

    @pytest.mark.asyncio
    async def test_transcript_accumulation(self):
        """Verify transcript parts accumulate correctly."""
        from inbound.agent import SarvamStreamingSTT

        stt = SarvamStreamingSTT()
        # Simulate transcript parts (without connecting)
        stt._transcript_parts.append("Hello")
        stt._transcript_parts.append("how are you")
        assert stt.latest_transcript == "Hello how are you"

    @pytest.mark.asyncio
    async def test_clear_transcript(self):
        """Verify clear resets state."""
        from inbound.agent import SarvamStreamingSTT

        stt = SarvamStreamingSTT()
        stt._transcript_parts.append("test")
        stt.clear_transcript()
        assert stt.latest_transcript == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
