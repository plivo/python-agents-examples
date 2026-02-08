"""
Integration tests for Grok-Plivo Voice Agent.

Test Levels:
1. Unit Tests - Test individual components (audio conversion, phone normalization)
2. Local Integration - Test WebSocket flow without external services
3. Grok Integration - Test xAI Grok Realtime API connection
4. Plivo Integration - Test Plivo API configuration

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
    uv run pytest tests/test_integration.py -v -k "grok"
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import os
import struct
import time
import uuid

import httpx
import plivo
import pytest
import websockets
from dotenv import load_dotenv

# Import audio functions from agent module
from agent import pcm_to_ulaw, ulaw_to_pcm

load_dotenv()

# Configuration from environment
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-fast-voice")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PUBLIC_URL = os.getenv("PUBLIC_URL", "")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

LOCAL_WS_URL = f"ws://localhost:{SERVER_PORT}/ws"
LOCAL_HTTP_URL = f"http://localhost:{SERVER_PORT}"


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

    @pytest.fixture
    def server_running(self):
        """Check if the local server is running."""
        try:
            response = httpx.get(LOCAL_HTTP_URL, timeout=2.0)
            return response.status_code == 200
        except Exception:
            pytest.skip("Local server not running. Start with: uv run python server.py")

    @pytest.mark.asyncio
    async def test_local_health_check(self, server_running):
        """Test the health check endpoint."""
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_local_answer_webhook(self, server_running):
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
    async def test_local_websocket_connection(self, server_running):
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
    async def test_local_audio_quality(self, server_running):
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
# GROK INTEGRATION TESTS
# =============================================================================


class TestGrokIntegration:
    """Integration tests for xAI Grok Realtime API."""

    @pytest.fixture
    def grok_configured(self):
        """Check if Grok API is configured."""
        if not XAI_API_KEY:
            pytest.skip("XAI_API_KEY not configured")

    @pytest.mark.asyncio
    async def test_grok_connection(self, grok_configured):
        """Test basic Grok Realtime API connection."""
        import websockets as ws_lib

        url = f"wss://api.x.ai/v1/realtime?model={GROK_MODEL}"
        headers = {"Authorization": f"Bearer {XAI_API_KEY}"}

        async with ws_lib.connect(url, additional_headers=headers, max_size=None) as ws:
            # Send session config
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": "Say hello briefly.",
                            "voice": "Sal",
                            "turn_detection": None,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                        },
                    }
                )
            )

            # Wait for session confirmation (skip conversation.created, ping, etc.)
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "session.updated":
                    break
                if msg.get("type") == "error":
                    pytest.fail(f"Grok error: {msg}")
            assert msg.get("type") == "session.updated"

    @pytest.mark.asyncio
    async def test_grok_text_to_audio(self, grok_configured):
        """Test sending text and receiving audio from Grok."""
        import websockets as ws_lib

        url = f"wss://api.x.ai/v1/realtime?model={GROK_MODEL}"
        headers = {"Authorization": f"Bearer {XAI_API_KEY}"}

        audio_chunks = []

        async with ws_lib.connect(url, additional_headers=headers, max_size=None) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "instructions": "Say hello briefly.",
                            "voice": "Sal",
                            "turn_detection": None,
                            "input_audio_format": "pcm16",
                            "output_audio_format": "pcm16",
                        },
                    }
                )
            )

            # Wait for session confirmation
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("type") == "session.updated":
                    break

            # Send text message
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Say hello"}],
                        },
                    }
                )
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {"modalities": ["text", "audio"]},
                    }
                )
            )

            try:
                async with asyncio.timeout(15):
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "response.output_audio.delta":
                            delta = msg.get("delta", "")
                            if delta:
                                audio_chunks.append(base64.b64decode(delta))
                        elif msg.get("type") == "response.done":
                            break
            except asyncio.TimeoutError:
                pass

        assert len(audio_chunks) > 0, "No audio received from Grok"
        total_audio = sum(len(chunk) for chunk in audio_chunks)
        assert total_audio > 1000, f"Audio too short: {total_audio} bytes"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
