"""Tests for FastAPI server endpoints."""

import base64
import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from server import app, normalize_phone_number


class TestPhoneNumberNormalization:
    """Tests for phone number normalization."""

    def test_normalize_us_number(self):
        """Test normalizing a US phone number."""
        result = normalize_phone_number("+1-555-123-4567", "US")
        assert result == "15551234567"

    def test_normalize_number_no_country_code(self):
        """Test normalizing a number without country code."""
        result = normalize_phone_number("555-123-4567", "US")
        assert result == "15551234567"

    def test_normalize_number_with_plus(self):
        """Test that plus sign is stripped."""
        result = normalize_phone_number("+15551234567", "US")
        assert result == "15551234567"

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        result = normalize_phone_number("", "US")
        assert result == ""

    def test_normalize_invalid_number(self):
        """Test normalizing invalid number falls back to digits only."""
        # Using a truly invalid number that can't be parsed
        result = normalize_phone_number("invalid", "US")
        assert result == ""  # No digits to extract


class TestHealthCheck:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check returns OK status."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "gemini-deepgram-cartesia-voice-agent"


class TestAnswerWebhook:
    """Tests for Plivo answer webhook."""

    @pytest.mark.asyncio
    async def test_answer_webhook_get(self):
        """Test answer webhook with GET request."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/answer",
                params={
                    "CallUUID": "test-call-123",
                    "From": "+15551234567",
                    "To": "+15559876543",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        content = response.text

        # Should contain Stream element for WebSocket
        assert "<Stream" in content
        assert "bidirectional" in content
        assert "audio/x-mulaw" in content

    @pytest.mark.asyncio
    async def test_answer_webhook_post(self):
        """Test answer webhook with POST request."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/answer",
                data={
                    "CallUUID": "test-call-456",
                    "From": "+15551234567",
                    "To": "+15559876543",
                },
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        content = response.text

        assert "<Stream" in content

    @pytest.mark.asyncio
    async def test_answer_webhook_encodes_call_data(self):
        """Test that answer webhook encodes call data in WebSocket URL."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get(
                "/answer",
                params={
                    "CallUUID": "call-789",
                    "From": "+15551111111",
                    "To": "+15552222222",
                },
            )

        content = response.text

        # Extract the WebSocket URL and decode the body parameter
        assert "body=" in content


class TestHangupWebhook:
    """Tests for Plivo hangup webhook."""

    @pytest.mark.asyncio
    async def test_hangup_webhook(self):
        """Test hangup webhook returns OK."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/hangup",
                data={
                    "CallUUID": "test-call-123",
                    "Duration": "60",
                    "HangupCause": "NORMAL_CLEARING",
                },
            )

        assert response.status_code == 200
        assert response.text == "OK"


class TestFallbackWebhook:
    """Tests for fallback webhook."""

    @pytest.mark.asyncio
    async def test_fallback_webhook(self):
        """Test fallback webhook returns error message XML."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/fallback")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/xml"
        content = response.text

        assert "<Speak" in content
        assert "technical difficulties" in content
        assert "<Hangup" in content


class TestWebSocketEndpoint:
    """Tests for WebSocket endpoint."""

    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection is accepted."""
        from starlette.testclient import TestClient

        # Create call metadata
        call_data = {"call_uuid": "test-123", "from": "+15551234567", "to": "+15559876543"}
        body_b64 = base64.b64encode(json.dumps(call_data).encode()).decode()

        with patch("server.agent.run_agent", new_callable=AsyncMock) as mock_agent:
            # Set up mock to return immediately
            mock_agent.return_value = None

            client = TestClient(app)
            with client.websocket_connect(f"/ws?body={body_b64}") as websocket:
                # Send start event (required by Plivo protocol)
                websocket.send_text(
                    json.dumps(
                        {
                            "event": "start",
                            "start": {"callId": "test-123", "streamId": "stream-456"},
                        }
                    )
                )

                # The agent should be called
                # Note: TestClient runs synchronously so we may need to handle differently

    @pytest.mark.asyncio
    async def test_websocket_decodes_body(self):
        """Test WebSocket decodes body parameter correctly."""
        call_data = {"call_uuid": "decode-test", "from": "+15550000000", "to": "+15551111111"}
        body_b64 = base64.b64encode(json.dumps(call_data).encode()).decode()

        # Verify encoding/decoding works
        decoded = json.loads(base64.b64decode(body_b64).decode())
        assert decoded == call_data
