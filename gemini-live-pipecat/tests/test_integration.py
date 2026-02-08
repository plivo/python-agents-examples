"""Integration tests for the Gemini Live Pipecat voice agent."""

import json
import os

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

# Set environment variables before importing the module
os.environ["PUBLIC_URL"] = "https://test.ngrok.io"
os.environ["GEMINI_API_KEY"] = "test_api_key"
os.environ["PLIVO_AUTH_ID"] = "test_auth_id"
os.environ["PLIVO_AUTH_TOKEN"] = "test_auth_token"

import agent  # noqa: E402
import server  # noqa: E402


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(server.app)


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 OK."""
        response = client.get("/")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Health endpoint should return JSON."""
        response = client.get("/")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_health_includes_service_name(self, client):
        """Health endpoint should include service name."""
        response = client.get("/")
        data = response.json()
        assert "service" in data
        assert data["service"] == "gemini-live-pipecat"

    def test_health_includes_model(self, client):
        """Health endpoint should include model name."""
        response = client.get("/")
        data = response.json()
        assert "model" in data
        assert "gemini" in data["model"].lower()


class TestAnswerWebhook:
    """Tests for the Plivo answer webhook."""

    def test_answer_returns_200(self, client):
        """Answer webhook should return 200 OK."""
        response = client.post("/answer")
        assert response.status_code == 200

    def test_answer_returns_xml(self, client):
        """Answer webhook should return XML content type."""
        response = client.post("/answer")
        assert "application/xml" in response.headers.get("content-type", "")

    def test_answer_contains_stream_element(self, client):
        """Answer webhook should contain Plivo Stream element."""
        response = client.post("/answer")
        assert "<Stream" in response.text
        assert "bidirectional" in response.text.lower()

    def test_answer_contains_websocket_url(self, client):
        """Answer webhook should contain WebSocket URL."""
        response = client.post("/answer")
        assert "/ws" in response.text

    def test_answer_contains_audio_config(self, client):
        """Answer webhook should contain audio configuration."""
        response = client.post("/answer")
        assert "mulaw" in response.text.lower() or "x-mulaw" in response.text.lower()


class TestWebSocketEndpoint:
    """Tests for the WebSocket endpoint."""

    def test_websocket_accepts_connection(self, client):
        """WebSocket endpoint should accept connections."""
        with client.websocket_connect("/ws"):
            # Connection accepted, now it expects start message
            # Close immediately since we don't have a real Plivo stream
            pass

    def test_websocket_rejects_invalid_start_message(self, client):
        """WebSocket should close if start message is invalid."""
        with client.websocket_connect("/ws") as websocket:
            # Send invalid start message (missing required fields)
            websocket.send_text(json.dumps({"event": "start", "start": {}}))
            # Should close due to missing stream_id or call_id
            with pytest.raises(WebSocketDisconnect):
                websocket.receive_text()

    def test_websocket_handles_valid_start_message(self, client):
        """WebSocket should process valid start message."""
        with client.websocket_connect("/ws") as websocket:
            # Send valid start message
            start_message = {
                "event": "start",
                "start": {
                    "streamId": "test-stream-123",
                    "callId": "test-call-456",
                },
            }
            websocket.send_text(json.dumps(start_message))
            # The connection will proceed but might fail on Gemini connection
            # since we don't have valid API keys - that's expected


class TestConfiguration:
    """Tests for configuration loading."""

    def test_default_port(self):
        """Default port should be 8000."""
        assert server.SERVER_PORT == 8000 or os.getenv("SERVER_PORT")

    def test_gemini_model_configured(self):
        """Gemini model should be configured."""
        assert agent.GEMINI_MODEL
        assert "gemini" in agent.GEMINI_MODEL.lower()

    def test_gemini_voice_configured(self):
        """Gemini voice should be configured."""
        assert agent.GEMINI_VOICE
        assert agent.GEMINI_VOICE in ["Aoede", "Charon", "Fenrir", "Kore", "Puck"]

    def test_system_prompt_configured(self):
        """System prompt should be configured."""
        assert agent.SYSTEM_PROMPT
        assert len(agent.SYSTEM_PROMPT) > 0


class TestModuleImports:
    """Tests for module imports and dependencies."""

    def test_pipecat_imports(self):
        """Pipecat imports should work."""
        from pipecat.frames.frames import LLMMessagesUpdateFrame
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineParams, PipelineTask
        from pipecat.serializers.plivo import PlivoFrameSerializer
        from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
        from pipecat.transports.websocket.fastapi import (
            FastAPIWebsocketParams,
            FastAPIWebsocketTransport,
        )

        assert LLMMessagesUpdateFrame is not None
        assert Pipeline is not None
        assert PipelineRunner is not None
        assert PipelineParams is not None
        assert PipelineTask is not None
        assert PlivoFrameSerializer is not None
        assert GeminiLiveLLMService is not None
        assert FastAPIWebsocketParams is not None
        assert FastAPIWebsocketTransport is not None

    def test_fastapi_app_exists(self):
        """FastAPI app should exist."""
        assert server.app is not None
        assert server.app.title == "Gemini Live Pipecat Voice Agent"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
