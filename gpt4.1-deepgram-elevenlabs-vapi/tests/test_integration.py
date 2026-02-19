"""
Integration tests for GPT-4.1 Vapi Voice Agent.

Test Levels:
1. Unit Tests - Test individual components (phone normalization, call manager)
2. Local Integration - Test server endpoints and webhook handling
3. Vapi Integration - Test Vapi API connection and assistant creation
4. Plivo Integration - Test Plivo API credentials

Run tests:
    uv run pytest tests/test_integration.py -v

Run specific test level:
    uv run pytest tests/test_integration.py -v -k "unit"
    uv run pytest tests/test_integration.py -v -k "local"
    uv run pytest tests/test_integration.py -v -k "vapi"
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest
from dotenv import load_dotenv

from utils import normalize_phone_number

load_dotenv()

# Configuration from environment
VAPI_PRIVATE_KEY = os.getenv("VAPI_PRIVATE_KEY", "")
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")

TEST_PORT = 18001
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"


# =============================================================================
# UNIT TESTS - Test individual components
# =============================================================================


class TestUnitPhoneNormalization:
    """Unit tests for phone number normalization."""

    def test_normalize_e164_format(self):
        """Test normalizing E.164 formatted numbers."""
        result = normalize_phone_number("+16572338892")
        assert result == "16572338892"

    def test_normalize_with_spaces(self):
        """Test normalizing numbers with spaces."""
        result = normalize_phone_number("+1 657-233-8892")
        assert result == "16572338892"

    def test_normalize_local_format(self):
        """Test normalizing local format numbers."""
        result = normalize_phone_number("(657) 233-8892")
        assert result == "16572338892"

    def test_normalize_empty_string(self):
        """Test normalizing empty string."""
        result = normalize_phone_number("")
        assert result == ""


class TestUnitCallManager:
    """Unit tests for outbound call manager."""

    def test_create_call(self):
        """Test creating an outbound call record."""
        from outbound.agent import CallManager

        manager = CallManager()
        record = manager.create_call(
            phone_number="+15551234567",
            campaign_id="test-campaign",
            opening_reason="your free trial sign-up",
            objective="qualify the lead",
        )

        assert record.call_id
        assert record.phone_number == "+15551234567"
        assert record.campaign_id == "test-campaign"
        assert record.status == "initiating"
        assert "free trial" in record.system_prompt
        assert "free trial" in record.initial_message

    def test_update_status(self):
        """Test updating call record status."""
        from outbound.agent import CallManager

        manager = CallManager()
        record = manager.create_call(phone_number="+15551234567")

        updated = manager.update_status(record.call_id, "ringing", vapi_call_id="vapi-123")
        assert updated is not None
        assert updated.status == "ringing"
        assert updated.vapi_call_id == "vapi-123"

    def test_get_call_by_vapi_id(self):
        """Test looking up a call by Vapi call ID."""
        from outbound.agent import CallManager

        manager = CallManager()
        record = manager.create_call(phone_number="+15551234567")
        manager.update_status(record.call_id, "ringing", vapi_call_id="vapi-456")

        found = manager.get_call_by_vapi_id("vapi-456")
        assert found is not None
        assert found.call_id == record.call_id

    def test_get_active_calls(self):
        """Test getting active calls."""
        from outbound.agent import CallManager

        manager = CallManager()
        r1 = manager.create_call(phone_number="+15551111111")
        r2 = manager.create_call(phone_number="+15552222222")
        manager.update_status(r1.call_id, "connected")
        manager.update_status(r2.call_id, "completed")

        active = manager.get_active_calls()
        assert len(active) == 1
        assert active[0].call_id == r1.call_id

    def test_determine_outcome(self):
        """Test outcome determination from ended reason."""
        from outbound.agent import determine_outcome

        assert determine_outcome("customer-did-not-answer", 0) == "no_answer"
        assert determine_outcome("customer-busy", 0) == "busy"
        assert determine_outcome("error-some-error", 0) == "failed"
        assert determine_outcome("customer-ended-call", 30) == "success"
        assert determine_outcome("assistant-ended-call", 45) == "success"

    def test_campaign_calls(self):
        """Test getting calls by campaign."""
        from outbound.agent import CallManager

        manager = CallManager()
        manager.create_call(phone_number="+15551111111", campaign_id="camp-1")
        manager.create_call(phone_number="+15552222222", campaign_id="camp-1")
        manager.create_call(phone_number="+15553333333", campaign_id="camp-2")

        camp1 = manager.get_calls_by_campaign("camp-1")
        assert len(camp1) == 2


class TestUnitAssistantConfig:
    """Unit tests for Vapi assistant configuration."""

    def test_build_inbound_config(self):
        """Test building inbound assistant config."""
        from inbound.agent import build_assistant_config

        config = build_assistant_config(
            server_url="https://example.com/vapi/webhook",
            from_number="+15551234567",
        )

        assert config["firstMessage"]
        assert config["transcriber"]["provider"] == "deepgram"
        assert config["model"]["provider"] == "openai"
        assert config["model"]["model"] == "gpt-4.1"
        assert config["voice"]["provider"] == "11labs"
        assert config["serverUrl"] == "https://example.com/vapi/webhook"
        assert config["backgroundDenoisingEnabled"] is True
        assert "stopSpeakingPlan" in config
        assert "startSpeakingPlan" in config
        assert "transcriptionEndpointingPlan" in config["startSpeakingPlan"]

    def test_build_outbound_config(self):
        """Test building outbound assistant config."""
        from outbound.agent import CallManager, build_outbound_assistant_config

        manager = CallManager()
        record = manager.create_call(
            phone_number="+15551234567",
            opening_reason="your free trial",
            objective="qualify lead",
        )

        config = build_outbound_assistant_config(
            server_url="https://example.com/vapi/webhook",
            record=record,
        )

        assert config["firstMessage"]
        assert "free trial" in config["firstMessage"]
        assert config["transcriber"]["provider"] == "deepgram"
        assert config["model"]["provider"] == "openai"
        assert config["voice"]["provider"] == "11labs"
        assert config["backgroundDenoisingEnabled"] is True
        assert "stopSpeakingPlan" in config

    def test_inbound_config_has_vad_settings(self):
        """Test that inbound config includes VAD and turn detection settings."""
        from inbound.agent import build_assistant_config

        config = build_assistant_config(server_url="https://example.com/vapi/webhook")

        # VAD settings
        stop_plan = config["stopSpeakingPlan"]
        assert stop_plan["numWords"] == 2
        assert stop_plan["voiceSeconds"] == 0.2
        assert stop_plan["backoffSeconds"] == 1.0

        # Turn detection settings (nested inside startSpeakingPlan)
        start_plan = config["startSpeakingPlan"]
        assert start_plan["waitSeconds"] == 0.4
        endpointing = start_plan["transcriptionEndpointingPlan"]
        assert endpointing["onPunctuationSeconds"] == 0.1
        assert endpointing["onNoPunctuationSeconds"] == 1.5
        assert endpointing["onNumberSeconds"] == 0.5


class TestUnitWebhookHandling:
    """Unit tests for webhook event handling."""

    @pytest.mark.asyncio
    async def test_handle_tool_calls(self):
        """Test handling tool calls from Vapi."""
        from inbound.agent import handle_tool_calls

        message = {
            "toolCallList": [
                {
                    "id": "tc-1",
                    "function": {
                        "name": "send_sms",
                        "arguments": {
                            "phone_number": "+15551234567",
                            "message": "Your order has shipped!",
                        },
                    },
                }
            ]
        }

        results = await handle_tool_calls(message)
        assert len(results) == 1
        assert results[0]["toolCallId"] == "tc-1"

        result_data = json.loads(results[0]["result"])
        assert result_data["status"] == "sent"

    @pytest.mark.asyncio
    async def test_handle_end_call_tool(self):
        """Test handling end_call tool call."""
        from inbound.agent import handle_tool_calls

        message = {
            "toolCallList": [
                {
                    "id": "tc-2",
                    "function": {
                        "name": "end_call",
                        "arguments": {"reason": "Customer satisfied"},
                    },
                }
            ]
        }

        results = await handle_tool_calls(message)
        assert len(results) == 1
        result_data = json.loads(results[0]["result"])
        assert result_data["status"] == "call_ending"

    @pytest.mark.asyncio
    async def test_handle_unknown_tool(self):
        """Test handling unknown tool call."""
        from inbound.agent import handle_tool_calls

        message = {
            "toolCallList": [
                {
                    "id": "tc-3",
                    "function": {
                        "name": "nonexistent_function",
                        "arguments": {},
                    },
                }
            ]
        }

        results = await handle_tool_calls(message)
        result_data = json.loads(results[0]["result"])
        assert "error" in result_data


# =============================================================================
# LOCAL INTEGRATION TESTS
# =============================================================================


class TestLocalIntegration:
    """Integration tests using local HTTP connection."""

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
            assert data["orchestrator"] == "vapi"
            assert "gpt-4.1" in data["model"]

    @pytest.mark.asyncio
    async def test_local_webhook_assistant_request(self, server_process):
        """Test the Vapi assistant-request webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/vapi/webhook",
                json={
                    "message": {
                        "type": "assistant-request",
                        "call": {
                            "id": "test-call-123",
                            "customer": {"number": "+15551234567"},
                        },
                    }
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "assistant" in data

            assistant = data["assistant"]
            assert assistant["transcriber"]["provider"] == "deepgram"
            assert assistant["model"]["provider"] == "openai"
            assert assistant["model"]["model"] == "gpt-4.1"
            assert assistant["voice"]["provider"] == "11labs"
            assert assistant["backgroundDenoisingEnabled"] is True
            assert "stopSpeakingPlan" in assistant
            assert "startSpeakingPlan" in assistant
            assert "transcriptionEndpointingPlan" in assistant["startSpeakingPlan"]

    @pytest.mark.asyncio
    async def test_local_webhook_tool_calls(self, server_process):
        """Test the Vapi tool-calls webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/vapi/webhook",
                json={
                    "message": {
                        "type": "tool-calls",
                        "toolCallList": [
                            {
                                "id": "tc-test",
                                "function": {
                                    "name": "send_sms",
                                    "arguments": {
                                        "phone_number": "+15551234567",
                                        "message": "Test message",
                                    },
                                },
                            }
                        ],
                    }
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "results" in data
            assert len(data["results"]) == 1
            assert data["results"][0]["toolCallId"] == "tc-test"

    @pytest.mark.asyncio
    async def test_local_webhook_status_update(self, server_process):
        """Test the Vapi status-update webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/vapi/webhook",
                json={
                    "message": {
                        "type": "status-update",
                        "status": "in-progress",
                        "call": {"id": "test-call-123"},
                    }
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True

    @pytest.mark.asyncio
    async def test_local_webhook_end_of_call(self, server_process):
        """Test the Vapi end-of-call-report webhook."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/vapi/webhook",
                json={
                    "message": {
                        "type": "end-of-call-report",
                        "call": {"id": "test-call-123"},
                        "durationSeconds": 45,
                        "endedReason": "customer-ended-call",
                        "summary": "Customer asked about product pricing.",
                    }
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["ok"] is True


# =============================================================================
# VAPI INTEGRATION TESTS
# =============================================================================


class TestVapiIntegration:
    """Integration tests for Vapi API."""

    @pytest.fixture
    def vapi_configured(self):
        """Check if Vapi API is configured."""
        if not VAPI_PRIVATE_KEY:
            pytest.skip("VAPI_PRIVATE_KEY not configured")

    @pytest.mark.asyncio
    async def test_vapi_list_assistants(self, vapi_configured):
        """Test Vapi API connection by listing assistants."""
        from vapi import AsyncVapi

        client = AsyncVapi(token=VAPI_PRIVATE_KEY)
        assistants = await client.assistants.list()
        assert assistants is not None


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
        import plivo

        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
        account = client.account.get()
        assert account is not None

    def test_plivo_phone_number_exists(self, plivo_configured):
        """Test that the configured phone number exists."""
        import phonenumbers
        import plivo

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
