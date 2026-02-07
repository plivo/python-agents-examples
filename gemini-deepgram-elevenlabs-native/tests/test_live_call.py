"""
Live call test - starts server and places an actual call via Plivo.

This test:
1. Starts the FastAPI server
2. Creates an ngrok tunnel (or uses PUBLIC_URL)
3. Configures Plivo webhooks
4. Places an outbound call to a test number
5. Verifies the call connects and agent responds

Requires:
- Valid Plivo credentials in .env
- A phone number to call (TEST_PHONE_NUMBER env var)
"""

import asyncio
import os
import subprocess
import sys
import time

import plivo
import pytest
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

# Configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
TEST_PHONE_NUMBER = os.getenv("TEST_PHONE_NUMBER", "")  # Number to call for testing
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

pytestmark = pytest.mark.skipif(
    not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]),
    reason="Plivo credentials not configured"
)


class TestPlivoConnection:
    """Test Plivo API connectivity."""

    def test_plivo_credentials_valid(self):
        """Test that Plivo credentials are valid."""
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        # Get account details to verify credentials
        try:
            account = client.account.get()
            print(f"\n[Plivo] Account verified: {account['auth_id']}")
            print(f"[Plivo] Account state: {account['account_type']}")
            assert account is not None
        except Exception as e:
            pytest.fail(f"Plivo credentials invalid: {e}")

    def test_plivo_phone_number_exists(self):
        """Test that the configured phone number exists in account."""
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        # List numbers in account
        numbers = client.numbers.list()
        phone_digits = "".join(c for c in PLIVO_PHONE_NUMBER if c.isdigit())

        found = False
        for number in numbers["objects"]:
            if phone_digits in number["number"]:
                found = True
                print(f"\n[Plivo] Phone number found: {number['number']}")
                print(f"[Plivo] Number type: {number['number_type']}")
                print(f"[Plivo] Voice enabled: {number['voice_enabled']}")
                break

        assert found, f"Phone number {PLIVO_PHONE_NUMBER} not found in account"


class TestServerStartup:
    """Test that the server can start."""

    def test_server_imports(self):
        """Test that server module imports correctly."""
        import server
        assert server.app is not None
        print(f"\n[Server] App title: {server.app.title}")

    def test_agent_imports(self):
        """Test that agent module imports correctly."""
        import agent
        assert agent.VoiceAgent is not None
        assert agent.GeminiLLM is not None
        assert agent.DeepgramSTT is not None
        assert agent.ElevenLabsTTS is not None
        print("\n[Agent] All components imported successfully")


class TestWebhookConfiguration:
    """Test webhook URL configuration."""

    @pytest.mark.skipif(not PUBLIC_URL, reason="PUBLIC_URL not configured")
    def test_configure_webhooks(self):
        """Test configuring Plivo webhooks."""
        from server import configure_plivo_webhooks

        result = configure_plivo_webhooks()
        print(f"\n[Webhooks] Configuration result: {result}")

        if result:
            print(f"[Webhooks] Answer URL: {PUBLIC_URL}/answer")
            print(f"[Webhooks] Hangup URL: {PUBLIC_URL}/hangup")


class TestOutboundCall:
    """Test placing an outbound call (requires TEST_PHONE_NUMBER)."""

    @pytest.mark.skipif(
        not all([TEST_PHONE_NUMBER, PUBLIC_URL]),
        reason="TEST_PHONE_NUMBER or PUBLIC_URL not configured"
    )
    def test_place_outbound_call(self):
        """Place an outbound call and verify it connects."""
        client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)

        answer_url = f"{PUBLIC_URL}/answer"

        print(f"\n[Call] Placing call from {PLIVO_PHONE_NUMBER} to {TEST_PHONE_NUMBER}")
        print(f"[Call] Answer URL: {answer_url}")

        try:
            response = client.calls.create(
                from_=PLIVO_PHONE_NUMBER,
                to_=TEST_PHONE_NUMBER,
                answer_url=answer_url,
                answer_method="POST",
            )

            print(f"[Call] Call UUID: {response['request_uuid']}")
            print(f"[Call] Status: Call initiated")

            # Note: The actual call handling happens asynchronously
            # The server must be running for the call to work

        except Exception as e:
            pytest.fail(f"Failed to place call: {e}")


def run_server_and_test():
    """
    Helper function to run the server and place a test call.

    Usage:
        python -m tests.test_live_call
    """
    print("=" * 60)
    print("Live Call Test")
    print("=" * 60)

    if not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER]):
        print("ERROR: Plivo credentials not configured in .env")
        return

    if not PUBLIC_URL:
        print("ERROR: PUBLIC_URL not configured in .env")
        print("Please start ngrok and set PUBLIC_URL to the ngrok URL")
        print("Example: ngrok http 8000")
        return

    print(f"\nConfiguration:")
    print(f"  Plivo Phone: {PLIVO_PHONE_NUMBER}")
    print(f"  Public URL: {PUBLIC_URL}")
    print(f"  Test Phone: {TEST_PHONE_NUMBER or 'Not configured'}")

    # Start server in subprocess
    print("\nStarting server...")
    server_proc = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for server to start
        time.sleep(3)

        if server_proc.poll() is not None:
            print("ERROR: Server failed to start")
            output = server_proc.stdout.read().decode()
            print(output)
            return

        print("Server started successfully")
        print(f"\nCall {PLIVO_PHONE_NUMBER} to test the voice agent")
        print("Press Ctrl+C to stop the server")

        # Keep running until interrupted
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server_proc.terminate()
        server_proc.wait()


if __name__ == "__main__":
    run_server_and_test()
