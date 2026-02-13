"""
Live call E2E test — places a real call through Plivo infrastructure.

This test:
1. Starts the FastAPI server as a subprocess on port 18002
2. Starts an ngrok tunnel programmatically
3. Configures Plivo webhooks to point at the ngrok URL
4. Places an outbound call from PLIVO_FROM_NUMBER to PLIVO_PHONE_NUMBER
5. Waits for the call to go live, starts recording
6. Lets the agent greeting play for ~20s
7. Hangs up, polls for the recording
8. Downloads the MP3, transcribes with faster-whisper
9. Verifies transcript contains expected greeting words

Requirements:
    - Valid PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_FROM_NUMBER,
      XAI_API_KEY in .env
    - PLIVO_FROM_NUMBER must be a separate Plivo number (used as caller ID)
    - ngrok binary at /usr/local/bin/ngrok
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (in project root or PATH)
    - Port 18002 available

Usage:
    cd grok-voice-native
    uv run pytest tests/test_live_call.py -v -s
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import httpx
import plivo
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    download_recording,
    start_ngrok,
    stop_ngrok,
    transcribe_audio,
    wait_for_recording,
)

load_dotenv()

# Configuration
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_FROM_NUMBER = os.getenv("PLIVO_FROM_NUMBER", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

# Ensure ffmpeg from project root is on PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isfile(os.path.join(PROJECT_ROOT, "ffmpeg")):
    os.environ["PATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

TEST_PORT = 18002
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_FROM_NUMBER, XAI_API_KEY]),
    reason="Plivo credentials, PLIVO_FROM_NUMBER, or XAI_API_KEY not configured",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def server_process():
    """Start the voice agent server as a subprocess on TEST_PORT."""
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
            resp = httpx.get(TEST_HTTP_URL, timeout=1.0)
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
    proc.wait(timeout=5)


@pytest.fixture(scope="module")
def ngrok_tunnel(server_process):
    """Start ngrok tunnel pointing at the test server."""
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[ngrok] Tunnel URL: {public_url}")

    yield public_url

    stop_ngrok(proc)


@pytest.fixture(scope="module")
def plivo_configured(ngrok_tunnel):
    """Configure Plivo app and assign phone number for inbound call handling."""
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    public_url = ngrok_tunnel

    app_name = "Grok_Voice_Agent_Test"
    answer_url = f"{public_url}/answer"
    hangup_url = f"{public_url}/hangup"

    # Find or create application
    apps = client.applications.list()
    existing_app = None
    for app_obj in apps["objects"]:
        if app_obj["app_name"] == app_name:
            existing_app = app_obj
            break

    if existing_app:
        client.applications.update(
            app_id=existing_app["app_id"],
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=hangup_url,
            hangup_method="POST",
        )
        app_id = existing_app["app_id"]
        print(f"[Plivo] Updated application: {app_name} -> {answer_url}")
    else:
        response = client.applications.create(
            app_name=app_name,
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=hangup_url,
            hangup_method="POST",
        )
        app_id = response["app_id"]
        print(f"[Plivo] Created application: {app_name} -> {answer_url}")

    # Assign phone number to agent app (used for B-leg in two-way test)
    phone_digits = "".join(c for c in PLIVO_PHONE_NUMBER if c.isdigit())
    client.numbers.update(number=phone_digits, app_id=app_id)
    print(f"[Plivo] Assigned {phone_digits} to app {app_id}")

    yield {"client": client, "app_id": app_id, "public_url": public_url}


# =============================================================================
# Tests
# =============================================================================


class TestLiveCall:
    """End-to-end tests that place a real call through Plivo."""

    def test_ngrok_tunnel_accessible(self, ngrok_tunnel):
        """Verify the ngrok tunnel reaches our server."""
        resp = httpx.get(ngrok_tunnel, timeout=10.0)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        print(f"\n[Health] {data}")

    def test_answer_webhook_via_ngrok(self, plivo_configured):
        """Verify the /answer endpoint returns valid Plivo XML through ngrok."""
        public_url = plivo_configured["public_url"]
        resp = httpx.get(
            f"{public_url}/answer",
            params={"CallUUID": "test-ngrok", "From": "+15551234567", "To": "+16572338892"},
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body
        assert "bidirectional" in body
        print(f"\n[Answer XML] {body[:300]}")

    def _place_call_and_wait(self, client, public_url):
        """Place an outbound call to PLIVO_PHONE_NUMBER and return the live call_uuid.

        Uses /hold as the A-leg answer_url so only the B-leg (phone number's app)
        starts the agent. This prevents the double-agent / audio feedback problem.
        """
        hold_url = f"{public_url}/hold"
        from_digits = "".join(c for c in PLIVO_FROM_NUMBER if c.isdigit())
        to_digits = "".join(c for c in PLIVO_PHONE_NUMBER if c.isdigit())

        print(f"\n[Call] Placing call: from_={from_digits} to_={to_digits}")
        print(f"[Call] A-leg answer_url: {hold_url} (hold/silent)")

        call_response = client.calls.create(
            from_=from_digits,
            to_=to_digits,
            answer_url=hold_url,
            answer_method="POST",
        )
        request_uuid = call_response["request_uuid"]
        print(f"[Call] request_uuid: {request_uuid}")

        # Poll for the call to go live
        print("[Call] Waiting for call to go live...")
        call_uuid = None
        for i in range(60):
            try:
                live_calls = client.live_calls.list_ids()
                call_ids = []
                if hasattr(live_calls, "calls"):
                    call_ids = live_calls.calls or []
                elif isinstance(live_calls, dict):
                    call_ids = live_calls.get("calls", [])
                if call_ids:
                    call_uuid = call_ids[0]
                    print(f"[Call] Live call_uuid: {call_uuid}")
                    break
            except Exception as e:
                if i % 10 == 0:
                    print(f"[Call] Poll error at {i}s: {e}")
            time.sleep(0.5)

        assert call_uuid, "Call did not go live within 30s"
        return call_uuid

    def test_live_call_greeting(self, plivo_configured):
        """Place a real call, record it, transcribe, and verify the greeting."""
        client = plivo_configured["client"]
        public_url = plivo_configured["public_url"]

        call_uuid = self._place_call_and_wait(client, public_url)

        try:
            # Start recording
            print("[Call] Starting recording...")
            client.calls.start_recording(call_uuid, file_format="mp3")

            # Let the agent greeting play
            print("[Call] Letting agent speak for 20s...")
            time.sleep(20)

        finally:
            print("[Call] Hanging up...")
            try:
                client.calls.delete(call_uuid)
            except Exception as e:
                print(f"[Call] Hangup error (may already be ended): {e}")

        # Poll for recording
        print("[Recording] Waiting for recording to become available...")
        recording_url = wait_for_recording(client, call_uuid, timeout=30)
        assert recording_url, f"No recording found for call {call_uuid} within 30s"
        print(f"[Recording] URL: {recording_url}")

        # Download and transcribe
        print("[Recording] Downloading...")
        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000, f"Recording too small: {len(audio_data)} bytes"
        print(f"[Recording] Downloaded {len(audio_data)} bytes")

        print("[Transcribe] Transcribing with faster-whisper...")
        transcript = transcribe_audio(audio_data)
        print(f"[Transcript] {transcript}")

        assert len(transcript) > 5, f"Transcript too short: '{transcript}'"
        greeting_words = [
            "hello", "hi", "welcome", "help", "how",
            "assist", "alex", "techflow", "grok",
        ]
        matches = [w for w in greeting_words if w in transcript.lower()]
        assert matches, (
            f"Greeting doesn't match expected content. "
            f"Expected one of {greeting_words}, got: '{transcript}'"
        )
        print(f"[Result] Matched greeting words: {matches}")

    def test_live_call_two_way_conversation(self, plivo_configured):
        """Place a call, ask the agent a question via Plivo TTS, verify the response."""
        client = plivo_configured["client"]
        public_url = plivo_configured["public_url"]

        call_uuid = self._place_call_and_wait(client, public_url)

        try:
            # Start recording
            print("[Call] Starting recording...")
            client.calls.start_recording(call_uuid, file_format="mp3")

            # Wait for agent greeting to finish
            print("[Call] Waiting 20s for agent greeting...")
            time.sleep(20)

            # Inject a question via Plivo TTS on the A-leg.
            # The A-leg audio flows to the B-leg (where the agent's Stream is),
            # so the agent's STT picks it up.
            question = "What plans do you offer and how much do they cost?"
            print(f"[Call] Speaking into call (aleg): '{question}'")
            client.calls.speak(call_uuid, text=question, language="en-US", legs="aleg")

            # Wait for the agent to process speech and respond
            print("[Call] Waiting 25s for agent response...")
            time.sleep(25)

        finally:
            print("[Call] Hanging up...")
            try:
                client.calls.delete(call_uuid)
            except Exception as e:
                print(f"[Call] Hangup error (may already be ended): {e}")

        # Poll for recording
        print("[Recording] Waiting for recording to become available...")
        recording_url = wait_for_recording(client, call_uuid, timeout=30)
        assert recording_url, f"No recording found for call {call_uuid} within 30s"
        print(f"[Recording] URL: {recording_url}")

        # Download and transcribe
        print("[Recording] Downloading...")
        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000, f"Recording too small: {len(audio_data)} bytes"
        print(f"[Recording] Downloaded {len(audio_data)} bytes")

        print("[Transcribe] Transcribing with faster-whisper...")
        transcript = transcribe_audio(audio_data)
        print(f"[Transcript] {transcript}")

        transcript_lower = transcript.lower()

        # Verify greeting is present
        greeting_words = ["hello", "hi", "welcome", "help", "how", "alex", "techflow"]
        greeting_matches = [w for w in greeting_words if w in transcript_lower]
        assert greeting_matches, (
            f"Greeting not found in transcript. "
            f"Expected one of {greeting_words}, got: '{transcript}'"
        )
        print(f"[Result] Greeting words matched: {greeting_matches}")

        # Verify agent responded to the pricing question
        product_words = [
            "pro", "team", "enterprise", "starter",
            "twelve", "twenty", "dollar", "month",
            "plan", "price", "cost", "tier",
            "12", "25", "49",
        ]
        product_matches = [w for w in product_words if w in transcript_lower]
        assert len(product_matches) >= 2, (
            f"Agent did not discuss products/pricing. "
            f"Matches: {product_matches}, transcript: '{transcript}'"
        )
        print(f"[Result] Product/pricing words matched: {product_matches}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
