"""
Outbound call E2E tests — verifies the outbound calling feature end-to-end.

Tests:
1. POST /outbound/call returns call_id and status tracking works
2. /outbound/answer returns valid Stream XML
3. Full outbound call cycle: place real call, record, transcribe, verify greeting
4. Status lifecycle transitions (initiating -> ringing -> connected -> completed)
5. Programmatic hangup via POST /outbound/hangup/{call_id}

Requirements:
    - Valid PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_FROM_NUMBER,
      XAI_API_KEY in .env
    - PLIVO_FROM_NUMBER must be a Plivo number (used as caller ID for outbound)
    - PLIVO_PHONE_NUMBER is the number to call (can be your personal phone)
    - ngrok binary at /usr/local/bin/ngrok
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (in project root or PATH)
    - Port 18003 available

Usage:
    cd grok-voice-native
    uv run pytest tests/test_outbound_call.py -v -s
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

TEST_PORT = 18003
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_FROM_NUMBER, XAI_API_KEY]),
    reason="Plivo credentials, PLIVO_FROM_NUMBER, or XAI_API_KEY not configured",
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def ngrok_tunnel():
    """Start ngrok tunnel pointing at TEST_PORT.

    Started before the server so we can pass PUBLIC_URL to the server process.
    ngrok only needs the port number, not a running server.
    """
    proc, public_url = start_ngrok(TEST_PORT)
    print(f"\n[ngrok] Tunnel URL: {public_url}")

    yield public_url

    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(ngrok_tunnel):
    """Start the voice agent server as a subprocess on TEST_PORT.

    Depends on ngrok_tunnel so PUBLIC_URL is available for Plivo answer_url.
    """
    env = os.environ.copy()
    env["SERVER_PORT"] = str(TEST_PORT)
    env["PUBLIC_URL"] = ngrok_tunnel

    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    proc = subprocess.Popen(
        [sys.executable, "-m", "outbound.server"],
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
def plivo_client():
    """Create a Plivo REST client."""
    return plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)


# =============================================================================
# Tests
# =============================================================================


class TestOutboundCall:
    """End-to-end tests for outbound calling."""

    def test_initiate_outbound_call_api(self, server_process, ngrok_tunnel):
        """POST /outbound/call returns call_id and status tracking works."""
        public_url = ngrok_tunnel

        # Initiate an outbound call via the API
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_PHONE_NUMBER,
                "campaign_id": "test-campaign-1",
                "opening_reason": "your recent demo request for TechFlow Teams",
                "objective": "qualify interest and book a meeting with sales",
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        print(f"\n[Outbound] Initiate response: {data}")

        # Should have a call_id
        assert "call_id" in data, f"Expected call_id in response: {data}"
        call_id = data["call_id"]

        # Check status endpoint
        status_resp = httpx.get(
            f"{public_url}/outbound/status/{call_id}",
            timeout=10.0,
        )
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        print(f"[Outbound] Status: {status_data}")
        assert status_data["call_id"] == call_id
        assert status_data["status"] in ("ringing", "connected", "completed", "failed", "no_answer")

        # Wait a moment then try to hang up the call
        time.sleep(3)
        hangup_resp = httpx.post(
            f"{public_url}/outbound/hangup/{call_id}",
            timeout=10.0,
        )
        print(f"[Outbound] Hangup response: {hangup_resp.json()}")

    def test_outbound_answer_webhook(self, server_process, ngrok_tunnel):
        """Verify /outbound/answer returns valid Plivo Stream XML."""
        public_url = ngrok_tunnel

        resp = httpx.get(
            f"{public_url}/outbound/answer",
            params={
                "call_id": "test-call-123",
                "CallUUID": "test-uuid-456",
                "From": PLIVO_FROM_NUMBER,
                "To": PLIVO_PHONE_NUMBER,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text
        print(f"\n[Outbound Answer XML] {body[:500]}")

        assert "<Stream" in body, "Response should contain <Stream> element"
        assert "bidirectional" in body, "Stream should be bidirectional"
        assert "ws" in body.lower(), "Stream should contain WebSocket URL"

    def test_outbound_call_full_cycle(self, server_process, ngrok_tunnel, plivo_client):
        """Place a real outbound call, record, transcribe, verify outbound greeting."""
        public_url = ngrok_tunnel

        # Initiate outbound call via the API
        opening_reason = "your recent demo request for TechFlow Teams"
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_PHONE_NUMBER,
                "campaign_id": "test-full-cycle",
                "opening_reason": opening_reason,
                "objective": "qualify interest and book a meeting with sales",
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        call_id = data.get("call_id")
        assert call_id, f"No call_id in response: {data}"
        print(f"\n[Outbound] Call initiated: {data}")

        # Wait for call to go live
        print("[Outbound] Waiting for call to connect...")
        call_uuid = None
        for i in range(60):
            try:
                live_calls = plivo_client.live_calls.list_ids()
                call_ids = []
                if hasattr(live_calls, "calls"):
                    call_ids = live_calls.calls or []
                elif isinstance(live_calls, dict):
                    call_ids = live_calls.get("calls", [])
                if call_ids:
                    call_uuid = call_ids[0]
                    print(f"[Outbound] Live call_uuid: {call_uuid}")
                    break
            except Exception as e:
                if i % 10 == 0:
                    print(f"[Outbound] Poll error at {i}s: {e}")
            time.sleep(0.5)

        if not call_uuid:
            print("[Outbound] Call did not go live — skipping recording verification")
            pytest.skip("Call did not connect (callee may not have answered)")

        try:
            # Start recording
            print("[Outbound] Starting recording...")
            plivo_client.calls.start_recording(call_uuid, file_format="mp3")

            # Let the outbound agent greeting play
            print("[Outbound] Letting agent speak for 20s...")
            time.sleep(20)

        finally:
            print("[Outbound] Hanging up...")
            try:
                plivo_client.calls.delete(call_uuid)
            except Exception as e:
                print(f"[Outbound] Hangup error (may already be ended): {e}")

        # Poll for recording
        print("[Recording] Waiting for recording to become available...")
        recording_url = wait_for_recording(plivo_client, call_uuid, timeout=30)
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

        # Verify the outbound greeting mentions the reason/identity
        outbound_words = [
            "alex", "techflow", "demo", "trial", "reaching out",
            "hi", "hello", "good time",
        ]
        matches = [w for w in outbound_words if w in transcript.lower()]
        assert matches, (
            f"Outbound greeting doesn't match expected content. "
            f"Expected one of {outbound_words}, got: '{transcript}'"
        )
        print(f"[Result] Matched outbound words: {matches}")

    def test_outbound_call_status_lifecycle(self, server_process, ngrok_tunnel):
        """Verify status transitions: initiating -> ringing -> connected -> completed."""
        public_url = ngrok_tunnel

        # Initiate a call
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_PHONE_NUMBER,
                "opening_reason": "your recent free trial sign-up for TechFlow",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")
        assert call_id, f"No call_id: {data}"

        # Check initial status should be ringing (Plivo API was called)
        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        print(f"\n[Lifecycle] Initial status: {status['status']}")
        assert status["status"] in ("ringing", "failed"), f"Unexpected status: {status['status']}"

        if status["status"] == "failed":
            print("[Lifecycle] Call failed to initiate — cannot test full lifecycle")
            return

        # Wait a bit for the call to potentially connect or fail
        time.sleep(5)

        # Check status again
        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        print(f"[Lifecycle] After 5s: {status['status']}")
        assert status["status"] in ("ringing", "connected", "completed", "failed", "no_answer")

        # Try programmatic hangup
        time.sleep(2)
        hangup_resp = httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=10.0)
        hangup_data = hangup_resp.json()
        print(f"[Lifecycle] Hangup response: {hangup_data}")

        # Final status check
        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        print(f"[Lifecycle] Final status: {status['status']}")

    def test_outbound_hangup_programmatic(self, server_process, ngrok_tunnel, plivo_client):
        """POST /outbound/hangup/{call_id} ends an active call."""
        public_url = ngrok_tunnel

        # Initiate a call
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_PHONE_NUMBER,
                "opening_reason": "your recent demo request for TechFlow Teams",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")
        assert call_id, f"No call_id: {data}"
        print(f"\n[Hangup] Call initiated: call_id={call_id}")

        # Wait for the call to potentially connect
        time.sleep(5)

        # Try to hang up
        hangup_resp = httpx.post(
            f"{public_url}/outbound/hangup/{call_id}",
            timeout=10.0,
        )
        hangup_data = hangup_resp.json()
        print(f"[Hangup] Response: {hangup_data}")

        # Verify the call status is now completed
        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        print(f"[Hangup] Final status: {status['status']}")
        assert status["status"] in ("completed", "failed", "no_answer"), (
            f"Expected terminal status, got: {status['status']}"
        )

    def test_outbound_campaign_endpoint(self, server_process, ngrok_tunnel):
        """GET /outbound/campaign/{campaign_id} returns calls for a campaign."""
        public_url = ngrok_tunnel
        campaign_id = "test-campaign-endpoint"

        # Initiate a call with this campaign
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_PHONE_NUMBER,
                "campaign_id": campaign_id,
                "opening_reason": "your recent demo request for TechFlow Teams",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")
        print(f"\n[Campaign] Call initiated: {call_id}")

        # Wait briefly then clean up
        time.sleep(3)
        httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=10.0)

        # Check campaign endpoint
        camp_resp = httpx.get(
            f"{public_url}/outbound/campaign/{campaign_id}",
            timeout=10.0,
        )
        assert camp_resp.status_code == 200
        camp_data = camp_resp.json()
        print(f"[Campaign] Response: {camp_data}")

        assert camp_data["campaign_id"] == campaign_id
        assert camp_data["total"] >= 1
        call_ids = [c["call_id"] for c in camp_data["calls"]]
        assert call_id in call_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
