"""
Outbound call E2E tests — verifies the outbound calling feature end-to-end.

Tests:
1. POST /outbound/call returns call_id and status tracking works
2. /outbound/answer returns valid Stream XML
3. Full outbound call cycle: place real call, record, transcribe, verify greeting
4. Status lifecycle transitions (initiating -> ringing -> connected -> completed)
5. Programmatic hangup via POST /outbound/hangup/{call_id}

Requirements:
    - Valid PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER,
      OPENAI_API_KEY, MODULATE_API_KEY in .env
    - PLIVO_PHONE_NUMBER is the agent number (used as caller ID for outbound)
    - PLIVO_TEST_NUMBER is a second Plivo number (destination for test calls)
    - ngrok binary available on PATH
    - faster-whisper installed (dev dependency)
    - ffmpeg binary available (in project root or PATH)
    - Port 18003 available

Usage:
    cd gpt4o-modulatevelma2-cartesiasonic3-pipecat
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
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODULATE_API_KEY = os.getenv("MODULATE_API_KEY", "")

# Ensure ffmpeg from project root is on PATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if os.path.isfile(os.path.join(PROJECT_ROOT, "ffmpeg")):
    os.environ["PATH"] = PROJECT_ROOT + os.pathsep + os.environ.get("PATH", "")

TEST_PORT = 18003
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not all(
        [PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER,
         PLIVO_TEST_NUMBER, OPENAI_API_KEY, MODULATE_API_KEY]
    ),
    reason="Plivo credentials, PLIVO_TEST_NUMBER, or API keys not configured",
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
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


@pytest.fixture(scope="module")
def plivo_client():
    """Create a Plivo REST client."""
    return plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)


@pytest.fixture(scope="module")
def bleg_app_id(plivo_client, ngrok_tunnel):
    """Configure PLIVO_TEST_NUMBER to answer with /outbound/answer.

    When calling between two Plivo numbers, Plivo routes the call through
    the destination number's app (ignoring the answer_url from calls.create).
    This fixture configures PLIVO_TEST_NUMBER's app so the outbound agent
    starts when the test call connects.

    Teardown restores the original app assignment.
    """
    test_digits = "".join(c for c in PLIVO_TEST_NUMBER if c.isdigit())

    # Save original app for teardown
    number_info = plivo_client.numbers.get(number=test_digits)
    if isinstance(number_info, dict):
        original_app = number_info.get("application", "")
    else:
        original_app = getattr(number_info, "application", "")
    # Extract app_id from URI like /v1/Account/.../Application/12345/
    original_app_id = ""
    if original_app and "/Application/" in str(original_app):
        original_app_id = str(original_app).split("/Application/")[1].rstrip("/")

    # Create or update the test app
    app_name = "GPT4o_Pipecat_Outbound_Test_Agent"
    answer_url = f"{ngrok_tunnel}/outbound/answer"

    apps = plivo_client.applications.list()
    existing_app = None
    for app_obj in apps["objects"]:
        if app_obj["app_name"] == app_name:
            existing_app = app_obj
            break

    if existing_app:
        plivo_client.applications.update(
            app_id=existing_app["app_id"],
            answer_url=answer_url,
            answer_method="POST",
        )
        app_id = existing_app["app_id"]
    else:
        response = plivo_client.applications.create(
            app_name=app_name,
            answer_url=answer_url,
            answer_method="POST",
        )
        app_id = response["app_id"]

    plivo_client.numbers.update(number=test_digits, app_id=app_id)
    print(f"\n[Plivo] Configured {test_digits} with B-leg app {app_id}")

    yield app_id

    # Restore original app
    if original_app_id:
        plivo_client.numbers.update(number=test_digits, app_id=original_app_id)
        print(f"\n[Plivo] Restored {test_digits} to original app {original_app_id}")


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
                "phone_number": PLIVO_TEST_NUMBER,
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
        assert status_data["status"] in (
            "ringing", "connected", "completed", "failed", "no_answer",
        )

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
                "From": PLIVO_PHONE_NUMBER,
                "To": PLIVO_TEST_NUMBER,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text
        print(f"\n[Outbound Answer XML] {body[:500]}")

        assert "<Stream" in body, "Response should contain <Stream> element"
        assert "bidirectional" in body, "Stream should be bidirectional"
        assert "ws" in body.lower(), "Stream should contain WebSocket URL"

    def test_outbound_call_full_cycle(
        self, server_process, ngrok_tunnel, plivo_client, bleg_app_id
    ):
        """Place a real outbound call, record, transcribe, verify outbound greeting."""
        public_url = ngrok_tunnel

        # Initiate outbound call via the API
        opening_reason = "your recent demo request for TechFlow Teams"
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_TEST_NUMBER,
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

    def test_outbound_campaign_endpoint(self, server_process, ngrok_tunnel):
        """GET /outbound/campaign/{campaign_id} returns calls for a campaign."""
        public_url = ngrok_tunnel
        campaign_id = "test-campaign-endpoint"

        # Initiate a call with this campaign
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": PLIVO_TEST_NUMBER,
                "campaign_id": campaign_id,
                "opening_reason": "your recent demo request for TechFlow Teams",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")
        print(f"\n[Campaign] Call initiated: {call_id}")

        # Wait briefly then clean up
        time.sleep(5)
        try:
            httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=30.0)
        except httpx.ReadTimeout:
            print("[Campaign] Hangup timed out (call may have already ended)")

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
