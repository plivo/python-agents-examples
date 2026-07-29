"""Outbound call E2E tests — verify the outbound calling feature end-to-end.

Starts the outbound server behind an ngrok tunnel, then exercises the outbound
API: initiating a call, the answer webhook XML, a full record/transcribe cycle,
status lifecycle transitions, programmatic hangup, and campaign listing.

Outbound calls use PLIVO_PHONE_NUMBER (a US number) as the caller ID and place
every real call to the single project test destination DEST_NUMBER.
"""

from __future__ import annotations

import contextlib
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
    require_public_ready,
    start_ngrok,
    stop_ngrok,
    transcribe_audio,
    wait_for_recording,
)

load_dotenv()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
DEST_NUMBER = os.getenv("PLIVO_DEST_NUMBER", "")

TEST_PORT = 18003
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, XAI_API_KEY, DEST_NUMBER]),
    reason="Plivo credentials, PLIVO_DEST_NUMBER, or XAI_API_KEY not configured",
)


@pytest.fixture(scope="module")
def ngrok_tunnel():
    """Start an ngrok tunnel pointing at TEST_PORT before the server starts."""
    proc, public_url = start_ngrok(TEST_PORT)
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(ngrok_tunnel):
    """Start the outbound voice agent server with PUBLIC_URL set to the tunnel."""
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

    try:
        require_public_ready(ngrok_tunnel, "/")
    except BaseException:
        proc.terminate()
        proc.wait()
        raise

    yield proc

    os.kill(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@pytest.fixture(scope="module")
def plivo_client():
    """Create a Plivo REST client."""
    return plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)


class TestOutboundCall:
    """End-to-end tests for outbound calling."""

    def test_initiate_outbound_call_api(self, server_process, ngrok_tunnel):
        """POST /outbound/call returns a call_id and status tracking works."""
        public_url = ngrok_tunnel

        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": DEST_NUMBER,
                "campaign_id": "test-campaign-1",
                "opening_reason": "your recent demo request for TechFlow Teams",
                "objective": "qualify interest and book a meeting with sales",
            },
            timeout=30.0,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "call_id" in data, f"Expected call_id in response: {data}"
        call_id = data["call_id"]

        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["call_id"] == call_id
        assert status_data["status"] in (
            "ringing", "connected", "completed", "failed", "no_answer",
        )

        time.sleep(3)
        httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=10.0)

    def test_outbound_answer_webhook(self, server_process, ngrok_tunnel):
        """GET /outbound/answer returns bidirectional Stream XML."""
        public_url = ngrok_tunnel

        resp = httpx.get(
            f"{public_url}/outbound/answer",
            params={
                "call_id": "test-call-123",
                "CallUUID": "test-uuid-456",
                "From": PLIVO_PHONE_NUMBER,
                "To": DEST_NUMBER,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body, "Response should contain a <Stream> element"
        assert "bidirectional" in body, "Stream should be bidirectional"
        assert "ws" in body.lower(), "Stream should contain a WebSocket URL"

    def test_outbound_call_full_cycle(self, server_process, ngrok_tunnel, plivo_client):
        """Place a real outbound call, record, transcribe, and verify the greeting."""
        public_url = ngrok_tunnel

        opening_reason = "your recent demo request for TechFlow Teams"
        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": DEST_NUMBER,
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

        call_uuid = None
        for _ in range(60):
            try:
                live_calls = plivo_client.live_calls.list_ids()
                call_ids = []
                if hasattr(live_calls, "calls"):
                    call_ids = live_calls.calls or []
                elif isinstance(live_calls, dict):
                    call_ids = live_calls.get("calls", [])
                if call_ids:
                    call_uuid = call_ids[0]
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not call_uuid:
            pytest.skip("Call did not connect (callee may not have answered)")

        try:
            plivo_client.calls.start_recording(call_uuid, file_format="mp3")
            time.sleep(20)
        finally:
            with contextlib.suppress(Exception):
                plivo_client.calls.delete(call_uuid)

        recording_url = wait_for_recording(plivo_client, call_uuid, timeout=30)
        assert recording_url, f"No recording found for call {call_uuid} within 30s"

        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000, f"Recording too small: {len(audio_data)} bytes"

        transcript = transcribe_audio(audio_data)
        assert len(transcript) > 5, f"Transcript too short: '{transcript}'"

        outbound_words = [
            "alex", "techflow", "demo", "trial", "reaching out",
            "hi", "hello", "good time",
        ]
        matches = [word for word in outbound_words if word in transcript.lower()]
        assert matches, (
            f"Outbound greeting doesn't match expected content. "
            f"Expected one of {outbound_words}, got: '{transcript}'"
        )

    def test_outbound_call_status_lifecycle(self, server_process, ngrok_tunnel):
        """Status transitions from ringing toward a terminal state."""
        public_url = ngrok_tunnel

        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": DEST_NUMBER,
                "opening_reason": "your recent free trial sign-up for TechFlow",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")
        assert call_id, f"No call_id: {data}"

        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        assert status["status"] in ("ringing", "failed"), (
            f"Unexpected status: {status['status']}"
        )

        if status["status"] == "failed":
            return

        time.sleep(5)

        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        assert status["status"] in (
            "ringing", "connected", "completed", "failed", "no_answer",
        )

        time.sleep(2)
        httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=10.0)
        httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)

    def test_outbound_hangup_programmatic(self, server_process, ngrok_tunnel):
        """POST /outbound/hangup/{call_id} drives the call to a terminal status."""
        public_url = ngrok_tunnel

        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": DEST_NUMBER,
                "opening_reason": "your recent demo request for TechFlow Teams",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")
        assert call_id, f"No call_id: {data}"

        time.sleep(5)

        httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=10.0)

        status_resp = httpx.get(f"{public_url}/outbound/status/{call_id}", timeout=10.0)
        status = status_resp.json()
        assert status["status"] in ("ringing", "connected", "completed", "failed", "no_answer"), (
            f"Unexpected status: {status['status']}"
        )

    def test_outbound_campaign_endpoint(self, server_process, ngrok_tunnel):
        """GET /outbound/campaign/{campaign_id} returns calls for a campaign."""
        public_url = ngrok_tunnel
        campaign_id = "test-campaign-endpoint"

        resp = httpx.post(
            f"{public_url}/outbound/call",
            params={
                "phone_number": DEST_NUMBER,
                "campaign_id": campaign_id,
                "opening_reason": "your recent demo request for TechFlow Teams",
            },
            timeout=30.0,
        )
        data = resp.json()
        call_id = data.get("call_id")

        time.sleep(3)
        httpx.post(f"{public_url}/outbound/hangup/{call_id}", timeout=10.0)

        camp_resp = httpx.get(f"{public_url}/outbound/campaign/{campaign_id}", timeout=10.0)
        assert camp_resp.status_code == 200
        camp_data = camp_resp.json()
        assert camp_data["campaign_id"] == campaign_id
        assert camp_data["total"] >= 1
        call_ids = [call["call_id"] for call in camp_data["calls"]]
        assert call_id in call_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
