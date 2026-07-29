"""Live inbound call E2E tests — place a real call through Plivo infrastructure.

Starts the inbound server behind an ngrok tunnel, assigns the agent number to a
Plivo application pointed at the tunnel, then places a call so the agent answers,
records it, and transcribes the audio to verify the greeting and a two-way turn.
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
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")

TEST_PORT = 18002
TEST_HTTP_URL = f"http://localhost:{TEST_PORT}"

pytestmark = pytest.mark.skipif(
    not all([PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER, XAI_API_KEY]),
    reason="Plivo credentials, PLIVO_TEST_NUMBER, or XAI_API_KEY not configured",
)


@pytest.fixture(scope="module")
def ngrok_tunnel():
    """Start an ngrok tunnel pointing at TEST_PORT."""
    proc, public_url = start_ngrok(TEST_PORT)
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(ngrok_tunnel):
    """Start the inbound voice agent server with PUBLIC_URL set to the tunnel."""
    env = os.environ.copy()
    env["SERVER_PORT"] = str(TEST_PORT)
    env["PUBLIC_URL"] = ngrok_tunnel

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
def plivo_configured(server_process, ngrok_tunnel):
    """Configure a Plivo application and assign the agent number to it."""
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    public_url = ngrok_tunnel

    app_name = "XAI_Realtime_Agent_Test"
    answer_url = f"{public_url}/answer"
    hangup_url = f"{public_url}/hangup"

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
    else:
        response = client.applications.create(
            app_name=app_name,
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=hangup_url,
            hangup_method="POST",
        )
        app_id = response["app_id"]

    phone_digits = "".join(c for c in PLIVO_PHONE_NUMBER if c.isdigit())
    client.numbers.update(number=phone_digits, app_id=app_id)

    yield {"client": client, "app_id": app_id, "public_url": public_url}


class TestLiveCall:
    """End-to-end tests that place a real call through Plivo."""

    def test_ngrok_tunnel_accessible(self, server_process, ngrok_tunnel):
        """The ngrok tunnel reaches the running server."""
        resp = httpx.get(ngrok_tunnel, timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_answer_webhook_via_ngrok(self, plivo_configured):
        """The /answer endpoint returns bidirectional Stream XML through ngrok."""
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

    def _place_call_and_wait(self, client, public_url):
        """Place a call from the US Plivo number to the receiver and return the live call UUID."""
        answer_url = f"{public_url}/answer"
        from_digits = "".join(c for c in PLIVO_PHONE_NUMBER if c.isdigit())
        to_digits = "".join(c for c in PLIVO_TEST_NUMBER if c.isdigit())

        client.calls.create(
            from_=from_digits,
            to_=to_digits,
            answer_url=answer_url,
            answer_method="POST",
        )

        call_uuid = None
        for _ in range(60):
            try:
                live_calls = client.live_calls.list_ids()
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

        assert call_uuid, "Call did not go live within 30s"
        return call_uuid

    def test_live_call_greeting(self, plivo_configured):
        """Place a real call, record it, transcribe, and verify the greeting."""
        client = plivo_configured["client"]
        public_url = plivo_configured["public_url"]

        call_uuid = self._place_call_and_wait(client, public_url)

        try:
            client.calls.start_recording(call_uuid, file_format="mp3")
            time.sleep(20)
        finally:
            with contextlib.suppress(Exception):
                client.calls.delete(call_uuid)

        recording_url = wait_for_recording(client, call_uuid, timeout=30)
        assert recording_url, f"No recording found for call {call_uuid} within 30s"

        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000, f"Recording too small: {len(audio_data)} bytes"

        transcript = transcribe_audio(audio_data)
        assert len(transcript) > 5, f"Transcript too short: '{transcript}'"

        greeting_words = ["hello", "hi", "welcome", "help", "how", "assist", "alex", "techflow"]
        matches = [word for word in greeting_words if word in transcript.lower()]
        assert matches, (
            f"Greeting doesn't match expected content. "
            f"Expected one of {greeting_words}, got: '{transcript}'"
        )

    def test_live_call_two_way_conversation(self, plivo_configured):
        """Place a call, ask a question via Plivo TTS, and verify the agent responds."""
        client = plivo_configured["client"]
        public_url = plivo_configured["public_url"]

        call_uuid = self._place_call_and_wait(client, public_url)

        try:
            client.calls.start_recording(call_uuid, file_format="mp3")
            time.sleep(20)

            question = "What plans do you offer and how much do they cost?"
            client.calls.speak(call_uuid, text=question, language="en-US", legs="bleg")

            time.sleep(25)
        finally:
            with contextlib.suppress(Exception):
                client.calls.delete(call_uuid)

        recording_url = wait_for_recording(client, call_uuid, timeout=30)
        assert recording_url, f"No recording found for call {call_uuid} within 30s"

        audio_data = download_recording(recording_url)
        assert len(audio_data) > 1000, f"Recording too small: {len(audio_data)} bytes"

        transcript = transcribe_audio(audio_data)
        transcript_lower = transcript.lower()

        greeting_words = ["hello", "hi", "welcome", "help", "how", "alex", "techflow"]
        greeting_matches = [word for word in greeting_words if word in transcript_lower]
        assert greeting_matches, (
            f"Greeting not found in transcript. "
            f"Expected one of {greeting_words}, got: '{transcript}'"
        )

        product_words = [
            "pro", "team", "enterprise", "starter",
            "twelve", "twenty", "dollar", "month",
            "plan", "price", "cost", "tier",
            "12", "25", "49",
        ]
        product_matches = [word for word in product_words if word in transcript_lower]
        assert len(product_matches) >= 2, (
            f"Agent did not discuss products/pricing. "
            f"Matches: {product_matches}, transcript: '{transcript}'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
