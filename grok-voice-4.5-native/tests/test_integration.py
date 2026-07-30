"""Unit and local integration tests for the xAI realtime  example."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import time
import uuid

import httpx
import pytest
import websockets
from dotenv import load_dotenv

from inbound.agent import XAIRealtimeAgent
from utils import normalize_phone_number

load_dotenv()

TEST_PORT = 18001
LOCAL_WS_URL = f"ws://localhost:{TEST_PORT}/ws"
LOCAL_HTTP_URL = f"http://localhost:{TEST_PORT}"


class FakeWebSocket:
    """Minimal WebSocket stand-in that yields queued frames then stops."""

    def __init__(self, incoming: list[str]):
        self._incoming = list(incoming)
        self.sent: list[dict] = []

    async def receive_text(self) -> str:
        if self._incoming:
            return self._incoming.pop(0)
        raise Exception("1000 (normal closure)")

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))


def make_agent(websocket: object = None, stream_id: str = "stream-1") -> XAIRealtimeAgent:
    return XAIRealtimeAgent(
        websocket=websocket,
        call_id="call-1",
        from_number="+14155550123",
        stream_id=stream_id,
    )


class TestUnitConfig:
    """Unit tests for session configuration and phone normalization."""

    def test_session_config_uses_mulaw_passthrough_and_server_vad(self):
        session = make_agent()._build_session_config()["session"]

        assert session["audio"]["input"]["format"]["type"] == "audio/pcmu"
        assert session["audio"]["output"]["format"]["type"] == "audio/pcmu"
        assert session["turn_detection"] == {"type": "server_vad"}
        assert "input_audio_format" not in session
        assert any(tool["name"] == "end_call" for tool in session["tools"])

    @pytest.mark.asyncio
    async def test_receive_from_plivo_forwards_payload_unchanged(self):
        payload = "dGVzdC1tdWxhdw=="
        ws = FakeWebSocket(
            [
                json.dumps({"event": "media", "media": {"payload": payload}}),
                json.dumps({"event": "stop"}),
            ]
        )
        agent = make_agent(ws)
        agent._running = True

        sent: list[dict] = []

        async def capture(message: dict) -> None:
            sent.append(message)

        agent._realtime_send = capture  # type: ignore[method-assign]

        await agent._receive_from_plivo()

        appended = [
            message for message in sent if message.get("type") == "input_audio_buffer.append"
        ]
        assert len(appended) == 1
        assert appended[0]["audio"] == payload

    def test_normalize_phone_number_e164(self):
        assert normalize_phone_number("+1 657-233-8892") == "16572338892"


class TestLocalIntegration:
    """Local server integration checks without placing real calls."""

    @pytest.fixture(scope="class")
    def server_process(self):
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
        async with httpx.AsyncClient() as client:
            response = await client.get(LOCAL_HTTP_URL)
            assert response.status_code == 200
            assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_local_answer_webhook(self, server_process):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{LOCAL_HTTP_URL}/answer",
                params={"CallUUID": "test123", "From": "+15551234567", "To": "+16572338892"},
            )
            assert response.status_code == 200
            assert "application/xml" in response.headers["content-type"]
            assert "<Stream" in response.text

    @pytest.mark.asyncio
    async def test_local_websocket_receives_audio(self, server_process):
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
