"""Shared test helpers for live call tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import httpx
import plivo
import pytest

NGROK_BIN = os.getenv("NGROK_BIN", "ngrok")
NGROK_API = "http://localhost:4040/api/tunnels"


def require_public_ready(public_url: str, path: str = "/", timeout: float = 10.0) -> None:
    """Skip the test unless the public URL serves HTTP 200.

    A broken or stale ngrok tunnel with nothing behind it answers 502, and a
    missing tunnel raises a connection error. Either way the live server is not
    actually reachable, so the test should skip rather than fail.
    """
    url = public_url.rstrip("/") + path
    try:
        resp = httpx.get(url, timeout=timeout)
    except Exception as exc:
        pytest.skip(f"Public URL {url} not reachable: {exc}")
    if resp.status_code != 200:
        pytest.skip(f"Public URL {url} returned {resp.status_code}; live server not available")


def start_ngrok(port: int) -> tuple[subprocess.Popen, str]:
    """Start ngrok tunnel and return (process, public_url)."""
    proc = subprocess.Popen(
        [NGROK_BIN, "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    public_url = None
    for _ in range(30):
        try:
            resp = httpx.get(NGROK_API, timeout=1.0)
            if resp.status_code == 200:
                tunnels = resp.json().get("tunnels", [])
                for tunnel in tunnels:
                    if tunnel.get("proto") == "https":
                        public_url = tunnel["public_url"]
                        break
                if public_url:
                    break
        except Exception:
            pass
        time.sleep(0.5)

    if not public_url:
        proc.terminate()
        proc.wait()
        pytest.skip("ngrok did not start or no HTTPS tunnel found")

    return proc, public_url


def stop_ngrok(proc: subprocess.Popen) -> None:
    """Stop ngrok process."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
        proc.wait()


def wait_for_recording(
    client: plivo.RestClient, call_uuid: str, timeout: float = 30.0
) -> str | None:
    """Poll for a recording to appear for the given call UUID."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            recordings = client.recordings.list(call_uuid=call_uuid)
            objects = recordings.get("objects", []) if isinstance(recordings, dict) else []
            if not objects and hasattr(recordings, "objects"):
                objects = recordings.objects or []
            if objects:
                recording = objects[0]
                if isinstance(recording, dict):
                    url = recording.get("recording_url", "")
                else:
                    url = getattr(recording, "recording_url", "")
                if url:
                    return url
        except Exception:
            pass
        time.sleep(3)
    return None


def download_recording(url: str) -> bytes:
    """Download recording audio from URL."""
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return resp.content


def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe MP3 audio using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as file_handle:
        file_handle.write(audio_data)
        tmp_path = file_handle.name

    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        os.unlink(tmp_path)
