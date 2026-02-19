"""Shared test helpers for live call tests."""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import httpx
import pytest

NGROK_BIN = os.getenv("NGROK_BIN", "ngrok")
NGROK_API = "http://localhost:4040/api/tunnels"


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
                for t in tunnels:
                    if t.get("proto") == "https":
                        public_url = t["public_url"]
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


def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe MP3 audio using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name

    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)
