"""Shared utilities and audio processing.

This module provides:
- Phone number normalization
- Audio format conversion (μ-law <-> PCM, resampling)
- Plivo <-> Deepgram Voice Agent audio helpers (pass-through: both sides speak
  μ-law 8kHz, so no transcoding happens on the hot path)
- Cloudflare quick tunnel helpers for the ``--tunnel`` one-command local mode
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading

import phonenumbers
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# =============================================================================
# Configuration — only what utils.py functions consume
# =============================================================================

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# =============================================================================
# Phone Number Utilities
# =============================================================================


def normalize_phone_number(phone: str, default_region: str = DEFAULT_COUNTRY_CODE) -> str:
    """Normalize phone number to E.164 format (digits only, no leading +)."""
    if not phone:
        return ""

    try:
        parsed = phonenumbers.parse(phone, default_region)
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164.lstrip("+")
    except phonenumbers.NumberParseException as e:
        logger.warning(f"Failed to parse phone number '{phone}': {e}")
        return "".join(c for c in phone if c.isdigit())


# =============================================================================
# Audio Conversion
# =============================================================================


def plivo_to_deepgram(mulaw_8k: bytes) -> bytes:
    """Convert Plivo audio to Deepgram Voice Agent input format (pass-through).

    The agent's Settings declare ``audio.input = {encoding: "mulaw", sample_rate: 8000}``,
    which is exactly what Plivo streams, so the bytes are forwarded unchanged.
    """
    return mulaw_8k


def deepgram_to_plivo(mulaw_8k: bytes) -> bytes:
    """Convert Deepgram Voice Agent output audio to Plivo format (pass-through).

    The agent's Settings declare ``audio.output = {encoding: "mulaw", sample_rate: 8000,
    container: "none"}``, so Deepgram emits raw headerless μ-law 8kHz that Plivo can
    play directly.
    """
    return mulaw_8k


# =============================================================================
# Cloudflare quick tunnel (``--tunnel``)
# =============================================================================
#
# ``cloudflared tunnel --url http://localhost:PORT`` creates a free, account-less
# public HTTPS URL (https://<random>.trycloudflare.com) that proxies HTTP and
# WebSockets to the local server, so Plivo can reach /answer and /ws without ngrok.

TUNNEL_URL_PATTERN = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
CLOUDFLARED_INSTALL_URL = "https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/"


class TunnelError(RuntimeError):
    """The quick tunnel could not be started."""


def parse_tunnel_url(line: str) -> str:
    """Return the trycloudflare.com URL in a cloudflared log line, or ``""``."""
    match = TUNNEL_URL_PATTERN.search(line)
    return match.group(0) if match else ""


def start_quick_tunnel(port: int, timeout_s: float = 30.0) -> tuple[str, subprocess.Popen]:
    """Start a Cloudflare quick tunnel to ``localhost:port``.

    Returns ``(public_url, process)``; stop it with :func:`stop_tunnel`.
    Raises :class:`TunnelError` if cloudflared is missing or no URL appears in time.
    """
    binary = shutil.which("cloudflared")
    if not binary:
        raise TunnelError(f"cloudflared not found on PATH. Install it: {CLOUDFLARED_INSTALL_URL}")

    proc = subprocess.Popen(
        [binary, "tunnel", "--no-autoupdate", "--url", f"http://localhost:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # cloudflared logs (including the URL) go to stderr
        text=True,
    )
    found: list[str] = []
    ready = threading.Event()

    def _read_logs() -> None:
        # Keep draining stderr for the tunnel's lifetime so the pipe never fills up.
        assert proc.stderr is not None
        for line in proc.stderr:
            if not found and (url := parse_tunnel_url(line)):
                found.append(url)
                ready.set()
        ready.set()  # process exited

    threading.Thread(target=_read_logs, name="cloudflared-logs", daemon=True).start()

    if not ready.wait(timeout_s) or not found:
        stop_tunnel(proc)
        raise TunnelError(f"cloudflared did not report a public URL within {timeout_s:.0f}s")
    return found[0], proc


def stop_tunnel(proc: subprocess.Popen | None) -> None:
    """Terminate a tunnel started by :func:`start_quick_tunnel` (idempotent)."""
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
