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
import time
import urllib.request

import numpy as np
import phonenumbers
from dotenv import load_dotenv
from loguru import logger
from scipy import signal as scipy_signal

load_dotenv()

# =============================================================================
# Configuration — only what utils.py functions consume
# =============================================================================

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# Audio sample rates
PLIVO_SAMPLE_RATE = 8000  # Plivo uses 8kHz μ-law
DEEPGRAM_SAMPLE_RATE = 8000  # Deepgram Voice Agent is configured for μ-law 8kHz in and out

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
# Audio Conversion Utilities
# =============================================================================

# μ-law decoding table (ITU-T G.711)
_ULAW_DECODE_TABLE = np.array(
    [
        -32124,
        -31100,
        -30076,
        -29052,
        -28028,
        -27004,
        -25980,
        -24956,
        -23932,
        -22908,
        -21884,
        -20860,
        -19836,
        -18812,
        -17788,
        -16764,
        -15996,
        -15484,
        -14972,
        -14460,
        -13948,
        -13436,
        -12924,
        -12412,
        -11900,
        -11388,
        -10876,
        -10364,
        -9852,
        -9340,
        -8828,
        -8316,
        -7932,
        -7676,
        -7420,
        -7164,
        -6908,
        -6652,
        -6396,
        -6140,
        -5884,
        -5628,
        -5372,
        -5116,
        -4860,
        -4604,
        -4348,
        -4092,
        -3900,
        -3772,
        -3644,
        -3516,
        -3388,
        -3260,
        -3132,
        -3004,
        -2876,
        -2748,
        -2620,
        -2492,
        -2364,
        -2236,
        -2108,
        -1980,
        -1884,
        -1820,
        -1756,
        -1692,
        -1628,
        -1564,
        -1500,
        -1436,
        -1372,
        -1308,
        -1244,
        -1180,
        -1116,
        -1052,
        -988,
        -924,
        -876,
        -844,
        -812,
        -780,
        -748,
        -716,
        -684,
        -652,
        -620,
        -588,
        -556,
        -524,
        -492,
        -460,
        -428,
        -396,
        -372,
        -356,
        -340,
        -324,
        -308,
        -292,
        -276,
        -260,
        -244,
        -228,
        -212,
        -196,
        -180,
        -164,
        -148,
        -132,
        -120,
        -112,
        -104,
        -96,
        -88,
        -80,
        -72,
        -64,
        -56,
        -48,
        -40,
        -32,
        -24,
        -16,
        -8,
        0,
        32124,
        31100,
        30076,
        29052,
        28028,
        27004,
        25980,
        24956,
        23932,
        22908,
        21884,
        20860,
        19836,
        18812,
        17788,
        16764,
        15996,
        15484,
        14972,
        14460,
        13948,
        13436,
        12924,
        12412,
        11900,
        11388,
        10876,
        10364,
        9852,
        9340,
        8828,
        8316,
        7932,
        7676,
        7420,
        7164,
        6908,
        6652,
        6396,
        6140,
        5884,
        5628,
        5372,
        5116,
        4860,
        4604,
        4348,
        4092,
        3900,
        3772,
        3644,
        3516,
        3388,
        3260,
        3132,
        3004,
        2876,
        2748,
        2620,
        2492,
        2364,
        2236,
        2108,
        1980,
        1884,
        1820,
        1756,
        1692,
        1628,
        1564,
        1500,
        1436,
        1372,
        1308,
        1244,
        1180,
        1116,
        1052,
        988,
        924,
        876,
        844,
        812,
        780,
        748,
        716,
        684,
        652,
        620,
        588,
        556,
        524,
        492,
        460,
        428,
        396,
        372,
        356,
        340,
        324,
        308,
        292,
        276,
        260,
        244,
        228,
        212,
        196,
        180,
        164,
        148,
        132,
        120,
        112,
        104,
        96,
        88,
        80,
        72,
        64,
        56,
        48,
        40,
        32,
        24,
        16,
        8,
        0,
    ],
    dtype=np.int16,
)


def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    """Convert μ-law encoded audio to 16-bit PCM.

    μ-law is a companding algorithm used in telephony (G.711 standard).
    This replaces the deprecated audioop.ulaw2lin function.

    Args:
        ulaw_data: μ-law encoded audio bytes

    Returns:
        16-bit PCM audio bytes
    """
    ulaw_samples = np.frombuffer(ulaw_data, dtype=np.uint8)
    pcm_samples = _ULAW_DECODE_TABLE[ulaw_samples]
    return pcm_samples.tobytes()


def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM audio to μ-law encoding.

    This replaces the deprecated audioop.lin2ulaw function.

    Args:
        pcm_data: 16-bit PCM audio bytes

    Returns:
        μ-law encoded audio bytes
    """
    BIAS = 0x84
    CLIP = 32635

    pcm_samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.int32)
    sign = (pcm_samples >> 8) & 0x80
    pcm_samples = np.where(sign != 0, -pcm_samples, pcm_samples)
    pcm_samples = np.clip(pcm_samples, 0, CLIP) + BIAS

    # Find segment using vectorized log2
    segment = np.floor(np.log2(pcm_samples >> 7)).astype(np.int32)
    segment = np.clip(segment, 0, 7)

    # Build μ-law byte
    ulaw = sign | ((segment << 4) | ((pcm_samples >> (segment + 3)) & 0x0F))
    ulaw = ~ulaw & 0xFF

    return ulaw.astype(np.uint8).tobytes()


def resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes:
    """Resample audio from one sample rate to another.

    Args:
        audio_data: Raw PCM audio bytes (16-bit signed integers)
        input_rate: Source sample rate in Hz
        output_rate: Target sample rate in Hz

    Returns:
        Resampled audio as bytes
    """
    if input_rate == output_rate:
        return audio_data

    samples = np.frombuffer(audio_data, dtype=np.int16)
    ratio = output_rate / input_rate
    new_length = int(len(samples) * ratio)
    resampled = scipy_signal.resample(samples.astype(np.float64), new_length)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


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
CLOUDFLARED_INSTALL_HINT = (
    "Install cloudflared: `brew install cloudflared` (macOS), or see "
    "https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/"
    "downloads/ (Linux/Windows)"
)


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
        raise TunnelError(f"cloudflared not found on PATH. {CLOUDFLARED_INSTALL_HINT}")

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


def wait_until_reachable(url: str, timeout_s: float = 60.0, interval_s: float = 1.0) -> bool:
    """Poll ``url`` until it answers HTTP 200 (new quick-tunnel DNS can take a few seconds)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(interval_s)
    return False
