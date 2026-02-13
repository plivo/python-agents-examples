"""Shared configuration, utilities, and audio processing.

This module provides:
- Configuration loaded from environment variables
- Phone number normalization
- Audio format conversion (μ-law <-> PCM, resampling)
- Silero VAD processor for voice activity detection
"""

from __future__ import annotations

import os

import numpy as np
import phonenumbers
from dotenv import load_dotenv
from loguru import logger
from scipy import signal as scipy_signal
from silero_vad import load_silero_vad

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

# Server
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# Plivo credentials
PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_FROM_NUMBER = os.getenv("PLIVO_FROM_NUMBER", "")
DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# Public URL for webhooks (ngrok URL or production domain)
PUBLIC_URL = os.getenv("PUBLIC_URL", "")

# xAI / Grok
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-3-fast-voice")
GROK_VOICE = os.getenv("GROK_VOICE", "Sal")
XAI_REALTIME_URL = "wss://api.x.ai/v1/realtime"

# Audio format constants
PLIVO_SAMPLE_RATE = 8000  # Plivo uses 8kHz μ-law
GROK_SAMPLE_RATE = 24000  # Grok default PCM sample rate
VAD_SAMPLE_RATE = 16000  # Silero VAD operates at 16kHz

# Silero VAD configuration
VAD_CHUNK_SAMPLES = 512  # 32ms at 16kHz (Silero expects 512 samples at 16kHz)
VAD_START_THRESHOLD = 0.5  # Speech probability to trigger speech start
VAD_END_THRESHOLD = 0.35  # Speech probability below this to consider silence
VAD_MIN_SILENCE_MS = 300  # Minimum silence duration (ms) to trigger speech end
VAD_PRE_SPEECH_PAD_MS = 150  # Audio to keep before speech start for context

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
    """Convert μ-law encoded audio to 16-bit PCM."""
    ulaw_samples = np.frombuffer(ulaw_data, dtype=np.uint8)
    pcm_samples = _ULAW_DECODE_TABLE[ulaw_samples]
    return pcm_samples.tobytes()


def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM audio to μ-law encoding."""
    BIAS = 0x84
    CLIP = 32635

    pcm_samples = np.frombuffer(pcm_data, dtype=np.int16).astype(np.int32)
    sign = (pcm_samples >> 8) & 0x80
    pcm_samples = np.where(sign != 0, -pcm_samples, pcm_samples)
    pcm_samples = np.clip(pcm_samples, 0, CLIP) + BIAS

    segment = np.floor(np.log2(pcm_samples >> 7)).astype(np.int32)
    segment = np.clip(segment, 0, 7)

    ulaw = sign | ((segment << 4) | ((pcm_samples >> (segment + 3)) & 0x0F))
    ulaw = ~ulaw & 0xFF

    return ulaw.astype(np.uint8).tobytes()


def resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes:
    """Resample audio from one sample rate to another."""
    if input_rate == output_rate:
        return audio_data

    samples = np.frombuffer(audio_data, dtype=np.int16)
    ratio = output_rate / input_rate
    new_length = int(len(samples) * ratio)
    resampled = scipy_signal.resample(samples.astype(np.float64), new_length)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


def plivo_to_grok(mulaw_8k: bytes) -> bytes:
    """Convert Plivo audio (μ-law 8kHz) to Grok format (PCM16 24kHz)."""
    pcm_8k = ulaw_to_pcm(mulaw_8k)
    return resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, GROK_SAMPLE_RATE)


def grok_to_plivo(pcm_24k: bytes) -> bytes:
    """Convert Grok audio (PCM16 24kHz) to Plivo format (μ-law 8kHz)."""
    pcm_8k = resample_audio(pcm_24k, GROK_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
    return pcm_to_ulaw(pcm_8k)


def plivo_to_vad(mulaw_8k: bytes) -> np.ndarray:
    """Convert Plivo audio (μ-law 8kHz) to Silero VAD format (float32 16kHz)."""
    pcm_8k = ulaw_to_pcm(mulaw_8k)
    pcm_16k = resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, VAD_SAMPLE_RATE)
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32)
    return samples / 32768.0  # Normalize to [-1, 1]


# =============================================================================
# Silero VAD Processor
# =============================================================================


class SileroVADProcessor:
    """Processes audio frames through Silero VAD for speech detection.

    Accumulates audio in a buffer and runs VAD when enough samples are available.
    Tracks speech state transitions (silence -> speaking -> silence) to determine
    when the user has finished a turn.
    """

    def __init__(self):
        self._model = load_silero_vad(onnx=True)
        self._buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_frames = 0
        self._min_silence_frames = int(
            VAD_MIN_SILENCE_MS / (VAD_CHUNK_SAMPLES / VAD_SAMPLE_RATE * 1000)
        )

    def reset(self) -> None:
        """Reset VAD state for a new turn."""
        self._model.reset_states()
        self._buffer = np.array([], dtype=np.float32)
        self._is_speaking = False
        self._silence_frames = 0

    def process(self, audio_f32: np.ndarray) -> tuple[bool, bool]:
        """Process audio and return (speech_started, speech_ended) events.

        Args:
            audio_f32: Float32 audio samples normalized to [-1, 1] at 16kHz.

        Returns:
            Tuple of (speech_started, speech_ended) booleans. Only one can be
            True at a time. Both False means no state change.
        """
        import torch

        self._buffer = np.concatenate([self._buffer, audio_f32])

        speech_started = False
        speech_ended = False

        while len(self._buffer) >= VAD_CHUNK_SAMPLES:
            chunk = self._buffer[:VAD_CHUNK_SAMPLES]
            self._buffer = self._buffer[VAD_CHUNK_SAMPLES:]

            chunk_tensor = torch.from_numpy(chunk)
            speech_prob = self._model(chunk_tensor, VAD_SAMPLE_RATE).item()

            if not self._is_speaking:
                if speech_prob >= VAD_START_THRESHOLD:
                    self._is_speaking = True
                    self._silence_frames = 0
                    speech_started = True
                    logger.debug(f"VAD: speech started (prob={speech_prob:.2f})")
            else:
                if speech_prob < VAD_END_THRESHOLD:
                    self._silence_frames += 1
                    if self._silence_frames >= self._min_silence_frames:
                        self._is_speaking = False
                        self._silence_frames = 0
                        speech_ended = True
                        logger.debug(f"VAD: speech ended (prob={speech_prob:.2f})")
                else:
                    self._silence_frames = 0

        return speech_started, speech_ended

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
