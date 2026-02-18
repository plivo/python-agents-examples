"""Shared utilities and audio processing.

This module provides:
- Phone number normalization
- Audio format conversion (μ-law <-> PCM, resampling)
- Silero VAD processor for voice activity detection
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime
import phonenumbers
from dotenv import load_dotenv
from loguru import logger
from scipy import signal as scipy_signal

load_dotenv()

# =============================================================================
# Configuration (only constants consumed by utility functions)
# =============================================================================

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# Audio format constants
PLIVO_SAMPLE_RATE = 8000  # Plivo uses 8kHz μ-law
GEMINI_INPUT_RATE = 16000  # Gemini expects 16kHz PCM input
GEMINI_OUTPUT_RATE = 24000  # Gemini outputs 24kHz PCM
VAD_SAMPLE_RATE = 16000  # Silero VAD operates at 16kHz

# Silero VAD configuration
VAD_CHUNK_SAMPLES = 512  # 32ms at 16kHz (Silero expects 512 samples at 16kHz)
VAD_START_THRESHOLD = 0.5  # Speech probability to trigger speech start
VAD_END_THRESHOLD = 0.35  # Speech probability below this to consider silence
VAD_MIN_SILENCE_MS = 300  # Minimum silence duration (ms) to trigger speech end

# ONNX-based Silero VAD configuration (used by SileroVAD class)
VAD_CONFIDENCE_THRESHOLD = float(os.getenv("VAD_CONFIDENCE", "0.5"))
VAD_ONNX_SAMPLE_RATE = 8000
VAD_FRAME_SAMPLES = 256  # 32ms at 8kHz
VAD_FRAME_BYTES = VAD_FRAME_SAMPLES * 2  # 512 bytes PCM16
VAD_SPEECH_FRAMES_THRESHOLD = 2  # ~64ms to confirm speech
VAD_SILENCE_FRAMES_THRESHOLD = 10  # ~320ms to confirm silence
VAD_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)

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
        -32124, -31100, -30076, -29052, -28028, -27004, -25980, -24956,
        -23932, -22908, -21884, -20860, -19836, -18812, -17788, -16764,
        -15996, -15484, -14972, -14460, -13948, -13436, -12924, -12412,
        -11900, -11388, -10876, -10364, -9852, -9340, -8828, -8316,
        -7932, -7676, -7420, -7164, -6908, -6652, -6396, -6140,
        -5884, -5628, -5372, -5116, -4860, -4604, -4348, -4092,
        -3900, -3772, -3644, -3516, -3388, -3260, -3132, -3004,
        -2876, -2748, -2620, -2492, -2364, -2236, -2108, -1980,
        -1884, -1820, -1756, -1692, -1628, -1564, -1500, -1436,
        -1372, -1308, -1244, -1180, -1116, -1052, -988, -924,
        -876, -844, -812, -780, -748, -716, -684, -652,
        -620, -588, -556, -524, -492, -460, -428, -396,
        -372, -356, -340, -324, -308, -292, -276, -260,
        -244, -228, -212, -196, -180, -164, -148, -132,
        -120, -112, -104, -96, -88, -80, -72, -64,
        -56, -48, -40, -32, -24, -16, -8, 0,
        32124, 31100, 30076, 29052, 28028, 27004, 25980, 24956,
        23932, 22908, 21884, 20860, 19836, 18812, 17788, 16764,
        15996, 15484, 14972, 14460, 13948, 13436, 12924, 12412,
        11900, 11388, 10876, 10364, 9852, 9340, 8828, 8316,
        7932, 7676, 7420, 7164, 6908, 6652, 6396, 6140,
        5884, 5628, 5372, 5116, 4860, 4604, 4348, 4092,
        3900, 3772, 3644, 3516, 3388, 3260, 3132, 3004,
        2876, 2748, 2620, 2492, 2364, 2236, 2108, 1980,
        1884, 1820, 1756, 1692, 1628, 1564, 1500, 1436,
        1372, 1308, 1244, 1180, 1116, 1052, 988, 924,
        876, 844, 812, 780, 748, 716, 684, 652,
        620, 588, 556, 524, 492, 460, 428, 396,
        372, 356, 340, 324, 308, 292, 276, 260,
        244, 228, 212, 196, 180, 164, 148, 132,
        120, 112, 104, 96, 88, 80, 72, 64,
        56, 48, 40, 32, 24, 16, 8, 0,
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


def plivo_to_gemini(mulaw_8k: bytes) -> bytes:
    """Convert Plivo audio (μ-law 8kHz) to Gemini format (PCM16 16kHz)."""
    pcm_8k = ulaw_to_pcm(mulaw_8k)
    return resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, GEMINI_INPUT_RATE)


def gemini_to_plivo(pcm_24k: bytes) -> bytes:
    """Convert Gemini audio (PCM16 24kHz) to Plivo format (μ-law 8kHz)."""
    pcm_8k = resample_audio(pcm_24k, GEMINI_OUTPUT_RATE, PLIVO_SAMPLE_RATE)
    return pcm_to_ulaw(pcm_8k)


def plivo_to_vad(mulaw_8k: bytes) -> np.ndarray:
    """Convert Plivo audio (μ-law 8kHz) to Silero VAD format (float32 16kHz)."""
    pcm_8k = ulaw_to_pcm(mulaw_8k)
    pcm_16k = resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, VAD_SAMPLE_RATE)
    samples = np.frombuffer(pcm_16k, dtype=np.int16).astype(np.float32)
    return samples / 32768.0  # Normalize to [-1, 1]


# =============================================================================
# Silero VAD (ONNX Runtime)
# =============================================================================


def _get_silero_model_path() -> str:
    """Get path to Silero VAD ONNX model, downloading if needed."""
    cache_dir = Path.home() / ".cache" / "silero-vad"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / "silero_vad.onnx"

    if not model_path.exists():
        logger.info(f"Downloading Silero VAD model to {model_path}...")
        urllib.request.urlretrieve(VAD_MODEL_URL, str(model_path))
        logger.info("Silero VAD model downloaded successfully")

    return str(model_path)


class SileroVAD:
    """Lightweight Silero VAD using ONNX Runtime (no PyTorch needed)."""

    def __init__(self, model_path: str):
        self._session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        # LSTM state: 2 layers, 1 batch, 128 hidden units
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        # Context window: 32 samples at 8kHz
        self._context = np.zeros(32, dtype=np.float32)
        self._sr = np.array(VAD_ONNX_SAMPLE_RATE, dtype=np.int64)

    def reset_states(self) -> None:
        """Zero out LSTM state for a fresh start."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(32, dtype=np.float32)

    def process_frame(self, pcm_bytes: bytes) -> float:
        """Process a single audio frame and return speech confidence.

        Args:
            pcm_bytes: 512 bytes of 16-bit PCM at 8kHz (256 samples)

        Returns:
            Speech confidence between 0.0 and 1.0
        """
        # Convert int16 PCM to float32 normalized to [-1, 1]
        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Prepend context window
        input_frame = np.concatenate((self._context, samples)).reshape(1, -1)

        # Run ONNX inference
        ort_inputs = {
            "input": input_frame,
            "state": self._state,
            "sr": self._sr,
        }
        ort_outputs = self._session.run(None, ort_inputs)

        confidence = ort_outputs[0].item()
        self._state = ort_outputs[1]

        return confidence


class SileroVADProcessor:
    """Processes audio frames through Silero VAD for speech detection.

    Accumulates audio in a buffer and runs VAD when enough samples are available.
    Tracks speech state transitions (silence -> speaking -> silence) to determine
    when the user has finished a turn.

    Uses ONNX-based SileroVAD for inference without PyTorch dependency.
    """

    def __init__(self):
        model_path = _get_silero_model_path()
        self._model = SileroVAD(model_path)
        self._buffer = bytearray()
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_count = 0
        self._min_silence_frames = VAD_SILENCE_FRAMES_THRESHOLD

    def reset(self) -> None:
        """Reset VAD state for a new turn."""
        self._model.reset_states()
        self._buffer = bytearray()
        self._is_speaking = False
        self._silence_frames = 0
        self._speech_count = 0

    def process(self, pcm_8k: bytes) -> tuple[bool, bool]:
        """Process 8kHz PCM audio and return (speech_started, speech_ended) events.

        Args:
            pcm_8k: PCM16 audio at 8kHz (variable length from Plivo)

        Returns:
            Tuple of (speech_started, speech_ended) booleans.
        """
        self._buffer.extend(pcm_8k)

        speech_started = False
        speech_ended = False

        while len(self._buffer) >= VAD_FRAME_BYTES:
            frame_bytes = bytes(self._buffer[:VAD_FRAME_BYTES])
            self._buffer = self._buffer[VAD_FRAME_BYTES:]

            try:
                confidence = self._model.process_frame(frame_bytes)
            except Exception:
                continue

            is_speech = confidence > VAD_CONFIDENCE_THRESHOLD

            if not self._is_speaking:
                if is_speech:
                    self._speech_count += 1
                    if self._speech_count >= VAD_SPEECH_FRAMES_THRESHOLD:
                        self._is_speaking = True
                        self._silence_frames = 0
                        speech_started = True
                        logger.debug(f"VAD: speech started (confidence={confidence:.2f})")
                else:
                    self._speech_count = 0
            else:
                if not is_speech:
                    self._silence_frames += 1
                    if self._silence_frames >= self._min_silence_frames:
                        self._is_speaking = False
                        self._silence_frames = 0
                        self._speech_count = 0
                        speech_ended = True
                        logger.debug(f"VAD: speech ended (confidence={confidence:.2f})")
                else:
                    self._silence_frames = 0

        return speech_started, speech_ended

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking
