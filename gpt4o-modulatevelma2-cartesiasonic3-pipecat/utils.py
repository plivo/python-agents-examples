"""Shared utilities, audio processing, and the Modulate STT service.

This module provides:
- Phone number normalization
- Audio format conversion (μ-law <-> PCM, resampling)
- ModulateSTTService: Velma-2 streaming as a Pipecat STT service

Note: VAD is handled by Pipecat framework (vad_analyzer on the user aggregator).

The STT service lives here rather than in inbound/ or outbound/ because both
agents use it and the canonical structure has no services/ directory. It is
transport-level audio code, which is what this module already owns.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator

import numpy as np
import phonenumbers
import websockets
from dotenv import load_dotenv
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from scipy import signal as scipy_signal

load_dotenv()

# =============================================================================
# Configuration (only constants consumed by utility functions)
# =============================================================================

DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "US")

# Audio format constants
PLIVO_SAMPLE_RATE = 8000  # Plivo uses 8kHz μ-law
CARTESIA_SAMPLE_RATE = 24000  # Cartesia Sonic default PCM output rate
MODULATE_SAMPLE_RATE = 8000  # Velma-2 receives s16le at the pipeline rate

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


def plivo_to_cartesia(mulaw_8k: bytes) -> bytes:
    """Convert Plivo audio (μ-law 8kHz) to Cartesia format (PCM16 24kHz)."""
    pcm_8k = ulaw_to_pcm(mulaw_8k)
    return resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, CARTESIA_SAMPLE_RATE)


def cartesia_to_plivo(pcm_24k: bytes) -> bytes:
    """Convert Cartesia audio (PCM16 24kHz) to Plivo format (μ-law 8kHz)."""
    pcm_8k = resample_audio(pcm_24k, CARTESIA_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
    return pcm_to_ulaw(pcm_8k)


# =============================================================================
# Modulate Velma-2 STT
# =============================================================================

MODULATE_STT_URL = "wss://platform.modulate.ai/api/velma-2-streaming"
MODULATE_STT_MODEL = "velma-2"


class ModulateSTTService(WebsocketSTTService):
    """Modulate Velma-2 streaming as a Pipecat STT service.

    Velma returns more than a transcript on a single socket: each finalised clip
    carries an optional emotion, accent and synthetic-voice score. The transcript
    drives the pipeline; the extra signals are logged and exposed on the
    TranscriptionFrame's ``result`` for anything downstream that wants them.

    Protocol (https://docs.modulate.ai/api-reference/velma/streaming):
      - API key goes in the query string, not a header. A bad key closes with 4001.
      - Exactly one configuration text frame must precede any audio.
      - Audio is raw binary; we send s16le at the pipeline's own sample rate.
      - ``partial_clip`` -> interim transcript, ``clip`` -> final.

    Note: ``behavior_detection`` events are documented for this endpoint but were
    not observed on the streaming socket during testing (2026-10-05). Behaviours
    come back reliably from the batch endpoint. The handler is kept because it is
    correct per the docs and costs nothing.
    """

    def __init__(
        self,
        *,
        api_key: str,
        behaviors: list[str] | None = None,
        produce_topics: bool = False,
        produce_summary: bool = False,
        sample_rate: int | None = None,
        **kwargs,
    ) -> None:
        # Velma-2 auto-detects language and exposes no model selector, so
        # language is explicitly None rather than left NOT_GIVEN; the base
        # class asserts every settings field has a real value.
        super().__init__(
            sample_rate=sample_rate,
            settings=STTSettings(model=MODULATE_STT_MODEL, language=None),
            **kwargs,
        )
        self._api_key = api_key
        self._config = json.dumps(
            {
                "behaviors": behaviors or [],
                "produce_topics": produce_topics,
                "produce_summary": produce_summary,
            }
        )
        self._websocket = None
        self._receive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    # -- lifecycle ------------------------------------------------------------

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame) -> None:
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        await super().cancel(frame)
        await self._disconnect()

    async def _connect(self) -> None:
        await self._connect_websocket()
        await super()._connect()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        if self._websocket:
            return
        url = (
            f"{MODULATE_STT_URL}?api_key={self._api_key}"
            f"&audio_format=s16le&sample_rate={self.sample_rate}&num_channels=1"
        )
        try:
            self._websocket = await websockets.connect(url, max_size=None)
        except Exception as e:
            logger.error(f"Modulate connect failed: {e}")
            self._websocket = None
            return
        await self._websocket.send(self._config)
        logger.info(f"Modulate Velma-2 connected at {self.sample_rate}Hz")

    async def _disconnect_websocket(self) -> None:
        if not self._websocket:
            return
        try:
            await self._websocket.send("")  # end-of-stream signal
            await self._websocket.close()
        except Exception as e:
            logger.debug(f"Modulate close: {e}")
        finally:
            self._websocket = None

    # -- audio in / transcripts out -------------------------------------------

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Forward caller audio. Transcripts arrive on the receive task."""
        if self._websocket is None:
            await self._connect()
        if self._websocket is None:
            logger.warning("Modulate unavailable, dropping audio")
            yield None
            return
        try:
            await self._websocket.send(audio)
        except Exception as e:
            logger.warning(f"Modulate send failed: {e}")
        yield None

    async def _receive_messages(self) -> None:
        async for message in self._websocket:
            if isinstance(message, bytes):
                continue
            try:
                await self._handle_event(json.loads(message))
            except json.JSONDecodeError:
                logger.warning(f"Non-JSON from Modulate: {message[:120]}")

    @staticmethod
    def _language(code: str | None) -> Language | None:
        if not code:
            return None
        try:
            return Language(code)
        except (ValueError, KeyError):
            return None

    async def _handle_event(self, event: dict) -> None:
        kind = event.get("type")

        if kind == "partial_clip":
            text = (event.get("partial_clip") or {}).get("text", "")
            if text:
                await self.push_frame(
                    InterimTranscriptionFrame(text, self._user_id, time_now_iso8601())
                )

        elif kind == "clip":
            clip = event.get("clip") or {}
            text = clip.get("text", "")
            if not text:
                return
            extras = {
                k: clip.get(k)
                for k in ("emotion", "accent", "deepfake_score", "speaker_label")
                if clip.get(k) is not None
            }
            if extras:
                logger.info(f"[Velma] {text[:48]!r} {extras}")
            await self.push_frame(
                TranscriptionFrame(
                    text,
                    self._user_id,
                    time_now_iso8601(),
                    self._language(clip.get("language")),
                    result=clip,
                )
            )
            await self.stop_processing_metrics()

        elif kind == "clip_update":
            # A refinement of a clip already transcribed. Log the better emotion
            # and accent, but do not push another frame or the LLM sees the same
            # words twice.
            clip = event.get("clip_update") or {}
            logger.debug(f"[Velma] refined: {clip.get('emotion')} {clip.get('accent')}")

        elif kind == "behavior_detection":
            detection = event.get("detection") or {}
            if detection.get("detected"):
                logger.warning(
                    f"[Velma] behaviour {detection.get('behavior_name')} "
                    f"@ {detection.get('confidence')}"
                )

        elif kind == "error":
            logger.error(f"Modulate error: {event.get('error')}")
