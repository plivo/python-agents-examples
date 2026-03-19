"""
xAI TTS Service for Pipecat
============================
Custom TTSService subclass for xAI's TTS API (POST /v1/tts).

xAI returns MP3 audio (not PCM), so we accumulate the full response
then decode to PCM via pydub before yielding audio frames.
"""

import io
from typing import AsyncGenerator, Optional

import httpx
from loguru import logger
from pydub import AudioSegment

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

XAI_TTS_URL = "https://api.x.ai/v1/tts"

VALID_VOICES = ["Ara", "Eve", "Rex", "Sal", "Leo"]


class XaiTTSService(TTSService):
    """xAI Text-to-Speech service.

    Sends text to POST /v1/tts and decodes the returned MP3 to PCM.
    TTFB is measured as time from request start to first byte received.
    """

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str = "Ara",
        language: str = "en",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._api_key = api_key
        self._voice_id = voice_id
        self._language = language
        self._client = httpx.AsyncClient(timeout=30.0)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()

            resp = await self._client.post(
                XAI_TTS_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "voice_id": self._voice_id,
                    "language": self._language,
                },
            )

            await self.stop_ttfb_metrics()

            if resp.status_code != 200:
                logger.error(f"{self} xAI TTS error (status {resp.status_code}): {resp.text[:200]}")
                yield ErrorFrame(error=f"xAI TTS error: {resp.status_code}")
                return

            await self.start_tts_usage_metrics(text)

            # Decode MP3 → PCM
            mp3_bytes = resp.content
            audio = AudioSegment.from_mp3(io.BytesIO(mp3_bytes))
            audio = audio.set_frame_rate(self.sample_rate).set_channels(1).set_sample_width(2)
            pcm_data = audio.raw_data

            # Yield frames
            CHUNK_SIZE = 8192
            yield TTSStartedFrame(context_id=context_id)
            for i in range(0, len(pcm_data), CHUNK_SIZE):
                chunk = pcm_data[i : i + CHUNK_SIZE]
                yield TTSAudioRawFrame(chunk, self.sample_rate, 1, context_id=context_id)
            yield TTSStoppedFrame(context_id=context_id)

        except Exception as e:
            logger.error(f"{self} xAI TTS exception: {e}")
            yield ErrorFrame(error=f"xAI TTS error: {e}")
