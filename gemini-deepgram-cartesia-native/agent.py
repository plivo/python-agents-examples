"""
Voice agent using Gemini LLM, Deepgram STT, and Cartesia TTS.

This module provides a voice agent that:
- Connects to Deepgram for real-time speech-to-text
- Uses Google Gemini for conversational responses
- Uses Cartesia for text-to-speech
- Handles bidirectional audio streaming with Plivo telephony

No frameworks are used - direct API integration only.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from typing import TYPE_CHECKING

import aiohttp
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from loguru import logger
from scipy import signal as scipy_signal

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_MODEL = os.getenv("DEEPGRAM_MODEL", "nova-2-phonecall")

CARTESIA_API_KEY = os.getenv("CARTESIA_API_KEY", "")
CARTESIA_VOICE_ID = os.getenv("CARTESIA_VOICE_ID", "79a125e8-cd45-4c13-8a67-188112f4dd22")
CARTESIA_MODEL = os.getenv("CARTESIA_MODEL", "sonic-2")

# Audio format constants
PLIVO_SAMPLE_RATE = 8000  # Plivo uses 8kHz u-law
DEEPGRAM_SAMPLE_RATE = 8000  # Deepgram can accept 8kHz
CARTESIA_SAMPLE_RATE = 24000  # Cartesia outputs 24kHz PCM

# =============================================================================
# System Prompt
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """
You are Alex, a friendly and professional customer service agent for TechFlow,
a software company that provides cloud-based productivity tools.

## Your Personality
- Warm, patient, and empathetic
- Professional but conversational
- You use natural speech patterns
- You never sound robotic or overly formal

## Audio Output Rules
- Your responses will be converted to speech, so never use special characters
- Spell out numbers naturally: say "twenty three dollars" not "$23"
- Keep responses concise - aim for 1-3 sentences unless explaining something complex
- Use natural pauses by breaking up longer responses

## Your Capabilities
You can help customers with:
1. Product information about TechFlow Pro, Teams, and Enterprise plans
2. Billing questions and payment issues
3. Technical support for basic issues
4. General inquiries

## Product Knowledge
- TechFlow Pro: twelve dollars per month, for individuals, 100GB storage
- TechFlow Teams: twenty five dollars per user per month, up to 25 people, 500GB
- TechFlow Enterprise: Custom pricing, unlimited users and storage, dedicated support

## Conversation Flow
1. Greet the caller warmly and ask how you can help
2. Listen and acknowledge their concern before jumping to solutions
3. Ask clarifying questions if needed
4. Provide clear, helpful responses
5. Confirm the customer is satisfied before ending
6. End with a friendly closing

## Important Guidelines
- Keep the conversation moving naturally
- Be honest if you cannot help and offer alternatives
- Ask if there is anything else before ending
""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# =============================================================================
# Audio Conversion Utilities
# =============================================================================

# u-law decoding table (ITU-T G.711)
_ULAW_DECODE_TABLE = np.array([
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
], dtype=np.int16)


def ulaw_to_pcm(ulaw_data: bytes) -> bytes:
    """Convert u-law encoded audio to 16-bit PCM."""
    ulaw_samples = np.frombuffer(ulaw_data, dtype=np.uint8)
    pcm_samples = _ULAW_DECODE_TABLE[ulaw_samples]
    return pcm_samples.tobytes()


def pcm_to_ulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM audio to u-law encoding."""
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
    if len(samples) == 0:
        return audio_data

    ratio = output_rate / input_rate
    new_length = int(len(samples) * ratio)
    resampled = scipy_signal.resample(samples.astype(np.float64), new_length)
    return np.clip(resampled, -32768, 32767).astype(np.int16).tobytes()


# =============================================================================
# Deepgram STT Client
# =============================================================================


class DeepgramSTT:
    """Real-time speech-to-text using Deepgram WebSocket API."""

    def __init__(self, on_transcript: callable):
        self.on_transcript = on_transcript
        self._ws = None
        self._session: aiohttp.ClientSession | None = None
        self._running = False

    async def connect(self) -> None:
        """Connect to Deepgram WebSocket."""
        self._session = aiohttp.ClientSession()
        self._running = True

        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?model={DEEPGRAM_MODEL}"
            f"&encoding=linear16"
            f"&sample_rate={DEEPGRAM_SAMPLE_RATE}"
            f"&channels=1"
            f"&punctuate=true"
            f"&interim_results=false"
        )

        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        self._ws = await self._session.ws_connect(url, headers=headers)
        logger.info("Connected to Deepgram STT")

        asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Receive transcription results from Deepgram."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    if data.get("type") == "Results":
                        channel = data.get("channel", {})
                        alternatives = channel.get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "")
                            if transcript.strip():
                                logger.info(f"STT transcript: {transcript}")
                                await self.on_transcript(transcript)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"Deepgram WebSocket error: {msg.data}")
                    break
        except Exception as e:
            if self._running:
                logger.error(f"Deepgram receive error: {e}")

    async def send_audio(self, pcm_audio: bytes) -> None:
        """Send PCM audio to Deepgram."""
        if self._ws and not self._ws.closed:
            await self._ws.send_bytes(pcm_audio)

    async def close(self) -> None:
        """Close the Deepgram connection."""
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()


# =============================================================================
# Cartesia TTS Client
# =============================================================================


class CartesiaTTS:
    """Text-to-speech using Cartesia WebSocket API for streaming."""

    def __init__(self):
        self._session: aiohttp.ClientSession | None = None
        self._ws = None
        self._context_id = None

    async def connect(self) -> None:
        """Initialize the HTTP session."""
        self._session = aiohttp.ClientSession()
        logger.info("Cartesia TTS client initialized")

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to speech using Cartesia HTTP API."""
        if not self._session:
            await self.connect()

        url = "https://api.cartesia.ai/tts/bytes"
        headers = {
            "X-API-Key": CARTESIA_API_KEY,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json",
        }
        payload = {
            "model_id": CARTESIA_MODEL,
            "transcript": text,
            "voice": {"mode": "id", "id": CARTESIA_VOICE_ID},
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": CARTESIA_SAMPLE_RATE,
            },
        }

        try:
            async with self._session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 200:
                    audio_data = await resp.read()
                    logger.debug(f"Cartesia TTS: synthesized {len(audio_data)} bytes")
                    return audio_data
                else:
                    error = await resp.text()
                    logger.error(f"Cartesia TTS error {resp.status}: {error}")
                    return b""
        except Exception as e:
            logger.error(f"Cartesia TTS request failed: {e}")
            return b""

    async def close(self) -> None:
        """Close the session."""
        if self._session:
            await self._session.close()


# =============================================================================
# Gemini LLM Client
# =============================================================================


class GeminiLLM:
    """Conversational LLM using Google Gemini."""

    def __init__(self, system_prompt: str):
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._system_prompt = system_prompt
        self._conversation_history: list[types.Content] = []

    async def generate_response(self, user_text: str) -> str:
        """Generate a response to user input."""
        self._conversation_history.append(
            types.Content(role="user", parts=[types.Part(text=user_text)])
        )

        try:
            response = await self._client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=self._conversation_history,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    temperature=0.7,
                    max_output_tokens=256,
                ),
            )

            assistant_text = response.text or ""
            logger.info(f"LLM response: {assistant_text[:100]}...")

            self._conversation_history.append(
                types.Content(role="model", parts=[types.Part(text=assistant_text)])
            )

            return assistant_text

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return "I'm sorry, I'm having trouble processing that. Could you please repeat?"


# =============================================================================
# Voice Agent
# =============================================================================


class VoiceAgent:
    """Voice agent that orchestrates STT, LLM, and TTS."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.from_number = from_number
        self.to_number = to_number

        self._running = False
        self._processing_lock = asyncio.Lock()
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()

        prompt = system_prompt or SYSTEM_PROMPT
        if from_number:
            prompt += f"\n\nCurrent caller's phone number: {from_number}"

        self._llm = GeminiLLM(prompt)
        self._tts = CartesiaTTS()
        self._stt = DeepgramSTT(on_transcript=self._handle_transcript)

    async def _handle_transcript(self, transcript: str) -> None:
        """Handle transcribed text from STT."""
        async with self._processing_lock:
            if not transcript.strip():
                return

            logger.info(f"Processing user input: {transcript}")

            # Get LLM response
            response_text = await self._llm.generate_response(transcript)
            if not response_text:
                return

            # Synthesize speech
            tts_audio = await self._tts.synthesize(response_text)
            if not tts_audio:
                return

            # Convert to Plivo format (24kHz PCM -> 8kHz u-law)
            pcm_8k = resample_audio(tts_audio, CARTESIA_SAMPLE_RATE, PLIVO_SAMPLE_RATE)
            ulaw_audio = pcm_to_ulaw(pcm_8k)

            # Queue for sending
            await self._audio_queue.put(ulaw_audio)

    async def run(self) -> None:
        """Run the voice agent session."""
        logger.info(f"Starting voice agent for call {self.call_id}")
        self._running = True

        try:
            await self._stt.connect()
            await self._tts.connect()

            # Send initial greeting
            greeting = "Hello! Thank you for calling TechFlow. How can I help you today?"
            greeting_audio = await self._tts.synthesize(greeting)
            if greeting_audio:
                pcm_8k = resample_audio(
                    greeting_audio, CARTESIA_SAMPLE_RATE, PLIVO_SAMPLE_RATE
                )
                ulaw_audio = pcm_to_ulaw(pcm_8k)
                await self._audio_queue.put(ulaw_audio)

            # Run concurrent tasks
            tasks = [
                asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
                asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
            ]

            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")

        except Exception as e:
            logger.error(f"Voice agent error: {e}")
        finally:
            self._running = False
            await self._stt.close()
            await self._tts.close()
            for task in tasks:
                if not task.done():
                    task.cancel()
            logger.info(f"Voice agent ended for call {self.call_id}")

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo and send to STT."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        ulaw_audio = base64.b64decode(payload)
                        pcm_audio = ulaw_to_pcm(ulaw_audio)
                        await self._stt.send_audio(pcm_audio)

                elif event == "stop":
                    logger.info("Received stop event from Plivo")
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _send_to_plivo(self) -> None:
        """Send audio from queue to Plivo."""
        CHUNK_SIZE = 160  # 20ms at 8kHz u-law

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(
                        self._audio_queue.get(), timeout=0.1
                    )

                    # Send in chunks
                    for i in range(0, len(audio), CHUNK_SIZE):
                        chunk = audio[i : i + CHUNK_SIZE]
                        if len(chunk) < CHUNK_SIZE:
                            chunk = chunk + b"\xff" * (CHUNK_SIZE - len(chunk))

                        message = {
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": base64.b64encode(chunk).decode("utf-8"),
                            },
                        }
                        await self.websocket.send_text(json.dumps(message))
                        await asyncio.sleep(0.018)  # ~20ms pacing

                except TimeoutError:
                    continue

        except asyncio.CancelledError:
            pass


# =============================================================================
# Public API
# =============================================================================


async def run_agent(
    websocket: WebSocket,
    call_id: str,
    from_number: str = "",
    to_number: str = "",
    system_prompt: str | None = None,
) -> None:
    """Run a voice agent session for an incoming call."""
    agent = VoiceAgent(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
    )
    await agent.run()
