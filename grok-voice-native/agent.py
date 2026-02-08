"""
Voice agent using xAI Grok Voice Agent API for speech-to-speech conversations.

This module provides a production-ready voice agent that:
- Connects to xAI's Grok Realtime API for real-time speech processing
- Uses Silero VAD for client-side voice activity detection and turn management
- Handles bidirectional audio streaming with Plivo telephony
- Supports function calling for actions during conversations
- Manages audio format conversion between Plivo (μ-law 8kHz) and Grok (PCM 24kHz)

Usage:
    from agent import run_agent

    # In your WebSocket handler:
    await run_agent(websocket, call_id, from_number, to_number)

Configuration (via environment variables):
    XAI_API_KEY: xAI API key (required)
    GROK_MODEL: Model name (default: grok-3-fast-voice)
    GROK_VOICE: Voice name (default: Sal)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from scipy import signal as scipy_signal
from silero_vad import load_silero_vad

if TYPE_CHECKING:
    from fastapi import WebSocket

load_dotenv()

# =============================================================================
# Configuration
# =============================================================================

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
# System Prompt
# =============================================================================


def _load_system_prompt() -> str:
    return (Path(__file__).parent / "system_prompt.md").read_text().strip()


DEFAULT_SYSTEM_PROMPT = _load_system_prompt()
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

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
# Tool Functions
# =============================================================================


async def check_order_status(order_number: str | None, email: str | None) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    logger.info(f"Checking order: number={order_number}, email={email}")

    if not order_number and not email:
        return {"status": "error", "message": "Need order number or email"}

    statuses = [
        {
            "status": "shipped",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "shipping_carrier": "FedEx",
            "tracking_number": f"FX{random.randint(1000000000, 9999999999)}",
            "estimated_delivery": (datetime.now() + timedelta(days=2)).strftime("%B %d"),
            "items": "TechFlow Pro Annual Subscription",
        },
        {
            "status": "processing",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "message": "Order is being prepared and will ship within 24 hours",
            "items": "TechFlow Teams License (5 seats)",
        },
        {
            "status": "delivered",
            "order_number": order_number or f"TF-{random.randint(100000, 999999)}",
            "delivered_date": (datetime.now() - timedelta(days=1)).strftime("%B %d"),
            "signed_by": "Front Desk",
            "items": "TechFlow Enterprise Setup Kit",
        },
    ]
    return random.choice(statuses)


async def send_sms(phone_number: str, message: str) -> dict[str, Any]:
    """Send SMS to customer. Replace with your actual implementation."""
    logger.info(f"Sending SMS to {phone_number}: {message[:50]}...")

    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "sent",
        "phone_number": phone_number,
        "message_preview": message[:50] + "..." if len(message) > 50 else message,
        "confirmation_id": f"SMS{random.randint(100000, 999999)}",
    }


async def schedule_callback(
    phone_number: str, reason: str, preferred_time: str, department: str
) -> dict[str, Any]:
    """Schedule a callback. Replace with your actual implementation."""
    logger.info(f"Scheduling callback: {phone_number}, {department}")

    if not phone_number:
        return {"status": "error", "message": "Phone number required"}

    return {
        "status": "scheduled",
        "callback_id": f"CB{random.randint(100000, 999999)}",
        "phone_number": phone_number,
        "department": department,
        "scheduled_time": preferred_time or "within 2 business hours",
        "reason": reason,
    }


async def transfer_call(department: str, reason: str) -> dict[str, Any]:
    """Transfer call to human agent. Replace with your actual implementation."""
    logger.info(f"Transferring to {department}: {reason}")

    return {
        "status": "transferring",
        "department": department,
        "reason": reason,
        "estimated_wait": "less than 2 minutes",
    }


# =============================================================================
# Silero VAD Processor
# =============================================================================


class SileroVADProcessor:
    """Processes audio frames through Silero VAD for speech detection.

    Accumulates audio in a buffer and runs VAD when enough samples are available.
    Tracks speech state transitions (silence → speaking → silence) to determine
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


# =============================================================================
# Grok Voice Bot
# =============================================================================


class GrokVoiceBot:
    """Manages a voice conversation session between Plivo and xAI Grok Realtime API."""

    def __init__(
        self,
        websocket: WebSocket,
        call_id: str,
        from_number: str = "",
        to_number: str = "",
        system_prompt: str | None = None,
        initial_message: str = "Hello, I'm calling for help.",
    ):
        self.websocket = websocket
        self.call_id = call_id
        self.from_number = from_number
        self.to_number = to_number
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.initial_message = initial_message

        self._running = False
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._vad = SileroVADProcessor()
        self._grok_ws = None
        self._is_responding = False  # Track if Grok is currently generating a response

    def _build_tools(self) -> list[dict[str, Any]]:
        """Build tool definitions for Grok function calling."""
        return [
            {
                "type": "function",
                "name": "check_order_status",
                "description": "Look up the status of a customer's order.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_number": {
                            "type": "string",
                            "description": "Order number (usually starts with TF-)",
                        },
                        "email": {
                            "type": "string",
                            "description": "Customer's email if order number unavailable",
                        },
                    },
                },
            },
            {
                "type": "function",
                "name": "send_sms",
                "description": "Send a text message to the customer's phone.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string", "description": "Phone number"},
                        "message": {"type": "string", "description": "Message content"},
                    },
                    "required": ["phone_number", "message"],
                },
            },
            {
                "type": "function",
                "name": "schedule_callback",
                "description": "Schedule a callback from a specialist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phone_number": {"type": "string", "description": "Phone number"},
                        "reason": {"type": "string", "description": "Why callback is needed"},
                        "preferred_time": {"type": "string", "description": "Preferred time"},
                        "department": {"type": "string", "description": "Department"},
                    },
                    "required": ["phone_number", "reason", "department"],
                },
            },
            {
                "type": "function",
                "name": "transfer_call",
                "description": "Transfer call to human agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "department": {"type": "string", "description": "Department"},
                        "reason": {"type": "string", "description": "Transfer reason"},
                    },
                    "required": ["department", "reason"],
                },
            },
            {
                "type": "function",
                "name": "end_call",
                "description": "End the call gracefully.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Reason for ending"},
                        "resolution": {
                            "type": "string",
                            "description": "How issue was resolved",
                        },
                    },
                },
            },
        ]

    async def _handle_function_call(self, name: str, call_id: str, arguments: str) -> None:
        """Execute a function call and send the result back to Grok."""
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            args = {}

        logger.info(f"Function call: {name} with args: {args}")

        try:
            if name == "check_order_status":
                result = await check_order_status(
                    order_number=args.get("order_number"),
                    email=args.get("email"),
                )
            elif name == "send_sms":
                result = await send_sms(
                    phone_number=args.get("phone_number", ""),
                    message=args.get("message", ""),
                )
            elif name == "schedule_callback":
                result = await schedule_callback(
                    phone_number=args.get("phone_number", ""),
                    reason=args.get("reason", ""),
                    preferred_time=args.get("preferred_time", ""),
                    department=args.get("department", "general"),
                )
            elif name == "transfer_call":
                result = await transfer_call(
                    department=args.get("department", "support"),
                    reason=args.get("reason", "Customer requested transfer"),
                )
            elif name == "end_call":
                logger.info(f"Ending call: {args.get('reason')}")
                self._running = False
                result = {"status": "call_ending", "reason": args.get("reason", "")}
            else:
                result = {"error": f"Unknown function: {name}"}

        except Exception as e:
            logger.error(f"Error in function {name}: {e}")
            result = {"error": str(e)}

        # Send function result back to Grok
        await self._grok_send(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(result),
                },
            }
        )
        # Request the model to continue with a response
        await self._grok_send(
            {
                "type": "response.create",
                "response": {"modalities": ["text", "audio"]},
            }
        )

    async def _grok_send(self, message: dict[str, Any]) -> None:
        """Send a JSON message to the Grok WebSocket."""
        if self._grok_ws:
            await self._grok_ws.send(json.dumps(message))

    def _build_session_config(self) -> dict[str, Any]:
        """Build Grok session configuration."""
        system_prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return {
            "type": "session.update",
            "session": {
                "instructions": system_prompt,
                "voice": GROK_VOICE,
                "turn_detection": None,  # Disable server VAD; using Silero VAD
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "tools": self._build_tools(),
            },
        }

    async def run(self) -> None:
        """Run the voice bot session."""
        import websockets

        logger.info(f"Starting Grok bot session for call {self.call_id}")
        self._running = True

        ws_url = f"{XAI_REALTIME_URL}?model={GROK_MODEL}"
        headers = {
            "Authorization": f"Bearer {XAI_API_KEY}",
        }

        try:
            async with websockets.connect(
                ws_url,
                additional_headers=headers,
                max_size=None,
            ) as grok_ws:
                self._grok_ws = grok_ws
                logger.info("Connected to xAI Grok Realtime API")

                # Configure the session
                await self._grok_send(self._build_session_config())

                # Wait for session confirmation
                while True:
                    msg = json.loads(await grok_ws.recv())
                    if msg.get("type") == "session.updated":
                        logger.info("Grok session configured")
                        break
                    elif msg.get("type") == "error":
                        logger.error(f"Grok session error: {msg}")
                        return

                # Send initial message to trigger greeting
                await self._grok_send(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": self.initial_message}],
                        },
                    }
                )
                await self._grok_send(
                    {
                        "type": "response.create",
                        "response": {"modalities": ["text", "audio"]},
                    }
                )

                await self._run_streaming_tasks(grok_ws)

        except Exception as e:
            logger.error(f"Bot session error: {e}")
        finally:
            self._running = False
            self._grok_ws = None
            logger.info(f"Bot session ended for call {self.call_id}")

    async def _run_streaming_tasks(self, grok_ws) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
            asyncio.create_task(self._receive_from_grok(grok_ws), name="grok_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} failed: {task.exception()}")
        finally:
            self._running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    async def _receive_from_plivo(self) -> None:
        """Receive audio from Plivo, run VAD, and forward to Grok."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)

                        # Convert to PCM 24kHz and send to Grok input buffer
                        pcm_24k = plivo_to_grok(mulaw_audio)
                        audio_b64 = base64.b64encode(pcm_24k).decode("utf-8")
                        await self._grok_send(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }
                        )

                        # Run Silero VAD on the audio
                        vad_audio = plivo_to_vad(mulaw_audio)
                        speech_started, speech_ended = self._vad.process(vad_audio)

                        if speech_started and self._is_responding:
                            # Barge-in: user started speaking while agent responds
                            logger.info("Barge-in detected, cancelling response")
                            await self._grok_send({"type": "response.cancel"})
                            self._is_responding = False
                            # Clear the Plivo send queue
                            while not self._send_queue.empty():
                                try:
                                    self._send_queue.get_nowait()
                                except asyncio.QueueEmpty:
                                    break

                        if speech_ended:
                            # User finished speaking - commit and request response
                            logger.debug("VAD: committing audio buffer")
                            await self._grok_send({"type": "input_audio_buffer.commit"})
                            await self._grok_send(
                                {
                                    "type": "response.create",
                                    "response": {"modalities": ["text", "audio"]},
                                }
                            )
                            self._vad.reset()

                elif event == "text":  # For testing - inject text as user input
                    text = message.get("text", "")
                    if text:
                        logger.info(f"Injecting text: {text[:50]}...")
                        await self._grok_send(
                            {
                                "type": "conversation.item.create",
                                "item": {
                                    "type": "message",
                                    "role": "user",
                                    "content": [{"type": "input_text", "text": text}],
                                },
                            }
                        )
                        await self._grok_send(
                            {
                                "type": "response.create",
                                "response": {"modalities": ["text", "audio"]},
                            }
                        )

                elif event == "stop":
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_grok(self, grok_ws) -> None:
        """Receive events from Grok and queue audio for Plivo."""
        try:
            async for raw_message in grok_ws:
                if not self._running:
                    return

                message = json.loads(raw_message)
                event_type = message.get("type", "")

                if event_type == "response.output_audio.delta":
                    # Audio chunk from Grok
                    audio_b64 = message.get("delta", "")
                    if audio_b64:
                        pcm_24k = base64.b64decode(audio_b64)
                        plivo_audio = grok_to_plivo(pcm_24k)
                        await self._send_queue.put(plivo_audio)

                elif event_type == "response.created":
                    self._is_responding = True

                elif event_type == "response.done":
                    self._is_responding = False
                    logger.debug("Grok response complete")

                elif event_type == "response.function_call_arguments.done":
                    # Function call from Grok
                    await self._handle_function_call(
                        name=message.get("name", ""),
                        call_id=message.get("call_id", ""),
                        arguments=message.get("arguments", ""),
                    )

                elif event_type == "response.output_audio_transcript.delta":
                    # Transcript for logging
                    transcript = message.get("delta", "")
                    if transcript:
                        logger.debug(f"Agent: {transcript}")

                elif event_type == "error":
                    logger.error(f"Grok error: {message.get('error', message)}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if "close" not in str(e).lower():
                logger.error(f"Grok receiver error: {e}")

    async def _send_to_plivo(self) -> None:
        """Send queued audio to Plivo WebSocket in 20ms chunks."""
        PLIVO_CHUNK_SIZE = 160  # 20ms at 8kHz μ-law
        audio_buffer = bytearray()

        try:
            while self._running:
                try:
                    audio = await asyncio.wait_for(self._send_queue.get(), timeout=0.1)
                    audio_buffer.extend(audio)

                    while len(audio_buffer) >= PLIVO_CHUNK_SIZE:
                        chunk = bytes(audio_buffer[:PLIVO_CHUNK_SIZE])
                        audio_buffer = audio_buffer[PLIVO_CHUNK_SIZE:]

                        message = {
                            "event": "playAudio",
                            "media": {
                                "contentType": "audio/x-mulaw",
                                "sampleRate": 8000,
                                "payload": base64.b64encode(chunk).decode("utf-8"),
                            },
                        }
                        await self.websocket.send_text(json.dumps(message))

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
    initial_message: str = "Hello, I'm calling for help.",
) -> None:
    """Run a voice agent session for an incoming call.

    Args:
        websocket: FastAPI WebSocket connection from Plivo
        call_id: Unique identifier for this call
        from_number: Caller's phone number
        to_number: Called phone number
        system_prompt: Custom system prompt (default: DEFAULT_SYSTEM_PROMPT)
        initial_message: Message to trigger agent greeting
    """
    agent = GrokVoiceBot(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
    )
    await agent.run()
