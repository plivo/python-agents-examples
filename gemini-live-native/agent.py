"""
Voice agent using Google Gemini Live API for speech-to-speech conversations.

This module provides a production-ready voice agent that:
- Connects to Google's Gemini Live API for real-time speech processing
- Handles bidirectional audio streaming with Plivo telephony
- Supports function calling for actions during conversations
- Manages audio format conversion between Plivo (μ-law 8kHz) and Gemini (PCM 16kHz/24kHz)

Usage:
    from agent import run_agent

    # In your WebSocket handler:
    await run_agent(websocket, call_id, from_number, to_number)

Configuration (via environment variables):
    GEMINI_API_KEY: Google AI API key (required)
    GEMINI_MODEL: Model name (default: gemini-2.5-flash-native-audio-preview-12-2025)
    GEMINI_VOICE: Voice name (default: Kore)
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import random
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
GEMINI_VOICE = os.getenv("GEMINI_VOICE", "Kore")

# Audio format constants
PLIVO_SAMPLE_RATE = 8000  # Plivo uses 8kHz μ-law
GEMINI_INPUT_RATE = 16000  # Gemini expects 16kHz PCM input
GEMINI_OUTPUT_RATE = 24000  # Gemini outputs 24kHz PCM
AUDIO_CHUNK_SIZE = 1024  # Bytes per chunk sent to Gemini

# =============================================================================
# System Prompt
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """
You are Alex, a friendly and professional customer service agent for TechFlow,
a software company that provides cloud-based productivity tools.

## Your Personality
- Warm, patient, and empathetic
- Professional but conversational - you're talking to a real person
- You use natural speech patterns with occasional filler words
- You never sound robotic or overly formal

## Audio Output Rules
- Your responses will be converted to speech, so never use special characters
- Spell out numbers naturally: say "twenty three dollars" not "$23"
- Keep responses concise - aim for 1-3 sentences unless explaining something complex
- Use natural pauses by breaking up longer responses

## Your Capabilities
You can help customers with:
1. Checking order status - ask for their order number or email
2. Product information about TechFlow Pro, Teams, and Enterprise plans
3. Billing questions and payment issues
4. Technical support for basic issues
5. Scheduling callbacks for complex issues
6. Sending confirmation texts to their phone

## Product Knowledge
- TechFlow Pro: twelve dollars per month, for individuals, 100GB storage
- TechFlow Teams: twenty five dollars per user per month, up to 25 people, 500GB
- TechFlow Enterprise: Custom pricing, unlimited users and storage, dedicated support

## When to Use Your Tools
- check_order_status: when customer asks about an order, delivery, or purchase
- send_sms: when customer needs information texted to them
- schedule_callback: for complex issues or when customer wants a specialist
- transfer_call: when customer asks for a human or you cannot help
- end_call: when conversation is complete and customer says goodbye

## Conversation Flow
1. Greet the caller warmly and ask how you can help
2. Listen and acknowledge their concern before jumping to solutions
3. Ask clarifying questions if needed
4. Provide clear, helpful responses
5. Confirm the customer is satisfied before ending
6. End with a friendly closing

## Handling Difficult Situations
- If frustrated, acknowledge their feelings first
- Never argue or get defensive
- Be honest if you cannot help and offer alternatives
- If unsure, say so honestly rather than guessing

## Important Guidelines
- Never make up order information - always use check_order_status
- Ask for phone number before sending SMS if not available
- Keep the conversation moving naturally
- Ask if there is anything else before ending
""".strip()

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# =============================================================================
# Audio Conversion Utilities (public API for use in tests)
# =============================================================================

# μ-law decoding table (ITU-T G.711)
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


def plivo_to_gemini(mulaw_8k: bytes) -> bytes:
    """Convert Plivo audio (μ-law 8kHz) to Gemini format (PCM16 16kHz)."""
    pcm_8k = ulaw_to_pcm(mulaw_8k)
    return resample_audio(pcm_8k, PLIVO_SAMPLE_RATE, GEMINI_INPUT_RATE)


def gemini_to_plivo(pcm_24k: bytes) -> bytes:
    """Convert Gemini audio (PCM16 24kHz) to Plivo format (μ-law 8kHz)."""
    pcm_8k = resample_audio(pcm_24k, GEMINI_OUTPUT_RATE, PLIVO_SAMPLE_RATE)
    return pcm_to_ulaw(pcm_8k)


# =============================================================================
# Tool Functions (simple functions instead of abstract class)
# =============================================================================


async def check_order_status(order_number: str | None, email: str | None) -> dict[str, Any]:
    """Look up order status. Replace with your actual implementation."""
    logger.info(f"Checking order: number={order_number}, email={email}")

    if not order_number and not email:
        return {"status": "error", "message": "Need order number or email"}

    # Mock implementation - replace with actual order system integration
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

    # Mock implementation - replace with actual SMS integration (e.g., Plivo SMS)
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

    # Mock implementation - replace with actual scheduling system
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

    # Mock implementation - replace with actual call transfer logic
    return {
        "status": "transferring",
        "department": department,
        "reason": reason,
        "estimated_wait": "less than 2 minutes",
    }


# =============================================================================
# Gemini Voice Bot
# =============================================================================


class GeminiVoiceBot:
    """Manages a voice conversation session between Plivo and Gemini Live API."""

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
        self._audio_buffer = bytearray()
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._tools = self._build_tools()

    def _build_tools(self) -> list[types.Tool]:
        """Build Gemini function calling tools."""
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="check_order_status",
                        description="Look up the status of a customer's order.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "order_number": types.Schema(
                                    type=types.Type.STRING,
                                    description="Order number (usually starts with TF-)",
                                ),
                                "email": types.Schema(
                                    type=types.Type.STRING,
                                    description="Customer's email if order number unavailable",
                                ),
                            },
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="send_sms",
                        description="Send a text message to the customer's phone.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "phone_number": types.Schema(
                                    type=types.Type.STRING, description="Phone number"
                                ),
                                "message": types.Schema(
                                    type=types.Type.STRING, description="Message content"
                                ),
                            },
                            required=["phone_number", "message"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="schedule_callback",
                        description="Schedule a callback from a specialist.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "phone_number": types.Schema(
                                    type=types.Type.STRING, description="Phone number"
                                ),
                                "reason": types.Schema(
                                    type=types.Type.STRING, description="Why callback is needed"
                                ),
                                "preferred_time": types.Schema(
                                    type=types.Type.STRING, description="Preferred time"
                                ),
                                "department": types.Schema(
                                    type=types.Type.STRING, description="Department"
                                ),
                            },
                            required=["phone_number", "reason", "department"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="transfer_call",
                        description="Transfer call to human agent.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "department": types.Schema(
                                    type=types.Type.STRING, description="Department"
                                ),
                                "reason": types.Schema(
                                    type=types.Type.STRING, description="Transfer reason"
                                ),
                            },
                            required=["department", "reason"],
                        ),
                    ),
                    types.FunctionDeclaration(
                        name="end_call",
                        description="End the call gracefully.",
                        parameters=types.Schema(
                            type=types.Type.OBJECT,
                            properties={
                                "reason": types.Schema(
                                    type=types.Type.STRING, description="Reason for ending"
                                ),
                                "resolution": types.Schema(
                                    type=types.Type.STRING, description="How issue was resolved"
                                ),
                            },
                        ),
                    ),
                ]
            )
        ]

    async def _handle_function_call(self, function_call: types.FunctionCall) -> str:
        """Route function calls to appropriate handlers."""
        name = function_call.name
        args = function_call.args or {}
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

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error in function {name}: {e}")
            return json.dumps({"error": str(e)})

    def _build_session_config(self) -> types.LiveConnectConfig:
        """Build Gemini Live session configuration."""
        system_prompt = self.system_prompt

        if self.from_number:
            call_time = datetime.now().strftime("%I:%M %p on %A, %B %d")
            system_prompt += f"""

## Current Call Context
- Caller's phone number: {self.from_number}
- Call ID: {self.call_id}
- Time: {call_time}

You can use the caller's phone number for SMS or callbacks without asking."""

        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=GEMINI_VOICE)
                )
            ),
            system_instruction=types.Content(parts=[types.Part(text=system_prompt)]),
            tools=self._tools,
        )

    async def run(self) -> None:
        """Run the voice bot session."""
        logger.info(f"Starting bot session for call {self.call_id}")
        self._running = True

        try:
            config = self._build_session_config()

            async with self._client.aio.live.connect(
                model=GEMINI_MODEL, config=config
            ) as session:
                logger.info("Connected to Gemini Live API")

                await session.send_client_content(
                    turns=types.Content(
                        role="user", parts=[types.Part(text=self.initial_message)]
                    ),
                    turn_complete=True,
                )

                await self._run_streaming_tasks(session)

        except Exception as e:
            logger.error(f"Bot session error: {e}")
        finally:
            self._running = False
            logger.info(f"Bot session ended for call {self.call_id}")

    async def _run_streaming_tasks(self, session) -> None:
        """Run the concurrent streaming tasks."""
        tasks = [
            asyncio.create_task(self._receive_from_plivo(session), name="plivo_rx"),
            asyncio.create_task(self._receive_from_gemini(session), name="gemini_rx"),
            asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
        ]

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
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

    async def _receive_from_plivo(self, session) -> None:
        """Receive audio from Plivo WebSocket and forward to Gemini."""
        try:
            while self._running:
                data = await self.websocket.receive_text()
                message = json.loads(data)
                event = message.get("event")

                if event == "media":
                    payload = message.get("media", {}).get("payload", "")
                    if payload:
                        mulaw_audio = base64.b64decode(payload)
                        pcm_audio = plivo_to_gemini(mulaw_audio)

                        self._audio_buffer.extend(pcm_audio)
                        if len(self._audio_buffer) >= AUDIO_CHUNK_SIZE:
                            chunk = bytes(self._audio_buffer[:AUDIO_CHUNK_SIZE])
                            self._audio_buffer = self._audio_buffer[AUDIO_CHUNK_SIZE:]
                            await session.send_realtime_input(
                                audio=types.Blob(data=chunk, mime_type="audio/pcm")
                            )

                elif event == "text":  # For testing - inject text as user input
                    text = message.get("text", "")
                    if text:
                        logger.info(f"Injecting text: {text[:50]}...")
                        await session.send_client_content(
                            turns=types.Content(role="user", parts=[types.Part(text=text)]),
                            turn_complete=True,
                        )

                elif event == "stop":
                    break

        except Exception as e:
            if "1000" not in str(e):
                logger.error(f"Plivo receiver error: {e}")

    async def _receive_from_gemini(self, session) -> None:
        """Receive audio from Gemini and queue for Plivo."""
        try:
            while self._running:
                try:
                    async for response in session.receive():
                        if not self._running:
                            return

                        if response.server_content:
                            model_turn = response.server_content.model_turn
                            if model_turn and model_turn.parts:
                                for part in model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        plivo_audio = gemini_to_plivo(part.inline_data.data)
                                        await self._send_queue.put(plivo_audio)

                        if response.tool_call:
                            for fc in response.tool_call.function_calls:
                                result = await self._handle_function_call(fc)
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            id=fc.id,
                                            name=fc.name,
                                            response={"result": result},
                                        )
                                    ]
                                )

                except Exception as e:
                    if "cancelled" in str(e).lower():
                        raise
                    logger.error(f"Gemini receive error: {e}")
                    raise

        except asyncio.CancelledError:
            pass

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
    agent = GeminiVoiceBot(
        websocket=websocket,
        call_id=call_id,
        from_number=from_number,
        to_number=to_number,
        system_prompt=system_prompt,
        initial_message=initial_message,
    )
    await agent.run()
