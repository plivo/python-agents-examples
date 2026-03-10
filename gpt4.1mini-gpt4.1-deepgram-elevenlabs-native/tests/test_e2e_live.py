"""E2E live tests with real APIs — measures key voice AI metrics.

Tests each component individually and the full pipeline end-to-end,
reporting latency and performance numbers.

Run: uv run pytest tests/test_e2e_live.py -v -s
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import struct
import time

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CONVERSATION_MODEL = os.getenv("OPENAI_CONVERSATION_MODEL", "gpt-4.1-mini")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_flash_v2_5")
HUBSPOT_ACCESS_TOKEN = os.getenv("HUBSPOT_ACCESS_TOKEN", "")
CAL_COM_API_KEY = os.getenv("CAL_COM_API_KEY", "")
CAL_COM_EVENT_TYPE_ID = os.getenv("CAL_COM_EVENT_TYPE_ID", "")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# Simulated tool latency ranges (ms) for mock tests
TOOL_LATENCY_RANGES = {
    "hubspot_lookup": (200, 800),
    "hubspot_create": (300, 1000),
    "hubspot_deal": (400, 1200),
    "calcom_booking": (500, 1500),
    "plivo_sms": (200, 600),
    "slack_webhook": (100, 400),
}


async def simulate_tool_latency(tool_name: str) -> float:
    """Sleep for a random duration simulating real tool API latency. Returns ms slept."""
    lo, hi = TOOL_LATENCY_RANGES.get(tool_name, (200, 800))
    delay_ms = random.uniform(lo, hi)
    await asyncio.sleep(delay_ms / 1000)
    return delay_ms


def skip_if_no_key(key_val: str, name: str):
    if not key_val:
        pytest.skip(f"{name} not configured")


# =============================================================================
# 1. DEEPGRAM STT LATENCY
# =============================================================================


class TestDeepgramSTT:
    """Measure Deepgram nova-3 streaming STT latency."""

    @pytest.mark.asyncio
    async def test_stt_connection_and_transcript_latency(self):
        """Connect to Deepgram, send audio, measure time to first transcript."""
        skip_if_no_key(DEEPGRAM_API_KEY, "DEEPGRAM_API_KEY")
        import websockets

        url = (
            "wss://api.deepgram.com/v1/listen"
            "?model=nova-3&encoding=linear16&sample_rate=8000"
            "&channels=1&interim_results=true&endpointing=false"
        )
        headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}"}

        # Generate 1 second of 440Hz tone as PCM16 8kHz
        samples = []
        for i in range(8000):
            sample = int(16000 * __import__("math").sin(2 * 3.14159 * 440 * i / 8000))
            samples.append(sample)
        pcm_audio = struct.pack(f"{len(samples)}h", *samples)

        t_connect_start = time.monotonic()
        async with websockets.connect(url, additional_headers=headers) as ws:
            t_connected = time.monotonic()
            connect_ms = (t_connected - t_connect_start) * 1000

            # Send audio in 20ms chunks (160 samples = 320 bytes)
            chunk_size = 320
            t_first_send = time.monotonic()
            for i in range(0, len(pcm_audio), chunk_size):
                await ws.send(pcm_audio[i : i + chunk_size])
                await asyncio.sleep(0.005)  # pace slightly

            # Wait for any transcript
            t_first_transcript = None
            transcript_text = ""
            try:
                async with asyncio.timeout(5):
                    while True:
                        msg = json.loads(await ws.recv())
                        if msg.get("type") == "Results":
                            channel = msg.get("channel", {})
                            alts = channel.get("alternatives", [{}])
                            if alts and alts[0].get("transcript"):
                                t_first_transcript = time.monotonic()
                                transcript_text = alts[0]["transcript"]
                                break
            except (asyncio.TimeoutError, Exception):
                pass

            # Close
            await ws.send(json.dumps({"type": "CloseStream"}))

        print("\n--- Deepgram STT Metrics ---")
        print(f"  WebSocket connect:     {connect_ms:6.0f} ms")
        if t_first_transcript:
            stt_latency = (t_first_transcript - t_first_send) * 1000
            print(f"  Time to 1st transcript:{stt_latency:6.0f} ms")
            print(f"  Transcript:            '{transcript_text}'")
        else:
            stt_latency = None
            print("  No transcript received (tone may not produce text — this is OK)")

        assert connect_ms < 3000, f"Deepgram connect too slow: {connect_ms}ms"


# =============================================================================
# 2. ELEVENLABS TTS LATENCY
# =============================================================================


class TestElevenLabsTTS:
    """Measure ElevenLabs streaming TTS latency."""

    @pytest.mark.asyncio
    async def test_tts_ttfb_and_throughput(self):
        """Measure time-to-first-byte and total synthesis time."""
        skip_if_no_key(ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY")

        text = "Hi there, thanks for calling TechFlow Solutions. How can I help you today?"
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech"
            f"/{ELEVENLABS_VOICE_ID}/stream"
            f"?output_format=pcm_24000"
        )

        t0 = time.monotonic()
        total_bytes = 0
        first_chunk_time = None
        chunk_count = 0

        async with httpx.AsyncClient(timeout=30.0) as client, client.stream(
            "POST",
            url,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={"text": text, "model_id": ELEVENLABS_MODEL_ID},
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes(chunk_size=4800):
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                total_bytes += len(chunk)
                chunk_count += 1

        t_end = time.monotonic()
        ttfb = (first_chunk_time - t0) * 1000 if first_chunk_time else 0
        total_ms = (t_end - t0) * 1000
        audio_duration = total_bytes / (24000 * 2)  # 24kHz 16-bit mono

        print("\n--- ElevenLabs TTS Metrics ---")
        print(f"  Text length:           {len(text):6d} chars")
        print(f"  TTFB:                  {ttfb:6.0f} ms")
        print(f"  Total synthesis:       {total_ms:6.0f} ms")
        print(f"  Audio duration:        {audio_duration:6.1f} s")
        print(f"  Chunks:                {chunk_count:6d}")
        print(f"  Total bytes:           {total_bytes:6d}")
        print(f"  Realtime factor:       {audio_duration / (total_ms / 1000):6.1f}x")

        assert ttfb < 2000, f"TTS TTFB too high: {ttfb}ms"
        assert total_bytes > 0, "No audio produced"


# =============================================================================
# 3. OPENAI LLM LATENCY — MINI (CONVERSATION MODEL)
# =============================================================================


class TestOpenAIMini:
    """Measure GPT-4.1 mini latency for conversational and delegation turns."""

    @pytest.mark.asyncio
    async def test_mini_conversational_latency(self):
        """Measure latency for a pure conversational response (no delegation)."""
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")

        messages = [
            {
                "role": "system",
                "content": "You are a friendly sales agent. Keep responses to 1-2 sentences.",
            },
            {"role": "user", "content": "Hi, I'm interested in your workflow automation tools."},
        ]

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": messages,
                    "max_tokens": 150,
                },
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {})

        print("\n--- GPT-4.1 Mini (Conversational) ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Prompt tokens:         {tokens.get('prompt_tokens', '?'):>6}")
        print(f"  Completion tokens:     {tokens.get('completion_tokens', '?'):>6}")
        print(f"  Response:              '{text[:80]}'")

        assert latency < 5000, f"Mini too slow: {latency}ms"

    @pytest.mark.asyncio
    async def test_mini_delegation_latency(self):
        """Measure latency when mini decides to delegate (tool call)."""
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")

        routing_tool = [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_reasoning",
                    "description": (
                        "Delegate a task to the reasoning system for CRM lookups, "
                        "data updates, lead scoring, meeting booking, SMS, or ending the call."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "Task description"},
                            "spoken_filler": {
                                "type": "string",
                                "description": "Short phrase to say while processing",
                            },
                        },
                        "required": ["task", "spoken_filler"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a sales agent. When the caller provides contact info, "
                    "delegate to reasoning to look it up. Always include a spoken_filler."
                ),
            },
            {"role": "user", "content": "Can you look me up? My email is john@acme.com."},
        ]

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": messages,
                    "max_tokens": 300,
                    "tools": routing_tool,
                },
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        message = data["choices"][0]["message"]
        tokens = data.get("usage", {})

        tool_calls = message.get("tool_calls", [])
        delegated = len(tool_calls) > 0 and tool_calls[0]["function"]["name"] == (
            "delegate_to_reasoning"
        )

        filler = ""
        task = ""
        if delegated:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            filler = args.get("spoken_filler", "")
            task = args.get("task", "")

        print("\n--- GPT-4.1 Mini (Delegation Decision) ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Delegated:             {delegated}")
        print(f"  Prompt tokens:         {tokens.get('prompt_tokens', '?'):>6}")
        print(f"  Completion tokens:     {tokens.get('completion_tokens', '?'):>6}")
        print(f"  Filler:                '{filler[:80]}'")
        print(f"  Task:                  '{task[:80]}'")

        assert delegated, "Mini should have delegated for a CRM lookup request"
        assert filler, "Mini should have provided a spoken_filler"
        assert latency < 5000, f"Mini delegation too slow: {latency}ms"


# =============================================================================
# 4. OPENAI LLM LATENCY — GPT-4.1 (REASONING MODEL)
# =============================================================================


class TestOpenAIReasoning:
    """Measure GPT-4.1 reasoning model latency."""

    @pytest.mark.asyncio
    async def test_reasoning_tool_call_latency(self):
        """Measure latency for reasoning model to decide on tool calls."""
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_contact",
                    "description": "Search for a contact in HubSpot CRM by email or phone.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_or_phone": {"type": "string", "description": "Email or phone"},
                        },
                        "required": ["email_or_phone"],
                    },
                },
            },
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning system. Use tools to complete the task. "
                    "Return JSON results when done.\n\n"
                    "Delegated task: Look up contact john@acme.com in CRM"
                ),
            },
            {"role": "user", "content": "Can you look me up? My email is john@acme.com."},
        ]

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_REASONING_MODEL,
                    "messages": messages,
                    "max_tokens": 300,
                    "tools": tools,
                },
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        message = data["choices"][0]["message"]
        tokens = data.get("usage", {})

        tool_calls = message.get("tool_calls", [])
        called_tool = tool_calls[0]["function"]["name"] if tool_calls else "none"

        print("\n--- GPT-4.1 (Reasoning — Tool Decision) ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Tool called:           {called_tool}")
        print(f"  Prompt tokens:         {tokens.get('prompt_tokens', '?'):>6}")
        print(f"  Completion tokens:     {tokens.get('completion_tokens', '?'):>6}")

        assert called_tool == "lookup_contact", f"Expected lookup_contact, got {called_tool}"
        assert latency < 10000, f"Reasoning too slow: {latency}ms"

    @pytest.mark.asyncio
    async def test_reasoning_json_response_latency(self):
        """Measure latency for reasoning model to produce structured JSON after tool results."""
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a reasoning system. Respond with ONLY valid JSON.\n"
                    'Include "status", "actions_taken", "data", "summary".\n\n'
                    "Delegated task: Look up contact john@acme.com"
                ),
            },
            {"role": "user", "content": "Look up my account, email john@acme.com"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_contact",
                        "arguments": '{"email_or_phone": "john@acme.com"}',
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": json.dumps({
                    "status": "found",
                    "contact_id": "123",
                    "firstname": "John",
                    "lastname": "Smith",
                    "company": "Acme Corp",
                    "email": "john@acme.com",
                }),
            },
        ]

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_REASONING_MODEL,
                    "messages": messages,
                    "max_tokens": 500,
                },
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {})

        is_json = False
        try:
            json.loads(content)
            is_json = True
        except json.JSONDecodeError:
            pass

        print("\n--- GPT-4.1 (Reasoning — JSON Response) ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Valid JSON:            {is_json}")
        print(f"  Prompt tokens:         {tokens.get('prompt_tokens', '?'):>6}")
        print(f"  Completion tokens:     {tokens.get('completion_tokens', '?'):>6}")
        print(f"  Response:              '{content[:120]}'")

        assert latency < 10000, f"Reasoning JSON too slow: {latency}ms"


# =============================================================================
# 5. MINI FOLLOW-UP LATENCY (crafting spoken response from JSON)
# =============================================================================


class TestMiniFollowup:
    """Measure mini's latency to craft a spoken response from reasoning results."""

    @pytest.mark.asyncio
    async def test_mini_followup_from_json(self):
        """Measure time for mini to craft a natural response from tool results."""
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")

        reasoning_result = {
            "status": "success",
            "actions_taken": ["lookup_contact"],
            "data": {
                "contact_id": "123",
                "firstname": "John",
                "lastname": "Smith",
                "company": "Acme Corp",
                "email": "john@acme.com",
                "existing_deal": "$25,000 in pipeline",
            },
            "summary": "Found existing contact with open deal",
        }

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly sales agent for TechFlow Solutions. "
                    "Keep responses to 1-2 sentences."
                ),
            },
            {"role": "user", "content": "Can you look me up? My email is john@acme.com."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "reasoning_result",
                    "type": "function",
                    "function": {
                        "name": "reasoning_result",
                        "arguments": json.dumps(reasoning_result),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "reasoning_result",
                "content": json.dumps(reasoning_result),
            },
            {
                "role": "system",
                "content": (
                    "The reasoning system completed the task. Results are above. "
                    "Craft a natural spoken response (1-3 sentences). "
                    "Do NOT mention tools or delegation."
                ),
            },
        ]

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": messages,
                    "max_tokens": 150,
                },
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {})

        print("\n--- GPT-4.1 Mini (Follow-up from JSON) ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Prompt tokens:         {tokens.get('prompt_tokens', '?'):>6}")
        print(f"  Completion tokens:     {tokens.get('completion_tokens', '?'):>6}")
        print(f"  Response:              '{text[:120]}'")

        assert latency < 5000, f"Mini follow-up too slow: {latency}ms"


# =============================================================================
# 6. FULL PIPELINE — SIMULATED DELEGATION TURN (no Plivo)
# =============================================================================


class TestFullPipeline:
    """Measure the full dual-LLM + TTS pipeline for a delegation turn."""

    @pytest.mark.asyncio
    async def test_full_delegation_turn_latency(self):
        """Simulate a complete delegation turn and measure all segments.

        Timeline:
        t0: User text arrives
        t1: Mini responds with delegation + filler   (mini_latency)
        t2: Filler TTS first byte                    (filler_ttfb)
        t3: Reasoning model starts (concurrent with filler TTS)
        t4: Reasoning done                           (reasoning_latency)
        t5: Mini follow-up from JSON                 (followup_latency)
        t6: Response TTS first byte                  (response_ttfb)
        """
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")
        skip_if_no_key(ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY")

        # — Step 1: Mini decides to delegate —
        routing_tool = [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_reasoning",
                    "description": "Delegate tasks requiring CRM, scheduling, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "spoken_filler": {"type": "string"},
                        },
                        "required": ["task", "spoken_filler"],
                    },
                },
            },
        ]
        messages_step1 = [
            {
                "role": "system",
                "content": (
                    "You are a sales agent. When CRM lookup is needed, delegate. "
                    "Include a spoken_filler."
                ),
            },
            {"role": "user", "content": "My email is sarah@bigcorp.io, can you look me up?"},
        ]

        t_start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r1 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": messages_step1,
                    "max_tokens": 300,
                    "tools": routing_tool,
                },
            )
            r1.raise_for_status()
        t_mini = time.monotonic()
        mini_latency = (t_mini - t_start) * 1000

        d1 = r1.json()
        msg1 = d1["choices"][0]["message"]
        tc = msg1.get("tool_calls", [{}])[0]
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        filler_text = args.get("spoken_filler", "One moment please.")
        task_desc = args.get("task", "Look up contact")

        # — Step 2: Filler TTS + Reasoning (concurrent) —
        filler_ttfb = None
        filler_bytes = 0

        async def stream_filler():
            nonlocal filler_ttfb, filler_bytes
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech"
                f"/{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000"
            )
            async with httpx.AsyncClient(timeout=30.0) as c, c.stream(
                "POST", url,
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={"text": filler_text, "model_id": ELEVENLABS_MODEL_ID},
            ) as resp:
                resp.raise_for_status()
                first = True
                async for chunk in resp.aiter_bytes(chunk_size=4800):
                    if first:
                        filler_ttfb = (time.monotonic() - t_mini) * 1000
                        first = False
                    filler_bytes += len(chunk)

        # Simulate reasoning with tool latency (mock CRM call)
        reasoning_latency = None
        sim_tool_latency = None

        async def run_reasoning():
            nonlocal reasoning_latency, sim_tool_latency
            t_r = time.monotonic()

            # Simulate HubSpot lookup latency
            sim_tool_latency = await simulate_tool_latency("hubspot_lookup")

            reasoning_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a reasoning system. Return ONLY valid JSON.\n"
                        f"Delegated task: {task_desc}"
                    ),
                },
                {
                    "role": "user",
                    "content": "My email is sarah@bigcorp.io, can you look me up?",
                },
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_sim",
                        "type": "function",
                        "function": {
                            "name": "lookup_contact",
                            "arguments": '{"email_or_phone": "sarah@bigcorp.io"}',
                        },
                    }],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_sim",
                    "content": json.dumps({
                        "status": "found",
                        "contact_id": "456",
                        "firstname": "Sarah",
                        "lastname": "Chen",
                        "company": "BigCorp",
                    }),
                },
            ]
            async with httpx.AsyncClient(timeout=30.0) as c:
                resp = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENAI_REASONING_MODEL,
                        "messages": reasoning_messages,
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
            reasoning_latency = (time.monotonic() - t_r) * 1000
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"status": "success", "summary": content}

        # Run filler + reasoning concurrently
        reasoning_result, _ = await asyncio.gather(
            run_reasoning(), stream_filler(),
        )

        # — Step 3: Mini follow-up —
        followup_messages = [
            {
                "role": "system",
                "content": "You are a friendly sales agent. Keep responses to 1-2 sentences.",
            },
            {
                "role": "user",
                "content": "My email is sarah@bigcorp.io, can you look me up?",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "rr",
                    "type": "function",
                    "function": {
                        "name": "reasoning_result",
                        "arguments": json.dumps(reasoning_result),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "rr",
                "content": json.dumps(reasoning_result),
            },
            {
                "role": "system",
                "content": (
                    "Reasoning completed. Craft a natural response (1-3 sentences). "
                    "Do NOT mention tools."
                ),
            },
        ]

        t_fu_start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r3 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": followup_messages,
                    "max_tokens": 150,
                },
            )
            r3.raise_for_status()
        t_fu_done = time.monotonic()
        followup_latency = (t_fu_done - t_fu_start) * 1000
        followup_text = r3.json()["choices"][0]["message"]["content"]

        # — Step 4: Response TTS TTFB —
        response_ttfb = None
        response_bytes = 0
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech"
            f"/{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000"
        )
        async with httpx.AsyncClient(timeout=30.0) as client, client.stream(
            "POST", url,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={"text": followup_text, "model_id": ELEVENLABS_MODEL_ID},
        ) as resp:
            resp.raise_for_status()
            first = True
            async for chunk in resp.aiter_bytes(chunk_size=4800):
                if first:
                    response_ttfb = (time.monotonic() - t_fu_done) * 1000
                    first = False
                response_bytes += len(chunk)

        t_end = time.monotonic()
        total_ms = (t_end - t_start) * 1000
        filler_audio_s = filler_bytes / (24000 * 2)
        response_audio_s = response_bytes / (24000 * 2)

        # — Report —
        print("\n" + "=" * 60)
        print("  FULL PIPELINE — DELEGATION TURN (SIMULATED TOOL LATENCY)")
        print("=" * 60)
        print(f"  1. Mini routing decision:    {mini_latency:6.0f} ms")
        print(f"  2. Filler TTS TTFB:          {filler_ttfb or 0:6.0f} ms (after mini)")
        print(f"     Filler audio duration:    {filler_audio_s:6.1f} s")
        print(f"  3. Reasoning (incl. sim tool):{reasoning_latency or 0:5.0f} ms (concurrent)")
        print(f"     ├─ Simulated tool call:   {sim_tool_latency or 0:6.0f} ms")
        print(f"     └─ LLM reasoning only:    "
              f"{(reasoning_latency or 0) - (sim_tool_latency or 0):6.0f} ms")
        print(f"  4. Mini follow-up:           {followup_latency:6.0f} ms")
        print(f"  5. Response TTS TTFB:        {response_ttfb or 0:6.0f} ms")
        print(f"     Response audio duration:  {response_audio_s:6.1f} s")
        print("  ─────────────────────────────────────────")
        print(f"  Total wall-clock:            {total_ms:6.0f} ms")
        print(f"  Time to filler audio:        {mini_latency + (filler_ttfb or 0):6.0f} ms")
        print(f"  Time to answer audio:        {total_ms - response_audio_s * 1000:6.0f} ms")
        print("  ─────────────────────────────────────────")
        print(f"  Filler:    '{filler_text[:60]}'")
        print(f"  Response:  '{followup_text[:80]}'")
        print("=" * 60)

        # Key assertions
        time_to_filler = mini_latency + (filler_ttfb or 0)
        assert time_to_filler < 3000, (
            f"Time to filler audio {time_to_filler}ms > 3s — caller hears silence too long"
        )


# =============================================================================
# 7. LIVE HUBSPOT API LATENCY
# =============================================================================


class TestHubSpotLive:
    """Measure real HubSpot API latencies."""

    @pytest.mark.asyncio
    async def test_hubspot_search_contact_latency(self):
        """Search for a contact by email — measures real CRM lookup time."""
        skip_if_no_key(HUBSPOT_ACCESS_TOKEN, "HUBSPOT_ACCESS_TOKEN")

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts/search",
                headers={
                    "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "filterGroups": [{
                        "filters": [{
                            "propertyName": "email",
                            "operator": "EQ",
                            "value": "rross@diocesan.com",
                        }],
                    }],
                    "properties": [
                        "firstname", "lastname", "email", "company", "phone",
                    ],
                    "limit": 1,
                },
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        results = data.get("results", [])
        found = len(results) > 0
        contact = results[0]["properties"] if found else {}

        print("\n--- HubSpot: Contact Search ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Found:                 {found}")
        if found:
            print(f"  Name:                  {contact.get('firstname')} {contact.get('lastname')}")
            print(f"  Email:                 {contact.get('email')}")
            print(f"  Company:               {contact.get('company')}")

        assert latency < 5000, f"HubSpot search too slow: {latency}ms"

    @pytest.mark.asyncio
    async def test_hubspot_list_contacts_latency(self):
        """List contacts — measures basic CRM read time."""
        skip_if_no_key(HUBSPOT_ACCESS_TOKEN, "HUBSPOT_ACCESS_TOKEN")

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.hubapi.com/crm/v3/objects/contacts?limit=5",
                headers={"Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}"},
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        count = len(data.get("results", []))

        print("\n--- HubSpot: List Contacts ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Contacts returned:     {count}")

        assert latency < 5000, f"HubSpot list too slow: {latency}ms"

    @pytest.mark.asyncio
    async def test_hubspot_list_deals_latency(self):
        """List deals — measures deal pipeline read time."""
        skip_if_no_key(HUBSPOT_ACCESS_TOKEN, "HUBSPOT_ACCESS_TOKEN")

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.hubapi.com/crm/v3/objects/deals?limit=5",
                headers={"Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}"},
            )
            response.raise_for_status()
        latency = (time.monotonic() - t0) * 1000
        data = response.json()
        count = len(data.get("results", []))

        print("\n--- HubSpot: List Deals ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Deals returned:        {count}")

        assert latency < 5000, f"HubSpot deals too slow: {latency}ms"


# =============================================================================
# 8. FULL PIPELINE — LIVE HUBSPOT DELEGATION TURN
# =============================================================================


class TestFullPipelineLiveHubSpot:
    """Full delegation pipeline with real HubSpot API call."""

    @pytest.mark.asyncio
    async def test_full_delegation_with_live_hubspot(self):
        """Complete delegation turn: mini → filler TTS → reasoning + real HubSpot → mini follow-up.

        This is the most realistic test — the reasoning model actually calls HubSpot
        via a real HTTP request, not a mocked tool result.
        """
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")
        skip_if_no_key(ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY")
        skip_if_no_key(HUBSPOT_ACCESS_TOKEN, "HUBSPOT_ACCESS_TOKEN")

        # — Step 1: Mini decides to delegate —
        routing_tool = [
            {
                "type": "function",
                "function": {
                    "name": "delegate_to_reasoning",
                    "description": "Delegate tasks requiring CRM, scheduling, etc.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string"},
                            "spoken_filler": {"type": "string"},
                        },
                        "required": ["task", "spoken_filler"],
                    },
                },
            },
        ]
        messages_step1 = [
            {
                "role": "system",
                "content": (
                    "You are a sales agent. When CRM lookup is needed, delegate. "
                    "Include a spoken_filler — 2-3 sentences to keep the caller engaged."
                ),
            },
            {
                "role": "user",
                "content": "Hi, I'm Ryan Ross from Diocesan. Can you pull up my account?",
            },
        ]

        t_start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r1 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": messages_step1,
                    "max_tokens": 300,
                    "tools": routing_tool,
                },
            )
            r1.raise_for_status()
        t_mini = time.monotonic()
        mini_latency = (t_mini - t_start) * 1000

        d1 = r1.json()
        msg1 = d1["choices"][0]["message"]
        tc = msg1.get("tool_calls", [{}])[0]
        args = json.loads(tc.get("function", {}).get("arguments", "{}"))
        filler_text = args.get("spoken_filler", "One moment please.")
        task_desc = args.get("task", "Look up contact")

        # — Step 2: Filler TTS + Reasoning with REAL HubSpot (concurrent) —
        filler_ttfb = None
        filler_bytes = 0

        async def stream_filler():
            nonlocal filler_ttfb, filler_bytes
            url = (
                f"https://api.elevenlabs.io/v1/text-to-speech"
                f"/{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000"
            )
            async with httpx.AsyncClient(timeout=30.0) as c, c.stream(
                "POST", url,
                headers={
                    "xi-api-key": ELEVENLABS_API_KEY,
                    "Content-Type": "application/json",
                },
                json={"text": filler_text, "model_id": ELEVENLABS_MODEL_ID},
            ) as resp:
                resp.raise_for_status()
                first = True
                async for chunk in resp.aiter_bytes(chunk_size=4800):
                    if first:
                        filler_ttfb = (time.monotonic() - t_mini) * 1000
                        first = False
                    filler_bytes += len(chunk)

        reasoning_latency = None
        hubspot_latency = None

        async def run_reasoning_with_live_hubspot():
            nonlocal reasoning_latency, hubspot_latency
            t_r = time.monotonic()

            # Step 2a: GPT-4.1 decides which tool to call
            reasoning_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a reasoning system. Use tools to complete tasks. "
                        "When done, respond with ONLY valid JSON.\n\n"
                        f"Delegated task: {task_desc}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Hi, I'm Ryan Ross from Diocesan. Can you pull up my account?"
                    ),
                },
            ]
            tools = [{
                "type": "function",
                "function": {
                    "name": "lookup_contact",
                    "description": "Search HubSpot CRM by email or phone.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "email_or_phone": {
                                "type": "string",
                                "description": "Email or phone",
                            },
                        },
                        "required": ["email_or_phone"],
                    },
                },
            }]

            async with httpx.AsyncClient(timeout=30.0) as c:
                resp = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENAI_REASONING_MODEL,
                        "messages": reasoning_messages,
                        "max_tokens": 300,
                        "tools": tools,
                    },
                )
                resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]

            # Step 2b: Execute REAL HubSpot lookup
            tool_result = {"status": "not_found"}
            if msg.get("tool_calls"):
                tc_r = msg["tool_calls"][0]
                tc_args = json.loads(tc_r["function"]["arguments"])
                search_val = tc_args.get("email_or_phone", "")

                t_hs = time.monotonic()
                async with httpx.AsyncClient(timeout=30.0) as c:
                    # Try email search first
                    hs_resp = await c.post(
                        "https://api.hubapi.com/crm/v3/objects/contacts/search",
                        headers={
                            "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "filterGroups": [{
                                "filters": [{
                                    "propertyName": "email",
                                    "operator": "CONTAINS_TOKEN",
                                    "value": search_val,
                                }],
                            }],
                            "properties": [
                                "firstname", "lastname", "email",
                                "company", "phone",
                            ],
                            "limit": 1,
                        },
                    )
                    hs_resp.raise_for_status()
                hubspot_latency = (time.monotonic() - t_hs) * 1000

                hs_data = hs_resp.json()
                results = hs_data.get("results", [])
                if results:
                    props = results[0]["properties"]
                    tool_result = {
                        "status": "found",
                        "contact_id": results[0]["id"],
                        **props,
                    }

                reasoning_messages.append(msg)
                reasoning_messages.append({
                    "role": "tool",
                    "tool_call_id": tc_r["id"],
                    "content": json.dumps(tool_result),
                })

            # Step 2c: GPT-4.1 generates structured JSON from real results
            async with httpx.AsyncClient(timeout=30.0) as c:
                resp2 = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": OPENAI_REASONING_MODEL,
                        "messages": reasoning_messages,
                        "max_tokens": 500,
                    },
                )
                resp2.raise_for_status()
            reasoning_latency = (time.monotonic() - t_r) * 1000
            content = resp2.json()["choices"][0]["message"]["content"]
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"status": "success", "summary": content}

        # Run filler + reasoning concurrently
        reasoning_result, _ = await asyncio.gather(
            run_reasoning_with_live_hubspot(), stream_filler(),
        )

        # — Step 3: Mini follow-up —
        followup_messages = [
            {
                "role": "system",
                "content": (
                    "You are a friendly sales agent. Keep responses to 1-2 sentences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Hi, I'm Ryan Ross from Diocesan. Can you pull up my account?"
                ),
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "rr",
                    "type": "function",
                    "function": {
                        "name": "reasoning_result",
                        "arguments": json.dumps(reasoning_result),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": "rr",
                "content": json.dumps(reasoning_result),
            },
            {
                "role": "system",
                "content": (
                    "Reasoning completed. Craft a natural response (1-3 sentences). "
                    "Do NOT mention tools."
                ),
            },
        ]

        t_fu_start = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r3 = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_CONVERSATION_MODEL,
                    "messages": followup_messages,
                    "max_tokens": 150,
                },
            )
            r3.raise_for_status()
        t_fu_done = time.monotonic()
        followup_latency = (t_fu_done - t_fu_start) * 1000
        followup_text = r3.json()["choices"][0]["message"]["content"]

        # — Step 4: Response TTS TTFB —
        response_ttfb = None
        response_bytes = 0
        url = (
            f"https://api.elevenlabs.io/v1/text-to-speech"
            f"/{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000"
        )
        async with httpx.AsyncClient(timeout=30.0) as client, client.stream(
            "POST", url,
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={"text": followup_text, "model_id": ELEVENLABS_MODEL_ID},
        ) as resp:
            resp.raise_for_status()
            first = True
            async for chunk in resp.aiter_bytes(chunk_size=4800):
                if first:
                    response_ttfb = (time.monotonic() - t_fu_done) * 1000
                    first = False
                response_bytes += len(chunk)

        t_end = time.monotonic()
        total_ms = (t_end - t_start) * 1000
        filler_audio_s = filler_bytes / (24000 * 2)
        response_audio_s = response_bytes / (24000 * 2)

        # — Report —
        print("\n" + "=" * 60)
        print("  FULL PIPELINE — LIVE HUBSPOT DELEGATION TURN")
        print("=" * 60)
        print(f"  1. Mini routing decision:    {mini_latency:6.0f} ms")
        print(f"  2. Filler TTS TTFB:          {filler_ttfb or 0:6.0f} ms (after mini)")
        print(f"     Filler audio duration:    {filler_audio_s:6.1f} s")
        print(f"  3. Reasoning (with HubSpot): {reasoning_latency or 0:5.0f} ms (concurrent)")
        print(f"     ├─ HubSpot API call:      {hubspot_latency or 0:6.0f} ms (REAL)")
        print(f"     └─ LLM reasoning only:    "
              f"{(reasoning_latency or 0) - (hubspot_latency or 0):6.0f} ms")
        print(f"  4. Mini follow-up:           {followup_latency:6.0f} ms")
        print(f"  5. Response TTS TTFB:        {response_ttfb or 0:6.0f} ms")
        print(f"     Response audio duration:  {response_audio_s:6.1f} s")
        print("  ─────────────────────────────────────────")
        print(f"  Total wall-clock:            {total_ms:6.0f} ms")
        print(f"  Time to filler audio:        {mini_latency + (filler_ttfb or 0):6.0f} ms")
        print(f"  Time to answer audio:        {total_ms - response_audio_s * 1000:6.0f} ms")
        print("  ─────────────────────────────────────────")
        print(f"  Filler:    '{filler_text[:60]}'")
        print(f"  Response:  '{followup_text[:80]}'")
        print(f"  Reasoning: {json.dumps(reasoning_result)[:120]}")
        print("=" * 60)

        # Key assertions
        time_to_filler = mini_latency + (filler_ttfb or 0)
        assert time_to_filler < 3000, (
            f"Time to filler audio {time_to_filler}ms > 3s"
        )


# =============================================================================
# 9. LIVE SLACK WEBHOOK LATENCY
# =============================================================================


class TestSlackLive:
    """Measure real Slack webhook latency."""

    @pytest.mark.asyncio
    async def test_slack_webhook_latency(self):
        """Post a test message to Slack — measures webhook response time."""
        skip_if_no_key(SLACK_WEBHOOK_URL, "SLACK_WEBHOOK_URL")

        payload = {
            "text": "Lead Qualification Test",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "Test: Lead Qualification Agent",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": "*Name:* Test Contact"},
                        {"type": "mrkdwn", "text": "*Company:* TestCorp"},
                        {"type": "mrkdwn", "text": "*Score:* 75/100"},
                        {"type": "mrkdwn", "text": "*Source:* E2E test suite"},
                    ],
                },
            ],
        }

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                SLACK_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
        latency = (time.monotonic() - t0) * 1000

        print("\n--- Slack: Webhook POST ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Status:                {response.status_code}")
        print(f"  Response:              '{response.text[:80]}'")

        assert response.status_code == 200, f"Slack returned {response.status_code}"
        assert latency < 5000, f"Slack webhook too slow: {latency}ms"


# =============================================================================
# 10. LIVE CAL.COM API LATENCY
# =============================================================================


class TestCalComLive:
    """Measure real Cal.com API latency."""

    @pytest.mark.asyncio
    async def test_calcom_list_event_types_latency(self):
        """List event types — measures Cal.com API read time."""
        skip_if_no_key(CAL_COM_API_KEY, "CAL_COM_API_KEY")

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.cal.com/v2/event-types",
                headers={
                    "Authorization": f"Bearer {CAL_COM_API_KEY}",
                    "cal-api-version": "2024-06-14",
                    "Content-Type": "application/json",
                },
            )
        latency = (time.monotonic() - t0) * 1000

        print("\n--- Cal.com: List Event Types ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Status:                {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            event_types = data.get("data", [])
            print(f"  Event types found:     {len(event_types)}")
            for et in event_types[:3]:
                print(f"    - {et.get('title', '?')} (id={et.get('id', '?')}, "
                      f"{et.get('lengthInMinutes', '?')} min)")
        else:
            print(f"  Response:              '{response.text[:120]}'")

        assert response.status_code == 200, (
            f"Cal.com returned {response.status_code}: {response.text[:200]}"
        )
        assert latency < 5000, f"Cal.com list too slow: {latency}ms"

    @pytest.mark.asyncio
    async def test_calcom_availability_latency(self):
        """Check availability slots — measures Cal.com scheduling read time."""
        skip_if_no_key(CAL_COM_API_KEY, "CAL_COM_API_KEY")

        # First get an event type ID
        async with httpx.AsyncClient(timeout=30.0) as client:
            et_resp = await client.get(
                "https://api.cal.com/v2/event-types",
                headers={
                    "Authorization": f"Bearer {CAL_COM_API_KEY}",
                    "cal-api-version": "2024-06-14",
                    "Content-Type": "application/json",
                },
            )
        if et_resp.status_code != 200:
            pytest.skip("Cannot list event types")

        event_types = et_resp.json().get("data", [])
        if not event_types:
            pytest.skip("No event types configured in Cal.com")

        event_type_id = event_types[0]["id"]

        # Check availability for the next 7 days
        from datetime import datetime, timedelta, timezone
        start = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        end = (datetime.now(tz=timezone.utc) + timedelta(days=7)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                "https://api.cal.com/v2/slots/available",
                params={
                    "startTime": start,
                    "endTime": end,
                    "eventTypeId": event_type_id,
                },
                headers={
                    "Authorization": f"Bearer {CAL_COM_API_KEY}",
                    "cal-api-version": "2024-06-14",
                    "Content-Type": "application/json",
                },
            )
        latency = (time.monotonic() - t0) * 1000

        print("\n--- Cal.com: Availability Check ---")
        print(f"  Latency:               {latency:6.0f} ms")
        print(f"  Status:                {response.status_code}")
        print(f"  Event type:            {event_types[0].get('title', '?')} (id={event_type_id})")

        if response.status_code == 200:
            data = response.json()
            slots = data.get("data", {})
            total_slots = sum(len(v) for v in slots.values()) if isinstance(slots, dict) else 0
            print(f"  Available slots (7d):  {total_slots}")
            # Show first 3 days with slots
            if isinstance(slots, dict):
                for date_key in sorted(slots.keys())[:3]:
                    day_slots = slots[date_key]
                    print(f"    {date_key}: {len(day_slots)} slots")
        else:
            print(f"  Response:              '{response.text[:120]}'")

        assert response.status_code == 200, (
            f"Cal.com availability returned {response.status_code}: {response.text[:200]}"
        )
        assert latency < 5000, f"Cal.com availability too slow: {latency}ms"


# =============================================================================
# 11. END-TO-END MULTI-TURN CONVERSATION
# =============================================================================


# Shared helpers for the E2E conversation test

ROUTING_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "delegate_to_reasoning",
            "description": (
                "Delegate a task to the reasoning system when the conversation "
                "requires CRM lookups, data updates, lead scoring, meeting booking, "
                "SMS sending, sales notifications, or ending the call."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "spoken_filler": {"type": "string"},
                },
                "required": ["task", "spoken_filler"],
            },
        },
    },
]

REASONING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_contact",
            "description": "Search HubSpot CRM by email or phone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email_or_phone": {"type": "string"},
                },
                "required": ["email_or_phone"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_or_update_contact",
            "description": "Create or update a contact in HubSpot CRM.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "firstname": {"type": "string"},
                    "lastname": {"type": "string"},
                    "company": {"type": "string"},
                    "jobtitle": {"type": "string"},
                    "phone": {"type": "string"},
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_lead",
            "description": "Score lead using BANT criteria and create a deal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string"},
                    "company_size": {"type": "string"},
                    "budget": {"type": "string"},
                    "timeline": {"type": "string"},
                    "authority": {"type": "string"},
                    "need": {"type": "string"},
                },
                "required": ["contact_id", "budget", "timeline", "authority", "need"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_sales",
            "description": "Post a notification to the sales Slack channel.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "company": {"type": "string"},
                    "email": {"type": "string"},
                    "score": {"type": "integer"},
                    "notes": {"type": "string"},
                },
                "required": ["name", "company", "score"],
            },
        },
    },
]

SYSTEM_PROMPT_FOR_TEST = (
    "You are Alex, an AI sales development representative (SDR) for TechFlow Solutions, "
    "a B2B SaaS company providing workflow automation tools.\n\n"
    "Qualify leads using BANT (Budget, Authority, Need, Timeline).\n"
    "Keep responses to 1-2 sentences. Use backchannel signals.\n"
    "Never ask more than one question at a time.\n\n"
    "When you need CRM lookups, scoring, or notifications, call delegate_to_reasoning "
    "with a task description and a spoken_filler phrase.\n"
    "Handle conversational turns directly without delegating."
)


async def _call_openai(model, messages, tools=None, max_tokens=300):
    """Shared OpenAI API caller."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        r.raise_for_status()
        return r.json()


async def _synthesize_tts(text):
    """Synthesize text via ElevenLabs, return (pcm_bytes, ttfb_ms, total_ms)."""
    url = (
        f"https://api.elevenlabs.io/v1/text-to-speech"
        f"/{ELEVENLABS_VOICE_ID}/stream?output_format=pcm_24000"
    )
    chunks = []
    ttfb = None
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=30.0) as client, client.stream(
        "POST", url,
        headers={"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"},
        json={"text": text, "model_id": ELEVENLABS_MODEL_ID},
    ) as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes(chunk_size=4800):
            if ttfb is None:
                ttfb = (time.monotonic() - t0) * 1000
            chunks.append(chunk)
    total = (time.monotonic() - t0) * 1000
    return b"".join(chunks), ttfb or 0, total


async def _execute_real_tool(name, args):
    """Execute a real tool call against live APIs. Returns result dict."""
    if name == "lookup_contact":
        email_or_phone = args.get("email_or_phone", "")
        filter_field = "email" if "@" in email_or_phone else "phone"
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts/search",
                headers={
                    "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "filterGroups": [{"filters": [{
                        "propertyName": filter_field,
                        "operator": "EQ",
                        "value": email_or_phone,
                    }]}],
                    "properties": [
                        "email", "phone", "firstname", "lastname",
                        "company", "jobtitle", "lifecyclestage",
                    ],
                },
            )
            r.raise_for_status()
            results = r.json().get("results", [])
            if results:
                props = results[0]["properties"]
                return {
                    "status": "found",
                    "contact_id": results[0]["id"],
                    "name": f"{props.get('firstname', '')} {props.get('lastname', '')}".strip(),
                    "email": props.get("email", ""),
                    "company": props.get("company", ""),
                    "job_title": props.get("jobtitle", ""),
                }
            return {"status": "not_found"}

    if name == "create_or_update_contact":
        email = args.pop("email", "")
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api.hubapi.com/crm/v3/objects/contacts",
                headers={
                    "Authorization": f"Bearer {HUBSPOT_ACCESS_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={"properties": {"email": email, **args}},
            )
            if r.status_code == 409:
                return {"status": "already_exists", "email": email}
            r.raise_for_status()
            return {"status": "created", "contact_id": r.json().get("id", "")}

    if name == "score_lead":
        # Simulate BANT scoring (real scoring creates deals — skip in test)
        score = 0
        if any(w in args.get("budget", "").lower() for w in ["allocated", "approved", "ready"]):
            score += 25
        if any(w in args.get("authority", "").lower() for w in ["decision", "final", "ceo"]):
            score += 25
        if any(w in args.get("need", "").lower() for w in ["urgent", "critical", "pain"]):
            score += 25
        if any(w in args.get("timeline", "").lower() for w in ["now", "asap", "this month"]):
            score += 25
        return {
            "status": "scored",
            "score": score,
            "max_score": 100,
            "qualified": score >= 50,
            "stage": "Qualified to Buy" if score >= 70 else "Presentation Scheduled",
        }

    if name == "notify_sales":
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                SLACK_WEBHOOK_URL,
                json={
                    "text": "E2E Test: Qualified Lead",
                    "blocks": [
                        {
                            "type": "header",
                            "text": {"type": "plain_text", "text": "E2E Test: Qualified Lead"},
                        },
                        {
                            "type": "section",
                            "fields": [
                                {"type": "mrkdwn", "text": f"*Name:* {args.get('name', '?')}"},
                                {
                            "type": "mrkdwn",
                            "text": f"*Company:* {args.get('company', '?')}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Score:* {args.get('score', '?')}/100",
                        },
                            ],
                        },
                    ],
                },
                headers={"Content-Type": "application/json"},
            )
            return {"status": "notified" if r.status_code == 200 else "error"}

    return {"status": "unknown_tool", "name": name}


async def _run_reasoning_with_real_tools(task_desc, conversation_history):
    """Run the reasoning model with real tool execution.

    Returns (result_dict, latency_ms, tool_log).
    """
    reasoning_system = (
        "You are the reasoning and tool-execution system for a voice agent. "
        "Use the available tools to complete the task. When done, respond with "
        "a JSON object with: status, actions_taken, data, summary.\n"
        "IMPORTANT: Return ONLY valid JSON. Do NOT write spoken text.\n\n"
        f"Delegated task: {task_desc}"
    )
    messages = [
        {"role": "system", "content": reasoning_system},
        *conversation_history,
    ]

    tool_log = []
    t0 = time.monotonic()

    for _ in range(5):
        result = await _call_openai(OPENAI_REASONING_MODEL, messages, tools=REASONING_TOOLS)
        msg = result["choices"][0]["message"]

        if not msg.get("tool_calls"):
            latency = (time.monotonic() - t0) * 1000
            raw = msg.get("content", "{}")
            try:
                return json.loads(raw), latency, tool_log
            except json.JSONDecodeError:
                return {"status": "success", "summary": raw}, latency, tool_log

        messages.append(msg)
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments")
            fn_args = json.loads(raw_args) if raw_args else {}
            t_tool = time.monotonic()
            fn_result = await _execute_real_tool(fn_name, fn_args)
            tool_latency = (time.monotonic() - t_tool) * 1000
            tool_log.append({
                "tool": fn_name,
                "latency_ms": round(tool_latency),
                "result_status": fn_result.get("status"),
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(fn_result),
            })

    latency = (time.monotonic() - t0) * 1000
    return {"status": "max_rounds"}, latency, tool_log


class TestE2EConversation:
    """End-to-end multi-turn lead qualification conversation with real APIs and audio playback."""

    @pytest.mark.asyncio
    async def test_full_lead_qualification_conversation(self):
        """Simulate a complete inbound lead qualification call.

        Multi-turn conversation:
        1. Greeting (agent speaks first)
        2. Caller introduces themselves → agent responds conversationally
        3. Caller provides email → agent delegates to CRM lookup
        4. Discovery questions (2 turns) → agent responds directly
        5. Caller reveals BANT info → agent delegates to score lead
        6. Agent wraps up → delegates to notify sales

        Uses real APIs: OpenAI, ElevenLabs, HubSpot, Slack.
        Plays audio through speakers for each agent response.
        """
        skip_if_no_key(OPENAI_API_KEY, "OPENAI_API_KEY")
        skip_if_no_key(ELEVENLABS_API_KEY, "ELEVENLABS_API_KEY")
        skip_if_no_key(HUBSPOT_ACCESS_TOKEN, "HUBSPOT_ACCESS_TOKEN")
        skip_if_no_key(SLACK_WEBHOOK_URL, "SLACK_WEBHOOK_URL")

        import io
        import struct
        import subprocess
        import tempfile

        def pcm_to_wav(pcm_data, sample_rate=24000):
            buf = io.BytesIO()
            buf.write(b"RIFF")
            buf.write(struct.pack("<I", 36 + len(pcm_data)))
            buf.write(b"WAVEfmt ")
            buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
            buf.write(b"data")
            buf.write(struct.pack("<I", len(pcm_data)))
            buf.write(pcm_data)
            return buf.getvalue()

        def play_audio(pcm_data):
            wav = pcm_to_wav(pcm_data)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(wav)
                f.flush()
                subprocess.run(["afplay", f.name], check=True)
            os.unlink(f.name)

        conversation_history = []
        turn_metrics = []
        total_start = time.monotonic()

        async def process_turn(
            user_text, turn_num, expect_delegation=False, play=True,
        ):
            """Process one conversation turn through the dual-LLM pipeline."""
            t_turn_start = time.monotonic()
            print(f"\n  {'─' * 56}")
            print(f"  TURN {turn_num}: CALLER")
            print(f"  \"{user_text}\"")

            conversation_history.append({"role": "user", "content": user_text})
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_FOR_TEST},
                *conversation_history,
            ]

            # Step 1: Mini routing
            t0 = time.monotonic()
            result = await _call_openai(
                OPENAI_CONVERSATION_MODEL, messages, tools=ROUTING_TOOL,
            )
            mini_latency = (time.monotonic() - t0) * 1000
            msg = result["choices"][0]["message"]
            delegated = bool(msg.get("tool_calls"))

            metrics = {
                "turn": turn_num,
                "user_text": user_text[:60],
                "delegated": delegated,
                "mini_routing_ms": round(mini_latency),
            }

            if not delegated:
                # Direct conversational response
                agent_text = msg.get("content", "")
                conversation_history.append({"role": "assistant", "content": agent_text})

                print(f"\n  TURN {turn_num}: AGENT (direct)")
                print(f"  \"{agent_text}\"")

                if play:
                    pcm, ttfb, tts_total = await _synthesize_tts(agent_text)
                    metrics["tts_ttfb_ms"] = round(ttfb)
                    metrics["tts_total_ms"] = round(tts_total)
                    metrics["audio_duration_s"] = round(len(pcm) / (24000 * 2), 1)
                    print("  [Playing audio...]")
                    play_audio(pcm)

                metrics["total_ms"] = round((time.monotonic() - t_turn_start) * 1000)
                turn_metrics.append(metrics)
                return agent_text

            # Delegated turn
            tc = msg["tool_calls"][0]
            raw_args = tc["function"].get("arguments")
            args = json.loads(raw_args) if raw_args else {}
            task_desc = args.get("task", "")
            spoken_filler = args.get("spoken_filler", "One moment please.")

            conversation_history.append(msg)
            conversation_history.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({"status": "delegated", "task": task_desc}),
            })

            print(f"\n  TURN {turn_num}: AGENT [filler]")
            print(f"  \"{spoken_filler}\"")
            print(f"  (delegating: {task_desc[:70]})")

            # Step 2: Filler TTS + reasoning concurrently
            filler_pcm = None
            filler_ttfb = 0
            reasoning_result = None
            reasoning_latency = 0
            tool_log = []

            async def do_filler():
                nonlocal filler_pcm, filler_ttfb
                filler_pcm, filler_ttfb, _ = await _synthesize_tts(spoken_filler)

            async def do_reasoning():
                nonlocal reasoning_result, reasoning_latency, tool_log
                reasoning_result, reasoning_latency, tool_log = (
                    await _run_reasoning_with_real_tools(task_desc, conversation_history)
                )

            await asyncio.gather(do_filler(), do_reasoning())

            metrics["filler_ttfb_ms"] = int(filler_ttfb)
            metrics["filler_audio_s"] = round(len(filler_pcm) / (24000 * 2), 1) if filler_pcm else 0
            metrics["reasoning_ms"] = int(reasoning_latency)
            metrics["tools_called"] = [t["tool"] for t in tool_log]
            metrics["tool_latencies_ms"] = {t["tool"]: t["latency_ms"] for t in tool_log}

            if play and filler_pcm:
                print("  [Playing filler audio...]")
                play_audio(filler_pcm)

            # Step 3: Mini follow-up from reasoning results
            conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "reasoning_result",
                    "type": "function",
                    "function": {
                        "name": "reasoning_result",
                        "arguments": json.dumps(reasoning_result),
                    },
                }],
            })
            conversation_history.append({
                "role": "tool",
                "tool_call_id": "reasoning_result",
                "content": json.dumps(reasoning_result),
            })

            followup_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_FOR_TEST},
                *conversation_history,
                {
                    "role": "system",
                    "content": (
                        "The reasoning system completed the task. Results are above. "
                        "Craft a natural spoken response (1-3 sentences). "
                        "Do NOT mention tools or delegation."
                    ),
                },
            ]

            t_fu = time.monotonic()
            fu_result = await _call_openai(
                OPENAI_CONVERSATION_MODEL, followup_messages, tools=None,
            )
            followup_latency = (time.monotonic() - t_fu) * 1000
            agent_text = fu_result["choices"][0]["message"].get("content", "")
            conversation_history.append({"role": "assistant", "content": agent_text})

            metrics["followup_ms"] = round(followup_latency)

            print(f"\n  TURN {turn_num}: AGENT [response]")
            print(f"  \"{agent_text}\"")

            if play:
                pcm, ttfb, tts_total = await _synthesize_tts(agent_text)
                metrics["tts_ttfb_ms"] = round(ttfb)
                metrics["tts_total_ms"] = round(tts_total)
                metrics["audio_duration_s"] = round(len(pcm) / (24000 * 2), 1)
                print("  [Playing response audio...]")
                play_audio(pcm)

            metrics["total_ms"] = round((time.monotonic() - t_turn_start) * 1000)
            turn_metrics.append(metrics)
            return agent_text

        # ── THE CONVERSATION ──────────────────────────────────────

        print("\n" + "=" * 60)
        print("  E2E LEAD QUALIFICATION CONVERSATION")
        print("  (real APIs: OpenAI, ElevenLabs, HubSpot, Slack)")
        print("=" * 60)

        # Turn 0: Agent greeting
        t_greet = time.monotonic()
        greeting_result = await _call_openai(
            OPENAI_CONVERSATION_MODEL,
            [
                {"role": "system", "content": SYSTEM_PROMPT_FOR_TEST},
                {"role": "user", "content": "Hello, I'm calling for help."},
            ],
        )
        greeting_latency = (time.monotonic() - t_greet) * 1000
        greeting = greeting_result["choices"][0]["message"]["content"]
        conversation_history.append({"role": "user", "content": "Hello, I'm calling for help."})
        conversation_history.append({"role": "assistant", "content": greeting})

        print(f"\n  {'─' * 56}")
        print("  TURN 0: AGENT [greeting]")
        print(f"  \"{greeting}\"")
        pcm, ttfb, _ = await _synthesize_tts(greeting)
        print("  [Playing greeting...]")
        play_audio(pcm)
        turn_metrics.append({
            "turn": 0,
            "user_text": "(inbound call)",
            "delegated": False,
            "mini_routing_ms": round(greeting_latency),
            "tts_ttfb_ms": round(ttfb),
            "audio_duration_s": round(len(pcm) / (24000 * 2), 1),
        })

        # Turn 1: Caller introduces themselves (conversational)
        await process_turn(
            "Hi Alex, I'm Ryan Ross from Diocesan. We're a church publishing company "
            "and I'm looking at ways to automate some of our internal workflows.",
            turn_num=1,
        )

        # Turn 2: Caller provides email → triggers CRM lookup
        await process_turn(
            "Sure, my email is rross@diocesan.com.",
            turn_num=2,
            expect_delegation=True,
        )

        # Turn 3: Discovery — Need (conversational)
        await process_turn(
            "Right now we're manually routing all our print orders through email. "
            "It's a big pain point — things get lost and it takes forever.",
            turn_num=3,
        )

        # Turn 4: Discovery — Budget + Authority (conversational)
        await process_turn(
            "I'm the VP of Operations so I make the final call on tools like this. "
            "We've got budget approved for this quarter, around fifty thousand.",
            turn_num=4,
        )

        # Turn 5: Timeline → triggers lead scoring
        await process_turn(
            "We need something in place this month ideally. The sooner the better.",
            turn_num=5,
            expect_delegation=True,
        )

        # Turn 6: Wrap-up → triggers sales notification
        await process_turn(
            "That sounds great. Yes, please have someone reach out to me.",
            turn_num=6,
            expect_delegation=True,
        )

        total_elapsed = (time.monotonic() - total_start) * 1000

        # ── METRICS REPORT ────────────────────────────────────────

        print("\n\n" + "=" * 70)
        print("  E2E CONVERSATION METRICS REPORT")
        print("=" * 70)

        total_delegated = sum(1 for m in turn_metrics if m.get("delegated"))
        total_direct = sum(1 for m in turn_metrics if not m.get("delegated"))
        avg_direct_ms = 0
        direct_times = [m["mini_routing_ms"] for m in turn_metrics if not m.get("delegated")]
        if direct_times:
            avg_direct_ms = sum(direct_times) / len(direct_times)

        print("\n  Summary:")
        print(f"    Total turns:              {len(turn_metrics)}")
        print(f"    Direct (conversational):  {total_direct}")
        print(f"    Delegated (tool calls):   {total_delegated}")
        print(f"    Total elapsed:            {total_elapsed:,.0f} ms")
        print(f"    Avg direct turn latency:  {avg_direct_ms:,.0f} ms")

        print("\n  Per-turn breakdown:")
        print(f"  {'Turn':>5} {'Type':<10} {'Mini':>7} {'Filler':>7} {'Reason':>7} "
              f"{'Follow':>7} {'TTS':>7} {'Total':>7}  Tools")
        print(f"  {'─' * 5} {'─' * 10} {'─' * 7} {'─' * 7} {'─' * 7} "
              f"{'─' * 7} {'─' * 7} {'─' * 7}  {'─' * 20}")

        for m in turn_metrics:
            turn_type = "delegate" if m.get("delegated") else "direct"
            mini = f"{m.get('mini_routing_ms', 0):,d}"
            filler = f"{m.get('filler_ttfb_ms', '-'):>7}" if m.get("delegated") else f"{'—':>7}"
            reason = f"{m.get('reasoning_ms', '-'):>7}" if m.get("delegated") else f"{'—':>7}"
            follow = f"{m.get('followup_ms', '-'):>7}" if m.get("delegated") else f"{'—':>7}"
            tts = f"{m.get('tts_ttfb_ms', 0):,d}"
            total = f"{m.get('total_ms', 0):,d}"
            tools = ", ".join(m.get("tools_called", [])) or "—"
            print(f"  {m['turn']:>5} {turn_type:<10} {mini:>7} {filler} {reason} "
                  f"{follow} {tts:>7} {total:>7}  {tools}")

        # Tool latency summary
        all_tool_latencies = {}
        for m in turn_metrics:
            for tool, lat in m.get("tool_latencies_ms", {}).items():
                all_tool_latencies.setdefault(tool, []).append(lat)

        if all_tool_latencies:
            print("\n  Tool API latencies (real):")
            for tool, lats in sorted(all_tool_latencies.items()):
                avg = sum(lats) / len(lats)
                print(f"    {tool:<25} avg={avg:,.0f}ms  calls={len(lats)}  "
                      f"values={[f'{lat}ms' for lat in lats]}")

        print("\n" + "=" * 70)

        # Assertions
        assert len(turn_metrics) >= 6, "Expected at least 6 conversation turns"
        assert total_elapsed < 120_000, f"Conversation took too long: {total_elapsed}ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
