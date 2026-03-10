"""Test the dual-LLM routing: GPT-4.1 mini (conversation) → GPT-4.1 (reasoning + tools).

Verifies:
  1. Mini handles conversational turns directly (no delegation)
  2. Mini delegates to reasoning when tools are needed
  3. Full routing loop: mini → delegate → GPT-4.1 executes tools → mini follow-up
  4. Routing is consistent across different utterance types

Run: uv run pytest tests/test_dual_llm.py -v -s
"""

from __future__ import annotations

import json
import os
import time

import httpx
import pytest
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_CONVERSATION_MODEL = os.getenv("OPENAI_CONVERSATION_MODEL", "gpt-4.1-mini")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL", "gpt-4.1")

SYSTEM_PROMPT = (
    "You are Alex, an AI sales development representative (SDR) for TechFlow Solutions, "
    "a B2B SaaS company providing workflow automation tools.\n\n"
    "Qualify leads using BANT (Budget, Authority, Need, Timeline).\n"
    "Keep responses to 1-2 sentences. Use backchannel signals.\n"
    "Never ask more than one question at a time.\n\n"
    "When you need CRM lookups, scoring, meeting booking, SMS, sales notifications, "
    "or to end the call, call delegate_to_reasoning with a task description and a "
    "spoken_filler phrase.\n"
    "Handle conversational turns directly without delegating."
)

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
                "properties": {"email_or_phone": {"type": "string"}},
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
            "name": "send_sms",
            "description": "Send an SMS follow-up to the caller.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["phone_number", "message"],
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
    {
        "type": "function",
        "function": {
            "name": "schedule_meeting",
            "description": "Book a demo meeting via Cal.com.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string"},
                    "name": {"type": "string"},
                    "preferred_time": {"type": "string"},
                },
                "required": ["email", "name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_call",
            "description": "End the call after summarizing outcomes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string"},
                    "outcome": {"type": "string"},
                },
                "required": ["reason"],
            },
        },
    },
]


# =============================================================================
# Helpers
# =============================================================================


async def call_openai(model, messages, tools=None, max_tokens=300):
    """Call OpenAI API and return the full response JSON."""
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=30.0) as client:
        t0 = time.monotonic()
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        latency_ms = (time.monotonic() - t0) * 1000
        r.raise_for_status()
        data = r.json()
        data["_latency_ms"] = latency_ms
        return data


def extract_response(result):
    """Extract message from OpenAI result. Returns (text, tool_calls, finish_reason)."""
    msg = result["choices"][0]["message"]
    return (
        msg.get("content", ""),
        msg.get("tool_calls"),
        result["choices"][0]["finish_reason"],
    )


def mock_tool_result(name, args):
    """Return a mock tool result for testing (no real API calls)."""
    if name == "lookup_contact":
        return {
            "status": "found",
            "contact_id": "hs-12345",
            "name": "Jane Smith",
            "email": args.get("email_or_phone", "jane@acme.com"),
            "company": "Acme Corp",
            "job_title": "VP Operations",
        }
    if name == "create_or_update_contact":
        return {"status": "created", "contact_id": "hs-67890"}
    if name == "score_lead":
        return {
            "status": "scored", "score": 75, "max_score": 100,
            "qualified": True, "stage": "Qualified to Buy",
        }
    if name == "send_sms":
        return {"status": "sent", "to": args.get("phone_number", "")}
    if name == "notify_sales":
        return {"status": "notified"}
    if name == "schedule_meeting":
        return {"status": "booked", "time": "2026-03-15T10:00:00Z"}
    if name == "end_call":
        return {"status": "ended", "reason": args.get("reason", "")}
    return {"status": "unknown_tool"}


# =============================================================================
# Test 1: Conversational turns → Mini responds directly (no delegation)
# =============================================================================


class TestMiniDirectResponse:
    """Verify GPT-4.1 mini handles conversational turns without delegating."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("utterance", [
        "Hi, I'm interested in learning more about TechFlow.",
        "We're currently using spreadsheets to track everything.",
        "That sounds interesting, tell me more.",
        "Yeah, our team has about 50 people.",
        "What kind of pricing do you offer?",
    ])
    async def test_conversational_turns_no_delegation(self, utterance):
        """Mini should return text directly for conversational utterances."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {"role": "user", "content": utterance},
        ]
        result = await call_openai(OPENAI_CONVERSATION_MODEL, messages, tools=ROUTING_TOOL)
        text, tool_calls, finish_reason = extract_response(result)
        latency = result["_latency_ms"]

        print(f"\n  Utterance: '{utterance}'")
        print(f"  Model: {OPENAI_CONVERSATION_MODEL}")
        print(f"  Finish reason: {finish_reason}")
        print(f"  Response: '{text[:100]}'")
        print(f"  Tool calls: {tool_calls}")
        print(f"  Latency: {latency:.0f}ms")

        assert finish_reason == "stop", f"Expected 'stop', got '{finish_reason}'"
        assert tool_calls is None, f"Mini should NOT delegate for: '{utterance}'"
        assert text.strip(), "Mini should return non-empty text"


# =============================================================================
# Test 2: Tool-requiring turns → Mini delegates to reasoning
# =============================================================================


class TestMiniDelegation:
    """Verify GPT-4.1 mini delegates when tools are needed."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("utterance,expected_keywords", [
        (
            "My email is jane@acme.com, can you look me up in your CRM?",
            ["look", "email", "jane", "crm", "contact"],
        ),
        (
            "Can you send me a text message with the pricing details?",
            ["sms", "text", "send", "message"],
        ),
    ])
    async def test_tool_turns_trigger_delegation(self, utterance, expected_keywords):
        """Mini should call delegate_to_reasoning for tool-requiring utterances."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {
                "role": "user",
                "content": "We're a 200-person company looking at workflow automation.",
            },
            {
                "role": "assistant",
                "content": "Got it, great size for our platform. What tools do you use?",
            },
            {"role": "user", "content": utterance},
        ]
        result = await call_openai(
            OPENAI_CONVERSATION_MODEL, messages, tools=ROUTING_TOOL,
        )
        _text, tool_calls, finish_reason = extract_response(result)
        latency = result["_latency_ms"]

        print(f"\n  Utterance: '{utterance}'")
        print(f"  Model: {OPENAI_CONVERSATION_MODEL}")
        print(f"  Finish reason: {finish_reason}")
        print(f"  Tool calls: {tool_calls is not None}")
        if tool_calls:
            tc = tool_calls[0]
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])
            print(f"  Function: {fn_name}")
            print(f"  Task: {fn_args.get('task', '')[:100]}")
            print(f"  Filler: {fn_args.get('spoken_filler', '')}")
        print(f"  Latency: {latency:.0f}ms")

        assert tool_calls is not None, f"Mini should delegate for: '{utterance}'"
        assert tool_calls[0]["function"]["name"] == "delegate_to_reasoning"
        fn_args = json.loads(tool_calls[0]["function"]["arguments"])
        assert "task" in fn_args, "Delegation must include a task"
        assert "spoken_filler" in fn_args, "Delegation must include spoken_filler"
        task_lower = fn_args["task"].lower()
        assert any(kw in task_lower for kw in expected_keywords), (
            f"Task should mention one of {expected_keywords}: '{fn_args['task']}'"
        )


# =============================================================================
# Test 3: Reasoning model executes tools correctly
# =============================================================================


class TestReasoningToolExecution:
    """Verify GPT-4.1 picks the right tools for delegated tasks."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("task_desc,expected_tool", [
        (
            "Look up the contact jane@acme.com in HubSpot CRM",
            "lookup_contact",
        ),
        (
            "Create contact: email=bob@widgets.io, firstname=Bob, "
            "lastname=Lee, company=Widgets Inc",
            "create_or_update_contact",
        ),
        (
            "Score this lead: contact_id=hs-12345, budget=allocated $50k, "
            "timeline=this quarter, authority=VP decision maker, need=urgent workflow automation",
            "score_lead",
        ),
        (
            "Notify the sales team: Jane Smith from Acme Corp, score 75/100, "
            "interested in enterprise plan",
            "notify_sales",
        ),
    ])
    async def test_reasoning_picks_correct_tool(self, task_desc, expected_tool):
        """GPT-4.1 should call the correct tool for each delegated task."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the reasoning and tool-execution system for a voice agent. "
                    "Use the available tools to complete the delegated task.\n\n"
                    f"Delegated task: {task_desc}"
                ),
            },
        ]
        result = await call_openai(OPENAI_REASONING_MODEL, messages, tools=REASONING_TOOLS)
        _text, tool_calls, finish_reason = extract_response(result)
        latency = result["_latency_ms"]

        print(f"\n  Task: '{task_desc[:80]}'")
        print(f"  Model: {OPENAI_REASONING_MODEL}")
        print(f"  Finish reason: {finish_reason}")
        if tool_calls:
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                fn_args = json.loads(tc["function"]["arguments"])
                print(f"  Tool: {fn_name}({json.dumps(fn_args)[:100]})")
        print(f"  Latency: {latency:.0f}ms")

        assert tool_calls is not None, f"Reasoning should call a tool for: '{task_desc}'"
        tool_names = [tc["function"]["name"] for tc in tool_calls]
        assert expected_tool in tool_names, (
            f"Expected '{expected_tool}' in {tool_names}"
        )


# =============================================================================
# Test 4: Full routing loop — mini → delegate → reasoning → tools → mini follow-up
# =============================================================================


class TestFullRoutingLoop:
    """End-to-end dual-LLM routing with mock tool execution."""

    @pytest.mark.asyncio
    async def test_crm_lookup_full_loop(self):
        """User provides email → mini delegates → reasoning looks up → mini responds."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {
                "role": "user",
                "content": "My email is jane@acme.com — am I in your system?",
            },
        ]
        timings = {}

        # Step 1: Mini with routing tool
        print("\n  === Step 1: GPT-4.1 mini (conversation) ===")
        t0 = time.monotonic()
        result = await call_openai(
            OPENAI_CONVERSATION_MODEL, conversation, tools=ROUTING_TOOL,
        )
        timings["mini_routing"] = (time.monotonic() - t0) * 1000
        _text, tool_calls, _finish_reason = extract_response(result)

        assert tool_calls is not None, "Mini should delegate for CRM lookup"
        tc = tool_calls[0]
        fn_args = json.loads(tc["function"]["arguments"])
        spoken_filler = fn_args["spoken_filler"]
        task_desc = fn_args["task"]
        print(f"  Filler: '{spoken_filler}'")
        print(f"  Task: '{task_desc[:100]}'")
        print(f"  Latency: {timings['mini_routing']:.0f}ms")

        # Add mini's delegation to history
        conversation.append(result["choices"][0]["message"])
        conversation.append({
            "role": "tool",
            "tool_call_id": tc["id"],
            "content": json.dumps({"status": "delegated", "task": task_desc}),
        })

        # Step 2: Reasoning model executes tools
        print("\n  === Step 2: GPT-4.1 (reasoning + tool execution) ===")
        reasoning_messages = [
            {
                "role": "system",
                "content": (
                    "You are the reasoning and tool-execution system for a voice agent. "
                    "Use the available tools to complete the task. When done, respond with "
                    "a JSON summary.\n\n"
                    f"Delegated task: {task_desc}"
                ),
            },
        ]
        t0 = time.monotonic()
        reason_result = await call_openai(
            OPENAI_REASONING_MODEL, reasoning_messages, tools=REASONING_TOOLS,
        )
        timings["reasoning_initial"] = (time.monotonic() - t0) * 1000
        _, reason_tools, _ = extract_response(reason_result)

        assert reason_tools is not None, "Reasoning should call lookup_contact"
        tool_log = []

        # Execute tool calls in a loop (reasoning may chain multiple)
        while reason_tools:
            reason_msg = reason_result["choices"][0]["message"]
            reasoning_messages.append(reason_msg)

            for rtc in reason_tools:
                fn_name = rtc["function"]["name"]
                fn_args = json.loads(rtc["function"]["arguments"])
                mock_result = mock_tool_result(fn_name, fn_args)
                tool_log.append({"tool": fn_name, "args": fn_args, "result": mock_result})
                args_s = json.dumps(fn_args)[:80]
                mock_s = json.dumps(mock_result)[:80]
                print(f"  Tool: {fn_name}({args_s}) → {mock_s}")
                reasoning_messages.append({
                    "role": "tool",
                    "tool_call_id": rtc["id"],
                    "content": json.dumps(mock_result),
                })

            t0 = time.monotonic()
            reason_result = await call_openai(
                OPENAI_REASONING_MODEL, reasoning_messages, tools=REASONING_TOOLS,
            )
            timings["reasoning_followup"] = (time.monotonic() - t0) * 1000
            reason_text, reason_tools, _ = extract_response(reason_result)

        # Parse reasoning result
        reasoning_output = reason_text or json.dumps(tool_log[-1]["result"])
        print(f"  Reasoning output: {reasoning_output[:120]}")
        r_init = timings['reasoning_initial']
        r_follow = timings.get('reasoning_followup', 0)
        print(f"  Latency: {r_init:.0f}ms + {r_follow:.0f}ms")

        # Step 3: Mini crafts spoken follow-up
        print("\n  === Step 3: GPT-4.1 mini (follow-up response) ===")
        conversation.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "reasoning_result",
                "type": "function",
                "function": {
                    "name": "reasoning_result",
                    "arguments": reasoning_output,
                },
            }],
        })
        conversation.append({
            "role": "tool",
            "tool_call_id": "reasoning_result",
            "content": reasoning_output,
        })

        followup_messages = [
            *conversation,
            {
                "role": "system",
                "content": (
                    "The reasoning system has completed the delegated task. "
                    "Craft a natural spoken response for the caller based on the results. "
                    "Keep it conversational and concise (1-3 sentences). "
                    "Do NOT mention tools or delegation."
                ),
            },
        ]
        t0 = time.monotonic()
        followup_result = await call_openai(
            OPENAI_CONVERSATION_MODEL, followup_messages, tools=None,
        )
        timings["mini_followup"] = (time.monotonic() - t0) * 1000
        followup_text, followup_tools, _ = extract_response(followup_result)

        print(f"  Response: '{followup_text}'")
        print(f"  Latency: {timings['mini_followup']:.0f}ms")

        assert followup_tools is None, "Follow-up should be text only, no tools"
        assert followup_text.strip(), "Follow-up should not be empty"
        # The response should reference the lookup result
        text_lower = followup_text.lower()
        assert any(w in text_lower for w in ["jane", "acme", "found", "system", "record"]), (
            f"Follow-up should reference the CRM result: '{followup_text}'"
        )

        # Summary
        total = sum(timings.values())
        print("\n  === Timing Summary ===")
        for step, ms in timings.items():
            print(f"  {step}: {ms:.0f}ms")
        print(f"  TOTAL: {total:.0f}ms")

    @pytest.mark.asyncio
    async def test_conversational_turn_no_reasoning(self):
        """Conversational turn should complete in a single mini call — no reasoning needed."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {
                "role": "user",
                "content": "We have about 150 employees, drowning in manual processes.",
            },
        ]

        print("\n  === Single mini call (no delegation) ===")
        t0 = time.monotonic()
        result = await call_openai(
            OPENAI_CONVERSATION_MODEL, conversation, tools=ROUTING_TOOL,
        )
        latency = (time.monotonic() - t0) * 1000
        text, tool_calls, _finish_reason = extract_response(result)

        print(f"  Response: '{text[:120]}'")
        print(f"  Tool calls: {tool_calls}")
        print(f"  Latency: {latency:.0f}ms")

        assert tool_calls is None, "Should NOT delegate for conversational turn"
        assert text.strip(), "Should return a response"
        assert latency < 5000, f"Single mini call should be fast, got {latency:.0f}ms"


# =============================================================================
# Test 5: Routing consistency — same utterance type routes the same way
# =============================================================================


class TestRoutingConsistency:
    """Run the same utterance 3 times and verify routing is consistent."""

    @pytest.mark.asyncio
    async def test_conversational_routes_consistently(self):
        """Conversational utterance should never trigger delegation across 3 runs."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {"role": "user", "content": "Tell me about your workflow automation features."},
        ]

        results = []
        for i in range(3):
            result = await call_openai(OPENAI_CONVERSATION_MODEL, messages, tools=ROUTING_TOOL)
            _, tool_calls, finish_reason = extract_response(result)
            routed = "delegate" if tool_calls else "direct"
            results.append(routed)
            print(f"  Run {i + 1}: {routed} (finish={finish_reason})")

        assert all(r == "direct" for r in results), (
            f"Conversational turn should always route direct, got: {results}"
        )

    @pytest.mark.asyncio
    async def test_tool_turn_routes_consistently(self):
        """Tool-requiring utterance should always trigger delegation across 3 runs."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {
                "role": "user",
                "content": "My email is sarah@bigcorp.com — look me up in your CRM.",
            },
        ]

        results = []
        for i in range(3):
            result = await call_openai(OPENAI_CONVERSATION_MODEL, messages, tools=ROUTING_TOOL)
            _, tool_calls, finish_reason = extract_response(result)
            routed = "delegate" if tool_calls else "direct"
            results.append(routed)
            print(f"  Run {i + 1}: {routed} (finish={finish_reason})")

        assert all(r == "delegate" for r in results), (
            f"Tool turn should always route to delegation, got: {results}"
        )


# =============================================================================
# Test 6: Latency budget check
# =============================================================================


class TestLatencyBudget:
    """Verify each step stays within acceptable latency bounds."""

    @pytest.mark.asyncio
    async def test_mini_latency_under_2s(self):
        """Mini conversation response should be under 2 seconds."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "assistant", "content": "Hi! I'm Alex from TechFlow. How can I help?"},
            {"role": "user", "content": "What does TechFlow do exactly?"},
        ]
        result = await call_openai(OPENAI_CONVERSATION_MODEL, messages, tools=ROUTING_TOOL)
        latency = result["_latency_ms"]
        text, _, _ = extract_response(result)
        print(f"\n  Mini response ({latency:.0f}ms): '{text[:80]}'")
        assert latency < 2000, f"Mini should respond in <2s, got {latency:.0f}ms"

    @pytest.mark.asyncio
    async def test_reasoning_latency_under_5s(self):
        """Reasoning model tool call should be under 5 seconds."""
        if not OPENAI_API_KEY:
            pytest.skip("OPENAI_API_KEY not configured")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the reasoning system. Use the tools to complete this task.\n\n"
                    "Delegated task: Look up contact jane@acme.com in HubSpot CRM"
                ),
            },
        ]
        result = await call_openai(OPENAI_REASONING_MODEL, messages, tools=REASONING_TOOLS)
        latency = result["_latency_ms"]
        _, tool_calls, _ = extract_response(result)
        if tool_calls:
            print(f"\n  Reasoning tool call ({latency:.0f}ms): {tool_calls[0]['function']['name']}")
        assert latency < 5000, f"Reasoning should respond in <5s, got {latency:.0f}ms"
