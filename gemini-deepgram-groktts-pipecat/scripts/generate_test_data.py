#!/usr/bin/env python3
"""
Generate synthetic session data for dashboard testing.

Creates sessions for each TTS provider with realistic pipeline TTFB values.
STT and LLM TTFB are held constant; only TTS TTFB varies by provider.

Usage:
    python scripts/generate_test_data.py
"""

import glob
import json
import os
import random
import time
import uuid
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sessions")

CONVERSATIONS = [
    [
        ("Yeah hi, can you tell me what the weather is like in San Francisco this weekend?",
         "It looks like San Francisco will be partly cloudy on Saturday with a high of sixty-two, and sunny on Sunday reaching sixty-seven degrees."),
        ("Hmm okay... should I bring a jacket?",
         "I'd recommend a light jacket for the evenings. Temperatures will drop to the low fifties after sunset."),
        ("Oh right, and um... I need to set a reminder to call my dentist on Monday morning.",
         "Done! I've set a reminder for Monday at nine AM to call your dentist."),
        ("What do I have on my calendar for Monday?",
         "On Monday you have a team standup at ten, lunch with Sarah at noon, and now your dentist call reminder at nine AM."),
        ("Can you move the standup to ten thirty?",
         "Sure, I've moved your team standup to ten thirty AM on Monday."),
        ("Perfect, that's all. Thanks!",
         "You're welcome! Have a great weekend."),
    ],
    [
        ("Yeah so I'm thinking about a trip to Tokyo next month. What's the best time to visit?",
         "March and April are wonderful for Tokyo. Cherry blossom season starts in late March and the weather is mild."),
        ("How long is the flight from San Francisco?",
         "A direct flight from San Francisco to Tokyo is about eleven hours."),
        ("Actually wait, could you help me convert fifty euros to US dollars?",
         "Fifty euros is approximately fifty-four US dollars at current exchange rates."),
        ("And how about two hundred thousand yen?",
         "Two hundred thousand Japanese yen is approximately one thousand three hundred and thirty US dollars."),
        ("What's a good budget for a week in Tokyo?",
         "For a comfortable trip, I'd budget around two thousand to three thousand dollars for a week."),
        ("Great, that's really helpful. Thanks!",
         "Happy to help! Let me know when you're ready to book flights."),
    ],
    [
        ("Hi, can you check what the temperature in the house is right now?",
         "Your thermostat reads seventy-two degrees. The upstairs is slightly warmer at seventy-four."),
        ("Can you turn it down to sixty-eight?",
         "Done, I've set the thermostat to sixty-eight degrees."),
        ("Add milk and eggs to my shopping list.",
         "I've added milk and eggs to your shopping list. You now have six items."),
        ("What else is on there?",
         "Your shopping list has bread, chicken breast, olive oil, spinach, milk, and eggs."),
        ("Oh wait, actually remove the spinach.",
         "Got it, I've removed spinach from your shopping list."),
        ("Can you also turn on the porch lights?",
         "The porch lights are now on. Would you like me to set them on a timer?"),
    ],
    [
        ("Hey yeah, what time is it in London right now?",
         "It's currently eight fifteen PM in London."),
        ("Set a timer for twenty minutes.",
         "Your twenty-minute timer is set and counting down."),
        ("How do you spell the word Mediterranean?",
         "Mediterranean is spelled M-E-D-I-T-E-R-R-A-N-E-A-N."),
        ("What's the square root of one forty-four?",
         "The square root of one hundred forty-four is twelve."),
        ("Can you read me today's top news headline?",
         "The top headline today is about the Federal Reserve holding interest rates steady."),
        ("Interesting. That's all I needed!",
         "You're welcome! Have a great rest of your day."),
    ],
]

# TTS TTFB ranges per provider (ms) — the key differentiator
TTS_TTFB_RANGES = {
    "xai": (120, 280),
    "elevenlabs": (180, 400),
    "cartesia": (80, 200),
    "openai": (250, 500),
}

# STT and LLM are constant across all providers
STT_TTFB_RANGE = (150, 300)
LLM_TTFB_RANGE = (200, 450)


def generate_session(provider: str, base_time: float):
    session_id = str(uuid.uuid4())
    script = random.choice(CONVERSATIONS)
    num_turns = random.randint(4, min(7, len(script)))
    conversation = script[:num_turns]

    tts_lo, tts_hi = TTS_TTFB_RANGES[provider]
    turns = []
    cursor = base_time

    for i, (user_text, bot_text) in enumerate(conversation):
        turn_num = i + 1
        user_started = cursor + random.uniform(0.8, 2.5)
        speak_duration = len(user_text) * 0.04 + random.uniform(0.3, 0.8)
        user_stopped = user_started + speak_duration

        stt_ttfb = random.uniform(*STT_TTFB_RANGE)
        llm_ttfb = random.uniform(*LLM_TTFB_RANGE)
        tts_ttfb = random.uniform(tts_lo, tts_hi)
        total_latency = stt_ttfb + llm_ttfb + tts_ttfb + random.uniform(20, 80)
        latency_ms = round(total_latency)

        bot_started = user_stopped + latency_ms / 1000
        bot_speak_duration = len(bot_text) * 0.04 + random.uniform(0.3, 0.8)
        bot_stopped = bot_started + bot_speak_duration
        cursor = bot_stopped

        turn = {
            "turn_number": turn_num,
            "user_started_at": round(user_started, 3),
            "user_stopped_at": round(user_stopped, 3),
            "bot_started_at": round(bot_started, 3),
            "bot_stopped_at": round(bot_stopped, 3),
            "response_latency_ms": latency_ms,
            "user_text": user_text,
            "bot_text": bot_text,
            "pipeline": {
                "stt_ttfb_ms": round(stt_ttfb, 1),
                "llm_ttfb_ms": round(llm_ttfb, 1),
                "tts_ttfb_ms": round(tts_ttfb, 1),
            },
        }
        turns.append(turn)

    # Build summary
    latencies = [t["response_latency_ms"] for t in turns]
    stt_vals = [t["pipeline"]["stt_ttfb_ms"] for t in turns]
    llm_vals = [t["pipeline"]["llm_ttfb_ms"] for t in turns]
    tts_vals = [t["pipeline"]["tts_ttfb_ms"] for t in turns]

    dead_air_ms = sum(latencies)
    summary = {
        "total_turns": len(turns),
        "avg_response_latency_ms": round(sum(latencies) / len(latencies)),
        "min_response_latency_ms": min(latencies),
        "max_response_latency_ms": max(latencies),
        "avg_stt_ttfb_ms": round(sum(stt_vals) / len(stt_vals), 1),
        "avg_llm_ttfb_ms": round(sum(llm_vals) / len(llm_vals), 1),
        "avg_tts_ttfb_ms": round(sum(tts_vals) / len(tts_vals), 1),
        "dead_air_ms": dead_air_ms,
        "dead_air_s": round(dead_air_ms / 1000, 2),
    }

    first_user = turns[0]["user_started_at"]
    last_bot = turns[-1]["bot_stopped_at"]
    call_secs = round(last_bot - first_user, 2)

    session = {
        "session_id": session_id,
        "mode": provider,
        "started_at": datetime.fromtimestamp(base_time, tz=timezone.utc).isoformat(),
        "ended_at": datetime.fromtimestamp(base_time + call_secs + 5, tz=timezone.utc).isoformat(),
        "config": {"tts_provider": provider},
        "turns": turns,
        "summary": summary,
    }

    return session


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = glob.glob(os.path.join(DATA_DIR, "*.json"))
    if existing:
        print(f"Clearing {len(existing)} existing session files...")
        for f in existing:
            os.remove(f)

    providers = ["xai", "elevenlabs", "cartesia", "openai"]
    sessions_per_provider = 5
    now = time.time()
    total = 0

    for provider in providers:
        label = {"xai": "xAI", "elevenlabs": "ElevenLabs", "cartesia": "Cartesia", "openai": "OpenAI"}[provider]
        print(f"\nGenerating {sessions_per_provider} sessions for {label}:")
        for i in range(sessions_per_provider):
            base_time = now - (sessions_per_provider - i) * 300
            session = generate_session(provider, base_time)
            path = os.path.join(DATA_DIR, f"{session['session_id']}.json")
            with open(path, "w") as f:
                json.dump(session, f, indent=2)
            avg = session["summary"]["avg_response_latency_ms"]
            tts_avg = session["summary"]["avg_tts_ttfb_ms"]
            turns = session["summary"]["total_turns"]
            print(f"  [{i+1}] {turns} turns, avg={avg}ms, tts={tts_avg}ms  {session['session_id'][:8]}")
            total += 1

    print(f"\n{total} sessions written to {DATA_DIR}")


if __name__ == "__main__":
    main()
