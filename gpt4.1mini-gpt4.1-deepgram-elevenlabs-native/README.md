# gpt4.1mini-gpt4.1-deepgram-elevenlabs-native

Dual-LLM lead qualification voice agent using **GPT-4.1 mini** (fast conversation) + **GPT-4.1** (reasoning & tool execution), **Deepgram nova-3** (streaming STT), **ElevenLabs flash v2.5** (streaming TTS), **smart-turn-v2** (semantic turn detection), and **Plivo** telephony.

## Architecture

```
Plivo μ-law 8kHz
    ├──→ PCM 8kHz ──→ Deepgram nova-3 (streaming WebSocket STT)
    │                      │
    │                  transcript (final)
    │                      │
    ├──→ float32 16kHz ──→ Silero VAD (is_speech signal)
    │                      │
    │                  Smart-turn-v2 (semantic turn completion)
    │                      │
    │                  turn_complete? ──→ use latest Deepgram transcript
    │                                         │
    │                              GPT-4.1 mini (conversation, with tools)
    │                                 │              │
    │                            [text reply]   [tool_calls]
    │                                 │              │
    │                                 │     GPT-4.1 (reasoning model)
    │                                 │     executes: HubSpot, Cal.com,
    │                                 │     Plivo SMS, Slack, end_call
    │                                 │              │
    │                                 ▼              ▼
    │                              ElevenLabs flash v2.5 (streaming TTS)
    │                                      │
    │                                  PCM 24kHz → μ-law 8kHz
    │                                      │
    └──← 160-byte chunks ←────────────────┘
```

### Dual-LLM Routing

GPT-4.1 mini handles every turn with tool definitions. If it returns `tool_calls`, GPT-4.1 executes the tools and generates the follow-up response. If it returns text only, that's used directly. OpenAI's function-calling is the routing signal — no custom classifiers needed.

### Smart-Turn Detection

Instead of simple VAD silence thresholds, smart-turn-v2 uses a Wav2Vec2-based ML model to predict semantic turn completion. This avoids cutting off users mid-thought ("um...", "so...") while still responding quickly to complete utterances (~12ms inference).

### Tool Integrations

| Tool | System | Description |
|------|--------|-------------|
| `lookup_contact` | HubSpot API v3 | Search contacts by email/phone |
| `create_or_update_contact` | HubSpot API v3 | Create/update contact with lead info |
| `score_lead` | HubSpot API v3 | BANT scoring, create/update deal |
| `schedule_meeting` | Cal.com API | Book a demo meeting |
| `send_sms` | Plivo API | Send SMS follow-up |
| `notify_sales` | Slack Webhook | Post to Slack when lead qualified |
| `end_call` | Internal | End call gracefully |

## Setup

```bash
# Install dependencies
uv sync

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys

# Run inbound server
uv run python -m inbound.server

# Run outbound server (separate terminal)
uv run python -m outbound.server
```

## Required Environment Variables

- `OPENAI_API_KEY` — OpenAI API key (for both GPT-4.1 mini and GPT-4.1)
- `DEEPGRAM_API_KEY` — Deepgram API key
- `ELEVENLABS_API_KEY` — ElevenLabs API key
- `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_PHONE_NUMBER` — Plivo credentials

### Optional Integrations

- `HUBSPOT_ACCESS_TOKEN` — HubSpot private app token
- `CAL_COM_API_KEY`, `CAL_COM_EVENT_TYPE_ID` — Cal.com booking
- `SLACK_WEBHOOK_URL` — Slack incoming webhook

## Testing

```bash
# Unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration tests (starts server, needs API keys)
uv run pytest tests/test_integration.py -v -k "local"

# Lint
uv run ruff check .
```

## Outbound Calls

```bash
# Initiate an outbound call
curl -X POST "http://localhost:8000/outbound/call?phone_number=+15551234567&opening_reason=trial+follow-up&objective=qualify+and+book+demo"

# Check call status
curl "http://localhost:8000/outbound/status/{call_id}"
```
