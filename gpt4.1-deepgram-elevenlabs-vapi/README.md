# GPT-4.1 + Deepgram + ElevenLabs + Vapi Voice Agent

Production-ready voice agent using **Vapi** as the orchestrator with:
- **Deepgram Nova-3** for speech-to-text
- **OpenAI GPT-4.1** for conversation intelligence
- **ElevenLabs** for text-to-speech
- **Plivo** for telephony (via SIP trunking)

## Architecture

Unlike native examples that manage WebSocket audio streams directly, Vapi is a **hosted orchestrator** that handles the entire voice pipeline server-side:

```
Inbound:  Caller -> Plivo (SIP) -> Vapi (STT/LLM/TTS/VAD) -> This Server (webhooks)
Outbound: This Server -> Vapi API -> Plivo (SIP) -> Callee
```

**What Vapi manages:**
- Real-time audio streaming and format conversion
- Voice Activity Detection (VAD) with configurable sensitivity
- Turn detection and endpointing (punctuation, silence, smart)
- Interruption handling (barge-in) with sub-100ms response
- Background noise denoising

**What this server handles:**
- Dynamic assistant configuration via `assistant-request` webhook
- Tool/function execution (order lookup, SMS, callbacks, transfers)
- Call lifecycle tracking (status updates, end-of-call reports)
- Outbound call initiation via Vapi API

## VAD and Turn Detection

Vapi provides server-side VAD with configurable parameters:

### Stop Speaking Plan (Interruption Detection)
- `numWords: 2` — minimum words before agent stops (filters noise)
- `voiceSeconds: 0.2` — how long user must speak before agent stops
- `backoffSeconds: 1.0` — quiet period after interruption

### Start Speaking Plan (Turn Detection)
- `waitSeconds: 0.4` — base wait before agent starts speaking
- Nested `transcriptionEndpointingPlan`:
  - `onPunctuationSeconds: 0.1` — wait after punctuation detected
  - `onNoPunctuationSeconds: 1.5` — wait with no punctuation
  - `onNumberSeconds: 0.5` — wait after number detected

### Background Denoising
Enabled by default to filter ambient noise before VAD processing.

## Prerequisites

1. **Vapi account** — Sign up at [vapi.ai](https://vapi.ai)
2. **Plivo account** with Zentrunk (SIP trunking) enabled
3. **API keys** for OpenAI, Deepgram, and ElevenLabs (can be provided via Vapi dashboard as BYOK)

## Setup

### 1. Install Dependencies

```bash
cd gpt4.1-deepgram-elevenlabs-vapi
uv sync
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Set Up Plivo SIP Trunk with Vapi

#### Automated Setup (Recommended)

```bash
uv run python setup_sip.py
```

This script automates the full setup:
1. Creates an IP ACL in Plivo with Vapi's signaling IPs
2. Creates outbound Zentrunk (Vapi -> Plivo -> PSTN)
3. Creates inbound Zentrunk (PSTN -> Plivo -> Vapi)
4. Registers the trunk as a BYO SIP credential in Vapi
5. Imports the Plivo phone number into Vapi
6. Updates `.env` with `VAPI_PHONE_NUMBER_ID`

**Prerequisite**: Zentrunk must be enabled in Plivo Console (Voice > Zentrunk).

#### Manual Setup

<details>
<summary>Click to expand manual setup steps</summary>

##### Outbound Trunk (Vapi -> Plivo -> PSTN)

1. In **Plivo Console** > Zentrunk > Outbound Trunk > IP Access Control List
2. Whitelist Vapi's signaling IPs:
   - `44.229.228.186/32`
   - `44.238.177.138/32`
3. Create an outbound trunk and copy the **Termination SIP Domain** (e.g., `{id}.zt.plivo.com`)

4. Register the trunk with Vapi:
```bash
curl -X POST https://api.vapi.ai/credential \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VAPI_PRIVATE_KEY" \
  -d '{
    "provider": "byo-sip-trunk",
    "name": "Plivo Trunk",
    "gateways": [{ "ip": "YOUR_ID.zt.plivo.com" }]
  }'
```

5. Import your Plivo phone number:
```bash
curl -X POST https://api.vapi.ai/phone-number \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $VAPI_PRIVATE_KEY" \
  -d '{
    "provider": "byo-phone-number",
    "number": "+1XXXXXXXXXX",
    "credentialId": "YOUR_CREDENTIAL_ID",
    "name": "Plivo Number"
  }'
```

Set the returned `id` as `VAPI_PHONE_NUMBER_ID` in your `.env`.

##### Inbound Trunk (PSTN -> Plivo -> Vapi)

1. In **Plivo Console** > Zentrunk > Inbound Trunks
2. Create a new inbound trunk
3. Add Origination URI: `sip.vapi.ai;transport=udp`
4. Update your Plivo phone number to use this Zentrunk inbound trunk

</details>

### 4. Configure Vapi Webhook

Set your server's webhook URL in the Vapi phone number configuration:
- **Server URL**: `https://your-domain.com/vapi/webhook`

For local development, use ngrok:
```bash
ngrok http 8000
# Then set PUBLIC_URL=https://xxx.ngrok.io in .env
```

### 5. Start the Server

**Inbound:**
```bash
uv run python -m inbound.server
```

**Outbound:**
```bash
uv run python -m outbound.server
```

## Usage

### Inbound Calls

1. Call your Plivo phone number
2. Plivo routes via SIP to Vapi
3. Vapi sends `assistant-request` webhook to your server
4. Your server returns assistant configuration with GPT-4.1, Deepgram, ElevenLabs
5. Vapi manages the conversation, sending tool-call webhooks as needed

### Outbound Calls

```bash
curl -X POST "http://localhost:8000/outbound/call?phone_number=%2B15551234567&opening_reason=your+free+trial&objective=qualify+the+lead"
```

### Check Call Status

```bash
curl "http://localhost:8000/outbound/status/{call_id}"
```

## API Endpoints

### Inbound Server

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/vapi/webhook` | POST | Vapi webhook (assistant-request, tool-calls, status, etc.) |

### Outbound Server

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/outbound/call` | POST | Initiate outbound call |
| `/outbound/status/{call_id}` | GET | Get call status |
| `/outbound/campaign/{campaign_id}` | GET | Get campaign calls |
| `/vapi/webhook` | POST | Vapi webhook (tool-calls, status, end-of-call) |

## Testing

```bash
# Unit tests (offline, no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration tests (starts server, tests webhooks)
uv run pytest tests/test_integration.py -v -k "local"

# Vapi API tests (requires VAPI_PRIVATE_KEY)
uv run pytest tests/test_integration.py -v -k "vapi"

# All tests
uv run pytest tests/ -v
```

## Key Differences from Native Examples

| Aspect | Native Examples | This Vapi Example |
|--------|-----------------|-------------------|
| Audio pipeline | Custom WebSocket + format conversion | Vapi handles everything |
| VAD | Client-side Silero in utils.py | Server-side, configured via assistant config |
| Turn detection | Manual audio buffer commit | Vapi's transcription endpointing |
| Plivo connection | WebSocket streaming (`<Stream>` XML) | SIP trunking (Zentrunk) |
| Agent class | Custom class with plivo_rx/api_rx/plivo_tx tasks | Webhook handlers only |
| Barge-in | Client-side detection + response.cancel | Vapi's interruption pipeline (<100ms) |
| utils.py | Audio conversion + VAD processor | Phone normalization only |

## File Structure

```
gpt4.1-deepgram-elevenlabs-vapi/
├── inbound/
│   ├── __init__.py
│   ├── agent.py              # Vapi assistant config + webhook handlers
│   ├── server.py             # FastAPI: /, /vapi/webhook
│   └── system_prompt.md      # Inbound system prompt
├── outbound/
│   ├── __init__.py
│   ├── agent.py              # CallManager + outbound Vapi config
│   ├── server.py             # FastAPI: /, /outbound/call, /vapi/webhook
│   └── system_prompt.md      # Outbound system prompt
├── utils.py                  # Phone normalization (minimal — Vapi handles audio)
├── setup_sip.py              # Automated Plivo SIP trunk + Vapi phone number setup
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_integration.py   # Unit + local integration + API tests
│   ├── test_e2e_live.py
│   ├── test_live_call.py
│   ├── test_multiturn_voice.py
│   └── test_outbound_call.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```
