# Grok Voice (Think Fast 1.0) + Plivo Voice Agent (Native)

Native speech-to-speech voice agent using the xAI realtime API with the `grok-voice-think-fast-1.0` model, Plivo telephony, and server-side turn detection.

## Features

- **Speech-to-Speech**: Native audio using the xAI realtime API (no separate STT/TTS)
- **Server-Side VAD**: Turn detection handled by the realtime API
- **Barge-in Support**: Callers can interrupt the agent mid-response with immediate audio clearing
- **Multi-turn Conversations**: Maintains context across the call
- **Function Calling**: Order status, SMS, callbacks, transfers, and call control
- **Auto-Configuration**: Automatically configures Plivo webhooks on startup
- **No Orchestration**: Direct API integration without frameworks
- **Inbound and Outbound**: Supports both receiving and placing calls

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- xAI API key with realtime API access
- Plivo account with voice-enabled phone numbers
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```sh
cd xai-realtime-native
uv sync
```

### 2. Configure environment

```sh
cp .env.example .env
```

Edit `.env` with your credentials:

```sh
XAI_API_KEY=your_xai_api_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Start ngrok

```sh
ngrok http 8000
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 4. Run the server

```sh
# Inbound (receives calls)
uv run python -m inbound.server

# Outbound (places calls)
uv run python -m outbound.server
```

The inbound server will:
1. Start on port 8000
2. Auto-configure Plivo webhooks for your phone number
3. Display `Ready! Call +1234567890 to test`

### 5. Make a test call

#### Inbound
Call your Plivo phone number and start talking to the agent.

#### Outbound
With the outbound server running:

```sh
curl -X POST "http://localhost:8000/outbound/call?phone_number=+1234567890"
```

## Project Structure

```text
xai-realtime-native/
├── utils.py               # Phone normalization helpers
├── inbound/
│   ├── agent.py           # XAIRealtimeAgent + tools + run_agent for inbound calls
│   ├── server.py          # Standalone inbound FastAPI app
│   └── system_prompt.md   # Inbound call system prompt
├── outbound/
│   ├── agent.py           # XAIRealtimeAgent + tools + CallManager for outbound
│   ├── server.py          # Standalone outbound FastAPI app
│   └── system_prompt.md   # Outbound call system prompt
├── tests/                 # Integration and live-call tests
├── pyproject.toml         # Project dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

## How It Works

```text
┌─────────┐     ┌─────────────┐     ┌─────────────┐
│  Phone  │────▶│   Plivo     │────▶│   Server    │
│  Call   │◀────│  (PSTN)     │◀────│  (FastAPI)  │
└─────────┘     └─────────────┘     └──────┬──────┘
                                           │
                     WebSocket (μ-law 8kHz)│
                                           ▼
                                    ┌─────────────┐
                                    │   Agent     │
                                    │ (Realtime   │
                                    │   bridge)   │
                                    │             │
                                    │     xAI     │
                                    │  Realtime   │
                                    └─────────────┘
```

1. **Incoming or Outbound Call**: Plivo receives or places the call and hits your webhook
2. **WebSocket Setup**: Server returns XML to establish a bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **Realtime Session**: Server opens a separate WebSocket to the xAI realtime API
5. **Turn Detection**: xAI server-side VAD detects when the caller starts and stops speaking
6. **AI Processing**: xAI generates streaming audio responses
7. **Barge-in**: If the caller speaks during playback, the server clears Plivo audio immediately
8. **Response Streaming**: Agent forwards model audio back to Plivo as `playAudio`

## Audio Formats

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo → Agent | μ-law | 8 kHz |
| Agent → xAI | `audio/pcmu` | 8 kHz |
| xAI → Agent | `audio/pcmu` | 8 kHz |
| Agent → Plivo | μ-law | 8 kHz |

## Turn Detection

This example uses **server-side turn detection** from the xAI realtime API.

### What that means

- the agent forwards telephony audio directly to xAI
- xAI decides when speech starts and stops
- xAI emits interruption signals like `input_audio_buffer.speech_started`
- the server reacts by sending `clearAudio` to Plivo for barge-in

### Tradeoff

- **Pros**: simpler architecture, fewer moving parts, no local VAD model
- **Cons**: less control over turn timing than a client-side VAD pipeline

## Function Calling

The agent includes these tool functions in each `agent.py`:

| Function | Description |
|----------|-------------|
| `check_order_status` | Look up order by number or email |
| `send_sms` | Send text message to customer |
| `schedule_callback` | Schedule callback from specialist |
| `transfer_call` | Transfer to human agent |
| `end_call` | End the conversation gracefully |

`send_sms` is wired to the Plivo SMS API, so you can use it directly to try the live SMS flow with your configured `PLIVO_PHONE_NUMBER`.

To add a new tool, define the function and add its schema to `_build_tools()`.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_API_KEY` | xAI API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks (ngrok) | Required |
| `SERVER_PORT` | Server port | `8000` |
| `XAI_REALTIME_MODEL` | Realtime voice model (`grok-voice-latest` tracks the newest) | `grok-voice-think-fast-1.0` |
| `XAI_VOICE` | Voice name | `Sal` |
| `DEFAULT_COUNTRY_CODE` | ISO 3166-1 alpha-2 code for phone parsing | `US` |
| `SYSTEM_PROMPT` | Override the default system prompt | TechFlow agent |

`PLIVO_PHONE_NUMBER` is the live voice and SMS source number for this example. In our setup, use a US Plivo number as the outbound caller ID and SMS source.

## Testing

This xAI variant currently has one practical test path: local and API-level checks. The dedicated live-call test files are placeholders and should not be presented as a ready validation path yet.

### Run unit and local integration tests

```sh
uv sync --group dev
uv run --group dev python -m pytest tests/test_integration.py -v -k "unit or local"
```

### Live-call coverage status

- `tests/test_e2e_live.py` is available for API-level validation.
- `tests/test_live_call.py` and `tests/test_outbound_call.py` are still scaffolds for future live-call automation.
- For now, validate inbound and outbound telephony manually by running the servers and placing real calls.

## Troubleshooting

- **No audio from agent**: Check `XAI_API_KEY`, `PLIVO_AUTH_ID`, and `PLIVO_AUTH_TOKEN`, and confirm the call answered path reached your server.
- **WebSocket bridge never starts**: Make sure `PUBLIC_URL` matches your active ngrok tunnel and that Plivo can reach `/answer` and `/ws`.
- **Call connects but there is silence**: Confirm the bridge is using `audio/x-mulaw;rate=8000` on the Plivo side and `audio/pcmu` on the xAI side.
- **Barge-in is not working**: Ensure the Plivo WebSocket continues delivering caller audio during playback so xAI can emit `input_audio_buffer.speech_started`.
- **Slow turn detection**: This example uses xAI server-side VAD, so interruption and end-of-turn timing depend on the realtime model rather than local VAD tuning.
- **Outbound call uses the wrong caller ID**: Set `PLIVO_PHONE_NUMBER` to the correct Plivo number before starting `outbound.server`.
