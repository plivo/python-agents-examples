# GPT Realtime Native Voice Agent

Real-time voice agent using OpenAI GPT Realtime 1.5 API for speech-to-speech conversations with Plivo telephony and Silero VAD for turn detection.

## Features

- **Speech-to-Speech**: Native audio using OpenAI GPT Realtime API (no separate STT/TTS)
- **Silero VAD**: Client-side voice activity detection for natural turn-taking
- **Barge-in Support**: Users can interrupt the agent mid-response with immediate audio clearing
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Function Calling**: Order status, SMS, callbacks, transfers, and call control
- **Auto-Configuration**: Automatically configures Plivo webhooks on startup
- **No Orchestration**: Direct API integration without frameworks

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key with GPT Realtime API access
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd gpt-realtime-native
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
OPENAI_API_KEY=your_openai_api_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Start ngrok

```bash
ngrok http 8000
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 4. Run the server

```bash
# Inbound (receives calls)
uv run python -m inbound.server

# Outbound (places calls)
uv run python -m outbound.server
```

The inbound server will:
1. Start on port 8000
2. Auto-configure Plivo webhooks for your phone number
3. Display "Ready! Call +1234567890 to test"

### 5. Make a test call

Call your Plivo phone number and start talking to the agent.

## Project Structure

```
gpt-realtime-native/
├── utils.py               # Config + phone utils + audio conversion + VAD
├── inbound/
│   ├── agent.py           # GPTRealtimeVoiceAgent + tools + run_agent for inbound calls
│   ├── server.py          # Standalone inbound FastAPI app
│   └── system_prompt.md   # Inbound call system prompt
├── outbound/
│   ├── agent.py           # GPTRealtimeVoiceAgent + tools + CallManager for outbound
│   ├── server.py          # Standalone outbound FastAPI app
│   └── system_prompt.md   # Outbound call system prompt (with template variables)
├── tests/                 # Integration and E2E tests
├── pyproject.toml         # Project dependencies
├── .env.example           # Environment variable template
└── README.md              # This file
```

## How It Works

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐
│  Phone  │────▶│   Plivo     │────▶│   Server    │
│  Call   │◀────│  (PSTN)     │◀────│  (FastAPI)  │
└─────────┘     └─────────────┘     └──────┬──────┘
                                           │
                     WebSocket (μ-law 8kHz)│
                                           ▼
                                    ┌─────────────┐
                                    │   Agent     │
                                    │ ┌─────────┐ │
                                    │ │ Silero  │ │
                                    │ │  VAD    │ │
                                    │ └────┬────┘ │
                                    │      │      │
                                    │ ┌────▼────┐ │
                                    │ │ OpenAI  │ │
                                    │ │Realtime │ │
                                    │ └─────────┘ │
                                    └─────────────┘
```

1. **Incoming Call**: Plivo receives call and hits `/answer` webhook
2. **WebSocket Setup**: Server returns XML to establish bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **VAD Processing**: Silero VAD detects when the user starts/stops speaking
5. **Turn Management**: On speech end, audio buffer is committed to OpenAI
6. **AI Processing**: GPT Realtime processes speech and generates audio response
7. **Barge-in**: If user speaks during response, the response is cancelled and Plivo playback is cleared
8. **Response Streaming**: Agent converts PCM 24kHz → μ-law 8kHz for Plivo

## Audio Formats

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo → Agent | μ-law | 8 kHz |
| Agent → Silero VAD | PCM float32 | 16 kHz |
| Agent → OpenAI | PCM16 | 24 kHz |
| OpenAI → Agent | PCM16 | 24 kHz |
| Agent → Plivo | μ-law | 8 kHz |

## Silero VAD Configuration

The agent uses client-side Silero VAD instead of OpenAI's built-in server VAD for more control over turn detection:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VAD_START_THRESHOLD` | 0.5 | Speech probability to trigger speech start |
| `VAD_END_THRESHOLD` | 0.35 | Probability below this = silence |
| `VAD_MIN_SILENCE_MS` | 300 | Minimum silence duration to end turn |
| `VAD_CHUNK_SAMPLES` | 512 | Samples per VAD frame (32ms at 16kHz) |

These can be tuned in `utils.py` for your use case. Alternatively, you can disable client-side VAD and use OpenAI's server-side turn detection by setting `turn_detection` in the session config:

```python
"turn_detection": {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500
}
```

## Function Calling

The agent includes these tool functions in each `agent.py`. Replace them with your own implementations:

| Function | Description |
|----------|-------------|
| `check_order_status` | Look up order by number or email |
| `send_sms` | Send text message to customer |
| `schedule_callback` | Schedule callback from specialist |
| `transfer_call` | Transfer to human agent |
| `end_call` | End the conversation gracefully |

To add a new tool, define the function and add its schema to `_build_tools()`:

```python
# 1. Add the tool function
async def get_weather(city: str) -> dict[str, Any]:
    """Get current weather for a city."""
    # Your implementation here
    return {"temperature": "72F", "conditions": "sunny"}

# 2. Add the schema in _build_tools()
{
    "type": "function",
    "name": "get_weather",
    "description": "Get current weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
}

# 3. Add the handler in _handle_function_call()
elif name == "get_weather":
    result = await get_weather(city=args.get("city", ""))
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks (ngrok) | Required |
| `SERVER_PORT` | Server port | `8000` |
| `GPT_REALTIME_MODEL` | OpenAI Realtime model name | `gpt-realtime-1.5` |
| `GPT_REALTIME_VOICE` | Voice name | `alloy` |
| `DEFAULT_COUNTRY_CODE` | ISO 3166-1 alpha-2 code for phone parsing | `US` |
| `SYSTEM_PROMPT` | Override the default system prompt | TechFlow agent |

### Available Voices

| Name | Description |
|------|-------------|
| alloy | Neutral, balanced (default) |
| ash | Warm, conversational |
| ballad | Expressive, storytelling |
| coral | Clear, friendly |
| echo | Smooth, professional |
| sage | Calm, authoritative |
| shimmer | Bright, energetic |
| verse | Versatile, adaptive |

## Testing

The test suite includes unit tests, integration tests, and end-to-end live call tests.

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test levels
uv run pytest tests/test_integration.py -v -k "unit"    # Unit tests (offline)
uv run pytest tests/test_integration.py -v -k "local"   # Local integration
uv run pytest tests/test_integration.py -v -k "openai"  # OpenAI API tests
uv run pytest tests/test_e2e_live.py -v -s              # E2E with real OpenAI API
uv run pytest tests/test_live_call.py -v -s             # Real phone calls via Plivo
uv run pytest tests/test_outbound_call.py -v -s         # Outbound call tests
uv run pytest tests/test_multiturn_voice.py -v -s       # Multi-turn conversation
```

**Requirements for live call tests:**
- Valid Plivo credentials and phone numbers in `.env`
- Valid OpenAI API key in `.env`
- `PLIVO_TEST_NUMBER` — a second Plivo number on the same account (acts as caller for inbound tests, destination for outbound tests)
- ngrok binary available on PATH
- `faster-whisper` (dev dependency, for transcription verification)

## Deployment

### Docker

```bash
# Build the image
docker build -t gpt-realtime-voice-agent .

# Run inbound server (default)
docker run -p 8000:8000 --env-file .env gpt-realtime-voice-agent

# Run outbound server
docker run -p 8000:8000 --env-file .env gpt-realtime-voice-agent \
  uv run python -m outbound.server
```

## Troubleshooting

### No audio heard on call

Plivo requires specific audio format for `playAudio` events. Common mistakes:
- Using `"contentType": "audio/x-mulaw;rate=8000"` (wrong — rate must be separate field)
- Missing `sampleRate` field
- Sending chunks larger than 160 bytes (20ms at 8kHz)

### Connection refused to OpenAI API

- Verify `OPENAI_API_KEY` is correct
- Check that your API key has GPT Realtime API access
- Ensure you're using the GA API format (not beta) — the GA API does not use the `OpenAI-Beta` header

### Agent doesn't respond after speaking

- Check Silero VAD thresholds — `VAD_MIN_SILENCE_MS` may be too high
- Lower `VAD_END_THRESHOLD` if speech endings aren't being detected
- Review server logs for VAD debug messages
- Consider switching to server-side VAD (`turn_detection: server_vad`) for simpler turn management

### Double greeting on call start

- Ensure no `conversation.item.create` with an initial user message is sent before `response.create`
- The agent should only send a bare `response.create` after session setup — the system prompt drives the greeting

### WebSocket disconnects immediately

- Ensure ngrok is running and URL in `.env` matches
- Check Plivo credentials are correct
- Verify the server is accessible from the internet

### Audio quality issues

- 8kHz telephony is lower quality than OpenAI's native 24kHz output
- This is expected — audio is downsampled for phone compatibility
