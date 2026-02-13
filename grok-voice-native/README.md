# Grok Voice Agent (Native)

Real-time voice agent using xAI Grok Voice Agent API for speech-to-speech conversations with Plivo telephony and Silero VAD for turn detection.

## Features

- **Speech-to-Speech**: Native audio using Grok Realtime API (no separate STT/TTS)
- **Silero VAD**: Client-side voice activity detection for natural turn-taking
- **Barge-in Support**: Users can interrupt the agent mid-response
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Function Calling**: Order status, SMS, callbacks, transfers, and call control
- **Auto-Configuration**: Automatically configures Plivo webhooks on startup
- **No Orchestration**: Direct API integration without frameworks

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- xAI API key with Grok Voice Agent API access
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd grok-voice-native
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
XAI_API_KEY=your_xai_api_key
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
grok-voice-native/
├── utils.py               # Config + phone utils + audio conversion + VAD
├── inbound/
│   ├── agent.py           # GrokVoiceAgent + tools + run_agent for inbound calls
│   ├── server.py          # Standalone inbound FastAPI app
│   └── system_prompt.md   # Inbound call system prompt
├── outbound/
│   ├── agent.py           # GrokVoiceAgent + tools + CallManager for outbound
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
                                    │ │  Grok   │ │
                                    │ │Realtime │ │
                                    │ └─────────┘ │
                                    └─────────────┘
```

1. **Incoming Call**: Plivo receives call and hits `/answer` webhook
2. **WebSocket Setup**: Server returns XML to establish bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **VAD Processing**: Silero VAD detects when the user starts/stops speaking
5. **Turn Management**: On speech end, audio buffer is committed to Grok
6. **AI Processing**: Grok processes speech and generates audio response
7. **Barge-in**: If user speaks during response, the response is cancelled
8. **Response Streaming**: Agent converts PCM 24kHz → μ-law 8kHz for Plivo

## Audio Formats

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo → Agent | μ-law | 8 kHz |
| Agent → Silero VAD | PCM float32 | 16 kHz |
| Agent → Grok | PCM16 | 24 kHz |
| Grok → Agent | PCM16 | 24 kHz |
| Agent → Plivo | μ-law | 8 kHz |

## Silero VAD Configuration

The agent uses client-side Silero VAD instead of Grok's built-in server VAD for more control over turn detection:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VAD_START_THRESHOLD` | 0.5 | Speech probability to trigger speech start |
| `VAD_END_THRESHOLD` | 0.35 | Probability below this = silence |
| `VAD_MIN_SILENCE_MS` | 300 | Minimum silence duration to end turn |
| `VAD_CHUNK_SAMPLES` | 512 | Samples per VAD frame (32ms at 16kHz) |

These can be tuned in `utils.py` for your use case.

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
| `XAI_API_KEY` | xAI API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks (ngrok) | Required |
| `SERVER_PORT` | Server port | `8000` |
| `GROK_MODEL` | Grok model name | `grok-3-fast-voice` |
| `GROK_VOICE` | Voice name | `Sal` |

### Available Voices

| Name | Gender | Tone |
|------|--------|------|
| Ara | Female | Warm, friendly (default) |
| Rex | Male | Confident, professional |
| Sal | Neutral | Smooth, versatile |
| Eve | Female | Energetic, upbeat |
| Leo | Male | Authoritative, commanding |

## Testing

The test suite includes unit tests, integration tests, and end-to-end live call tests.

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test levels
uv run pytest tests/test_integration.py -v              # Unit + integration tests
uv run pytest tests/test_e2e_live.py -v -s              # E2E with real Grok API
uv run pytest tests/test_live_call.py -v -s             # Real phone calls via Plivo
uv run pytest tests/test_outbound_call.py -v -s         # Outbound call tests
uv run pytest tests/test_multiturn_voice.py -v -s       # Multi-turn conversation
```

**Requirements for live call tests:**
- Valid Plivo credentials and phone numbers in `.env`
- Valid xAI API key in `.env`
- ngrok installed at `/usr/local/bin/ngrok`
- `faster-whisper` (dev dependency, for transcription verification)

## Deployment

### Docker

```bash
# Build the image
docker build -t grok-voice-agent .

# Run inbound server (default)
docker run -p 8000:8000 --env-file .env grok-voice-agent

# Run outbound server
docker run -p 8000:8000 --env-file .env grok-voice-agent \
  uv run python -m outbound.server
```

## Troubleshooting

### No audio heard on call

Plivo requires specific audio format for `playAudio` events. Common mistakes:
- Using `"contentType": "audio/x-mulaw;rate=8000"` (wrong - rate must be separate)
- Missing `sampleRate` field
- Sending chunks larger than 160 bytes (20ms at 8kHz)

### Connection refused to xAI API

- Verify `XAI_API_KEY` is correct
- Check that your API key has Grok Voice Agent API access
- The Voice Agent API is only available in `us-east-1` region

### Agent doesn't respond after speaking

- Check Silero VAD thresholds - the `VAD_MIN_SILENCE_MS` may be too high
- Lower `VAD_END_THRESHOLD` if speech endings aren't being detected
- Review server logs for VAD debug messages

### WebSocket disconnects immediately

- Ensure ngrok is running and URL in `.env` matches
- Check Plivo credentials are correct
- Verify the server is accessible from the internet

### Audio quality issues

- 8kHz telephony is lower quality than Grok's native 24kHz output
- This is expected - audio is downsampled for phone compatibility
