# Gemini Live Voice Agent

Real-time voice agent using Google Gemini Live API for speech-to-speech conversations with Plivo telephony.

## Features

- **Speech-to-Speech**: Native audio using Gemini Live API (no separate STT/TTS)
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Function Calling**: Order status, SMS, callbacks, transfers, and call control
- **Auto-Configuration**: Automatically configures Plivo webhooks on startup
- **Low Latency**: Real-time bidirectional audio streaming

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google AI API key with Gemini Live API access
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd gemini-live-plivo
uv sync
```

### 2. Configure environment

```bash
cp env.example .env
```

Edit `.env` with your credentials:

```bash
GEMINI_API_KEY=your_gemini_api_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Start ngrok

```bash
ngrok http 7860
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 4. Run the server

```bash
uv run python server.py
```

The server will:
1. Start on port 7860
2. Auto-configure Plivo webhooks for your phone number
3. Display "Ready! Call +1234567890 to test"

### 5. Make a test call

Call your Plivo phone number and start talking to the agent.

## Project Structure

```
gemini-live-plivo/
├── agent.py            # Voice agent with Gemini Live API
├── server.py           # FastAPI server with Plivo webhooks
├── tests/              # Integration and voice tests
├── pyproject.toml      # Project dependencies
├── env.example         # Environment variable template
└── Dockerfile          # Container deployment
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
                                    │  (Gemini    │
                                    │   Live)     │
                                    └─────────────┘
```

1. **Incoming Call**: Plivo receives call and hits `/answer` webhook
2. **WebSocket Setup**: Server returns XML to establish bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **Format Conversion**: Agent converts μ-law 8kHz → PCM16 16kHz for Gemini
5. **AI Processing**: Gemini Live processes speech and generates response
6. **Response Streaming**: Agent converts PCM16 24kHz → μ-law 8kHz for Plivo

## Audio Formats

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo → Agent | μ-law | 8 kHz |
| Agent → Gemini | PCM16 | 16 kHz |
| Gemini → Agent | PCM16 | 24 kHz |
| Agent → Plivo | μ-law | 8 kHz |

Audio conversion uses numpy for Python 3.11+ compatibility.

## Function Calling

The agent includes these functions:

| Function | Description |
|----------|-------------|
| `check_order_status` | Look up order by number or email |
| `send_sms` | Send text message to customer |
| `schedule_callback` | Schedule callback from specialist |
| `transfer_call` | Transfer to human agent |
| `end_call` | End the conversation gracefully |

To add custom functions, edit `agent.py`:

```python
# Add function declaration in _build_tools()
types.FunctionDeclaration(
    name="my_function",
    description="What it does",
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "param1": types.Schema(type=types.Type.STRING, description="..."),
        },
        required=["param1"],
    ),
)

# Add handler in _handle_function_call()
elif name == "my_function":
    result = await my_function(args.get("param1"))
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks (ngrok) | Required |
| `SERVER_PORT` | Server port | `7860` |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash-native-audio-preview-12-2025` |
| `GEMINI_VOICE` | Voice name | `Kore` |

### Available Voices

Aoede, Charon, Fenrir, Kore, Puck, and others.

## Testing

### Run unit and integration tests

```bash
uv sync --group dev
uv run pytest tests/test_integration.py -v
```

### Run multi-turn voice test

Requires ffmpeg for TTS audio generation:

```bash
# macOS - download static binary
curl -L "https://evermeet.cx/ffmpeg/getrelease/ffmpeg/zip" -o ffmpeg.zip
unzip ffmpeg.zip && chmod +x ffmpeg

# Run test with ffmpeg in PATH
PATH="$PWD:$PATH" uv run python tests/test_multiturn_voice.py
```

## Deployment

### Docker

```bash
docker build -t gemini-live-voice-agent .
docker run -p 7860:7860 --env-file .env gemini-live-voice-agent
```

## Troubleshooting

### No audio heard on call (incorrectPayload error)

Plivo requires specific audio format for `playAudio` events:

```json
{
  "event": "playAudio",
  "media": {
    "contentType": "audio/x-mulaw",
    "sampleRate": 8000,
    "payload": "base64..."
  }
}
```

Common mistakes:
- Using `"contentType": "audio/x-mulaw;rate=8000"` (wrong - rate must be separate)
- Missing `sampleRate` field
- Sending chunks larger than 160 bytes (20ms at 8kHz)

### Call drops after agent's first response

This happens when Gemini's `session.receive()` iterator exits after `turn_complete`. The session is still alive - you must loop and call `receive()` again:

```python
while self._running:
    async for response in session.receive():
        # Handle response...
    # Iterator exited (turn complete) - loop to continue listening
```

### No audio from Gemini

- Verify `GEMINI_API_KEY` is correct
- Check that the model supports audio output (use `gemini-2.5-flash-native-audio-preview-12-2025`)
- Review server logs for connection errors

### WebSocket disconnects immediately

- Ensure ngrok is running and URL in `.env` matches
- Check Plivo credentials are correct
- Verify the server is accessible from the internet

### Audio quality issues

- 8kHz telephony is lower quality than Gemini's native 24kHz output
- This is expected - audio is downsampled for phone compatibility

## Implementation Notes

For developers extending this code:

1. **Gemini API uses keyword arguments**:
   ```python
   await session.send_client_content(turns=..., turn_complete=True)
   await session.send_tool_response(function_responses=[...])
   ```

2. **FunctionResponse requires `id`**:
   ```python
   types.FunctionResponse(id=fc.id, name=fc.name, response={...})
   ```

3. **Audio conversion** uses numpy-based μ-law encoding/decoding.

4. **Plivo audio chunks** must be 160 bytes (20ms at 8kHz μ-law) - larger chunks cause `incorrectPayload` errors.
