# Gemini Live Voice Agent with Silero VAD Turn Detection

Real-time voice agent using Google Gemini Live API for speech-to-speech conversations with Plivo telephony and **Silero VAD (ONNX)**-based voice activity detection for turn detection and barge-in interruption handling.

## Features

- **Speech-to-Speech**: Native audio using Gemini Live API (no separate STT/TTS)
- **Two-Layer VAD**: Client-side Silero VAD + server-side Gemini AutomaticActivityDetection
- **ML-Based VAD**: Silero VAD uses an ONNX neural network for best-in-class speech detection accuracy
- **Barge-In Interruption**: User can interrupt the agent mid-speech for natural conversation flow
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Function Calling**: Order status, SMS, callbacks, transfers, and call control
- **Auto-Configuration**: Automatically configures Plivo webhooks on startup
- **Low Latency**: Real-time bidirectional audio streaming

## How Turn Detection Works

This agent uses a **two-layer VAD architecture** for responsive turn-taking:

### Layer 1 — Client-Side VAD (Silero ONNX)
- ML-based neural network VAD running via ONNX Runtime (no PyTorch needed)
- Buffers two Plivo frames (~40ms) to form one 256-sample Silero frame (~32ms)
- Returns a confidence score (0.0-1.0) for each frame
- When confidence exceeds threshold while the agent is talking, triggers instant interruption
- Drains audio queue and sends Plivo `clearAudio` to stop playback immediately
- Model automatically downloaded from GitHub on first run and cached locally

### Layer 2 — Server-Side VAD (Gemini AutomaticActivityDetection)
- Configured with HIGH sensitivity for speech start/end detection
- `prefix_padding_ms=100` captures speech onset
- `silence_duration_ms=500` for natural pause handling
- `START_OF_ACTIVITY_INTERRUPTS` enables server-side barge-in
- `TURN_INCLUDES_ONLY_ACTIVITY` filters non-speech audio

When the user speaks while the agent is talking:
1. Client-side Silero VAD detects speech (~64ms confirmation)
2. Agent audio queue is drained, Plivo `clearAudio` is sent
3. Server-side VAD confirms interruption via `server_content.interrupted`
4. Gemini generates a new response based on what the user said

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Google AI API key with Gemini Live API access
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd gemini-live-native
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
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
ngrok http 8000
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 4. Run the server

```bash
uv run python server.py
```

The server will:
1. Download Silero VAD model on first run (cached at `~/.cache/silero-vad/`)
2. Start on port 8000
3. Auto-configure Plivo webhooks for your phone number
4. Display "Ready! Call +1234567890 to test"

### 5. Make a test call

Call your Plivo phone number and start talking to the agent. Try interrupting the agent mid-sentence to test barge-in.

## Project Structure

```
gemini-live-native/
├── agent.py            # Voice agent with Gemini Live API + Silero VAD
├── server.py           # FastAPI server with Plivo webhooks
├── tests/              # Integration and voice tests
├── pyproject.toml      # Project dependencies
├── .env.example         # Environment variable template
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
                                    │             │
                                    │  Silero VAD │
                                    │  (ONNX)     │
                                    └─────────────┘
```

1. **Incoming Call**: Plivo receives call and hits `/answer` webhook
2. **WebSocket Setup**: Server returns XML to establish bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **VAD Processing**: Silero VAD analyzes buffered 8kHz PCM frames for speech
5. **Format Conversion**: Agent converts μ-law 8kHz → PCM16 16kHz for Gemini
6. **AI Processing**: Gemini Live processes speech and generates response
7. **Response Streaming**: Agent converts PCM16 24kHz → μ-law 8kHz for Plivo
8. **Interruption**: If user speaks during playback, audio is cleared instantly

## Audio Formats

| Stage | Format | Sample Rate |
|-------|--------|-------------|
| Plivo → Agent | μ-law | 8 kHz |
| Agent VAD (Silero) | PCM16 float32 | 8 kHz |
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
| `SERVER_PORT` | Server port | `8000` |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.5-flash-native-audio-preview-12-2025` |
| `GEMINI_VOICE` | Voice name | `Kore` |
| `VAD_CONFIDENCE` | Silero confidence threshold (0.0-1.0) | `0.5` |

### VAD Tuning

- **VAD_CONFIDENCE**: Speech detection threshold. Lower values (e.g., 0.3) detect speech more aggressively, higher values (e.g., 0.7) require stronger speech signals. Default `0.5` works well for telephony.

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
docker build -t gemini-live-voice-agent-silero .
docker run -p 8000:8000 --env-file .env gemini-live-voice-agent-silero
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

### Silero model download fails

- The model is downloaded from GitHub on first run
- If behind a firewall, manually download `silero_vad.onnx` and place at `~/.cache/silero-vad/silero_vad.onnx`
- URL: https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

### Interruption not working

- Check logs for "VAD: user speech started" and "Interruption triggered" messages
- Try lowering `VAD_CONFIDENCE` if speech isn't being detected
- Ensure `stream_id` is being passed (needed for Plivo `clearAudio`)

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

5. **Silero VAD frame buffering**: Silero needs 256 samples (512 bytes PCM16) at 8kHz per frame, but Plivo sends 160 samples (320 bytes PCM16) per packet. The agent buffers two Plivo frames to fill one Silero frame.

6. **ONNX Runtime**: Silero VAD runs as a lightweight ONNX model with LSTM state, avoiding the need for PyTorch. The model is ~2MB and cached locally.
