# Gemini Live Pipecat Voice Agent

Real-time voice agent using Pipecat framework with Google Gemini Live API for speech-to-speech conversations with Plivo telephony.

## Features

- **Pipecat Framework**: Modular pipeline architecture for voice AI
- **Speech-to-Speech**: Native audio using Gemini Live API
- **Multi-turn Conversations**: Maintains context across conversation turns
- **Voice Activity Detection**: Silero VAD for natural turn-taking
- **Low Latency**: Real-time bidirectional audio streaming

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Google AI API key with Gemini Live API access
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

### 1. Install dependencies

```bash
cd gemini-live-pipecat
uv sync
```

Or with pip:

```bash
pip install -e .
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
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Start ngrok

```bash
ngrok http 8000
```

Copy the ngrok URL to `PUBLIC_URL` in your `.env` file.

### 4. Configure Plivo webhook

Set your Plivo phone number's Answer URL to:
```
https://your-ngrok-url.ngrok-free.app/answer
```

### 5. Run the server

```bash
uv run python voice_agent.py
```

### 6. Make a test call

Call your Plivo phone number and start talking to the agent.

## Project Structure

```
gemini-live-pipecat/
├── voice_agent.py      # Main application with Pipecat pipeline
├── pyproject.toml      # Project dependencies
├── .env.example         # Environment variable template
└── README.md           # This file
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
                                    │  Pipecat    │
                                    │  Pipeline   │
                                    │             │
                                    │ ┌─────────┐ │
                                    │ │ Gemini  │ │
                                    │ │  Live   │ │
                                    │ └─────────┘ │
                                    └─────────────┘
```

1. **Incoming Call**: Plivo receives call and hits `/answer` webhook
2. **WebSocket Setup**: Server returns XML to establish bidirectional stream
3. **Audio Streaming**: Plivo streams μ-law 8kHz audio via WebSocket
4. **Pipecat Pipeline**: Audio flows through the pipeline with VAD and context management
5. **Gemini Processing**: Gemini Live processes speech and generates response
6. **Response Streaming**: Audio is streamed back through Pipecat to Plivo

## Pipecat Pipeline Architecture

The pipeline uses Pipecat's modular architecture with Gemini Multimodal Live:

```python
Pipeline([
    transport.input(),   # Receive audio from Plivo
    llm,                 # Gemini Multimodal Live (speech-to-speech)
    transport.output(),  # Send audio to Plivo
])
```

Since Gemini Multimodal Live handles both speech recognition and synthesis natively,
the pipeline is simpler than traditional STT → LLM → TTS architectures.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PUBLIC_URL` | Public URL for webhooks (ngrok) | Required |
| `SERVER_PORT` | Server port | `8000` |
| `GEMINI_MODEL` | Gemini model name | `models/gemini-2.5-flash-native-audio-preview-12-2025` |
| `GEMINI_VOICE` | Gemini voice name | `Puck` |
| `SYSTEM_PROMPT` | Custom system prompt | Default assistant prompt |

### Available Voices

Aoede, Charon, Fenrir, Kore, Puck

## Comparison with gemini-live-native

| Feature | gemini-live-pipecat | gemini-live-native |
|---------|--------------------|--------------------|
| Framework | Pipecat | None (direct API) |
| Code complexity | Lower | Higher |
| Customization | Via Pipecat processors | Full control |
| Audio conversion | Handled by Pipecat | Manual implementation |
| VAD | Silero via Pipecat | Custom or none |

Use **gemini-live-pipecat** when you want:
- Quick setup with less code
- Built-in VAD and audio handling
- Easy pipeline customization
- Integration with other Pipecat services

Use **gemini-live-native** when you need:
- Maximum control over audio processing
- Custom audio format handling
- Minimal dependencies
- Production-optimized performance

## Troubleshooting

### No audio heard on call

- Verify `GEMINI_API_KEY` is correct
- Check that Plivo credentials are set
- Ensure ngrok is running and URL matches `PUBLIC_URL`

### WebSocket disconnects immediately

- Check Plivo webhook configuration
- Verify ngrok tunnel is active
- Review server logs for errors

### Agent doesn't respond

- Verify Gemini Live API access is enabled for your API key
- Check the model name is correct
- Review logs for Gemini API errors

## License

MIT
