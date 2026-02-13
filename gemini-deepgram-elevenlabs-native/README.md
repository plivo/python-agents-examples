# Gemini + Deepgram + ElevenLabs Voice Agent (Native)

A voice agent that uses Google Gemini for conversation, Deepgram for speech-to-text, and ElevenLabs for text-to-speech. This implementation uses direct API integration without any orchestration frameworks.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Plivo     │────▶│  Deepgram   │────▶│   Gemini    │
│  (Phone)    │     │   (STT)     │     │   (LLM)     │
│             │◀────│             │◀────│             │
└─────────────┘     └─────────────┘     └─────────────┘
       │                                       │
       │            ┌─────────────┐            │
       └───────────▶│ ElevenLabs  │◀───────────┘
                    │   (TTS)     │
                    └─────────────┘
```

**Audio Flow:**
1. Caller speaks → Plivo captures audio (u-law 8kHz)
2. Audio converted to PCM → sent to Deepgram for transcription
3. Transcript sent to Gemini for response generation
4. Response text sent to ElevenLabs for speech synthesis
5. TTS audio (PCM 24kHz) converted to u-law 8kHz → sent back to caller

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- [ngrok](https://ngrok.com/) for local development
- API keys for:
  - [Plivo](https://www.plivo.com/) - Telephony
  - [Google AI Studio](https://aistudio.google.com/) - Gemini API
  - [Deepgram](https://deepgram.com/) - Speech-to-text
  - [ElevenLabs](https://elevenlabs.io/) - Text-to-speech

## Setup

1. **Clone and navigate to the project:**
   ```bash
   cd gemini-deepgram-elevenlabs-native
   ```

2. **Create environment file:**
   ```bash
   cp .env.example .env
   ```

3. **Configure your `.env` file** with your API keys and Plivo phone number.

4. **Install dependencies:**
   ```bash
   uv sync
   ```

5. **Start ngrok** (in a separate terminal):
   ```bash
   ngrok http 8000
   ```

6. **Update `PUBLIC_URL`** in your `.env` with the ngrok HTTPS URL.

7. **Run the server:**
   ```bash
   uv run python server.py
   ```

8. **Call your Plivo phone number** to test the voice agent.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `PLIVO_AUTH_ID` | Plivo account auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo account auth token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `GEMINI_API_KEY` | Google AI API key | Required |
| `GEMINI_MODEL` | Gemini model name | `gemini-2.0-flash` |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required |
| `DEEPGRAM_MODEL` | Deepgram model | `nova-2-phonecall` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Required |
| `ELEVENLABS_VOICE_ID` | ElevenLabs voice ID | Rachel |
| `ELEVENLABS_MODEL` | ElevenLabs model | `eleven_flash_v2_5` |
| `SERVER_PORT` | Server port | `8000` |

## How It Works

### Components

- **server.py**: FastAPI server handling Plivo webhooks and WebSocket connections
- **agent.py**: Voice agent implementation with STT, LLM, and TTS integration

### Audio Format Conversion

- **Plivo**: u-law 8kHz mono (telephony standard)
- **Deepgram**: Linear PCM 16-bit 8kHz
- **ElevenLabs**: Linear PCM 16-bit 24kHz

The agent handles all necessary conversions between these formats using NumPy and SciPy.

### Conversation Flow

1. Server receives incoming call webhook from Plivo
2. Returns XML instructing Plivo to open bidirectional WebSocket
3. Agent connects to Deepgram WebSocket for real-time STT
4. User speech is transcribed and sent to Gemini
5. Gemini response is synthesized via ElevenLabs
6. TTS audio is streamed back to caller through Plivo

## Customization

### System Prompt

Modify the `DEFAULT_SYSTEM_PROMPT` in `agent.py` or set the `SYSTEM_PROMPT` environment variable.

### Voice Selection

Change `ELEVENLABS_VOICE_ID` in your `.env` file. Browse available voices at [ElevenLabs](https://elevenlabs.io/voice-library).

### STT Model

Change `DEEPGRAM_MODEL` for different accuracy/speed tradeoffs. Options include:
- `nova-2-phonecall` - Optimized for phone audio
- `nova-2` - General purpose
- `nova-2-meeting` - Optimized for meetings

## Troubleshooting

### No audio from agent
- Check ElevenLabs API key and voice ID
- Verify audio format conversion is working
- Check server logs for TTS errors

### Poor transcription quality
- Ensure using `nova-2-phonecall` model for phone audio
- Check audio is being received from Plivo (check logs)

### Webhook not receiving calls
- Verify ngrok is running and URL is correct in `.env`
- Check Plivo console for webhook configuration
- Ensure `PUBLIC_URL` uses HTTPS
