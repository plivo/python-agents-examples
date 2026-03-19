# xAI Grok TTS Voice Showcase

Voice agent showcasing xAI's Grok TTS across 5 voices, 7 scenarios, multilingual auto-detection, and expressive speech tags — all over real phone calls using Pipecat + Plivo telephony.

## Features

- **5 Distinct Voices**: Ara, Eve, Rex, Sal, Leo — each with a tailored conversation scenario
- **Multilingual Auto-Detect**: Switch languages mid-conversation — xAI TTS detects and renders automatically
- **Expressive Speech Tags**: Inline tags (`[laugh]`, `[pause]`, `[gasp]`) and wrapping tags (`<whisper>`, `<shout>`, `<sing>`)
- **Live Waveform Replay**: Web-based audio visualizer with real-time waveform + synced transcript
- **Pipeline Metrics**: Per-turn STT/LLM/TTS TTFB breakdown with dashboard
- **Call Recording**: Automatic Plivo call recording with playback in dashboard

## Architecture

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐
│  Phone  │────▶│   Plivo     │────▶│  Pipecat    │
│  Call   │◀────│  (PSTN)     │◀────│  Pipeline   │
└─────────┘     └─────────────┘     └──────┬──────┘
                                           │
                    μ-law 8kHz WebSocket    │
                                           ▼
                              ┌──────────────────────┐
                              │   Deepgram STT       │
                              │   Google Gemini LLM   │
                              │   xAI Grok TTS       │
                              │   Smart Turn v3      │
                              └──────────────────────┘
```

## Voice Scenarios

| Voice | Scenario | Description |
|-------|----------|-------------|
| **Ara** | Dramatic Storytelling | Captivating narrator with vivid emotion and pacing |
| **Eve** | Empathetic Counselor | Warm, understanding listener who mirrors emotions |
| **Rex** | Excited Travel Guide | Enthusiastic guide with infectious energy |
| **Sal** | Casual Friend Chat | Witty, laid-back friend with natural conversation |
| **Leo** | News Anchor | Authoritative delivery with varied pacing |
| **Ara** | Multilingual Auto-Detect | Switches languages on request — Hindi, Spanish, Japanese, etc. |
| **Ara** | Expressive Speech Tags | Uses `[laugh]`, `<whisper>`, `<sing>` for theatrical delivery |

## Prerequisites

- Python 3.11+
- [ngrok](https://ngrok.com/) (for local development)
- API keys: [xAI](https://console.x.ai/), [Deepgram](https://deepgram.com/), [Google AI](https://aistudio.google.com/)
- [Plivo](https://www.plivo.com/) account with a phone number

## Quick Start

### 1. Install dependencies

```bash
pip install -e .
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
DEEPGRAM_API_KEY=your_deepgram_key
GOOGLE_API_KEY=your_google_key
XAI_TTS_API_KEY=your_xai_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_FROM_NUMBER=14155551234
PLIVO_TO_NUMBER=17375551234
```

### 3. Start ngrok

```bash
ngrok http 8000
```

### 4. Select voice and scenario

Edit `.env`:

```bash
XAI_VOICE=Ara
SCENARIO=storytelling
```

### 5. Run the bot

```bash
python bot.py -t plivo -x your-ngrok-host.ngrok-free.dev --port 8000
```

### 6. Make the call

```bash
source .env
curl -X POST "https://api.plivo.com/v1/Account/${PLIVO_AUTH_ID}/Call/" \
  -u "${PLIVO_AUTH_ID}:${PLIVO_AUTH_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"from\": \"${PLIVO_FROM_NUMBER}\", \"to\": \"${PLIVO_TO_NUMBER}\", \"answer_url\": \"https://your-ngrok-host.ngrok-free.dev/\"}"
```

### 7. View the dashboard

```bash
python dashboard.py
# Open http://localhost:8080
```

Click any session to play the recording with a live waveform visualizer and synced transcript. Use the replay page for clean screen recordings.

## Project Structure

```
xai-voice-showcase/
├── bot.py                  # Pipecat voice agent with voice/scenario switching
├── services/
│   └── xai_tts.py          # Custom xAI TTS service (POST /v1/tts, MP3→PCM)
├── metrics_observer.py     # Per-turn pipeline metrics (STT/LLM/TTS TTFB)
├── dashboard.py            # FastAPI dashboard + replay server
├── server.py               # Pipecat runner placeholder
├── static/
│   ├── dashboard.html      # Dashboard with inline waveform player
│   └── replay.html         # Standalone replay page for screen recording
├── scripts/
│   └── generate_test_data.py
├── data/
│   └── sessions/           # Session JSON files (gitignored)
├── .env.example            # Environment variable template
├── pyproject.toml          # Project dependencies
└── README.md
```

## How It Works

1. **Bot starts** → Pipecat creates a FastAPI server with a WebSocket endpoint
2. **Plivo call** → Plivo hits the webhook, returns XML to open a bidirectional audio stream
3. **Pipeline** → Audio flows through: Deepgram STT → Google Gemini LLM → xAI TTS
4. **Smart Turn v3** → Detects when the user is done speaking (not just silence)
5. **xAI TTS** → Sends text to `POST /v1/tts`, decodes MP3→PCM, streams to caller
6. **Metrics** → Observer captures per-turn latency waterfall and transcriptions
7. **Recording** → Plivo records the call, URL is saved to session JSON

## xAI TTS Details

### Custom Service (`services/xai_tts.py`)

xAI's TTS API returns MP3 (not streaming PCM), so the service:
1. POSTs text to `https://api.x.ai/v1/tts`
2. Receives full MP3 response
3. Decodes to PCM via pydub
4. Yields audio frames to the pipeline

### Speech Tags

The expressive scenario instructs the LLM to include xAI speech tags:

```
Inline:   [laugh] [sigh] [gasp] [pause] [clears throat]
Wrapping: <whisper>text</whisper> <shout>text</shout> <sing>text</sing>
```

### Multilingual Auto-Detection

xAI TTS automatically detects the language of the input text. The multilingual scenario instructs the LLM to respond in whatever language the user requests — xAI handles the rest.

Supported: English, Chinese, Hindi, Arabic, Japanese, Korean, Portuguese, Spanish, and more.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `XAI_TTS_API_KEY` | xAI API key | Required |
| `DEEPGRAM_API_KEY` | Deepgram API key | Required |
| `GOOGLE_API_KEY` | Google Gemini API key | Required |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_FROM_NUMBER` | Plivo caller number | Required |
| `PLIVO_TO_NUMBER` | Number to call | Required |
| `XAI_VOICE` | Voice name | `Ara` |
| `SCENARIO` | Scenario key | `storytelling` |
| `SMART_TURN_STOP_SECS` | Turn detection timeout | `3.0` |
| `DASHBOARD_PORT` | Dashboard server port | `8080` |

### Available Voices

| Name | Tone |
|------|------|
| Ara | Warm, dramatic |
| Eve | Empathetic, caring |
| Rex | Energetic, enthusiastic |
| Sal | Casual, witty |
| Leo | Authoritative, measured |

## Dashboard

The dashboard at `http://localhost:8080` provides:

- **Session tab**: Waveform audio player with real-time visualization + synced transcript
- **Compare tab**: Side-by-side TTS TTFB and pipeline breakdown charts
- **Voices tab**: Aggregate metrics grouped by voice

Each session also has a standalone **Replay page** (`/replay/{session_id}`) designed for clean screen recordings.

## Troubleshooting

### No audio on call
- xAI TTS is non-streaming (full MP3 response) — expect 1-3s TTS TTFB
- Check xAI API key has credits at https://console.x.ai

### Bot gets interrupted immediately
- Phone line noise triggers VAD — the bot uses `start_secs=0.4` and `min_volume=0.8` to reduce false triggers
- A 2-second delay before greeting helps the line settle

### SSL errors
- Set `SSL_CERT_FILE` before running: `SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")`
