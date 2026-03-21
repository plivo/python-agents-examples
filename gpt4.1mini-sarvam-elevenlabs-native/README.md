# GPT-4.1 Mini + Sarvam STT + ElevenLabs TTS — Native Voice Agent

Native orchestration (raw WebSockets + asyncio, no framework). Plivo μ-law 8kHz (160B/20ms chunks) feeds into Sarvam.ai `saaras:v3` streaming STT over WebSocket at 8kHz PCM (hosted in India, no resample needed) and local Silero VAD (ONNX v5, 512 samples/32ms at 16kHz). LLM is OpenAI `gpt-4.1-mini` via HTTP (non-streaming, 300 max tokens, 5 tool functions). TTS is ElevenLabs `eleven_flash_v2_5` via WebSocket `stream-input` — text sent in sentence chunks, audio returned as PCM16 24kHz, resampled to 8kHz μ-law for Plivo. VAD start threshold set to 0.85 to reject echo (agent playback registers 0.51–0.74, real speech 0.93+), end threshold 0.35, 500ms min silence. Barge-in cancels in-flight LLM/TTS tasks, drains the send queue, and sends `clearAudio` to Plivo.

## Pipeline Architecture

```
┌──────────┐       ┌────────────┐       ┌─────────────────────────────────────────────────────┐
│  Phone   │──────▶│   Plivo    │──────▶│                  Voice Agent                        │
│  (PSTN)  │◀──────│  Gateway   │◀──────│                                                     │
└──────────┘       └────────────┘       │  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
                    μ-law 8kHz          │  │ Silero   │  │ Sarvam.ai │  │  OpenAI GPT-4.1  │  │
                    bidirectional       │  │ VAD      │  │ STT (WS)  │  │  mini (HTTP)     │  │
                    WebSocket           │  │ (local)  │  │           │  │                  │  │
                                        │  └────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
                                        │       │              │                 │             │
                                        │       │    ┌─────────▼─────────────────▼──────┐     │
                                        │       │    │       Turn State Machine         │     │
                                        │       │    │  speech_start → barge-in/cancel  │     │
                                        │       │    │  speech_end   → commit turn      │     │
                                        │       │    └─────────┬────────────────────────┘     │
                                        │       │              │                              │
                                        │  ┌────▼──────────────▼──────────────────────────┐   │
                                        │  │          ElevenLabs TTS (WebSocket)           │   │
                                        │  │  sentence-chunked input → streaming PCM out  │   │
                                        │  └──────────────────────────────────────────────┘   │
                                        └─────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Service | Protocol | Model / Engine | Region |
|-----------|---------|----------|----------------|--------|
| **Telephony** | Plivo | WebSocket (μ-law 8kHz) | — | US (Plivo PSTN gateway) |
| **STT** | Sarvam.ai | WebSocket streaming | `saaras:v3` | India (`api.sarvam.ai`) |
| **LLM** | OpenAI | HTTP (non-streaming) | `gpt-4.1-mini` (300 max tokens) | US (`api.openai.com`) |
| **TTS** | ElevenLabs | WebSocket streaming | `eleven_flash_v2_5` | US (`api.elevenlabs.io`) |
| **VAD** | Silero | Local (ONNX) | `silero_vad` v5 | Local (no network) |
| **Turn detection** | Custom | — | Debounced frame counter | Local |

## Audio Pipeline

Every audio frame passes through multiple sample-rate conversions. The pipeline is optimized for telephony (8kHz μ-law) while each AI service operates at its native rate:

| Hop | Format | Sample Rate | Frame Size | Notes |
|-----|--------|-------------|------------|-------|
| Plivo → Agent | μ-law (base64) | 8 kHz | 160 bytes (20ms) | G.711 decode table, no codec library |
| Agent → Sarvam STT | PCM16 (base64 JSON) | 8 kHz | 320 bytes | Sarvam accepts 8kHz directly, no resample needed |
| Agent → Silero VAD | float32 | 16 kHz | 512 samples (32ms) | Resampled via scipy; ~2 Plivo frames per VAD frame |
| Agent → ElevenLabs TTS | text (JSON) | — | sentence chunks | WebSocket `stream-input` endpoint |
| ElevenLabs → Agent | PCM16 (base64 JSON) | 24 kHz | variable | Decoded and resampled to 8kHz |
| Agent → Plivo | μ-law (base64) | 8 kHz | 160 bytes (20ms) | Chunked exactly to 20ms for smooth playback |

### Audio Conversion Functions

```
plivo_to_sarvam_streaming()  — μ-law 8kHz → PCM16 8kHz (decode only, no resample)
plivo_to_vad()               — μ-law 8kHz → float32 16kHz (decode + resample)
elevenlabs_to_plivo()        — PCM16 24kHz → μ-law 8kHz (resample + encode)
```

## Turn Detection & VAD

Turn detection uses **client-side Silero VAD** running locally as an ONNX model. This gives sub-millisecond inference and full control over turn boundaries, independent of any API's server-side VAD.

### VAD Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `VAD_START_THRESHOLD` | **0.85** | Speech probability to trigger `speech_start`. Set high to reject echo of the agent's own audio playing back through the earpiece (echo registers at 0.5–0.74; real speech at 0.93+) |
| `VAD_END_THRESHOLD` | **0.35** | Probability below this is classified as silence |
| `VAD_MIN_SILENCE_MS` | **500** | Minimum consecutive silence before triggering `speech_end`. Prevents premature turn commits during natural pauses |
| `VAD_CHUNK_SAMPLES` | **512** | 32ms frames at 16kHz. Silero requires exactly 512 samples |
| `VAD_PRE_SPEECH_PAD_MS` | **150** | Audio context retained before speech start |

### Echo Rejection

The 0.85 start threshold was tuned empirically from live call data:

- **Agent echo** (own TTS playing back through earpiece): VAD probability 0.51–0.74
- **Real user speech**: VAD probability 0.93–0.99

At the default 0.5 threshold, echo triggers false barge-ins that cut off the agent mid-sentence, causing the caller to hear silence. The 0.85 threshold eliminates echo triggers while reliably detecting real speech.

### Turn State Machine

```
IDLE ──[VAD speech_start]──▶ SPEAKING
  │                              │
  │                     [VAD speech_end]
  │                              │
  │                              ▼
  │                      CHECK TRANSCRIPT
  │                        ╱          ╲
  │               (ready)╱            ╲(pending)
  │                    ╱                ╲
  │                   ▼                  ▼
  │            COMMIT TURN         WAIT FOR STT
  │                   │                  │
  │                   ▼           [transcript arrives]
  │             LLM → TTS               │
  │                   │                  ▼
  │                   ▼            COMMIT TURN
  │              IS_PLAYING              │
  │                   │                  ▼
  │          [playedStream]        LLM → TTS
  │                   │
  └───────────────────┘
```

### Barge-in Handling

When `speech_start` fires during `IS_PLAYING`:
1. Cancel in-flight TTS task (if streaming)
2. Cancel in-flight LLM turn task (if waiting for response)
3. Drain the send queue (discard buffered audio)
4. Send `clearAudio` event to Plivo (stops playback immediately)
5. Clear STT transcript buffer (discard echo transcription)
6. Reset `_is_playing = False`

## STT: Sarvam.ai Streaming

Sarvam provides **real-time streaming STT** optimized for Indian English accents, connected via a persistent WebSocket:

- **Endpoint**: `wss://api.sarvam.ai/speech-to-text/ws`
- **Model**: `saaras:v3`
- **Input**: PCM16 at 8kHz (Sarvam accepts 8kHz natively — no upsampling needed)
- **Language**: `en-IN` (configurable via `SARVAM_STT_LANGUAGE`)
- **Latency**: Transcripts arrive via `data` events, typically within 400–500ms of speech end

Audio is forwarded to STT continuously — even during agent playback. This ensures the STT has already processed user speech by the time VAD fires `speech_end`. On barge-in, the transcript buffer is cleared to discard echo.

## LLM: OpenAI GPT-4.1 mini

The LLM is called via standard **HTTP chat completions** (non-streaming):

- **Endpoint**: `https://api.openai.com/v1/chat/completions`
- **Model**: `gpt-4.1-mini`
- **Max tokens**: 300 (keeps responses concise for voice)
- **Function calling**: 5 tools (order status, SMS, callback, transfer, end call)
- **Observed latency**: 770ms–2000ms depending on conversation length

The full conversation history is sent each turn. Tool calls trigger a follow-up completion to generate the spoken response from tool results.

## TTS: ElevenLabs WebSocket Streaming

TTS uses the ElevenLabs **WebSocket streaming API** (`stream-input`) instead of the HTTP endpoint. This enables sentence-level progressive synthesis — audio generation starts before the full text is delivered:

- **Endpoint**: `wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input`
- **Model**: `eleven_flash_v2_5` (optimized for low latency)
- **Voice**: Rachel (`21m00Tcm4TlvDq8ikWAM`)
- **Output format**: PCM16 at 24kHz
- **Voice settings**: stability=0.5, similarity_boost=0.8

### WebSocket Protocol Flow

1. **BOS** (beginning of stream): Send initial config with API key and voice settings
2. **Text chunks**: Send each sentence as `{"text": "sentence. "}` — audio starts generating immediately
3. **EOS** (end of stream): Send `{"text": ""}` to flush remaining audio
4. **Receive**: Audio arrives as base64-encoded PCM chunks in JSON messages

Text is split at sentence boundaries (`[.!?]` followed by whitespace) so ElevenLabs can begin synthesis on the first sentence while remaining sentences are still being sent.

### Observed Latency

| Metric | Value | Notes |
|--------|-------|-------|
| TTS TTFB | 280–380ms | Time from WebSocket connect to first audio chunk |
| TTS total | 1.1–1.7s | Full synthesis including all chunks |
| TTFS (time to first speech) | 1.5–2.3s | End-to-end: speech_end → audio playing (includes LLM + TTS) |

The LLM (non-streaming) is the latency bottleneck. TTS WebSocket TTFB is consistently under 400ms.

## Concurrent Task Architecture

The agent runs three persistent asyncio tasks plus on-demand turn tasks:

| Task | Name | Role |
|------|------|------|
| `plivo_rx` | `_receive_from_plivo` | Decode Plivo audio → forward to STT + VAD → detect turns |
| `plivo_tx` | `_send_to_plivo` | Drain send queue → chunk to 160 bytes → send `playAudio` |
| `stt_watch` | `_watch_transcripts` | Convergence gate: if VAD ended but STT hasn't delivered, wait and commit |
| `turn_N` | `_process_text_turn` | On-demand: LLM completion → TTS synthesis → queue audio |

Tasks coordinate via `asyncio.Queue` (send queue), `asyncio.Event` flags (`_is_playing`, `_running`), and `asyncio.Lock` (`_turn_lock` serializes turns).

## Features

- **Inbound & Outbound**: Full support for receiving and placing calls
- **Function Calling**: Order status, SMS, callbacks, transfers, call control
- **Barge-in**: Sub-second interruption with audio clearing
- **Playback Checkpoints**: Plivo `checkpoint` events track when audio finishes playing
- **Structured Logging**: Per-turn metrics (LLM latency, TTS TTFB, audio duration, TTFS)
- **OpenTelemetry**: Optional tracing spans for LLM and TTS operations
- **Auto-Configuration**: Server configures Plivo webhooks on startup

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- Sarvam.ai API key
- ElevenLabs API key
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

```bash
cd gpt4.1mini-sarvam-elevenlabs-native
uv sync
cp .env.example .env
# Edit .env with your credentials

# Start ngrok
ngrok http 8000

# Run inbound server
uv run python -m inbound.server

# Or outbound server
uv run python -m outbound.server
```

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `OPENAI_MODEL` | LLM model | `gpt-4.1-mini` |
| `SARVAM_API_KEY` | Sarvam.ai API key | Required |
| `SARVAM_STT_LANGUAGE` | STT language code | `en-IN` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Required |
| `ELEVENLABS_VOICE_ID` | Voice ID | `21m00Tcm4TlvDq8ikWAM` (Rachel) |
| `ELEVENLABS_MODEL_ID` | TTS model | `eleven_flash_v2_5` |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `LOG_LEVEL` | Logging verbosity | `normal` (`verbose` / `quiet`) |

## Testing

```bash
# Unit tests (offline)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration (starts server, needs API keys)
uv run pytest tests/test_integration.py -v -k "local"

# API integration (ElevenLabs WebSocket TTS, Sarvam STT, OpenAI)
uv run pytest tests/test_integration.py -v -k "not unit and not local"

# Live phone call tests (needs PLIVO_TEST_NUMBER)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
```

## Deployment

```bash
docker build -t gpt4mini-sarvam-elevenlabs .
docker run -p 8000:8000 --env-file .env gpt4mini-sarvam-elevenlabs
```
