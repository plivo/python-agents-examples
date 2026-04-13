# GPT-5.4 Mini + Speechmatics STT + ElevenLabs TTS -- Native Voice Agent

Native orchestration (raw WebSockets + asyncio, no framework). Plivo mu-law 8kHz (160B/20ms chunks) feeds Speechmatics Voice Agent API (`adaptive` profile) over WebSocket at 8kHz PCM s16le binary frames (no resample needed) and local Silero VAD (ONNX v5, 512 samples/32ms at 16kHz). LLM is OpenAI `gpt-5.4-mini` via HTTP SSE streaming (sentence-buffered output for progressive TTS). TTS is ElevenLabs `eleven_flash_v2_5` via WebSocket `stream-input` -- text sent in sentence chunks, audio returned as PCM16 24kHz, resampled to 8kHz mu-law for Plivo. VAD start threshold 0.85 to reject echo (agent playback registers 0.51-0.74, real speech 0.93+), end threshold 0.35, 500ms min silence. Barge-in cancels in-flight LLM/TTS tasks, drains the send queue, and sends `clearAudio` to Plivo.

## Architecture

```
┌──────────┐       ┌────────────┐       ┌──────────────────────────────────────────────────────────┐
│  Phone   │──────▶│   Plivo    │──────▶│                    Voice Agent                           │
│  (PSTN)  │◀──────│  Gateway   │◀──────│                                                          │
└──────────┘       └────────────┘       │  ┌──────────┐  ┌────────────────┐  ┌─────────────────┐  │
                    μ-law 8kHz          │  │ Silero   │  │ Speechmatics   │  │ OpenAI GPT-5.4  │  │
                    bidirectional       │  │ VAD      │  │ Voice Agent    │  │  Mini (HTTP SSE)│  │
                    WebSocket           │  │ (local)  │  │ (WebSocket)    │  │                 │  │
                                        │  └────┬─────┘  └──────┬─────────┘  └────────┬────────┘  │
                                        │       │               │                     │           │
                                        │       │    ┌──────────▼─────────────────────▼──────┐    │
                                        │       │    │       Turn State Machine              │    │
                                        │       │    │  speech_start → barge-in/cancel       │    │
                                        │       │    │  EndOfTurn    → commit turn           │    │
                                        │       │    └──────────┬────────────────────────────┘    │
                                        │       │               │                                 │
                                        │  ┌────▼───────────────▼─────────────────────────────┐   │
                                        │  │          ElevenLabs TTS (WebSocket)               │   │
                                        │  │  sentence-chunked input → streaming PCM out       │   │
                                        │  └──────────────────────────────────────────────────┘   │
                                        └──────────────────────────────────────────────────────────┘
```

### Component Summary

| Component | Service | Protocol | Model / Engine | Region |
|-----------|---------|----------|----------------|--------|
| **Telephony** | Plivo | WebSocket (μ-law 8kHz) | — | US (Plivo PSTN gateway) |
| **STT** | Speechmatics | WebSocket streaming | Voice Agent API `adaptive` profile | EU (`preview.rt.speechmatics.com`) |
| **LLM** | OpenAI | HTTP SSE streaming | `gpt-5.4-mini` | US (`api.openai.com`) |
| **TTS** | ElevenLabs | WebSocket streaming | `eleven_flash_v2_5` | US (`api.elevenlabs.io`) |
| **VAD** | Silero | Local (ONNX) | `silero_vad` v5 | Local (no network) |
| **Turn detection** | Custom | — | Debounced frame counter + EndOfTurn | Local |

## Audio Pipeline

| Hop | Format | Sample Rate | Frame Size | Notes |
|-----|--------|-------------|------------|-------|
| Plivo → Agent | μ-law (base64) | 8 kHz | 160 bytes (20ms) | G.711 decode table, no codec library |
| Agent → Speechmatics STT | PCM s16le (binary) | 8 kHz | 320 bytes | Sent as raw binary WebSocket frames; no resample needed |
| Agent → Silero VAD | float32 | 16 kHz | 512 samples (32ms) | Resampled via scipy; ~2 Plivo frames per VAD frame |
| Agent → ElevenLabs TTS | text (JSON) | — | sentence chunks | WebSocket `stream-input` endpoint |
| ElevenLabs → Agent | PCM16 (base64 JSON) | 24 kHz | variable | Decoded and resampled to 8kHz |
| Agent → Plivo | μ-law (base64) | 8 kHz | 160 bytes (20ms) | Chunked exactly to 20ms for smooth playback |

### Audio Conversion Functions

```
plivo_to_speechmatics()  — μ-law 8kHz → PCM16 8kHz (decode only, no resample)
plivo_to_vad()           — μ-law 8kHz → float32 16kHz (decode + resample)
elevenlabs_to_plivo()    — PCM16 24kHz → μ-law 8kHz (resample + encode)
```

## Turn Detection & VAD

Turn detection uses **client-side Silero VAD** running locally as an ONNX model combined with **Speechmatics `EndOfTurn` events**. Client-side VAD gives sub-millisecond inference and full barge-in control; Speechmatics `EndOfTurn` provides an additional signal for when the server-side model determines the speaker has finished.

### VAD Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `VAD_START_THRESHOLD` | **0.85** | Speech probability to trigger `speech_start`. Set high to reject echo of the agent's own audio playing back through the earpiece (echo registers at 0.51–0.74; real speech 0.93+) |
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
  │                  [VAD speech_end or EndOfTurn]
  │                              │
  │                              ▼
  │                      CHECK TRANSCRIPT
  │                        ╱           ╲
  │               (ready)╱             ╲(pending)
  │                    ╱                 ╲
  │                   ▼                   ▼
  │            COMMIT TURN          WAIT FOR STT
  │                   │                   │
  │                   ▼          [transcript arrives]
  │             LLM → TTS                 │
  │                   │                   ▼
  │                   ▼             COMMIT TURN
  │              IS_PLAYING               │
  │                   │                   ▼
  │          [playedStream]         LLM → TTS
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

## STT: Speechmatics Voice Agent API

Speechmatics provides **real-time streaming STT** via the Voice Agent WebSocket API with intelligent end-of-turn detection:

- **Endpoint**: `wss://preview.rt.speechmatics.com/v2/agent/{profile}`
- **Default profile**: `adaptive` (configurable via `SPEECHMATICS_PROFILE`)
- **Input format**: PCM s16le at 8kHz, sent as raw binary WebSocket frames (no JSON wrapping, no resample)
- **Language**: `en` with partials enabled
- **Key events**: `AddPartialSegment`, `AddSegment`, `EndOfTurn`, `SpeechStarted`, `SpeechEnded`, `EndOfTurnPrediction`

Audio is forwarded to STT continuously — even during agent playback. This ensures the STT has already processed user speech by the time VAD fires `speech_end`. On barge-in, the transcript buffer is cleared to discard echo.

### Speechmatics Profiles

| Profile | Description |
|---------|-------------|
| `agile` | Fastest response; lower accuracy; best for highly interactive scenarios |
| `adaptive` | Balanced accuracy and latency; **default** |
| `smart` | Highest accuracy; higher latency; best for complex vocabulary |
| `external` | Bring-your-own model configuration |

### Speechmatics Protocol Flow

1. **Connect**: `wss://preview.rt.speechmatics.com/v2/agent/{profile}` with `Authorization: Bearer {key}`
2. **StartRecognition**: send JSON config with audio format (`raw/pcm_s16le/8000`) and transcription config
3. **RecognitionStarted**: wait for server ACK before forwarding audio
4. **Audio**: send raw PCM binary frames (no encoding, no framing)
5. **EndOfStream**: send `{"message": "EndOfStream", "last_seq_no": 0}` to close gracefully

## LLM: OpenAI GPT-5.4 Mini

The LLM is called via **HTTP streaming chat completions** (SSE) with sentence-level buffering to start TTS before the full response arrives:

- **Endpoint**: `https://api.openai.com/v1/chat/completions`
- **Model**: `gpt-5.4-mini`
- **Streaming**: `stream=True` — tokens are buffered and flushed at sentence boundaries (`[.!?]`)
- **Function calling**: 5 tools (order status, SMS, callback, transfer, end call)

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
- Speechmatics API key
- ElevenLabs API key
- Plivo account with a phone number
- ngrok (for local development)

## Quick Start

```bash
cd gpt5.4-speechmaticsagent-elevenflashv2.5-native
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
| `OPENAI_MODEL` | LLM model | `gpt-5.4-mini` |
| `SPEECHMATICS_API_KEY` | Speechmatics API key | Required |
| `SPEECHMATICS_PROFILE` | Voice Agent profile | `adaptive` |
| `ELEVENLABS_API_KEY` | ElevenLabs API key | Required |
| `ELEVENLABS_VOICE_ID` | Voice ID | `21m00Tcm4TlvDq8ikWAM` (Rachel) |
| `ELEVENLABS_MODEL_ID` | TTS model | `eleven_flash_v2_5` |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Your Plivo phone number | Required |
| `PUBLIC_URL` | Public URL for webhooks | Required |
| `SERVER_PORT` | Server port | `8000` |
| `DEFAULT_COUNTRY_CODE` | Default region for phone parsing | `US` |
| `LOG_LEVEL` | Logging verbosity | `normal` (`verbose` / `quiet`) |
| `LOG_FORMAT` | Log format | `text` (`json`) |
| `LOG_FILE` | File path for JSON log sink | — |
| `REDIS_EVENTS_URL` | Redis URL for Streams sink | — |
| `REDIS_STREAM_KEY` | Redis stream key | `voice-agent:events` |

### VAD Tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `VAD_START_THRESHOLD` | `0.85` | Speech probability to start detection (raise to reject echo) |
| `VAD_END_THRESHOLD` | `0.35` | Probability below which audio is classified as silence |
| `VAD_MIN_SILENCE_MS` | `500` | Minimum silence duration before committing turn |
| `VAD_CHUNK_SAMPLES` | `512` | Silero frame size in samples (32ms at 16kHz) |
| `VAD_PRE_SPEECH_PAD_MS` | `150` | Audio retained before speech start event |

## API Endpoints

### Inbound Server (`python -m inbound.server`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET/POST` | `/answer` | Plivo answer webhook — returns `<Stream>` XML |
| `POST` | `/hangup` | Plivo hangup webhook |
| `POST` | `/fallback` | Fallback webhook for error recovery |
| `GET/POST` | `/hold` | Hold endpoint (keeps call alive during tests) |
| `WS` | `/ws` | Bidirectional audio stream with Plivo |

### Outbound Server (`python -m outbound.server`)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/outbound/call` | Initiate an outbound call |
| `GET/POST` | `/outbound/answer` | Plivo answer webhook for outbound calls |
| `POST` | `/outbound/hangup` | Plivo hangup webhook for outbound calls |
| `WS` | `/outbound/ws` | Bidirectional audio stream for outbound calls |
| `GET` | `/outbound/calls` | List active outbound calls |
| `GET` | `/outbound/calls/{call_id}` | Get status of a specific call |

#### Initiate Outbound Call

```bash
curl -X POST http://localhost:8000/outbound/call \
  -H "Content-Type: application/json" \
  -d '{
    "to": "+14155551234",
    "from": "+18005550100",
    "context": "Follow up on order TF-123456"
  }'
```

## Testing

```bash
# Unit tests (offline — no API keys needed)
uv run pytest tests/test_integration.py -v -k "unit"

# Local integration (starts server subprocess, tests WebSocket flow with real APIs)
uv run pytest tests/test_integration.py -v -k "local"

# API integration (ElevenLabs WebSocket TTS, Speechmatics STT, OpenAI)
uv run pytest tests/test_integration.py -v -k "not unit and not local"

# E2E live call tests (need PLIVO_TEST_NUMBER set in .env)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
uv run pytest tests/test_multiturn_voice.py -v -s
```

## Deployment

```bash
# Build image
docker build -t gpt5.4-speechmaticsagent-elevenflashv2.5 .

# Run with .env file
docker run -p 8000:8000 --env-file .env gpt5.4-speechmaticsagent-elevenflashv2.5

# Run inbound only
docker run -p 8000:8000 --env-file .env gpt5.4-speechmaticsagent-elevenflashv2.5 \
  python -m inbound.server

# Run outbound only
docker run -p 8000:8000 --env-file .env gpt5.4-speechmaticsagent-elevenflashv2.5 \
  python -m outbound.server
```

### Production Checklist

- [ ] `PUBLIC_URL` points to a stable HTTPS URL (not ngrok)
- [ ] Plivo webhooks auto-configured on server startup (check logs for confirmation)
- [ ] `LOG_FORMAT=json` for structured log ingestion
- [ ] `LOG_FILE` set for persistent log storage (or use `REDIS_EVENTS_URL`)
- [ ] `OTEL_EXPORTER_OTLP_ENDPOINT` set for distributed tracing (optional)
- [ ] Reverse proxy (nginx/caddy) terminates TLS; `PUBLIC_URL` uses `https://`

## Observability

### Log Levels

| `LOG_LEVEL` | What is logged |
|-------------|----------------|
| `verbose` | Every pipeline event: per-packet stats, VAD frame probabilities, queue sizes, TTFB |
| `normal` | Key events: turn lifecycle, STT results, LLM responses, TTS timing (default) |
| `quiet` | Errors and session start/end only |

### Log Sinks

- **stderr (text)**: Default. Human-readable via loguru.
- **stderr (JSON)**: Set `LOG_FORMAT=json` — structured fields, suitable for log aggregation (Datadog, Loki).
- **File sink**: Set `LOG_FILE=/path/to/agent.log` — JSON format, 100MB rotation, 7-day retention.
- **Redis Streams**: Set `REDIS_EVENTS_URL=redis://...` — publishes each log record to `REDIS_STREAM_KEY` (default `voice-agent:events`) for real-time dashboards.

### OpenTelemetry

Install the observability extra and set the OTLP endpoint:

```bash
uv sync --extra observability

export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
uv run python -m inbound.server
```

OpenLLMetry (Traceloop) auto-instrumentation is enabled automatically when installed — it traces LLM calls with token counts, latencies, and model metadata. httpx spans cover all outbound HTTP calls (OpenAI + ElevenLabs REST).
