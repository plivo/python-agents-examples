# GPT-4o + Modulate Velma-2 STT + Cartesia Sonic TTS — Pipecat Voice Agent

Pipecat framework orchestration. Plivo μ-law 8kHz streams into `FastAPIWebsocketTransport` via `PlivoFrameSerializer`, which decodes to PCM16 and drives Silero VAD (ONNX v5) on the user aggregator. STT is Modulate `velma-2` over WebSocket at `wss://platform.modulate.ai/api/velma-2-streaming`, taking raw s16le at the pipeline sample rate with the API key as a query parameter and one JSON config frame before any audio; `partial_clip` events become interim transcripts and `clip` events final ones, each carrying optional emotion, accent and synthetic-voice score which are logged and attached to the frame's `result`. LLM is OpenAI `gpt-4o` with one registered tool, `search_the_web`, backed by Tavily at `search_depth=fast` (measured 1.27s median against 5.46s for `advanced`, which is long enough on a live call that the caller assumes the line dropped). TTS is Cartesia `sonic-3.6`, returned as PCM and resampled to μ-law 8kHz by the serializer. Barge-in is handled by Pipecat interruption handling, which cancels in-flight LLM and TTS frames and emits `clearAudio` to Plivo.

## What makes this example different

Two things, both about what the agent knows:

- **Modulate as the STT.** Velma-2 returns more than words. Each finalised clip can carry an emotion label, an accent, and a synthetic-voice score on the same socket that delivers the transcript, so the agent has signals a plain STT never surfaces. The transcript drives the pipeline; the extras are logged and left on `TranscriptionFrame.result` for anything downstream.
- **Tavily as a tool.** The LLM calls `search_the_web` when it is not confident, rather than inventing an answer. The system prompt tells it to search instead of guessing and to say where an answer came from.

## Pipeline Architecture

```
Plivo (μ-law 8kHz)
   → FastAPIWebsocketTransport + PlivoFrameSerializer
   → Modulate Velma-2 STT (WebSocket, s16le)
   → LLMContextAggregator (Silero VAD)
   → OpenAI gpt-4o  ⇄  Tavily search_the_web (HTTP tool call)
   → Cartesia Sonic TTS
   → transport.output() → Plivo
```

| Hop | Format | Rate |
|---|---|---|
| Plivo → serializer | μ-law | 8 kHz |
| Serializer → Modulate STT | PCM16 (s16le) | pipeline rate |
| Modulate STT → gpt-4o | Text | N/A |
| gpt-4o → Cartesia | Text | N/A |
| Cartesia → serializer | PCM16 | 24 kHz |
| Serializer → Plivo | μ-law | 8 kHz |

## Prerequisites

- Python 3.10+
- A Plivo account, auth ID, auth token, and a phone number
- OpenAI API key
- Modulate API key (platform.modulate.ai, API Keys tab)
- Cartesia API key
- Tavily API key (app.tavily.com)
- ngrok or another public tunnel for webhooks

## Setup

```bash
cd gpt4o-modulatevelma2-cartesiasonic3-pipecat
uv sync
cp .env.example .env     # then fill it in
```

Start a tunnel and put its host in `PUBLIC_URL`:

```bash
ngrok http 8000
```

## Running

Inbound:

```bash
uv run python -m inbound.server
```

Outbound:

```bash
uv run python -m outbound.server
```

The server configures the webhooks on `PLIVO_PHONE_NUMBER` at startup, so an
inbound call to that number reaches the agent with no further setup.

## Project Structure

```
gpt4o-modulatevelma2-cartesiasonic3-pipecat/
├── inbound/
│   ├── agent.py             # Pipecat pipeline + Tavily tool
│   ├── server.py            # FastAPI: /answer, /ws, /hangup
│   └── system_prompt.md
├── outbound/
│   ├── agent.py             # Same pipeline + CallManager
│   ├── server.py            # FastAPI: /outbound/answer, /outbound/hangup, /ws
│   └── system_prompt.md
├── utils.py                 # Audio conversion, phone utils, ModulateSTTService
├── tests/
├── pyproject.toml
├── .env.example
└── README.md
```

`ModulateSTTService` lives in `utils.py` because both agents use it and the
canonical structure has no `services/` directory.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `MODULATE_API_KEY` | Modulate API key | Required |
| `CARTESIA_API_KEY` | Cartesia API key | Required |
| `TAVILY_API_KEY` | Tavily API key | Required for the search tool |
| `LLM_MODEL` | OpenAI model | `gpt-4o` |
| `TTS_MODEL` | Cartesia model | `sonic-3.6` |
| `TTS_VOICE` | Cartesia voice ID | British Reading Lady |
| `TAVILY_SEARCH_DEPTH` | `fast`, `basic` or `advanced` | `fast` |
| `PLIVO_AUTH_ID` | Plivo auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo auth token | Required |
| `PLIVO_PHONE_NUMBER` | Number to auto-configure | Required |
| `PUBLIC_URL` | Public tunnel URL | Required |
| `SERVER_PORT` | Port | `8000` |

## Measurements

Taken against the live APIs on 5 October 2026.

**Tavily search latency**, same query, three runs each. This happens mid-call
while the caller waits in silence, so it governs the default.

| `search_depth` | `include_answer` | median | worst |
|---|---|---|---|
| `advanced` | `advanced` | 5.46s | 5.79s |
| `basic` | `basic` | 3.80s | 4.60s |
| **`fast`** | **`advanced`** | **1.27s** | **1.53s** |
| `fast` | `basic` | 0.47s | 1.75s |

`fast` is a different index rather than a truncated `advanced`, so answer
quality holds.

**Modulate signal timing**, on a 9.4 second call fed in real time:

| Signal | Arrives |
|---|---|
| Partial transcript | 3.2s |
| Final transcript | 13.3s |
| Topics | 77.4s |
| Summary | 78.0s |
| `behavior_detection` | not observed on streaming |

Behaviours such as jailbreak and vishing detection returned 98–99.5% confidence
from the **batch** endpoint on identical audio, but no `behavior_detection`
event arrived on the streaming socket across two samples and two behaviour sets.
Treat behaviours, topics and summary as post-call signals; the transcript and
the per-clip emotion and synthetic-voice scores are the live ones.

## Troubleshooting

**No audio in either direction** — check `PUBLIC_URL` matches the running
tunnel and that the Plivo number's answer URL points at `/answer`.

**Modulate closes immediately with code 4001** — the API key is wrong or
missing. It goes in the query string, not a header.

**The agent talks over the caller** — Silero VAD is attached to the user
aggregator; confirm `allow_interruptions=True` on `PipelineParams`.

**Long silences before answers** — `TAVILY_SEARCH_DEPTH` is probably set to
`advanced`. See the measurements above.
