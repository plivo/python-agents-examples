# GPT-4o + Modulate Velma-2 + Cartesia Sonic 3 (Pipecat Voice Agent)

Pipecat framework orchestration over Plivo telephony. Plivo's bidirectional stream (base64 μ-law 8kHz in JSON over WebSocket) enters `FastAPIWebsocketTransport` through `PlivoFrameSerializer`, which decodes and resamples it to PCM16 16kHz mono, the pipeline input rate; on the way out the same serializer resamples Cartesia's PCM16 24kHz to μ-law 8kHz `playAudio` events. Those are the only two resampling points.
- STT: Modulate `velma-2` over WebSocket (`wss://platform.modulate.ai/api/velma-2-streaming`), raw s16le 16kHz mono, API key in the query string, one JSON configuration frame before any audio; `partial_clip` events become interim transcripts and `clip` events final ones, with per-clip emotion, accent, synthetic-voice score, speaker label and language left on `TranscriptionFrame.result`.
- LLM: OpenAI `gpt-4o` (streaming Chat Completions over HTTPS) with one tool, `search_the_web`, backed by Tavily over HTTPS (`search_depth=fast`, `include_answer=advanced`, `max_results=5`, 5s hard timeout). TTS: Cartesia `sonic-3.6` over WebSocket, `pcm_s16le` 24kHz.
- VAD: Pipecat `SileroVADAnalyzer` (Silero ONNX) on the user aggregator, 512-sample (32ms) frames at 16kHz, with Pipecat's defaults, not tuned for this example: `confidence` 0.7, `start_secs` 0.2, `stop_secs` 0.2, `min_volume` 0.6.
- Barge-in: speech start while the agent is talking makes Pipecat emit an `InterruptionFrame`, which cancels the in-flight LLM response and TTS audio, and the serializer sends Plivo `clearAudio`. Inbound opens with an LLM-generated line; outbound speaks the `greeting` query param of the `answer_url` verbatim through a `TTSSpeakFrame`.

## Features

- **Modulate as the STT.** Velma-2 returns more than words. Each finalised clip can carry an emotion label, an accent, and a synthetic-voice score on the same socket that delivers the transcript, so the agent has signals a plain STT never surfaces. The transcript drives the pipeline; the extras are logged at DEBUG and left on `TranscriptionFrame.result` for anything downstream.
- **Tavily as a tool.** The LLM calls `search_the_web` when it is not confident, rather than inventing an answer. The system prompt tells it to search instead of guessing and to say where an answer came from. Every failure path (no key, error, timeout, no answer) hands the LLM a sentence it can speak.
- **Inbound and outbound.** Two self-contained servers: `inbound.server` (port 8000) answers calls to your Plivo number, `outbound.server` (port 8001) answers calls you place with Plivo's Make Call API and speaks a per-call greeting.
- **Webhook authentication, always on.** Every Plivo HTTP webhook must carry a valid V3 signature; unsigned requests get 403 and the servers refuse to start without `PLIVO_AUTH_TOKEN`.
- **Turn taking and barge-in by Pipecat.** Silero VAD on the user aggregator; an interruption cancels the LLM and TTS and flushes Plivo's playback buffer with `clearAudio`.
- **Prompts in files.** `inbound/system_prompt.md` and `outbound/system_prompt.md` are the only prompt source.

## Prerequisites

- Python 3.11+ and [uv](https://docs.astral.sh/uv/). Pipecat 1.x, which this example is written against (`pipecat-ai>=1.12.0`), does not support Python 3.10
- A Plivo account: auth ID, auth token and a voice-enabled phone number
- OpenAI API key
- Modulate API key (platform.modulate.ai, API Keys tab)
- Cartesia API key
- Tavily API key (app.tavily.com). Optional: without it the tool tells the LLM that web search is not configured
- ngrok or another public HTTPS tunnel for webhooks

## Quick Start

```bash
cd gpt4o-modulatevelma2-cartesiasonic3-pipecat
uv sync
cp .env.example .env     # then fill it in
```

Start a tunnel and put its HTTPS URL in `PUBLIC_URL` (scheme and host only, exactly as Plivo will call it):

```bash
ngrok http 8000          # inbound; use 8001 for the outbound server
```

Inbound:

```bash
uv run python -m inbound.server
```

On startup the inbound server creates or updates the Plivo application `GPT4o_ModulateVelma2_CartesiaSonic3_Pipecat`, points its answer and hangup URLs at `PUBLIC_URL`, and attaches `PLIVO_PHONE_NUMBER` to it. Call that number to talk to the agent. With `PUBLIC_URL` or `PLIVO_PHONE_NUMBER` unset it skips this step and you configure the number yourself.

Outbound:

```bash
uv run python -m outbound.server
```

The outbound server configures nothing in Plivo. Place a call with the Make Call API as shown in [Outbound Calls](#outbound-calls).

Both servers can run at once: they listen on different ports (`SERVER_PORT`, `OUTBOUND_SERVER_PORT`), each behind its own tunnel or one host that routes to both.

## Project Structure

```
gpt4o-modulatevelma2-cartesiasonic3-pipecat/
├── inbound/
│   ├── __init__.py
│   ├── agent.py             # ModulateSTTService, search_the_web tool, run_agent() pipeline
│   ├── server.py            # FastAPI: /answer, /hangup, /fallback, /hold, /ws; Plivo auto-config
│   └── system_prompt.md     # Inbound system prompt
├── outbound/
│   ├── __init__.py
│   ├── agent.py             # Same service and tool (identical copies) + verbatim greeting
│   ├── server.py            # FastAPI: /outbound/answer, /outbound/hangup, /ws
│   └── system_prompt.md     # Outbound system prompt
├── utils.py                 # Phone number normalization (no audio helpers: Pipecat converts)
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # sys.path setup
│   ├── helpers.py           # Server subprocess, webhook signing, test-only μ-law codec, caller TTS, ngrok, recording
│   ├── test_integration.py  # Offline unit tests + local integration
│   ├── test_e2e_live.py     # Real LLM/STT/TTS over a simulated Plivo stream (no phone call)
│   ├── test_live_call.py    # Real inbound call
│   ├── test_multiturn_voice.py  # Multi-turn + barge-in over a simulated Plivo stream
│   └── test_outbound_call.py    # Real outbound call placed with the Make Call API
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

`ModulateSTTService` and `search_the_web` live in `agent.py` of each direction, as identical copies, because each direction is self-contained. `utils.py` holds only `normalize_phone_number` (used by both servers). It has no μ-law, PCM or resampling helpers and the example has no `numpy`/`scipy` runtime dependency of its own: `PlivoFrameSerializer` does every conversion on the call path. The tests keep a small pure-Python G.711 codec and downsampler in `tests/helpers.py` for RMS checks, transcription and caller speech.

## How It Works

### Pipeline Architecture

```
Plivo (μ-law 8kHz)
   → FastAPIWebsocketTransport + PlivoFrameSerializer
   → Modulate Velma-2 STT (WebSocket, s16le 16kHz)
   → LLMContextAggregator (user side, Silero VAD)
   → OpenAI gpt-4o  ⇄  Tavily search_the_web (HTTPS tool call)
   → Cartesia Sonic TTS (WebSocket, PCM16 24kHz)
   → LLMContextAggregator (assistant side)
   → transport.output() → Plivo
```

| Hop | Protocol | Format | Rate |
|---|---|---|---|
| Plivo → serializer | WebSocket, base64 in JSON `media` events | μ-law | 8 kHz |
| Serializer → Modulate STT and Silero VAD | WebSocket binary frames (STT), in process (VAD) | PCM16 (s16le) mono | 16 kHz |
| Modulate STT → gpt-4o | Pipecat frames, then HTTPS | Text | N/A |
| gpt-4o ⇄ Tavily | HTTPS | JSON | N/A |
| gpt-4o → Cartesia | WebSocket | Text | N/A |
| Cartesia → serializer | WebSocket | PCM16 (`pcm_s16le`) | 24 kHz |
| Serializer → Plivo | WebSocket, base64 in JSON `playAudio` events | μ-law | 8 kHz |

16 kHz and 24 kHz are Pipecat's default pipeline input and output rates (`PipelineParams.audio_in_sample_rate` / `audio_out_sample_rate`); the agents do not override them. The lock file resolves a single Pipecat version, 1.12.0 (Python 3.11+); the rates and the VAD defaults below are that version's. `run_agent()` uses `PipelineWorker` and `WorkerRunner` (`add_workers(worker)` then `run()`), not `PipelineTask` / `PipelineRunner`, which Pipecat deprecated in 1.3.0 and removes in 2.0.0. `WorkerRunner()` is built with its defaults, so `handle_sigterm` stays `False` and uvicorn keeps its own SIGTERM handler.

### Call flow

1. Plivo calls the answer webhook (`/answer` or `/outbound/answer`). The server verifies the signature and returns `<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">` with a `wss://…/ws?body=…` URL. `body` is percent-encoded base64 JSON holding `call_uuid`, `from`, `to` (and `greeting` for outbound).
2. Plivo opens `/ws` and sends `start`, then `media` events. The server reads the `start` event and calls `run_agent()`, which assembles the pipeline above.
3. Opening line. Inbound: the LLM is prompted with a stand-in caller turn ("Hello, I'm calling for help.") that is not kept in the conversation context, so the opening line is LLM-generated. Outbound: the greeting goes straight to TTS and is recorded in the context as an assistant turn, so the LLM does not introduce itself again.
4. Turn taking. `SileroVADAnalyzer` (`confidence` 0.7, `start_secs` 0.2, `stop_secs` 0.2, `min_volume` 0.6: Pipecat's defaults) on the user aggregator marks speech start and end; Modulate's `clip` transcripts fill the user turn.
5. Barge-in. Speech start while the agent is talking interrupts it: Pipecat cancels the in-flight LLM response and TTS audio and the serializer sends `{"event": "clearAudio"}` so Plivo drops what it has buffered.
6. Tool calls. When the LLM calls `search_the_web`, the handler queries Tavily and returns a short answer plus up to two source titles. The handler is bounded at 5 seconds, with a 7 second Pipecat function-call timeout behind it.
7. Hangup. When Plivo closes the stream the pipeline ends. The agent does not hang up calls itself (`auto_hang_up=False`).

### Endpoints

| Server | Endpoint | Method | Signature-checked | Description |
|---|---|---|---|---|
| inbound | `/` | GET | No | Health check |
| inbound | `/answer` | GET/POST | Yes | Answer webhook: returns `<Stream>` XML |
| inbound | `/hangup` | POST | Yes | Hangup webhook: logs `CallUUID`, `Duration`, `HangupCause` |
| inbound | `/fallback` | POST | Yes | Fallback webhook: speaks an apology and hangs up |
| inbound | `/hold` | GET/POST | Yes | Returns `<Wait length="120">`; used as the calling leg's answer URL in live tests |
| inbound | `/ws` | WebSocket | No | Plivo audio stream; runs the agent |
| outbound | `/` | GET | No | Health check |
| outbound | `/outbound/answer` | GET/POST | Yes | Answer webhook (your `answer_url`): reads the optional `greeting` query param and returns `<Stream>` XML |
| outbound | `/outbound/hangup` | POST | Yes | Hangup webhook (your optional `hangup_url`): logs only |
| outbound | `/ws` | WebSocket | No | Plivo audio stream; runs the agent |

## Webhook authentication

Both servers are exposed on a public URL, so they accept only webhook requests that come from Plivo. The check always runs; there is no setting to turn it off.

- **Plivo webhooks** (`/answer`, `/hangup`, `/fallback`, `/hold`, `/outbound/answer`, `/outbound/hangup`) must carry a valid [Plivo V3 signature](https://www.plivo.com/docs/voice/concepts/signature-validation): the `X-Plivo-Signature-V3` and `X-Plivo-Signature-V3-Nonce` headers, keyed with `PLIVO_AUTH_TOKEN`. `verify_plivo_signature()`, a FastAPI dependency in each `server.py`, checks them with the Plivo SDK's `validate_v3_signature()`. The signed params are the form fields for POST and the query string for GET. Anything else gets **403** and a warning with the path and reason (`missing …`, `signature mismatch`, `PLIVO_AUTH_TOKEN or PUBLIC_URL not set`). Signatures are never logged.
- **`/ws`** is not a Plivo webhook, so its handshake is not signed and it has no check of its own, like the other examples in this repo. The stream URL is only handed to Plivo in the signed answer webhook's `<Stream>` XML.

**`PUBLIC_URL` must be exactly the URL Plivo is configured to call.** Plivo signs the URL it requested. Behind a tunnel or reverse proxy the server sees `http://localhost:8000/...` instead, so the signed URL is rebuilt as `PUBLIC_URL` (trailing `/` stripped) + request path + raw query string; `request.url` is not used. The scheme, host and any path prefix must match the answer and hangup URLs in the Plivo application or Make Call request. With `PUBLIC_URL` unset every webhook is rejected. The outbound `answer_url` query string (`greeting`) is part of what Plivo signs, so editing it invalidates the signature.

**`PLIVO_AUTH_TOKEN` is required.** With it empty, both servers log the reason and exit with status 1 before listening. Use a subaccount's auth token if the number and calls belong to a subaccount, because Plivo signs with the token of the account that owns the call. The tests use a dummy token and sign their requests the way Plivo does.

## Outbound Calls

You place outbound calls with [Plivo's Make Call API](https://www.plivo.com/docs/voice/api/call/make-a-call). The outbound server has no dial endpoint and keeps no call records: it answers Plivo's webhooks and bridges audio.

```
your shell / backend ──POST /v1/Account/{auth_id}/Call/──► Plivo ──dials──► callee
                                                             │ callee answers
                        /outbound/answer?greeting=…       ◄─┘ (answer_url)
                          └─► <Stream> ─► /ws ─► run_agent(greeting)
```

With the outbound server running and reachable at `PUBLIC_URL` (the numbers below are placeholders; `set -a && source .env && set +a` exports the credentials):

```bash
curl -X POST "https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/" \
  -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"from": "+14155550100", "to": "+14155550123",
       "answer_url": "https://your-host.example.com/outbound/answer?greeting=Hi%2C%20this%20is%20a%20quick%20call%20about%20your%20appointment%20tomorrow.%20Is%20now%20a%20good%20time%3F",
       "hangup_url": "https://your-host.example.com/outbound/hangup",
       "answer_method": "POST", "hangup_method": "POST"}'
```

URL-encode the query value. From Python, `plivo.RestClient().calls.create(from_=..., to_=..., answer_url=..., answer_method="POST", hangup_url=...)` does the same (see `tests/test_outbound_call.py`). At startup the server logs the `answer_url` and `hangup_url` to use for its `PUBLIC_URL`.

| `answer_url` param | Used for | When missing |
|---|---|---|
| `greeting` | Spoken **verbatim** by TTS when the callee answers, and recorded in the LLM context as the assistant's first turn | `DEFAULT_GREETING` in `outbound/agent.py` ("Hello, this is an automated assistant calling. Is now a good time to talk?") |

The greeting is the only per-call input. The system prompt is `outbound/system_prompt.md` as is, with no per-call templating; it tells the LLM that the greeting has already been spoken, so write the prompt for your use case there. For status, duration or recordings, use Plivo's Call API and call logs, or your `hangup_url`.

## Configuration

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key (LLM) | Required |
| `MODULATE_API_KEY` | Modulate API key (STT); sent in the WebSocket query string | Required |
| `CARTESIA_API_KEY` | Cartesia API key (TTS) | Required |
| `TAVILY_API_KEY` | Tavily API key. Without it `search_the_web` tells the LLM that web search is not configured | Empty |
| `LLM_MODEL` | OpenAI model | `gpt-4o` |
| `TTS_MODEL` | Cartesia model ID. `sonic-3.6` is Cartesia's current stable Sonic model and tracks its latest stable snapshot; pin a dated snapshot such as `sonic-3.6-2026-08-27` for unchanging behaviour. Older IDs (`sonic-3.5`, `sonic-3`) take the same request shape and voice IDs | `sonic-3.6` |
| `TTS_VOICE` | Cartesia voice ID, from the Cartesia voice library or its List Voices API | `71a7ad14-091c-4e8e-a314-022ece01c121` (a Cartesia library voice) |
| `TAVILY_SEARCH_DEPTH` | Passed to Tavily as `search_depth`: `ultra-fast`, `fast`, `basic` or `advanced` | `fast` |
| `PLIVO_AUTH_ID` | Plivo auth ID (inbound auto-configuration; your own Make Call requests) | Required for auto-configuration |
| `PLIVO_AUTH_TOKEN` | Plivo auth token. Also the key for webhook signature checks: both servers refuse to start without it | Required |
| `PLIVO_PHONE_NUMBER` | Inbound: the number auto-configured at startup. Outbound: not used by the server beyond the health check; use it as `from` in your Make Call request | Empty |
| `PUBLIC_URL` | Public HTTPS URL of the server. Signatures are checked against it and the `wss://` stream URL is built from it | Required |
| `SERVER_PORT` | Inbound server port | `8000` |
| `OUTBOUND_SERVER_PORT` | Outbound server port (different, so both can run at once) | `8001` |
| `DEFAULT_COUNTRY_CODE` | Default region (ISO 3166-1 alpha-2) for phone numbers without a country prefix | `US` |
| `PLIVO_TEST_NUMBER` | Tests only: second Plivo number (caller for inbound tests, destination for outbound tests) | Empty |
| `NGROK_BIN` | Tests only: path to the ngrok binary | `ngrok` |
| `FFMPEG_DIR` | Tests only: directory holding an `ffmpeg` binary for faster-whisper | Empty |
| `TEST_LOG_DIR` | Tests only: where test server logs are written | System temp dir |

The Modulate endpoint (`MODULATE_STT_URL`), the Tavily timeout (`TAVILY_TIMEOUT_SECS = 5.0`) and `max_results=5` are constants in the agent files, not env vars.

### System prompt

Each direction has one prompt source: `inbound/system_prompt.md` and `outbound/system_prompt.md`. There is no env override. To customise a prompt, edit the file, or in Docker mount your own file over it (see [Docker](#docker)).

## Adapting this to your use case

### Modulate: behaviour presets

`ModulateSTTService` is defined in `inbound/agent.py` and, as an identical copy, in
`outbound/agent.py` (each direction is self-contained). It takes three options beyond
the API key, all sent in the one configuration frame that precedes any audio. Both
agents construct it with the API key only, so change the call in `run_agent()`:

```python
stt = ModulateSTTService(
    api_key=MODULATE_API_KEY,
    behaviors=["preset:vishing", "preset:jailbreak_attempt"],
    produce_topics=True,
    produce_summary=True,
)
```

Velma ships **164 behaviour presets**, which is the main thing to tailor. They
cover far more than fraud: compliance exposure, CX quality, sales signals, and
agent self-monitoring. List them all, with the criteria each one uses, from the
API itself rather than trusting a copy in a README:

```bash
curl -s -H "X-API-Key: $MODULATE_API_KEY" \
  https://platform.modulate.ai/api/velma-2-batch/list-presets | jq -r '.presets[].identifier'
```

A sample of what is in there, by the use case you would pick it for:

| Use case | Presets |
|---|---|
| Fraud and account takeover | `vishing`, `account_impersonation`, `credential_solicitation`, `return_fraud_attempt`, `refund_abuse`, `remote_access_request` |
| Agent security | `jailbreak_attempt`, `ai_agent_manipulation`, `hallucination_policy`, `inapropriate_ai_agent_content` |
| Compliance | `fdcpa_violation_risk`, `regulation_b_fair_lending_risk`, `do_not_call_violation_risk`, `recording_consent_omission`, `consent_withdrawal` |
| CX and QA | `issue_resolved`, `issue_not_resolved`, `monologuing`, `repetition_loop`, `unaddressed_question`, `sop_greeting_the_customer` |
| Sales | `objection_timing`, `objection_trust`, `budget_qualification`, `champion_identification`, `price_sensitivity` |
| Caller welfare | `caller_under_duress`, `personal_vulnerability`, `suicidal_and_self_injurious_ideation`, `mandatory_reporting_trigger` |
| Verticals | `finance_*`, `insurance_*`, `retail_*`, `logistics_*`, `medical_*`, `saas_*` |

Three things to know before you rely on them:

- **Prefix every name with `preset:`.** An unknown identifier fails the whole
  request with `422 invalid behavior preset`, so a typo is loud rather than silent.
- **Undetected behaviours are omitted from the response, not returned as false.**
  Asking for six and getting five back means one did not fire.
- **Behaviours did not arrive on the streaming socket during testing.** See the
  measurements below. The same audio and presets return them reliably from the
  batch endpoint, so treat behaviours as a post-call pass.

What *is* live on every finalised clip, with no configuration at all, is
`emotion`, `accent`, `deepfake_score`, `speaker_label` and `language`. These are
logged at DEBUG (never the transcript text) and left on `TranscriptionFrame.result`
for anything downstream.

### Tavily: shaping the search

`search_the_web()` in `inbound/agent.py` (and its identical copy in
`outbound/agent.py`) calls `AsyncTavilyClient.search` with `query`, `search_depth`,
`include_answer="advanced"`, `max_results=5` and `timeout`. The whole call is bounded
by `TAVILY_TIMEOUT_SECS = 5.0`; past that the tool tells the LLM it could not look the
answer up. The arguments worth changing per use case:

| Argument | Why you would change it |
|---|---|
| `search_depth` | Latency. Set with `TAVILY_SEARCH_DEPTH` (`ultra-fast`, `fast`, `basic` or `advanced`). `fast` is the default here for the reason in the measurements below |
| `max_results` | Fewer results is less context for the LLM and a faster turn. The example ships `5`. One spot comparison by the author on 5 October 2026 measured 0.50s for `3` against 2.28s for `5`; the search depth and answer mode of that comparison were not recorded, so it is not comparable with the table below |
| `include_domains` | Ground answers in sources you trust, such as your own docs or a price list |
| `exclude_domains` | Keep competitors or aggregators out of the answer |
| `topic="news"` + `days` | Recency, when the agent fields questions about current events |
| `include_answer` | `advanced` writes a fuller synthesis, `basic` is terser and quicker |

```python
response = await asyncio.wait_for(
    client.search(
        query=query,
        search_depth=TAVILY_SEARCH_DEPTH,
        include_domains=["yourcompany.com"],   # answer only from your own docs
        include_answer="advanced",
        max_results=3,
        timeout=TAVILY_TIMEOUT_SECS,
    ),
    timeout=TAVILY_TIMEOUT_SECS,
)
```

Make the same edit in both agent files; `tests/test_integration.py` asserts the two
copies stay identical and pins the arguments that ship.

One gotcha: `country` is rejected with `BadRequestError` when `search_depth` is
`fast` or `ultra-fast`. Localising results means giving up the fast index.

## Measurements

Taken by the example's author against the live APIs on 5 October 2026. They are
not re-measured by the test suite.

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

## Testing

The tests keep webhook authentication on. They sign webhook requests the way Plivo does (`plivo_signature_headers()` / `signed_webhook()` in `tests/helpers.py`, built on the Plivo SDK) and open `/ws` with the stream URL returned by a signed answer webhook. Servers started for local tests get a dummy auth token, `PUBLIC_URL` set to their own `http://localhost:<port>` URL, and no Plivo auth ID or phone number, so they cannot reconfigure a real Plivo number. Every test that needs a live service skips, with the missing names in the reason, unless all the keys it needs are set.

Run from this directory:

```bash
uv sync   # includes the dev group
uv run ruff check .

# Unit tests: offline, no API keys, no network
uv run pytest tests/test_integration.py -v -k unit

# Local server over a simulated Plivo stream + OpenAI key check
# (needs OPENAI_API_KEY, MODULATE_API_KEY, CARTESIA_API_KEY)
uv run pytest tests/test_integration.py -v -k "local or OpenAI"

# Plivo credential and number checks, read-only (needs PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER)
uv run pytest tests/test_integration.py -v -k Plivo

# E2E with the real LLM, STT and TTS, no phone call (same three API keys, plus ffmpeg)
uv run pytest tests/test_e2e_live.py -v -s

# Multi-turn + barge-in over a simulated Plivo stream (same three API keys, nothing else:
# the caller's speech is synthesised with OpenAI TTS as raw PCM, so no ffmpeg, gTTS or pydub)
uv run pytest tests/test_multiturn_voice.py -v -s

# Real calls (the three API keys, Plivo credentials, PLIVO_TEST_NUMBER, ngrok, ffmpeg)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
```

| Test file | Port | What it covers |
|---|---|---|
| `test_integration.py` (`-k unit`) | none | The test-only μ-law codec and downsampler, phone normalization, `ModulateSTTService._handle_event`, the Tavily tool with a stubbed client, prompts and the outbound greeting, `PipelineWorker` / `WorkerRunner` wiring with no Pipecat `DeprecationWarning` from the agents, server routes, webhook authentication for both servers |
| `test_integration.py` (`-k local`) | 18001 | Health, signed and unsigned `/answer`, `playAudio` and speech energy from the opening line |
| `test_e2e_live.py` | 18005 | Opening line transcribed with faster-whisper |
| `test_multiturn_voice.py` | 18004 | Three spoken user turns each answered; speaking over an answer produces `clearAudio`. Local only (server subprocess + simulated Plivo stream); skips only when one of the three API keys is missing, and uses `OPENAI_API_KEY` (`gpt-4o-mini-tts`) for the caller's speech |
| `test_live_call.py` | 18002 | Real inbound call: signed webhook through the tunnel, recorded and transcribed opening line |
| `test_outbound_call.py` | 18003 | Real outbound call via `calls.create(answer_url=…/outbound/answer?greeting=…)`: greeting spoken, hangup webhook received |

The live call tests use `PLIVO_TEST_NUMBER`, a second Plivo number on the same account. They assign test-only Plivo applications to the numbers involved and restore the original application afterwards. They start their own ngrok agent and skip if one is already running.

From the repo root:

```bash
./scripts/validate-example.sh gpt4o-modulatevelma2-cartesiasonic3-pipecat
```

Two of its checks do not apply to this pipeline and fail: `PLIVO_CHUNK_SIZE = 160` (Pipecat's output transport and `PlivoFrameSerializer` frame the audio; the agent has no `_send_to_plivo`) and `vad_enabled=True` (VAD is attached as `vad_analyzer` on the user aggregator, not as a transport flag).

## Docker

```bash
docker build -t gpt4o-modulatevelma2-cartesiasonic3-pipecat .

# Inbound (default command): configures the Plivo number from PUBLIC_URL on startup
docker run -p 8000:8000 --env-file .env -e PUBLIC_URL=https://your-host.example.com \
  gpt4o-modulatevelma2-cartesiasonic3-pipecat

# Outbound (listens on 8001, OUTBOUND_SERVER_PORT)
docker run -p 8001:8001 --env-file .env -e PUBLIC_URL=https://your-host.example.com \
  gpt4o-modulatevelma2-cartesiasonic3-pipecat uv run python -m outbound.server
```

The image exposes 8000 (inbound) and 8001 (outbound). It is based on `python:3.12-slim` by default; pass `--build-arg BASE_IMAGE=...` to change it. It runs `uv sync --locked --no-install-project --no-dev --extra streaming`, so the `streaming` extra (Redis) is installed and the `observability` extra is not.

To use your own system prompt without editing the repo, mount a file over the one in the image (`WORKDIR` is `/app`):

```bash
docker run -v "$PWD/my_prompt.md:/app/inbound/system_prompt.md:ro" \
  -p 8000:8000 --env-file .env -e PUBLIC_URL=https://your-host.example.com \
  gpt4o-modulatevelma2-cartesiasonic3-pipecat

docker run -v "$PWD/my_outbound_prompt.md:/app/outbound/system_prompt.md:ro" \
  -p 8001:8001 --env-file .env -e PUBLIC_URL=https://your-host.example.com \
  gpt4o-modulatevelma2-cartesiasonic3-pipecat uv run python -m outbound.server
```

Use a host that keeps WebSocket connections open for the length of a call.

## Troubleshooting

**The server exits at startup with "PLIVO_AUTH_TOKEN is empty".** Webhook signatures cannot be checked without it. Set `PLIVO_AUTH_TOKEN`.

**403 on `/answer` or `/outbound/answer`, or the call drops at once.** The signature did not verify. The log line `Rejected Plivo webhook …` gives the reason. Check that `PUBLIC_URL` is exactly the scheme and host Plivo calls (the running tunnel, `https`), and that `PLIVO_AUTH_TOKEN` belongs to the account or subaccount that owns the number.

**No audio in either direction.** Check `PUBLIC_URL` matches the running tunnel and that the Plivo number's answer URL points at `/answer` (inbound) or that your `answer_url` points at `/outbound/answer` (outbound).

**Modulate closes immediately with code 4001.** The API key is wrong or missing. It goes in the query string, not a header. The service logs the rejection and does not reconnect.

**The agent talks over the caller.** Interruptions are on by default in Pipecat and driven by the Silero VAD attached to the user aggregator in `run_agent()`. Check that the `vad_analyzer` is still passed in `LLMUserAggregatorParams`, and raise or lower `VADParams` (`confidence`, `start_secs`) if speech starts are missed.

**Long silences before answers.** `TAVILY_SEARCH_DEPTH` is probably set to `advanced`. See the measurements above. A search that passes 5 seconds is abandoned and the agent says it could not look the answer up.

**The agent says web search is not configured.** `TAVILY_API_KEY` is empty.
