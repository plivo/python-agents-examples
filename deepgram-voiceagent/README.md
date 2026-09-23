# Deepgram Voice Agent API — Plivo Voice Agent

Plivo bidirectional audio streaming bridged over raw `websockets` + asyncio (no SDK/framework) to the Deepgram Voice Agent API (`wss://agent.deepgram.com/v1/agent/converse`, `Authorization: Token`), a managed listen → think → speak pipeline on one WebSocket. Plivo μ-law 8kHz (base64 JSON, 160B/20ms) is forwarded unchanged as binary frames and Settings request raw μ-law 8kHz output (`container: none`), so there is no transcoding or resampling in either direction.
- Models are `.env` config passed verbatim into Settings. Defaults: listen Deepgram Flux `flux-general-en` (`version: v2`, `eot_threshold` 0.7, `eot_timeout_ms` 5000) for end-of-turn detection with no client-side VAD; think `open_ai` `gpt-4.1-mini` hosted by Deepgram with 5 client-side functions (`end_call`/`transfer_call` use `defer_until_eot`); speak Aura-2 `aura-2-thalia-en`.
- Barge-in on Deepgram `UserStartedSpeaking` while audio is playing: Deepgram cancels its LLM/TTS; the client drains the send queue, sends Plivo `clearAudio` and drops late agent audio until the next user end-of-turn (`ConversationText` user / `EndOfTurn`).
- Playback end is tracked with Plivo `checkpoint`/`playedStream`; `end_call` hangs up via Plivo REST `calls.delete` after the goodbye's checkpoint is played (15s fallback deadline).
- Optional saved agent configuration: with `DEEPGRAM_INBOUND_AGENT_ID`/`DEEPGRAM_OUTBOUND_AGENT_ID` set, Settings sends `agent: <uuid>` (a config published from this code via `python -m inbound.agent --publish`) and the per-call context (`UpdatePrompt`) and greeting (`InjectAgentMessage`) follow `SettingsApplied`.

## Features

- **One upstream connection**: Deepgram runs STT, the LLM and TTS behind one WebSocket, billed by Deepgram. No OpenAI, Anthropic or Google key is needed for Deepgram-managed LLMs.
- **Models are configuration**: listen, think and speak models are `.env` values in Deepgram's own vocabulary (see [Model Configuration](#model-configuration)); changing them needs no code changes.
- **Zero transcoding**: Plivo μ-law 8kHz is forwarded as-is, and Deepgram returns μ-law 8kHz ready for `playAudio`.
- **Server-side turn detection**: Deepgram owns turn-taking (Flux end-of-turn by default), so there is no client-side VAD and no torch, Silero or ONNX dependencies.
- **Barge-in**: `UserStartedSpeaking` → queue drain + `clearAudio` (with `streamId`), plus a drop gate for late audio from the cancelled response.
- **Function calling**: order status, SMS, callbacks, transfers and `end_call`, all executed client-side through `FunctionCallRequest`/`FunctionCallResponse`.
- **Graceful hangup**: the goodbye plays in full, the checkpoint is acknowledged, and then the call is hung up via REST.
- **Inbound and outbound**: the inbound server auto-configures the Plivo application and number webhooks on startup; the outbound server passes per-call answer/hangup URLs and tracks status by call and campaign.
- **Structured events**: `call_answered`, `user_text`, `agent_text`, `turn_complete` (with Deepgram `LatencyReport` fields) and `session_end`.
- **Saved agent configurations (optional)**: publish the agent block once from code and reference it by UUID per direction; see [Saved agent configurations](#saved-agent-configurations-optional).
- **Text injection**: Plivo `{"event": "text", "text": "..."}` messages are forwarded as Deepgram `InjectUserMessage` (used by the E2E tests).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A Deepgram API key with Voice Agent API access ([console.deepgram.com](https://console.deepgram.com))
- A Plivo account with a phone number
- ngrok, for local development

## Quick Start

### 1. Install dependencies

```bash
cd deepgram-voiceagent
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
DEEPGRAM_API_KEY=your_deepgram_api_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

### 3. Start ngrok

```bash
ngrok http 8000
```

Copy the HTTPS URL into `PUBLIC_URL`.

### 4. Run the server

```bash
# Inbound (receive calls) — auto-configures the Plivo answer/hangup webhooks
uv run python -m inbound.server

# Outbound (place calls) — no number auto-config; answer/hangup URLs are set per call
uv run python -m outbound.server
```

### 5. Make a test call

- **Inbound**: call your Plivo number. The greeting plays right after the Deepgram handshake.
- **Outbound**:

  ```bash
  curl -X POST -G "http://localhost:8000/outbound/call" \
    --data-urlencode "phone_number=+1234567890" \
    --data-urlencode "opening_reason=you requested a demo"
  ```

## Project Structure

```
deepgram-voiceagent/
├── inbound/
│   ├── __init__.py
│   ├── agent.py            # DeepgramVoiceAgent, FUNCTION_DEFINITIONS, _build_settings(), run_agent()
│   ├── server.py           # FastAPI: /, /answer, /ws, /hangup, /fallback, /hold + _hangup_call()
│   └── system_prompt.md    # Inbound system prompt
├── outbound/
│   ├── __init__.py
│   ├── agent.py            # Same agent + OutboundCallRecord, CallManager, build_outbound_prompt()
│   ├── server.py           # FastAPI: /outbound/call, /outbound/answer, /ws, status, hangup, campaign
│   └── system_prompt.md    # Outbound prompt ({{opening_reason}}, {{objective}}, {{context}})
├── utils.py                # μ-law codec, resample_audio, plivo_to_deepgram/deepgram_to_plivo (pass-through), normalize_phone_number
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── helpers.py          # ngrok, recording, transcription
│   ├── test_integration.py # Unit + local + Deepgram/Plivo integration
│   ├── test_observability.py
│   ├── test_e2e_live.py    # Real Deepgram, no phone call
│   ├── test_live_call.py   # Real inbound call
│   ├── test_multiturn_voice.py  # Multi-turn + barge-in on a real call
│   └── test_outbound_call.py    # Real outbound call
├── pyproject.toml
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

## How It Works

```
┌─────────┐     ┌─────────────┐     ┌──────────────────────────┐     ┌──────────────────────────────┐
│  Phone  │────▶│   Plivo     │────▶│  FastAPI server (/ws)    │────▶│  Deepgram Voice Agent API    │
│  (PSTN) │◀────│  Gateway    │◀────│  DeepgramVoiceAgent      │◀────│  wss://agent.deepgram.com    │
└─────────┘     └─────────────┘     │                          │     │                              │
                 μ-law 8kHz JSON    │  plivo_rx   ─ binary ──▶ │     │  listen: flux-general-en *   │
                 (base64, 20ms)     │  deepgram_rx ◀─ binary ─ │     │  think:  gpt-4.1-mini *      │
                                    │  plivo_tx  (160B chunks) │     │  speak:  aura-2-thalia-en *  │
                                    └──────────────────────────┘     └──────────────────────────────┘
                                          μ-law 8kHz binary frames + JSON control events
```

\* Default models; each is set in `.env` (see [Model Configuration](#model-configuration)).

Three concurrent asyncio tasks, following the canonical `FIRST_COMPLETED` pattern:

| Task | Method | Role |
|------|--------|------|
| `plivo_rx` | `_receive_from_plivo` | `media` → binary frame to Deepgram (buffered until `SettingsApplied`, capped at 2s / 16000B); `text` → `InjectUserMessage`; `playedStream` → `_on_played_stream()`; `clearedAudio`; `stop` ends the task |
| `deepgram_rx` | `_receive_from_deepgram` | `_handshake()`, then `_handle_deepgram_event()` for JSON; binary audio → `_on_agent_audio()` → send queue |
| `plivo_tx` | `_send_to_plivo` | 160-byte `playAudio` chunks and `_Checkpoint` sentinels; after 0.1s with an empty queue runs `_idle_housekeeping()` (`KeepAlive` if nothing sent upstream for 5s, hangup deadline) |

### Call sequence

1. `/answer` (inbound) or `/outbound/answer` (outbound) returns `<Stream bidirectional keepCallAlive contentType="audio/x-mulaw;rate=8000">` pointing at `/ws`. Call metadata (`call_uuid`, `from`, `to`, `parent_call_uuid`, `sip_headers`, plus `is_outbound`/`call_id` for outbound) travels as base64 JSON in `?body=`.
2. `/ws` waits for Plivo `start` (`callId`, `streamId`), then calls `run_agent(...)` with `hangup_callback=functools.partial(_hangup_call, callId)`. Outbound loads `system_prompt` and `initial_message` from the `CallManager` record.
3. **Handshake** (`_handshake()`): `Welcome{request_id}` → send `Settings` exactly once → `SettingsApplied`, all within 10s. No audio or text goes upstream before `SettingsApplied`; buffered input is flushed right after.
4. **Greeting**: Deepgram speaks `agent.greeting`. Binary μ-law audio may arrive before `ConversationText{assistant}` (→ `agent_text`, turn 1). `AgentAudioDone` → a `_Checkpoint` is queued behind the last chunk → Plivo `playedStream` → `turn_complete`.
5. **User turn**: Flux detects end-of-turn → `ConversationText{user}` (→ `user_text`, TTFS clock starts) and `EndOfTurn` → `ConversationText{assistant}` (→ `agent_text`) and binary audio, interleaved with single-key `LatencyReport` messages → `AgentAudioDone` → checkpoint → `playedStream` → `turn_complete`.
6. **Barge-in**: `UserStartedSpeaking` drains the send queue and `_tx_buffer`, sends `clearAudio`, and forgets the pending checkpoint. If audio was playing, it also emits `turn_complete(barge_in=true)` and sets the drop gate (see Implementation Notes).
7. **Function call**: `FunctionCallRequest{functions:[{id, name, arguments, client_side}]}` → `_handle_function_call(name, arguments)` for each `client_side` entry (10s timeout) → `FunctionCallResponse{id, name, content}`.
8. **Hangup**: `end_call` sets a pending hangup with a 15s deadline; the LLM then speaks its goodbye → `AgentAudioDone` → checkpoint → `playedStream` → `_finish_hangup()` → `hangup_callback()` (Plivo `calls.delete`) → tasks stop → `session_end`.

## What Runs When

| When | File | What it does |
|---|---|---|
| Module import | `inbound/agent.py`, `outbound/agent.py` | `load_dotenv()`, read the `DEEPGRAM_*` config, load `system_prompt.md`, define `FUNCTION_DEFINITIONS` and `build_agent_config()`. No network calls. |
| Server start: `uv run python -m inbound.server` | `inbound/server.py` → `main()` | Logging sinks (text/JSON/file/Redis) and optional OTel are set up when the module loads. Then `check_saved_agent_config()` runs (see [Startup check](#startup-check)), then `configure_plivo_webhooks()` creates or updates the Plivo application and assigns `PLIVO_PHONE_NUMBER` when `PUBLIC_URL` is set, then uvicorn starts. |
| Server start: `uv run python -m outbound.server` | `outbound/server.py` → `main()` | Same startup check, then uvicorn. There is no number auto-config, because answer and hangup URLs are passed per call. |
| `POST /outbound/call` | `outbound/server.py` | `CallManager.create_call()` builds the per-call prompt and greeting, then Plivo `calls.create` dials the number. |
| Plivo answers (`/answer` or `/outbound/answer`) | `server.py` | Returns `<Stream bidirectional keepCallAlive>` XML that points Plivo at `/ws`. Call metadata travels in `?body=`. |
| Each call (`/ws`) | `server.py` → `run_agent()` in `agent.py` | Accepts the WebSocket, reads Plivo's `start` event, then runs `DeepgramVoiceAgent.run()`. That opens a **new** Deepgram WebSocket for the call, sends Settings (inline block or saved UUID), personalizes a saved session, and runs the `plivo_rx` / `deepgram_rx` / `plivo_tx` tasks until the call ends. There is no Deepgram connection before a call arrives. |
| `end_call` tool | `agent.py` → `hangup_callback` | After the goodbye has played, the agent calls `_hangup_call()` from `server.py`, which hangs up via the Plivo REST API. Plivo credentials never leave `server.py`. |
| Manual CLI: `uv run python -m inbound.agent --publish` / `--list` / `--delete` | `agent.py` → `main()` | Manages saved agent configurations over the Deepgram REST API. No server is started. |
| Shared helpers | `utils.py` | μ-law codec, resampling, `plivo_to_deepgram` / `deepgram_to_plivo` (pass-through), phone normalization. |

## Audio Formats

| Hop | Format | Sample Rate | Frame Size | Notes |
|-----|--------|-------------|------------|-------|
| Plivo → Agent | μ-law (base64 JSON) | 8 kHz | 160 bytes (20ms) | `media` events |
| Agent → Deepgram | μ-law (binary WS frame) | 8 kHz | 160 bytes | `plivo_to_deepgram()` is a pass-through |
| Deepgram → Agent | μ-law (binary WS frame, `container: none`) | 8 kHz | variable | raw, no WAV header |
| Agent → Plivo | μ-law (base64 JSON `playAudio`) | 8 kHz | 160 bytes (20ms) | `deepgram_to_plivo()` is a pass-through; the tail is padded with `0xFF` (μ-law silence) |

`utils.py` still ships `ulaw_to_pcm`, `pcm_to_ulaw` and `resample_audio` (numpy + scipy). The agent does not call them; they are there for tests and for switching to a `linear16` encoding. The `0xFF` padding is applied only to a partial final chunk when a checkpoint is dequeued.

## Deepgram Settings

`_build_settings()` builds the single `Settings` message sent after `Welcome`:

```json
{
  "type": "Settings",
  "tags": ["plivo", "deepgram-voiceagent"],
  "audio": {
    "input":  {"encoding": "mulaw", "sample_rate": 8000},
    "output": {"encoding": "mulaw", "sample_rate": 8000, "container": "none"}
  },
  "agent": {
    "listen": {
      "provider": {
        "type": "deepgram", "version": "v2", "model": "flux-general-en",
        "eot_threshold": 0.7, "eot_timeout_ms": 5000
      }
    },
    "think": {
      "provider": {"type": "open_ai", "model": "gpt-4.1-mini", "temperature": 0.7},
      "prompt": "<_build_system_prompt(): system prompt + '## Current Call Context' when from_number is set>",
      "functions": "<FUNCTION_DEFINITIONS: 5 entries, see Function Calling>"
    },
    "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
    "greeting": "<initial_message: AGENT_GREETING (inbound) or the CallManager greeting (outbound)>"
  }
}
```

- The listen/think/speak values come from the `DEEPGRAM_LISTEN_*`, `DEEPGRAM_THINK_*` and `DEEPGRAM_SPEAK_*` env vars (defaults shown); see [Model Configuration](#model-configuration). `tags` is `["plivo", EXAMPLE_NAME]`.
- `agent.greeting` is spoken **verbatim** by TTS; it is not an instruction to the LLM. The outbound `CallManager.create_call()` therefore builds a literal greeting from `opening_reason`, or uses `DEFAULT_OUTBOUND_GREETING`.
- Do **not** add `agent.language` or `speak.provider.language`. With this configuration Deepgram replies with an `Error` and the session ends.
- `eot_threshold` sets how confident Flux must be before ending the turn; lower values reply faster but risk cutting the caller off. `eot_timeout_ms` forces an end of turn after that much silence, whatever the confidence.
- `build_agent_config()` returns the `agent` block without `greeting` and without call context. Inline mode is `build_agent_config(prompt=<per-call prompt>)` plus `greeting`; `--publish` stores `build_agent_config()` as is.

## Saved agent configurations (optional)

Deepgram can store the `agent` block of Settings as a reusable **agent configuration** under your project, identified by a UUID. Settings then carries `"agent": "<uuid>"` instead of the inline block. This example supports it opt-in, per direction; with the variables empty (the default) nothing changes.

Use it when:

- several servers or deployments should run the **same pinned config**, and you want to see which one a call used (`session_end.agent_config`);
- you want **rollback by switching an ID**: publish a new config, point the env var at it, and point it back if needed.

Trade-offs:

- **Immutable.** Only metadata can change. Any change to the prompt, functions or models means publishing a new UUID and updating the env var. Old configs stay until you delete them.
- **API only.** There is no console UI for these configs. Create, list and delete them through the REST API (this example's CLI wraps it).
- **Visible to every project member.** Never put secrets in the prompt or function definitions.
- **Per-call context arrives after connect.** Reference by UUID is all-or-nothing: you cannot mix it with inline `agent` fields. The caller context (and, for outbound calls, the campaign details) is appended with `UpdatePrompt`, and the greeting is sent with `InjectAgentMessage`. Both go out right after `SettingsApplied`, before buffered caller audio is flushed. In live checks the injected greeting started about 0.6 s after `SettingsApplied`, against about 0.1 s for an inline `agent.greeting`. The delay comes from `InjectAgentMessage` itself, not from `UpdatePrompt`.
- **Env config is baked in at publish time.** The saved config captures the `DEEPGRAM_LISTEN_*`, `DEEPGRAM_THINK_*` and `DEEPGRAM_SPEAK_*` values and `SYSTEM_PROMPT` as they were when you ran `--publish`. Changing those env vars later has no effect until you publish again.

### Publish, use, delete

```bash
# Create a config from this code's settings (no server is started); prints the UUID and the .env line
uv run python -m inbound.agent --publish     # -> DEEPGRAM_INBOUND_AGENT_ID=<uuid>
uv run python -m outbound.agent --publish    # -> DEEPGRAM_OUTBOUND_AGENT_ID=<uuid>

# List the project's configs (UUID + metadata), delete old ones
uv run python -m inbound.agent --list
uv run python -m inbound.agent --delete <uuid>
```

Put the printed line in `.env` and restart the server. The session start log shows `settings: saved agent config <uuid>` (or `settings: inline`). The project comes from `DEEPGRAM_PROJECT_ID`, or else from the single project that `GET /v1/projects` returns for the key. If the key sees several projects, the CLI stops and asks you to set `DEEPGRAM_PROJECT_ID`. Each config is created with the metadata `{"example": "deepgram-voiceagent", "direction": "inbound" | "outbound"}`.

**Key scopes.** Creating and deleting configs needs a `DEEPGRAM_API_KEY` with the `agent:write` scope, and listing needs `agent:read`. A key without these scopes gets `403 INSUFFICIENT_PERMISSIONS`, which the CLI reports along with the missing scope. Running calls against a saved config uses the same single key; no separate admin key is needed.

**What is published.** The published config is `build_agent_config()`: the listen, think and speak providers, `FUNCTION_DEFINITIONS` and the static base prompt. It has no greeting and no per-call context.

- **Inbound:** the base prompt is `system_prompt.md` (or `SYSTEM_PROMPT`). Each call appends the `## Current Call Context` block (caller number, call ID, time) with `UpdatePrompt`, then injects `AGENT_GREETING`.
- **Outbound:** the template's `{{opening_reason}}`, `{{objective}}` and `{{context}}` placeholders are replaced with pointers such as `[the objective under "This Call" below]`. Each call appends a `## This Call` section with `UpdatePrompt`. That section holds the greeting already spoken, the opening reason, the objective and the additional context, followed by the call context. The `CallManager` greeting is then injected. Inline mode still renders the template exactly as before.

### Startup check

When the server starts (`main()` in `inbound/server.py` / `outbound/server.py`, before uvicorn accepts calls), it runs `check_saved_agent_config()` from the matching `agent.py`:

- **Inline mode** (no agent ID): it logs `Deepgram agent settings: inline (listen=… think=… speak=… functions=5)` and makes no API call.
- **Saved mode**: it fetches the config (`GET .../agents/{uuid}`, which needs `agent:read`) and logs the models the config actually contains. It then warns about drift between the config and what `--publish` would create from the deployed code and env:
  - a model env var that is **explicitly set** but differs from the config, e.g. `DEEPGRAM_THINK_MODEL=gpt-5.4-mini is ignored in saved mode: the saved config has think.provider.model='gpt-4.1-mini'. Re-publish to apply it.`;
  - a base prompt or `FUNCTION_DEFINITIONS` that differs from the code (function names only in the config or only in the code are listed).
- **Unknown ID** (404/400): the server logs how to fix it and **exits with code 1**. Without the check, Deepgram would answer every call's Settings with `INTERNAL_SERVER_ERROR` ("resolving agent ID") and each call would end without a greeting.
- **Can't verify** (e.g. a key without `agent:read`, or a network error): it logs a warning and starts anyway.

### How the agent ID and the model env vars interact

With an agent ID set, Deepgram receives **only the ID**. The model env vars matter only when `--publish` runs, because that is when their values are copied into the config. The inbound direction is shown here; outbound works the same with `DEEPGRAM_OUTBOUND_AGENT_ID`.

| # | `DEEPGRAM_INBOUND_AGENT_ID` | Model env vars | Settings sent to Deepgram | Models used on calls | Startup check |
|---|---|---|---|---|---|
| 1 | unset | unset | full inline `agent` block | defaults (`flux-general-en`, `open_ai`/`gpt-4.1-mini`, `aura-2-thalia-en`) | logs "inline" |
| 2 | unset | e.g. `DEEPGRAM_THINK_MODEL=gpt-5.4-mini` | inline block with that model | `gpt-5.4-mini` (after restart) | logs "inline" |
| 3 | set | unset, or equal to the published values | `"agent": "<uuid>"`, then `UpdatePrompt` + `InjectAgentMessage` | the values in effect at `--publish` time | logs the saved models |
| 4 | set | changed after publishing | the UUID only | still the published models | ⚠️ warns that the env var is ignored |
| 5 | set | `system_prompt.md` / `SYSTEM_PROMPT` or `FUNCTION_DEFINITIONS` changed without re-publishing | the UUID only | the published prompt and functions | ⚠️ warns about prompt/function drift |
| 6 | set to a deleted or wrong ID | any | — | — | ❌ server exits with code 1 |
| 7 | inbound set, outbound unset | set | inbound: UUID; outbound: inline | inbound: published models; outbound: current env | each server checks its own direction |
| 8 | either | `AGENT_GREETING` set | inline: `agent.greeting`; saved: `InjectAgentMessage` | n/a | n/a |
| 9 | n/a | `DEEPGRAM_PROJECT_ID` | not used on calls | n/a | used by the check and by `--publish`/`--list`/`--delete` |

To change a model or the prompt in saved mode: run `--publish` again (you get a new UUID), put the new UUID in the env var, restart the server, and delete the old config.

**Turn handling.** The injected greeting comes back as `ConversationText` (role `assistant`) followed by `AgentAudioDone`, the same events as an inline greeting. It therefore still counts as turn 1 and emits `agent_text` and `turn_complete`.

Behaviour verified live (2026-09-23):

- `POST /v1/projects/{project_id}/agents` with `{"config": "<agent block as a JSON string>", "metadata": {...}}` returns `{"agent_uuid": "..."}`. Deepgram's docs say `agent_id`, so the code accepts both.
- `GET .../agents` returns a plain JSON list. `GET .../agents/{uuid}` returns `agent_uuid`, `member_id`, `api_version`, `config` and `metadata`. `DELETE .../agents/{uuid}` returns 200.
- In a UUID session, `UpdatePrompt` gets a `PromptUpdated` reply and **appends** to the saved prompt. The LLM answered "What phone number am I calling from?" using the appended caller number.
- `InjectAgentMessage` is spoken verbatim. A saved config without `greeting` stays silent until it arrives.
- Client-side functions from the saved config arrive as `FunctionCallRequest` exactly as in inline mode, and `FunctionCallResponse` works unchanged.

## Outbound Call API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/outbound/call` | POST | Initiate a call. Query params: `phone_number` (required), `campaign_id`, `opening_reason`, `objective`, `context` |
| `/outbound/answer` | GET/POST | Plivo answer webhook (set per call, `?call_id=`); returns `<Stream>` to `/ws` |
| `/outbound/hangup` | POST | Plivo hangup webhook; sets `completed` and `outcome` via `determine_outcome()` |
| `/outbound/status/{call_id}` | GET | Call status and details |
| `/outbound/hangup/{call_id}` | POST | End a `ringing`/`connected` call via Plivo REST |
| `/outbound/campaign/{campaign_id}` | GET | All calls for a campaign |
| `/ws` | WebSocket | Plivo audio stream (shared path with inbound) |
| `/hold`, `/` | GET/POST, GET | Silent 120s `<Wait>`; health check |

Status flow: `initiating` → `ringing` → `connected` → `completed` (or `failed`). `outcome` is one of `success`, `no_answer`, `busy`, `failed`. Errors are returned as `{"error": ...}` JSON bodies.

```bash
curl -X POST "http://localhost:8000/outbound/call" \
  -G \
  --data-urlencode "phone_number=+1234567890" \
  --data-urlencode "campaign_id=demo-campaign" \
  --data-urlencode "opening_reason=your recent demo request for TechFlow Teams" \
  --data-urlencode "objective=qualify interest and book a meeting with sales"
```

With `opening_reason`, the greeting becomes: *"Hi, this is Alex from TechFlow. I'm reaching out because {opening_reason}. Is now a good time for a quick chat?"* `build_outbound_prompt(opening_reason, objective, context)` substitutes `{{opening_reason}}`, `{{objective}}` and `{{context}}` into `outbound/system_prompt.md`.

## Function Calling

The functions are declared in `agent.think.functions` without an `endpoint`, so Deepgram sends them to the client (`client_side: true`).

| Function | Description | `defer_until_eot` |
|----------|-------------|-------------------|
| `check_order_status(order_number, email)` | Look up an order (mock data) | — |
| `send_sms(phone_number, message)` | Send a text message (mock) | — |
| `schedule_callback(phone_number, reason, preferred_time, department)` | Schedule a specialist callback (mock) | — |
| `transfer_call(department, reason)` | Transfer to a human (mock; no Plivo transfer) | yes |
| `end_call(reason, resolution)` | Handled inline: sets the pending hangup, returns `{"status": "call_ending"}` | yes |

To add a function, append a `{"name", "description", "parameters"}` JSON-schema entry to `FUNCTION_DEFINITIONS` and a branch in `DeepgramVoiceAgent._handle_function_call()`, in both `inbound/agent.py` and `outbound/agent.py`. `arguments` may arrive as a JSON string or a dict. Whatever the handler returns is serialized with `json.dumps` into `FunctionCallResponse.content`; a timeout returns `{"error": "function timed out"}`.

## Observability

Each structured event is a loguru record bound with `event=<name>` and `call_id` (the Plivo `ParentCallUUID` when present, otherwise the stream's `callId`). Records go to stderr (text), or JSON on stderr with `LOG_FORMAT=json`, plus an optional JSON file sink (`LOG_FILE`, 100 MB rotation, 7-day retention) and Redis Streams (`REDIS_EVENTS_URL` → `XADD` to `REDIS_STREAM_KEY`, maxlen ~10000; needs `--extra streaming`).

| Event | When | Key fields |
|-------|------|------------|
| `call_answered` | Session starts, before connecting to Deepgram | `call_id`, `leg_call_id`, `from_number`, `to_number`, `sip_headers`, `stream_id` |
| `user_text` | `ConversationText{user}` or a Plivo `text` injection (its Deepgram echo is deduplicated) | `call_id`, `turn`, `text` |
| `agent_text` | Each `ConversationText{assistant}` | `call_id`, `turn`, `text` |
| `turn_complete` | `playedStream` for the turn's checkpoint (or `AgentAudioDone` when there is no `streamId`), or a barge-in | `call_id`, `turn`, `barge_in`, `user_text`, `agent_text`, `plivo_rx_bytes`, `plivo_tx_chunks`, `playback_ms`, `total_latency_ms`, `tts_latency_ms`, `ttt_latency_ms`, `latency_report` |
| `session_end` | Teardown | `call_id`, `duration_s`, `turns`, `barge_ins`, `errors`, `ttfs_avg_ms`, `ttfs_samples`, `rx_bytes`, `tx_chunks`, `deepgram_request_id` |

Latency fields come from Deepgram `LatencyReport` messages (seconds, converted to ms). Every numeric key is merged into `latency_report` as `<key>_ms` (`stt_latency_ms`, `ttt_token_latency_ms`, `ttt_text_latency_ms`, `ttt_tool_latency_ms`, `tts_latency_ms`, `total_latency_ms`), and the report resets at each new user turn. `total_latency_ms` and `tts_latency_ms` are copied to the top-level fields; `ttt_latency_ms` takes `ttt_text_latency`. `AgentStartedSpeaking` latencies are also read if Deepgram sends them. `playback_ms` is the time from sending the checkpoint to `playedStream`. TTFS is measured on the client, from `ConversationText{user}` (or text injection) to the next `playAudio` chunk.

`LOG_LEVEL` is read by `agent.py` and gates only the agent's pipeline logs. Structured events, the session-start line, warnings and errors are logged at every level.

| `LOG_LEVEL` | What is logged |
|-------------|----------------|
| `verbose` | Everything in `normal`, plus `Settings sent`, `AgentThinking`, `EndOfTurn` trigger, `History`, non-STT `LatencyReport`s, unhandled event types (e.g. the `FunctionCallResponse` echo), `KeepAlive` sent, checkpoint sent/stale, dropped late audio, packet counts every 500 and queue size |
| `normal` (default) | `Welcome`, `SettingsApplied` time and flush counts, first audio in/out, user/agent text per turn, tool calls and results, barge-ins, `playedStream` + playback time, TTFS, hangup |
| `quiet` | Only structured events, session start, warnings (`Warning`, `InjectionRefused`) and errors |

OpenTelemetry: run `uv sync --extra observability`. The agent wraps `run()` in a single `session` span (tracer `voice-agent`, attribute `call_id` = first 8 characters). Spans are exported over OTLP gRPC only when `OTEL_EXPORTER_OTLP_ENDPOINT` is set. The server also enables Traceloop (OpenLLMetry) and httpx auto-instrumentation when those packages are installed.

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `DEEPGRAM_API_KEY` | Deepgram API key (sent as `Authorization: Token`) | Required |
| `DEEPGRAM_AGENT_URL` | Voice Agent WebSocket URL (EU: `wss://api.eu.deepgram.com/v1/agent/converse`) | `wss://agent.deepgram.com/v1/agent/converse` |
| `DEEPGRAM_LISTEN_MODEL` | `agent.listen.provider.model` | `flux-general-en` |
| `DEEPGRAM_LISTEN_EOT_THRESHOLD` | `eot_threshold` (sent only for `flux-*` models) | `0.7` |
| `DEEPGRAM_LISTEN_EOT_TIMEOUT_MS` | `eot_timeout_ms` (sent only for `flux-*` models) | `5000` |
| `DEEPGRAM_LISTEN_LANGUAGE` | `language` (sent only for non-Flux models, when set) | — |
| `DEEPGRAM_THINK_PROVIDER` | `agent.think.provider.type` | `open_ai` |
| `DEEPGRAM_THINK_MODEL` | `agent.think.provider.model` | `gpt-4.1-mini` |
| `DEEPGRAM_THINK_TEMPERATURE` | `agent.think.provider.temperature` | `0.7` |
| `DEEPGRAM_SPEAK_MODEL` | `agent.speak.provider.model` (`flux-*` voices also send `version: v2`) | `aura-2-thalia-en` |
| `AGENT_GREETING` | Inbound greeting, spoken verbatim (outbound ignores it) | `Hi, this is Alex from TechFlow. I'm built with the Deepgram Voice Agent API on Plivo. How can I help you today?` |
| `DEEPGRAM_INBOUND_AGENT_ID` | Saved agent config UUID for inbound calls (`inbound.agent --publish`); empty = inline Settings | — |
| `DEEPGRAM_OUTBOUND_AGENT_ID` | Saved agent config UUID for outbound calls (`outbound.agent --publish`); empty = inline Settings | — |
| `DEEPGRAM_PROJECT_ID` | Project for `--publish`/`--list`/`--delete`; empty = the key's only project | — |
| `SYSTEM_PROMPT` | Inbound: replaces `system_prompt.md`. Outbound: used only when no `CallManager` record is found (records always use `build_outbound_prompt()`) | — |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token | Required |
| `PLIVO_PHONE_NUMBER` | Plivo number (inbound auto-config, and caller ID for outbound) | Required |
| `PLIVO_TEST_NUMBER` | Second Plivo number for live call tests | — |
| `PUBLIC_URL` | Public HTTPS URL for webhooks; `https://` → `wss://` for the stream URL | Required |
| `SERVER_PORT` | Server port | `8000` |
| `DEFAULT_COUNTRY_CODE` | Default region for phone parsing | `US` |
| `LOG_LEVEL` | Agent pipeline log verbosity (`verbose` / `normal` / `quiet`) | `normal` |
| `LOG_FORMAT` | `json` replaces the stderr sink with serialized JSON | `text` |
| `LOG_FILE` | Path for an additional JSON log sink | — |
| `REDIS_EVENTS_URL` | Redis URL for the Streams sink | — |
| `REDIS_STREAM_KEY` | Redis stream key | `voice-agent:events` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for tracing | — |

## Model Configuration

The Voice Agent API is a managed pipeline, but you choose the model for each stage. The `DEEPGRAM_*` model variables map one-to-one onto fields of Deepgram's `Settings` message and take **Deepgram's own values verbatim**: model ids and provider types exactly as Deepgram lists them. The example adds no aliases of its own. It adds only the protocol fields Deepgram requires for a model family:

- **Listen**: `flux-*` models are sent with `version: v2` plus `eot_threshold`/`eot_timeout_ms`. Other models (for example `nova-3`) are sent without them, plus `language` if `DEEPGRAM_LISTEN_LANGUAGE` is set.
- **Think**: `DEEPGRAM_THINK_PROVIDER` is Deepgram's `think.provider.type` (`open_ai`, `anthropic`, `google`, `groq`, `nvidia`, …). `DEEPGRAM_THINK_MODEL` is a model id Deepgram supports for that provider. Deepgram publishes the live list at `GET https://agent.deepgram.com/v1/agent/settings/think/models` (with your API key) and in its [LLM models docs](https://developers.deepgram.com/docs/voice-agent-llm-models). The LLM you choose sets the pricing tier.
- **Speak**: any Deepgram voice id from the [TTS models list](https://developers.deepgram.com/docs/voice-agent-tts-models). `aura-2-*` voices use the default v1 API; `flux-*` voices are sent with `version: v2`. Never add `language` to a Deepgram speak provider, because Deepgram rejects it. Non-Deepgram speak providers (Cartesia, ElevenLabs, OpenAI and others) use a different provider shape and, for most of them, your own endpoint and keys. They need a code change in `_build_speak_provider()`.

### Tested combinations

Each row was checked live against Deepgram (2026-09-22) using this example's `_build_settings()`. Every combination produced `SettingsApplied`, a spoken greeting, and a correct spoken answer to an injected question ("What plans do you offer?"), with no `Error` or `Warning` events. "First audio" is the time from `InjectUserMessage` to the first agent audio byte, from one sample per combination, so treat it as indicative only.

| Listen | Think (`provider` / `model`) | Speak | First audio | Notes |
|---|---|---|---|---|
| `flux-general-en` | `open_ai` / `gpt-4.1-mini` | `aura-2-thalia-en` | ~1.0 s | **Default**; also verified on real Plivo calls (see [Testing](#testing)) |
| `flux-general-en` | `open_ai` / `gpt-5.4-mini` | `aura-2-thalia-en` | ~0.9 s | |
| `flux-general-en` | `google` / `gemini-2.5-flash` | `aura-2-thalia-en` | ~1.1 s | |
| `flux-general-en` | `open_ai` / `gpt-4.1-mini` | `flux-alexis-en` | ~1.0 s | Flux TTS voice (sent with `version: v2`) |
| `nova-3` (`DEEPGRAM_LISTEN_LANGUAGE=en`) | `anthropic` / `claude-haiku-4-5` | `aura-2-thalia-en` | ~0.7 s | No Flux EOT params are sent; end-of-turn comes from Deepgram's non-Flux listen path. Claude tended to add markdown, which the prompt's "no markdown" rule guards against |

Only the default row went through the full Plivo call suites. For any other combination, run `tests/test_e2e_live.py` with your `.env` values before relying on it.

## Dependencies

- Runtime: `fastapi`, `uvicorn[standard]`, `websockets>=15.0`, `plivo`, `python-dotenv`, `python-multipart`, `loguru`, `numpy`, `scipy`, `phonenumbers`. No torch, Silero, ONNX, OpenAI or Deepgram SDK.
- `observability` extra: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `opentelemetry-instrumentation-httpx`, `traceloop-sdk`.
- `streaming` extra: `redis[hiredis]`.
- `dev` group: `ruff`, `pre-commit`, `pytest`, `pytest-asyncio`, `httpx`, `faster-whisper`, `gtts`, `pydub`, `audioop-lts` (Python 3.13+).

## Pricing

The Voice Agent API bills per minute of connected session, and the rate depends on the LLM tier. `gpt-4.1-mini` falls in Deepgram's **Standard** tier, about **$0.075/min** pay-as-you-go, covering Flux STT, the hosted LLM and Aura-2 TTS. Plivo telephony is billed separately. Rates change, so check [deepgram.com/pricing](https://deepgram.com/pricing) for current numbers. Choosing a larger LLM (for example `gpt-4.1`) moves the session to a higher tier.

## Testing

Observed latency (from `turn_complete` / `session_end` in the live call and multi-turn tests):

| Metric | Median | Range | Notes |
|---|---|---|---|
| TTFS (user end-of-turn → first `playAudio` to Plivo) | ~950 ms | 611–1917 ms | 1917 ms turn included a tool call |
| Deepgram `total_latency` | ~1030 ms | 731–1397 ms | from `LatencyReport` |
| Deepgram `ttt_*_latency` (LLM) | — | 514–1206 ms | `gpt-4.1-mini`; Deepgram may emit `SLOW_THINK_REQUEST` warnings above ~5 s |
| Deepgram `tts_latency` | — | 93–127 ms | Aura-2 |
| `end_call` → REST hangup | ~4.4 s | — | goodbye spoken + `playedStream`, then `calls.delete` (`NORMAL_CLEARING`) |

Run from this directory:

```bash
uv sync   # includes the dev group
uv run ruff check .

# Unit tests (offline, no API keys)
uv run pytest tests/test_integration.py tests/test_observability.py -v -k unit

# Local server + direct Deepgram handshake/greeting (needs DEEPGRAM_API_KEY)
uv run pytest tests/test_integration.py -v -k "local or Deepgram"

# Saved agent config: publish -> connect by UUID -> UpdatePrompt/InjectAgentMessage -> delete
# (needs agent:read/agent:write scopes; skipped with the reason otherwise)
uv run pytest tests/test_integration.py -v -s -k SavedConfigIntegration

# E2E with real Deepgram, no phone call (greeting, text question, end_call).
# Runs every test twice: [inline] and [saved] (publishes a config for the run, then deletes it)
uv run pytest tests/test_e2e_live.py -v -s

# Real calls (need Plivo creds, PLIVO_TEST_NUMBER, ngrok)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
uv run pytest tests/test_multiturn_voice.py -v -s
```

The live call tests use `PLIVO_TEST_NUMBER`, a second Plivo number on the same account. It is the caller for inbound tests and the destination for outbound tests.

From the repo root:

```bash
./scripts/validate-example.sh deepgram-voiceagent
```

## Deployment

### Docker

```bash
docker build -t deepgram-voiceagent .

# Inbound (default)
docker run -p 8000:8000 --env-file .env deepgram-voiceagent

# Outbound
docker run -p 8000:8000 --env-file .env deepgram-voiceagent \
  uv run python -m outbound.server
```

The image (default `python:3.12-slim`) runs `uv sync --locked --no-install-project --no-dev --extra streaming`, so the Redis sink is available; the `observability` extra is not installed. A different base image can be passed with `--build-arg BASE_IMAGE=...`.

## Troubleshooting

### 401 / handshake rejected

Deepgram expects `Authorization: Token <key>`, not `Bearer`. Also check that the key's project has Voice Agent API access.

### `Error` right after `Settings`

A field in `Settings` is invalid. The most common cause is an `agent.language` or `speak.provider.language` field, which this configuration rejects. Other causes are an unknown model id or provider type (`DEEPGRAM_LISTEN_MODEL`, `DEEPGRAM_THINK_PROVIDER`/`DEEPGRAM_THINK_MODEL`, `DEEPGRAM_SPEAK_MODEL`) or a malformed function schema. The `Error` event's `description` field names the problem.

### Silence on the call / no greeting

- Audio was sent before `SettingsApplied`; it must be buffered until then.
- The encoding does not match: input and output must both be `mulaw` at 8000 Hz, with output `container: "none"`. With a WAV container, a header gets played as noise.
- The `playAudio` payload is malformed: `contentType` must be `audio/x-mulaw`, `sampleRate` must be a separate field set to 8000, and chunks must be 160 bytes.

### Session drops when no audio is flowing

Deepgram closes sessions that receive nothing. Plivo media normally streams continuously, but if it stops, `plivo_tx` sends `{"type":"KeepAlive"}` once nothing has been sent upstream for 5s (visible with `LOG_LEVEL=verbose`).

### Agent repeats its greeting / speaks the prompt

`agent.greeting` is spoken verbatim. Put instructions in the system prompt, not in the greeting.

### Call does not hang up after goodbye

The hangup waits for `playedStream` on the goodbye's checkpoint; if Plivo never acknowledges it, the 15s deadline (checked in `_idle_housekeeping()`) forces it. `_hangup_call()` skips the REST call when `PLIVO_AUTH_ID`/`PLIVO_AUTH_TOKEN` are missing (logged as `Skipping REST hangup`); otherwise look for `hangup callback ERROR` in the logs.

## Implementation Notes

Behaviour observed in live testing against Deepgram, and how the code handles it:

1. **Settings once, nothing before `SettingsApplied`.** Media and text that arrive during the handshake are buffered (audio capped at 2s / 16000B, oldest dropped) and flushed before `_settings_applied` is set, so live input cannot overtake them.
2. **No `AgentStartedSpeaking` / `AgentThinking` in practice.** Latency arrives only as single-key `LatencyReport` messages: `stt_latency` (roughly every audio frame), then `ttt_text_latency` / `ttt_token_latency` / `ttt_tool_latency`, `tts_latency` and `total_latency`, all in seconds. `_on_latency_report()` merges them into `turn_complete`'s `latency_report` and `*_latency_ms` fields. The `AgentStartedSpeaking` handler is kept in case Deepgram sends it.
3. **Greeting audio before its text.** Binary greeting audio can arrive before `ConversationText{assistant}`. The code does not wait for the text; `_on_agent_audio()` starts playback on the first frame, and turn 1 is assigned whichever arrives.
4. **`InjectUserMessage` echo.** Deepgram echoes an injected message back as `ConversationText{user}` and sends `EndOfTurn{trigger: "manual"}`. The echo is matched against `_last_injected_text` and skipped, so the turn is counted once. Injecting while audio is playing triggers a local barge-in first.
5. **Drop gate after barge-in.** A few binary frames of the cancelled response can still arrive after `UserStartedSpeaking`. `_drop_agent_audio` discards them and clears on `ConversationText{user}`, `EndOfTurn`, `AgentThinking`, `FunctionCallRequest` or `AgentStartedSpeaking`. It is deliberately **not** cleared by `ConversationText{assistant}`, because the new response's first frame can precede it and a late sentence of the cancelled response can follow it.
6. **`end_call` ordering.** The LLM calls `end_call` first and then speaks the goodbye. The function only marks the hangup as pending (15s deadline); the hangup happens at the next `playedStream` for the goodbye's checkpoint, or at the deadline. If the goodbye is cut off by a barge-in, the hangup runs on that response's `AgentAudioDone`.
7. **Echoes and history.** Deepgram echoes each `FunctionCallResponse` back and sends `History` events. Both are only logged at `LOG_LEVEL=verbose`.
8. **Checkpoint through the queue.** `AgentAudioDone` enqueues a `_Checkpoint` sentinel behind the audio, so Plivo's `checkpoint` is always sent after the last chunk. A `playedStream` whose name does not match the pending checkpoint is ignored. With no `streamId` (local tests), `AgentAudioDone` completes playback directly.
9. **Hangup is owned by the server.** The agent never reads Plivo credentials. `server.py` passes `hangup_callback=functools.partial(_hangup_call, callId)`, which runs `calls.delete` in a thread.
10. **Python 3.10.** Timeouts catch `(TimeoutError, asyncio.TimeoutError)`, and the Deepgram WebSocket connects with `max_size=None` and `ping_interval=20`.
