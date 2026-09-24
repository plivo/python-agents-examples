# Deepgram Voice Agent API — Plivo Voice Agent

Plivo bidirectional audio streaming bridged over raw `websockets` + asyncio (no SDK/framework) to the Deepgram Voice Agent API (`wss://agent.deepgram.com/v1/agent/converse`, `Authorization: Token`), a managed listen → think → speak pipeline on one WebSocket. Plivo μ-law 8kHz (base64 JSON, 160B/20ms) is forwarded unchanged as binary frames and Settings request raw μ-law 8kHz output (`container: none`), so there is no transcoding or resampling in either direction.
- Models are `.env` config passed verbatim into Settings. Defaults: listen Deepgram Flux `flux-general-en` (`version: v2`, `eot_threshold` 0.7, `eot_timeout_ms` 5000) for end-of-turn detection with no client-side VAD; think `open_ai` `gpt-4.1-mini` hosted by Deepgram with 5 client-side functions (`end_call`/`transfer_call` use `defer_until_eot`); speak Aura-2 `aura-2-thalia-en`.
- Barge-in on Deepgram `UserStartedSpeaking` while audio is playing: Deepgram cancels its LLM/TTS; the client drains the send queue, sends Plivo `clearAudio` and drops late agent audio until the next user end-of-turn (`ConversationText` user / `EndOfTurn`).
- Playback end is tracked with Plivo `checkpoint`/`playedStream`; `end_call` hangs up via Plivo REST `calls.delete` after the goodbye's checkpoint is played (15s fallback deadline).
- Two agent-definition paths per direction: inline (default) sends the full `agent` block in Settings with `agent.greeting`; with `DEEPGRAM_INBOUND_AGENT_ID`/`DEEPGRAM_OUTBOUND_AGENT_ID` set to a Deepgram reusable agent config UUID, Settings sends `agent: "<uuid>"`, then `UpdatePrompt` (per-call context; outbound adds a "This Call" section) and `InjectAgentMessage` (greeting) follow `SettingsApplied`.

## Features

- **One upstream connection**: Deepgram runs STT, the LLM and TTS behind one WebSocket, billed by Deepgram. No OpenAI, Anthropic or Google key is needed for Deepgram-managed LLMs.
- **Models are configuration**: listen, think and speak models are `.env` values in Deepgram's own vocabulary (see [Model Configuration](#model-configuration)); changing them needs no code changes.
- **Zero transcoding**: Plivo μ-law 8kHz is forwarded as-is, and Deepgram returns μ-law 8kHz ready for `playAudio`.
- **Server-side turn detection**: Deepgram owns turn-taking (Flux end-of-turn by default), so there is no client-side VAD and no torch, Silero or ONNX dependencies.
- **Barge-in**: `UserStartedSpeaking` → queue drain + `clearAudio` (with `streamId`), plus a drop gate for late audio from the cancelled response.
- **Function calling**: order status, SMS, callbacks, transfers and `end_call`, all executed client-side through `FunctionCallRequest`/`FunctionCallResponse`.
- **Graceful hangup**: the goodbye plays in full, the checkpoint is acknowledged, and then the call is hung up via REST.
- **Inbound and outbound**: the inbound server auto-configures the Plivo application and number webhooks on startup. Outbound calls are placed with Plivo's Make Call API directly; the per-call context (`opening_reason`, `objective`, `context`) rides on the `answer_url` query string, and the outbound server only answers the webhooks and bridges audio.
- **Structured events**: `call_answered`, `user_text`, `agent_text`, `turn_complete` (with Deepgram `LatencyReport` fields) and `session_end`.
- **Inline or reusable agent config**: send the agent definition in every `Settings` (default), or store it once in your Deepgram project and reference it by UUID; see [Choosing a path](#choosing-a-path-inline-vs-reusable-agent-config).
- **Text injection**: Plivo `{"event": "text", "text": "..."}` messages are forwarded as Deepgram `InjectUserMessage` (used by the E2E tests).

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- A Deepgram API key with Voice Agent API access ([console.deepgram.com](https://console.deepgram.com))
- A Plivo account with a phone number
- For a local run, `cloudflared` for `--tunnel` (or ngrok); for hosting, Docker

## Quick Start

Deepgram has no telephony, so a small bridge server has to sit between Plivo and Deepgram. That server is this example. Plivo just needs a public URL to reach it.

```
Caller ──► Plivo ══ WebSocket ══► this server (bridge) ══ WebSocket ══► Deepgram Voice Agent
                                        ▲
                         reachable at PUBLIC_URL (tunnel or host)
```

| | Option A: one command (local) | Option B: Docker (any host) |
|---|---|---|
| **Use for** | Trying it on your laptop | Running on a server |
| **Public URL** | Created for you (free Cloudflare quick tunnel) | Your host's HTTPS URL |
| **Needs** | `uv`, `cloudflared` | Docker, a host that allows long-lived WebSockets |
| **Plivo number setup** | Automatic | Automatic when `PUBLIC_URL` is set |

For both options, first put four values in `.env`:

```bash
git clone https://github.com/plivo/python-agents-examples && cd python-agents-examples/deepgram-voiceagent
cp .env.example .env   # set DEEPGRAM_API_KEY, PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER
```

### Option A: one command, local

[Install `cloudflared`](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/) once, then start the agent with a tunnel:

```bash
uv run python -m inbound.server --tunnel
```

What `--tunnel` does:

1. Starts `cloudflared tunnel --url http://localhost:$SERVER_PORT` (inbound, default 8000; the outbound server uses `$OUTBOUND_SERVER_PORT`, default 8001). The quick tunnel needs no Cloudflare account.
2. Uses the `https://<random>.trycloudflare.com` URL as `PUBLIC_URL`.
3. Points your Plivo number at it, creating or updating the `Deepgram_VoiceAgent` Plivo application. Plivo only accepts a URL once it can resolve the hostname, so for a new tunnel the server retries in the background for up to 3 minutes. In testing this took about 70 s.
4. Logs `Ready! Call +<number> to talk to the agent (Ctrl+C to stop)` once the server accepts connections and Plivo has accepted the URL.

Call the number. Ctrl+C stops the server and the tunnel. The URL changes on every run and the number is re-pointed each time.

For outbound calls, run `uv run python -m outbound.server --tunnel`. It listens on port 8001, so it can run at the same time as the inbound server, each with its own tunnel URL. Once it accepts connections it logs `Ready!` with a ready-to-paste cURL for Plivo's Make Call API with this server's answer URL; see [Outbound Calls](#outbound-calls). In short:

```bash
set -a && source .env && set +a   # PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN in the shell
curl -X POST "https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/" \
  -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"from": "<your Plivo number>", "to": "+1234567890",
       "answer_url": "https://<random>.trycloudflare.com/outbound/answer?opening_reason=you%20requested%20a%20demo",
       "hangup_url": "https://<random>.trycloudflare.com/outbound/hangup",
       "answer_method": "POST", "hangup_method": "POST"}'
```

### Option B: Docker

```bash
docker build -t deepgram-voiceagent .
docker run --env-file .env -e PUBLIC_URL=https://your-host.example.com -p 8000:8000 deepgram-voiceagent
```

The container listens on port 8000 and configures the Plivo number from `PUBLIC_URL` on startup. For outbound calls, append `uv run python -m outbound.server` to `docker run` and publish port 8001 instead:

```bash
docker run --env-file .env -e PUBLIC_URL=https://your-outbound-host.example.com -p 8001:8001 deepgram-voiceagent \
  uv run python -m outbound.server
```

Then place calls with Plivo's Make Call API using `answer_url=https://your-outbound-host.example.com/outbound/answer`. See [Deployment](#deployment) for suitable hosts.

### Other ways to expose the server

Without `--tunnel`, any tunnel works. For example, run `ngrok http 8000`, set `PUBLIC_URL` to its HTTPS URL, then `uv run python -m inbound.server`. For outbound, tunnel port 8001 (`ngrok http 8001`) instead.

To run both servers at once behind fixed URLs, give each process its own `PUBLIC_URL` on the command line, e.g. `PUBLIC_URL=https://out.example.com uv run python -m outbound.server`. A variable set in the shell takes precedence over `.env`, because `load_dotenv()` does not override variables that are already set.

## Project Structure

```
deepgram-voiceagent/
├── inbound/
│   ├── __init__.py
│   ├── agent.py            # DeepgramVoiceAgent, FUNCTION_DEFINITIONS, _build_settings(), run_agent()
│   ├── server.py           # FastAPI: /, /answer, /ws, /hangup, /fallback, /hold + _hangup_call(), Plivo auto-config, startup log of the agent path + reusable-config ID check
│   └── system_prompt.md    # Inbound system prompt
├── outbound/
│   ├── __init__.py
│   ├── agent.py            # Same agent + build_outbound_prompt(), build_outbound_greeting(), outbound call-context labels
│   ├── server.py           # FastAPI: /, /outbound/answer (answer_url call details -> Stream body), /outbound/hangup (logs), /ws + _hangup_call()
│   └── system_prompt.md    # Outbound prompt ({{greeting}}, {{opening_reason}}, {{objective}}, {{context}})
├── utils.py                # μ-law codec, resample_audio, plivo_to_deepgram/deepgram_to_plivo (pass-through), normalize_phone_number, --tunnel helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── helpers.py          # ngrok, recording, transcription, Plivo webhook signing, Deepgram agent-config REST helpers
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

1. `/answer` (inbound) or `/outbound/answer` (outbound) returns `<Stream bidirectional keepCallAlive contentType="audio/x-mulaw;rate=8000">` pointing at `/ws`. Call metadata (`call_uuid`, `from`, `to`, `parent_call_uuid`, `sip_headers`, plus the `answer_url`'s `opening_reason`/`objective`/`context` for outbound) travels as base64 JSON in `?body=`.
2. `/ws` waits for Plivo `start` (`callId`, `streamId`), then calls `run_agent(...)` with `hangup_callback=functools.partial(_hangup_call, callId)`. Outbound `run_agent()` renders the prompt (`build_outbound_prompt()`) and greeting (`build_outbound_greeting()`) from the call details.
3. **Handshake** (`_handshake()`): `Welcome{request_id}` → send `Settings` exactly once → `SettingsApplied`, all within 10s. No audio or text goes upstream before `SettingsApplied`; buffered input is flushed right after.
4. **Greeting**: Deepgram speaks `agent.greeting`. Binary μ-law audio may arrive before `ConversationText{assistant}` (→ `agent_text`, turn 1). `AgentAudioDone` → a `_Checkpoint` is queued behind the last chunk → Plivo `playedStream` → `turn_complete`.
5. **User turn**: Flux detects end-of-turn → `ConversationText{user}` (→ `user_text`, TTFS clock starts) and `EndOfTurn` → `ConversationText{assistant}` (→ `agent_text`) and binary audio, interleaved with single-key `LatencyReport` messages → `AgentAudioDone` → checkpoint → `playedStream` → `turn_complete`.
6. **Barge-in**: `UserStartedSpeaking` drains the send queue and `_tx_buffer`, sends `clearAudio`, and forgets the pending checkpoint. If audio was playing, it also emits `turn_complete(barge_in=true)` and sets the drop gate (see Implementation Notes).
7. **Function call**: `FunctionCallRequest{functions:[{id, name, arguments, client_side}]}` → `_handle_function_call(name, arguments)` for each `client_side` entry (10s timeout) → `FunctionCallResponse{id, name, content}`.
8. **Hangup**: `end_call` sets a pending hangup with a 15s deadline; the LLM then speaks its goodbye → `AgentAudioDone` → checkpoint → `playedStream` → `_finish_hangup()` → `hangup_callback()` (Plivo `calls.delete`) → tasks stop → `session_end`.

## What Runs When

| When | File | What it does |
|---|---|---|
| Module import | `inbound/agent.py`, `outbound/agent.py` | `load_dotenv()`, read the `DEEPGRAM_*` config, load `system_prompt.md`, define `FUNCTION_DEFINITIONS`. No network calls. |
| Server start: `uv run python -m inbound.server` | `inbound/server.py` → `main()` | Logging sinks (text/JSON/file/Redis) and optional OTel are set up when the module loads. Then `describe_deepgram_agent()` logs the active path, e.g. `Deepgram agent: inline (listen=flux-general-en, think=open_ai/gpt-4.1-mini, speak=aura-2-thalia-en)` or `Deepgram agent: reusable config <uuid> …`. On the reusable path, `verify_deepgram_agent_id()` looks the UUID up in the API key's Deepgram project (`GET /v1/projects`, then `/projects/{id}/agents/{uuid}`) and **exits with code 1 if it isn't found**; when found, it keeps the config's model names for trace attributes. It warns and starts anyway if it can't check (network error, key without `agent:read`). `check_webhook_auth_config()` then **exits with code 1 if `PLIVO_AUTH_TOKEN` is empty** (see [Webhook authentication](#webhook-authentication)). Then `configure_plivo_webhooks()` creates or updates the Plivo application and assigns `PLIVO_PHONE_NUMBER` when `PUBLIC_URL` is set (with `--tunnel`, in a background thread that retries until Plivo accepts the new URL), then uvicorn starts. `Ready! Call +N to talk to the agent (Ctrl+C to stop)` is logged once, when both the server accepts connections (a local TCP connect to `SERVER_PORT`; the lifespan hook runs before uvicorn binds) and the Plivo setup succeeded. No Ready line if the Plivo setup fails or is skipped. |
| Server start: `uv run python -m outbound.server` | `outbound/server.py` → `main()` | Same agent-path log line, reusable-config ID check and webhook-auth check, then uvicorn on `OUTBOUND_SERVER_PORT` (default 8001). Once the server accepts connections it logs `Ready!` with a cURL for Plivo's Make Call API and this server's answer URL, ending with `(Ctrl+C to stop)`. There is no number auto-config, because you pass the answer and hangup URLs with each call. |
| You place a call (Plivo Make Call API) | your shell / backend → Plivo | `POST https://api.plivo.com/v1/Account/{auth_id}/Call/` with `from`, `to`, `answer_url=<PUBLIC_URL>/outbound/answer?opening_reason=…&objective=…&context=…` and optional `hangup_url`. The server has no dial endpoint and keeps no call records. |
| Plivo answers (`/answer` or `/outbound/answer`) | `server.py` | `verify_plivo_signature()` checks Plivo's V3 signature (403 if invalid). Then returns `<Stream bidirectional keepCallAlive>` XML that points Plivo at `/ws`. Call metadata (and, outbound, the `answer_url` call details) travels as base64 JSON in `?body=`, percent-encoded so a `+` in the base64 isn't read as a space. |
| Each call (`/ws`) | `server.py` → `run_agent()` in `agent.py` | Accepts the WebSocket, reads Plivo's `start` event, then runs `DeepgramVoiceAgent.run()`. That opens a **new** Deepgram WebSocket for the call, sends Settings (inline block or reusable config UUID), sends `UpdatePrompt` + `InjectAgentMessage` on the reusable path, and runs the `plivo_rx` / `deepgram_rx` / `plivo_tx` tasks until the call ends. There is no Deepgram connection before a call arrives. |
| `end_call` tool | `agent.py` → `hangup_callback` | After the goodbye has played, the agent calls `_hangup_call()` from `server.py`, which hangs up via the Plivo REST API. Plivo credentials never leave `server.py`. |
| One-off setup (optional): create a reusable config | your shell → Deepgram REST API | The `curl` commands in [Creating a reusable config](#creating-a-reusable-config) post this example's agent definition once. The example contains no code for it; servers only read the UUID from the env var and check at startup that it exists. |
| Shared helpers | `utils.py` | μ-law codec, resampling, `plivo_to_deepgram` / `deepgram_to_plivo` (pass-through), phone normalization. |

## Webhook authentication

Both servers are exposed on a public URL, so they accept only requests that come from Plivo. The check below always runs; there is no setting to turn it off:

- **Plivo webhooks** (`/answer`, `/hangup`, `/fallback`, `/hold`, `/outbound/answer`, `/outbound/hangup`) must carry a valid [Plivo V3 signature](https://www.plivo.com/docs/voice/concepts/signature-validation) (`X-Plivo-Signature-V3` + `X-Plivo-Signature-V3-Nonce`, an HMAC-SHA256 signature keyed with `PLIVO_AUTH_TOKEN`). `verify_plivo_signature()`, a FastAPI dependency in each `server.py`, checks them with the Plivo SDK's `validate_v3_signature()`. The signed params are the form fields for POST and the query string for GET. Otherwise the request gets **403** and a warning is logged with the path and reason (`missing …` or `signature mismatch`). Signatures are never logged.
- **`/ws`** is not a Plivo webhook, so its handshake is not signed, and it has no extra check of its own, like the other examples in this repo. The stream URL (`wss://…/ws?body=<metadata>`) is only handed to Plivo in the signed answer webhook's `<Stream>` XML.

**`PUBLIC_URL` must be exactly the URL Plivo is configured to call.** Plivo signs the URL it requested. Behind a tunnel or reverse proxy, this server sees `http://localhost:8000/...` instead, so the signed URL is rebuilt as `PUBLIC_URL` (trailing `/` stripped) + request path + raw query string. `request.url` is not used. The scheme (`https` vs `http`), the host and any path prefix must match the answer and hangup URLs in the Plivo application or Make Call request. `PUBLIC_URL=https://agent.example.com` and `https://agent.example.com/` are equivalent. `http://agent.example.com` or another hostname for the same server is not. The outbound `answer_url` query string (`opening_reason`, `objective`, `context`) is part of what Plivo signs: for a POST, the Plivo SDK's base string is `<url>?<query params sorted, URL-decoded>.<form params sorted, name+value concatenated>`. Editing the query string therefore invalidates the signature. This was verified live against Plivo's own signatures (`Plivo signature verified: POST /outbound/answer + query string` in the log).

**`--tunnel`**: the quick-tunnel URL becomes `PUBLIC_URL` at startup, and the verifier reads it on each request, so signatures are checked against the tunnel URL. The inbound server points the number at that same URL. For outbound, use the tunnel URL from the `Ready!` line in your `answer_url`.

**`PLIVO_AUTH_TOKEN` is required**: with it empty, the server refuses to start. Tests use a dummy token and sign their requests the way Plivo does. Use a subaccount's auth token if the number and calls belong to a subaccount, because Plivo signs with the token of the account that owns the call.

## Audio Formats

| Hop | Format | Sample Rate | Frame Size | Notes |
|-----|--------|-------------|------------|-------|
| Plivo → Agent | μ-law (base64 JSON) | 8 kHz | 160 bytes (20ms) | `media` events |
| Agent → Deepgram | μ-law (binary WS frame) | 8 kHz | 160 bytes | `plivo_to_deepgram()` is a pass-through |
| Deepgram → Agent | μ-law (binary WS frame, `container: none`) | 8 kHz | variable | raw, no WAV header |
| Agent → Plivo | μ-law (base64 JSON `playAudio`) | 8 kHz | 160 bytes (20ms) | `deepgram_to_plivo()` is a pass-through; the tail is padded with `0xFF` (μ-law silence) |

`utils.py` also provides `ulaw_to_pcm`, `pcm_to_ulaw` and `resample_audio` (numpy + scipy). The agent does not call them; the tests use them, and they are what switching to a `linear16` encoding would need. The `0xFF` padding is applied only to a partial final chunk when a checkpoint is dequeued.

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
      "prompt": "<_build_system_prompt(): system prompt + '## Current Call Context' when the numbers are known>",
      "functions": "<FUNCTION_DEFINITIONS: 5 entries, see Function Calling>"
    },
    "speak": {"provider": {"type": "deepgram", "model": "aura-2-thalia-en"}},
    "greeting": "<initial_message: AGENT_GREETING (inbound) or build_outbound_greeting() (outbound)>"
  }
}
```

- The listen/think/speak values come from the `DEEPGRAM_LISTEN_*`, `DEEPGRAM_THINK_*` and `DEEPGRAM_SPEAK_*` env vars (defaults shown); see [Model Configuration](#model-configuration). `tags` is `["plivo", EXAMPLE_NAME]`.
- `agent.greeting` is spoken **verbatim** by TTS; it is not an instruction to the LLM. Outbound `build_outbound_greeting()` therefore builds a literal greeting from `opening_reason`, or uses `DEFAULT_OUTBOUND_GREETING`.
- The call context labels the numbers by direction. Inbound: `Caller's phone number` (Plivo `From`). Outbound: `Customer's phone number (the person you called)` (`To`, used for SMS and callbacks) and `Our business phone number (caller ID the customer sees)` (`From`, given if the customer asks how to reach us).
- Do **not** add `agent.language` or `speak.provider.language`. With this configuration Deepgram replies with an `Error` and the session ends.
- `eot_threshold` sets how confident Flux must be before ending the turn; lower values reply faster but risk cutting the caller off. `eot_timeout_ms` forces an end of turn after that much silence, whatever the confidence.
- This is Path 1 (inline). With a reusable config UUID set, `agent` is just that UUID string; see [Choosing a path](#choosing-a-path-inline-vs-reusable-agent-config).

## Choosing a path: inline vs reusable agent config

The agent definition (listen/think/speak providers, prompt, functions) can reach Deepgram in two ways. You choose per direction with one env var; nothing else in the code changes.

| | Path 1: inline (default) | Path 2: reusable agent config |
|---|---|---|
| **How to choose** | `DEEPGRAM_INBOUND_AGENT_ID` / `DEEPGRAM_OUTBOUND_AGENT_ID` empty | Set the var to a config UUID |
| **Where the config lives** | This repo: `DEEPGRAM_*` env vars, `system_prompt.md`, `FUNCTION_DEFINITIONS`, built by `_build_settings()` on every call | Your Deepgram project, stored once from this repo's code (see below) |
| **Good for** | Development, one deployment, changes shipped with the code | Several deployments pinned to one config, per-customer configs, A/B tests, rollback by switching a UUID |
| **How to change the config** | Edit env/code, restart | Create a new config, point the env var at the new UUID, restart, delete the old one (configs are immutable) |
| **Greeting latency** (after `SettingsApplied`, measured) | ~0.1 s (`agent.greeting`) | ~0.6 s (`InjectAgentMessage`) |
| **Tested on real Plivo calls** | ✅ | ✅ |

```
Path 1: inline                                 Path 2: reusable config
Welcome                                        Welcome
Settings {agent: {listen, think, speak,        Settings {agent: "<uuid>"}
  prompt + call context, functions, greeting}} SettingsApplied
SettingsApplied                                UpdatePrompt        (call context; outbound: "This Call")
greeting audio                                 InjectAgentMessage  (greeting)
                                               greeting audio
```

On Path 2 a UUID reference is all-or-nothing: no inline `agent` fields can be mixed in. So right after `SettingsApplied`, and before buffered caller audio is flushed, the agent appends the per-call context with `UpdatePrompt` and speaks the greeting with `InjectAgentMessage`. The injected greeting comes back as `ConversationText` (assistant) plus `AgentAudioDone`, like an inline greeting, so it is still turn 1. The extra ~0.5 s comes from `InjectAgentMessage`, not from `UpdatePrompt`. The session start log shows `settings: saved agent config <uuid>` or `settings: inline`, and `session_end.agent_config` carries the UUID or `inline`.

[Deepgram's reusable agent configurations](https://developers.deepgram.com/docs/reusable-agent-configurations) exist so that an agent definition can be stored once and referenced by ID instead of being resent in every `Settings`. Deepgram lists these use cases: per-customer configs, regional or regulatory compliance, A/B testing voices or prompts, and multi-agent architectures.

With a UUID set, Deepgram receives **only the UUID**. The `DEEPGRAM_LISTEN_*`, `DEEPGRAM_THINK_*`, `DEEPGRAM_SPEAK_*` vars, the prompt file (`inbound/system_prompt.md`; outbound: `outbound/system_prompt.md` rendered by `build_outbound_prompt()` with pointers to "This Call") and `FUNCTION_DEFINITIONS` matter only at the moment you create the config. `AGENT_GREETING` (inbound), the outbound greeting and "This Call" details, and the per-call context still apply on every call.

### Creating a reusable config

This is a one-off setup step, done with Deepgram's REST API. The example has no code for it; the server only reads the UUID and checks at startup that it exists. The key needs the `agent:write` scope to create or delete and `agent:read` to list. Run from this directory:

```bash
set -a && source .env && set +a   # DEEPGRAM_API_KEY (and any DEEPGRAM_* model vars) in the shell

# Project id (the first project; pick another from the list if your key sees several)
export DG_PROJECT_ID=$(curl -s https://api.deepgram.com/v1/projects \
  -H "Authorization: Token $DEEPGRAM_API_KEY" \
  | uv run python -c 'import json, sys; print(json.load(sys.stdin)["projects"][0]["project_id"])')
```

Each command below prints this example's own inline agent definition (from `_build_settings()`, without the greeting or per-call context) as the create body and posts it. It returns `{"agent_uuid": "<uuid>"}`; put that UUID in `.env` and restart the server.

**Inbound** (`DEEPGRAM_INBOUND_AGENT_ID`):

```bash
uv run python - <<'EOF' | curl -sS -X POST "https://api.deepgram.com/v1/projects/$DG_PROJECT_ID/agents" \
  -H "Authorization: Token $DEEPGRAM_API_KEY" -H "Content-Type: application/json" -d @-
import json
from inbound.agent import DeepgramVoiceAgent

agent = DeepgramVoiceAgent(websocket=None, call_id="create-config", agent_config_id="")
config = agent._build_settings()["agent"]
del config["greeting"]  # sent per call with InjectAgentMessage
meta = {"example": "deepgram-voiceagent", "direction": "inbound"}
print(json.dumps({"config": json.dumps(config), "metadata": meta}))
EOF
```

**Outbound** (`DEEPGRAM_OUTBOUND_AGENT_ID`). The saved prompt cannot hold one call's details, so the template's `{{greeting}}`, `{{opening_reason}}`, `{{objective}}` and `{{context}}` are filled with pointers to the `## This Call` section. Each call appends that section with `UpdatePrompt`: the greeting already spoken, the opening reason, the objective and the extra context (neutral wording for any the `answer_url` did not carry), followed by the call context.

```bash
uv run python - <<'EOF' | curl -sS -X POST "https://api.deepgram.com/v1/projects/$DG_PROJECT_ID/agents" \
  -H "Authorization: Token $DEEPGRAM_API_KEY" -H "Content-Type: application/json" -d @-
import json
from outbound.agent import DeepgramVoiceAgent, build_outbound_prompt

prompt = build_outbound_prompt(
    opening_reason='[the opening reason under "This Call" below]',
    objective='[the objective under "This Call" below]',
    context='See "This Call" below.',
    greeting='the greeting quoted under "This Call" below.',
)
agent = DeepgramVoiceAgent(
    websocket=None, call_id="create-config", system_prompt=prompt, agent_config_id=""
)
config = agent._build_settings()["agent"]
del config["greeting"]  # sent per call with InjectAgentMessage
meta = {"example": "deepgram-voiceagent", "direction": "outbound"}
print(json.dumps({"config": json.dumps(config), "metadata": meta}))
EOF
```

List and delete:

```bash
curl -s "https://api.deepgram.com/v1/projects/$DG_PROJECT_ID/agents" \
  -H "Authorization: Token $DEEPGRAM_API_KEY"

curl -sS -X DELETE "https://api.deepgram.com/v1/projects/$DG_PROJECT_ID/agents/<uuid>" \
  -H "Authorization: Token $DEEPGRAM_API_KEY"
```

**Tools caveat.** The function schemas live in the Deepgram config, but the handlers run in `agent.py` (`_handle_function_call()`). A reusable config must reference only tools this example implements, which the commands above guarantee because they copy `FUNCTION_DEFINITIONS`. If a config offers a tool that `agent.py` does not implement and the LLM calls it, the call does not crash: the agent replies with a `FunctionCallResponse` whose content is `{"error": "Unknown function: <name>"}`, and the LLM carries on. The reverse also holds: a tool missing from the config is never called. Without `end_call` in the config, for example, the agent can't hang up. After editing `FUNCTION_DEFINITIONS`, create a new config.

### Facts verified live (2026-09-23)

- The create response field is `agent_uuid` (Deepgram's docs say `agent_id`).
- Create requires `metadata` as well as `config` (a JSON **string**). Without `metadata` it returns `400 missing field 'metadata'`.
- List returns a plain JSON list (`[]` when empty). `DELETE .../agents/{uuid}` returns 200.
- Configs are immutable. `PUT .../agents/{uuid}` updates only `metadata`, which is required. A `config` in the `PUT` body is silently ignored, and the response is still 200.
- There is no console UI for these configs yet; use the REST API.
- Configs, including their prompts, and variables are visible to all members of the project. Don't put secrets in them.
- The API key needs `agent:write` to create and delete, and `agent:read` to list. Without them, the API returns `403 INSUFFICIENT_PERMISSIONS`. Calls that use a config need only the normal key.
- In a UUID session, `UpdatePrompt` gets a `PromptUpdated` reply and **appends** to the saved prompt. The LLM answered "What phone number am I calling from?" from the appended caller number. `InjectAgentMessage` is spoken verbatim, and a config without `greeting` stays silent until it arrives. Client-side functions arrive as `FunctionCallRequest` exactly as they do inline.

## Outbound Calls

You place outbound calls with [Plivo's Make Call API](https://www.plivo.com/docs/voice/api/call/make-a-call). This server has no dial endpoint and keeps no call records: it answers Plivo's webhooks and bridges audio.

```
your shell / backend ──POST /v1/Account/{auth_id}/Call/──► Plivo ──dials──► callee
                                                             │ callee answers
                        /outbound/answer?opening_reason=…  ◄─┘ (answer_url)
                          └─► <Stream> ─► /ws ─► run_agent(opening_reason, objective, context)
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/outbound/answer` | GET/POST | Plivo answer webhook (your `answer_url`). Reads the optional query params `opening_reason`, `objective` and `context`, plus Plivo's `CallUUID`/`From`/`To`/`ParentCallUUID`/`SIP-*` fields, and returns `<Stream>` to `/ws` with them in the base64 `body` |
| `/outbound/hangup` | POST | Plivo hangup webhook (your optional `hangup_url`); logs `CallUUID`, `Duration`, `HangupCause` |
| `/ws` | WebSocket | Plivo audio stream; runs the agent |
| `/` | GET | Health check |

Once it accepts connections, the server logs a `Ready!` line with the cURL below, its own `PUBLIC_URL` and `PLIVO_PHONE_NUMBER` filled in. Credentials stay shell references (`set -a && source .env && set +a` exports them):

```bash
curl -X POST "https://api.plivo.com/v1/Account/$PLIVO_AUTH_ID/Call/" \
  -u "$PLIVO_AUTH_ID:$PLIVO_AUTH_TOKEN" -H "Content-Type: application/json" \
  -d '{"from": "+14155550100", "to": "+1234567890",
       "answer_url": "https://your-host.example.com/outbound/answer?opening_reason=your%20recent%20demo%20request&objective=book%20a%20meeting%20with%20sales",
       "hangup_url": "https://your-host.example.com/outbound/hangup",
       "answer_method": "POST", "hangup_method": "POST"}'
```

URL-encode each query value. From Python, `plivo.RestClient().calls.create(from_=..., to_=..., answer_url=..., answer_method="POST", hangup_url=...)` does the same (see `tests/test_outbound_call.py`).

| `answer_url` param | Used for | When missing |
|---|---|---|
| `opening_reason` | Greeting: *"Hi, this is Alex from TechFlow. I'm reaching out because {opening_reason}. Is now a good time for a quick chat?"*; the prompt's "you are calling about" | `DEFAULT_OUTBOUND_GREETING` ("…following up on your recent interest in our products…") and a matching neutral reason |
| `objective` | The prompt's objective | Qualify interest and offer a meeting with sales |
| `context` | The prompt's "Additional Context" | "No additional context was provided for this call." |

`build_outbound_prompt()` fills `{{greeting}}` (the exact greeting spoken), `{{opening_reason}}`, `{{objective}}` and `{{context}}` in `outbound/system_prompt.md`, so the LLM never sees an unfilled placeholder. On the reusable-config path the same values go into the "This Call" `UpdatePrompt` section. For status, duration or recordings, use Plivo's Call API and call logs, or your `hangup_url`.

## Function Calling

The functions are declared in `agent.think.functions` without an `endpoint`, so Deepgram sends them to the client (`client_side: true`).

| Function | Description | `defer_until_eot` |
|----------|-------------|-------------------|
| `check_order_status(order_number, email)` | Look up an order (mock data) | — |
| `send_sms(phone_number, message)` | Send a text message (mock) | — |
| `schedule_callback(phone_number, reason, preferred_time, department)` | Schedule a specialist callback (mock) | — |
| `transfer_call(department, reason)` | Transfer to a human (mock; no Plivo transfer) | yes |
| `end_call(reason, resolution)` | Handled inline: sets the pending hangup, returns `{"status": "call_ending"}` | yes |

To add a function, append a `{"name", "description", "parameters"}` JSON-schema entry to `FUNCTION_DEFINITIONS` and a branch in `DeepgramVoiceAgent._handle_function_call()`, in both `inbound/agent.py` and `outbound/agent.py`. On the reusable-config path, create a new config afterwards so Deepgram offers the new schema (see the [tools caveat](#creating-a-reusable-config)). `arguments` may arrive as a JSON string or a dict. Whatever the handler returns is serialized with `json.dumps` into `FunctionCallResponse.content`; a timeout returns `{"error": "function timed out"}`.

## Observability

Two independent outputs. Only the console log is on by default:

| Output | Where it goes | Who uses it | Enable |
|--------|---------------|-------------|--------|
| **Structured loguru events** (`call_answered`, `user_text`, `agent_text`, `turn_complete`, `session_end`) | the console (stderr; text, or JSON with `LOG_FORMAT=json`), a JSONL file (`LOG_FILE`, 100 MB rotation, 7-day retention), and the Redis stream `voice-agent:events` (`REDIS_EVENTS_URL` → `XADD` to `REDIS_STREAM_KEY`, maxlen ~10000) | Dashboards and the hosting app read the Redis stream for live transcripts and per-turn metrics; the file and JSON console output feed log pipelines | Console always; `LOG_FILE`; `REDIS_EVENTS_URL` + `--extra streaming` |
| **OpenTelemetry spans** (tracer `voice-agent`) | Any OTLP backend (Jaeger, Grafana Tempo, Honeycomb, Datadog, …) over OTLP gRPC | Latency debugging: where a turn's time went, per call | `uv sync --extra observability` and set `OTEL_EXPORTER_OTLP_ENDPOINT` |

### Structured events

Each event is a loguru record bound with `event=<name>` and `call_id` (the Plivo `ParentCallUUID` when present, otherwise the stream's `callId`).

| Event | When | Key fields |
|-------|------|------------|
| `call_answered` | Session starts, before connecting to Deepgram | `call_id`, `leg_call_id`, `from_number`, `to_number`, `sip_headers`, `stream_id` |
| `user_text` | `ConversationText{user}` or a Plivo `text` injection (its Deepgram echo is deduplicated) | `call_id`, `turn`, `text` |
| `agent_text` | Each `ConversationText{assistant}` | `call_id`, `turn`, `text` |
| `turn_complete` | `playedStream` for the turn's checkpoint (or `AgentAudioDone` when there is no `streamId`), or a barge-in | `call_id`, `turn`, `barge_in`, `user_text`, `agent_text`, `plivo_rx_bytes`, `plivo_tx_chunks`, `playback_ms`, `total_latency_ms`, `tts_latency_ms`, `ttt_latency_ms`, `latency_report` |
| `session_end` | Teardown | `call_id`, `duration_s`, `turns`, `barge_ins`, `errors`, `ttfs_avg_ms`, `ttfs_samples`, `rx_bytes`, `tx_chunks`, `deepgram_request_id`, `agent_config` (UUID or `inline`) |

Latency fields come from Deepgram `LatencyReport` messages (seconds, converted to ms). Every numeric key is merged into `latency_report` as `<key>_ms` (`stt_latency_ms`, `ttt_token_latency_ms`, `ttt_text_latency_ms`, `ttt_tool_latency_ms`, `tts_latency_ms`, `total_latency_ms`), and the report resets at each new user turn. `total_latency_ms` and `tts_latency_ms` are copied to the top-level fields; `ttt_latency_ms` takes `ttt_text_latency`. `AgentStartedSpeaking` latencies are also read if Deepgram sends them. `playback_ms` is the time from sending the checkpoint to `playedStream`. TTFS is measured on the client, from `ConversationText{user}` (or text injection) to the next `playAudio` chunk.

`LOG_LEVEL` is read by `agent.py` and gates only the agent's pipeline logs. Structured events, the session-start line, warnings and errors are logged at every level.

| `LOG_LEVEL` | What is logged |
|-------------|----------------|
| `verbose` | Everything in `normal`, plus `Settings sent`, `AgentThinking`, `EndOfTurn` trigger, `History`, non-STT `LatencyReport`s, unhandled event types (e.g. the `FunctionCallResponse` echo), `KeepAlive` sent, checkpoint sent/stale, dropped late audio, packet counts every 500 and queue size |
| `normal` (default) | `Welcome`, `SettingsApplied` time and flush counts, first audio in/out, user/agent text per turn, tool calls and results, barge-ins, `playedStream` + playback time, TTFS, hangup |
| `quiet` | Only structured events, session start, warnings (`Warning`, `InjectionRefused`) and errors |

### OpenTelemetry spans

Without the `observability` extra, `opentelemetry` fails to import and every tracing hook is a no-op (one attribute check per hook, no errors). With it installed but no `OTEL_EXPORTER_OTLP_ENDPOINT`, spans are created and dropped. `server.py` also initialises Traceloop (OpenLLMetry) when `traceloop-sdk` is installed. It is only an optional export destination: the LLM runs inside Deepgram and no LLM SDK or httpx client is on the call path, so it adds no spans of its own.

The LLM, STT and TTS all run inside Deepgram, so the spans are built from the events it sends back, plus client-side timing for tools and playback:

```
session                         whole call: run()
│   call_id, leg_call_id, gen_ai.system="deepgram", gen_ai.request.model,
│   deepgram.think.provider, deepgram.listen.model, deepgram.speak.model,
│   deepgram.agent_config (UUID or "inline"), deepgram.request_id
│   events: deepgram.error (+ status ERROR), deepgram.warning when no turn is open
├── plivo_rx                    task: Plivo audio/text/playedStream in
├── deepgram_rx                 task: Deepgram events + agent audio in
├── plivo_tx                    task: playAudio + checkpoints out
├── turn                        turn 1 = greeting: SettingsApplied -> its turn_complete
│   └── playback                first playAudio -> playedStream (interrupted=true on barge-in)
└── turn                        user turn: EndOfTurn / ConversationText(user) / injected text
    │   turn, call_id, turn.source (greeting|audio|text), eot.trigger, user_text,
    │   agent_text, barge_in, playback_ms, turn.completed, deepgram.<key>_ms (LatencyReport)
    │   events: barge_in, deepgram.warning (e.g. SLOW_THINK_REQUEST)
    ├── stt                     stt_latency          (audio turns only)
    ├── llm                     ttt_tool_latency     (the pass that called a tool)
    ├── tool.check_order_status FunctionCallRequest -> FunctionCallResponse sent
    │                           tool.name, tool.call_id, tool.arguments, tool.result, tool.status
    ├── llm                     ttt_text_latency     (ttt_token_latency if text is absent)
    ├── tts                     tts_latency
    └── playback                first playAudio of the turn -> playedStream
```

A turn ends when `turn_complete` is emitted. It also ends with `turn.completed=false` if the next turn starts first (`turn.ended_by="next_turn"`) or the call ends mid-turn (`turn.ended_by="session_end"`), so no span is left open. Task spans cancelled at teardown carry `cancelled=true`; they are not marked as errors.

Live trace (real Deepgram, inline config, greeting plus one injected question; start offset from session start, then duration):

```
session                  +    0ms  21983ms  gpt-4.1-mini, flux-general-en, aura-2-thalia-en, inline
  plivo_rx               +  297ms  21649ms
  deepgram_rx            +  297ms  21652ms  cancelled=true
  plivo_tx               +  298ms  21651ms
  turn                   +  413ms   8359ms  turn=1 source=greeting playback_ms=4547
    playback             +  524ms   8248ms  interrupted=false
  turn                   + 9274ms  12126ms  turn=2 source=text eot.trigger=manual
    llm                  + 9312ms    729ms  ttt_tool_latency
    llm                  + 9325ms   1327ms  ttt_text_latency
    tool.check_order_status +10041ms   1ms  status=processing {"order_number":"TF-123456"}
    tts                  +10655ms    124ms  tts_latency
    playback             +10755ms  10645ms  interrupted=false
```

**What Deepgram exposes, and what it doesn't:**

| Signal | Exposed | Used for |
|--------|---------|----------|
| Final user / assistant text (`ConversationText`) | Yes | `user_text`, `agent_text` |
| End of turn (`EndOfTurn` with `trigger`), `UserStartedSpeaking` | Yes | `turn` start, `eot.trigger`, `barge_in` |
| Stage durations (`LatencyReport`: `stt_latency`, `ttt_token/text/tool_latency`, `tts_latency`, `total_latency`, in seconds) | Yes | `stt` / `llm` / `tts` spans, `deepgram.*_ms` |
| Tool calls (`FunctionCallRequest`), answered client-side | Yes | `tool.<name>` spans |
| `AgentAudioDone`, `Warning`, `Error` | Yes | checkpoint, span events, session status |
| Playback (Plivo `checkpoint` / `playedStream`) | Plivo | `playback` spans |
| Token counts, cost | No | — |
| Raw LLM request / response | No | — |
| Interim transcripts, STT confidence | No | — |
| Per-stage start/end timestamps | No (durations only) | stt/llm/tts start times are back-computed |

**Span placement is approximate.** `stt`, `llm` and `tts` end when their `LatencyReport` arrives and start that duration earlier, so they show how long each stage took, not exactly when it ran. The client measures arrival, which trails Deepgram's own clock by network time. `stt` uses the last `stt_latency` received within 2s before end of turn (Deepgram sends one roughly per audio frame). `ttt_text_latency` is counted from end of turn, so on a tool turn it overlaps the tool pass and the tool call. `tool.*` and `playback` are timed on the client from the messages it sends and receives.

**Models on the reusable-config path.** With a config UUID, the `DEEPGRAM_*` model env vars don't describe the saved config. At startup `verify_deepgram_agent_id()` already fetches the config, so it also reads the listen/think/speak models from it (configs are immutable, so they can't go stale) and `server.py` passes them to `run_agent(saved_agent_models=...)`. No network call is made per call. If the check could not run or failed, the model attributes read `saved-config`.

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
| `DEEPGRAM_INBOUND_AGENT_ID` | Reusable agent config UUID for inbound calls (see [Creating a reusable config](#creating-a-reusable-config)); empty = inline Settings | — |
| `DEEPGRAM_OUTBOUND_AGENT_ID` | Reusable agent config UUID for outbound calls; empty = inline Settings | — |
| `PLIVO_AUTH_ID` | Plivo Auth ID | Required |
| `PLIVO_AUTH_TOKEN` | Plivo Auth Token. Also the key for webhook signature checks | Required |
| `PLIVO_PHONE_NUMBER` | Plivo number (inbound auto-config; outbound: the `from` shown in the startup cURL) | Required |
| `PLIVO_TEST_NUMBER` | Second Plivo number for live call tests | — |
| `PUBLIC_URL` | Public HTTPS URL for webhooks; `https://` → `wss://` for the stream URL. Must match the URL Plivo calls: signatures are checked against it | Required |
| `SERVER_PORT` | Inbound server port | `8000` |
| `OUTBOUND_SERVER_PORT` | Outbound server port (differs from inbound so both can run at once) | `8001` |
| `DEFAULT_COUNTRY_CODE` | Default region for phone parsing | `US` |
| `LOG_LEVEL` | Agent pipeline log verbosity (`verbose` / `normal` / `quiet`) | `normal` |
| `LOG_FORMAT` | `json` replaces the stderr sink with serialized JSON | `text` |
| `LOG_FILE` | Path for an additional JSON log sink | — |
| `REDIS_EVENTS_URL` | Redis URL for the Streams sink | — |
| `REDIS_STREAM_KEY` | Redis stream key | `voice-agent:events` |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP endpoint for tracing | — |

### System prompt

Each direction has one prompt source: `inbound/system_prompt.md` and `outbound/system_prompt.md`. There is no `SYSTEM_PROMPT` env var. The env file is shared by both directions, and a multi-line prompt doesn't fit an env file or `docker --env-file`. To customise a prompt, edit the file. To customise it in Docker without editing the repo, mount your own file over it (the image's `WORKDIR` is `/app`):

```bash
docker run -v "$PWD/my_prompt.md:/app/inbound/system_prompt.md:ro" \
  -p 8000:8000 --env-file .env -e PUBLIC_URL=https://your-host.example.com deepgram-voiceagent

docker run -v "$PWD/my_outbound_prompt.md:/app/outbound/system_prompt.md:ro" \
  -p 8001:8001 --env-file .env -e PUBLIC_URL=https://your-host.example.com deepgram-voiceagent \
  uv run python -m outbound.server
```

The outbound file is a template: keep `{{greeting}}`, `{{opening_reason}}`, `{{objective}}` and `{{context}}` where the per-call values should go (any you leave out are simply not used). On the reusable-config path the prompt is fixed when the config is created, so create a new config after changing it.

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

- Runtime: `fastapi`, `uvicorn[standard]`, `websockets>=15.0`, `plivo`, `httpx` (the reusable-config ID check at startup), `python-dotenv`, `python-multipart`, `loguru`, `numpy`, `scipy`, `phonenumbers`. No torch, Silero, ONNX, OpenAI or Deepgram SDK.
- `observability` extra: `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`, `traceloop-sdk`.
- `streaming` extra: `redis[hiredis]`.
- `dev` group: `ruff`, `pre-commit`, `pytest`, `pytest-asyncio`, `faster-whisper`, `opentelemetry-sdk` (the span-tree unit tests use its in-memory exporter; the tests also use the runtime `httpx`).

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

The tests keep webhook authentication on. They sign webhook requests the way Plivo does (`plivo_signature_headers()` / `signed_webhook()` in `tests/helpers.py`, built on the Plivo SDK). They open `/ws` with the stream URL returned by a signed answer webhook. Local servers use a test auth token, with `PUBLIC_URL` set to their `http://localhost:<port>` URL.

Run from this directory:

```bash
uv sync   # includes the dev group
uv run ruff check .

# Unit tests (offline, no API keys)
uv run pytest tests/test_integration.py tests/test_observability.py -v -k unit

# Local server + direct Deepgram handshake/greeting (needs DEEPGRAM_API_KEY)
uv run pytest tests/test_integration.py -v -k "local or Deepgram"

# Reusable agent config: create (README body) -> connect by UUID -> UpdatePrompt/InjectAgentMessage
# -> delete (needs agent:read/agent:write scopes; skipped with the reason otherwise)
uv run pytest tests/test_integration.py -v -s -k SavedConfigIntegration

# E2E with real Deepgram, no phone call (greeting, text question, end_call).
# Runs every test twice: [inline] and [saved] (creates a config for the run, then deletes it)
uv run pytest tests/test_e2e_live.py -v -s

# Real calls (need Plivo creds, PLIVO_TEST_NUMBER, ngrok)
uv run pytest tests/test_live_call.py -v -s
uv run pytest tests/test_outbound_call.py -v -s
uv run pytest tests/test_multiturn_voice.py -v -s
```

The live call tests use `PLIVO_TEST_NUMBER`, a second Plivo number on the same account. It is the caller for inbound tests and the destination for outbound tests. Optional test-only env vars: `NGROK_BIN` (ngrok binary, default `ngrok`), `FFMPEG_DIR` (directory holding an `ffmpeg` binary for faster-whisper) and `TEST_LOG_DIR` (where test server logs go, default the system temp dir).

From the repo root:

```bash
./scripts/validate-example.sh deepgram-voiceagent
```

## Deployment

Use any host that runs containers and keeps WebSocket connections open for the length of a call:

| Host | Works? | Notes |
|---|---|---|
| Fly.io, Render, Railway, a VM or Kubernetes | ✅ | Long-lived WebSockets supported |
| Google Cloud Run | ✅ | Raise the request timeout (max 60 min) above your longest call |
| Vercel/Netlify functions, AWS Lambda | ❌ | No long-lived WebSocket server |

```bash
docker build -t deepgram-voiceagent .

# Inbound (default): configures the Plivo number from PUBLIC_URL on startup
docker run -p 8000:8000 --env-file .env -e PUBLIC_URL=https://your-host.example.com deepgram-voiceagent

# Outbound (listens on 8001, OUTBOUND_SERVER_PORT)
docker run -p 8001:8001 --env-file .env -e PUBLIC_URL=https://your-host.example.com deepgram-voiceagent \
  uv run python -m outbound.server
```

The image exposes 8000 (inbound) and 8001 (outbound). It is based on `python:3.12-slim` by default; pass `--build-arg BASE_IMAGE=...` to change it. It runs `uv sync --locked --no-install-project --no-dev --extra streaming`, so the Redis sink is available but the `observability` extra is not installed. The image does not include `cloudflared`, so `--tunnel` is for local runs only.

## Troubleshooting

### `--tunnel`: "cloudflared not found" or no "Ready!" line

- **"cloudflared not found":** [install `cloudflared`](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/) so it's on your `PATH`, then run again.
- **`Waiting for Plivo to accept …` for a while:** this is normal. Plivo rejects a new `trycloudflare.com` hostname (`Must be a valid url`) until it resolves, and the server keeps retrying for up to 3 minutes before `Ready!`. With `outbound.server --tunnel`, Plivo's Make Call API rejects the new `answer_url` the same way during that window; retry the call after a minute.
- **`Plivo did not accept … within 180s`:** restart with `--tunnel` to get a new URL, or use ngrok or a deployed host with a fixed `PUBLIC_URL`. Quick tunnels are meant for development and have no uptime guarantee.

### Calls rejected: 403 on `/answer` / `/outbound/answer`, or the call drops at once

The log shows `Rejected Plivo webhook POST /answer: signature mismatch …`. The URL rebuilt from `PUBLIC_URL` doesn't match the one Plivo signed. Check the following:

- `PUBLIC_URL` has the same scheme and host as the application's answer URL, or the `answer_url` you passed to Make Call. For example, don't mix `http://` and `https://`, and don't mix an old tunnel hostname with a new one.
- `PLIVO_AUTH_TOKEN` belongs to the account (or subaccount) that owns the number or call.
- Nothing between Plivo and the server rewrites the path or query string.

`missing X-Plivo-Signature-V3` means the request did not come from Plivo, for example a hand-made `curl`. To call the server by hand, sign the request with your `PLIVO_AUTH_TOKEN` the way Plivo does (see `plivo_signature_headers()` in `tests/helpers.py`).

### 401 / handshake rejected

Deepgram expects `Authorization: Token <key>`, not `Bearer`. Also check that the key's project has Voice Agent API access.

### `Error` right after `Settings`

A field in `Settings` is invalid. The most common cause is an `agent.language` or `speak.provider.language` field, which this configuration rejects. Other causes are an unknown model id or provider type (`DEEPGRAM_LISTEN_MODEL`, `DEEPGRAM_THINK_PROVIDER`/`DEEPGRAM_THINK_MODEL`, `DEEPGRAM_SPEAK_MODEL`) or a malformed function schema. The `Error` event's `description` field names the problem.

### Reusable config: every call fails, or old models/prompt are used

- **Server exits at startup with "… is not an agent config in the API key's Deepgram project"**: the UUID in `DEEPGRAM_INBOUND_AGENT_ID` / `DEEPGRAM_OUTBOUND_AGENT_ID` was deleted or mistyped. Without this check, Deepgram would reply to every call's `Settings` with an `Error` (`INTERNAL_SERVER_ERROR`, "resolving agent ID") and calls would end without a greeting. List the configs (see [Creating a reusable config](#creating-a-reusable-config)) and fix the env var, or empty it to go back to inline. If the log instead says `Could not verify … starting unverified`, the check couldn't run (network, or a key without `agent:read`), and a bad UUID would only show up on the first call.
- Changes to `DEEPGRAM_*` model vars, `system_prompt.md` or `FUNCTION_DEFINITIONS` have no effect on this path, because the config was fixed when it was created. Create a new config, switch the UUID and restart.

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
2. **No `AgentStartedSpeaking` / `AgentThinking` in practice.** Latency arrives only as single-key `LatencyReport` messages: `stt_latency` (roughly every audio frame), then `ttt_text_latency` / `ttt_token_latency` / `ttt_tool_latency`, `tts_latency` and `total_latency`, all in seconds. `_on_latency_report()` merges them into `turn_complete`'s `latency_report` and `*_latency_ms` fields. The `AgentStartedSpeaking` handler reads its latency fields if Deepgram sends it.
3. **Greeting audio before its text.** Binary greeting audio can arrive before `ConversationText{assistant}`. The code does not wait for the text; `_on_agent_audio()` starts playback on the first frame, and turn 1 is assigned whichever arrives.
4. **`InjectUserMessage` echo.** Deepgram echoes an injected message back as `ConversationText{user}` and sends `EndOfTurn{trigger: "manual"}`. The echo is matched against `_last_injected_text` and skipped, so the turn is counted once. Injecting while audio is playing triggers a local barge-in first.
5. **Drop gate after barge-in.** A few binary frames of the cancelled response can still arrive after `UserStartedSpeaking`. `_drop_agent_audio` discards them and clears on `ConversationText{user}`, `EndOfTurn`, `AgentThinking`, `FunctionCallRequest` or `AgentStartedSpeaking`. It is deliberately **not** cleared by `ConversationText{assistant}`, because the new response's first frame can precede it and a late sentence of the cancelled response can follow it.
6. **`end_call` ordering.** The LLM calls `end_call` first and then speaks the goodbye. The function only marks the hangup as pending (15s deadline); the hangup happens at the next `playedStream` for the goodbye's checkpoint, or at the deadline. If the goodbye is cut off by a barge-in, the hangup runs on that response's `AgentAudioDone`.
7. **Echoes and history.** Deepgram echoes each `FunctionCallResponse` back and sends `History` events. Both are only logged at `LOG_LEVEL=verbose`.
8. **Checkpoint through the queue.** `AgentAudioDone` enqueues a `_Checkpoint` sentinel behind the audio, so Plivo's `checkpoint` is always sent after the last chunk. A `playedStream` whose name does not match the pending checkpoint is ignored. With no `streamId` (local tests), `AgentAudioDone` completes playback directly.
9. **Hangup is owned by the server.** The agent never reads Plivo credentials. `server.py` passes `hangup_callback=functools.partial(_hangup_call, callId)`, which runs `calls.delete` in a thread.
10. **Python 3.10.** Timeouts catch `(TimeoutError, asyncio.TimeoutError)`, and the Deepgram WebSocket connects with `max_size=None` and `ping_interval=20`.
