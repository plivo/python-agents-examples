# Voice Agent Examples — Project Constitution

This repo contains production-ready voice agent examples using Plivo telephony. Every example follows the same structure regardless of AI API or orchestration approach.

## Naming Convention

`{llm-provider+series}-{stt-provider+series}-{tts-provider+series}-{orchestration}[-{variant}]`

**Every component always includes provider name + model series.** The series identifies the API contract; the size variant (mini/nano/pro/flash) is config in `.env`, not part of the folder name.

### LLM component: `{provider}{version}`

Drop the size class (mini/nano/pro/flash) — it's `.env` config. Only include size when two different sizes are used together in the same example.

| Model | Folder component | Notes |
|---|---|---|
| `gpt-5.4-mini` | `gpt5.4` | drop "mini" |
| `gpt-4.1` | `gpt4.1` | |
| `gpt-4.1-mini` (alone) | `gpt4.1` | drop "mini" |
| `gpt-4.1-mini` + `gpt-4.1` (dual) | `gpt4.1mini-gpt4.1` | two sizes → keep both |
| `gpt-4o-mini` | `gpt4o` | drop "mini" |
| `gemini-2.0-flash` | `gemini2` | drop "flash" |
| `gemini-2.5-flash` (live API) | `gemini2.5-live` | drop "flash"; `-live` = S2S API type |
| `gemini-3.1-flash` (live API) | `gemini3.1-live` | drop "flash"; `-live` = S2S API type |
| `gpt-realtime-1.5` (S2S) | `gptrealtime1.5` | "realtime" is the model name |
| `grok-3-fast-voice` (S2S) | `grok3-voice` | |

### Voice AI (STT) component: `{provider}{model-name}{version}`

| Model | Folder component |
|---|---|
| Deepgram `nova-2-phonecall` | `deepgramnova2` |
| Deepgram `nova-3` | `deepgramnova3` |
| Deepgram `flux` | `deepgramflux` |
| AssemblyAI `u3-rt-pro` | `assemblyaiu3` |
| Modulate `velma-2` | `modulatevelma2` |
| Sarvam STT | `sarvam` (no named model series) |

### Voice AI (TTS) component: `{provider}{model-name}{version}`

| Model | Folder component |
|---|---|
| ElevenLabs `eleven_flash_v2_5` | `elevenflashv2.5` |
| Cartesia `sonic-2` | `cartesiasonic2` |
| Cartesia `sonic-3` and its `sonic-3.x` point releases (e.g. `sonic-3.6`) | `cartesiasonic3` |
| OpenAI `gpt-4o-mini-tts` | `openaitts4o` |
| Grok `grok-3-fast-voice` (TTS only) | `groktts3` |

### Examples

`gpt5.4-assemblyaiu3-cartesiasonic3-native`, `gemini2.5-live-pipecat`, `gpt4.1-deepgramnova3-elevenflashv2.5-native`, `deepgram-voiceagent` (managed platform — see below)

Orchestration types:
- **native** — raw websockets/SDK, custom asyncio task management, client-side Silero VAD (default)
- **pipecat** / **livekit** / **vapi** — framework-based Pipeline, framework-managed VAD
- **managed platform** — hosted voice-agent product; no orchestration token in the name (see "Managed Voice-Agent Platforms")

Variants:
- **`-no-vad`** — explicitly opts out of client-side VAD (e.g., `gemini2.5-live-native-no-vad` relies on server-side VAD)
- **`-webrtcvad`** — uses WebRTC VAD instead of Silero (e.g., `gemini2.5-live-native-webrtcvad`)
- All new native examples include Silero VAD by default. These suffixes are the exception, not the rule.

### Managed Voice-Agent Platforms

A managed platform runs the whole conversation loop (STT, LLM, TTS, turn detection, barge-in) as one hosted product; the example only bridges Plivo audio to it and answers its client-side events. These are named after the **product**, not its component models, and do not use the orchestration/VAD tokens above:

`{provider}-{product}[-{variant}]` — e.g. `deepgram-voiceagent` (Deepgram "Voice Agent API")

- **`{provider}`** — the company name, lowercased.
- **`{product}`** — the platform's *own* name for the product, as branded in its docs/API reference: lowercased, spaces and punctuation removed, generic suffixes like "API" dropped. Honor each brand's naming; do **not** reuse another platform's product term or invent a shared category word. If the product name is the company name, write it once. If a provider has several voice-agent products, use the one whose API the example actually calls.
- **No model components** — model choices are `.env` config. Env vars mirror the platform's own config schema and accept its documented values verbatim (model ids, provider types); never define aliases or wrapper values on top. The README lists tested combinations.
- **No `native`/`-no-vad`/orchestration token** — turn detection and barge-in are owned by the platform. The client still must flush Plivo playback (`clearAudio`) on the platform's interruption event.
- **One example per platform.** A different model combination is a README/`.env` change, not a new example.
- Declare the category in `pyproject.toml` so `scripts/validate-example.sh` applies the right checks:
  ```toml
  [tool.voice-agent-example]
  category = "managed-platform"
  ```
- Canonical file structure, config placement, audio rules and the 3-task asyncio pattern still apply (`_receive_from_{api}()` handles the platform's events). No Silero/VAD code.

**Variants (`-{variant}`) are not predefined and require human review.** Add one only when the *integration contract* changes in a way `.env` config cannot express — e.g. how Plivo audio reaches the platform changes, or a pipeline stage moves out of the platform into this example's own code. When proposing one: name what differs in the integration (not a model, and not a vendor unless the vendor *is* the difference), keep it to one or two lowercase words, check it doesn't collide with an existing token, and call it out in the PR description for a reviewer to approve before the directory is created. The validator only checks the name's format.

Legacy: `gpt4.1-deepgramnova3-elevenflashv2.5-vapi` predates this rule and is still named by components; it will be revisited separately.

## Canonical File Structure (ALL examples)

```
{example-name}/
├── inbound/
│   ├── __init__.py
│   ├── agent.py              # AI-specific voice agent class (or framework pipeline)
│   ├── server.py             # FastAPI: /answer, /ws, /hangup
│   └── system_prompt.md      # System prompt for inbound calls
├── outbound/
│   ├── __init__.py
│   ├── agent.py              # Same agent class + outbound prompt/greeting rendered from per-call context
│   ├── server.py             # FastAPI: /outbound/answer (answer_url context → <Stream>), /outbound/hangup, /ws
│   └── system_prompt.md      # System prompt for outbound calls
├── utils.py                  # Phone utils; audio conversion + VAD only where this example converts audio (see "utils.py Requirements")
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # sys.path setup (copy from grok3-voice-native)
│   ├── helpers.py            # ngrok, recording, transcription (copy from grok3-voice-native); test-only decoder when utils.py has no codec
│   ├── test_integration.py   # Unit + local integration tests
│   ├── test_e2e_live.py      # E2E with real API (no phone call)
│   ├── test_live_call.py     # Real inbound call test
│   ├── test_multiturn_voice.py  # Multi-turn conversation test
│   └── test_outbound_call.py # Real outbound call test
├── pyproject.toml
├── .env.example              # Leading dot (industry standard)
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
└── README.md
```

No exceptions. S2S, pipeline, and framework examples all use this structure.

## Config Constant Placement

Constants live where they are consumed:

**`server.py`** owns (duplicated in inbound/outbound — each file is self-contained):
- `SERVER_PORT`, `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_PHONE_NUMBER`, `PUBLIC_URL`

**`agent.py`** owns:
- API keys, model names, voice names, API URLs
- `PLIVO_CHUNK_SIZE = 160` (used in `_send_to_plivo`; not defined in framework examples, where the transport chunks the audio)
- `SYSTEM_PROMPT` (loaded only from `system_prompt.md`; no env override, see "System Prompt")

**`utils.py`** owns only what its functions consume:
- Audio sample rates: `PLIVO_SAMPLE_RATE`, `{API}_SAMPLE_RATE`, `VAD_SAMPLE_RATE` (only when `utils.py` has conversion functions; an example with none, e.g. `gpt4o-modulatevelma2-cartesiasonic3-pipecat`, defines no sample-rate constants there)
- VAD params (native only): `VAD_START_THRESHOLD`, `VAD_END_THRESHOLD`, `VAD_MIN_SILENCE_MS`, `VAD_CHUNK_SAMPLES`
- `DEFAULT_COUNTRY_CODE`

## System Prompt

The system prompt is loaded only from `inbound/system_prompt.md` / `outbound/system_prompt.md`. No `SYSTEM_PROMPT` (or similar) env override: it is a second source of truth, the env is shared by both directions, and multi-line prompts don't fit env files / `docker --env-file`. Customise by editing the file, or by mounting another file over it (e.g. `docker run -v ./my_prompt.md:/app/inbound/system_prompt.md …`).

## Outbound Calls

New examples use the simple outbound path: Plivo Make Call API → `answer_url` (`/outbound/answer?greeting=…`, the greeting as an optional query param, spoken verbatim; a default applies when absent) → `<Stream>` (greeting in the base64 `body`) → `/ws` → agent. The system prompt is `outbound/system_prompt.md` as is (no per-call templating; customize the use case in the file). The server has no dial endpoint and no `CallManager`/campaign/status tracking; `/outbound/hangup` only logs. Reference: `deepgram-voiceagent/`. Existing examples with `CallManager`, `OutboundCallRecord` and `POST /outbound/call` are legacy; don't add them to new ones.

## utils.py Requirements

Only utility functions and their internal constants. No server or agent config.

**Principle.** An audio helper is required in `utils.py` exactly when this example's own code converts audio on the call path. Whoever converts owns the code: when a framework's serializer/transport or a hosted platform does it, no helper is required here and none is carried unused. Conversion code never lives inline in `agent.py` / `server.py`.

Always required: `normalize_phone_number(phone: str, default_region: str) -> str`.

Audio helpers, by name:
- Codec set: `ulaw_to_pcm(ulaw_data: bytes) -> bytes` (G.711 decode table), `pcm_to_ulaw(pcm_data: bytes) -> bytes` (G.711 encode), `resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes`
- Direction wrappers: `plivo_to_{api}(mulaw_8k: bytes) -> bytes` (Plivo audio to API format), `{api}_to_plivo(pcm: bytes) -> bytes` (API audio to Plivo format). In a cascaded pipeline `{api}` is the service on that side: the STT for input, the TTS for output.
- Silero set: `plivo_to_vad(mulaw_8k: bytes) -> np.ndarray` (float32 16kHz), `SileroVADProcessor` class (reference: `grok3-voice-native/utils.py`)

| Who converts audio on the call path | Required in `utils.py` besides `normalize_phone_number` | Reference |
|---|---|---|
| **Native**, API needs PCM or another rate | Codec set, both wrappers, Silero set | `grok3-voice-native/` |
| **Native `-no-vad`** | Codec set, both wrappers. No VAD code | `gemini2.5-live-native-no-vad/` |
| **Native `-webrtcvad`** | Codec set, both wrappers. No Silero set: the `webrtcvad.Vad` instance lives in `agent.py` and is fed `ulaw_to_pcm` output at 8kHz | `gemini2.5-live-native-webrtcvad/` |
| **Native or managed platform, μ-law 8kHz end to end** (API accepts and emits Plivo's own format, no client-side VAD) | Both wrappers as documented pass-throughs (the body returns its argument). Codec set, decode table and sample-rate constants omitted | `deepgram-voiceagent/` (Deepgram Voice Agent configured for `mulaw` 8000 in and out) |
| **Managed platform** configured for PCM or another rate | Codec set, both wrappers. No VAD code | none yet |
| **Framework** (pipecat / livekit), serializer or transport converts | Nothing | `gpt4o-modulatevelma2-cartesiasonic3-pipecat/` (`PlivoFrameSerializer`) |
| **Framework** whose own code touches raw audio (a custom processor or service that decodes, encodes or resamples) | The helpers that code calls, defined in `utils.py` and imported from there | none yet |
| **Hosted orchestration**, call audio never reaches this server | Nothing | `gpt4.1-deepgramnova3-elevenflashv2.5-vapi/` (Plivo SIP trunk to Vapi; the server only handles webhooks) |

Rules that follow from the principle:
- A native example with Silero or WebRTC VAD decodes for the VAD, so it always carries the codec set, even when the API itself takes μ-law.
- A direction with no conversion still has its wrapper, as a pass-through, so the audio path is readable from `utils.py`.
- An example with no codec set does not carry `numpy`/`scipy` as runtime dependencies for it. Tests that decode recordings (RMS, transcription) or build caller audio keep a small decoder in `tests/helpers.py` instead.
- Unused code that exists only to satisfy this list is not required.
- Legacy: `gemini2.5-live-pipecat/` and `gpt4o-deepgramnova3-openaitts4o-pipecat/` still carry codec helpers that only their tests call. Tolerated, not a model.

`scripts/validate-example.sh` enforces this from the syntax tree: native and managed-platform examples must define the helpers above (the pass-through row is granted only when both wrappers return their argument, no codec function is defined or referenced, and no non-test code imports a conversion library); framework and hosted examples skip the helper checks; every example fails if non-test code outside `utils.py` imports `audioop`, calls a library resampler, or defines its own μ-law/resample function.

## VAD Strategy

**Native examples**: client-side Silero VAD (`SileroVADProcessor`).
- VAD runs in `plivo_rx` task alongside audio forwarding
- Speech start during AI response triggers barge-in (`response.cancel` or equivalent)
- Speech end triggers turn commit (`input_audio_buffer.commit` + `response.create` or equivalent)
- Reference: `grok3-voice-native/utils.py` (SileroVADProcessor), `grok3-voice-native/inbound/agent.py` (integration)

**Framework examples** (Pipecat/LiveKit/Vapi): configure the framework's own VAD or turn detection in code. No separate Silero, no VAD code in `utils.py`.
- Pipecat 1.x: `vad_analyzer=SileroVADAnalyzer()` on `LLMUserAggregatorParams` (the params of the user side of `LLMContextAggregatorPair`). Reference: `gpt4o-modulatevelma2-cartesiasonic3-pipecat/inbound/agent.py`
- Pipecat below 1.0 (legacy): `vad_enabled=True` in transport params. Pipecat 1.x `TransportParams` has no such field and ignores it. Reference: `gemini2.5-live-pipecat/inbound/agent.py`
- LiveKit: `vad=` on the session. Hosted assistant config (Vapi): its speaking-plan keys.

## Audio Pipeline Rules

- `PLIVO_CHUNK_SIZE = 160` — exactly 20ms at 8kHz mono μ-law. Defined in `agent.py._send_to_plivo()`. Native and managed-platform examples only: a framework's transport does the chunking, so framework examples do not define it.
- Plivo WebSocket sends/receives base64 μ-law at 8kHz
- playAudio JSON format: `{"event": "playAudio", "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": "<base64>"}}`
  - Framework examples (Pipecat, LiveKit, Vapi, …) usually do not build this message: the framework's transport emits it. The validator requires the dict literal in native and managed-platform agents; in a framework agent it is checked only if the agent builds one itself.
- Answer webhook returns `<Stream>` XML: `bidirectional=True`, `keepCallAlive=True`, `contentType="audio/x-mulaw;rate=8000"`

## Agent Structure

**Native orchestration**: custom agent class with these methods:
- `__init__`, `run()`, `_run_streaming_tasks()` (3 concurrent tasks)
- `_receive_from_plivo()` — plivo_rx: decode audio, run VAD, forward to API
- `_receive_from_{api}()` — api_rx: receive API events, queue audio for plivo
- `_send_to_plivo()` — plivo_tx: chunk audio to 160 bytes, send playAudio
- Public `run_agent()` function wraps class instantiation

**Framework orchestration**: `run_agent()` function assembles the framework's pipeline and runs it to completion. No custom agent class needed.

**Pipecat 1.x** (reference: `gpt4o-modulatevelma2-cartesiasonic3-pipecat/inbound/agent.py`):
- New examples build a `PipelineWorker` (`pipecat.pipeline.worker`) around the `Pipeline` and run it with `WorkerRunner` (`pipecat.workers.runner`): `await runner.add_workers(worker)`, then `await runner.run()`.
- `PipelineTask` / `PipelineRunner` are deprecated aliases of those two classes (deprecated in Pipecat 1.3.0, removed in 2.0.0). Only the legacy examples locked to Pipecat 0.0.x use them (`gemini2.5-live-pipecat`, `gpt4o-deepgramnova3-openaitts4o-pipecat`); do not use them in new code.

**Pipecat runner signal handling**:
- `WorkerRunner` installs its handlers with `loop.add_signal_handler(...)` when `run()` starts. A loop-level handler **replaces** uvicorn's handler for that signal and Pipecat never removes it, so after the first call uvicorn never receives that signal again and the process hangs instead of shutting down.
- Defaults are `handle_sigint=True`, `handle_sigterm=False`. Inside uvicorn construct `WorkerRunner(handle_sigint=False)` and never pass `handle_sigterm=True`.
- `handle_sigint=True` / `handle_sigterm=True` are only appropriate for standalone scripts where the runner owns the process lifecycle.
- Legacy `PipelineRunner` (Pipecat 0.0.x) has the same two arguments and defaults and installs the handlers in `__init__`; never pass it `handle_sigterm=True`.
- `PipelineWorker` defaults: idle timeout 300s (`idle_timeout_secs`), cancel timeout 20s (`cancel_timeout_secs`), relevant for shutdown timing.

## WebSocket Protocol

1. Plivo sends `{"event": "start", "start": {"callId": "...", "streamId": "..."}}` — handle first
2. Plivo sends `{"event": "media", "media": {"payload": "<base64 μ-law>"}}` — audio data
3. Plivo sends `{"event": "stop"}` — call ended
4. Agent sends `{"event": "playAudio", "media": {...}}` — response audio
5. Agent sends `{"event": "clearAudio"}` — on barge-in to stop playback

## Webhook Authentication

New examples authenticate Plivo's HTTP webhooks and the `/ws` stream (reference: `deepgram-voiceagent/`, README "Webhook authentication"):
- **Plivo HTTP webhooks** (answer, hangup, fallback, …): verify the V3 signature (`X-Plivo-Signature-V3` + `-Nonce`) with `plivo.utils.validate_v3_signature`, keyed with `PLIVO_AUTH_TOKEN`, via a FastAPI dependency in `server.py`. Rebuild the signed URL as `PUBLIC_URL` + request path + raw query string (never `request.url`: behind a tunnel it is `http://localhost…`), read at request time so `--tunnel` works. Signed params are the form fields for POST and the query string for GET. Failure: 403, plus a warning with the path and reason, never the signature.
- **Always on**: no env switch disables the check. With an empty `PLIVO_AUTH_TOKEN`, refuse to start. Tests use a dummy `PLIVO_AUTH_TOKEN` and sign requests the way Plivo does.
- **`/ws` stream**: Plivo sends the same two signature headers when it connects. Verify them with the same dependency as the webhooks (`@app.websocket("/ws", dependencies=…)`), method `GET`, no params. The signed URL is `http://` + `PUBLIC_URL` host + request path: always the `http` scheme, and without the query string. On failure raise `WebSocketException(code=1008)` before `accept()` (the client sees HTTP 403) and log a warning with the path and reason. The signature does not cover `?body=`. Tests sign the connection the way Plivo does (`stream_signature_headers()` in `tests/helpers.py`).
- Keep percent-encoding the base64 `body` in the stream URL so a `+` doesn't arrive as a space.

## Asyncio Patterns (Native)

```python
# Task management — always use this pattern
tasks = [
    asyncio.create_task(self._receive_from_plivo(), name="plivo_rx"),
    asyncio.create_task(self._receive_from_{api}(ws), name="{api}_rx"),
    asyncio.create_task(self._send_to_plivo(), name="plivo_tx"),
]
try:
    done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        if task.exception():
            logger.error(f"Task {task.get_name()} failed: {task.exception()}")
finally:
    self._running = False
    for task in tasks:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
```

Note: `_pending` with underscore prefix avoids RUF059 lint warning.

## Package Management

- **Always use `uv`** — never `pip`, `pip install`, or `python -m pip`
- Each example has its own virtualenv (`.venv/` inside the example directory)
- `uv sync` to install deps, `uv add {pkg}` to add new deps, `uv run` to execute commands
- All commands run through `uv run`: `uv run pytest`, `uv run ruff check .`, `uv run python -m inbound.server`
- `uv.lock` is committed to git for reproducible builds
- **Python floor**: 3.10+ is the repo-wide default (`requires-python = ">=3.10"`, ruff `target-version = "py310"`). An example raises the floor only when a dependency requires it, and then `requires-python` and ruff `target-version` name the same floor, the Dockerfile base image runs a Python at or above it, and the README Prerequisites state it. Reference: `gpt4o-modulatevelma2-cartesiasonic3-pipecat/` (`pipecat-ai>=1.0` requires Python 3.11+: `>=3.11`, `py311`, `python:3.12-slim`)

### Dockerfile `uv sync` and optional dependencies

Every example must include `[project.optional-dependencies]` with `observability` and `streaming` extras (reference: `gpt4.1-sarvam-elevenflashv2.5-native/pyproject.toml`). The Dockerfile's `uv sync` command must include `--extra streaming` so Redis is available at runtime. If `pyproject.toml` defines a `streaming` extra but the Dockerfile omits `--extra streaming`, the container will fail at runtime when streaming features are used.

## Git Workflow

- **Never commit directly to `main`**. Always create a feature branch first:
  `git checkout -b {example-name}` (or `git checkout -b fix/{description}` for fixes)
- Push to the `fork` remote (not `origin`, which has IP restrictions):
  `git push -u fork {branch-name}`
- Open a PR from the fork branch to `origin/main` when ready.

## Code Quality

- `from __future__ import annotations` at top of every `.py` file
- `loguru` for logging (not stdlib `logging`)
- No hardcoded API keys — always `os.getenv()`
- `python-dotenv` with `load_dotenv()` at module level
- All imports lazy where heavy (e.g., `import torch` inside methods)

## Lint

Ruff with: `select = ["E", "W", "F", "I", "B", "UP", "SIM", "RUF"]`, `line-length = 100`, `target-version = "py310"` (or the example's raised Python floor, see "Package Management")

Run: `uv run ruff check .`

## Testing

**Unit tests** (`-k "unit"`): offline, no API keys needed
- `TestUnitAudioConversion`: ulaw↔pcm roundtrip, silence detection. Tests the `utils.py` codec when `utils.py` has one; otherwise the test-only decoder in `tests/helpers.py` (reference: `gpt4o-modulatevelma2-cartesiasonic3-pipecat/tests/`)
- `TestUnitPhoneNormalization`: E.164 formatting

**Local integration** (`-k "local"`): starts server subprocess, tests WebSocket flow with real API
- `TestLocalIntegration`: health check, answer webhook XML, WebSocket audio flow

**E2E live call tests**: real Plivo calls, recording, transcription
- `test_live_call.py`: inbound call → greeting verification
- `test_outbound_call.py`: outbound call → greeting verification
- `test_multiturn_voice.py`: multi-turn + barge-in verification

Test infra: `conftest.py` sets `sys.path`, `helpers.py` has ngrok/recording/transcription utils, plus the test-only decoder when `utils.py` has no codec.

**Server subprocess teardown** in `server_process` fixture — always use SIGTERM with SIGKILL fallback:
```python
os.kill(proc.pid, signal.SIGTERM)
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
```
Pipecat servers may not exit on SIGTERM alone once a runner (`WorkerRunner`, or legacy `PipelineRunner`) that was allowed to install signal handlers has been active (see "Pipecat runner signal handling" above). Native servers typically exit cleanly on SIGTERM, but the fallback pattern is safe for all examples.

Run: `uv run pytest tests/test_integration.py -v -k "unit"` (offline)

## Reference Files

- **Primary reference**: `grok3-voice-native/` — complete native example with Silero VAD
- `grok3-voice-native/utils.py` — SileroVADProcessor class, audio conversion
- `grok3-voice-native/inbound/agent.py` — native agent pattern with VAD + barge-in
- `grok3-voice-native/outbound/agent.py` — legacy `CallManager` outbound pattern; do not copy it into new examples (see "Outbound Calls")
- `grok3-voice-native/tests/` — full test suite to replicate
- `gemini2.5-live-native-no-vad/` — alternative native pattern (SDK-based, server-side VAD, no client-side VAD)
- `gpt4o-modulatevelma2-cartesiasonic3-pipecat/inbound/agent.py`: framework reference for Pipecat 1.x (`PipelineWorker` + `WorkerRunner`, `vad_analyzer` on the user aggregator, `PlivoFrameSerializer`, no audio helpers in `utils.py`)
- `gemini2.5-live-pipecat/inbound/agent.py`: legacy framework reference (Pipecat 0.0.x: `PipelineTask` + `PipelineRunner`, `vad_enabled=True`)
- `deepgram-voiceagent/` — managed voice-agent platform reference (raw WebSocket bridge, platform-side turn detection, checkpoint-based playback tracking)
- `deepgram-voiceagent/outbound/` — outbound reference: Plivo Make Call API → `answer_url` query params → `<Stream>` → agent

## README Demo Description (Required)

The text between H1 (`#`) and the first H2 (`##`) in each README is displayed as the demo description in the hosting app. It must be **5 lines or fewer** but pack maximum technical detail. Use `gpt4.1mini-sarvam-elevenlabs-native/README.md` as the reference.

### Format

Text and bullet lists only — **no tables, no diagrams, no code blocks** between H1 and first H2. Write a dense description (≤5 lines) that traces the full pipeline from telephony input to audio output, naming every component along the way. Include:

- Orchestration approach (native/framework)
- Each component: service name, model/engine, protocol (WS/HTTP), audio format, sample rate, region
- VAD: engine, frame size, threshold values with empirical tuning rationale (echo vs speech probability ranges)
- Barge-in: what gets cancelled and what event is sent
- Any notable audio conversions (resample or no-resample)

### Rules

- No vague descriptions ("production-ready", "best-in-class") — every word should be a technical fact
- No tables, diagrams, or code blocks — the hosting app doesn't render them properly. Bullet lists are fine
- Do not include observed latency — that belongs in the detailed sections below
- The rest of the README (after the first H2) can use tables, diagrams, and full detail

## Slash Commands (Phase Workflow)

```
/scaffold-example {name} {description}   # Phase 1: directory structure + boilerplate
/implement-agent {name} {api-docs-url}    # Phase 2: write agent.py + utils.py
/test-example {name}                       # Phase 3: create tests + run them
/review-example {name}                     # Phase 4: quality gate checklist
/document-example {name}                   # Phase 5: README + .env.example validation
```

Each phase gets a fresh context window. Run sequentially.

## CI Validation

```bash
./scripts/validate-example.sh {example-name}
```

Exit 0 = pass, exit 1 = fail. Checks structure, lint, unit tests, config placement, `utils.py` helpers.
