# Voice Agent Examples — Project Constitution

This repo contains production-ready voice agent examples using Plivo telephony. Every example follows the same structure regardless of AI API or orchestration approach.

## Naming Convention

`{ai-provider}-{optional-stt}-{optional-tts}-{orchestration}[-{variant}]`

Examples: `grok-voice-native`, `gemini-deepgram-elevenlabs-native`, `gemini-live-pipecat`

Orchestration types:
- **native** — raw websockets/SDK, custom asyncio task management, client-side Silero VAD (default)
- **pipecat** / **livekit** / **vapi** — framework-based Pipeline, framework-managed VAD

Variants:
- **`-no-vad`** — explicitly opts out of client-side VAD (e.g., `gemini-live-native-no-vad` relies on server-side VAD)
- **`-webrtcvad`** — uses WebRTC VAD instead of Silero (e.g., `gemini-live-native-webrtcvad`)
- All new native examples include Silero VAD by default. These suffixes are the exception, not the rule.

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
│   ├── agent.py              # Same agent class + OutboundCallRecord, CallManager
│   ├── server.py             # FastAPI: /outbound/call, /outbound/ws, etc.
│   └── system_prompt.md      # System prompt for outbound calls
├── utils.py                  # Audio conversion, VAD (if native), phone utils
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # sys.path setup (copy from grok-voice-native)
│   ├── helpers.py            # ngrok, recording, transcription (copy from grok-voice-native)
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
- `PLIVO_CHUNK_SIZE = 160` (used in `_send_to_plivo`)
- `SYSTEM_PROMPT` (loaded from `system_prompt.md`)

**`utils.py`** owns only what its functions consume:
- Audio sample rates: `PLIVO_SAMPLE_RATE`, `{API}_SAMPLE_RATE`, `VAD_SAMPLE_RATE`
- VAD params (native only): `VAD_START_THRESHOLD`, `VAD_END_THRESHOLD`, `VAD_MIN_SILENCE_MS`, `VAD_CHUNK_SAMPLES`
- `DEFAULT_COUNTRY_CODE`

## utils.py Requirements

Only utility functions and their internal constants. No server or agent config.

Required functions:
- `ulaw_to_pcm(ulaw_data: bytes) -> bytes` — G.711 decode table
- `pcm_to_ulaw(pcm_data: bytes) -> bytes` — G.711 encode
- `resample_audio(audio_data: bytes, input_rate: int, output_rate: int) -> bytes`
- `plivo_to_{api}(mulaw_8k: bytes) -> bytes` — Plivo audio to API format
- `{api}_to_plivo(pcm: bytes) -> bytes` — API audio to Plivo format
- `normalize_phone_number(phone: str, default_region: str) -> str`

For native examples, also:
- `plivo_to_vad(mulaw_8k: bytes) -> np.ndarray` — float32 16kHz for Silero
- `SileroVADProcessor` class (reference: `grok-voice-native/utils.py`)

For framework examples: no VAD in utils (framework handles it).

## VAD Strategy

**Native examples**: client-side Silero VAD (`SileroVADProcessor`).
- VAD runs in `plivo_rx` task alongside audio forwarding
- Speech start during AI response triggers barge-in (`response.cancel` or equivalent)
- Speech end triggers turn commit (`input_audio_buffer.commit` + `response.create` or equivalent)
- Reference: `grok-voice-native/utils.py` (SileroVADProcessor), `grok-voice-native/inbound/agent.py` (integration)

**Framework examples** (Pipecat/LiveKit): use `vad_enabled=True` in transport params. No separate Silero.

## Audio Pipeline Rules

- `PLIVO_CHUNK_SIZE = 160` — exactly 20ms at 8kHz mono μ-law. Defined in `agent.py._send_to_plivo()`.
- Plivo WebSocket sends/receives base64 μ-law at 8kHz
- playAudio JSON format: `{"event": "playAudio", "media": {"contentType": "audio/x-mulaw", "sampleRate": 8000, "payload": "<base64>"}}`
- Answer webhook returns `<Stream>` XML: `bidirectional=True`, `keepCallAlive=True`, `contentType="audio/x-mulaw;rate=8000"`

## Agent Structure

**Native orchestration**: custom agent class with these methods:
- `__init__`, `run()`, `_run_streaming_tasks()` (3 concurrent tasks)
- `_receive_from_plivo()` — plivo_rx: decode audio, run VAD, forward to API
- `_receive_from_{api}()` — api_rx: receive API events, queue audio for plivo
- `_send_to_plivo()` — plivo_tx: chunk audio to 160 bytes, send playAudio
- Public `run_agent()` function wraps class instantiation

**Framework orchestration**: `run_agent()` function assembles Pipeline. No custom class needed.

**Pipecat PipelineRunner signal handling**:
- Use `PipelineRunner()` (default `handle_sigterm=False`) when running inside uvicorn.
- Do NOT use `PipelineRunner(handle_sigterm=True)` — it calls `loop.add_signal_handler(signal.SIGTERM, ...)` in `__init__`, which **replaces** uvicorn's SIGTERM handler. After the pipeline finishes, uvicorn's handler is never restored, so uvicorn never receives a shutdown signal and the process hangs indefinitely.
- `handle_sigterm=True` is only appropriate for standalone scripts where PipelineRunner owns the process lifecycle.
- PipelineRunner idle timeout is 300s, cancel timeout is 20s — relevant for shutdown timing.

## WebSocket Protocol

1. Plivo sends `{"event": "start", "start": {"callId": "...", "streamId": "..."}}` — handle first
2. Plivo sends `{"event": "media", "media": {"payload": "<base64 μ-law>"}}` — audio data
3. Plivo sends `{"event": "stop"}` — call ended
4. Agent sends `{"event": "playAudio", "media": {...}}` — response audio
5. Agent sends `{"event": "clearAudio"}` — on barge-in to stop playback

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

## Code Quality

- `from __future__ import annotations` at top of every `.py` file
- `loguru` for logging (not stdlib `logging`)
- No hardcoded API keys — always `os.getenv()`
- `python-dotenv` with `load_dotenv()` at module level
- All imports lazy where heavy (e.g., `import torch` inside methods)

## Lint

Ruff with: `select = ["E", "W", "F", "I", "B", "UP", "SIM", "RUF"]`, `line-length = 100`, `target-version = "py310"`

Run: `uv run ruff check .`

## Testing

**Unit tests** (`-k "unit"`): offline, no API keys needed
- `TestUnitAudioConversion`: ulaw↔pcm roundtrip, silence detection
- `TestUnitPhoneNormalization`: E.164 formatting

**Local integration** (`-k "local"`): starts server subprocess, tests WebSocket flow with real API
- `TestLocalIntegration`: health check, answer webhook XML, WebSocket audio flow

**E2E live call tests**: real Plivo calls, recording, transcription
- `test_live_call.py`: inbound call → greeting verification
- `test_outbound_call.py`: outbound call → greeting verification
- `test_multiturn_voice.py`: multi-turn + barge-in verification

Test infra: `conftest.py` sets `sys.path`, `helpers.py` has ngrok/recording/transcription utils.

**Server subprocess teardown** in `server_process` fixture — always use SIGTERM with SIGKILL fallback:
```python
os.kill(proc.pid, signal.SIGTERM)
try:
    proc.wait(timeout=5)
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
```
Pipecat servers may not exit on SIGTERM alone when a PipelineRunner has been active (see "Pipecat PipelineRunner signal handling" above). Native servers typically exit cleanly on SIGTERM, but the fallback pattern is safe for all examples.

Run: `uv run pytest tests/test_integration.py -v -k "unit"` (offline)

## Reference Files

- **Primary reference**: `grok-voice-native/` — complete native example with Silero VAD
- `grok-voice-native/utils.py` — SileroVADProcessor class, audio conversion
- `grok-voice-native/inbound/agent.py` — native agent pattern with VAD + barge-in
- `grok-voice-native/outbound/agent.py` — OutboundCallRecord, CallManager pattern
- `grok-voice-native/tests/` — full test suite to replicate
- `gemini-live-native-no-vad/` — alternative native pattern (SDK-based, server-side VAD, no client-side VAD)
- `gemini-live-pipecat/inbound/agent.py` — framework Pipeline reference

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

Exit 0 = pass, exit 1 = fail. Checks structure, lint, unit tests, config placement.
