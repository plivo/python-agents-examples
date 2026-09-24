# Implement Voice Agent

**Phase 2**: Write the actual agent logic for `inbound/agent.py`, `outbound/agent.py`, and API-specific parts of `utils.py`.

## Arguments

- `$ARGUMENTS` should contain: `{example-name}` and optionally an API documentation URL

## Instructions

Read `CLAUDE.md` for rules and patterns. Read the scaffolded files in `{example-name}/` to understand the current state.

### 1. Research the API

If an API docs URL is provided, fetch it to understand:
- WebSocket/streaming endpoint URL and authentication
- Audio input/output formats and sample rates
- Session configuration and turn management
- Event types and message protocol
- Function calling / tool use support

Also read the reference implementations:
- `grok3-voice-native/inbound/agent.py` — native pattern with Silero VAD, barge-in, turn management
- `deepgram-voiceagent/outbound/` — outbound pattern: Plivo Make Call API → `answer_url` query params → `<Stream>` → agent (`grok3-voice-native/outbound/` uses the legacy `CallManager`; don't copy it)
- `gemini2.5-live-native/inbound/agent.py` — alternative native pattern (SDK-based)

### 2. Update utils.py

Set correct audio sample rates for the API:
- `{API}_SAMPLE_RATE` (or `{API}_INPUT_RATE` / `{API}_OUTPUT_RATE` if they differ)
- Update `plivo_to_{api}()` and `{api}_to_plivo()` with correct rates
- Only modify utility-owned constants — do NOT add server or agent config here

### 3. Implement inbound/agent.py

**For native orchestration**, implement the full agent class:

```
_receive_from_plivo():
  - Decode base64 μ-law from Plivo WebSocket messages
  - Convert to API format via plivo_to_{api}()
  - Run Silero VAD via plivo_to_vad() + self._vad.process()
  - On speech_started + self._is_responding: trigger barge-in
    - Cancel API response (response.cancel or equivalent)
    - Drain send queue
    - Send clearAudio to Plivo
  - On speech_ended: commit audio buffer + request response
  - Handle "stop" event to break loop

_receive_from_{api}():
  - Connect to API WebSocket/SDK
  - Send session configuration (model, voice, turn_detection=None, tools)
  - Send initial message to trigger greeting
  - Process incoming events:
    - Audio delta → convert via {api}_to_plivo() → queue for send
    - Response created/done → track _is_responding state
    - Function calls → dispatch to handler
    - Errors → log

_send_to_plivo():
  - Dequeue audio from send queue
  - Buffer and chunk to PLIVO_CHUNK_SIZE (160 bytes)
  - Send as playAudio JSON via WebSocket
```

Include all tool functions from the scaffold (check_order_status, send_sms, schedule_callback, transfer_call, end_call).

**For framework orchestration**, implement `run_agent()`:
- Configure framework transport with Plivo WebSocket
- Set `vad_enabled=True` in transport params
- Assemble Pipeline with appropriate services
- Start the pipeline

### 4. Implement outbound/agent.py

Copy the inbound agent logic, then add (reference: `deepgram-voiceagent/outbound/agent.py`):
- `build_outbound_prompt()` for template variable substitution, with neutral fallbacks so no `{{...}}` placeholder is ever sent
- A greeting builder from `opening_reason` (default greeting when it is empty)
- An outbound call context that labels `To` as the customer's number and `From` as our caller ID
- `run_agent()` that accepts `opening_reason`, `objective`, `context` (from the `answer_url` query string) and renders the prompt and greeting

Do not add `CallManager`, `OutboundCallRecord` or `determine_outcome()` (legacy), and do not read the prompt from a `SYSTEM_PROMPT` env var; `system_prompt.md` is the only source.

### 5. Update pyproject.toml

Replace placeholder deps with real ones:
```bash
cd {example-name}
uv add {api-specific-package}
```

### 6. Update .env.example

Replace API placeholder vars with real variable names matching the `os.getenv()` calls in agent.py and utils.py.

## Verification

After implementation:
1. `uv run ruff check .` from the example directory — must be clean
2. `uv run python -c "from utils import *; print('utils OK')"` — imports work
3. `uv run python -c "from inbound.agent import run_agent; print('inbound OK')"` — imports work
4. `uv run python -c "from outbound.agent import run_agent, build_outbound_prompt; print('outbound OK')"` — imports work

Fix any lint or import errors before declaring Phase 2 complete.
