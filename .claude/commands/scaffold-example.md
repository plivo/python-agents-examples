# Scaffold a New Voice Agent Example

**Phase 1**: Create directory structure and boilerplate for a new voice agent example.

## Arguments

- `$ARGUMENTS` should contain: `{example-name} "{description}"` and optionally `native` or `framework`
- If orchestration type is not specified, ask the user: native (raw websockets, Silero VAD) or framework (Pipecat/LiveKit)?

## Instructions

Read `CLAUDE.md` for the canonical file structure and rules. Use `grok3-voice-native/` as the primary reference.

### 0. Validate the example name

Before creating any files, verify `{example-name}` follows the naming convention:
`{llm-provider+series}-{stt-provider+series}-{tts-provider+series}-{orchestration}[-{variant}]`

**Every component always includes provider name + model series.** The series identifies the API contract; the size variant (mini/nano/pro/flash) is `.env` config, not part of the folder name. Only include size when two different sizes are used together.

**LLM:** `{provider}{version}` — drop size class unless two sizes used together
- `gpt5.4`, `gpt4.1`, `gpt4o`, `gemini2`, `grok3`
- Dual-LLM: `gpt4.1mini-gpt4.1` (two sizes → keep both)
- S2S/multimodal: `gemini2.5-live`, `gemini3.1-live`, `gptrealtime1.5`, `grok3-voice`

**STT:** `{provider}{model-name}{version}`
- `deepgramnova2`, `deepgramnova3`, `deepgramflux`, `assemblyaiu3`, `sarvam`

**TTS:** `{provider}{model-name}{version}`
- `elevenflashv2.5`, `cartesiasonic2`, `cartesiasonic3`, `openaitts4o`, `groktts3`

Examples: `gpt5.4-assemblyaiu3-cartesiasonic3-native`, `gemini2.5-live-pipecat`, `gpt4.1-deepgramnova3-elevenflashv2.5-native`

- Must end with a known orchestration type: `native`, `pipecat`, `livekit`, `vapi`
- If a variant suffix follows the orchestration type, it must be a known variant: `no-vad`, `webrtcvad`
- Unknown suffixes should be flagged — ask the user before proceeding

**Managed voice-agent platform** (hosted product runs STT/LLM/TTS, turn detection and barge-in — see CLAUDE.md "Managed Voice-Agent Platforms"): the name is `{provider}-{product}[-{variant}]` instead, e.g. `deepgram-voiceagent`.
- `{product}` must mirror the platform's own branding for the product; do not reuse another platform's term. Confirm the product name with the user before creating files.
- No model/orchestration/VAD tokens. Add `[tool.voice-agent-example]` `category = "managed-platform"` to `pyproject.toml`.
- Variants are not predefined — if one seems needed, propose it and get the user's explicit approval.

If the name does not match, stop and ask the user to provide a corrected name.

### 1. Create the full directory structure

Create ALL of these files under `{example-name}/`:

```
inbound/__init__.py
inbound/agent.py
inbound/server.py
inbound/system_prompt.md
outbound/__init__.py
outbound/agent.py
outbound/server.py
outbound/system_prompt.md
utils.py
tests/__init__.py
tests/conftest.py
tests/helpers.py
tests/test_integration.py
tests/test_e2e_live.py
tests/test_live_call.py
tests/test_multiturn_voice.py
tests/test_outbound_call.py
pyproject.toml
.env.example
.gitignore
.pre-commit-config.yaml
Dockerfile
README.md
```

### 2. Copy shared boilerplate verbatim

Copy these files from `grok3-voice-native/` WITHOUT modification:
- `tests/conftest.py`
- `tests/helpers.py`
- `.gitignore`
- `.pre-commit-config.yaml`

### 3. Copy and adapt server files

Copy `grok3-voice-native/inbound/server.py` → `{example-name}/inbound/server.py`:
- Replace "Grok" with the new agent name in FastAPI title/description
- Replace `grok-plivo-voice-agent` with the new service name
- Replace Plivo app name (`Grok_Voice_Agent` → `{NewAgent}_Voice_Agent`)
- Keep ALL routes, webhook logic, and WebSocket handling identical

Do the same for `grok3-voice-native/outbound/server.py` → `{example-name}/outbound/server.py`.

### 4. Create system prompts

Copy `grok3-voice-native/inbound/system_prompt.md` and `outbound/system_prompt.md` as starting templates. The user can customize these later.

### 5. Create agent.py skeletons

**For native orchestration**:
- `inbound/agent.py`: Skeleton with the 3-task pattern from `grok3-voice-native/inbound/agent.py`
  - Class with `__init__`, `run()`, `_run_streaming_tasks()`, `_receive_from_plivo()`, `_receive_from_{api}()`, `_send_to_plivo()`
  - Import and use `SileroVADProcessor` from utils
  - All method bodies have `# TODO: Implement {api}-specific logic` comments
  - Include the tool functions (check_order_status, send_sms, etc.) from reference
  - Include public `run_agent()` function
- `outbound/agent.py`: Same skeleton + `OutboundCallRecord`, `CallManager`, `determine_outcome` from `grok3-voice-native/outbound/agent.py`

**For framework orchestration**:
- `inbound/agent.py`: Skeleton with `run_agent()` function that assembles a Pipeline
  - `# TODO: Configure {framework} services and pipeline` comments
  - No custom agent class (framework handles task management)
- `outbound/agent.py`: Similar skeleton + `OutboundCallRecord`, `CallManager`

### 6. Create utils.py

**For native**:
- Copy the full `grok3-voice-native/utils.py` structure
- Replace Grok-specific constants with placeholders for the new API:
  - `{API}_SAMPLE_RATE = 24000  # TODO: Set correct sample rate`
  - Rename `plivo_to_grok` → `plivo_to_{api}`, `grok_to_plivo` → `{api}_to_plivo`
- Keep `SileroVADProcessor` class, `plivo_to_vad()`, all audio conversion functions, `normalize_phone_number()`
- Keep only utility-owned constants (sample rates, VAD params, DEFAULT_COUNTRY_CODE)
- Do NOT include server constants (SERVER_PORT, PLIVO_AUTH_ID, etc.) or agent constants (API keys, model names)

**For framework**:
- Same audio conversion functions but NO SileroVADProcessor, NO plivo_to_vad()
- No VAD constants

### 7. Create pyproject.toml

Based on `grok3-voice-native/pyproject.toml`:
- Update project name and description
- Keep common deps: fastapi, uvicorn, websockets, plivo, python-dotenv, python-multipart, loguru, numpy, scipy, phonenumbers
- For native: add `silero-vad>=5.1`, `torch>=2.0.0`
- Add `# TODO: Add {api}-specific dependencies` comment
- Keep dev deps and ruff/pytest config identical

### 8. Create .env.example

Based on `grok3-voice-native/.env.example`:
- Replace xAI-specific vars with placeholders for the new API
- Keep Plivo section, DEFAULT_COUNTRY_CODE, PUBLIC_URL, SERVER_PORT
- For native: keep Silero VAD configuration section
- For framework: omit VAD section

### 9. Create Dockerfile

Based on `grok3-voice-native/Dockerfile` — adjust only the COPY paths if needed.

### 10. Create README.md

Create a minimal placeholder README with the example name and description. Full README is generated in Phase 5 (`/document-example`).

## Verification

After creating all files, run:
```bash
ls -la {example-name}/
ls -la {example-name}/inbound/
ls -la {example-name}/outbound/
ls -la {example-name}/tests/
```

Confirm all files exist. Report the file count and any issues.
