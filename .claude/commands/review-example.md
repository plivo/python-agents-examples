# Review Voice Agent Example

**Phase 4**: Quality gate — run a comprehensive checklist against the example. Blocks documentation until all checks pass.

## Arguments

- `$ARGUMENTS` should contain: `{example-name}`

## Instructions

Read `CLAUDE.md` for all rules. Then systematically check every item below.

### Detection

First, determine orchestration type by reading `{example-name}/inbound/agent.py`:
- If it imports from `pipecat` or `livekit` → **framework**
- Otherwise → **native**

### Checklist

Run through EVERY item. Report PASS or FAIL with details for each.

#### Naming (1 check)

0. **Directory name follows convention**: Must match `{provider}-...-{orchestration}[-{variant}]` where orchestration is `native|pipecat|livekit|vapi` and variant (if present) is `no-vad|webrtcvad`

#### Structure (8 checks)

1. **All canonical files exist**: Check every file from the canonical structure in CLAUDE.md
2. **`.env.example` naming**: Must be `.env.example` (with leading dot), NOT `env.example`
3. **No stale `env.example`**: File `env.example` (without dot) must NOT exist
4. **`__init__.py` files**: Must exist in `inbound/`, `outbound/`, `tests/`
5. **`system_prompt.md`**: Must exist in both `inbound/` and `outbound/`
6. **`pyproject.toml` fields**: Has `name`, `version`, `description`, `requires-python`, `dependencies`
7. **Test files complete**: All 7 test files exist in `tests/`
8. **Dockerfile exists**: And COPY paths match actual structure

#### Config Placement (4 checks)

9. **Server constants in server.py**: `SERVER_PORT`, `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_PHONE_NUMBER`, `PUBLIC_URL` are imported from utils or defined in server.py — NOT in agent.py
10. **Agent constants in agent.py**: API keys, model names, voice names, `PLIVO_CHUNK_SIZE`, `SYSTEM_PROMPT` — NOT in utils.py
11. **Utils only has utility constants**: No `SERVER_PORT`, `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_PHONE_NUMBER`, `PUBLIC_URL`, API keys, or model names in utils.py
12. **No config leakage**: grep for common config constants to verify placement

#### Audio Pipeline (4 checks)

13. **PLIVO_CHUNK_SIZE = 160**: Defined in agent.py `_send_to_plivo()` method
14. **playAudio format correct**: `contentType: "audio/x-mulaw"`, `sampleRate: 8000`, base64 payload
15. **Stream XML correct**: `bidirectional=True`, `keepCallAlive=True`, `contentType="audio/x-mulaw;rate=8000"`
16. **Sample rates correct**: Check `plivo_to_{api}()` and `{api}_to_plivo()` use correct rates

#### VAD (4 checks — native only, skip for framework)

17. **SileroVADProcessor used**: Imported and instantiated in agent.__init__
18. **plivo_to_vad exists**: In utils.py, converts to float32 16kHz
19. **VAD-driven turn management**: speech_ended triggers buffer commit + response.create
20. **Barge-in implemented**: speech_started + _is_responding triggers response.cancel + queue drain

#### VAD (1 check — framework only)

17f. **vad_enabled=True**: Set in framework transport params

#### Agent Pattern (3 checks)

21. **3-task pattern** (native): plivo_rx, {api}_rx, plivo_tx tasks created and managed
22. **`_pending` pattern**: Uses `done, _pending = await asyncio.wait(...)` (not `done, pending =`)
23. **Task cleanup in finally**: All tasks cancelled with `contextlib.suppress(asyncio.CancelledError)`

#### Code Quality (5 checks)

24. **`from __future__ import annotations`**: Present in all .py files (except __init__.py)
25. **loguru logging**: Uses `from loguru import logger`, NOT `import logging`
26. **No hardcoded keys**: No API keys, tokens, or secrets in source code
27. **python-dotenv**: `load_dotenv()` called in utils.py
28. **Lint clean**: `uv run ruff check .` returns 0 errors

#### Testing (3 checks)

29. **Unit tests pass**: `uv run pytest tests/test_integration.py -v -k "unit"` — all pass
30. **E2E test structure**: test_live_call.py has fixtures for server_process, ngrok_tunnel, plivo_configured
31. **Outbound test exists**: test_outbound_call.py tests the outbound flow

## Output Format

```
## Review: {example-name}
Orchestration: native | framework

### Structure
1. [PASS] All canonical files exist
2. [PASS] .env.example naming correct
...

### Config Placement
9. [FAIL] SERVER_PORT found in utils.py — should be in server.py only
...

### Summary
PASSED: 28/31
FAILED: 3/31
BLOCKED: Cannot proceed to /document-example until all checks pass.

### Fixes Required
- Move SERVER_PORT from utils.py to server.py
- ...
```

If ALL checks pass: "All checks passed. Ready for Phase 5: /document-example {example-name}"

If any FAIL: List required fixes. Do NOT proceed to documentation. Fix the issues first, then re-run `/review-example`.
