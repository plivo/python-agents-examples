# Create and Run Tests

**Phase 3**: Write the test suite and run all tests for a voice agent example.

## Arguments

- `$ARGUMENTS` should contain: `{example-name}`

## Instructions

Read `CLAUDE.md` for testing requirements. Read the existing test files in `{example-name}/tests/` and the agent code to understand what to test.

Use `grok3-voice-native/tests/` as the primary reference for test patterns.

**Signed requests.** The servers check Plivo's signature on every webhook and on `/ws` (CLAUDE.md "Webhook Authentication"), so tests must sign what they send. Reference: `deepgram-voiceagent/tests/` (`grok3-voice-native` predates this).
- Test servers run with a dummy `PLIVO_AUTH_TOKEN` and `PUBLIC_URL` set to their own `http://localhost:<port>` URL
- Webhooks: `signed_webhook()` / `plivo_signature_headers()` from `tests/helpers.py`
- `/ws`: take the stream URL from a signed answer webhook's XML and connect with `stream_signature_headers(stream_url, token)`
- Never add a switch that turns the check off for tests

### 1. Write test_integration.py

Create 4 test classes following `grok3-voice-native/tests/test_integration.py`:

**TestUnitAudioConversion** (offline, `-k "unit"`). Tests the `utils.py` codec when utils.py has one. When it has none (μ-law pass-through, or a framework that converts), put a small test-only decoder in `tests/helpers.py` and test that instead (reference: `gpt4o-modulatevelma2-cartesiasonic3-pipecat/tests/`); never add a codec to utils.py just for tests:
- `test_ulaw_to_pcm_conversion` — silence bytes → PCM, check size and amplitude
- `test_pcm_to_ulaw_conversion` — PCM silence → μ-law, check size
- `test_audio_roundtrip` — 440Hz sine wave, PCM→ulaw→PCM, check correlation > 0.9

**TestUnitPhoneNormalization** (offline, `-k "unit"`):
- `test_normalize_e164_format` — "+16572338892" roundtrip
- `test_normalize_with_spaces` — "+1 657-233-8892" → E.164
- `test_normalize_local_format` — "(657) 233-8892" → E.164

**TestUnitWebhookAuth** (offline, `-k "unit"`, FastAPI `TestClient`; reference: `deepgram-voiceagent/tests/test_integration.py`):
- every webhook route accepts a signed request and returns 403 to an unsigned one
- `/ws` runs the agent when signed; unsigned or wrongly signed is refused (close code 1008) and no agent starts
- the server refuses to start with an empty `PLIVO_AUTH_TOKEN`

**TestLocalIntegration** (starts server, needs API key):
- `server_process` fixture: starts `inbound.server` on TEST_PORT
- `test_local_health_check` — GET / returns 200
- `test_local_answer_webhook` — signed POST /answer returns XML with `<Stream>`; unsigned returns 403
- `test_local_websocket_connection` — connect with signed headers, send start event, receive playAudio; an unsigned connection gets HTTP 403
- `test_local_audio_quality` — receive audio chunks, verify RMS > 500

**Test{API}Integration** (needs API key):
- `test_{api}_connection` — connect to API, configure session, verify response
- `test_{api}_text_to_audio` — send text, receive audio chunks

### 2. Write test_e2e_live.py

Reference: `grok3-voice-native/tests/test_e2e_live.py`

Test that the agent works end-to-end with the real API (no phone call):
- Start server subprocess
- Connect via WebSocket
- Send start event + audio frames
- Verify audio response is received
- Verify response is contextually appropriate

### 3. Write test_live_call.py

Reference: `grok3-voice-native/tests/test_live_call.py`

Real inbound call test:
1. Start server as subprocess
2. Start ngrok tunnel (using helpers.start_ngrok)
3. Configure Plivo webhooks
4. Place call from PLIVO_TEST_NUMBER to PLIVO_PHONE_NUMBER
5. Wait for call to go live, start recording
6. Let greeting play ~20s
7. Hang up, poll for recording
8. Download MP3, transcribe with faster-whisper
9. Verify transcript contains greeting words

Skip if credentials not configured: `pytestmark = pytest.mark.skipif(...)`

### 4. Write test_outbound_call.py

Similar to test_live_call.py but for outbound:
1. Start outbound server subprocess
2. Start ngrok tunnel
3. Place the call with Plivo's Make Call API (`client.calls.create(from_=PLIVO_PHONE_NUMBER, to_=PLIVO_TEST_NUMBER, answer_url=<tunnel>/outbound/answer?opening_reason=..., ...)`)
4. Wait for call to connect
5. Record, transcribe, verify greeting

### 5. Write test_multiturn_voice.py

Multi-turn conversation test:
1. Same setup as live call
2. Wait for greeting
3. Inject question via Plivo TTS (`client.calls.speak`)
4. Wait for response
5. Verify response is contextually appropriate
6. Optionally test barge-in

### 6. Run all tests

```bash
cd {example-name}

# First, run lint
uv run ruff check .

# Run unit tests (must pass offline)
uv run pytest tests/test_integration.py -v -k "unit"

# Run local integration tests (needs API key)
uv run pytest tests/test_integration.py -v -k "local"

# Run API integration tests (needs API key)
uv run pytest tests/test_integration.py -v -k "not unit and not local"
```

### 7. Fix any failures

- Fix lint errors first
- Fix unit test failures (these MUST pass)
- Fix integration test failures if API key is available
- Report results clearly: which tests pass, which skip, which fail and why

## Verification

All of these must succeed:
1. `uv run ruff check .` — zero errors
2. `uv run pytest tests/test_integration.py -v -k "unit"` — all pass
3. Test files exist: `test_integration.py`, `test_e2e_live.py`, `test_live_call.py`, `test_outbound_call.py`, `test_multiturn_voice.py`

Report: X unit tests passed, Y integration tests passed/skipped, Z lint errors.
