# xAI Realtime + Plivo Voice Agent (Native)

Real-time voice agent using the xAI realtime API for speech-to-speech conversations over Plivo telephony. This variant keeps the audio path as simple as possible: it forwards Plivo's mu-law 8 kHz audio straight to xAI and back with no transcoding, and relies on server-side VAD for turn detection.

For a variant that transcodes audio and uses local VAD, see the other xAI example variants in this repository.

## How this variant differs

| | reference variant | this example |
| --- | --- | --- |
| Audio path | Transcodes audio and uses extra processing | mu-law 8 kHz passthrough (`audio/pcmu`) |
| Turn detection | Client-side/local VAD | Realtime server-side VAD (`server_vad`) |
| Extra dependencies | numpy, scipy, torch, silero-vad | none |

The trade-off: fewer moving parts and lower latency, against the fine-grained control a local VAD gives you.

## Features

- Speech-to-speech using the xAI realtime API (no separate STT/TTS)
- mu-law 8 kHz passthrough — no audio conversion
- Server-side turn detection
- Barge-in — the caller can interrupt the agent mid-response
- Function calling (order status, SMS, callbacks, transfers, call control)
- Auto-configuration of Plivo webhooks on startup
- Inbound and outbound calls

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- xAI API key with realtime API access
- Plivo account with a phone number
- ngrok (for local development)

## Quick start

1. Install dependencies.

```sh
cd xai-realtime-native
uv sync
```

2. Copy the environment template.

```sh
cp .env.example .env
```

3. Edit `.env` with your credentials:

```sh
XAI_API_KEY=your_xai_api_key
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
PLIVO_PHONE_NUMBER=+1234567890
PLIVO_TEST_NUMBER=+1234567891
PUBLIC_URL=https://your-ngrok-url.ngrok-free.app
```

4. Start ngrok.

```sh
ngrok http 8000
```

Copy the ngrok URL into `PUBLIC_URL`.

5. Run the server.

```sh
# Inbound (receives calls)
uv run python -m inbound.server

# Outbound (places calls)
uv run python -m outbound.server
```

The inbound server auto-configures your Plivo number's answer webhook on startup. Call the number to talk to the agent. To place an outbound call:

```sh
curl -X POST "http://localhost:8000/outbound/call?phone_number=+1234567890"
```

## How it works

```
Phone call ──▶ Plivo ──▶ FastAPI server ──▶ xAI realtime API
           ◀──       ◀──               ◀──
```

1. Plivo hits `/answer` and receives `<Stream>` XML pointing at the server's WebSocket.
2. Plivo streams the call audio (mu-law 8 kHz) over the WebSocket.
3. The server forwards each mu-law payload to xAI as `input_audio_buffer.append`, unchanged.
4. xAI streams mu-law audio back, which the server relays to Plivo as `playAudio`.
5. On xAI's `input_audio_buffer.speech_started`, the server sends `clearAudio` to Plivo so the agent stops immediately for barge-in.

## Project structure

```
xai-realtime-native/
├── inbound/
├── outbound/
├── tests/
├── utils.py
├── pyproject.toml
├── Dockerfile
└── .env.example
```

## Configuration

| Variable | Description |
| --- | --- |
| `XAI_API_KEY` | xAI API key |
| `XAI_REALTIME_MODEL` | Optional model name appended to the realtime URL when explicitly set |
| `XAI_VOICE` | Voice name (default `Sal`) |
| `PLIVO_AUTH_ID` / `PLIVO_AUTH_TOKEN` | Plivo credentials |
| `PLIVO_PHONE_NUMBER` | Number to auto-configure on startup |
| `PLIVO_TEST_NUMBER` | Second Plivo number used by live-call tests |
| `PUBLIC_URL` | Public HTTPS URL for webhooks (ngrok in development) |
| `SERVER_PORT` | Server port (default 8000) |

## Tests

Run the offline unit and local integration checks:

```sh
uv run --group dev python -m pytest tests/test_integration.py -v -k "unit or local"
```

Run the live API or telephony checks only after you set the required credentials:

```sh
uv run --group dev python -m pytest tests/test_e2e_live.py -v
uv run --group dev python -m pytest tests/test_live_call.py -v -s
uv run --group dev python -m pytest tests/test_outbound_call.py -v -s
```
