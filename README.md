# Python Voice Agent Examples

A collection of production-ready voice AI agent examples built with [Plivo](https://www.plivo.com/). Each example demonstrates a different combination of AI models and frameworks for building real-time phone-based voice agents on the Plivo voice AI platform.

## How It Works

All examples follow the same general pattern:

```
┌─────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Phone  │────▶│   Plivo     │────▶│   Server    │────▶│  AI Agent   │
│  Call   │◀────│  (Voice AI) │◀────│  (FastAPI)  │◀────│             │
└─────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

1. A phone call comes in (or is initiated) through Plivo
2. Plivo hits a webhook on your FastAPI server
3. The server establishes a bidirectional WebSocket for audio streaming
4. The AI agent processes speech and generates responses in real-time

## Examples

Each example directory is self-contained with its own dependencies, environment configuration, and documentation. Directory names follow the convention `{llm}-{stt}-{tts}-{framework}`, or `{provider}-{product}` for managed voice-agent platforms (see [CONTRIBUTING.md](./CONTRIBUTING.md) for details).

### Speech-to-Speech (S2S)

These examples use models that handle both speech input and output natively — the simplest architecture with the fewest moving parts.

| Example | Model | Framework | Highlights |
|---------|-------|-----------|------------|
| [gemini2.5-live-native](./gemini2.5-live-native/) | Gemini 2.5 Live | None | Direct API integration, function calling, auto-webhook config |
| [gemini2.5-live-pipecat](./gemini2.5-live-pipecat/) | Gemini 2.5 Live | Pipecat | Modular pipeline, built-in VAD, less code |
| [gptrealtime1.5-native](./gptrealtime1.5-native/) | GPT Realtime 1.5 | None | Silero VAD, barge-in support, function calling |
| [grok3-voice-native](./grok3-voice-native/) | Grok 3 Voice | None | Silero VAD, barge-in support, function calling |

### Managed Voice-Agent Platforms

These examples connect Plivo to a hosted voice-agent product that runs STT, LLM, TTS, turn detection and barge-in on its side. Models are chosen in `.env` using the platform's own values.

| Example | Platform | Default models (configurable) | Highlights |
|---------|----------|-------------------------------|------------|
| [deepgram-voiceagent](./deepgram-voiceagent/) | Deepgram Voice Agent API | Flux STT, GPT-4.1 mini, Aura-2 TTS | Single WebSocket, μ-law 8kHz end to end (no transcoding), Flux end-of-turn, client-side function calling, checkpoint-tracked playback |

### STT + LLM + TTS Pipeline

These examples wire up separate providers for speech-to-text, language model, and text-to-speech — offering more flexibility to mix and match.

| Example | STT | LLM | TTS | Framework |
|---------|-----|-----|-----|-----------|
| [gemini2-deepgramnova2-cartesiasonic2-native](./gemini2-deepgramnova2-cartesiasonic2-native/) | Deepgram Nova 2 | Gemini 2 | Cartesia Sonic 2 | None |
| [gemini2-deepgramnova2-elevenflashv2.5-native](./gemini2-deepgramnova2-elevenflashv2.5-native/) | Deepgram Nova 2 | Gemini 2 | ElevenLabs Flash v2.5 | None |
| [gpt4o-deepgramnova3-openaitts4o-pipecat](./gpt4o-deepgramnova3-openaitts4o-pipecat/) | Deepgram Nova 3 | GPT-4o | OpenAI TTS 4o | Pipecat |
| [gpt5.4-assemblyaiu3-cartesiasonic3-native](./gpt5.4-assemblyaiu3-cartesiasonic3-native/) | AssemblyAI U3 | GPT 5.4 | Cartesia Sonic 3 | None |
| [daily-plivo](./daily-plivo/) | Deepgram | OpenAI | Cartesia | Pipecat + Daily |

## Prerequisites

- Python 3.10+ (3.12 recommended)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- [ngrok](https://ngrok.com/) for local development
- A [Plivo](https://www.plivo.com/) account with a phone number
- API keys for the AI services used by your chosen example

## Quick Start

1. **Choose an example** from the tables above and navigate to its directory:
   ```bash
   cd gemini2.5-live-native  # or any other example
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   # or: uv pip install -r requirements.txt
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` with your API keys and Plivo credentials.

4. **Start ngrok** (in a separate terminal):
   ```bash
   ngrok http 8000  # port varies by example
   ```

5. **Update `PUBLIC_URL`** in `.env` with your ngrok HTTPS URL.

6. **Run the server:**
   ```bash
   uv run python server.py  # entry point varies by example
   ```

7. **Call your Plivo phone number** to talk to the agent.

See each example's README for detailed setup and configuration.

## Contributing

Want to add a new voice agent example? See [CONTRIBUTING.md](./CONTRIBUTING.md) for the project naming convention, required structure, and submission guidelines.

## License

MIT
