# Contributing

Guidelines for contributing voice agent examples to this repository.

## Project Naming Convention

Each example project uses Plivo as the telephony platform. Project names follow this structure:

```
{llm-provider+series}-{stt-provider+series}-{tts-provider+series}-{orchestration}[-{variant}]
```

Every component includes the **provider name** and **model series**. The series identifies the API contract; the size variant (mini/nano/pro/flash) is config in `.env`, not part of the folder name.

### LLM: `{provider}{version}`

Drop the size class (mini/nano/pro/flash) unless two different sizes are used together.

| Model | Folder component | Notes |
|---|---|---|
| `gpt-5.4-mini` | `gpt5.4` | drop "mini" |
| `gpt-4.1-mini` + `gpt-4.1` (dual) | `gpt4.1mini-gpt4.1` | two sizes → keep both |
| `gemini-2.5-flash` (live API) | `gemini2.5-live` | drop "flash"; `-live` = S2S API |
| `gpt-realtime-1.5` (S2S) | `gptrealtime1.5` | "realtime" is the model name |
| `grok-3-fast-voice` (S2S) | `grok3-voice` | |

### STT/TTS: `{provider}{model-name}{version}`

| Model | Folder component |
|---|---|
| Deepgram `nova-3` | `deepgramnova3` |
| AssemblyAI `u3-rt-pro` | `assemblyaiu3` |
| ElevenLabs `eleven_flash_v2_5` | `elevenflashv2.5` |
| Cartesia `sonic-3` | `cartesiasonic3` |
| OpenAI `gpt-4o-mini-tts` | `openaitts4o` |

### Orchestration

| Value | Meaning |
|-------|---------|
| `native` | Direct API integration, no orchestration framework |
| `pipecat` | Uses the Pipecat framework |
| `livekit` | Uses the LiveKit Agents framework |
| `vapi` | Uses the Vapi framework |

### Managed voice-agent platforms: `{provider}-{product}[-{variant}]`

When a hosted product runs the whole STT → LLM → TTS loop (including turn detection and barge-in), name the example after that product instead of its component models — e.g. `deepgram-voiceagent` for Deepgram's "Voice Agent API".

- Use the platform's **own** product name as branded in its docs (lowercased, no spaces/punctuation, "API" dropped). Don't borrow another platform's term or apply a common label to all platforms.
- No model, orchestration, or VAD tokens. Models are `.env` config that accepts the platform's documented values verbatim.
- One example per platform; new model combinations go in its README.
- Variants are not predefined: propose one only when the integration itself changes (not a model swap), and get it approved in PR review.
- Add `[tool.voice-agent-example]` / `category = "managed-platform"` to `pyproject.toml`.

See the "Managed Voice-Agent Platforms" section of [CLAUDE.md](./CLAUDE.md) for the full rules.

### Examples

| Project Name | LLM | STT | TTS | Framework |
|---|---|---|---|---|
| `gpt5.4-assemblyaiu3-cartesiasonic3-native` | GPT 5.4 | AssemblyAI U3 | Cartesia Sonic 3 | None |
| `gemini2.5-live-native` | Gemini 2.5 (S2S) | Gemini 2.5 (S2S) | Gemini 2.5 (S2S) | None |
| `gpt4.1-deepgramnova3-elevenflashv2.5-vapi` | GPT 4.1 | Deepgram Nova 3 | ElevenLabs Flash v2.5 | Vapi |
| `deepgram-voiceagent` | Configurable (default GPT-4.1 mini, hosted by Deepgram) | Configurable (default Deepgram Flux) | Configurable (default Deepgram Aura-2) | Managed platform (Deepgram Voice Agent API) |

## Project Structure

Each example project should be a self-contained directory at the repository root:

```
{project-name}/
├── inbound/
│   ├── agent.py        # AI-specific voice agent class
│   ├── server.py       # FastAPI: /answer, /ws, /hangup
│   └── system_prompt.md
├── outbound/
│   ├── agent.py        # Same agent + OutboundCallRecord, CallManager
│   ├── server.py       # FastAPI: /outbound/call, /outbound/ws
│   └── system_prompt.md
├── utils.py            # Audio conversion, VAD, phone utils
├── tests/              # Unit, integration, e2e, live call tests
├── pyproject.toml
├── .env.example
├── .gitignore
├── Dockerfile
└── README.md
```

## Adding a New Example

1. Create a new directory following the naming convention
2. Include a README.md with:
   - Brief description of the voice agent
   - Prerequisites and dependencies
   - Setup and configuration steps
   - How to run the example
3. Use `.env.example` for environment variables (never commit actual credentials)
4. Test that the example works end-to-end with Plivo
