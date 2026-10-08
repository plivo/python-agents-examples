# Document Voice Agent Example

**Phase 5**: Generate comprehensive README and validate documentation completeness.

## Arguments

- `$ARGUMENTS` should contain: `{example-name}`

## Prerequisites

Phase 4 (`/review-example`) must have passed all checks. If not, run `/review-example` first.

## Instructions

Read `CLAUDE.md` for documentation requirements. Use `grok3-voice-native/README.md` as the template.

### 1. Generate README.md

Create a comprehensive README following this section structure (from `grok3-voice-native/README.md`):

#### Required Sections

1. **Title + one-line description**: `# {Example Name}\n\n{description}`
2. **Features**: bullet list of key features
3. **Prerequisites**: Python version, uv, API keys needed, Plivo account, ngrok
4. **Quick Start**: numbered steps
   - Install dependencies (`uv sync`)
   - Configure environment (`cp .env.example .env`)
   - Start ngrok
   - Run the server (inbound + outbound commands)
   - Make a test call
5. **Project Structure**: tree diagram matching actual files
6. **How It Works**: ASCII art diagram showing the audio pipeline flow
7. **Audio Formats**: table of stages, formats, and sample rates
8. **VAD Configuration** (native only): table of VAD parameters with defaults and descriptions
   OR **Framework Configuration** (framework only): framework-specific settings
9. **Function Calling**: table of available tools + example of adding a new one
10. **Configuration**: table of environment variables, descriptions, defaults
11. **Available Voices** (if applicable): table of voice options
12. **Testing**: commands for all test levels with descriptions
13. **Deployment**: Docker build and run commands
14. **Troubleshooting**: common issues and solutions specific to this API
15. **Webhook authentication**: which routes need Plivo's signature (the webhooks and `/ws`), that `PUBLIC_URL` must match the URL Plivo calls, what a rejected request looks like in the log, and that the stream signature does not cover the stream URL's query string. Reference: `deepgram-voiceagent/README.md`

#### Formatting Rules

- Use Markdown tables for structured data
- Include code blocks with language hints (`bash`, `python`)
- ASCII art diagrams for architecture (match grok3-voice-native style)
- Keep explanations concise but complete

### 2. Validate .env.example completeness

Scan all `os.getenv()` calls in the codebase:

```bash
grep -rn "os.getenv" {example-name}/ --include="*.py" | grep -v __pycache__ | grep -v .pyc
```

Verify that every environment variable referenced in code exists in `.env.example`.

Report any missing variables and add them.

### 3. Update root README.md

Read the root `README.md` and add the new example to the list of examples. Follow the existing format.

### 4. Final verification

Run these checks:
- `README.md` exists and has all required sections
- `.env.example` covers all `os.getenv()` calls
- Root README.md updated

## Output

Report:
- README.md created with N sections
- .env.example validated: X variables in code, Y in .env.example, Z missing
- Root README.md updated

Phase 5 complete. The example is ready for CI validation: `./scripts/validate-example.sh {example-name}`
