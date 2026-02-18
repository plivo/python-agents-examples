#!/usr/bin/env bash
# validate-example.sh — CI-runnable validation for voice agent examples
#
# Usage: ./scripts/validate-example.sh <example-name>
# Exit code: 0 = all checks pass, 1 = one or more checks failed
#
# This script validates that a voice agent example follows the canonical
# structure and conventions defined in CLAUDE.md.

set -euo pipefail

# =============================================================================
# Setup
# =============================================================================

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <example-name>"
    echo "Example: $0 grok-voice-native"
    exit 1
fi

EXAMPLE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
EXAMPLE_DIR="$REPO_ROOT/$EXAMPLE"

if [[ ! -d "$EXAMPLE_DIR" ]]; then
    echo "ERROR: Directory '$EXAMPLE_DIR' does not exist"
    exit 1
fi

PASS=0
FAIL=0
SKIP=0

pass() {
    echo "  [PASS] $1"
    PASS=$((PASS + 1))
}

fail() {
    echo "  [FAIL] $1"
    FAIL=$((FAIL + 1))
}

skip() {
    echo "  [SKIP] $1"
    SKIP=$((SKIP + 1))
}

# =============================================================================
# Detect orchestration type
# =============================================================================

ORCHESTRATION="native"
if grep -q "pipecat\|livekit" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
    ORCHESTRATION="framework"
fi

echo "=========================================="
echo "Validating: $EXAMPLE"
echo "Orchestration: $ORCHESTRATION"
echo "=========================================="
echo ""

# =============================================================================
# 0. Naming Convention Check
# =============================================================================

echo "--- Naming ---"

# Extract orchestration type and optional variant from directory name
# Convention: {provider}-{optional-stt}-{optional-tts}-{orchestration}[-{variant}]
KNOWN_ORCH="native|pipecat|livekit|vapi"
KNOWN_VARIANTS="no-vad|webrtcvad"

name_valid=false
# Check if name ends with a known orchestration type (with optional known variant)
if [[ "$EXAMPLE" =~ -($KNOWN_ORCH)$ ]]; then
    name_valid=true
elif [[ "$EXAMPLE" =~ -($KNOWN_ORCH)-($KNOWN_VARIANTS)$ ]]; then
    name_valid=true
fi

if $name_valid; then
    pass "Directory name follows naming convention"
else
    fail "Directory name '$EXAMPLE' does not follow naming convention ({provider}-...-{orchestration}[-{variant}])"
fi

echo ""

# =============================================================================
# 1. Structure Checks
# =============================================================================

echo "--- Structure ---"

# Required files (canonical structure)
REQUIRED_FILES=(
    "inbound/__init__.py"
    "inbound/agent.py"
    "inbound/server.py"
    "inbound/system_prompt.md"
    "outbound/__init__.py"
    "outbound/agent.py"
    "outbound/server.py"
    "outbound/system_prompt.md"
    "utils.py"
    "tests/__init__.py"
    "tests/conftest.py"
    "tests/helpers.py"
    "tests/test_integration.py"
    "tests/test_e2e_live.py"
    "tests/test_live_call.py"
    "tests/test_multiturn_voice.py"
    "tests/test_outbound_call.py"
    "pyproject.toml"
    ".env.example"
    ".gitignore"
    ".pre-commit-config.yaml"
    "Dockerfile"
    "README.md"
)

missing_files=()
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$EXAMPLE_DIR/$f" ]]; then
        missing_files+=("$f")
    fi
done

if [[ ${#missing_files[@]} -eq 0 ]]; then
    pass "All ${#REQUIRED_FILES[@]} canonical files exist"
else
    fail "Missing files: ${missing_files[*]}"
fi

# .env.example naming (leading dot)
if [[ -f "$EXAMPLE_DIR/.env.example" ]]; then
    pass ".env.example has leading dot"
else
    fail ".env.example missing (no leading dot)"
fi

# No stale env.example (without dot)
if [[ -f "$EXAMPLE_DIR/env.example" ]]; then
    fail "Stale env.example (without dot) exists — remove it"
else
    pass "No stale env.example"
fi

# pyproject.toml required fields
if [[ -f "$EXAMPLE_DIR/pyproject.toml" ]]; then
    pyproject_ok=true
    for field in "name" "version" "description" "requires-python"; do
        if ! grep -q "$field" "$EXAMPLE_DIR/pyproject.toml"; then
            pyproject_ok=false
            break
        fi
    done
    if $pyproject_ok; then
        pass "pyproject.toml has required fields"
    else
        fail "pyproject.toml missing required fields (name, version, description, requires-python)"
    fi
else
    fail "pyproject.toml not found"
fi

echo ""

# =============================================================================
# 2. Config Placement Checks
# =============================================================================

echo "--- Config Placement ---"

# Check utils.py does NOT contain server/agent config
if [[ -f "$EXAMPLE_DIR/utils.py" ]]; then
    leaked_configs=()

    # Server constants that should NOT be in utils.py
    for const in "SERVER_PORT" "PLIVO_AUTH_ID" "PLIVO_AUTH_TOKEN" "PLIVO_PHONE_NUMBER" "PUBLIC_URL"; do
        if grep -q "^${const}\s*=" "$EXAMPLE_DIR/utils.py" 2>/dev/null; then
            leaked_configs+=("$const")
        fi
    done

    if [[ ${#leaked_configs[@]} -eq 0 ]]; then
        pass "utils.py has no server/agent config constants"
    else
        fail "utils.py contains config that belongs elsewhere: ${leaked_configs[*]}"
    fi
else
    fail "utils.py not found"
fi

# Check PLIVO_CHUNK_SIZE is in agent.py, not utils.py
if grep -rq "PLIVO_CHUNK_SIZE" "$EXAMPLE_DIR/utils.py" 2>/dev/null; then
    fail "PLIVO_CHUNK_SIZE found in utils.py — should be in agent.py"
else
    pass "PLIVO_CHUNK_SIZE not in utils.py"
fi

echo ""

# =============================================================================
# 3. Audio Pipeline Checks
# =============================================================================

echo "--- Audio Pipeline ---"

# PLIVO_CHUNK_SIZE = 160
if grep -rq "PLIVO_CHUNK_SIZE.*=.*160\|PLIVO_CHUNK_SIZE = 160" "$EXAMPLE_DIR/inbound/agent.py" "$EXAMPLE_DIR/outbound/agent.py" 2>/dev/null; then
    pass "PLIVO_CHUNK_SIZE = 160 found in agent.py"
else
    fail "PLIVO_CHUNK_SIZE = 160 not found in agent.py"
fi

# playAudio format
if grep -rq "audio/x-mulaw" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
    pass "playAudio uses audio/x-mulaw content type"
else
    fail "playAudio content type 'audio/x-mulaw' not found in inbound/agent.py"
fi

# Stream XML content type
if grep -rq "audio/x-mulaw" "$EXAMPLE_DIR/inbound/server.py" 2>/dev/null; then
    pass "Stream XML uses audio/x-mulaw content type"
else
    fail "Stream XML content type not found in inbound/server.py"
fi

echo ""

# =============================================================================
# 4. VAD Checks
# =============================================================================

echo "--- VAD ---"

if [[ "$EXAMPLE" == *"-no-vad"* ]]; then
    skip "SileroVADProcessor (no-vad variant — uses server-side VAD)"
    skip "plivo_to_vad (no-vad variant)"
    skip "VAD-driven turn management (no-vad variant)"
    skip "Barge-in handling (no-vad variant)"
    skip "silero-vad dependency (no-vad variant)"
elif [[ "$EXAMPLE" == *"-webrtcvad"* ]]; then
    skip "SileroVADProcessor (webrtcvad variant — uses WebRTC VAD)"
    skip "plivo_to_vad (webrtcvad variant)"
    # webrtcvad still needs speech detection and barge-in
    if grep -rq "speech_ended\|is_speech" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "VAD-driven turn management"
    else
        fail "VAD-driven turn management not found in inbound/agent.py"
    fi
    if grep -rq "speech_started\|barge.in\|interrupt" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "Barge-in handling"
    else
        fail "Barge-in handling not found in inbound/agent.py"
    fi
    skip "silero-vad dependency (webrtcvad variant)"
elif [[ "$ORCHESTRATION" == "native" ]]; then
    # SileroVADProcessor used
    if grep -rq "SileroVADProcessor" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "SileroVADProcessor imported in inbound/agent.py"
    else
        fail "SileroVADProcessor not found in inbound/agent.py"
    fi

    # plivo_to_vad exists in utils
    if grep -q "def plivo_to_vad" "$EXAMPLE_DIR/utils.py" 2>/dev/null; then
        pass "plivo_to_vad() defined in utils.py"
    else
        fail "plivo_to_vad() not found in utils.py"
    fi

    # VAD-driven turn management (speech_ended)
    if grep -rq "speech_ended" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "VAD-driven turn management (speech_ended handling)"
    else
        fail "speech_ended handling not found in inbound/agent.py"
    fi

    # Barge-in (speech_started + response cancel)
    if grep -rq "speech_started" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "Barge-in handling (speech_started)"
    else
        fail "speech_started handling not found in inbound/agent.py"
    fi

    # silero-vad in pyproject.toml
    if grep -q "silero-vad" "$EXAMPLE_DIR/pyproject.toml" 2>/dev/null; then
        pass "silero-vad in pyproject.toml dependencies"
    else
        fail "silero-vad not found in pyproject.toml"
    fi
else
    # Framework: check vad_enabled
    if grep -rq "vad_enabled.*True\|vad_enabled=True" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "vad_enabled=True found in framework config"
    else
        fail "vad_enabled=True not found in inbound/agent.py"
    fi
    skip "SileroVADProcessor (framework uses built-in VAD)"
    skip "plivo_to_vad (framework uses built-in VAD)"
fi

echo ""

# =============================================================================
# 5. Code Quality Checks
# =============================================================================

echo "--- Code Quality ---"

# from __future__ import annotations
py_files_missing_annotations=()
while IFS= read -r pyfile; do
    # Skip __init__.py files
    if [[ "$(basename "$pyfile")" == "__init__.py" ]]; then
        continue
    fi
    if ! grep -q "from __future__ import annotations" "$pyfile" 2>/dev/null; then
        py_files_missing_annotations+=("$(basename "$pyfile")")
    fi
done < <(find "$EXAMPLE_DIR" -name "*.py" -not -path "*/__pycache__/*" -not -path "*/.venv/*")

if [[ ${#py_files_missing_annotations[@]} -eq 0 ]]; then
    pass "All .py files have 'from __future__ import annotations'"
else
    fail "Missing 'from __future__ import annotations' in: ${py_files_missing_annotations[*]}"
fi

# loguru usage
if grep -rq "from loguru import logger" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
    pass "Uses loguru for logging"
else
    fail "loguru not used in inbound/agent.py"
fi

# No hardcoded API keys (basic check)
hardcoded_found=false
for pattern in 'sk-[a-zA-Z0-9]{20,}' 'xai-[a-zA-Z0-9]{20,}' 'AIza[a-zA-Z0-9]{30,}'; do
    if grep -rqE "$pattern" "$EXAMPLE_DIR/" --include="*.py" 2>/dev/null; then
        hardcoded_found=true
        break
    fi
done
if $hardcoded_found; then
    fail "Possible hardcoded API key found in source code"
else
    pass "No hardcoded API keys detected"
fi

# No credentials in .env.example
if [[ -f "$EXAMPLE_DIR/.env.example" ]]; then
    cred_leak=false
    while IFS= read -r line; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ || -z "$line" ]] && continue
        # Check if value side has actual content (not empty, not a placeholder)
        key="${line%%=*}"
        value="${line#*=}"
        # Skip known safe defaults
        [[ "$key" == "SERVER_PORT" || "$key" == "DEFAULT_COUNTRY_CODE" ]] && continue
        [[ "$key" == *"MODEL"* || "$key" == *"VOICE"* ]] && continue
        # If value is non-empty and doesn't look like a placeholder
        if [[ -n "$value" && ! "$value" =~ ^your_ && ! "$value" =~ ^\{.*\}$ ]]; then
            # Allow simple defaults like "8000", "US", model names
            if [[ ${#value} -gt 30 ]]; then
                cred_leak=true
                break
            fi
        fi
    done < "$EXAMPLE_DIR/.env.example"

    if $cred_leak; then
        fail "Possible credentials in .env.example (values > 30 chars)"
    else
        pass "No credentials in .env.example"
    fi
fi

echo ""

# =============================================================================
# 6. Lint Check
# =============================================================================

echo "--- Lint ---"

if command -v uv &>/dev/null; then
    cd "$EXAMPLE_DIR"
    if uv run ruff check . 2>/dev/null; then
        pass "ruff lint clean"
    else
        fail "ruff lint errors found"
    fi
    cd "$REPO_ROOT"
else
    skip "uv not available — cannot run ruff"
fi

echo ""

# =============================================================================
# 7. Unit Tests
# =============================================================================

echo "--- Unit Tests ---"

if command -v uv &>/dev/null && [[ -f "$EXAMPLE_DIR/tests/test_integration.py" ]]; then
    cd "$EXAMPLE_DIR"
    if uv run python -m pytest tests/test_integration.py -v -k "unit" --tb=short 2>/dev/null; then
        pass "Unit tests pass"
    else
        fail "Unit tests failed"
    fi
    cd "$REPO_ROOT"
else
    skip "Cannot run unit tests (uv not available or test file missing)"
fi

echo ""

# =============================================================================
# 8. README Completeness
# =============================================================================

echo "--- README ---"

if [[ -f "$EXAMPLE_DIR/README.md" ]]; then
    readme_sections=0
    for section in "Features" "Prerequisites" "Quick Start" "Project Structure" "How It Works" "Configuration" "Testing"; do
        if grep -qi "## .*$section\|# .*$section" "$EXAMPLE_DIR/README.md" 2>/dev/null; then
            readme_sections=$((readme_sections + 1))
        fi
    done

    if [[ $readme_sections -ge 5 ]]; then
        pass "README.md has $readme_sections/7 key sections"
    else
        fail "README.md only has $readme_sections/7 key sections"
    fi
else
    fail "README.md not found"
fi

echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
TOTAL=$((PASS + FAIL + SKIP))
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped (total: $TOTAL)"
echo "=========================================="

if [[ $FAIL -gt 0 ]]; then
    echo "VALIDATION FAILED"
    exit 1
else
    echo "VALIDATION PASSED"
    exit 0
fi
