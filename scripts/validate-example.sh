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
    echo "Example: $0 grok3-voice-native"
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

# Orchestration tokens from the naming convention. Anything other than "native" is a
# framework: the framework (or hosted platform behind it) owns the audio transport and VAD.
KNOWN_ORCH="native|pipecat|livekit|vapi"
KNOWN_VARIANTS="no-vad|webrtcvad"

ORCHESTRATION="native"
if [[ "$EXAMPLE" =~ -($KNOWN_ORCH)(-($KNOWN_VARIANTS))?$ ]]; then
    if [[ "${BASH_REMATCH[1]}" != "native" ]]; then
        ORCHESTRATION="framework"
    fi
elif grep -q "pipecat\|livekit" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
    ORCHESTRATION="framework"
fi
# Managed voice-agent platforms declare themselves in pyproject.toml:
#   [tool.voice-agent-example]
#   category = "managed-platform"
# (names follow the platform's own product branding, so they can't be
# detected from the directory name alone)
if grep -Eq '^category *= *"managed-platform"' "$EXAMPLE_DIR/pyproject.toml" 2>/dev/null; then
    ORCHESTRATION="managed-platform"
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
name_valid=false
if [[ "$ORCHESTRATION" == "managed-platform" ]]; then
    # {provider}-{product}[-{variant}]: lowercase, hyphen-separated tokens.
    # The product token must mirror the platform's own branding and any variant
    # must be agreed in review (see CLAUDE.md) -- only the format is checked here.
    if [[ "$EXAMPLE" =~ ^[a-z0-9.]+(-[a-z0-9.]+)+$ ]]; then
        name_valid=true
    fi
# Check if name ends with a known orchestration type (with optional known variant)
elif [[ "$EXAMPLE" =~ -($KNOWN_ORCH)$ ]]; then
    name_valid=true
elif [[ "$EXAMPLE" =~ -($KNOWN_ORCH)-($KNOWN_VARIANTS)$ ]]; then
    name_valid=true
fi

if $name_valid; then
    pass "Directory name follows naming convention"
    if [[ "$ORCHESTRATION" == "managed-platform" ]]; then
        echo "  [NOTE] Managed-platform name: confirm product branding and any variant in human review"
    fi
elif [[ "$ORCHESTRATION" == "managed-platform" ]]; then
    fail "Directory name '$EXAMPLE' must be lowercase hyphen-separated {provider}-{product}[-{variant}]"
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
# 2b. utils.py Helper Checks
# =============================================================================

echo "--- utils.py Helpers ---"

# Which helpers utils.py must define depends on who converts audio on the call path
# (CLAUDE.md "utils.py Requirements"). Everything is read from the syntax tree, so a
# comment, docstring or string that merely names a helper does not count.
#
# utils_probe prints one "<key> <value>" line per fact:
#   defs         top-level functions and classes defined in utils.py (comma-separated, or "-")
#   to_api       the plivo_to_{api} direction wrappers in utils.py (plivo_to_vad excluded)
#   from_api     the {api}_to_plivo direction wrappers in utils.py
#   passthrough  yes  both directions have a wrapper and every wrapper's body is just
#                     "return <its first argument>" (the API takes and emits μ-law 8kHz)
#   converts     yes  non-test code references ulaw_to_pcm / pcm_to_ulaw / resample_audio
#                     or imports a conversion library (audioop, scipy, soxr, ...)
#   inline       conversion done outside utils.py in non-test code, as file:line(reason)
#                entries: an audioop import, a call to a library resampler, or a function
#                whose name says it converts μ-law/PCM or resamples
#   probe        ok   last line, so truncated output is never mistaken for a result
#
# It runs under the example's own interpreter (uv run python): a system python3 can be
# older than the example's requires-python and unable to parse its syntax. A probe that
# cannot run is reported as a FAIL below, never as a pass or a skip.
PROBE_RUNNER="python3"
if command -v uv &>/dev/null; then
    PROBE_RUNNER="uv run python"
fi

utils_probe() {
    (
        cd "$EXAMPLE_DIR"
        $PROBE_RUNNER - "$EXAMPLE_DIR" <<'PY' 2>/dev/null
import ast
import os
import re
import sys

root = sys.argv[1]
CODEC = ("ulaw_to_pcm", "pcm_to_ulaw", "resample_audio")
CONVERSION_LIBS = ("audioop", "scipy", "soxr", "samplerate", "librosa", "resampy")
RESAMPLE_LIBS = CONVERSION_LIBS[1:]
INLINE_DEF = re.compile(r"(u|mu|a)law.*(pcm|lin)|(pcm|lin).*(u|mu|a)law|resampl", re.IGNORECASE)
TEST_FILE = re.compile(r"^(test_.*|.*_test|conftest)\.py$")


def parse(path):
    with open(path, encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename=path)


def dotted_root(node):
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def is_passthrough(func):
    body = list(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    args = func.args.posonlyargs + func.args.args
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Name)
        and bool(args)
        and body[0].value.id == args[0].arg
    )


defs = []
to_api = []
from_api = []
passthrough = True
for stmt in parse(os.path.join(root, "utils.py")).body:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        defs.append(stmt.name)
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if stmt.name.startswith("plivo_to_") and stmt.name != "plivo_to_vad":
            to_api.append(stmt.name)
            passthrough = passthrough and is_passthrough(stmt)
        elif stmt.name.endswith("_to_plivo"):
            from_api.append(stmt.name)
            passthrough = passthrough and is_passthrough(stmt)

converts = False
inline = set()
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(
        d for d in dirnames if d not in ("tests", "__pycache__") and not d.startswith(".")
    )
    for filename in sorted(filenames):
        if not filename.endswith(".py") or TEST_FILE.match(filename):
            continue
        path = os.path.join(dirpath, filename)
        rel = os.path.relpath(path, root)
        in_utils = rel == "utils.py"
        nodes = list(ast.walk(parse(path)))
        lib_names = {}
        for node in nodes:
            line = getattr(node, "lineno", 0)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    lib = alias.name.split(".")[0]
                    if lib in CONVERSION_LIBS:
                        lib_names[alias.asname or lib] = lib
                        converts = True
                        if lib == "audioop" and not in_utils:
                            inline.add(f"{rel}:{line}(imports-audioop)")
            elif isinstance(node, ast.ImportFrom):
                lib = (node.module or "").split(".")[0]
                if node.level == 0 and lib in CONVERSION_LIBS:
                    for alias in node.names:
                        lib_names[alias.asname or alias.name] = lib
                    converts = True
                    if lib == "audioop" and not in_utils:
                        inline.add(f"{rel}:{line}(imports-audioop)")
                if any(alias.name in CODEC for alias in node.names):
                    converts = True
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not in_utils and INLINE_DEF.search(node.name):
                    inline.add(f"{rel}:{line}(defines-{node.name})")
            elif isinstance(node, ast.Name) and node.id in CODEC:
                converts = True
            elif isinstance(node, ast.Attribute) and node.attr in CODEC:
                converts = True
        if in_utils:
            continue
        for node in nodes:
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            lib = lib_names.get(dotted_root(func))
            if name.startswith("resample") and lib in RESAMPLE_LIBS:
                inline.add(f"{rel}:{node.lineno}(calls-{lib}-{name})")


def joined(items):
    return ",".join(items) if items else "-"


print("defs", joined(defs))
print("to_api", joined(to_api))
print("from_api", joined(from_api))
print("passthrough", "yes" if to_api and from_api and passthrough else "no")
print("converts", "yes" if converts else "no")
print("inline", joined(sorted(inline)))
print("probe", "ok")
PY
    )
}

# probe_field <key> prints that key's value from the probe output ("" if absent)
probe_field() {
    awk -v key="$1" '$1 == key { print $2 }' <<< "$PROBE_OUT"
}

# utils_defines <name> succeeds if utils.py defines that top-level function or class
utils_defines() {
    [[ ",$UTILS_DEFS," == *",$1,"* ]]
}

PROBE_OUT=""
UTILS_DEFS=""
if [[ ! -f "$EXAMPLE_DIR/utils.py" ]]; then
    skip "utils.py helpers (utils.py not found, reported under Config Placement)"
elif ! PROBE_OUT="$(utils_probe)" || [[ "$(probe_field probe)" != "ok" ]]; then
    PROBE_OUT=""
    fail "utils.py helpers not checked: '$PROBE_RUNNER' failed to run the probe or to parse the example's Python files"
else
    UTILS_DEFS="$(probe_field defs)"
    to_api="$(probe_field to_api)"
    from_api="$(probe_field from_api)"

    if utils_defines "normalize_phone_number"; then
        pass "normalize_phone_number() defined in utils.py"
    else
        fail "normalize_phone_number() not defined in utils.py"
    fi

    if [[ "$ORCHESTRATION" == "framework" ]]; then
        # The framework's serializer/transport, or the hosted platform, converts the audio.
        # Helpers this example's own code needs are still caught by the inline check below.
        skip "Direction wrappers plivo_to_{api} / {api}_to_plivo (framework or hosted platform converts audio)"
        skip "Codec set ulaw_to_pcm / pcm_to_ulaw / resample_audio (framework or hosted platform converts audio)"
    else
        # Native and managed-platform: this example's code bridges Plivo audio to the API.
        if [[ "$to_api" != "-" && "$from_api" != "-" ]]; then
            pass "Direction wrappers defined in utils.py ($to_api, $from_api)"
        else
            fail "utils.py must define plivo_to_{api}() and {api}_to_plivo() (found: $to_api / $from_api)"
        fi

        missing_codec=()
        for fn in "ulaw_to_pcm" "pcm_to_ulaw" "resample_audio"; do
            if ! utils_defines "$fn"; then
                missing_codec+=("$fn")
            fi
        done
        if [[ ${#missing_codec[@]} -eq 0 ]]; then
            pass "Codec set defined in utils.py (ulaw_to_pcm, pcm_to_ulaw, resample_audio)"
        elif [[ ${#missing_codec[@]} -eq 3 && "$(probe_field passthrough)" == "yes" \
            && "$(probe_field converts)" == "no" ]] && ! utils_defines "plivo_to_vad"; then
            # μ-law 8kHz end to end. Cannot be claimed by accident: both wrappers must
            # return their argument unchanged, no codec function may be defined or
            # referenced, and no non-test code may import a conversion library.
            skip "Codec set (μ-law 8kHz end to end: $to_api and $from_api are pass-throughs, nothing converts audio)"
        else
            fail "utils.py missing ${missing_codec[*]} (required unless both direction wrappers are pass-throughs and no code converts audio)"
        fi
    fi

    inline_conversion="$(probe_field inline)"
    if [[ "$inline_conversion" == "-" ]]; then
        pass "No audio conversion outside utils.py"
    else
        fail "Audio conversion outside utils.py (move it into utils.py): ${inline_conversion//,/, }"
    fi
fi

echo ""

# =============================================================================
# 3. Audio Pipeline Checks
# =============================================================================

echo "--- Audio Pipeline ---"

# PLIVO_CHUNK_SIZE = 160 (consumed by _send_to_plivo; a framework's transport does its own chunking)
if [[ "$ORCHESTRATION" == "framework" ]]; then
    skip "PLIVO_CHUNK_SIZE = 160 (framework transport chunks the audio)"
elif grep -rq "PLIVO_CHUNK_SIZE.*=.*160\|PLIVO_CHUNK_SIZE = 160" "$EXAMPLE_DIR/inbound/agent.py" "$EXAMPLE_DIR/outbound/agent.py" 2>/dev/null; then
    pass "PLIVO_CHUNK_SIZE = 160 found in agent.py"
else
    fail "PLIVO_CHUNK_SIZE = 160 not found in agent.py"
fi

# playAudio format and VAD configuration are read from the syntax tree, so a comment or
# docstring that merely mentions them does not count, and nothing here names a
# framework's classes (Pipecat, LiveKit, Vapi and others spell their transports differently).
#
# agent_probe <file> prints "<playaudio> <vad>":
#   playaudio  ok     the agent builds a playAudio dict with contentType "audio/x-mulaw"
#                     and sampleRate 8000
#              wrong  it builds one with a different (or unverifiable) format
#              none   it builds none (the framework's transport emits the message)
#   vad        the name of the VAD / turn-detection setting found, or "none"
agent_probe() {
    python3 - "$1" <<'PY' 2>/dev/null || echo "none none"
import ast
import re
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())

# Module-level NAME = <constant>, so "contentType": MULAW_TYPE style code resolves.
consts = {}
for stmt in tree.body:
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Constant):
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                consts[target.id] = stmt.value.value


def value_of(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    return None


VAD_KEYWORD = re.compile(r"^(vad|vad_analyzer|turn_detection|turn_detector|turn_analyzer)$")
VAD_DICT_KEY = re.compile(r"vad|speakingplan|turn_?detection|endpointing", re.IGNORECASE)

playaudio = "none"
vad = "none"
for node in ast.walk(tree):
    if isinstance(node, ast.Dict):
        items = {k.value: v for k, v in zip(node.keys, node.values) if isinstance(k, ast.Constant)}
        if "contentType" in items and ("sampleRate" in items or "payload" in items):
            good = (
                value_of(items["contentType"]) == "audio/x-mulaw"
                and value_of(items.get("sampleRate")) == 8000
            )
            if good:
                playaudio = "ok"
            elif playaudio != "ok":
                playaudio = "wrong"
        for key in items:
            if isinstance(key, str) and VAD_DICT_KEY.search(key) and vad == "none":
                vad = key
    elif isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg is None:
                continue
            is_none = isinstance(kw.value, ast.Constant) and kw.value.value is None
            if VAD_KEYWORD.match(kw.arg) and not is_none:
                vad = kw.arg
            elif kw.arg == "vad_enabled" and value_of(kw.value) is True:
                vad = "vad_enabled"
print(playaudio, vad)
PY
}

for side in inbound outbound; do
    agent_file="$EXAMPLE_DIR/$side/agent.py"
    [[ -f "$agent_file" ]] || continue
    read -r playaudio_kind _vad_kind <<< "$(agent_probe "$agent_file")"
    if [[ "$playaudio_kind" == "ok" ]]; then
        pass "$side/agent.py builds playAudio as audio/x-mulaw at 8000 Hz"
    elif [[ "$playaudio_kind" == "wrong" ]]; then
        fail "$side/agent.py builds a playAudio message that is not contentType audio/x-mulaw with sampleRate 8000"
    elif [[ "$ORCHESTRATION" == "framework" ]]; then
        skip "$side/agent.py playAudio format (emitted by the framework's transport)"
    else
        fail "$side/agent.py does not build a playAudio message with contentType audio/x-mulaw and sampleRate 8000"
    fi
done

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

if [[ "$ORCHESTRATION" == "managed-platform" ]]; then
    skip "SileroVADProcessor (managed platform — turn detection is platform-side)"
    skip "plivo_to_vad (managed platform)"
    skip "VAD-driven turn management (managed platform)"
    skip "silero-vad dependency (managed platform)"
    # The platform detects barge-in, but the client must still flush Plivo playback
    if grep -q "clearAudio" "$EXAMPLE_DIR/inbound/agent.py" 2>/dev/null; then
        pass "Barge-in handling (clearAudio on platform interruption event)"
    else
        fail "clearAudio not sent in inbound/agent.py (platform barge-in must flush Plivo playback)"
    fi
elif [[ "$EXAMPLE" == *"-no-vad"* ]]; then
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

    # plivo_to_vad and SileroVADProcessor defined in utils (from the utils.py probe above)
    if utils_defines "plivo_to_vad" && utils_defines "SileroVADProcessor"; then
        pass "plivo_to_vad() and SileroVADProcessor defined in utils.py"
    else
        fail "plivo_to_vad() and SileroVADProcessor must both be defined in utils.py"
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
    # Framework: VAD or turn detection must be configured in code. Frameworks spell it
    # differently (a vad=/vad_analyzer=/turn_detection= argument, or a key in a hosted
    # platform's assistant config), so the probe matches the concept, not a class name.
    read -r _playaudio_kind vad_kind <<< "$(agent_probe "$EXAMPLE_DIR/inbound/agent.py")"
    if [[ "$vad_kind" != "none" ]]; then
        pass "VAD / turn detection configured in framework config ($vad_kind)"
    else
        fail "no VAD or turn detection configured in inbound/agent.py"
    fi
    skip "SileroVADProcessor (framework uses built-in VAD)"
    skip "plivo_to_vad (framework uses built-in VAD)"
fi

echo ""

# =============================================================================
# 4b. Webhook Authentication (CLAUDE.md "Webhook Authentication")
# =============================================================================
# A server that verifies Plivo signatures (it references validate_v3_signature) must
# declare a dependency on every route except the health check "/", /ws included.
# Servers without the check are older examples and are skipped.

echo "--- Webhook Authentication ---"

for direction in inbound outbound; do
    server_file="$EXAMPLE_DIR/$direction/server.py"
    [[ -f "$server_file" ]] || continue
    auth_result=$(python3 - "$server_file" <<'PY' 2>/dev/null || echo "error"
import ast
import sys

source = open(sys.argv[1]).read()
if "validate_v3_signature" not in source:
    print("none")
    raise SystemExit
ROUTES = {"get", "post", "put", "delete", "patch", "api_route", "websocket"}
unsigned = []
for node in ast.walk(ast.parse(source)):
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        continue
    for dec in node.decorator_list:
        if not (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)):
            continue
        if dec.func.attr not in ROUTES or not dec.args:
            continue
        path = dec.args[0].value if isinstance(dec.args[0], ast.Constant) else "?"
        if path == "/":
            continue
        if not any(kw.arg == "dependencies" for kw in dec.keywords):
            unsigned.append(f"{dec.func.attr.upper()} {path}")
print("unsigned: " + ", ".join(unsigned) if unsigned else "ok")
PY
)
    case "$auth_result" in
        ok) pass "$direction/server.py: every Plivo route (webhooks and /ws) checks the signature" ;;
        none) skip "$direction/server.py: no Plivo signature check (older example)" ;;
        error) fail "$direction/server.py: could not be parsed for the signature check" ;;
        *) fail "$direction/server.py: routes without the signature dependency ($auth_result)" ;;
    esac
done

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
