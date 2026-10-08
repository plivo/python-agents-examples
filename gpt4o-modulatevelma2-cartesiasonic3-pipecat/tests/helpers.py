"""Shared test helpers for live call tests."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import secrets
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import httpx
import plivo
import pytest
from plivo.utils.signature_v3 import construct_get_url, construct_post_url, get_signature_v3

NGROK_BIN = os.getenv("NGROK_BIN", "ngrok")
NGROK_API = "http://127.0.0.1:4040/api/tunnels"
# Plivo rejects webhook URLs whose hostname it can't resolve yet ("Must be a valid url");
# a fresh trycloudflare.com hostname took ~70s to be accepted in testing.
PLIVO_URL_ACCEPT_TIMEOUT_S = 180.0
PLIVO_URL_RETRY_INTERVAL_S = 5.0

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Plivo auth token local test servers check webhook signatures with (never a real credential)
TEST_AUTH_TOKEN = "test-plivo-auth-token"
# Every key the pipeline needs to produce audio: LLM, STT and TTS
LIVE_API_KEYS = ("OPENAI_API_KEY", "MODULATE_API_KEY", "CARTESIA_API_KEY")
PLIVO_CALL_VARS = ("PLIVO_AUTH_ID", "PLIVO_AUTH_TOKEN", "PLIVO_PHONE_NUMBER", "PLIVO_TEST_NUMBER")


def missing_env(*names: str) -> list[str]:
    """The names among ``names`` that are unset or empty (for skip conditions)."""
    return [name for name in names if not os.getenv(name)]


def local_server_env(port: int) -> dict[str, str]:
    """Env for a server that only serves local tests: it can never touch a real Plivo number.

    No PLIVO_AUTH_ID / PLIVO_PHONE_NUMBER, so inbound auto-configuration is skipped whatever
    the developer's .env says; webhook auth is keyed with TEST_AUTH_TOKEN and PUBLIC_URL is
    the local URL, so tests sign requests the way Plivo does and <Stream> points at ws://localhost.
    """
    return {
        "PLIVO_AUTH_ID": "",
        "PLIVO_AUTH_TOKEN": TEST_AUTH_TOKEN,
        "PLIVO_PHONE_NUMBER": "",
        "PUBLIC_URL": f"http://localhost:{port}",
    }


# =============================================================================
# Test-only audio codec. The agent never converts audio itself (Pipecat's
# PlivoFrameSerializer does), so the G.711 codec lives here, not in utils.py.
# Pure Python: no numpy or scipy. Only synthesize_caller_speech() below uses
# gTTS, pydub and ffmpeg (all three come with the dev dependency group).
# =============================================================================

PLIVO_SAMPLE_RATE = 8000  # Plivo streams μ-law at 8kHz
_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def ulaw_to_pcm(ulaw_audio: bytes) -> bytes:
    """G.711 μ-law -> 16-bit little-endian PCM (recordings, RMS, transcripts)."""
    samples = []
    for byte in ulaw_audio:
        code = ~byte & 0xFF
        exponent = (code >> 4) & 0x07
        mantissa = code & 0x0F
        magnitude = (((mantissa << 3) + _ULAW_BIAS) << exponent) - _ULAW_BIAS
        samples.append(-magnitude if code & 0x80 else magnitude)
    return struct.pack(f"<{len(samples)}h", *samples)


def pcm_to_ulaw(pcm_audio: bytes) -> bytes:
    """16-bit little-endian PCM -> G.711 μ-law (a trailing odd byte is dropped)."""
    count = len(pcm_audio) // 2
    out = bytearray(count)
    for i, sample in enumerate(struct.unpack(f"<{count}h", pcm_audio[: count * 2])):
        sign = 0x80 if sample < 0 else 0x00
        magnitude = min(abs(sample), _ULAW_CLIP) + _ULAW_BIAS
        exponent = magnitude.bit_length() - 8  # 0..7: magnitude is 132..32767
        mantissa = (magnitude >> (exponent + 3)) & 0x0F
        out[i] = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return bytes(out)


def downsample_pcm16(pcm_audio: bytes, input_rate: int, output_rate: int) -> bytes:
    """Downsample PCM16 mono by a whole factor (24kHz -> 8kHz is 3).

    Each output sample is the mean of ``factor`` input samples: a box low-pass
    followed by decimation. Enough for speech fed to a VAD and an STT in a test;
    it is not a general-purpose resampler.
    """
    if input_rate == output_rate:
        return pcm_audio
    factor, remainder = divmod(input_rate, output_rate)
    if remainder or factor < 1:
        raise ValueError(f"{input_rate}Hz -> {output_rate}Hz is not a whole-factor downsample")
    count = len(pcm_audio) // 2
    samples = struct.unpack(f"<{count}h", pcm_audio[: count * 2])
    out = [sum(samples[i : i + factor]) // factor for i in range(0, count - factor + 1, factor)]
    return struct.pack(f"<{len(out)}h", *out)


def rms_of_ulaw(ulaw_audio: bytes) -> float:
    """RMS of μ-law audio on the PCM16 scale (silence is ~0, speech is in the thousands)."""
    pcm = ulaw_to_pcm(ulaw_audio)
    samples = struct.unpack(f"<{len(pcm) // 2}h", pcm)
    return (sum(s * s for s in samples) / max(len(samples), 1)) ** 0.5


def pcm_to_plivo(pcm_audio: bytes, sample_rate: int) -> bytes:
    """PCM16 mono at ``sample_rate`` -> what Plivo streams: μ-law 8kHz."""
    return pcm_to_ulaw(downsample_pcm16(pcm_audio, sample_rate, PLIVO_SAMPLE_RATE))


def ensure_ffmpeg_on_path() -> None:
    """Put a checked-in ffmpeg binary on PATH (faster-whisper needs it).

    Looks in FFMPEG_DIR, then in the example dir and each parent directory.
    """
    candidates = [Path(os.environ["FFMPEG_DIR"])] if os.getenv("FFMPEG_DIR") else []
    candidates += [PROJECT_DIR, *PROJECT_DIR.parents]
    for directory in candidates:
        if (directory / "ffmpeg").is_file():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            return


def ffmpeg_executable() -> str:
    """The ffmpeg binary pydub decodes MP3 with; no system install is needed.

    An ffmpeg already on PATH (or in FFMPEG_DIR / a parent directory) wins. Otherwise
    the binary inside the ``imageio-ffmpeg`` wheel (a dev dependency) is used: it ships
    in the wheel for macOS, Linux and Windows, so nothing is downloaded at test time.
    """
    ensure_ffmpeg_on_path()
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def synthesize_caller_speech(text: str) -> bytes:
    """``text`` spoken by gTTS, as μ-law 8kHz ready to send as Plivo media frames.

    gTTS (Google's public TTS endpoint: network, no API key) returns MP3; pydub decodes
    it with ffmpeg and resamples it to 8kHz mono PCM16, which is then μ-law encoded.
    The MP3 is decoded with ``from_file_using_temporary_files``, pydub's ffmpeg-only
    path: ``from_mp3`` also runs ffprobe, which the imageio-ffmpeg wheel does not ship.
    Errors are raised with the failing step named (a test that cannot get its caller
    speech fails; it does not skip).
    """
    import warnings

    from gtts import gTTS

    with warnings.catch_warnings():
        # pydub 0.25.1 has invalid-escape regexes (SyntaxWarning on Python 3.12+) and
        # warns at import when ffmpeg is not on PATH; the converter is set explicitly below
        warnings.simplefilter("ignore", SyntaxWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        from pydub import AudioSegment

    with tempfile.TemporaryDirectory() as tmp_dir:
        mp3_path = os.path.join(tmp_dir, "caller.mp3")
        try:
            gTTS(text=text, lang="en").save(mp3_path)
        except Exception as e:
            raise RuntimeError(
                f"gTTS could not synthesise {text!r} (it needs network access to "
                f"translate.google.com): {type(e).__name__}: {e}"
            ) from e
        AudioSegment.converter = ffmpeg_executable()
        try:
            audio = AudioSegment.from_file_using_temporary_files(mp3_path, format="mp3")
        except Exception as e:
            raise RuntimeError(
                f"pydub could not decode the gTTS MP3 with ffmpeg at "
                f"{AudioSegment.converter}: {type(e).__name__}: {e}"
            ) from e
    audio = audio.set_frame_rate(PLIVO_SAMPLE_RATE).set_channels(1).set_sample_width(2)
    return pcm_to_plivo(audio.raw_data, audio.frame_rate)


def server_log_path(name: str) -> Path:
    """Where a test server subprocess writes its logs (TEST_LOG_DIR or the temp dir)."""
    log_dir = Path(os.getenv("TEST_LOG_DIR") or tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}.log"


def start_server(
    module: str,
    port: int,
    log_path: Path,
    env_overrides: dict[str, str] | None = None,
    *,
    required: bool = False,
) -> subprocess.Popen:
    """Start ``python -m {module}`` on ``port`` with its output written to ``log_path``.

    Output goes to a file, not a pipe, so a chatty server can never block on a full pipe.
    If the health check doesn't come up within 15s the calling test is skipped, or
    failed when ``required`` is true (for tests that may only skip on missing keys).
    """
    env = os.environ.copy()
    env["SERVER_PORT"] = str(port)  # inbound.server
    env["OUTBOUND_SERVER_PORT"] = str(port)  # outbound.server
    for key, value in (env_overrides or {}).items():
        env[key] = value

    log_file = open(log_path, "w")  # noqa: SIM115 — closed by stop_server
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        cwd=str(PROJECT_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    proc.log_file = log_file  # type: ignore[attr-defined]

    for _ in range(30):
        try:
            if httpx.get(f"http://localhost:{port}/", timeout=1.0).status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            break
        time.sleep(0.5)

    stop_server(proc)
    message = f"Server did not start in time. Output:\n{log_path.read_text()[-2000:]}"
    if required:
        pytest.fail(message)
    pytest.skip(message)


def stop_server(proc: subprocess.Popen) -> None:
    """Stop a server subprocess: SIGTERM, wait 5s, then SIGKILL."""
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log_file = getattr(proc, "log_file", None)
    if log_file is not None:
        log_file.close()


def read_log_text(log_path: Path) -> str:
    """Everything a test server subprocess has logged so far (loguru text + uvicorn)."""
    return log_path.read_text(errors="replace")


def wait_for_log(log_path: Path, text: str, timeout: float = 15.0) -> bool:
    """Wait until ``text`` appears in the server log."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if text in read_log_text(log_path):
            return True
        time.sleep(0.5)
    return text in read_log_text(log_path)


def log_tail(log_path: Path, chars: int = 2000) -> str:
    """The end of the server log, for assertion messages."""
    return read_log_text(log_path)[-chars:]


# =============================================================================
# What the pipeline did, read from the server's DEBUG log (Pipecat's own lines)
# =============================================================================

# Pipecat broadcasts an interruption at the start of every user turn; the serializer
# turns it into clearAudio. It is a barge-in only when the bot was speaking at the time.
INTERRUPTION_LOG = "broadcasting interruption"
_BOT_STARTED = re.compile(r"- Bot started speaking$", re.MULTILINE)
_BOT_STOPPED = re.compile(r"- Bot stopped speaking$", re.MULTILINE)
_TTS_TEXT = re.compile(r"Generating TTS \[(.*)\]$", re.MULTILINE)
_FINAL_TRANSCRIPT = re.compile(r"(?<!INTERIM )TRANSCRIPTION: (['\"])(.*)\1 from ")


def log_offset(log_path: Path) -> int:
    """A position in the server log; pass it back to read only what is logged after it."""
    return len(read_log_text(log_path))


def tts_texts(log_text: str) -> list[str]:
    """Every sentence handed to TTS (``Generating TTS [...]``), in order."""
    return _TTS_TEXT.findall(log_text)


def final_transcripts(log_text: str) -> list[str]:
    """Every final STT transcript (TranscriptionLogObserver's ``TRANSCRIPTION:`` lines)."""
    return [match[1] for match in _FINAL_TRANSCRIPT.findall(log_text)]


def barge_ins(log_text: str) -> list[str]:
    """The interruption log lines that cut the bot off while it was speaking."""
    speaking = False
    cut_offs = []
    for line in log_text.splitlines():
        if _BOT_STARTED.search(line):
            speaking = True
        elif _BOT_STOPPED.search(line):
            speaking = False
        elif INTERRUPTION_LOG in line and speaking:
            cut_offs.append(line)
    return cut_offs


def tool_call_offsets(log_text: str, name: str = "") -> list[int]:
    """Where in ``log_text`` the LLM service started a tool call (``name``, or any tool).

    Pipecat's LLM service logs ``Calling function [<name>:<tool_call_id>] with arguments``
    when it runs a registered function.
    """
    return [m.start() for m in re.finditer(rf"Calling function \[{name or '[^:]+'}:", log_text)]


def _spoke_after_last_tool_call(log_text: str) -> bool:
    calls = tool_call_offsets(log_text)
    return not calls or bool(_BOT_STARTED.search(log_text, calls[-1]))


def _bot_activity(log_text: str) -> tuple[int, ...]:
    return (
        len(_BOT_STARTED.findall(log_text)),
        len(_BOT_STOPPED.findall(log_text)),
        log_text.count("LLM START RESPONSE"),
        log_text.count("LLM END RESPONSE"),
        len(tool_call_offsets(log_text)),
        len(_TTS_TEXT.findall(log_text)),
    )


def wait_for_bot_turn(
    log_path: Path, offset: int, timeout: float = 45.0, settle_secs: float = 2.0
) -> bool:
    """Wait until the agent has spoken a whole turn after ``offset`` and gone quiet.

    A turn is over when, in the log after ``offset``, the bot has started speaking at
    least once (and again after its last tool call, so a search in progress is waited
    for), every start has its "Bot stopped speaking", every LLM response has ended, and
    none of those counters (nor the number of sentences sent to TTS) has moved for
    ``settle_secs``. The settle time covers the gap between two sentences
    of one answer and the audio still in flight to the phone, so whoever speaks next
    does not talk over the agent. Returns False on timeout.
    """
    deadline = time.time() + timeout
    last_activity: tuple[int, ...] | None = None
    changed_at = time.time()
    while time.time() < deadline:
        log_text = read_log_text(log_path)[offset:]
        activity = _bot_activity(log_text)
        if activity != last_activity:
            last_activity, changed_at = activity, time.time()
        started, stopped, llm_starts, llm_ends, _, _ = activity
        quiet = stopped >= started and llm_ends >= llm_starts
        answered = started and _spoke_after_last_tool_call(log_text)
        if answered and quiet and time.time() - changed_at >= settle_secs:
            return True
        time.sleep(0.2)
    return False


# =============================================================================
# A scripted human on the far leg of a live call (test_live_call, test_outbound_call)
# =============================================================================

# Answer URL path of the human's leg. ngrok answers it itself (static_xml_policy); the
# request never reaches the example's servers.
SILENT_LEG_PATH = "/test-silent-leg/answer"
# Someone who picks up and listens. The background session recording answers the call
# without a sound (a <Wait>-only answer never picks up an inbound call) and records the
# leg from its first instant; the two <Wait>s keep the line open and silent for up to 4
# minutes. The human's lines are spoken into this leg with Plivo's Speak API meanwhile.
SILENT_LEG_XML = (
    '<?xml version="1.0" encoding="UTF-8"?><Response>'
    '<Record recordSession="true" redirect="false" maxLength="600"/>'
    '<Wait length="120"/><Wait length="120"/>'
    "</Response>"
)


@dataclass(frozen=True)
class SpokenTurn:
    """One line the human says, and how to recognise it and the agent's answer."""

    text: str  # spoken into the human's leg with Plivo's Speak API
    heard: tuple[str, ...]  # words the STT's final transcripts of the line must contain
    answer: tuple[str, ...] = ()  # words the agent's reply must contain (any phrasing)
    tool: str = ""  # tool the agent must call before it replies


def missing_words(words: tuple[str, ...] | list[str], text: str) -> list[str]:
    """The ``words`` that do not occur in ``text`` (case-insensitive)."""
    return [w for w in words if w.lower() not in text.lower()]


def speak_turns(
    client: plivo.RestClient, leg_uuid: str, turns: tuple[SpokenTurn, ...], log_path: Path, who: str
) -> list[int]:
    """Say each turn on ``leg_uuid`` and wait for the agent to finish answering it.

    The agent hears the lines as real telephone audio (Plivo Speak API, ``legs="aleg"``:
    towards the other party of that leg). Call this once the agent is quiet; each line
    is spoken only after ``wait_for_bot_turn`` says the previous answer is over, so the
    human never talks over the agent. Returns the log offset at which each line was
    spoken, plus the offset after the last answer: turn ``i`` is logged in
    ``log[offsets[i]:offsets[i + 1]]``.
    """
    offsets = []
    for turn in turns:
        offsets.append(log_offset(log_path))
        print(f"[{who}] says: '{turn.text}'")
        client.calls.speak(leg_uuid, text=turn.text, language="en-US", legs="aleg")
        assert wait_for_bot_turn(log_path, offsets[-1], timeout=60), (
            f"The agent never finished answering '{turn.text}'\n{log_tail(log_path)}"
        )
    offsets.append(log_offset(log_path))
    return offsets


def check_spoken_turns(turns: tuple[SpokenTurn, ...], log_text: str, offsets: list[int]) -> None:
    """Assert, from the server log, that every turn was heard and answered in speech.

    For each turn: the STT's final transcripts contain ``heard``; a non-empty reply went
    to TTS and contains ``answer``; ``tool`` (if any) was called and the agent spoke
    after the call; and nothing cut the agent off while it was speaking (``barge_ins``).
    Prints what was heard and said, turn by turn.
    """
    assert len(offsets) == len(turns) + 1, "offsets must be speak_turns()'s return value"
    for turn, start, stop in zip(turns, offsets, offsets[1:], strict=False):
        turn_log = log_text[start:stop]
        heard = " ".join(final_transcripts(turn_log))
        reply = " ".join(tts_texts(turn_log))
        print(f"[Turn] said     : '{turn.text}'")
        print(f"       STT heard: {final_transcripts(turn_log)}")
        print(f"       tool     : {len(tool_call_offsets(turn_log))} call(s)")
        print(f"       reply    : {tts_texts(turn_log)}")
        assert not missing_words(turn.heard, heard), (
            f"The STT transcribed '{turn.text}' as '{heard}': lacks "
            f"{missing_words(turn.heard, heard)}"
        )
        assert reply.strip(), f"No spoken reply to '{turn.text}'\n{turn_log[-1500:]}"
        assert not missing_words(turn.answer, reply), (
            f"The reply to '{turn.text}' lacks {missing_words(turn.answer, reply)}: '{reply}'"
        )
        if turn.tool:
            calls = tool_call_offsets(turn_log, turn.tool)
            assert calls, f"The agent answered '{turn.text}' without calling {turn.tool}: '{reply}'"
            after_tool = " ".join(tts_texts(turn_log[calls[-1] :]))
            assert after_tool.strip(), f"No spoken reply after the {turn.tool} call"
            assert _BOT_STARTED.search(turn_log, calls[-1]), (
                f"The agent never spoke after the {turn.tool} call"
            )
        assert not barge_ins(turn_log), (
            f"The agent was cut off after '{turn.text}': {barge_ins(turn_log)}"
        )


@contextlib.contextmanager
def number_on_app(
    client: plivo.RestClient, number_digits: str, app_name: str, answer_url: str, hangup_url: str
) -> Iterator[str]:
    """Assign ``number_digits`` to a test application; always put its own application back.

    Yields the test application's id. The restore is in a ``finally``: it runs whether
    the tests passed or failed, and also when the reassignment itself raised.
    ``hangup_url`` is always set: Plivo otherwise keeps the application's previous one.
    """
    tail = number_digits[-4:]
    original_app_id = get_app_id_for_number(client, number_digits)
    app_id = ""
    try:
        app_id = upsert_application(client, app_name, answer_url, hangup_url=hangup_url)
        client.numbers.update(number=number_digits, app_id=app_id)
        print(f"\n[Plivo] Number ...{tail} assigned to test app {app_name} ({app_id})")
        yield app_id
    finally:
        if not original_app_id or original_app_id == app_id:
            print(
                f"\n[Plivo] Number ...{tail} had no other application before the test "
                f"(found: {original_app_id or 'none'}); nothing to restore"
            )
        else:
            client.numbers.update(number=number_digits, app_id=original_app_id)
            print(f"\n[Plivo] Restored number ...{tail} to its original app {original_app_id}")


def assert_silent_leg_served(tunnel_url: str) -> str:
    """The silent leg's answer URL, after checking that ngrok really serves the XML."""
    url = f"{tunnel_url}{SILENT_LEG_PATH}"
    served = httpx.post(url, timeout=10.0)
    assert served.status_code == 200 and served.text == SILENT_LEG_XML, (
        f"ngrok does not serve the silent-leg XML at {SILENT_LEG_PATH}: "
        f"HTTP {served.status_code} (Traffic Policy needs ngrok v3)"
    )
    return url


# =============================================================================
# Plivo webhook and /ws stream signing (V3), /ws stream URLs — what Plivo itself sends
# =============================================================================


def plivo_signature_headers(
    method: str,
    url: str,
    auth_token: str,
    params: dict | None = None,
    nonce: str | None = None,
) -> dict[str, str]:
    """``X-Plivo-Signature-V3`` headers for a webhook request, signed the way Plivo signs.

    ``url`` is the full URL Plivo was configured to call, query string included. For POST,
    ``params`` are the form fields; for GET, Plivo's params are part of ``url``. The base
    string comes from the Plivo SDK (the same code the server validates with).
    """
    nonce = nonce or str(secrets.randbelow(10**20))
    if method.upper() == "GET":
        base_url = construct_get_url(url, dict(params or {}))
    else:
        base_url = construct_post_url(url, dict(params or {}))
    signature = get_signature_v3(auth_token.encode(), base_url.decode(), nonce.encode())
    return {"X-Plivo-Signature-V3": signature.decode(), "X-Plivo-Signature-V3-Nonce": nonce}


def signed_webhook(
    method: str, url: str, auth_token: str, data: dict | None = None, timeout: float = 10.0
) -> httpx.Response:
    """Send a Plivo-signed webhook request (form ``data`` for POST) to ``url``."""
    headers = plivo_signature_headers(method, url, auth_token, data)
    return httpx.request(method, url, data=data, headers=headers, timeout=timeout)


def stream_signature_headers(
    stream_url: str, auth_token: str, nonce: str | None = None
) -> dict[str, str]:
    """``X-Plivo-Signature-V3`` headers Plivo sends when it connects to ``stream_url``.

    Plivo signs the stream URL as ``http://`` + host + path, without the query string.
    """
    parts = urlsplit(stream_url)
    return plivo_signature_headers(
        "GET", f"http://{parts.netloc}{parts.path}", auth_token, None, nonce
    )


def stream_url_from_xml(xml: str) -> str:
    """The wss:// URL of the <Stream> element in an answer webhook response."""
    stream = ElementTree.fromstring(xml).find("Stream")
    assert stream is not None and stream.text, f"No <Stream> URL in: {xml}"
    return stream.text.strip()


def stream_query(xml: str) -> dict[str, str]:
    """Decoded query params (``body``) of the <Stream> URL."""
    return {k: v[0] for k, v in parse_qs(urlsplit(stream_url_from_xml(xml)).query).items()}


def stream_body(xml: str) -> dict:
    """The call metadata JSON carried in the <Stream> URL's ``body``."""
    return json.loads(base64.b64decode(stream_query(xml)["body"]))


def list_live_call_ids(client: plivo.RestClient) -> list[str]:
    """Return the UUIDs of all live calls on the account."""
    try:
        live_calls = client.live_calls.list_ids()
    except Exception:
        return []
    if hasattr(live_calls, "calls"):
        return list(live_calls.calls or [])
    if isinstance(live_calls, dict):
        return list(live_calls.get("calls", []))
    return []


def get_app_id_for_number(client: plivo.RestClient, number_digits: str) -> str:
    """Return the Plivo application ID currently assigned to a number ("" if none)."""
    try:
        info = client.numbers.get(number=number_digits)
    except Exception:
        return ""
    app = (
        info.get("application", "") if isinstance(info, dict) else getattr(info, "application", "")
    )
    if app and "/Application/" in str(app):
        return str(app).split("/Application/")[1].rstrip("/")
    return ""


def is_invalid_url_error(e: Exception) -> bool:
    """Plivo's API rejects answer/hangup URLs whose hostname it can't resolve yet."""
    return isinstance(e, plivo.exceptions.ValidationError) and "valid url" in str(e).lower()


def until_plivo_accepts_url(action, what: str, timeout_s: float = PLIVO_URL_ACCEPT_TIMEOUT_S):
    """Call ``action()`` until Plivo stops answering "Must be a valid url" (new tunnel host)."""
    deadline = time.time() + timeout_s
    while True:
        try:
            return action()
        except Exception as e:
            if not is_invalid_url_error(e) or time.time() >= deadline:
                raise
            print(f"[Plivo] {what}: hostname not accepted yet, retrying...")
            time.sleep(PLIVO_URL_RETRY_INTERVAL_S)


def upsert_application(
    client: plivo.RestClient, app_name: str, answer_url: str, hangup_url: str = ""
) -> str:
    """Create or update a Plivo application and return its app_id.

    Retries while Plivo rejects a tunnel hostname it can't resolve yet.
    """
    params = {"answer_url": answer_url, "answer_method": "POST"}
    if hangup_url:
        params.update(hangup_url=hangup_url, hangup_method="POST")
    # app_name filters by prefix server-side (the unfiltered list is paged at 20)
    apps = client.applications.list(app_name=app_name)
    existing = [a["app_id"] for a in apps["objects"] if a["app_name"] == app_name]
    if existing:
        app_id = existing[0]
        until_plivo_accepts_url(
            lambda: client.applications.update(app_id=app_id, **params),
            f"update application {app_name}",
        )
        return app_id
    return until_plivo_accepts_url(
        lambda: client.applications.create(app_name=app_name, **params),
        f"create application {app_name}",
    )["app_id"]


def _running_ngrok_pids() -> list[str]:
    """PIDs of ngrok processes on this machine (read-only ``pgrep``; for messages only)."""
    try:
        out = subprocess.run(["pgrep", "-f", "ngrok"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    return out.stdout.split()


def _ngrok_agent_running() -> bool:
    """True if an ngrok agent already answers on its local API (``NGROK_API``)."""
    try:
        httpx.get(NGROK_API, timeout=1.0)
    except httpx.HTTPError:
        return False
    return True


def static_xml_policy(responses: dict[str, str]) -> dict:
    """An ngrok Traffic Policy that answers ``{path: xml}`` at ngrok's edge.

    Requests for any other path go on to the tunnelled server untouched. This lets a
    test give Plivo an answer URL for the far leg of a call without adding a route to
    the example's servers and without a second tunnel: ngrok's free plan allows one
    agent, and every tunnel that agent opens gets the same (single) domain, so a second
    local server could not have a URL of its own. The XML is static and public for as
    long as the tunnel is up; it must not contain anything secret.
    """
    return {
        "on_http_request": [
            {
                "expressions": [f"req.url.path == '{path}'"],
                "actions": [
                    {
                        "type": "custom-response",
                        "config": {
                            "status_code": 200,
                            "headers": {"content-type": "application/xml"},
                            "body": xml,
                        },
                    }
                ],
            }
            for path, xml in responses.items()
        ]
    }


def start_ngrok(port: int, traffic_policy: dict | None = None) -> tuple[subprocess.Popen, str]:
    """Start our own ngrok tunnel to ``port`` and return (process, public_url).

    ``traffic_policy`` (see ``static_xml_policy``) is written to a temp file and passed
    as ``--traffic-policy-file``; ``stop_ngrok`` deletes the file.

    Never kills other ngrok processes: an agent already running on this machine belongs
    to another session or to a manual tunnel, and killing it would break that session's
    calls. ngrok's free plan allows only one agent session at a time, so instead of
    starting a second agent this skips the test when one is already up (its local API
    answers on ``NGROK_API``). Only the process started here is ever stopped, through its
    own ``Popen`` handle (``stop_ngrok``), on the failure path and at teardown.
    """
    if _ngrok_agent_running():
        pids = ", ".join(_running_ngrok_pids()) or "unknown"
        pytest.skip(
            f"ngrok is already running (pid {pids}), probably another session or a manual "
            "tunnel; stop it and re-run"
        )

    command = [NGROK_BIN, "http", str(port)]
    policy_path = ""
    if traffic_policy:
        # JSON is valid YAML, which is what ngrok reads; no YAML writer is needed
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as policy_file:
            json.dump(traffic_policy, policy_file)
            policy_path = policy_file.name
        command += ["--traffic-policy-file", policy_path]

    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    proc.policy_path = policy_path  # type: ignore[attr-defined]

    public_url = None
    for _ in range(30):
        if proc.poll() is not None:  # our agent exited (bad auth, session limit, ...)
            break
        try:
            resp = httpx.get(NGROK_API, timeout=1.0)
            if resp.status_code == 200:
                for t in resp.json().get("tunnels", []):
                    addr = t.get("config", {}).get("addr", "")
                    if t.get("proto") == "https" and str(port) in addr:
                        public_url = t["public_url"]
                        break
                if public_url:
                    break
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(0.5)

    if not public_url:
        stop_ngrok(proc)
        pytest.skip("ngrok did not start or no HTTPS tunnel found")

    return proc, public_url


def stop_ngrok(proc: subprocess.Popen) -> None:
    """Stop the ngrok process started by ``start_ngrok`` (only that one, by its handle)."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    policy_path = getattr(proc, "policy_path", "")
    if policy_path:
        Path(policy_path).unlink(missing_ok=True)


def wait_for_recording(
    client: plivo.RestClient, call_uuid: str, timeout: float = 30.0
) -> str | None:
    """Poll for a recording to appear for the given call UUID. Returns URL or None."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            recordings = client.recordings.list(call_uuid=call_uuid)
            objects = recordings.get("objects", []) if isinstance(recordings, dict) else []
            # Handle ListResponseObject
            if not objects and hasattr(recordings, "objects"):
                objects = recordings.objects or []
            if objects:
                rec = objects[0]
                if isinstance(rec, dict):
                    url = rec.get("recording_url", "")
                else:
                    url = getattr(rec, "recording_url", "")
                if url:
                    return url
        except Exception:
            pass
        time.sleep(3)
    return None


def download_recording(url: str) -> bytes:
    """Download recording audio from URL."""
    resp = httpx.get(url, follow_redirects=True, timeout=30.0)
    resp.raise_for_status()
    return resp.content


def transcribe_audio(audio_data: bytes) -> str:
    """Transcribe MP3 audio using faster-whisper."""
    from faster_whisper import WhisperModel

    model = WhisperModel("base", device="cpu", compute_type="int8")

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name

    try:
        segments, _ = model.transcribe(tmp_path, language="en")
        return " ".join(seg.text.strip() for seg in segments).strip()
    finally:
        os.unlink(tmp_path)


def _field(obj, name: str) -> str:
    value = obj.get(name, "") if isinstance(obj, dict) else getattr(obj, name, "")
    return str(value or "")


def place_call_and_wait(
    client: plivo.RestClient,
    from_digits: str,
    to_digits: str,
    answer_url: str,
    timeout: float = 30.0,
) -> tuple[str, str]:
    """Place a Plivo call between two Plivo numbers and wait until both legs are live.

    Returns ``(outbound_uuid, inbound_uuid)``: the API call we placed (whose far end we
    can ``speak`` into) and the inbound call that the destination number's app answers.
    Only calls that were not live before this call are considered. ``inbound_uuid`` may
    be "" if Plivo does not report the leg direction.
    """
    baseline = set(list_live_call_ids(client))
    response = client.calls.create(
        from_=from_digits, to_=to_digits, answer_url=answer_url, answer_method="POST"
    )
    print(f"[Call] request_uuid: {_field(response, 'request_uuid')}")

    outbound_uuid, inbound_uuid = "", ""
    deadline = time.time() + timeout
    while time.time() < deadline and not (outbound_uuid and inbound_uuid):
        for call_uuid in set(list_live_call_ids(client)) - baseline:
            try:
                direction = _field(client.live_calls.get(call_uuid), "direction")
            except Exception:
                direction = ""
            if direction == "outbound":
                outbound_uuid = call_uuid
            elif direction == "inbound":
                inbound_uuid = call_uuid
            elif not outbound_uuid:
                outbound_uuid = call_uuid
        time.sleep(0.5)
    if not outbound_uuid and inbound_uuid:
        outbound_uuid = inbound_uuid
    print(f"[Call] live legs: outbound={outbound_uuid} inbound={inbound_uuid}")
    return outbound_uuid, inbound_uuid


def hangup_quietly(client: plivo.RestClient, *call_uuids: str) -> None:
    """Hang up calls, ignoring ones that already ended."""
    for call_uuid in call_uuids:
        if not call_uuid:
            continue
        try:
            client.calls.delete(call_uuid)
        except Exception as e:
            print(f"[Call] hangup {call_uuid}: {e}")


def leg_transcripts(client: plivo.RestClient, legs: dict[str, str]) -> dict[str, str]:
    """Download and transcribe each leg's recording: ``{label: transcript}``.

    ``legs`` is ``{label: call_uuid}``, e.g. ``{"CALLER leg": ..., "AGENT leg": ...}``; the
    label is printed with the transcript. A Plivo leg recording carries both directions
    of that leg, so each transcript normally has both speakers: one started with the
    record API is mono (both voices mixed); the session recording of SILENT_LEG_XML is
    stereo, with the far party (the agent) on the left and that leg's own Speak audio
    on the right.
    Legs with no recording (or an empty one) are left out.
    """
    transcripts: dict[str, str] = {}
    for label, call_uuid in legs.items():
        if not call_uuid:
            continue
        url = wait_for_recording(client, call_uuid, timeout=45)
        if not url:
            print(f"[Recording] {label}: none for {call_uuid}")
            continue
        audio = download_recording(url)
        print(f"[Recording] {label} ({call_uuid}): {len(audio)} bytes")
        if len(audio) < 1000:
            continue
        transcripts[label] = transcribe_audio(audio)
        print(f"[Transcript] {label}: {transcripts[label]}")
    return transcripts


def best_transcript(client: plivo.RestClient, call_uuids: list[str]) -> str:
    """Download each leg's recording, transcribe it and return the longest transcript."""
    legs = {call_uuid: call_uuid for call_uuid in call_uuids}
    return max(leg_transcripts(client, legs).values(), key=len, default="")
