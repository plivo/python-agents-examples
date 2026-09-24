"""Shared test helpers for live call tests."""

from __future__ import annotations

import base64
import json
import os
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.etree import ElementTree

import httpx
import plivo
import pytest
from plivo.utils.signature_v3 import construct_get_url, construct_post_url, get_signature_v3

NGROK_BIN = os.getenv("NGROK_BIN", "ngrok")
NGROK_API = "http://127.0.0.1:4040/api/tunnels"

PROJECT_DIR = Path(__file__).resolve().parent.parent


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


def server_log_path(name: str) -> Path:
    """Where a test server subprocess writes its logs (TEST_LOG_DIR or the temp dir)."""
    log_dir = Path(os.getenv("TEST_LOG_DIR") or tempfile.gettempdir())
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{name}.log"


def start_server(
    module: str, port: int, log_path: Path, env_overrides: dict[str, str] | None = None
) -> subprocess.Popen:
    """Start ``python -m {module}`` on ``port`` with JSON logs written to ``log_path``.

    Output goes to a file, not a pipe, so a chatty server can never block on a full pipe.
    Skips the calling test if the health check doesn't come up within 15s.
    """
    env = os.environ.copy()
    env["SERVER_PORT"] = str(port)  # inbound.server
    env["OUTBOUND_SERVER_PORT"] = str(port)  # outbound.server
    env.setdefault("LOG_FORMAT", "json")
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
    pytest.skip(f"Server did not start in time. Output:\n{log_path.read_text()[-2000:]}")


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


def read_log_records(log_path: Path) -> list[dict]:
    """Parse the loguru JSON records (LOG_FORMAT=json) from a server log file."""
    records = []
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            records.append(json.loads(line)["record"])
        except (json.JSONDecodeError, KeyError):
            continue
    return records


def read_log_events(log_path: Path, event: str | None = None) -> list[dict]:
    """Return the structured ``extra`` dicts that carry an ``event`` key."""
    events = [r["extra"] for r in read_log_records(log_path) if "event" in r.get("extra", {})]
    if event is not None:
        events = [e for e in events if e["event"] == event]
    return events


def log_messages(log_path: Path) -> list[str]:
    """Return every log message text from a server log file."""
    return [r.get("message", "") for r in read_log_records(log_path)]


# =============================================================================
# Plivo webhook signing (V3) and /ws stream URLs — what Plivo itself sends
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


# =============================================================================
# Deepgram reusable agent configurations (REST) — live Path 2 tests only
# =============================================================================

DEEPGRAM_API_URL = "https://api.deepgram.com/v1"


def readme_create_script(direction: str) -> str:
    """The Python half of the README's create command for ``direction`` (one source of truth).

    It is the body of the ``uv run python - <<'EOF' | curl ...`` heredoc in "Creating a
    reusable config" and prints the create body ``{"config": "<json>", "metadata": {...}}``.
    """
    readme = (PROJECT_DIR / "README.md").read_text()
    marker = f"from {direction}.agent import DeepgramVoiceAgent"
    start = readme.rindex("import json\n", 0, readme.index(marker))
    return readme[start : readme.index("\nEOF\n", start) + 1]


def readme_create_body(direction: str, env: dict[str, str] | None = None) -> dict:
    """Run the README's create script and return the body it would POST."""
    result = subprocess.run(
        [sys.executable, "-"],
        input=readme_create_script(direction),
        capture_output=True,
        text=True,
        cwd=PROJECT_DIR,
        env=env,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def _deepgram_rest(method: str, path: str, body: dict | None = None):
    response = httpx.request(
        method,
        f"{DEEPGRAM_API_URL}{path}",
        json=body,
        headers={"Authorization": f"Token {os.environ['DEEPGRAM_API_KEY']}"},
        timeout=15,
    )
    if response.status_code == 403:
        pytest.skip(f"DEEPGRAM_API_KEY lacks agent:read/agent:write: {response.text}")
    response.raise_for_status()
    return response.json() if response.content else None


def deepgram_project_id() -> str:
    """The first project the key can see (as in the README)."""
    return _deepgram_rest("GET", "/projects")["projects"][0]["project_id"]


def create_agent_config(direction: str) -> str:
    """POST the README's create body for ``direction``; return the new config's UUID."""
    created = _deepgram_rest(
        "POST", f"/projects/{deepgram_project_id()}/agents", readme_create_body(direction)
    )
    return created["agent_uuid"]


def list_agent_configs() -> list[dict]:
    return _deepgram_rest("GET", f"/projects/{deepgram_project_id()}/agents")


def delete_agent_config(agent_uuid: str) -> None:
    _deepgram_rest("DELETE", f"/projects/{deepgram_project_id()}/agents/{agent_uuid}")


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


def upsert_application(
    client: plivo.RestClient, app_name: str, answer_url: str, hangup_url: str = ""
) -> str:
    """Create or update a Plivo application and return its app_id."""
    params = {"answer_url": answer_url, "answer_method": "POST"}
    if hangup_url:
        params.update(hangup_url=hangup_url, hangup_method="POST")
    # app_name filters by prefix server-side (the unfiltered list is paged at 20)
    apps = client.applications.list(app_name=app_name)
    for app_obj in apps["objects"]:
        if app_obj["app_name"] == app_name:
            client.applications.update(app_id=app_obj["app_id"], **params)
            return app_obj["app_id"]
    return client.applications.create(app_name=app_name, **params)["app_id"]


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


def start_ngrok(port: int) -> tuple[subprocess.Popen, str]:
    """Start our own ngrok tunnel to ``port`` and return (process, public_url).

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

    proc = subprocess.Popen(
        [NGROK_BIN, "http", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

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
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


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


def best_transcript(client: plivo.RestClient, call_uuids: list[str]) -> str:
    """Download each leg's recording, transcribe it and return the longest transcript."""
    transcript = ""
    for call_uuid in call_uuids:
        if not call_uuid:
            continue
        url = wait_for_recording(client, call_uuid, timeout=45)
        if not url:
            print(f"[Recording] none for {call_uuid}")
            continue
        audio = download_recording(url)
        print(f"[Recording] {call_uuid}: {len(audio)} bytes")
        if len(audio) < 1000:
            continue
        text = transcribe_audio(audio)
        print(f"[Transcript] ({call_uuid}): {text}")
        if len(text) > len(transcript):
            transcript = text
    return transcript
