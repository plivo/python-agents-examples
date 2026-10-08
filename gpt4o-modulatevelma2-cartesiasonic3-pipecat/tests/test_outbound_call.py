"""
Outbound call E2E tests: place the call with Plivo's Make Call API, as a user would.

Tests:
1. /outbound/answer (reached through the tunnel) returns Stream XML whose body carries
   the answer_url greeting; unsigned requests get 403
2. The server has no dial endpoint
3. Full outbound call cycle, as a natural conversation between the agent and one callee:
   plivo.RestClient().calls.create(answer_url=<tunnel>/outbound/answer?greeting=...,
   hangup_url=<tunnel>/outbound/hangup), then
     agent : the answer_url greeting, uninterrupted (the callee is silent)
     callee: "Yes, now is a good time, what is this call about?"
     agent : answers
     callee: "... what is the capital city of France?"
     agent : answers (Paris)
     callee: "... thank you very much, goodbye."
     agent : answers, then the callee hangs up

How the callee works. The agent calls from PLIVO_PHONE_NUMBER to PLIVO_TEST_NUMBER. A
call between two Plivo numbers creates a second, inbound call on PLIVO_TEST_NUMBER,
answered by that number's Plivo application. For the test the number is assigned to an
application whose answer URL returns SILENT_LEG_XML (tests/helpers.py): a background
session recording (it answers the call, makes no sound and records from the first
instant) followed by
<Wait>, so the line stays open and silent. That XML is served by ngrok itself, from a
Traffic Policy on the one tunnel (tests/helpers.py: static_xml_policy), so the example's
servers get no extra route and no second tunnel is needed. The callee's lines are spoken
into its leg with Plivo's Speak API and reach the agent as real telephone audio. The
callee never talks over the agent: before each line the test waits for the server log
to show that the agent finished its turn (Pipecat's "Bot started/stopped speaking",
LLM and TTS lines, then a short settle time). The number's original application is
restored afterwards, also when the test fails.

What is verified: the greeting's own words in the recordings' transcripts, no barge-in
(interruption -> clearAudio) during the greeting or anywhere else, each callee line
transcribed by Modulate, a spoken agent reply to each line (Paris for the question), the
signed answer webhook, the stream start, the hangup webhook and the pipeline ending.

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER,
      OPENAI_API_KEY, MODULATE_API_KEY and CARTESIA_API_KEY in .env
    - ngrok v3 binary (Traffic Policy support) on PATH (or NGROK_BIN), with no other
      ngrok agent running
    - faster-whisper installed (dev dependency), ffmpeg available
    - Port 18003 available

Usage:
    uv run pytest tests/test_outbound_call.py -v -s
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote, urlencode

import httpx
import plivo
import pytest
from dotenv import load_dotenv

from tests.helpers import (
    INTERRUPTION_LOG,
    LIVE_API_KEYS,
    PLIVO_CALL_VARS,
    SILENT_LEG_PATH,
    SILENT_LEG_XML,
    SpokenTurn,
    assert_silent_leg_served,
    check_spoken_turns,
    ensure_ffmpeg_on_path,
    final_transcripts,
    hangup_quietly,
    leg_transcripts,
    list_live_call_ids,
    log_offset,
    log_tail,
    missing_env,
    missing_words,
    number_on_app,
    read_log_text,
    server_log_path,
    signed_webhook,
    speak_turns,
    start_ngrok,
    start_server,
    static_xml_policy,
    stop_ngrok,
    stop_server,
    stream_body,
    tts_texts,
    wait_for_bot_turn,
    wait_for_log,
)
from utils import normalize_phone_number

load_dotenv()
ensure_ffmpeg_on_path()

PLIVO_AUTH_ID = os.getenv("PLIVO_AUTH_ID", "")
PLIVO_AUTH_TOKEN = os.getenv("PLIVO_AUTH_TOKEN", "")
PLIVO_PHONE_NUMBER = os.getenv("PLIVO_PHONE_NUMBER", "")
PLIVO_TEST_NUMBER = os.getenv("PLIVO_TEST_NUMBER", "")

TEST_PORT = 18003
LOG_PATH = server_log_path("modulate_outbound_server")
# Test-only Plivo application that answers the callee leg
BLEG_APP_NAME = "GPT4o_ModulateVelma2_CartesiaSonic3_Pipecat_Outbound_Test"
GREETING = (
    "Hello, this is a courtesy call from the voice agent demo about your appointment. "
    "Is now a good time for a quick chat?"
)
# Words of GREETING that the default greeting does not contain
GREETING_ONLY_WORDS = ["courtesy", "demo", "appointment", "quick chat"]


# Each line is one unbroken sentence, so the end-of-turn detector has no mid-line pause
# to mistake for the end of the callee's turn.
CALLEE_TURNS = (
    SpokenTurn(
        "Yes, now is a good time, what is this call about?",
        heard=("good time", "call"),
    ),
    # Checkable whatever the wording, and answerable with or without the web-search tool
    SpokenTurn(
        "Okay, and one quick question, what is the capital city of France?",
        heard=("capital", "france"),
        answer=("paris",),
    ),
    SpokenTurn("Great, thank you very much, goodbye.", heard=("thank",)),
)

_MISSING = missing_env(*PLIVO_CALL_VARS, *LIVE_API_KEYS)
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"not configured: {', '.join(_MISSING)}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tunnel_url():
    """Start the tunnel before the server so PUBLIC_URL can be passed to it."""
    proc, public_url = start_ngrok(TEST_PORT, static_xml_policy({SILENT_LEG_PATH: SILENT_LEG_XML}))
    print(f"\n[tunnel] URL: {public_url}")
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(tunnel_url):
    """Start the outbound server (SIGTERM -> wait(5) -> SIGKILL on teardown).

    The outbound server never configures a Plivo number; PLIVO_PHONE_NUMBER is blanked
    anyway so nothing in this process depends on it.
    """
    proc = start_server(
        "outbound.server",
        TEST_PORT,
        LOG_PATH,
        {"PUBLIC_URL": tunnel_url, "PLIVO_PHONE_NUMBER": ""},
    )
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_client():
    return plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)


@pytest.fixture(scope="module")
def bleg_app_id(plivo_client, tunnel_url):
    """Point PLIVO_TEST_NUMBER at the silent callee XML; always restore its application."""
    callee_url = assert_silent_leg_served(tunnel_url)
    test_digits = normalize_phone_number(PLIVO_TEST_NUMBER)
    with number_on_app(plivo_client, test_digits, BLEG_APP_NAME, callee_url, callee_url) as app_id:
        yield app_id


@pytest.fixture(autouse=True)
def hang_up_leftover_calls(plivo_client):
    """Hang up any call a test left live (e.g. one still ringing when it finished)."""
    baseline = set(list_live_call_ids(plivo_client))
    yield
    leftovers = set(list_live_call_ids(plivo_client)) - baseline
    if leftovers:
        print(f"\n[Cleanup] hanging up leftover calls: {leftovers}")
        hangup_quietly(plivo_client, *leftovers)


# =============================================================================
# Helpers
# =============================================================================


def _direction(client: plivo.RestClient, call_uuid: str) -> str:
    try:
        live = client.live_calls.get(call_uuid)
    except Exception:
        return ""
    value = live.get("direction", "") if isinstance(live, dict) else getattr(live, "direction", "")
    return str(value or "")


def _wait_for_legs(
    client: plivo.RestClient, baseline: set[str], timeout: float = 40.0
) -> tuple[str, str]:
    """(A-leg, B-leg): the outbound call we placed and the inbound call it created."""
    a_leg, b_leg = "", ""
    deadline = time.time() + timeout
    while time.time() < deadline and not (a_leg and b_leg):
        for call_uuid in set(list_live_call_ids(client)) - baseline:
            direction = _direction(client, call_uuid)
            if direction == "outbound":
                a_leg = call_uuid
            elif direction == "inbound":
                b_leg = call_uuid
        time.sleep(0.5)
    print(f"[Outbound] live legs: A={a_leg} B={b_leg}")
    return a_leg, b_leg


# =============================================================================
# Tests
# =============================================================================


class TestOutboundCall:
    """End-to-end tests for outbound calling via Plivo's Make Call API."""

    def test_outbound_answer_webhook(self, server_process, tunnel_url):
        """/outbound/answer returns valid Plivo Stream XML carrying the greeting."""
        query = urlencode(
            {
                "CallUUID": "test-uuid-456",
                "From": PLIVO_PHONE_NUMBER,
                "To": PLIVO_TEST_NUMBER,
                "greeting": GREETING,
            },
            quote_via=quote,
        )
        url = f"{tunnel_url}/outbound/answer?{query}"
        assert httpx.get(url, timeout=10.0).status_code == 403  # unsigned
        resp = signed_webhook("GET", url, PLIVO_AUTH_TOKEN)
        assert resp.status_code == 200
        body = resp.text
        assert "<Stream" in body
        assert 'bidirectional="true"' in body
        assert 'keepCallAlive="true"' in body
        assert "audio/x-mulaw;rate=8000" in body
        assert tunnel_url.replace("https://", "wss://") + "/ws?body=" in body
        meta = stream_body(body)
        assert meta["greeting"] == GREETING
        assert meta["call_uuid"] == "test-uuid-456"

        hint = f"answer_url={tunnel_url}/outbound/answer?greeting="
        assert hint in read_log_text(LOG_PATH), "startup log lacks the Make Call hint"

    def test_no_dial_endpoint(self, server_process, tunnel_url):
        """The server has no dial endpoint: calls are placed with the Plivo API directly."""
        assert httpx.post(f"{tunnel_url}/outbound/call", timeout=10.0).status_code == 404

    def test_outbound_call_full_cycle(self, server_process, tunnel_url, plivo_client, bleg_app_id):
        """Make Call API -> greeting heard in full -> three callee turns, each answered."""
        call_start = log_offset(LOG_PATH)
        baseline = set(list_live_call_ids(plivo_client))
        answer_url = f"{tunnel_url}/outbound/answer?" + urlencode(
            {"greeting": GREETING}, quote_via=quote
        )
        response = plivo_client.calls.create(
            from_=normalize_phone_number(PLIVO_PHONE_NUMBER),
            to_=normalize_phone_number(PLIVO_TEST_NUMBER),
            answer_url=answer_url,
            answer_method="POST",
            hangup_url=f"{tunnel_url}/outbound/hangup",
            hangup_method="POST",
        )
        request_uuid = (
            response.get("request_uuid", "")
            if isinstance(response, dict)
            else getattr(response, "request_uuid", "")
        )
        print(f"[Outbound] Make Call request_uuid={request_uuid}")
        assert request_uuid

        a_leg, b_leg = _wait_for_legs(plivo_client, baseline)
        ended = f"Outbound call ended: CallUUID={a_leg}"
        try:
            assert a_leg, f"The outbound call did not connect\n{log_tail(LOG_PATH)}"
            assert b_leg, "The callee leg (inbound call on PLIVO_TEST_NUMBER) did not go live"
            # The callee leg records itself from its answer XML; record the agent's leg too
            try:
                plivo_client.calls.start_recording(a_leg, file_format="mp3")
            except Exception as e:
                print(f"[Outbound] Recording failed on {a_leg}: {e}")

            # 1. The callee stays silent until the agent has finished its greeting
            assert wait_for_bot_turn(LOG_PATH, call_start, timeout=40), (
                f"The agent never finished its greeting\n{log_tail(LOG_PATH)}"
            )
            # 2-4. The callee speaks, then waits for the agent to finish its answer
            offsets = speak_turns(plivo_client, b_leg, CALLEE_TURNS, LOG_PATH, who="Callee")
            # The callee hangs up; Plivo ends the agent's leg and closes its stream
            hangup_quietly(plivo_client, b_leg)
            hangup_received = wait_for_log(LOG_PATH, ended, timeout=20)
        finally:
            hangup_quietly(plivo_client, a_leg, b_leg)

        log = read_log_text(LOG_PATH)
        greeting_log = log[call_start : offsets[0]]
        print(f"[Agent] greeting: {tts_texts(greeting_log)}")

        # The answer webhook was signed and carried the greeting; the stream started
        assert f"Outbound call answered: CallUUID={a_leg}" in log
        a_line = next(line for line in log.splitlines() if f"answered: CallUUID={a_leg}" in line)
        assert "greeting: from answer_url" in a_line
        assert f"Plivo stream started: callId={a_leg}" in log
        # One agent only: the callee leg never reached the outbound server
        assert b_leg not in log, "The callee leg was answered by the agent"

        # Nothing interrupted the greeting (the original problem: two agents greeting at
        # once): no interruption, hence no clearAudio, before the callee's first line.
        # The greeting went to TTS word for word.
        assert INTERRUPTION_LOG not in greeting_log, (
            f"Interruption during the greeting\n{greeting_log[-1500:]}"
        )
        assert final_transcripts(greeting_log) == [], (
            f"The agent heard speech during its greeting: {final_transcripts(greeting_log)}"
        )
        assert tts_texts(greeting_log) == [GREETING], tts_texts(greeting_log)

        # Each callee line was transcribed by Modulate and answered in speech; the callee
        # never talked over the agent, so no turn has a barge-in either
        check_spoken_turns(CALLEE_TURNS, log, offsets)

        # What was really heard on the line: both legs' recordings, transcribed locally
        transcripts = leg_transcripts(plivo_client, {"AGENT leg": a_leg, "CALLEE leg": b_leg})
        assert transcripts, f"No recording of {a_leg} or {b_leg} could be transcribed"
        heard_on_call = " ".join(transcripts.values()).lower().replace("-", " ")
        assert not missing_words(GREETING_ONLY_WORDS, heard_on_call), (
            f"The recordings lack {missing_words(GREETING_ONLY_WORDS, heard_on_call)} of the "
            f"answer_url greeting: {transcripts}"
        )
        for turn in CALLEE_TURNS:
            assert not missing_words(turn.answer, heard_on_call), (
                f"The recordings lack the answer {turn.answer} to '{turn.text}': {transcripts}"
            )

        # Hangup webhook received, and the agent stopped its pipeline on the stream close
        assert hangup_received, f"hangup_url webhook not received\n{log_tail(LOG_PATH)}"
        assert wait_for_log(LOG_PATH, f"Pipeline ended for outbound call {a_leg}", timeout=15), (
            f"The pipeline was still running after the hangup\n{log_tail(LOG_PATH)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
