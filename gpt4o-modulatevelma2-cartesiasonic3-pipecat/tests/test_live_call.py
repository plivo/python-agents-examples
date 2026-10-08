"""
Live inbound call E2E test: a real caller phones the agent and holds a conversation.

Tests:
1. The tunnel reaches the inbound server
2. /answer through the tunnel: signed like Plivo signs it -> Stream XML; unsigned -> 403
3. /hold through the tunnel (the server's silent test route): signed -> <Wait>; unsigned -> 403
4. A full multi-turn inbound call between one caller and the agent:
     caller: dials in and listens
     agent : its opening line (LLM-generated), uninterrupted
     caller: "What is the capital city of France?"
     agent : answers (Paris)
     caller: "What is the price of one bitcoin in US dollars right now?"
     agent : calls search_the_web (Tavily), then answers
     caller: "Great, thank you very much, goodbye."
     agent : answers, then the caller hangs up

How the caller works. The test places a call from PLIVO_TEST_NUMBER to
PLIVO_PHONE_NUMBER with Plivo's Make Call API. PLIVO_PHONE_NUMBER is assigned to a test
application whose answer URL is the inbound server's /answer on the tunnel, so that leg
runs the agent (the number's own application is restored afterwards, also on failure).
The calling leg answers with SILENT_LEG_XML (tests/helpers.py): a background session
recording, which makes no sound and records that leg from its first instant, followed
by <Wait>. ngrok serves that XML itself from a Traffic Policy on the one tunnel
(static_xml_policy), so it works on a free ngrok plan and needs no route on the server.
The caller's lines are spoken into its own leg with Plivo's Speak API and reach the
agent as real telephone audio. The caller never talks over the agent: before each line
the test waits for the server log to show the agent finished its turn
(wait_for_bot_turn: Pipecat's "Bot started/stopped speaking", LLM, tool-call and TTS
lines, then a short settle time).

What is verified: the opening line is heard in the recording before any caller speech
and nothing interrupted it; Modulate transcribed each caller line; a spoken reply
followed each one (Paris for the first); search_the_web was called for the second and
the agent spoke after it; the agent was never cut off mid-speech (no echo barge-in);
Plivo's signed /answer was accepted, the stream started, the /hangup webhook arrived
and the pipeline ended. Each leg's transcript is printed labelled CALLER leg / AGENT
leg; a Plivo leg recording carries both directions, so both contain both voices (the
caller leg's is stereo: agent left, caller right; the agent leg's is mono, mixed).

Requirements:
    - PLIVO_AUTH_ID, PLIVO_AUTH_TOKEN, PLIVO_PHONE_NUMBER, PLIVO_TEST_NUMBER,
      OPENAI_API_KEY, MODULATE_API_KEY, CARTESIA_API_KEY and TAVILY_API_KEY in .env
      (PLIVO_TEST_NUMBER is a second Plivo number)
    - ngrok v3 binary (Traffic Policy support) on PATH (or NGROK_BIN), with no other
      ngrok agent running
    - faster-whisper installed (dev dependency), ffmpeg available
    - Port 18002 available

Usage:
    uv run pytest tests/test_live_call.py -v -s
"""

from __future__ import annotations

import os
import re

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
    barge_ins,
    check_spoken_turns,
    ensure_ffmpeg_on_path,
    final_transcripts,
    hangup_quietly,
    leg_transcripts,
    log_offset,
    log_tail,
    missing_env,
    number_on_app,
    place_call_and_wait,
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

TEST_PORT = 18002
LOG_PATH = server_log_path("modulate_live_call_server")
# Test-only Plivo application (not the one inbound.server auto-configures)
APP_NAME = "GPT4o_ModulateVelma2_CartesiaSonic3_Pipecat_Test"

SEARCH_TOOL = "search_the_web"
# Each line is one unbroken sentence, so the end-of-turn detector has no mid-line pause
# to mistake for the end of the caller's turn.
CALLER_TURNS = (
    # An ordinary question with one checkable answer
    SpokenTurn(
        "What is the capital city of France?",
        heard=("capital", "france"),
        answer=("paris",),
    ),
    # A current price: inbound/system_prompt.md tells the agent to search, not guess. The
    # answer changes by the minute, so only the tool call and a spoken reply are checked.
    SpokenTurn(
        "What is the price of one bitcoin in US dollars right now?",
        heard=("price", "bitcoin"),
        tool=SEARCH_TOOL,
    ),
    SpokenTurn("Great, thank you very much, goodbye.", heard=("thank",)),
)
# The first distinctive word of the caller's first line, as a transcript says it
FIRST_CALLER_WORD = "capital"

_MISSING = missing_env(*PLIVO_CALL_VARS, *LIVE_API_KEYS, "TAVILY_API_KEY")
pytestmark = pytest.mark.skipif(bool(_MISSING), reason=f"not configured: {', '.join(_MISSING)}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def tunnel_url():
    """Start ngrok first: the server needs PUBLIC_URL to verify signatures and build wss://."""
    proc, public_url = start_ngrok(TEST_PORT, static_xml_policy({SILENT_LEG_PATH: SILENT_LEG_XML}))
    print(f"\n[tunnel] URL: {public_url}")
    yield public_url
    stop_ngrok(proc)


@pytest.fixture(scope="module")
def server_process(tunnel_url):
    """Start the inbound server (SIGTERM -> wait(5) -> SIGKILL on teardown).

    PLIVO_PHONE_NUMBER is blanked so the server skips its own webhook auto-config;
    the plivo_configured fixture assigns (and later restores) the number instead.
    """
    proc = start_server(
        "inbound.server",
        TEST_PORT,
        LOG_PATH,
        {"PUBLIC_URL": tunnel_url, "PLIVO_PHONE_NUMBER": ""},
    )
    print(f"[server] logs: {LOG_PATH}")
    yield proc
    stop_server(proc)


@pytest.fixture(scope="module")
def plivo_configured(server_process, tunnel_url):
    """Point PLIVO_PHONE_NUMBER at a test app on the tunnel; always restore its own app."""
    client = plivo.RestClient(auth_id=PLIVO_AUTH_ID, auth_token=PLIVO_AUTH_TOKEN)
    phone_digits = normalize_phone_number(PLIVO_PHONE_NUMBER)
    with number_on_app(
        client, phone_digits, APP_NAME, f"{tunnel_url}/answer", f"{tunnel_url}/hangup"
    ):
        yield {"client": client, "public_url": tunnel_url}


def _words(text: str) -> set[str]:
    """The distinctive (4+ letter) words of ``text``, lower-cased."""
    return set(re.findall(r"[a-z]{4,}", text.lower()))


# =============================================================================
# Tests
# =============================================================================


class TestLiveCall:
    """End-to-end tests that place a real call through Plivo."""

    def test_tunnel_accessible(self, server_process, tunnel_url):
        resp = httpx.get(tunnel_url, timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_answer_webhook_via_ngrok(self, plivo_configured):
        """Signed like Plivo signs it: accepted through the tunnel; unsigned: 403."""
        public_url = plivo_configured["public_url"]
        form = {"CallUUID": "test-ngrok", "From": "+15551234567", "To": "+16572338892"}
        assert httpx.post(f"{public_url}/answer", data=form, timeout=10.0).status_code == 403
        resp = signed_webhook("POST", f"{public_url}/answer", PLIVO_AUTH_TOKEN, form)
        assert resp.status_code == 200
        assert "<Stream" in resp.text
        assert 'bidirectional="true"' in resp.text
        assert 'keepCallAlive="true"' in resp.text
        assert "audio/x-mulaw;rate=8000" in resp.text
        assert public_url.replace("https://", "wss://") + "/ws?body=" in resp.text
        assert stream_body(resp.text)["call_uuid"] == "test-ngrok"

    def test_hold_webhook_via_ngrok(self, server_process, tunnel_url):
        """The server's silent /hold route still answers signed requests only."""
        form = {"CallUUID": "test-hold", "From": "+15551234567", "To": "+16572338892"}
        assert httpx.post(f"{tunnel_url}/hold", data=form, timeout=10.0).status_code == 403
        resp = signed_webhook("POST", f"{tunnel_url}/hold", PLIVO_AUTH_TOKEN, form)
        assert resp.status_code == 200
        assert '<Wait length="120"' in resp.text
        assert "<Stream" not in resp.text

    def test_live_call_conversation(self, plivo_configured):
        """A caller dials in, hears the opening line, asks two questions and says goodbye."""
        client = plivo_configured["client"]
        caller_url = assert_silent_leg_served(plivo_configured["public_url"])
        call_start = log_offset(LOG_PATH)
        # caller_leg: the call we place (it answers with the silent, self-recording XML);
        # agent_leg: the inbound call it creates on PLIVO_PHONE_NUMBER, answered by /answer
        caller_leg, agent_leg = place_call_and_wait(
            client,
            from_digits=normalize_phone_number(PLIVO_TEST_NUMBER),
            to_digits=normalize_phone_number(PLIVO_PHONE_NUMBER),
            answer_url=caller_url,
        )
        try:
            assert caller_leg, f"Call did not go live within 30s\n{log_tail(LOG_PATH)}"
            assert agent_leg and agent_leg != caller_leg, "The agent's leg did not go live"
            # The caller's leg records itself from its answer XML; record the agent's too
            try:
                client.calls.start_recording(agent_leg, file_format="mp3")
            except Exception as e:
                print(f"[Call] recording failed on {agent_leg}: {e}")

            # 1. The caller stays silent until the agent has finished its opening line
            assert wait_for_bot_turn(LOG_PATH, call_start, timeout=40), (
                f"The agent never finished its opening line\n{log_tail(LOG_PATH)}"
            )
            # 2-4. The caller speaks, then waits for the agent to finish its answer
            offsets = speak_turns(client, caller_leg, CALLER_TURNS, LOG_PATH, who="Caller")
            # The caller hangs up; Plivo ends the agent's leg and closes its stream
            hangup_quietly(client, caller_leg)
            hangup_received = wait_for_log(
                LOG_PATH, f"Call ended: CallUUID={agent_leg}", timeout=20
            )
        finally:
            hangup_quietly(client, caller_leg, agent_leg)

        log = read_log_text(LOG_PATH)[call_start:]
        offsets = [offset - call_start for offset in offsets]
        opening_log = log[: offsets[0]]
        opening_line = " ".join(tts_texts(opening_log))
        print(f"[Agent] opening line: {tts_texts(opening_log)}")

        # Plivo's own signed /answer webhook was accepted and its stream started; the
        # caller's leg was answered by ngrok, never by the server
        assert f"Incoming call: CallUUID={agent_leg}" in log, log_tail(LOG_PATH)
        assert f"Plivo stream started: callId={agent_leg}" in log
        assert "Rejected Plivo" not in log, "A webhook or the stream of this call was rejected"
        assert caller_leg not in log, "The caller's leg reached the server"

        # The opening line was spoken and nothing interrupted it: no interruption (hence
        # no clearAudio) and nothing heard before the caller's first line
        assert opening_line.strip(), f"No opening line went to TTS\n{opening_log[-1500:]}"
        assert INTERRUPTION_LOG not in opening_log, (
            f"Interruption during the opening line\n{opening_log[-1500:]}"
        )
        assert final_transcripts(opening_log) == [], (
            f"The agent heard speech during its opening line: {final_transcripts(opening_log)}"
        )

        # Each caller line was transcribed by Modulate and answered in speech; turn 2
        # called the search tool first. The caller never talked over the agent and its
        # leg sends back no echo, so the agent must never be cut off mid-speech.
        check_spoken_turns(CALLER_TURNS, log, offsets)
        assert not barge_ins(log), f"The agent was cut off mid-speech: {barge_ins(log)}"
        for line in log.splitlines():
            if "[Tavily]" in line or "Tavily search" in line or "[Latency]" in line:
                print(f"[Log] {line.split(' - ', 1)[-1]}")

        # What was really heard on the line: both legs' recordings, transcribed locally.
        # Each recording carries both directions of its leg, so each has both voices.
        transcripts = leg_transcripts(client, {"CALLER leg": caller_leg, "AGENT leg": agent_leg})
        assert "CALLER leg" in transcripts, f"No recording of the caller's leg {caller_leg}"
        # The caller's leg is recorded from its first instant: the opening line comes
        # before the caller's first words, and most of its words are there
        caller_heard = transcripts["CALLER leg"].lower()
        assert FIRST_CALLER_WORD in caller_heard, (
            f"The caller is not in its own recording: {caller_heard}"
        )
        before_caller = _words(caller_heard[: caller_heard.index(FIRST_CALLER_WORD)])
        opening_words = _words(opening_line)
        heard_share = len(opening_words & before_caller) / max(len(opening_words), 1)
        print(f"[Result] opening-line words heard before the caller spoke: {heard_share:.0%}")
        assert heard_share >= 0.6, (
            f"Opening line '{opening_line}' not heard before the caller spoke: '{caller_heard}'"
        )
        heard_on_call = " ".join(transcripts.values()).lower()
        assert "paris" in heard_on_call, f"The recordings lack the answer 'Paris': {transcripts}"

        # Hangup webhook received, and the agent stopped its pipeline on the stream close
        assert hangup_received, f"/hangup webhook not received\n{log_tail(LOG_PATH)}"
        assert wait_for_log(LOG_PATH, f"Pipeline ended for call {agent_leg}", timeout=15), (
            f"The pipeline was still running after the hangup\n{log_tail(LOG_PATH)}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
