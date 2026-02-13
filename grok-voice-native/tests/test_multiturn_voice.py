"""
Test multi-turn voice conversation with TTS-generated speech.

This is a standalone test script that demonstrates the voice agent
handling multiple conversation turns with real synthesized speech.

Requirements:
    - Server running: uv run python -m inbound.server
    - ffmpeg/ffprobe in PATH (for audio conversion)
    - gTTS and pydub installed (included in dev dependencies)

Usage:
    # Add ffmpeg to PATH if needed
    PATH="/path/to/ffmpeg:$PATH" uv run python tests/test_multiturn_voice.py
"""

import asyncio
import base64
import json
import os
import tempfile
import time
import uuid

import websockets

from utils import pcm_to_ulaw


def generate_tts_audio(text: str) -> bytes | None:
    """Generate speech audio using Google TTS and convert to μ-law."""
    try:
        from gtts import gTTS
        from pydub import AudioSegment
    except ImportError:
        print("  gTTS or pydub not available")
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        mp3_path = f.name

    try:
        tts = gTTS(text=text, lang="en")
        tts.save(mp3_path)

        audio = AudioSegment.from_mp3(mp3_path)
        audio = audio.set_frame_rate(8000).set_channels(1).set_sample_width(2)

        return pcm_to_ulaw(audio.raw_data)
    except Exception as e:
        print(f"  TTS error: {e}")
        return None
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


async def send_audio_and_wait(ws, audio_bytes: bytes, timeout: float = 15.0) -> dict:
    """Send audio and wait for response."""
    result = {"sent_chunks": 0, "recv_chunks": 0, "recv_bytes": 0, "ttfr": None}

    # Send in 20ms chunks (160 bytes at 8kHz μ-law)
    chunk_size = 160
    chunks = [audio_bytes[i : i + chunk_size] for i in range(0, len(audio_bytes), chunk_size)]

    start_time = time.time()

    for chunk in chunks:
        payload = base64.b64encode(chunk).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": payload}}))
        result["sent_chunks"] += 1
        await asyncio.sleep(0.02)

    # Wait for response
    last_received = time.time()

    while time.time() - start_time < timeout:
        silence = base64.b64encode(b"\xff" * 160).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))

        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                data = json.loads(message)
                if data.get("event") == "playAudio":
                    result["recv_chunks"] += 1
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        result["recv_bytes"] += len(base64.b64decode(payload))
                    last_received = time.time()
                    if result["ttfr"] is None:
                        result["ttfr"] = time.time() - start_time
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed:
            break

        if result["recv_chunks"] > 0 and (time.time() - last_received) > 3:
            break

        await asyncio.sleep(0.02)

    return result


async def wait_for_greeting(ws, timeout: float = 10.0) -> dict:
    """Wait for agent greeting."""
    result = {"recv_chunks": 0, "recv_bytes": 0}
    start_time = time.time()
    last_received = start_time

    while time.time() - start_time < timeout:
        silence = base64.b64encode(b"\xff" * 160).decode()
        await ws.send(json.dumps({"event": "media", "media": {"payload": silence}}))

        try:
            while True:
                message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                data = json.loads(message)
                if data.get("event") == "playAudio":
                    result["recv_chunks"] += 1
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        result["recv_bytes"] += len(base64.b64decode(payload))
                    last_received = time.time()
        except asyncio.TimeoutError:
            pass
        except websockets.exceptions.ConnectionClosed:
            break

        if result["recv_chunks"] > 0 and (time.time() - last_received) > 2:
            break

        await asyncio.sleep(0.02)

    return result


async def test_multiturn_voice():
    """Test a complete multi-turn voice conversation."""
    ws_url = "ws://localhost:8000/ws?body=eyJjYWxsX3V1aWQiOiAidGVzdCIsICJmcm9tIjogIisxNTU1MTIzNDU2NyIsICJ0byI6ICIrMTY1NzIzMzg4OTIifQ=="

    print("=" * 70)
    print("MULTI-TURN VOICE CONVERSATION TEST")
    print("=" * 70)

    turns = [
        "What are your business hours?",
        "Do you have any specials today?",
        "Thanks, goodbye!",
    ]

    print("\nGenerating TTS audio...")
    turn_audio = []
    for i, text in enumerate(turns):
        print(f'  Turn {i + 1}: "{text}"')
        audio = generate_tts_audio(text)
        if audio is None:
            print("  ERROR: Could not generate TTS audio")
            print("  Make sure ffmpeg is in PATH and gTTS/pydub are installed")
            return
        turn_audio.append((text, audio))
        print(f"    Generated {len(audio)} bytes ({len(audio) / 8000:.1f}s)")

    print("\nConnecting to WebSocket...")

    try:
        async with websockets.connect(ws_url, close_timeout=5) as ws:
            print("Connected\n")

            await ws.send(
                json.dumps(
                    {
                        "event": "start",
                        "start": {"callId": str(uuid.uuid4()), "streamId": str(uuid.uuid4())},
                    }
                )
            )

            results = []

            # Wait for greeting
            print("Turn 0: Agent Greeting")
            print("-" * 50)
            greeting = await wait_for_greeting(ws)
            results.append(("Greeting", 0, greeting["recv_bytes"], None))
            print(f"  Received {greeting['recv_chunks']} chunks ({greeting['recv_bytes']} bytes)\n")

            # Execute conversation turns
            for i, (text, audio) in enumerate(turn_audio):
                print(f'Turn {i + 1}: "{text}"')
                print("-" * 50)

                result = await send_audio_and_wait(ws, audio, timeout=20.0)
                results.append((text[:30], len(audio), result["recv_bytes"], result["ttfr"]))

                if result["recv_chunks"] > 0:
                    print(f"  Sent {result['sent_chunks']} chunks")
                    print(
                        f"  Received {result['recv_chunks']} chunks ({result['recv_bytes']} bytes)"
                    )
                    if result["ttfr"]:
                        print(f"  Time to first response: {result['ttfr']:.1f}s")
                else:
                    print("  No response received")
                print()

            # Summary
            print("=" * 70)
            print("SUMMARY")
            print("=" * 70)
            print(f"{'Turn':<5} {'Description':<32} {'Sent':>10} {'Recv':>10} {'TTFR':>8}")
            print("-" * 70)

            total_recv = 0
            for i, (desc, sent, recv, ttfr) in enumerate(results):
                ttfr_str = f"{ttfr:.1f}s" if ttfr else "-"
                print(f"{i:<5} {desc:<32} {sent:>9}B {recv:>9}B {ttfr_str:>8}")
                total_recv += recv

            print("-" * 70)
            print(f"Total received: {total_recv} bytes")
            print()

            if all(r[2] > 0 for r in results):
                print("SUCCESS: All turns completed!")
            else:
                print("PARTIAL: Some turns had no response")

    except ConnectionRefusedError:
        print("\nERROR: Could not connect to server")
        print("Make sure the server is running: uv run python -m inbound.server")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_multiturn_voice())
