#!/usr/bin/env python
"""
Live test script to verify the voice agent greets the caller first.
This simulates a Plivo WebSocket connection to test the greeting behavior.
"""

import asyncio
import base64
import json
import sys

import websockets


async def send_silence(websocket, stream_id: str):
    """Send silence frames to keep the connection alive."""
    # 160 bytes of silence (mu-law encoded)
    silence = base64.b64encode(b"\xff" * 160).decode()
    media_message = {
        "event": "media",
        "media": {
            "payload": silence,
            "streamId": stream_id,
        }
    }
    await websocket.send(json.dumps(media_message))


async def test_greeting():
    """Connect to the voice agent and verify greeting behavior."""
    uri = "ws://localhost:8080/ws/stream"
    stream_id = "test-stream-12345"

    print("Connecting to voice agent WebSocket...")

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Sending Plivo start message...")

            # Send Plivo start message (simulating incoming call)
            start_message = {
                "event": "start",
                "start": {
                    "streamId": stream_id,
                    "callId": "test-call-67890",
                    "from": "+1234567890",
                    "to": "+0987654321",
                }
            }
            await websocket.send(json.dumps(start_message))
            print("Start message sent. Sending silence frames and waiting for agent response...")

            # Listen for responses from the agent
            # The agent should send audio frames with the greeting
            greeting_received = False
            audio_bytes_total = 0
            timeout = 20  # seconds to wait for greeting

            try:
                start_time = asyncio.get_event_loop().time()
                silence_task = None

                async def send_periodic_silence():
                    """Send silence frames periodically to keep the stream active."""
                    while True:
                        await send_silence(websocket, stream_id)
                        await asyncio.sleep(0.02)  # 20ms intervals (50 packets/sec)

                # Start sending silence in the background
                silence_task = asyncio.create_task(send_periodic_silence())

                while asyncio.get_event_loop().time() - start_time < timeout:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=2.0)

                        # Parse the response
                        if isinstance(response, str):
                            data = json.loads(response)
                            event_type = data.get("event", "unknown")

                            if event_type == "playAudio":
                                # Plivo playAudio event - agent is speaking
                                payload = data.get("media", {}).get("payload", "")
                                if payload:
                                    audio_bytes = len(base64.b64decode(payload))
                                    audio_bytes_total += audio_bytes
                                    if not greeting_received:
                                        print(f"Receiving audio from agent...")
                                        greeting_received = True
                            elif event_type in ("clear", "media"):
                                pass  # Ignore clear and media events
                            elif "metrics" in str(data.get("type", "")):
                                pass  # Ignore metrics
                            else:
                                print(f"Received event: {event_type}")
                        else:
                            print(f"Received binary data: {len(response)} bytes")
                            audio_bytes_total += len(response)
                            greeting_received = True

                        # If we've received enough audio, consider the greeting complete
                        if audio_bytes_total > 5000:
                            print(f"Received substantial audio: {audio_bytes_total} bytes")
                            break

                    except asyncio.TimeoutError:
                        print(".", end="", flush=True)
                        continue

                # Cancel the silence task
                if silence_task:
                    silence_task.cancel()
                    try:
                        await silence_task
                    except asyncio.CancelledError:
                        pass

            except websockets.exceptions.ConnectionClosed as e:
                print(f"\nConnection closed: {e}")

            print("\n")
            if greeting_received:
                print(f"SUCCESS: Agent sent audio response (greeting) - {audio_bytes_total} bytes total")
                return True
            else:
                print("WARNING: No audio response received within timeout")
                print("This could mean:")
                print("  - Gemini API key is not configured or invalid")
                print("  - Network issues connecting to Gemini")
                print("  - The greeting mechanism needs adjustment")
                return False

    except ConnectionRefusedError:
        print("ERROR: Could not connect to voice agent.")
        print("Make sure the server is running: python voice_agent.py")
        return False
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Voice Agent Greeting Test")
    print("=" * 60)
    print()

    result = asyncio.run(test_greeting())

    print()
    print("=" * 60)
    sys.exit(0 if result else 1)
