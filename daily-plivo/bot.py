import os
import sys
import asyncio
import argparse
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import TextFrame

load_dotenv(override=True)

# Fix SSL certificate verification issues
import ssl
import certifi
import os

# Set SSL certificate path for Python
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

# Configure logging
logger.remove()
logger.add(sys.stderr, level="DEBUG")


async def main(
    room_url: str,
    token: str,
    phone_number: str,
    sip_uri: str
):
    """
    Main bot logic using Plivo for dial-out
    """
    # Configure Daily transport
    transport = DailyTransport(
        room_url,
        token,
        "AI Phone Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,  # Enable transcription to see if we're receiving audio
        )
    )
    
    # Initialize AI services
    # Note: SSL context may need to be passed differently depending on service implementation
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        model="nova-2-phonecall",
        language="en-US",
    )
    
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # British Lady
    )
    
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
    )
    
    # Set up the context and messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant on a phone call. Be concise and natural in your responses. Speak as if you're having a real conversation. Always greet callers warmly when they join.",
        }
    ]
    
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)
    
    # Build the pipeline
    # Note: TextFrame goes directly to LLM, bypassing STT
    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])
    
    # Create the task
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        )
    )
    
    # Event handlers for the transport
    @transport.event_handler("on_joined")
    async def on_joined(transport, participant):
        logger.info(f"Bot joined the room - waiting for call to connect via Plivo -> Daily SIP")
        logger.info(f"Bot participant info: {participant}")
        
    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        logger.info(f"=== PARTICIPANT JOINED ===")
        logger.info(f"Participant: {participant}")
        logger.info(f"Participant type: {type(participant)}")
        logger.info(f"Participant details: {participant if isinstance(participant, dict) else 'Not a dict'}")
        
        # Send a greeting when participant joins
        try:
            logger.info("Sending greeting to participant...")
            # Send a text frame through the pipeline to trigger LLM response
            greeting_text = "Hello! I'm your AI assistant. How can I help you today?"
            await task.queue_frames([TextFrame(greeting_text)])
            logger.info(f"Greeting queued: {greeting_text}")
        except Exception as e:
            logger.error(f"Error sending greeting: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        """
        Called when a participant leaves
        """
        logger.info(f"Participant left: {participant}, reason: {reason}")
        await task.cancel()
    
    @transport.event_handler("on_call_state_updated")
    async def on_call_state_updated(transport, state):
        """
        Called when call state changes
        """
        logger.info(f"Call state updated: {state}")
    
    @transport.event_handler("on_audio_frame")
    async def on_audio_frame(transport, frame):
        """
        Called when audio frame is received
        """
        logger.debug(f"Audio frame received: {len(frame) if hasattr(frame, '__len__') else 'unknown'} bytes")
    
    @transport.event_handler("on_transcription")
    async def on_transcription(transport, transcription):
        """
        Called when transcription is received
        """
        logger.info(f"Transcription received: {transcription}")
    
    # Run the pipeline
    runner = PipelineRunner()
    
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily Plivo Dial-out Bot")
    parser.add_argument("-u", "--url", type=str, required=True, help="Daily room URL")
    parser.add_argument("-t", "--token", type=str, required=True, help="Daily token")
    parser.add_argument("-n", "--number", type=str, required=True, help="Phone number to call")
    parser.add_argument("-s", "--sip", type=str, required=True, help="Daily SIP URI")
    
    args = parser.parse_args()
    
    asyncio.run(
        main(
            room_url=args.url,
            token=args.token,
            phone_number=args.number,
            sip_uri=args.sip
        )
    )