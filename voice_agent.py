import asyncio
import json
import os
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import Response
from dotenv import load_dotenv

from pipecat.frames.frames import EndFrame, LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator,
    LLMUserResponseAggregator,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService, OpenAITTSService
from pipecat.transports.network.fastapi_websocket import (
    FastAPIWebsocketTransport,
    FastAPIWebsocketParams,
)
from pipecat.serializers.plivo import PlivoFrameSerializer

from plivo import plivoxml

# Load environment variables
load_dotenv()

# Store active pipeline tasks
active_tasks = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for FastAPI startup/shutdown"""
    yield
    # Cleanup on shutdown
    for task in active_tasks.values():
        await task.cancel()


# Create FastAPI app
app = FastAPI(lifespan=lifespan)


@app.post("/answer")
async def answer_webhook(request: Request):
    """
    Plivo calls this endpoint when a call comes in.
    Returns XML to start streaming audio via WebSocket.
    """
    
    # Get ngrok URL from environment
    base_url = os.getenv("NGROK_URL", "")
    if not base_url:
        return Response(
            content="NGROK_URL not configured",
            status_code=500
        )
    
    # Create Plivo XML response
    response = plivoxml.ResponseElement()
    
    # Add greeting
    # response.add(
    #     plivoxml.SpeakElement(
    #         "Hello! Please wait while I connect you to our AI assistant.",
    #         voice="Polly.Joanna"
    #     )
    # )
    
    # Start audio stream to WebSocket
    ws_url = base_url.replace("https://", "wss://") + "/ws/stream"
    
    # CORRECTED: StreamElement takes the URL as the first argument (content)
    stream = plivoxml.StreamElement(
        ws_url,  # This is the 'content' parameter
        bidirectional=True,
        keepCallAlive=True,
        contentType="audio/x-mulaw;rate=8000"
    )
    
    response.add(stream)
    
    return Response(
        content=response.to_string(),
        media_type="application/xml"
    )

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint that handles real-time audio streaming from Plivo.
    """
    
    await websocket.accept()
    
    try:
        # Read the start message from Plivo
        start_data = await websocket.receive_text()
        start_message = json.loads(start_data)
        
        print(f"Received start message: {start_message}")
        
        # Extract Plivo-specific IDs from the start event
        start_info = start_message.get("start", {})
        stream_id = start_info.get("streamId")
        call_id = start_info.get("callId")
        
        if not stream_id or not call_id:
            print("Missing stream_id or call_id")
            await websocket.close()
            return
        
        # Create Plivo serializer with authentication
        serializer = PlivoFrameSerializer(
            stream_id=stream_id,
            call_id=call_id,
            auth_id=os.getenv("PLIVO_AUTH_ID"),
            auth_token=os.getenv("PLIVO_AUTH_TOKEN"),
        )
        
        # Create transport with Plivo serializer
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                add_wav_header=False,
                vad_enabled=True,
                vad_audio_passthrough=True,
                serializer=serializer,
            ),
        )
        
        # Initialize AI services
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            encoding="mulaw",
            sample_rate=8000,
        )
        
        tts = OpenAITTSService(
            api_key=os.getenv("OPENAI_API_KEY"),
            voice="alloy",
            model="tts-1",
        )
        
        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini",
        )
        
        # Create message aggregators
        user_response = LLMUserResponseAggregator()
        assistant_response = LLMAssistantResponseAggregator()
        
        # Create the pipeline
        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                user_response,
                llm,
                tts,
                assistant_response,
                transport.output(),
            ]
        )
        
        # Create initial conversation context
        messages = [
            {
                "role": "system",
                "content": """You are a helpful voice assistant. Keep your responses 
                concise and natural for voice conversation. Be friendly and professional.
                Ask clarifying questions when needed.""",
            }
        ]
        
        # Create and run the pipeline task
        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
        )
        
        # Store the task
        active_tasks[call_id] = task
        
        # Queue initial LLM messages
        await task.queue_frames([LLMMessagesFrame(messages)])
        
        # Run the pipeline
        runner = PipelineRunner()
        await runner.run(task)
        
    except Exception as e:
        print(f"Error in WebSocket handler: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        if call_id in active_tasks:
            del active_tasks[call_id]
        
        try:
            await websocket.close()
        except:
            pass


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "message": "Plivo Voice Agent is running"}


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Plivo Voice Agent...")
    print(f"Webhook URL: {os.getenv('NGROK_URL')}/answer")
    print(f"WebSocket URL: {os.getenv('NGROK_URL').replace('https://', 'wss://')}/ws/stream")
    
    uvicorn.run(app, host="0.0.0.0", port=8080)