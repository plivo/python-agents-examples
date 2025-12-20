# Daily Plivo SIP Dial-Out Example

This example demonstrates how to use Daily.co with Plivo for SIP dial-out functionality, using Daily as the WebRTC transport layer.

## Architecture Overview

The system consists of three main components:

1. **FastAPI Server** (`server.py`): Handles outbound call requests and orchestrates the call setup
2. **Bot Process** (`bot.py`): Manages the AI conversation using Daily.co WebRTC
3. **External Services**:
   - **Daily**: Provides WebRTC transport and SIP capabilities
   - **Plivo**: Handles actual phone call routing (instead of Twilio)
   - **AI Services**: OpenAI (LLM), Deepgram (STT), Cartesia (TTS)

### Call Flow

```
1. Client sends POST request to /outbound-call with phone number
   ↓
2. Server creates Daily room with SIP dial-out enabled
   ↓
3. Server spawns bot process with room details
   ↓
4. Bot joins Daily room via WebRTC
   ↓
5. Bot uses Plivo API to initiate call to target phone number
   ↓
6. Plivo connects call to Daily's SIP endpoint
   ↓
7. Audio flows: Phone ←→ Plivo ←→ Daily SIP ←→ Daily WebRTC ←→ Bot
   ↓
8. Bot processes audio through AI pipeline (STT → LLM → TTS)
```

## Key Differences from Twilio Implementation

### 1. Authentication
- **Twilio**: Uses Account SID and Auth Token
- **Plivo**: Uses Auth ID and Auth Token

### 2. API Structure
- **Twilio**: Uses `twilio.rest.Client`
- **Plivo**: Uses `plivo.RestClient`

### 3. Making Calls
**Twilio:**
```python
call = twilio_client.calls.create(
    from_=from_number,
    to=to_number,
    twiml=twiml_response
)
```

**Plivo:**
```python
call = plivo_client.calls.create(
    from_=from_number,
    to_=to_number,  # Note the underscore
    answer_url=answer_url,
    answer_method="POST"
)
```

### 4. Call Control
- **Twilio**: Uses TwiML (XML format)
- **Plivo**: Uses Plivo XML (similar but different element names)

## Prerequisites

1. **Plivo Account**
   - Sign up at https://www.plivo.com/
   - Get your Auth ID and Auth Token from the Plivo console
   - Purchase a phone number for outbound calling

2. **Daily Account**
   - Sign up at https://daily.co/
   - Get your API key from the Daily dashboard

3. **AI Service Keys**
   - OpenAI API key (for LLM)
   - Deepgram API key (for speech-to-text)
   - Cartesia API key (for text-to-speech)

4. **Python 3.10+**

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file from the example:
```bash
cp .env.example .env
```

4. Fill in your API credentials in `.env`:
```env
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
DAILY_API_KEY=your_daily_api_key
OPENAI_API_KEY=your_openai_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
CARTESIA_API_KEY=your_cartesia_api_key
```

## Running the Application

1. Start the server:
```bash
python server.py
```

This will start the FastAPI server on `http://localhost:8000`

2. Make an outbound call:
```bash
curl -X POST http://localhost:8000/outbound-call \
  -H "Content-Type: application/json" \
  -d '{
    "phone_number": "+1234567890",
    "from_number": "+1987654321"
  }'
```

Replace:
- `+1234567890` with the number you want to call
- `+1987654321` with your Plivo phone number

## Configuration Options

### Bot Personality
Edit the system message in `bot.py`:
```python
messages = [
    {
        "role": "system",
        "content": "You are a helpful AI assistant on a phone call..."
    }
]
```

### Voice Selection
Change the Cartesia voice in `bot.py`:
```python
tts = CartesiaTTSService(
    api_key=os.getenv("CARTESIA_API_KEY"),
    voice_id="79a125e8-cd45-4c13-8a67-188112f4dd22",  # Change this
)
```

Available voices: https://docs.cartesia.ai/voices

### LLM Model
Change the OpenAI model in `bot.py`:
```python
llm = OpenAILLMService(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo", etc.
)
```

## Important Notes

### Plivo XML Response Handler

You'll need to set up a webhook endpoint to handle Plivo's answer_url. When Plivo receives your call request, it will make a POST request to this URL expecting an XML response that tells it what to do with the call.

Example answer_url handler:
```python
@app.post("/plivo-answer")
async def plivo_answer(request: Request):
    """
    Plivo will call this endpoint when the call is answered.
    We need to return XML that connects the call to Daily's SIP endpoint.
    """
    form_data = await request.form()
    
    # Get the SIP URI from your storage (you'll need to implement this)
    sip_uri = get_sip_uri_for_call(form_data.get('CallUUID'))
    
    xml_response = f'''<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Dial>
        <User>{sip_uri}</User>
    </Dial>
</Response>'''
    
    return PlainTextResponse(xml_response, media_type="application/xml")
```

### SIP URI Format

Plivo expects SIP URIs in this format:
```
sip:username@domain
```

Daily provides SIP endpoints in a similar format. Make sure to extract and use the correct SIP URI from the Daily room configuration.

### Call Termination

The bot will automatically terminate when:
- The caller hangs up
- The `on_dialout_stopped` event is triggered
- An error occurs during the call

### Audio Quality

This example is configured for telephone-quality audio (8kHz sample rate). Deepgram's `nova-2-phonecall` model is optimized for phone calls.

## Troubleshooting

### Call doesn't connect
1. Check that your Plivo credentials are correct
2. Verify your Plivo phone number is active and has outbound calling enabled
3. Check the server logs for error messages

### No audio or bot doesn't respond
1. Verify all API keys are correct in `.env`
2. Check that Daily room was created successfully (check server logs)
3. Ensure your firewall allows WebRTC connections

### Call drops immediately
1. Make sure the SIP URI is correctly formatted
2. Verify that Daily's SIP endpoint is accessible
3. Check Plivo console for call logs and error messages

## Deployment Considerations

### Production Deployment

1. **Use a production-ready server**: Consider using Gunicorn or similar
2. **Set up proper logging**: Configure loguru for production logging
3. **Implement error handling**: Add more robust error handling and retry logic
4. **Secure your endpoints**: Add authentication to your API endpoints
5. **Monitor your calls**: Implement monitoring for call quality and success rates

### Webhook URL

For production, you'll need:
1. A publicly accessible HTTPS endpoint
2. Valid SSL certificate
3. Configure the webhook URL in your Plivo application settings

Example using ngrok for development:
```bash
ngrok http 8000
```

Then use the ngrok URL as your webhook endpoint.

## Additional Resources

- [Daily.co Documentation](https://docs.daily.co/)
- [Plivo Documentation](https://www.plivo.com/docs/)
- [Plivo XML Reference](https://www.plivo.com/docs/voice/xml/)
- [Plivo API Reference](https://www.plivo.com/docs/voice/api/)
- [Pipecat Documentation](https://docs.pipecat.ai/) (AI framework used by the bot)

## License

This example is provided as-is for educational purposes.

## Support

For issues related to:
- **Daily.co**: https://help.daily.co/
- **Plivo**: https://support.plivo.com/
- **Pipecat** (AI framework): https://github.com/pipecat-ai/pipecat
