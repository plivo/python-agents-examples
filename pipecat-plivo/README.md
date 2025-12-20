# Plivo Voice AI Agent

A voice AI agent built with Pipecat and Plivo that can handle phone calls using speech-to-text, LLM, and text-to-speech.

## Features

- Real-time phone call handling via Plivo
- Speech-to-text using Deepgram
- AI responses using OpenAI GPT-4o-mini
- Text-to-speech using OpenAI TTS
- WebSocket streaming for low latency

## Prerequisites

- Python 3.10 or higher (3.12 recommended)
- Plivo account with phone number
- OpenAI API key
- Deepgram API key
- ngrok for local development

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plivo-voice-agent.git
cd plivo-voice-agent
```

2. Create virtual environment:
```bash
uv venv
```

3. Install dependencies:
```bash
uv pip install -r requirements.txt
```

4. Create `.env` file:
```bash
cp .env.example .env
```

Edit `.env` with your actual API keys:
PLIVO_AUTH_ID=your_plivo_auth_id
PLIVO_AUTH_TOKEN=your_plivo_auth_token
OPENAI_API_KEY=your_openai_api_key
PLIVO_PHONE_NUMBER=+1234567890
DEEPGRAM_API_KEY=your_deepgram_api_key
NGROK_URL=https://your-ngrok-url.ngrok-free.app

## Usage

1. Start the voice agent:
```bash
python voice_agent.py
```

2. In another terminal, start ngrok:
```bash
ngrok http 8080
```

3. Configure Plivo webhook with your ngrok URL:
   - Answer URL: `https://your-ngrok-url.ngrok-free.app/answer`
   - Method: POST

4. Call your Plivo number to test!

## Project Structure

plivo-voice-agent/
├── voice_agent.py       # Main application
├── .env                 # Environment variables (not in git)
├── .env.example         # Example env file
├── .gitignore          # Git ignore rules
├── README.md           # This file
└── requirements.txt    # Python dependencies

## Configuration

See `.env.example` for required environment variables.

## License

MIT

## Contributing

Pull requests are welcome!


