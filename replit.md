# ProyectoXYZ-MVP

## Overview
A mental health support chat application built with FastAPI. Features a beautiful rose-pink themed UI with empathetic AI responses and audio message support.

## Project Structure
```
/
├── main.py              # FastAPI backend server
├── templates/
│   └── index.html       # Rose-themed chat interface with audio recording
├── audio_messages/      # Stored audio message files
├── pyproject.toml       # Python dependencies
└── replit.md            # This file
```

## Tech Stack
- **Backend**: FastAPI with Uvicorn
- **Frontend**: HTML + Tailwind CSS + Phosphor Icons + Web Audio API
- **Templating**: Jinja2
- **Audio**: WebM format stored on backend

## Features
- ✅ Beautiful rose-pink themed responsive UI
- ✅ Text-based chat with empathetic responses
- ✅ Audio message recording via microphone button
- ✅ Audio files stored on backend for processing
- ✅ Sidebar navigation (Home, Bookmarks, Crisis, Profile)
- ✅ Mobile-responsive design

## Endpoints
- `GET /` - Serves the chat interface
- `POST /chat` - Handles both text messages and audio files
  - Text: `{ "message": "user text" }`
  - Audio: `{ "audio": "base64-encoded-audio", "type": "audio/webm" }`

## Running the Application
The application runs on port 5000 with the command:
```bash
python main.py
```

## Future Enhancements
- Connect real LLM for intelligent empathetic responses
- Add speech-to-text for audio transcription
- Add persistent chat history with database
- Implement user authentication
- Add text-to-speech responses
- Add voice input functionality with audio playback
