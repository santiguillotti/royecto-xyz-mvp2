# Sesion de Psiquiatria de Marle - MVP

## Overview
A mental health support chat application built with FastAPI. Features a beautiful light sky-blue themed UI with empathetic AI responses, audio message support, and dual-mode interaction (professional therapist vs. empathetic friend).

## Project Structure
```
/
├── main.py              # FastAPI backend server with mode-based responses
├── templates/
│   └── index.html       # Light blue themed chat interface with audio recording & wave visualization
├── audio_messages/      # Stored audio message files
├── pyproject.toml       # Python dependencies
└── replit.md            # This file
```

## Tech Stack
- **Backend**: FastAPI with Uvicorn
- **Frontend**: HTML + Tailwind CSS + Phosphor Icons + Web Audio API + SVG animations
- **Templating**: Jinja2
- **Audio**: WebM format stored on backend

## Features
- ✅ Light sky-blue responsive UI with rose/purple accents
- ✅ Session title: "Sesion de Psiquiatria de Marle"
- ✅ Text-based chat with empathetic responses
- ✅ Audio message recording via microphone button
- ✅ Animated audio wave visualization (rose-to-purple gradient)
- ✅ Dual-mode interaction via "Herramientas":
  - "Terapeuta profesional" - Clinical, professional responses
  - "Amigo que te escucha" - Empathetic, conversational responses
- ✅ Audio files stored on backend for processing
- ✅ Sidebar navigation (Home, Bookmarks, Crisis, Profile)
- ✅ Mobile-responsive design

## Endpoints
- `GET /` - Serves the chat interface
- `POST /chat` - Handles text, audio, and mode selection
  - Text: `{ "message": "user text", "mode": "amigo|profesional" }`
  - Audio: `{ "audio": "base64-encoded-audio", "type": "audio/webm", "mode": "amigo|profesional" }`

## Running the Application
The application runs on port 5000 with the command:
```bash
python main.py
```

## Recent Updates
- Changed background to light sky blue (#E6F2FF)
- Updated title to "Sesion de Psiquiatria de Marle"
- Added Herramientas mode selector (Professional Therapist vs. Empathetic Friend)
- Implemented real-time audio wave visualization (like WhatsApp) using Web Audio API
- Waves now respond to actual audio frequency data while recording
- Fixed microphone button to properly start/stop recording
- Backend now responds differently based on selected mode

## Future Enhancements
- Connect real LLM (OpenAI, Anthropic, etc.) for intelligent responses
- Add speech-to-text for audio transcription
- Add persistent chat history with database
- Implement user authentication
- Add text-to-speech responses
- Add voice input functionality with audio playback
