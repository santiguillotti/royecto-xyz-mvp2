# ProyectoXYZ-MVP

## Overview
A mental health support chat application built with FastAPI. Features a beautiful rose-pink themed UI with empathetic AI responses.

## Project Structure
```
/
├── main.py              # FastAPI backend server
├── templates/
│   └── index.html       # Rose-themed chat interface
├── pyproject.toml       # Python dependencies
└── replit.md            # This file
```

## Tech Stack
- **Backend**: FastAPI with Uvicorn
- **Frontend**: HTML + Tailwind CSS + Phosphor Icons
- **Templating**: Jinja2

## Endpoints
- `GET /` - Serves the chat interface
- `POST /chat` - Receives user messages and returns empathetic responses

## Running the Application
The application runs on port 5000 with the command:
```bash
python main.py
```

## Future Enhancements
- Connect real LLM for intelligent empathetic responses
- Add persistent chat history with database
- Implement user authentication
- Add voice input functionality
