from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
import base64
import os
from pathlib import Path

app = FastAPI()

templates = Jinja2Templates(directory="templates")

AUDIO_DIR = Path("audio_messages")
AUDIO_DIR.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: dict):
    mode = payload.get("mode", "amigo")
    
    if "message" in payload:
        user_message = payload.get("message")
        if mode == "profesional":
            bot_response = f"Entiendo. Sobre lo que mencionas: '{user_message}'. Desde una perspectiva clínica, es importante explorar este aspecto más a fondo. ¿Podrías desarrollar más?"
        else:
            bot_response = f"Entiendo que dices: '{user_message}'. Cuéntame más sobre cómo te sientes."
    elif "audio" in payload:
        audio_data = payload.get("audio", "")
        audio_type = payload.get("type", "audio/webm")
        
        audio_bytes = base64.b64decode(audio_data)
        
        audio_filename = f"message_{len(list(AUDIO_DIR.glob('*')))}.webm"
        audio_path = AUDIO_DIR / audio_filename
        
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        
        if mode == "profesional":
            bot_response = f"Gracias por compartir tu mensaje de audio. Como profesional, puedo percibir tu tono y emoción. ¿Hay algo específico en lo que pueda ayudarte profesionalmente?"
        else:
            bot_response = f"Recibí tu mensaje de audio. Estoy aquí para escucharte y apoyarte. ¿Hay algo más que quieras compartir?"
    else:
        bot_response = "No entiendo el mensaje. Intenta de nuevo."
    
    return {"response": bot_response}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)
