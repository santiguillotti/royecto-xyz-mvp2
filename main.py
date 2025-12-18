import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
import os
from pathlib import Path
from openai import OpenAI
import logging
import uuid
from sqlalchemy import create_engine, Column, String, Text, Integer, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

AUDIO_DIR = Path("audio_messages")
AUDIO_DIR.mkdir(exist_ok=True)

# --- CONFIGURACIÓN DE BASE DE DATOS ---
# Si estamos en Render, usa la URL real. Si estamos en Replit probando, usa un archivo local sqlite.
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.warning("⚠️ Usando base de datos local SQLite (No persistente en Render Free)")
    DATABASE_URL = "sqlite:///./local_marle.db"

# Corrección para SQLAlchemy (Render usa postgres:// pero SQLAlchemy quiere postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELOS (TABLAS) ---
class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True) # El UUID de la cookie
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String) # user, assistant, system
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

# Crear tablas si no existen
Base.metadata.create_all(bind=engine)

# Dependencia para obtener la DB en cada petición
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

PROMPTS = {
    "amigo": "Eres un amigo cercano y empático. Tu tono es casual, cálido y comprensivo.",
    "profesional": "Eres un terapeuta profesional. Tu tono es clínico pero empático. Analizas patrones."
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    response = templates.TemplateResponse("index.html", {"request": request})

    # Gestión de Cookie
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    # Asegurar que la sesión existe en DB
    db_session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
    if not db_session:
        db_session = ChatSession(id=session_id)
        db.add(db_session)
        db.commit()

    return response

@app.post("/chat")
async def chat_endpoint(request: Request, response: Response, payload: dict, db: Session = Depends(get_db)):
    # 1. Recuperar Sesión
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    # Crear sesión en DB si es nueva
    if not db.query(ChatSession).filter(ChatSession.id == session_id).first():
        db.add(ChatSession(id=session_id))
        db.commit()

    if not client:
        return JSONResponse(content={"response": "⚠️ Error: Configura la API Key."}, status_code=500)

    try:
        user_text = ""
        mode = payload.get("mode", "amigo")

        # 2. Procesar Audio
        if "audio" in payload and payload["audio"]:
            audio_data = base64.b64decode(payload["audio"])
            temp_filename = AUDIO_DIR / f"temp_{session_id}.webm"
            with open(temp_filename, "wb") as f:
                f.write(audio_data)
            with open(temp_filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file, language="es"
                )
            user_text = transcription.text
        elif "message" in payload:
            user_text = payload["message"]

        if not user_text:
            return {"response": "No entendí."}

        # 3. Guardar Mensaje de Usuario en DB
        db.add(Message(session_id=session_id, role="user", content=user_text))
        db.commit()

        # 4. Recuperar Historial (Últimos 10 + System Prompt)
        # Primero, el system prompt actual
        system_instruction = PROMPTS.get(mode, PROMPTS["amigo"])
        messages_for_ai = [{"role": "system", "content": system_instruction}]

        # Luego, los últimos 10 mensajes de la DB
        last_messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.desc()).limit(10).all()

        # Los recuperamos en orden inverso (del más nuevo al más viejo), así que hay que darles la vuelta
        for msg in reversed(last_messages):
            messages_for_ai.append({"role": msg.role, "content": msg.content})

        # 5. Llamar a GPT
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_ai,
            max_tokens=300,
            temperature=0.7
        )
        ai_response = completion.choices[0].message.content

        # 6. Guardar Respuesta de IA en DB
        db.add(Message(session_id=session_id, role="assistant", content=ai_response))
        db.commit()

        prefix = f"*[🎙️ Escuché: \"{user_text}\"]*\n\n" if "audio" in payload else ""
        return {"response": prefix + ai_response}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"response": f"Error técnico: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)