import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends, BackgroundTasks
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

# --- CONFIGURACIÓN DE LOGS ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Directorio para audios
AUDIO_DIR = Path("audio_messages")
AUDIO_DIR.mkdir(exist_ok=True)

# --- 1. CONFIGURACIÓN DE BASE DE DATOS ---
DATABASE_URL = os.environ.get("DATABASE_URL")

# Fallback: Si no hay base de datos (local), usamos un archivo SQLite
if not DATABASE_URL:
    logger.warning("⚠️ DATABASE_URL no encontrada. Usando SQLite local.")
    DATABASE_URL = "sqlite:///./local_marle.db"

# Corrección para Render: Cambiar postgres:// a postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- 2. MODELOS DE DATOS (TABLAS) ---
class ChatSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, ForeignKey("sessions.id"))
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserProfile(Base):
    __tablename__ = "user_profiles"
    session_id = Column(String, ForeignKey("sessions.id"), primary_key=True)
    profile_json = Column(Text, default="{}")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Crear las tablas en la base de datos
Base.metadata.create_all(bind=engine)

# Dependencia para conectar a la DB en cada petición
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 3. CLIENTE OPENAI ---
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.error("❌ NO SE ENCONTRÓ LA OPENAI_API_KEY")
    client = None
else:
    client = OpenAI(api_key=api_key)

PROMPTS = {
    "amigo": "Eres un amigo cercano y empático. Tu tono es casual, cálido y comprensivo.",
    "profesional": "Eres un terapeuta profesional. Tu tono es clínico pero empático. Analizas patrones."
}

# --- 4. FUNCIÓN DEL ANALISTA (BACKGROUND) ---
def update_patient_profile(session_id: str, user_text: str, current_profile_json: str):
    if not client: return

    # Creamos una sesión de DB nueva para este hilo secundario
    db = SessionLocal()
    try:
        analyst_prompt = f"""
        Eres un analista psicológico silencioso. Extrae datos del usuario y actualiza su perfil JSON.

        Perfil Actual: {current_profile_json}
        Nuevo Mensaje: "{user_text}"

        Instrucciones:
        1. Busca: Nombre, Edad, Ubicación, Profesión, Estado Civil, o Temas Psicológicos Clave.
        2. Si hay datos nuevos, actualiza el JSON.
        3. NO inventes datos.
        4. Responde SOLO con el JSON válido.
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": analyst_prompt}],
            temperature=0.0,
            response_format={ "type": "json_object" }
        )

        new_profile_json = completion.choices[0].message.content

        # Guardar en DB
        profile = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        if not profile:
            profile = UserProfile(session_id=session_id, profile_json=new_profile_json)
            db.add(profile)
        else:
            profile.profile_json = new_profile_json

        db.commit()
        logger.info(f"🕵️‍♂️ Incógnito actualizó perfil: {new_profile_json}")

    except Exception as e:
        logger.error(f"Error en Analista: {e}")
    finally:
        db.close()

# --- 5. RUTAS DE LA APP ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    response = templates.TemplateResponse("index.html", {"request": request})

    # Manejo de Cookie de Sesión
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    # Crear sesión en DB si no existe
    if not db.query(ChatSession).filter(ChatSession.id == session_id).first():
        db.add(ChatSession(id=session_id))
        db.commit()

    return response

@app.post("/chat")
async def chat_endpoint(
    request: Request, 
    response: Response, 
    payload: dict, 
    background_tasks: BackgroundTasks, 
    db: Session = Depends(get_db)
):
    # 1. Recuperar o Crear Sesión
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        db.add(ChatSession(id=session_id))
        db.commit()

    if not client:
        return JSONResponse(content={"response": "⚠️ Error: API Key no configurada."}, status_code=500)

    try:
        user_text = ""
        mode = payload.get("mode", "amigo")

        # 2. Procesar Audio o Texto
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
            return {"response": "No pude escuchar nada."}

        # 3. Guardar mensaje usuario
        db.add(Message(session_id=session_id, role="user", content=user_text))
        db.commit()

        # 4. Contexto para la IA (Perfil + Historial)
        profile_db = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        current_profile_str = profile_db.profile_json if profile_db else "{}"

        base_prompt = PROMPTS.get(mode, PROMPTS["amigo"])
        system_instruction = f"""
        {base_prompt}
        DATOS CONOCIDOS DEL PACIENTE: {current_profile_str}
        """

        messages_for_ai = [{"role": "system", "content": system_instruction}]

        # Últimos 10 mensajes
        last_messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.desc()).limit(10).all()
        for msg in reversed(last_messages):
            messages_for_ai.append({"role": msg.role, "content": msg.content})

        # 5. Generar respuesta
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_ai,
            max_tokens=300,
            temperature=0.7
        )
        ai_response = completion.choices[0].message.content

        # 6. Guardar respuesta y activar analista
        db.add(Message(session_id=session_id, role="assistant", content=ai_response))
        db.commit()

        background_tasks.add_task(update_patient_profile, session_id, user_text, current_profile_str)

        prefix = f"*[🎙️ Escuché: \"{user_text}\"]*\n\n" if "audio" in payload else ""
        return {"response": prefix + ai_response}

    except Exception as e:
        logger.error(f"Error Crítico: {str(e)}")
        return JSONResponse(content={"response": "Ocurrió un error técnico."}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)