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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

AUDIO_DIR = Path("audio_messages")
AUDIO_DIR.mkdir(exist_ok=True)

# --- CONFIGURACIÓN DB ---
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    logger.warning("⚠️ Usando DB local (No persistente en Render)")
    DATABASE_URL = "sqlite:///./local_marle.db"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELOS (TABLAS) ---
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

# NUEVA TABLA: Perfil del Paciente
class UserProfile(Base):
    __tablename__ = "user_profiles"
    session_id = Column(String, ForeignKey("sessions.id"), primary_key=True)
    # Guardaremos todo el perfil como un JSON de texto para ser flexibles
    # Ej: {"nombre": "Santi", "edad": "30", "tema_clave": "ansiedad social"}
    profile_json = Column(Text, default="{}")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Crear tablas
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

PROMPTS = {
    "amigo": "Eres un amigo cercano y empático. Tu tono es casual.",
    "profesional": "Eres un terapeuta profesional. Tu tono es clínico pero empático."
}

# --- EL CEREBRO ANALISTA (SEGUNDO PLANO) ---
def update_patient_profile(session_id: str, user_text: str, current_profile_json: str):
    """
    Esta función corre en segundo plano. Analiza el texto y actualiza la DB.
    """
    if not client: return

    # Abrimos una nueva sesión de DB para este hilo secundario
    db = SessionLocal()
    try:
        # Instrucción para el Analista
        analyst_prompt = f"""
        Eres un analista psicológico silencioso. Tu trabajo es extraer datos del usuario y actualizar su perfil JSON.

        Perfil Actual: {current_profile_json}
        Nuevo Mensaje del Usuario: "{user_text}"

        Instrucciones:
        1. Identifica si el usuario mencionó: Nombre, Edad, Ubicación, Profesión, Estado Civil, o Temas Psicológicos Clave.
        2. Si hay datos nuevos o contradictorios, actualiza el JSON.
        3. NO inventes datos. Si no hay nada nuevo, devuelve el JSON igual.
        4. Responde SOLO con el JSON válido, sin texto extra.
        """

        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Usamos el modelo barato para esto
            messages=[{"role": "system", "content": analyst_prompt}],
            temperature=0.0, # Temperatura 0 para ser precisos
            response_format={ "type": "json_object" } # Forzamos respuesta JSON
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
        logger.info(f"🕵️‍♂️ Perfil actualizado para {session_id}: {new_profile_json}")

    except Exception as e:
        logger.error(f"Error en el analista: {e}")
    finally:
        db.close()


# --- RUTAS ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, db: Session = Depends(get_db)):
    response = templates.TemplateResponse("index.html", {"request": request})
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    # Crear sesión inicial
    if not db.query(ChatSession).filter(ChatSession.id == session_id).first():
        db_session = ChatSession(id=session_id)
        db.add(db_session)
        db.commit()

    return response

@app.post("/chat")
async def chat_endpoint(
    request: Request, 
    response: Response, 
    payload: dict, 
    background_tasks: BackgroundTasks, # Inyectamos el gestor de tareas de fondo
    db: Session = Depends(get_db)
):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        # Asegurar creación en DB
        db.add(ChatSession(id=session_id))
        db.commit()

    if not client:
        return JSONResponse(content={"response": "⚠️ Error: API Key."}, status_code=500)

    try:
        user_text = ""
        mode = payload.get("mode", "amigo")

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

        # Guardar mensaje usuario
        db.add(Message(session_id=session_id, role="user", content=user_text))
        db.commit()

        # --- RECUPERAR PERFIL PARA DARLE CONTEXTO A MARLE ---
        profile_db = db.query(UserProfile).filter(UserProfile.session_id == session_id).first()
        current_profile_str = profile_db.profile_json if profile_db else "{}"

        # Inyectamos el perfil en las instrucciones del sistema
        base_prompt = PROMPTS.get(mode, PROMPTS["amigo"])
        system_instruction_with_context = f"""
        {base_prompt}

        DATOS CONOCIDOS DEL PACIENTE (Úsalos sutilmente, no los repitas como robot):
        {current_profile_str}
        """

        # Armar historial para GPT
        messages_for_ai = [{"role": "system", "content": system_instruction_with_context}]
        last_messages = db.query(Message).filter(Message.session_id == session_id).order_by(Message.created_at.desc()).limit(10).all()
        for msg in reversed(last_messages):
            messages_for_ai.append({"role": msg.role, "content": msg.content})

        # Llamar a GPT (Marle)
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages_for_ai,
            max_tokens=300,
            temperature=0.7
        )
        ai_response = completion.choices[0].message.content

        # Guardar respuesta IA
        db.add(Message(session_id=session_id, role="assistant", content=ai_response))
        db.commit()

        # --- ACTIVAR AL ANALISTA (BACKGROUND TASK) ---
        # Esto sucede DESPUÉS de responderle al usuario, para no hacerlo esperar
        background_tasks.add_task(update_patient_profile, session_id, user_text, current_profile_str)

        prefix = f"*[🎙️ Escuché: \"{user_text}\"]*\n\n" if "audio" in payload else ""
        return {"response": prefix + ai_response}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"response": f"Error técnico: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)