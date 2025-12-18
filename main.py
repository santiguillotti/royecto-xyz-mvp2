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
import uuid  # Librería para generar IDs únicos

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="templates")

AUDIO_DIR = Path("audio_messages")
AUDIO_DIR.mkdir(exist_ok=True)

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.warning("⚠️ NO SE ENCONTRÓ LA API KEY.")
    client = None
else:
    client = OpenAI(api_key=api_key)

# --- MEMORIA POR SESIÓN ---
# Diccionario para guardar historiales separados por usuario
# Estructura: { "session_id_xyz": [ {role: user, ...}, ... ] }
user_sessions = {}

PROMPTS = {
    "amigo": "Eres un amigo cercano y empático. Tu tono es casual, cálido y comprensivo. Recuerda lo que el usuario te ha dicho antes.",
    "profesional": "Eres un terapeuta profesional. Tu tono es clínico pero empático. Analizas patrones. Recuerda el historial del paciente."
}

# --- GESTIÓN DE COOKIES ---
async def get_session_id(request: Request, response: Response):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
        # Seteamos la cookie para que el navegador la recuerde
        # httponly=True hace que javascript no pueda leerla (más seguro)
        response.set_cookie(key="session_id", value=session_id, httponly=True)
        logger.info(f"🆕 Nueva sesión creada: {session_id}")
    return session_id

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Generamos cookie si no existe al cargar la página
    response = templates.TemplateResponse("index.html", {"request": request})

    if not request.cookies.get("session_id"):
        new_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=new_id, httponly=True)

    return response

@app.post("/chat")
async def chat_endpoint(request: Request, response: Response, payload: dict):
    # Recuperamos el ID del usuario de la cookie
    session_id = request.cookies.get("session_id")

    # Si por alguna razón no tiene cookie, creamos una al vuelo
    if not session_id:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    # Inicializamos la memoria de ESTE usuario si no existe
    if session_id not in user_sessions:
        user_sessions[session_id] = []

    # Alias para la memoria de este usuario específico
    history = user_sessions[session_id]

    if not client:
        return JSONResponse(content={"response": "⚠️ Error: Configura la API Key."}, status_code=500)

    try:
        user_text = ""
        mode = payload.get("mode", "amigo")

        # 1. Procesar Audio
        if "audio" in payload and payload["audio"]:
            logger.info(f"🎤 Procesando audio usuario {session_id}...")
            audio_data = base64.b64decode(payload["audio"])
            # Usamos el session_id en el nombre del archivo para no mezclar audios
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

        # 2. Gestión de Memoria e Instrucciones
        system_instruction = PROMPTS.get(mode, PROMPTS["amigo"])

        # Lógica de System Prompt:
        # Si la historia está vacía, agregamos el System Prompt.
        # Si ya tiene datos, verificamos si el mensaje [0] es system y lo actualizamos si cambió el modo.
        if not history:
            history.append({"role": "system", "content": system_instruction})
        else:
            if history[0]["role"] == "system":
                history[0]["content"] = system_instruction

        # Agregar mensaje del usuario AL HISTORIAL DE SU SESIÓN
        history.append({"role": "user", "content": user_text})

        # Limitar memoria para no gastar tokens infinitos (Mantenemos últimos 10 mensajes + System Prompt)
        # Esto es un truco de ahorro de dinero
        if len(history) > 11:
            # Mantenemos el index 0 (System) y los últimos 10
            user_sessions[session_id] = [history[0]] + history[-10:]
            history = user_sessions[session_id]

        # 3. Llamar a GPT
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history, 
            max_tokens=300,
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content

        # Guardar respuesta
        history.append({"role": "assistant", "content": ai_response})

        prefix = ""
        if "audio" in payload:
            prefix = f"*[🎙️ Escuché: \"{user_text}\"]*\n\n"

        return {"response": prefix + ai_response}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"response": f"Error técnico: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)