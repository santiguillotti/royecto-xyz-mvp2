import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import base64
import os
from pathlib import Path
from openai import OpenAI
import logging

# Configuración de Logs (para ver errores en la consola)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configuración de Templates
templates = Jinja2Templates(directory="templates")

# Directorio temporal para audios
AUDIO_DIR = Path("audio_messages")
AUDIO_DIR.mkdir(exist_ok=True)

# --- 1. CONEXIÓN CON EL CEREBRO (OPENAI) ---
# Intentamos obtener la clave de los Secrets
api_key = os.environ.get("OPENAI_API_KEY")

# Verificación de seguridad
if not api_key:
    logger.warning("⚠️ NO SE ENCONTRÓ LA API KEY. La IA no funcionará.")
else:
    client = OpenAI(api_key=api_key)


# --- 2. PERSONALIDADES (SYSTEM PROMPTS) ---
PROMPTS = {
    "amigo": (
        "Eres un amigo cercano y empático. Tu tono es casual, cálido y comprensivo. "
        "Usas frases cortas y emojis ocasionalmente. No das consejos clínicos, solo "
        "escuchas y validas los sentimientos. Tu objetivo es que el usuario se sienta acompañado."
    ),
    "profesional": (
        "Eres un terapeuta profesional experto en psicología clínica y sobre todo en el metodo CBT (Cognitive Behavioral Therapy). Tu tono es calmado, "
        "profesional y reflexivo. Haces preguntas abiertas para invitar a la introspección. "
        "No juzgas. Si detectas riesgo grave, sugieres buscar ayuda local. "
        "Tus respuestas son estructuradas y orientadas al bienestar mental."
    )
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: dict):
    if not api_key:
        return JSONResponse(
            content={"response": "⚠️ Error: No has configurado la API Key en los Secrets de Replit."},
            status_code=500
        )

    try:
        user_text = ""
        mode = payload.get("mode", "amigo")

        # --- PASO A: PROCESAR AUDIO (SI EXISTE) ---
        if "audio" in payload and payload["audio"]:
            logger.info("🎤 Recibido audio. Procesando con Whisper...")

            # 1. Decodificar el Base64 a archivo real
            audio_data = base64.b64decode(payload["audio"])
            temp_filename = AUDIO_DIR / "temp_input.webm"

            with open(temp_filename, "wb") as f:
                f.write(audio_data)

            # 2. Enviar a OpenAI Whisper (Speech to Text)
            with open(temp_filename, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1", 
                    file=audio_file,
                    language="es" # Forzamos español para mejorar precisión
                )

            user_text = transcription.text
            logger.info(f"📝 Transcripción: {user_text}")

        elif "message" in payload:
            user_text = payload["message"]

        if not user_text:
            return {"response": "No pude entender el mensaje."}

        # --- PASO B: PENSAR RESPUESTA (GPT) ---
        logger.info(f"🧠 Consultando a GPT en modo: {mode}")

        system_instruction = PROMPTS.get(mode, PROMPTS["amigo"])

        completion = client.chat.completions.create(
            model="gpt-4o-mini", # Modelo rápido y barato (o usa gpt-3.5-turbo)
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_text}
            ],
            max_tokens=300, # Limitar longitud de respuesta
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content

        # (Opcional) Si vino de audio, agregamos un indicador visual
        prefix = ""
        if "audio" in payload:
            prefix = f"*[🎙️ Escuché: \"{user_text}\"]*\n\n"

        return {"response": prefix + ai_response}

    except Exception as e:
        logger.error(f"🔥 Error en el backend: {str(e)}")
        return JSONResponse(
            content={"response": f"Lo siento, ocurrió un error técnico: {str(e)}"},
            status_code=500
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)