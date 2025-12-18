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

# --- MEMORIA (LISTA GLOBAL) ---
# En un MVP real, esto debería ser una base de datos o por sesión de usuario.
# Aquí usamos una lista simple en memoria. Si reinicias el server, se borra.
conversation_history = []

PROMPTS = {
    "amigo": "Eres un amigo cercano y empático. Tu tono es casual, cálido y comprensivo. Recuerda lo que el usuario te ha dicho antes.",
    "profesional": "Eres un terapeuta profesional. Tu tono es clínico pero empático. Analizas patrones. Recuerda el historial del paciente."
}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(payload: dict):
    global conversation_history

    if not client:
        return JSONResponse(content={"response": "⚠️ Error: Configura la API Key."}, status_code=500)

    try:
        user_text = ""
        mode = payload.get("mode", "amigo")

        # 1. Procesar Audio
        if "audio" in payload and payload["audio"]:
            logger.info("🎤 Procesando audio...")
            audio_data = base64.b64decode(payload["audio"])
            temp_filename = AUDIO_DIR / "temp_input.webm"
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

        # Si es el primer mensaje, inicializamos el historial con el System Prompt
        if not conversation_history:
            conversation_history.append({"role": "system", "content": system_instruction})
        else:
            # Si cambiamos de modo a mitad de charla, actualizamos la instrucción base
            conversation_history[0] = {"role": "system", "content": system_instruction}

        # Agregar mensaje del usuario al historial
        conversation_history.append({"role": "user", "content": user_text})

        # 3. Llamar a GPT con TODO el historial
        logger.info(f"🧠 Enviando historial de {len(conversation_history)} mensajes a GPT...")

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history, # <-- AQUÍ ESTÁ LA MAGIA DE LA MEMORIA
            max_tokens=300,
            temperature=0.7
        )

        ai_response = completion.choices[0].message.content

        # Agregar respuesta de la IA al historial
        conversation_history.append({"role": "assistant", "content": ai_response})

        prefix = ""
        if "audio" in payload:
            prefix = f"*[🎙️ Escuché: \"{user_text}\"]*\n\n"

        return {"response": prefix + ai_response}

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return JSONResponse(content={"response": f"Error técnico: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)