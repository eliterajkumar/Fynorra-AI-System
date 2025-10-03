# main.py (modify your existing main.py)
import argparse
from typing import Generator, Tuple
import numpy as np
import os
import threading
import time
import uuid
from loguru import logger
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field
import uvicorn

from src.speech import SpeechService
from src.agent import Agent
from fastrtc import (
    AlgoOptions,
    ReplyOnPause,
    Stream,
)

# load environment variables
load_dotenv()

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# initialize services
speech_service = SpeechService()
agent = Agent()

default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

# ---- FastAPI start-demo endpoint ----
app = FastAPI(title="Fynorra Demo Starter")

class StartDemoPayload(BaseModel):
    name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=6)
    email: EmailStr
    service: str | None = "Demo"
    message: str | None = None
    source: str | None = "website"
    consent: bool = True

# In-memory map: session_id -> lead dict (only for short-lived local sessions)
SESSION_MAP: dict[str, dict] = {}

@app.post("/api/start_demo")
def start_demo(payload: StartDemoPayload):
    if not payload.consent:
        raise HTTPException(status_code=400, detail="Consent required.")
    # 1) generate session id
    session_id = str(uuid.uuid4())
    lead = payload.dict()
    lead["timestamp"] = int(time.time())
    lead["session_id"] = session_id

    # 2) save lead persistently using your existing lead_store tool if available
    try:
        # try to import your wrapper tool (optional)
        from tools.lead_store import store_lead
        # store_lead returns message string — pass through key fields
        store_msg = store_lead(
            name=lead["name"],
            phone=lead["phone"],
            email=lead["email"],
            service=lead.get("service"),
            message=lead.get("message"),
            source=lead.get("source"),
            consent=lead.get("consent"),
            idempotency_key=f"web-{session_id}"
        )
        logger.info(f"lead_store result: {store_msg}")
    except Exception as e:
        # if tool not available or fails, fall back to local file append (durable)
        logger.warning(f"lead_store not available or failed: {e}. Falling back to local save.")
        os.makedirs("data", exist_ok=True)
        with open("data/leads.jsonl", "a") as f:
            import json
            f.write(json.dumps(lead, ensure_ascii=False) + "\n")

    # 3) attach lead to agent session map so agent can use it when the user connects
    try:
        # Agent should expose a method to set session info (see Agent change below)
        agent.set_session_info(session_id, lead)
    except Exception as e:
        logger.warning(f"Agent doesn't support set_session_info: {e}")

    # 4) return session info and URL for frontend to open the demo UI (Gradio)
    gradio_url = os.environ.get("GRADIO_URL", "http://127.0.0.1:7867")
    return {"ok": True, "session_id": session_id, "gradio_url": gradio_url, "message": "Demo session created. Open demo UI and pass session_id."}

def run_fastapi():
    # run uvicorn in the same process bound to 8001
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("API_PORT", 8001)), log_level="info")

# ---- audio response handler (unchanged) ----
def response(
    audio: tuple[int, np.ndarray],
) -> Generator[Tuple[int, np.ndarray], None, None]:
    logger.info("🎙️ Received audio input")
    logger.debug("🔄 Transcribing audio...")
    transcript = speech_service.speech_to_text(audio, response_format="text")
    logger.info(f'👂 Transcribed: "{transcript}"')
    logger.debug("🧠 Running agent...")
    # If caller included session id metadata in transcript or via some channel, agent can use it
    agent_response = agent.invoke(transcript)
    response_text = agent_response["messages"][-1]["content"]
    logger.info(f'💬 Response: "{response_text}"')
    logger.debug("🔊 Generating speech...")
    yield from speech_service.text_to_speech(response_text, voice_id=default_voice_id)

def create_stream() -> Stream:
    return Stream(
        modality="audio",
        mode="send-receive",
        handler=ReplyOnPause(
            response,
            algo_options=AlgoOptions(
                speech_threshold=0.2,
            ),
        ),
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastRTC Voice Agent")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface (automatically provides a temporary phone number)",
    )
    args = parser.parse_args()

    # start FastAPI in background thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    logger.info("📡 Demo API started at http://0.0.0.0:8001 (local)")

    logger.info("🔊 Initialized speech service with ElevenLabs TTS provider")
    logger.info("🎤 Initialized speech service with Groq STT provider")
    logger.info("tts model preloaded during startup")
    
    stream = create_stream()
    logger.info("🎧 Stream handler configured")

    if args.phone:
        logger.info("🚀 Launching with FastRTC phone interface...")
        stream.fastphone()
    else:
        logger.info("🚀 Launching with Gradio UI...")
        stream.ui.launch()
