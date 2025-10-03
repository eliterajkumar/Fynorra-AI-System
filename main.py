# main.py (updated)
import argparse
from typing import Generator, Tuple
import numpy as np
import os
import threading
import time
import uuid
import base64
import tempfile
import json
import traceback
from loguru import logger
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, Field
import uvicorn
import soundfile as sf
import aiofiles
# from src.tools.lead_store import lead_store_tool
from fastapi.middleware.cors import CORSMiddleware

from src.speech import SpeechService
from src.agent import Agent
from fastrtc import (
    AlgoOptions,
    ReplyOnPause, 
    Stream,
)



# allow only your frontend origin in production; during dev you can use "*"
frontend_origin = os.environ.get("https://fynorra.com""*")  # set to e.g. https://your-site.com in prod

# ---- FastAPI app ----
app = FastAPI(title="Fynorra Demo Starter")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin] if frontend_origin != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# load environment variables
load_dotenv()

logger.remove()
logger.add(
    lambda msg: print(msg),
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <level>{message}</level>",
)

# initialize services (reuse these in endpoints)
speech_service = SpeechService()
agent = Agent()

default_voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")

# ---- Models ----
class StartDemoPayload(BaseModel):
    name: str = Field(..., min_length=1)
    phone: str = Field(..., min_length=6)
    email: EmailStr
    service: str | None = "Demo"
    message: str | None = None
    source: str | None = "website"
    consent: bool = True

class StreamPayload(BaseModel):
    session_id: str | None = None
    chunk: str  # data URI: data:audio/xxx;base64,...

class EndPayload(BaseModel):
    session_id: str

# ---- In-memory session store ----
SESSIONS: dict[str, dict] = {}  # session_id -> {"lead": {...}, "wav": path, "last_update": float}

# ---- Helpers ----
def _ensure_session(sid: str | None = None, lead: dict | None = None) -> str:
    if sid and sid in SESSIONS:
        return sid
    new_sid = sid or str(uuid.uuid4())
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix=f"sess-{new_sid}-")
    os.close(tmp_fd)
    # create an empty wav file header? we'll write on first append
    SESSIONS[new_sid] = {"lead": lead or {}, "wav": tmp_path, "last_update": time.time()}
    logger.debug(f"session created: {new_sid} -> {tmp_path}")
    return new_sid

_DATAURI_RE_START = "data:"
def _datauri_to_bytes(data_uri: str) -> bytes:
    if not data_uri.startswith(_DATAURI_RE_START):
        raise ValueError("Not a data URI")
    head, b64 = data_uri.split(",", 1)
    return base64.b64decode(b64)

async def _append_chunk_to_wav(chunk_bytes: bytes, dest_wav: str):
    # write chunk to temp file and read via soundfile for format support (webm/ogg)
    tfd, tpath = tempfile.mkstemp()
    os.close(tfd)
    async with aiofiles.open(tpath, "wb") as f:
        await f.write(chunk_bytes)
    try:
        arr, sr = sf.read(tpath, dtype="int16")
        if os.path.exists(dest_wav) and os.path.getsize(dest_wav) > 44:
            prev, prev_sr = sf.read(dest_wav, dtype="int16")
            if prev_sr != sr:
                logger.warning("sample rate mismatch - overwriting dest wav")
                combined = arr
            else:
                combined = np.concatenate([prev, arr], axis=0)
        else:
            combined = arr
        sf.write(dest_wav, combined, sr, subtype="PCM_16")
    finally:
        try:
            os.remove(tpath)
        except Exception:
            pass

def _tts_bytes_to_datauri(b: bytes) -> str:
    return "data:audio/wav;base64," + base64.b64encode(b).decode("ascii")

# ---- Routes ----
@app.post("/api/start_demo")
def start_demo(payload: StartDemoPayload):
    if not payload.consent:
        raise HTTPException(status_code=400, detail="Consent required.")
    session_id = str(uuid.uuid4())
    lead = payload.dict()
    lead["timestamp"] = int(time.time())
    lead["session_id"] = session_id

    # persist lead using tools.lead_store if available
    try:
        from src.tools.lead_store import store_lead
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
        logger.warning(f"lead_store not available/failed: {e}. Falling back to local save.")
        try:
            os.makedirs("data", exist_ok=True)
            with open("data/leads.jsonl", "a") as f:
                f.write(json.dumps(lead, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("failed to save lead locally")

    # attach to session map & agent
    sid = _ensure_session(session_id, lead)
    SESSIONS[sid]["lead"] = lead
    try:
        agent.set_session_info(sid, lead)
    except Exception:
        logger.debug("agent.set_session_info not implemented; continuing")

    gradio_url = os.environ.get("GRADIO_URL", "http://127.0.0.1:7867")
    return {"ok": True, "session_id": sid, "gradio_url": gradio_url, "message": "Demo session created."}

@app.post("/voice/stream")
async def voice_stream(payload: StreamPayload):
    try:
        sid = _ensure_session(payload.session_id)
        dest = SESSIONS[sid]["wav"]
        SESSIONS[sid]["last_update"] = time.time()

        # decode chunk
        try:
            chunk_bytes = _datauri_to_bytes(payload.chunk)
        except Exception as e:
            logger.error("invalid chunk data: %s", e)
            raise HTTPException(status_code=400, detail="Invalid chunk data")

        # append chunk to session wav
        try:
            await _append_chunk_to_wav(chunk_bytes, dest)
        except Exception:
            logger.exception("failed to append chunk")

        # Transcribe: prefer file-based API if available
        transcript = ""
        try:
            if hasattr(speech_service, "speech_to_text_from_file"):
                transcript = speech_service.speech_to_text_from_file(dest)
            else:
                arr, sr = sf.read(dest, dtype="int16")
                try:
                    transcript = speech_service.speech_to_text((sr, arr), response_format="text")
                except Exception:
                    # last fallback try file method if present
                    if hasattr(speech_service, "speech_to_text_from_file"):
                        transcript = speech_service.speech_to_text_from_file(dest)
        except Exception:
            logger.exception("STT failed")
            transcript = ""

        logger.info(f"[session {sid}] transcribed: {transcript}")

        # Call agent with session context if supported
        try:
            if hasattr(agent, "invoke"):
                try:
                    resp = agent.invoke(transcript, session_id=sid)
                except TypeError:
                    resp = agent.invoke(transcript)
            else:
                resp = {"messages": [{"content": "agent unavailable"}]}
        except Exception:
            logger.exception("agent invocation failed")
            resp = {"messages": [{"content": "Sorry, an error occurred."}]}

        try:
            resp_text = resp.get("messages", [{"content": ""}])[-1]["content"]
        except Exception:
            resp_text = str(resp)

        # TTS -> bytes -> data URI
        audio_datauri = ""
        try:
            if hasattr(speech_service, "text_to_speech_bytes"):
                b = speech_service.text_to_speech_bytes(resp_text, voice_id=default_voice_id)
                audio_datauri = _tts_bytes_to_datauri(b)
            elif hasattr(speech_service, "text_to_speech"):
                gen = speech_service.text_to_speech(resp_text, voice_id=default_voice_id)
                chunks = []
                sr = None
                for piece in gen:
                    if isinstance(piece, tuple) and len(piece) == 2:
                        sr, arr = piece
                        chunks.append(arr)
                    elif isinstance(piece, np.ndarray):
                        chunks.append(piece)
                if chunks and isinstance(chunks[0], np.ndarray):
                    import io
                    buf = io.BytesIO()
                    sf.write(buf, np.concatenate(chunks, axis=0), sr or 24000, format="WAV", subtype="PCM_16")
                    audio_datauri = _tts_bytes_to_datauri(buf.getvalue())
        except Exception:
            logger.exception("TTS conversion failed")
            audio_datauri = ""

        return {"session_id": sid, "text": transcript or resp_text, "audio": audio_datauri}
    except HTTPException:
        raise
    except Exception:
        logger.exception("Unexpected error in /voice/stream")
        raise HTTPException(status_code=500, detail="internal error")

@app.post("/voice/stream/end")
async def voice_stream_end(payload: EndPayload):
    sid = payload.session_id
    if not sid or sid not in SESSIONS:
        raise HTTPException(status_code=404, detail="session not found")
    try:
        if hasattr(agent, "end_session"):
            agent.end_session(sid)
    except Exception:
        pass
    info = SESSIONS.pop(sid, None)
    # optionally remove wav file or keep for debugging
    # try: os.remove(info['wav']) except: pass
    logger.info(f"Ended session {sid}")
    return {"ok": True, "session_id": sid}

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

# ---- main launcher ----
def run_fastapi():
    port = int(os.environ.get("API_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

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
    logger.info(f"📡 Demo API started at http://0.0.0.0:{os.environ.get('API_PORT', '8001')} (local)")

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
