# api/voice.py
import base64
import uuid
import logging
import os
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()
logger = logging.getLogger("voice-endpoint")

@router.post("/voice/stream")
async def voice_stream(req: Request):
    """
    Accepts small audio chunks as data-URI JSON payload:
      { "session_id": "...", "chunk": "data:audio/webm;base64,...." }

    Responds ASAP with:
      { "session_id": "...", "text": "...", "audio": "data:audio/wav;base64,..." (optional) }
    """
    body = await req.json()
    session_id = body.get("session_id") or str(uuid.uuid4())
    chunk = body.get("chunk")
    if not chunk:
        return JSONResponse(status_code=400, content={"error": "chunk required"})

    # decode chunk
    try:
        if chunk.startswith("data:"):
            _, b64 = chunk.split(",", 1)
            audio_bytes = base64.b64decode(b64)
        else:
            audio_bytes = base64.b64decode(chunk)
    except Exception as e:
        logger.exception("bad chunk decode")
        return JSONResponse(status_code=400, content={"error": "bad chunk format"})

    # TODO: Replace the lines below with:
    # 1) push bytes into streaming ASR / buffer for VAD detection
    # 2) when an utterance is complete, call LLM and generate TTS
    # 3) return TTS audio as data URI so client plays immediately
    # For now we return a small ACK or canned demo audio from env var DEMO_WAV_BASE64.

    demo_wav_b64 = os.environ.get("DEMO_WAV_BASE64", "")
    if demo_wav_b64:
        logger.info("Returning demo audio for session %s", session_id)
        return {"session_id": session_id, "audio": f"data:audio/wav;base64,{demo_wav_b64}", "text": "Demo response"}
    else:
        logger.debug("Received chunk for session %s (%d bytes)", session_id, len(audio_bytes))
        return {"session_id": session_id, "text": "Received chunk (demo)"}
