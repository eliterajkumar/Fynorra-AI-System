# api/main.py
from __future__ import annotations

import os
import uuid
import time
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import httpx
import jwt  # PyJWT

# redis import is optional / handled below
try:
    import redis
except Exception:
    redis = None

load_dotenv()

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-backend")

# ----- Env vars -----
LIVEKIT_URL = os.getenv("LIVEKIT_URL")  # e.g., wss://your-livekit-host
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
REDIS_URL = os.getenv("REDIS_URL", "").strip()  # if empty -> no redis

# If you want the server to refuse to start when LiveKit creds missing, keep this check.
# Otherwise you can relax it for local dev. I'm keeping it so token-generation only runs with keys.
if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
    logger.error("Missing LiveKit env variables. Please set LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.")
    raise RuntimeError("Missing LiveKit env variables")

# normalize rest base: if LIVEKIT_URL uses ws/wss convert to https for REST calls
def livekit_rest_base(lk_url: str) -> str:
    if lk_url.startswith("wss://"):
        return "https://" + lk_url[len("wss://") :]
    if lk_url.startswith("ws://"):
        return "http://" + lk_url[len("ws://") :]
    return lk_url

LIVEKIT_REST_BASE = livekit_rest_base(LIVEKIT_URL).rstrip("/")

# ----- Optional Redis init (graceful) -----
redis_client: Optional[object] = None
if REDIS_URL and redis:
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        logger.info("Redis initialized from %s", REDIS_URL)
    except Exception as e:
        logger.warning("Redis init failed: %s. Continuing without Redis.", e)
        redis_client = None
else:
    if not REDIS_URL:
        logger.info("REDIS_URL not set — running without Redis.")
    else:
        logger.info("redis library not available; running without Redis.")

app = FastAPI(title="AI Voice Backend")

# allow frontend origins to call — adjust allowed origins in production
# Add your deployed frontend origin to this list when moving to prod.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        # add your real frontend origin, e.g. "https://www.fynorra.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- models -----
class SessionCreate(BaseModel):
    user_name: str | None = None
    room: str | None = None  # optional: allow client to request room

def mask_name(n: str | None) -> str:
    if not n:
        return "unknown"
    if len(n) <= 2:
        return n[0] + "*"
    return n[0] + "*" * (len(n) - 2) + n[-1]

# ----- ensure room exists (non-fatal; prefer server SDK) -----
def ensure_room_exists(room_name: str) -> None:
    """
    Try to create LiveKit room using the server SDK if available.
    If not available, try REST call as best-effort. Never raise on failure.
    """
    try:
        # Preferred: livekit-server-sdk (if installed)
        from livekit_server_sdk import RoomServiceClient  # type: ignore

        if not (LIVEKIT_REST_BASE and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
            logger.debug("LiveKit params missing for SDK; skipping create_room")
            return

        svc = RoomServiceClient(LIVEKIT_REST_BASE, LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        try:
            svc.create_room({"name": room_name})
            logger.debug("ensure_room_exists (sdk) ok")
            return
        except Exception as e:
            logger.info("livekit sdk create_room non-fatal: %s", e)
            return
    except Exception:
        # SDK not installed; fallback to REST attempt (non-fatal)
        try:
            url = f"{LIVEKIT_REST_BASE}/v1/rooms"
            payload = {"name": room_name}
            resp = httpx.post(url, json=payload, timeout=6.0)
            if resp.status_code in (200, 201, 409):
                logger.debug("ensure_room_exists REST ok: %s", resp.status_code)
            else:
                logger.debug("ensure_room_exists REST non-fatal: %s %s", resp.status_code, resp.text)
        except Exception as exc:
            logger.debug("ensure_room_exists REST skipped: %s", exc)
        return

# ----- Join token generation (JWT) -----
def generate_join_token(identity: str, room: str, ttl_seconds: int = 300) -> str:
    now = int(time.time())
    exp = now + ttl_seconds
    nbf = now - 1
    jti = str(uuid.uuid4())
    payload = {
        "iss": LIVEKIT_API_KEY,
        "exp": exp,
        "nbf": nbf,
        "jti": jti,
        "sub": identity,
        "video": {
            "room": room,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
        },
    }
    token = jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")
    # PyJWT returns str in modern versions
    return token if isinstance(token, str) else token.decode("utf-8")

# ----- session endpoints -----
@app.post("/session")
async def create_session(payload: SessionCreate, request: Request):
    client_ip = request.client.host if request.client else "unknown"
    logger.info("session request from %s user=%s", client_ip, payload.user_name)

    session_id = str(uuid.uuid4())
    room_name = (payload.room or "restaurant-room").strip()

    # validate small things
    if payload.user_name and len(payload.user_name) > 100:
        raise HTTPException(status_code=400, detail="user_name too long")

    # Try to create room (non-fatal)
    ensure_room_exists(room_name)

    identity = payload.user_name or f"user_{session_id[:8]}"
    try:
        token = generate_join_token(identity=identity, room=room_name, ttl_seconds=300)
    except Exception as e:
        logger.exception("token generation failed")
        raise HTTPException(status_code=500, detail=str(e))

    expires_at = int(time.time() + 300)

    session_data = {
        "session_id": session_id,
        "room": room_name,
        "identity": identity,
        "masked_name": mask_name(payload.user_name),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "expires_at": expires_at,
        "client_ip": client_ip,
    }
    # store in redis if available
    if redis_client:
        try:
            redis_client.setex(f"session:{session_id}", 3600, json.dumps(session_data))
        except Exception as e:
            logger.warning("Redis setex failed: %s", e)
    else:
        logger.debug("Redis not configured; skipping session cache")

    return {"session_id": session_id, "room": room_name, "token": token, "expires_at": expires_at}

@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    if not redis_client:
        raise HTTPException(status_code=404, detail="session store not configured")
    data = redis_client.get(f"session:{session_id}")
    if not data:
        raise HTTPException(status_code=404, detail="session not found")
    return json.loads(data)

@app.get("/health")
async def health():
    return {"status": "ok"}

# ----- optional: include voice router if present -----
try:
    # attempt relative import for api package layout
    from .voice import router as voice_router  # type: ignore
except Exception:
    try:
        from voice import router as voice_router  # type: ignore
    except Exception:
        voice_router = None

if voice_router:
    app.include_router(voice_router)
    logger.info("voice router included")
else:
    logger.info("voice router not found — /voice/stream not registered")

# If you also run the worker from the same repo with `python api/main.py`,
# keep worker startup separate (your earlier worker run uses livekit.agents.cli)
# so we don't start it accidentally when running FastAPI server.
if __name__ == "__main__":
    logger.info("Starting FastAPI (uvicorn recommended).")
