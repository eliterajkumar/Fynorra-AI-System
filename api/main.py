# api/main.py
from __future__ import annotations
import os, uuid, time, json, logging
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx, redis, jwt  # PyJWT

load_dotenv()

# basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-backend")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")  # can be wss://... or https://...
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

if not (LIVEKIT_URL and LIVEKIT_API_KEY and LIVEKIT_API_SECRET):
    raise RuntimeError("Missing LiveKit env variables")

# normalize rest base: if LIVEKIT_URL uses ws/wss convert to https for REST calls
def livekit_rest_base(lk_url: str) -> str:
    if lk_url.startswith("wss://"):
        return "https://" + lk_url[len("wss://") :]
    if lk_url.startswith("ws://"):
        return "http://" + lk_url[len("ws://") :]
    return lk_url

LIVEKIT_REST_BASE = livekit_rest_base(LIVEKIT_URL).rstrip("/")

redis_client = redis.from_url(REDIS_URL, decode_responses=True)

app = FastAPI(title="AI Voice Backend")

# allow frontend origins to call — adjust allowed origins in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SessionCreate(BaseModel):
    user_name: str | None = None
    room: str | None = None  # optional: allow client to request room

def mask_name(n: str | None) -> str:
    if not n: return "unknown"
    if len(n) <= 2: return n[0] + "*"
    return n[0] + "*" * (len(n)-2) + n[-1]

def ensure_room_exists(room_name: str) -> None:
    """
    Try to create LiveKit room via REST API (idempotent). Uses LIVEKIT_REST_BASE.
    If the server does not support REST room creation (managed service), we ignore errors.
    """
    url = f"{LIVEKIT_REST_BASE}/v1/rooms"
    payload = {"name": room_name}
    try:
        resp = httpx.post(url, json=payload, auth=(LIVEKIT_API_KEY, LIVEKIT_API_SECRET), timeout=8.0)
        if resp.status_code in (200, 201, 409):
            logger.debug("ensure_room_exists ok: %s", resp.status_code)
            return
        logger.warning("room create returned %s: %s", resp.status_code, resp.text)
    except httpx.RequestError as e:
        # Non-fatal: some managed livekit endpoints don't expose REST admin API.
        logger.debug("could not call LiveKit REST create room: %s", e)

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
    return token if isinstance(token, str) else token.decode("utf-8")

@app.post("/session")
async def create_session(payload: SessionCreate, request: Request):
    # basic rate-limiting placeholder: later integrate real limiter
    client_ip = request.client.host if request.client else "unknown"
    logger.info("session request from %s user=%s", client_ip, payload.user_name)

    session_id = str(uuid.uuid4())
    room_name = (payload.room or "restaurant-room").strip()

    # validate small things
    if payload.user_name and len(payload.user_name) > 100:
        raise HTTPException(status_code=400, detail="user_name too long")

    # try to ensure room exists (non-fatal)
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
    try:
        redis_client.setex(f"session:{session_id}", 3600, json.dumps(session_data))
    except Exception as e:
        logger.warning("Redis setex failed: %s", e)

    return {"session_id": session_id, "room": room_name, "token": token, "expires_at": expires_at}

@app.get("/session/{session_id}")
async def get_session_state(session_id: str):
    data = redis_client.get(f"session:{session_id}")
    if not data:
        raise HTTPException(status_code=404, detail="session not found")
    return json.loads(data)

@app.get("/health")
async def health():
    return {"status": "ok"}
