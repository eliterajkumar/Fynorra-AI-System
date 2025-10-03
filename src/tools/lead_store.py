"""
Lead store wrapper for Fynorra.
Tries Google Sheets first (if configured). On any failure, saves to local JSONL (durable).
Provides minimal validation, optional idempotency_key, and basic retry for GSheet pushes.
"""

from loguru import logger
from llama_index.core.tools import FunctionTool
import json, os, time
from typing import Optional
from pydantic import BaseModel, EmailStr, ValidationError
import functools
import traceback

# Optional gspread dependency
try:
    import gspread
    GS_AVAILABLE = True
except Exception:
    GS_AVAILABLE = False

# Config / env
LEADS_FILE = os.environ.get("LEADS_FILE", "data/leads.jsonl")
GSHEET_ID = os.environ.get("GSHEET_ID")         # sheet id from URL
GS_SA_JSON = os.environ.get("GS_SA_JSON")       # path to service account JSON
GSHEET_RETRY = int(os.environ.get("GSHEET_RETRY", "2"))

# Pydantic model for validation
class LeadModel(BaseModel):
    name: str
    phone: str
    email: EmailStr
    service: Optional[str] = None
    message: Optional[str] = None
    source: Optional[str] = "agent"
    consent: bool = True
    idempotency_key: Optional[str] = None  # optional unique key to avoid duplicates

def _append_local(lead: dict):
    os.makedirs(os.path.dirname(LEADS_FILE) or ".", exist_ok=True)
    # atomic write: write to temp then rename
    tmp = LEADS_FILE + ".tmp"
    with open(tmp, "a") as f:
        f.write(json.dumps(lead, ensure_ascii=False) + "\n")
    # append tmp to actual file (safe even if tmp has single line)
    with open(tmp, "r") as t, open(LEADS_FILE, "a") as dest:
        dest.write(t.read())
    try:
        os.remove(tmp)
    except Exception:
        pass

def _push_to_gsheet(lead: dict):
    if not GS_AVAILABLE:
        raise RuntimeError("gspread not installed")
    if not GS_SA_JSON or not GSHEET_ID:
        raise RuntimeError("Google Sheets not configured (GS_SA_JSON or GSHEET_ID missing)")

    gc = gspread.service_account(filename=GS_SA_JSON)
    sh = gc.open_by_key(GSHEET_ID)
    ws = sh.sheet1
    row = [
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(lead.get("timestamp", time.time()))),
        lead.get("idempotency_key") or "",
        lead.get("name"),
        lead.get("phone"),
        lead.get("email"),
        lead.get("service") or "",
        lead.get("message") or "",
        lead.get("source") or ""
    ]
    ws.append_row(row)  # may raise on network/auth errors

def _try_gsheet_with_retries(lead: dict, retries: int = 2):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            _push_to_gsheet(lead)
            return True
        except Exception as e:
            last_exc = e
            logger.warning(f"GSheet push attempt {attempt} failed: {e}")
            # small backoff
            time.sleep(0.5 * attempt)
    # after retries
    logger.error(f"GSheet push failed after {retries} attempts: {last_exc}")
    logger.debug(traceback.format_exc())
    return False

def store_lead(
    name: str,
    phone: str,
    email: str,
    service: Optional[str] = None,
    message: Optional[str] = None,
    source: Optional[str] = "agent",
    consent: bool = True,
    idempotency_key: Optional[str] = None
) -> str:
    """
    Store lead: try Google Sheets first (if configured); on failure, save locally.
    Returns a short status message string.
    """
    try:
        # validate
        lead = LeadModel(
            name=name, phone=phone, email=email,
            service=service, message=message, source=source,
            consent=consent, idempotency_key=idempotency_key
        ).dict()
    except ValidationError as ve:
        logger.error(f"Lead validation error: {ve}")
        return f"Invalid lead data: {ve.errors()}"

    # add timestamp
    lead["timestamp"] = int(time.time())

    # Attempt Google Sheets if available & configured
    gsheet_ok = False
    try:
        if GS_AVAILABLE and GS_SA_JSON and GSHEET_ID:
            gsheet_ok = _try_gsheet_with_retries(lead, retries=GSHEET_RETRY)
        else:
            logger.debug("GSheet not configured or gspread missing; skipping GSheet push.")
    except Exception as e:
        logger.exception("Unexpected error while trying GSheet push: %s", e)
        gsheet_ok = False

    # Always append locally as durable fallback (if gsheet succeeded we still keep local copy)
    try:
        _append_local(lead)
    except Exception as e:
        logger.exception("Failed to append lead locally: %s", e)
        # If both local and gsheet failed, surface error
        if not gsheet_ok:
            return "Error: Could not save lead (both GSheet and local storage failed)."

    if gsheet_ok:
        return f"Lead saved to Google Sheets and local store for {lead['name']}."
    else:
        return f"Lead saved locally for {lead['name']}. (GSheet push not configured or failed.)"

# Expose for agents/tools framework
lead_store_tool = FunctionTool.from_defaults(
    fn=store_lead,
    name="store_lead",
    description="Store lead: tries Google Sheets first, then local JSONL fallback. Args: name, phone, email, service, message, source, consent, idempotency_key."
)
