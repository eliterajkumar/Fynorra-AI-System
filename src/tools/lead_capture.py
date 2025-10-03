"""
Lead capture tool for Fynorra Sales Agent.
"""
from loguru import logger
from llama_index.core.tools import FunctionTool
import json, os, time

LEADS_FILE = "data/leads.jsonl"

def capture_lead(name: str, phone: str, email: str, service: str = None, message: str = None) -> str:
    """
    Capture customer lead details and save locally (JSONL file).
    
    Args:
        name (str): Customer's name
        phone (str): Customer's phone number
        email (str): Customer's email address
        service (str): Requested service or interest
        message (str): Optional message or query
    
    Returns:
        str: Success confirmation message
    """
    try:
        os.makedirs(os.path.dirname(LEADS_FILE), exist_ok=True)
        lead = {
            "timestamp": int(time.time()),
            "name": name,
            "phone": phone,
            "email": email,
            "service": service,
            "message": message
        }
        with open(LEADS_FILE, "a") as f:
            f.write(json.dumps(lead, ensure_ascii=False) + "\n")
        logger.info(f"Lead captured: {lead}")
        return f"Lead saved for {name}. Our sales team will follow up shortly."
    except Exception as e:
        logger.error(f"Error capturing lead: {e}")
        return "Sorry, something went wrong while saving the lead."

# Expose as FunctionTool for agent use
lead_capture_tool = FunctionTool.from_defaults(
    fn=capture_lead,
    name="capture_lead",
    description="Capture customer lead details (name, phone, email, service, message)."
)
