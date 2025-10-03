"""
Google Sheets integration tool for storing leads.
"""
from loguru import logger
from llama_index.core.tools import FunctionTool
import gspread, os, time

SHEET_ID = os.environ.get("GSHEET_ID")
SERVICE_ACCOUNT_FILE = os.environ.get("GS_SA_JSON")

def save_to_gsheet(name: str, phone: str, email: str, service: str = None, message: str = None) -> str:
    """
    Append lead details to Google Sheet.
    """
    try:
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        sh = gc.open_by_key(SHEET_ID)
        ws = sh.sheet1
        ws.append_row([
            int(time.time()), name, phone, email, service or "", message or ""
        ])
        logger.info(f"Lead saved to Google Sheet: {name}, {email}")
        return f"Lead saved to Google Sheets for {name}."
    except Exception as e:
        logger.error(f"Error saving to Google Sheets: {e}")
        return "Error: Could not save to Google Sheet."

gsheet_tool = FunctionTool.from_defaults(
    fn=save_to_gsheet,
    name="save_to_gsheet",
    description="Save customer lead to Google Sheets (name, phone, email, service, message)."
)
