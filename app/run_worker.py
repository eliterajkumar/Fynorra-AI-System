# api/run_worker.py
"""
Start the LiveKit agent worker.
Run with: python api/run_worker.py
"""
from __future__ import annotations
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent-worker")

# import the agent entrypoint you already wrote (adjust path if different)
try:
    # if your entrypoint is in agents/restaurant.py or similar:
    from agents.restaurant import entrypoint
    from livekit.agents import cli, WorkerOptions
except Exception as e:
    logger.exception("Failed to import agent entrypoint: %s", e)
    raise

if __name__ == "__main__":
    logger.info("Starting agent worker (cli.run_app)...")
    # This will block and run your worker loop. Render will keep the process alive and show logs.
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
