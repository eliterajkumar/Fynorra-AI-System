import asyncio
import os
from dotenv import load_dotenv
from livekit.agents import cli, WorkerOptions, JobContext
from agents.restaurant import entrypoint  # import entrypoint function from your restaurant.py

load_dotenv()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
