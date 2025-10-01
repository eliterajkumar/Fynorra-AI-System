# agents/adapters/groq_llm.py
from __future__ import annotations
import os, httpx
from typing import List, Dict, Optional

class GroqLLM:
    def __init__(self, 
                 model: str = "llama-3.3-70b-versatile", 
                 max_tokens: int = 1024, 
                 temperature: float = 0.3):
        """
        Wrapper for Groq Chat Completions API.

        Default model = Groq's LLaMA 3.3 70B.
        Other models available:
        - "llama-3.3-70b-versatile"
        - "llama-3.3-8b-instruct"
        - "mixtral-8x7b"
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.key = os.getenv("GROQ_API_KEY")
        if not self.key:
            raise RuntimeError("GROQ_API_KEY not set in .env")

    async def chat(self, messages: List[Dict[str, str]]) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json"
        }
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(self.url, json=body, headers=headers)
            if r.status_code >= 400:
                raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")
            data = r.json()

        if "choices" in data and len(data["choices"]) > 0:
            return {"content": data["choices"][0]["message"].get("content", "")}
        return {"content": ""}

    async def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Simple helper if you just want to pass one prompt (user message)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        res = await self.chat(messages)
        return res["content"]
