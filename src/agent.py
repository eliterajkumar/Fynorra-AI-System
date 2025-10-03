# src/agent.py
"""
Receptionist-style Agent (no RAG).
Fast local handlers (greeting/menu/reservation/order/business info).
Falls back to ReActAgent (LLM) for general/complex queries.

Behavior changes per request:
 - Avoid blunt "no" or robotic error phrases.
 - If the agent doesn't have info, politely say "I don't have that info right now" and offer to learn.
 - All LLM responses should be returned so the caller can send them to TTS.
 - Return metadata "speak": True so the caller knows to TTS the reply.
"""
from __future__ import annotations

import inspect
import asyncio
import os
import re
import yaml
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv


# LlamaIndex ReActAgent (if installed). We only use it for LLM fallback.
try:
    from llama_index.core.agent import ReActAgent
except Exception:
    ReActAgent = None

from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage

from .llm_service import LLMService
# from .tools import get_weather  # optional tool

load_dotenv()


class Agent:    
    """
    Fynorra Sales Agent (Receptionist-style + Sales-focused)

    Responsibilities:
      - Fast local handlers for greetings, basic company info (services, pricing tiers,
        demo availability), and scheduling demos/meetings.
      - Act as a practical sales rep: qualify leads (BANT-like checks), pitch Fynorra services
        concisely, handle common objections, and present pricing/options.
      - Capture lead details (name, company, contact, needs, budget, timeline) and save to
        configured CRM/storage (e.g., CRM API or local leads queue).
      - Offer and schedule live demos or trials; provide demo links and next-step emails.
      - For complex or out-of-scope queries (technical deep-dive, legal, custom dev), fall
        back to LLM ReActAgent for draft responses or route to human sales engineer.
      - If missing info, politely ask user for details to continue qualification and log
        the conversation so agent can learn.
      - Respect rate-limits, privacy, and opt-out requests (GDPR/consent-aware).
      - Config-driven: reads company details, services, scripts, pricing, and CRM config
        from config/business.yaml and config/sales_playbook.yaml when present.
    """


    DEFAULT_SERVICES = [
        {"id": "S001", "name": "AI Chatbot (Basic rule-based)", "starting_price": "USD 100"},
        {"id": "S002", "name": "AI Chatbot (RAG + LLM)", "starting_price": "USD 1000"},
        {"id": "S003", "name": "Voice AI / Call Automation", "starting_price": "USD 1500"},
        {"id": "S004", "name": "Document Processing (OCR + Automation)", "starting_price": "USD 5000"},
        {"id": "S005", "name": "CRM Automation & Integrations", "starting_price": "USD 8,000"},
    ]

    GREETINGS = {"hi", "hello", "hey"}

    OFFER_TO_LEARN = (
        "I don't have that information right now."
        "I can have our team follow up with you if you'd like?"
    )

    def __init__(self, business_config_path: str | None = None):
        # load prompts (optional)
        prompts_path = Path(__file__).parent.parent / "config" / "prompts.yaml"
        base_prompt = ""
        if prompts_path.exists():
            try:
                with open(prompts_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                    base_prompt = cfg.get("system_prompts", {}).get("fynorra_sales_assistant", "") or ""
            except Exception as e:
                logger.debug("Failed loading prompts.yaml: %s", e)

        # system prompt guiding LLM behavior
        self.system_prompt = (
            base_prompt.strip()
            + "\n\nYou are Fynorra Sales Assistant. Answer conversationally in clear English. "
            "Be persuasive but honest, concise, and include a direct next step (e.g., 'Book a demo', 'Share pricing')."
        )

        # load business info
        self.business_info = self._load_business_info(business_config_path)

        # services list (from business config or defaults)
        self.SERVICES = self.business_info.get("services", self.DEFAULT_SERVICES)

        # LLM service
        self.llm_service = LLMService()

        # memory for conversation history
        self.memory = Memory.from_defaults(token_limit=4000, chat_history_token_ratio=0.8)
        self.memory.put_messages([ChatMessage(role="system", content=self.system_prompt)])

        # init ReActAgent if available
        self._init_agent()

        logger.debug("Fynorra Sales Agent initialized for business: %s", self.business_info.get("name"))

    def _load_business_info(self, business_config_path: str | None = None) -> dict:
        default_info = {
            "name": "FYNORRA AI SOLUTIONS PRIVATE LIMITED",
            "common_name": "Fynorra AI Solutions",
            "address": "H No 101A, Deep Enclave, Gali No 8, Vikas Nagar, Uttam Nagar West, Delhi, India, 110059",
            "opening_hours": "Mon-Fri 09:30-18:30 IST",
            "description": "AI-first consulting firm building custom AI assistants, voice agents, and automation for lead capture & support.",
            "founder": "Rajkumar",
            "services": self.DEFAULT_SERVICES,
            "contact": {"website": "https://fynorra.com", "email": "info@fynorra.com", "phone": ""},
            "notes": "Public records show incorporation and directors. Verify sensitive details before presenting as legal facts."
        }

        p = Path(business_config_path) if business_config_path else Path(__file__).parent.parent / "config" / "business.yaml"
        if not p.exists():
            return default_info
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                info = {**default_info}
                cp = data.get("company_profile") or data
                info["name"] = cp.get("common_name") or cp.get("company_name") or info["name"]
                info["address"] = cp.get("registered_address") or info["address"]
                info["opening_hours"] = cp.get("opening_hours") or info["opening_hours"]
                if data.get("company_notes", {}) and data["company_notes"].get("founder_override"):
                    info["founder"] = data["company_notes"]["founder_override"]
                else:
                    if cp.get("founder_public_names"):
                        info["founder"] = cp["founder_public_names"][0]
                    elif cp.get("directors"):
                        info["founder"] = cp["directors"][0]
                if "services" in data and isinstance(data["services"], list):
                    services_out = []
                    for s in data["services"]:
                        if isinstance(s, dict):
                            services_out.append({
                                "id": s.get("service_id", s.get("id", "S-?")),
                                "name": s.get("service_name", s.get("name", "Unnamed Service")),
                                "starting_price": s.get("estimated_dev_cost_min_usd", s.get("starting_price", ""))
                            })
                    if services_out:
                        info["services"] = services_out
                if data.get("contact"):
                    info["contact"] = {**info.get("contact", {}), **data.get("contact")}
                return info
        except Exception as e:
            logger.warning("Failed to load business config %s: %s", p, e)
            return default_info

    def _init_agent(self):
        """Initialize ReActAgent safely. If not available we keep react_agent=None."""
        llm_provider = os.getenv("LLM_PROVIDER", "groq").lower()
        try:
            llm = self.llm_service.llm
            if ReActAgent is None:
                logger.warning("ReActAgent class not found; LLM fallback disabled.")
                self.react_agent = None
                return
            self.react_agent = ReActAgent(tools=[], llm=llm, verbose=False)
            logger.info("ReActAgent initialized with provider: %s, model: %s", llm_provider, getattr(self.llm_service, "model", "unknown"))
        except Exception as e:
            logger.exception("Failed to initialize ReActAgent: %s", e)
            self.react_agent = None

    # ---------- Helpers ----------
    def _strip_think_tags(self, text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*\n?", "", text, flags=re.DOTALL).strip()

    def _is_greeting(self, text: str) -> bool:
        if not text:
            return False
        t = text.lower().strip()
        if t in self.GREETINGS:
            return True
        for g in self.GREETINGS:
            if t.startswith(g + " ") or t.startswith(g + "!"):
                return True
        return False

    def _services_text(self) -> str:
        lines = []
        lines.append(f"{self.business_info.get('common_name') or self.business_info.get('name')} offers:")
        for s in self.SERVICES:
            name = s.get("name")
            price = s.get("starting_price") or s.get("estimated_dev_cost_min_usd", "")
            lines.append(f"- {name}" + (f" (from {price})" if price else ""))
        lines.append("Would you like a short demo (20 mins) or a tailored proposal?")
        return " ".join(lines)

    def _conversationalize(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        if len(t.split()) <= 8 and not t.endswith("."):
            return "Sure — " + (t[0].lower() + t[1:] if t and t[0].isupper() else t)
        if t.lower().startswith(("please", "tell", "explain", "show")):
            return "Of course — " + t
        return t

    # ---------- Safe call to LLM (handles coroutine/sync mismatch) ----------
    def _call_react_agent_safe(self, input_text: str, chat_history):
        """
        Try calling several possible ReActAgent methods. Handle coroutine functions and awaitables.
        Use nest_asyncio if necessary to allow calling asyncio.run inside an existing loop.
        """
        if not self.react_agent:
            raise RuntimeError("ReActAgent not initialized or unavailable.")
        agent = self.react_agent
        candidates = ["chat", "run", "invoke", "generate", "__call__", "predict"]

        last_exc = None
        for name in candidates:
            if hasattr(agent, name):
                fn = getattr(agent, name)
                try:
                    if inspect.iscoroutinefunction(fn):
                        try:
                            return asyncio.run(fn(input_text, chat_history=chat_history))
                        except RuntimeError as e:
                            last_exc = e
                            try:
                                import nest_asyncio
                                nest_asyncio.apply()
                                return asyncio.run(fn(input_text, chat_history=chat_history))
                            except Exception as e2:
                                last_exc = e2
                                continue
                    result = fn(input_text, chat_history=chat_history)
                    if inspect.isawaitable(result):
                        try:
                            return asyncio.run(result)
                        except RuntimeError as e:
                            last_exc = e
                            try:
                                import nest_asyncio
                                nest_asyncio.apply()
                                return asyncio.run(result)
                            except Exception as e2:
                                last_exc = e2
                                continue
                    return result
                except Exception as e:
                    last_exc = e
                    logger.debug("Agent method %s raised: %s", name, e)
                    continue

        # fallback: if agent instance is callable
        try:
            if callable(agent):
                res = agent(input_text, chat_history=chat_history)
                if inspect.isawaitable(res):
                    return asyncio.run(res)
                return res
        except Exception as e:
            last_exc = e
            logger.debug("Agent direct-call fallback raised: %s", e)

        logger.error("ReActAgent call failed. Last exception: %s", last_exc)
        raise RuntimeError("ReActAgent has no usable call method.") from last_exc

    # ---------- Public API ----------
    def invoke(self, input_text: str, config: dict | None = None) -> dict:
        """
        Main entrypoint.
        Returns dict with messages and metadata. Always includes "speak": True.
        """
        if config is None:
            config = {"configurable": {"thread_id": "default_user"}}

        logger.info('💭 Thinking about: "%s"', input_text)
        text = (input_text or "").strip()

        # business-info quick answers
        bi = self._answer_business_info(text)
        if bi:
            assistant = self._conversationalize(bi)
            self._store_to_memory(input_text, assistant)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "speak": True}

        # greeting
        if self._is_greeting(text):
            assistant = f"Hi — welcome to {self.business_info.get('common_name') or self.business_info.get('name')}. I'm here to help you explore AI solutions and book a quick demo. How can I assist?"
            assistant = self._conversationalize(assistant)
            self._store_to_memory(input_text, assistant)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "speak": True}

        # ask about services / pitch
        if re.search(r"\b(service|services|offer|what do you do|what can you do)\b", text, flags=re.I):
            assistant = self._services_text()
            assistant = self._conversationalize(assistant)
            self._store_to_memory(input_text, assistant)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "speak": True}

        # pricing / demo booking
        if re.search(r"\b(price|pricing|cost|quote|estimate|demo|book a demo)\b", text, flags=re.I):
            assistant = (
                "We offer a 20-minute demo and a tailored proposal. "
                "Starter pricing is listed for each service, but final quotes depend on scope. "
                "Would you like to book a demo or share a few details so I can prepare a proposal?"
            )
            assistant = self._conversationalize(assistant)
            self._store_to_memory(input_text, assistant)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "speak": True}

        # lead capture intent — look for contact info or ask for it
        if re.search(r"\b(contact|email|phone|call me|reach|lead|interested)\b", text, flags=re.I):
            phone = None
            email = None
            em = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text)
            if em:
                email = em.group(1)
            pm = re.search(r"(\+?\d[\d\-\s]{6,}\d)", text)
            if pm:
                phone = re.sub(r"[^\d+]", "", pm.group(1))
            if email or phone:
                assistant = "Thanks — I've captured your contact. Our sales team will reach out within 48 hours. Would you like to book a demo now?"
                metadata = {}
                if email:
                    metadata["email"] = email
                if phone:
                    metadata["phone"] = phone
                assistant = self._conversationalize(assistant)
                self._store_to_memory(input_text, assistant)
                return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "lead": metadata, "speak": True}
            else:
                assistant = "I'd be glad to connect you. Could I have your name, email and/or phone so our team can reach out?"
                assistant = self._conversationalize(assistant)
                self._store_to_memory(input_text, assistant)
                return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "speak": True}

        # qualification questions (budget/timeline/priority)
        if re.search(r"\b(budget|timeline|when|soon|priority)\b", text, flags=re.I):
            assistant = "Great question — can you share your timeline and an estimated budget range? That helps us propose the right solution."
            assistant = self._conversationalize(assistant)
            self._store_to_memory(input_text, assistant)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant}], "speak": True}

        # fallback -> LLM/ReActAgent for general questions (no RAG)
        try:
            chat_history = self.memory.get()
            raw_resp = self._call_react_agent_safe(text, chat_history=chat_history)
            assistant_response = self._extract_text_from_raw_resp(raw_resp)
            assistant_response = self._strip_think_tags(assistant_response)
            assistant_response = self._conversationalize(assistant_response)
            if not assistant_response.strip():
                assistant_response = self.OFFER_TO_LEARN
            self._store_to_memory(input_text, assistant_response)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant_response}], "speak": True}
        except Exception as e:
            logger.exception("Error generating response via ReActAgent: %s", e)
            assistant_response = self.OFFER_TO_LEARN
            self._store_to_memory(input_text, assistant_response)
            return {"messages": [{"role": "user", "content": input_text}, {"role": "assistant", "content": assistant_response}], "speak": True}

    def _extract_text_from_raw_resp(self, raw_resp) -> str:
        if raw_resp is None:
            return ""
        if hasattr(raw_resp, "response"):
            return str(raw_resp.response)
        if hasattr(raw_resp, "text"):
            return str(getattr(raw_resp, "text"))
        if isinstance(raw_resp, dict):
            if "messages" in raw_resp and isinstance(raw_resp["messages"], list) and raw_resp["messages"]:
                last = raw_resp["messages"][-1]
                if isinstance(last, dict) and "content" in last:
                    return str(last["content"])
                return str(last)
            if "text" in raw_resp:
                return str(raw_resp["text"])
            return str(raw_resp)
        if isinstance(raw_resp, str):
            return raw_resp
        return str(raw_resp)

    def _store_to_memory(self, user_text: str, assistant_text: str):
        try:
            self.memory.put_messages([
                ChatMessage(role="user", content=user_text),
                ChatMessage(role="assistant", content=assistant_text)
            ])
        except Exception:
            logger.debug("Failed to write to memory; continuing.")

    def _answer_business_info(self, text: str) -> str | None:
        t = (text or "").lower()
        # founder / owner
        if re.search(r"\b(founder|owner|who started|who founded|ceo)\b", t):
            founder = self.business_info.get("founder")
            if founder:
                return f"Our founder is {founder}."
            return None
        if re.search(r"\b(name|company name|what(?:'s| is) your name)\b", t):
            return f"Our name is {self.business_info.get('common_name') or self.business_info.get('name')}."
        if re.search(r"\b(address|where are you|located)\b", t):
            return f"We are located at {self.business_info.get('address')}."
        if re.search(r"\b(contact|email|phone|website)\b", t):
            c = self.business_info.get("contact", {})
            web = c.get("website") or "fynorra.ai"
            email = c.get("email") or "sales@fynorra.ai"
            return f"You can reach us at {email} or visit {web}."
        if re.search(r"\b(about (the )?company|tell me about|what(?:'s| is) this company)\b", t):
            return self.business_info.get("description", "")
        return None

    def clear_history(self):
        """Reset chat memory and re-seed the system prompt."""
        self.memory = Memory.from_defaults(token_limit=4000, chat_history_token_ratio=0.8)
        self.memory.put_messages([ChatMessage(role="system", content=self.system_prompt)])
        return True


__all__ = ["Agent"]
