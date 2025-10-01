# agents/restaurant.py
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Annotated, Optional, Dict, List, Tuple, Union

# LLM adapter (may be heavy)
from agents.adapters.groq_llm import GroqLLM

import yaml
from dotenv import load_dotenv
from pydantic import Field

from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import function_tool
from livekit.agents.voice import Agent, AgentSession, RunContext
from livekit.agents.voice.room_io import RoomInputOptions
from livekit.plugins import cartesia, deepgram, openai, silero

# logger
logger = logging.getLogger("restaurant-example")
logger.setLevel(logging.INFO)

load_dotenv()

# voices config (cartesia voice ids)
voices = {
    "greeter": "794f9389-aac1-45b6-b726-9d9369183238",
    "reservation": "156fb8d2-335b-4950-9cb3-a2d33befec77",
    "takeaway": "6f84f4b8-58a2-430c-8c79-688dad597532",
    "checkout": "39b376fc-488e-4d0c-8b37-e00b72059fdd",
}


# ------------- User data (do NOT store CVV) -------------
@dataclass
class UserData:
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None

    reservation_time: Optional[str] = None

    order: Optional[List[str]] = None

    # store only masked/last4, never full PAN or CVV
    card_last4: Optional[str] = None
    card_masked: Optional[str] = None  # e.g. "**** **** **** 1234"

    expense: Optional[float] = None
    checked_out: Optional[bool] = None

    agents: Dict[str, Agent] = field(default_factory=dict)
    prev_agent: Optional[Agent] = None

    def summarize(self) -> str:
        """Return a short YAML summary — redact sensitive fields."""
        data = {
            "customer_name": self.customer_name or "unknown",
            "customer_phone": self.customer_phone or "unknown",
            "reservation_time": self.reservation_time or "unknown",
            "order": self.order or "unknown",
            "card_last4": self.card_last4 or "none",
            "expense": self.expense or "unknown",
            "checked_out": self.checked_out or False,
        }
        
        return yaml.dump(data)


# alias for run context type
RunContext_T = RunContext[UserData]


# ------------- helper function tools -------------
@function_tool()
async def update_name(
    name: Annotated[str, Field(description="The customer's name")],
    context: RunContext_T,
) -> str:
    
    userdata = context.userdata
    userdata.customer_name = name
    logger.info("update_name: %s", name)
    return f"The name is updated to {name}"


@function_tool()
async def update_phone(
    phone: Annotated[str, Field(description="The customer's phone number")],
    context: RunContext_T,
) -> str:
    
    userdata = context.userdata
    userdata.customer_phone = phone
    logger.info("update_phone: %s", phone)
    return f"The phone number is updated to {phone}"


@function_tool()
async def to_greeter(context: RunContext_T) -> Union[Agent, Tuple[Agent, str]]:
    """Route to the greeter agent."""
    curr_agent = context.session.current_agent
    return await curr_agent._transfer_to_agent("greeter", context)


# ------------- BaseAgent with context merging -------------
class BaseAgent(Agent):
    async def on_enter(self) -> None:
        agent_name = self.__class__.__name__
        logger.info("entering task %s", agent_name)

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # merge truncated previous agent chat if available
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add system instructions containing a redacted user summary
        chat_ctx.add_message(
            role="system",
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        # generate an initial reply (non-blocking)
        self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: RunContext_T) -> Tuple[Agent, str]:
        userdata = context.userdata
        current_agent = context.session.current_agent
        next_agent = userdata.agents.get(name)
        
        userdata.prev_agent = current_agent
        
        return next_agent, f"Transferring to {name}."


# ------------- agent implementations -------------
class Greeter(BaseAgent):
    def __init__(self, menu: str) -> None:
        # LLm initialization may be heavy; wrap in try/catch and fallback
        try:
            llm_instance = GroqLLM(model="llama-3.1-70b-versatile")
        except Exception as e:
            logger.warning("GroqLLM init failed, falling back to OpenAI: %s", e)
            try:
                # Use OpenAI adapter if available (lighter)
                llm_instance = openai.OpenAIModel()  # depends on plugin availability
            except Exception as e2:
                logger.error("Fallback LLM init failed: %s", e2)
                llm_instance = None

        # TTS: cartesia preferred; fallback to openai.realtime if needed
        try:
            tts_instance = cartesia.TTS(voice=voices["greeter"])
        except Exception:
            logger.warning("Cartesia TTS not available for greeter; using default TTS")
            tts_instance = None

        super().__init__(
            instructions=(
                f"You are a friendly restaurant receptionist. The menu is: {menu}\n"
                "Your jobs are to greet the caller and understand if they want to "
                "make a reservation or order takeaway. Guide them to the right agent using tools."
            ),
            llm=llm_instance,
            tts=tts_instance,
        )
        self.menu = menu

    @function_tool()
    async def to_reservation(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("reservation", context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context)


class Reservation(BaseAgent):
    def __init__(self) -> None:
        try:
            tts_instance = cartesia.TTS(voice=voices["reservation"])
        except Exception:
            logger.warning("Cartesia reservation TTS init failed")
            tts_instance = None

        super().__init__(
            instructions="You are a reservation agent at a restaurant. Your jobs are to ask for "
            "the reservation time, then customer's name, and phone number. Then "
            "confirm the reservation details with the customer.",
            tools=[update_name, update_phone, to_greeter],
            tts=tts_instance,
        )

    @function_tool()
    async def update_reservation_time(
        self,
        time: Annotated[str, Field(description="The reservation time")],
        context: RunContext_T,
    ) -> str:
        
        userdata = context.userdata
        userdata.reservation_time = time
        return f"The reservation time is updated to {time}"

    @function_tool()
    async def confirm_reservation(self, context: RunContext_T) -> Union[str, Tuple[Agent, str]]:
        userdata = context.userdata
        if not userdata.customer_name or not userdata.customer_phone:
            return "Please provide your name and phone number first."
        
        if not userdata.reservation_time:
            return "Please provide reservation time first."
        
        return await self._transfer_to_agent("greeter", context)


class Takeaway(BaseAgent):
    def __init__(self, menu: str) -> None:
        try:
            tts_instance = cartesia.TTS(voice=voices["takeaway"])
        except Exception:
            logger.warning("Cartesia takeaway TTS init failed")
            tts_instance = None

        super().__init__(
            instructions=(
                f"Your are a takeaway agent that takes orders from the customer. "
                f"Our menu is: {menu}\n"
                "Clarify special requests and confirm the order with the customer."
            ),
            tools=[to_greeter],
            tts=tts_instance,
        )

    @function_tool()
    async def update_order(
        self,
        items: Annotated[List[str], Field(description="The items of the full order")],
        context: RunContext_T,
    ) -> str:
        
        userdata = context.userdata
        userdata.order = items
        return f"The order is updated to {items}"

    @function_tool()
    async def to_checkout(self, context: RunContext_T) -> Union[str, Tuple[Agent, str]]:
        userdata = context.userdata
        if not userdata.order:
            return "No takeaway order found. Please make an order first."
        
        return await self._transfer_to_agent("checkout", context)


class Checkout(BaseAgent):
    def __init__(self, menu: str) -> None:
        try:
            tts_instance = cartesia.TTS(voice=voices["checkout"])
        except Exception:
            logger.warning("Cartesia checkout TTS init failed")
            tts_instance = None

        super().__init__(
            instructions=(
                f"You are a checkout agent at a restaurant. The menu is: {menu}\n"
                "Your are responsible for confirming the expense of the "
                "order and then collecting customer's name, phone number and credit card "
                "information. Do NOT store CVV. Store only last4 and masked value."
            ),
            tools=[update_name, update_phone, to_greeter],
            tts=tts_instance,
        )

    @function_tool()
    async def confirm_expense(
        self,
        expense: Annotated[float, Field(description="The expense of the order")],
        context: RunContext_T,
    ) -> str:
        
        userdata = context.userdata
        userdata.expense = expense
        return f"The expense is confirmed to be {expense}"

    @function_tool()
    async def update_credit_card(
        self,
        number: Annotated[str, Field(description="The credit card number")],
        expiry: Annotated[str, Field(description="The expiry date of the credit card")],
        cvv: Annotated[Optional[str], Field(description="The CVV of the credit card (do not store)")] = None,
        context: RunContext_T = None,
    ) -> str:
        """
        Store only masked/last4 and do NOT persist CVV.
        Prefer integrating a PCI-compliant payment gateway for real payments.
        """
        userdata = context.userdata
        if number and len(number) >= 4:
            last4 = number[-4:]
            userdata.card_last4 = last4
            userdata.card_masked = ("*" * (len(number) - 4)) + last4
        else:
            userdata.card_last4 = None
            userdata.card_masked = None

        # Do NOT keep CVV
        logger.info("Received credit card info for user; stored masked PAN ending %s", userdata.card_last4)
        return f"Received card ending with {userdata.card_last4}"

    @function_tool()
    async def confirm_checkout(self, context: RunContext_T) -> Union[str, Tuple[Agent, str]]:
        userdata = context.userdata
        if not userdata.expense:
            return "Please confirm the expense first."

        if not userdata.card_last4:
            return "Please provide card info first."

        userdata.checked_out = True
        # After successful checkout, hand back to greeter
        return await to_greeter(context)

    @function_tool()
    async def to_takeaway(self, context: RunContext_T) -> Tuple[Agent, str]:
        return await self._transfer_to_agent("takeaway", context)


# ------------- entrypoint -------------
async def entrypoint(ctx: JobContext):
    menu = "Pizza: $10, Salad: $5, Ice Cream: $3, Coffee: $2"
    userdata = UserData()
    userdata.agents.update(
        {
            "greeter": Greeter(menu),
            "reservation": Reservation(),
            "takeaway": Takeaway(menu),
            "checkout": Checkout(menu),
        }
    )

    # Safe init of AgentSession components: use try/except so worker doesn't crash on missing libs
    try:
        stt_impl = deepgram.STT()
    except Exception as e:
        logger.warning("Deepgram STT init failed: %s. Falling back to None.", e)
        stt_impl = None

    # LLm instance: we already attempted to init inside Greeter for safer fallback,
    # but here we still provide a default for the session (None allowed).
    try:
        default_llm = GroqLLM(model="llama-3.1-70b-versatile")
    except Exception as e:
        logger.warning("GroqLLM init at session-level failed: %s. Proceeding with None LLM (agents may set their own).", e)
        default_llm = None

    try:
        tts_impl_default = cartesia.TTS()
    except Exception as e:
        logger.warning("Cartesia global TTS init failed: %s. Proceeding with None TTS.", e)
        tts_impl_default = None

    try:
        vad_impl = silero.VAD.load()
    except Exception as e:
        logger.warning("Silero VAD load failed: %s. Proceeding with None VAD.", e)
        vad_impl = None

    session = AgentSession[UserData](
        userdata=userdata,
        stt=stt_impl,
        llm=default_llm,
        tts=tts_impl_default,
        vad=vad_impl,
        max_tool_steps=5,
    
    )

    # start session and join LiveKit room
    try:
        await session.start(
            agent=userdata.agents["greeter"],
            room=ctx.room,
            room_input_options=RoomInputOptions(),
        )
    except Exception as e:
        logger.exception("Failed to start AgentSession: %s", e)
        raise


# CLI run when file invoked directly
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
