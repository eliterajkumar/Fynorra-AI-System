"""
LLM service module for handling interactions with language models.
"""
import os
from loguru import logger
import litellm
from llama_index.llms.litellm import LiteLLM


class LLMService:
    """
    Service for interacting with Language Models through LiteLLM.
    This version is streamlined to only use Groq.
    """

    def __init__(self):
        """Initialize the LLM service with configuration from environment variables."""
        self.llm_provider = "groq"
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.5"))
        
        self.model = "groq/llama-3.1-8b-instant"

        # initialize LiteLLM instance
        self.llm = LiteLLM(model=self.model, temperature=self.temperature)
        
        logger.debug(f"LLM Service initialized with provider: {self.llm_provider}, model: {self.model}")

    def generate_response(self, messages):
        """
        Generate a response from the LLM based on the given messages.
        
        Args:
            messages (list): List of message dictionaries with role and content
            
        Returns:
            str: The generated response text
            
        Raises:
            Exception: If the model attempt fails
        """
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if not os.getenv("GROQ_API_KEY"):
            logger.warning("GROQ_API_KEY environment variable not set")
        
        logger.debug(f"Generating response with model: {self.model}")
        
        try:
            response = litellm.completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error with model: {str(e)}")
            return "i'm sorry, but i'm having trouble connecting to the server. please try again later."
