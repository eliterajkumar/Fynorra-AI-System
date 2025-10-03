"""
speech service module for tts and stt operations.
"""
import os
from typing import Generator, Tuple
import numpy as np
from loguru import logger
from dotenv import load_dotenv

from .tts import ElevenLabsTTS
from .stt import GroqSTT

load_dotenv()


class SpeechService:
    """
    handles all speech-related operations including text-to-speech and speech-to-text.
    This version is streamlined to use only ElevenLabs for TTS and Groq for STT.
    """

    def __init__(self):
        """
        initialize the speech service.
        """
        # initialize tts provider
        self.tts_provider = ElevenLabsTTS()
        logger.debug("Speech service initialized with ElevenLabs TTS provider")
        
        # initialize stt provider
        self.stt_provider = GroqSTT()
        logger.debug("Speech service initialized with Groq STT provider")
        
        # always preload tts model to reduce initial latency
        self.preload_tts()

    def preload_tts(self) -> None:
        """
        preload the active tts provider to reduce latency on first use.
        """
        if not hasattr(self.tts_provider, 'initialized') or not self.tts_provider.initialized:
            logger.info("Preloading ElevenLabs TTS model...")
            self.tts_provider.initialize()
            self.tts_provider.initialized = True
            logger.info("ElevenLabs TTS model preloaded successfully")

    def text_to_speech(
        self,
        text: str,
        voice_id: str = None,
        model_id: str = None,
        output_format: str = "mp3_44100_128",
        language: str = None,
        **kwargs
    ) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        convert text to speech using the active tts provider.
        
        args:
            text: text to synthesize
            voice_id: voice id or name (provider-specific)
            model_id: model id (provider-specific)
            output_format: output audio format (provider-specific)
            language: language code (provider-specific)
            
        yields:
            a tuple of (sample_rate, audio_array) for audio playback
        """
        if not text:
            logger.warning("empty text provided to text_to_speech")
            return
            
        # model should already be initialized, but check just in case
        if not hasattr(self.tts_provider, 'initialized') or not self.tts_provider.initialized:
            self.tts_provider.initialize()
            self.tts_provider.initialized = True
            
        logger.debug(f"converting text to speech using ElevenLabs, length: {len(text)}")
        
        yield from self.tts_provider.text_to_speech(
            text=text,
            voice_id=voice_id,
            model_id=model_id,
            output_format=output_format,
            language=language,
            **kwargs
        )

    def speech_to_text(
        self,
        audio: tuple[int, np.ndarray],
        model_id: str = None,
        language_code: str = None,
        prompt: str = None,
        temperature: float = 0,
        response_format: str = "text",
        **kwargs
    ) -> str:
        """
        convert speech to text using the active stt provider.
        
        args:
            audio: tuple containing sample rate and audio data
            model_id: model id (provider-specific)
            language_code: language code (provider-specific)
            prompt: optional prompt for context or spelling (groq and openai only)
            temperature: sampling temperature (groq and openai only)
            response_format: output format (groq and openai only)
            
        returns:
            transcribed text
        """
        if not audio or len(audio) != 2:
            logger.warning("invalid audio provided to speech_to_text")
            return ""
            
        # lazy initialization of provider
        if not hasattr(self.stt_provider, 'initialized') or not self.stt_provider.initialized:
            self.stt_provider.initialize()
            self.stt_provider.initialized = True
            
        provider_kwargs = kwargs.copy()
        provider_kwargs.update({
            "prompt": prompt,
            "temperature": temperature,
            "response_format": response_format
        })
            
        return self.stt_provider.speech_to_text(
            audio=audio,
            model_id=model_id,
            language_code=language_code,
            **provider_kwargs
        )
