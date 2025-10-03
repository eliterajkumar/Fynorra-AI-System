"""
text-to-speech providers package.
"""
from .provider import ProviderTTS
from .elevenlabs_tts import ElevenLabsTTS


__all__ = ["ProviderTTS", "ElevenLabsTTS"] 