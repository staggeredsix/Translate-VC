"""
Text-to-speech service.
Handles conversion of text to speech using the Coqui TTS model.
"""

import os
import torch
import logging
import tempfile
import soundfile as sf
import numpy as np
from TTS.api import TTS

from .. import config
from ..utils.audio_utils import normalize_audio, clean_temp_file
from ..utils.logging_utils import PerformanceLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("tts")

class TTSService:
    """Service for text-to-speech conversion"""
    
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(config.MODELS_DIR, "tts")
        self.device = config.DEVICE
        
        # Choose appropriate model based on GPU capabilities
        if config.ADA_OPTIMIZED:
            # VITS is better optimized for newer GPUs
            self.model_name = "tts_models/en/vctk/vits"
        else:
            # Tacotron2 works well on older hardware
            self.model_name = "tts_models/en/ljspeech/tacotron2-DDC"
    
    def initialize(self):
        """Initialize the TTS model"""
        if self.model:
            return self.model
            
        logger.info(f"Initializing TTS model: {self.model_name}")
        perf_logger.start_timer("tts_init")
        
        self.model = TTS(
            model_name=self.model_name, 
            progress_bar=False, 
            gpu=self.device == "cuda"
        )
            
        duration = perf_logger.end_timer("tts_init")
        logger.info(f"TTS model initialized in {duration:.2f}s")
        
        return self.model
    
    @torch.no_grad()
    def synthesize(self, text):
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            tuple: (audio_data, sample_rate) or None if failed
        """
        if not text.strip():
            return None
            
        # Initialize if needed
        if self.model is None:
            self.initialize()
            
        perf_logger.start_timer("synthesize")
        temp_file = None
        
        try:
            # Create a temporary file for the audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as out:
                temp_file = out.name
                
                # For Ada GPUs, use optimized TTS generation
                if config.ADA_OPTIMIZED and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=config.ADA_OPTIMIZED):
                        self.model.tts_to_file(text=text, file_path=temp_file)
                else:
                    self.model.tts_to_file(text=text, file_path=temp_file)
                
                # Read the audio file
                audio_bytes, sample_rate = sf.read(temp_file, dtype="float32")
                
                # Convert to int16 for better compatibility
                audio_int16 = normalize_audio(audio_bytes, "int16")
                
                duration = perf_logger.end_timer("synthesize")
                logger.debug(f"Speech synthesis completed in {duration:.3f}s - {len(text)} chars")
                
                return (sample_rate, audio_int16)
                
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            return None
        finally:
            # Clean up temporary file
            if temp_file:
                clean_temp_file(temp_file)
    
    def list_available_models(self):
        """
        List available TTS models.
        
        Returns:
            list: Available model names
        """
        if self.model is None:
            self.initialize()
            
        try:
            return self.model.list_models()
        except:
            # Fallback to common models
            return [
                "tts_models/en/ljspeech/tacotron2-DDC",
                "tts_models/en/vctk/vits",
                "tts_models/en/ljspeech/fast_pitch",
                "tts_models/en/ljspeech/glow-tts"
            ]
    
    def change_model(self, model_name):
        """
        Change the TTS model.
        
        Args:
            model_name: New model name
            
        Returns:
            bool: Success status
        """
        if model_name == self.model_name:
            return True
            
        try:
            logger.info(f"Changing TTS model from {self.model_name} to {model_name}")
            self.model_name = model_name
            self.model = None  # Force reinitialization
            self.initialize()
            return True
        except Exception as e:
            logger.error(f"Failed to change TTS model: {e}")
            return False

# Singleton instance
_instance = None

def get_tts_service():
    """Get the singleton TTSService instance"""
    global _instance
    if _instance is None:
        _instance = TTSService()
        _instance.initialize()
    return _instance