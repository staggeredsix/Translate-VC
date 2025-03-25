"""
Whisper speech-to-text service.
Handles speech transcription using the Whisper model.
"""

import os
import torch
import logging
import numpy as np
from faster_whisper import WhisperModel
from functools import lru_cache

from .. import config
from ..utils.audio_utils import save_audio_to_temp, normalize_audio, clean_temp_file
from ..utils.logging_utils import PerformanceLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("whisper")

class WhisperService:
    """Service for speech recognition using Whisper model"""
    
    def __init__(self):
        self.model = None
        self.model_size = config.WHISPER_MODEL_SIZE
        self.device = config.DEVICE
        self.model_path = os.path.join(config.MODELS_DIR, "whisper")
        
    def initialize(self):
        """Initialize the Whisper model with appropriate settings"""
        if self.model:
            return self.model
            
        logger.info(f"Initializing Whisper model (size: {self.model_size})")
        perf_logger.start_timer("whisper_init")
        
        compute_type = "float16" if self.device == "cuda" else "int8"
        
        # Use more aggressive optimizations for Ada GPUs
        if config.ADA_OPTIMIZED:
            self.model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type=compute_type,
                download_root=self.model_path,
                cpu_threads=8,  # Higher CPU thread count for pre/post processing
                num_workers=4,  # Parallel workers for data loading
                local_files_only=False,  # Use local cache if available
            )
        else:
            self.model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type=compute_type,
                download_root=self.model_path
            )
            
        duration = perf_logger.end_timer("whisper_init")
        logger.info(f"Whisper model initialized in {duration:.2f}s")
        
        return self.model
        
    def update_model_size(self, new_size):
        """
        Update the model size and reinitialize.
        
        Args:
            new_size: New model size (tiny, base, small, medium, large)
            
        Returns:
            bool: Success status
        """
        if new_size == self.model_size:
            return True
            
        valid_sizes = ["tiny", "base", "small", "medium", "large"]
        if new_size not in valid_sizes:
            logger.error(f"Invalid Whisper model size: {new_size}")
            return False
            
        logger.info(f"Changing Whisper model size from {self.model_size} to {new_size}")
        self.model_size = new_size
        self.model = None  # Force reinitialization
        config.update_config("WHISPER_MODEL_SIZE", new_size)
        return True
        
    @torch.no_grad()
    def transcribe(self, audio_input, language=None, vad_filter=True):
        """
        Transcribe audio to text.
        
        Args:
            audio_input: Audio data (numpy array) or path to audio file
            language: Language code or None for auto-detection
            vad_filter: Whether to apply voice activity detection filtering
            
        Returns:
            tuple: (transcribed_text, detected_language)
        """
        if self.model is None:
            self.initialize()
            
        perf_logger.start_timer("transcribe")
        
        # Process different input types
        temp_file = None
        try:
            if isinstance(audio_input, np.ndarray):
                # Normalize and save to temporary file
                audio_input = normalize_audio(audio_input, "float32")
                temp_file = save_audio_to_temp(audio_input)
                audio_path = temp_file
            elif isinstance(audio_input, str) and os.path.exists(audio_input):
                # Use existing file path
                audio_path = audio_input
            else:
                raise ValueError("Invalid audio input type")
                
            # Configure transcription settings
            whisper_settings = {
                'beam_size': 5 if config.ADA_OPTIMIZED else 4,
                'language': language,  # None for auto-detection
                'vad_filter': vad_filter,
                'vad_parameters': dict(min_silence_duration_ms=500)
            }
            
            # Run transcription with appropriate precision
            if config.ADA_OPTIMIZED and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    segments, info = self.model.transcribe(audio_path, **whisper_settings)
            else:
                segments, info = self.model.transcribe(audio_path, **whisper_settings)
                
            # Extract text and language
            text = " ".join([seg.text for seg in segments])
            detected_language = info.get("language", "en")
            
            duration = perf_logger.end_timer("transcribe")
            logger.debug(f"Transcription completed in {duration:.3f}s - {len(text)} chars, language: {detected_language}")
            
            return text, detected_language
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", "en"
        finally:
            # Clean up temporary file
            if temp_file:
                clean_temp_file(temp_file)
                
    @lru_cache(maxsize=32)
    def get_available_languages(self):
        """
        Get list of languages supported by the model.
        
        Returns:
            list: Available language codes
        """
        if self.model is None:
            self.initialize()
            
        try:
            return self.model.tokenizer.languages
        except:
            # Fallback to common languages if model doesn't expose language list
            return ["en", "fr", "de", "es", "it", "pt", "ru", "zh", "ja", "ar"]

# Singleton instance
_instance = None

def get_whisper_service():
    """Get the singleton WhisperService instance"""
    global _instance
    if _instance is None:
        _instance = WhisperService()
        _instance.initialize()
    return _instance