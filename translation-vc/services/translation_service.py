"""
Translation service.
Handles text translation between languages with verification capabilities.
"""

import os
import torch
import logging
from functools import lru_cache
from transformers import MarianMTModel, MarianTokenizer
import time

import config
from utils.logging_utils import PerformanceLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("translation")

class TranslationService:
    """Service for text translation using MarianMT models"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_path = os.path.join(config.MODELS_DIR, "marian")
        self.model_name = "Helsinki-NLP/opus-mt-en-ROMANCE"
        self.device = config.DEVICE
        
    def initialize(self):
        """Initialize the translation model and tokenizer"""
        if self.model and self.tokenizer:
            return self.model, self.tokenizer
            
        logger.info(f"Initializing translation model: {self.model_name}")
        perf_logger.start_timer("translation_init")
        
        # Initialize tokenizer with optimizations
        self.tokenizer = MarianTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.model_path,
            use_fast=True  # Use faster tokenizer implementation
        )
        
        # Initialize model with Ada-specific optimizations
        if config.ADA_OPTIMIZED:
            # Load with torch_dtype=torch.float16 directly for Ada GPUs
            self.model = MarianMTModel.from_pretrained(
                self.model_name,
                cache_dir=self.model_path,
                torch_dtype=torch.float16,  # Use FP16 for Ada GPUs
                device_map="auto"  # Automatically map to available devices
            )
        else:
            self.model = MarianMTModel.from_pretrained(
                self.model_name,
                cache_dir=self.model_path
            ).to(self.device)
            
        duration = perf_logger.end_timer("translation_init")
        logger.info(f"Translation model initialized in {duration:.2f}s")
        
        return self.model, self.tokenizer
        
    @lru_cache(maxsize=128)
    def translate_cached(self, text, source_lang, target_lang):
        """
        Cached wrapper for translate to avoid redundant translations.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            str: Translated text
        """
        return self.translate(text, source_lang, target_lang)
        
    @torch.no_grad()
    def translate(self, text, source_lang, target_lang):
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            str: Translated text
        """
        # Skip if text is empty or languages are the same
        if not text or source_lang == target_lang:
            return text
            
        # Initialize if needed
        if self.model is None or self.tokenizer is None:
            self.initialize()
            
        perf_logger.start_timer("translate")
        
        try:
            # For Ada GPUs, use context manager to enable mixed precision
            if config.ADA_OPTIMIZED and hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    translation_result = self._perform_translation(text, source_lang, target_lang)
            else:
                translation_result = self._perform_translation(text, source_lang, target_lang)
            
            duration = perf_logger.end_timer("translate")
            logger.debug(f"Translation completed in {duration:.3f}s - {len(text)} → {len(translation_result)} chars")
            
            return translation_result
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return text  # Return original text if translation fails
            
    def _perform_translation(self, text, source_lang, target_lang):
        """
        Internal function to perform the actual translation.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            str: Translated text
        """
        # Prepare input for translation
        input_text = f">>{target_lang}<< {text}"
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        
        # Move inputs to the same device as model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate translation with optimized settings for Ada GPUs
        generation_kwargs = {
            "max_length": 512,
            "early_stopping": True,
        }
        
        if config.ADA_OPTIMIZED:
            # Ada-specific optimizations
            generation_kwargs.update({
                "num_beams": 5,         # Slightly more beams for better quality
                "length_penalty": 0.7,   # Favor slightly longer translations
                "do_sample": False,      # Deterministic generation is faster
            })
        else:
            # Standard settings
            generation_kwargs.update({
                "num_beams": 4,
                "length_penalty": 0.6,
            })
        
        # Generate translation
        translated = self.model.generate(**inputs, **generation_kwargs)
        translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        # If translation verification is enabled, verify and possibly retry
        if config.TRANSLATION_VERIFICATION:
            verification_service = get_verification_service()
            if verification_service and len(text.split()) >= 5:
                # Only verify non-trivial translations
                verification_result = verification_service.verify_translation(
                    original_text=text,
                    translated_text=translated_text,
                    source_lang=source_lang,
                    target_lang=target_lang
                )
                
                # If quality is poor, retry with different parameters
                if verification_result.get("quality", "good") == "poor":
                    logger.warning(f"Poor translation detected (BLEU: {verification_result.get('score', 0):.2f}), retrying")
                    
                    # Retry with different parameters
                    retry_kwargs = generation_kwargs.copy()
                    retry_kwargs.update({
                        "num_beams": 8,          # More beams for better quality
                        "length_penalty": 0.8,   # Favor completeness
                        "do_sample": True,       # Enable sampling
                        "top_k": 50,             # Diverse generation
                        "temperature": 0.8       # Slightly random
                    })
                    
                    # Generate improved translation
                    translated = self.model.generate(**inputs, **retry_kwargs)
                    translated_text = self.tokenizer.decode(translated[0], skip_special_tokens=True)
        
        return translated_text

# Singleton instance
_instance = None

def get_translation_service():
    """Get the singleton TranslationService instance"""
    global _instance
    if _instance is None:
        _instance = TranslationService()
        _instance.initialize()
    return _instance

# Import here to avoid circular imports
from .verification_service import get_verification_service