"""
Translation verification service.
Provides quality verification for translations using back-translation and BLEU scoring.
"""

import os
import torch
import logging
from transformers import pipeline
from sacrebleu.metrics import BLEU
import nltk

import config
from utils.logging_utils import PerformanceLogger

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("verification")

# Ensure NLTK data is available
try:
    nltk.download('punkt', quiet=True)
except:
    logger.warning("Failed to download NLTK data, verification may be limited")

class VerificationService:
    """Service for verifying translation quality"""
    
    def __init__(self):
        self.verification_pipeline = None
        self.bleu_scorer = None
        self.model_path = os.path.join(config.MODELS_DIR, "verification")
        self.enabled = config.TRANSLATION_VERIFICATION
        self.quality_threshold = 30  # BLEU score threshold for "good" quality
        
    def initialize(self):
        """Initialize the verification models"""
        if not self.enabled:
            logger.info("Translation verification is disabled")
            return None
            
        if self.verification_pipeline and self.bleu_scorer:
            return self.verification_pipeline
            
        try:
            logger.info("Loading translation verification model...")
            perf_logger.start_timer("verification_init")
            
            # For Ada GPUs, we can use a more efficient pipeline with batching
            if config.ADA_OPTIMIZED:
                self.verification_pipeline = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-ROMANCE-en",  # Back-translation model
                    device=config.DEVICE,
                    torch_dtype=torch.float16,
                    batch_size=config.BATCH_SIZE
                )
            else:
                self.verification_pipeline = pipeline(
                    "translation",
                    model="Helsinki-NLP/opus-mt-ROMANCE-en",
                    device=config.DEVICE
                )
                
            # Initialize BLEU scorer for translation quality
            self.bleu_scorer = BLEU()
            
            duration = perf_logger.end_timer("verification_init")
            logger.info(f"Translation verification model loaded in {duration:.2f}s")
            
            return self.verification_pipeline
            
        except Exception as e:
            logger.warning(f"Failed to load verification model: {e}")
            self.enabled = False
            return None
    
    def enable(self, enabled=True):
        """
        Enable or disable verification.
        
        Args:
            enabled: Whether verification should be enabled
            
        Returns:
            bool: Current enabled state
        """
        self.enabled = enabled
        config.update_config("TRANSLATION_VERIFICATION", enabled)
        
        # Initialize model if enabling
        if enabled and not self.verification_pipeline:
            self.initialize()
            
        return self.enabled
    
    @torch.no_grad()
    def verify_translation(self, original_text, translated_text, source_lang, target_lang):
        """
        Verify translation quality using back-translation and BLEU scoring.
        
        Args:
            original_text: Original source text
            translated_text: Translated target text
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            dict: Verification results including score and quality assessment
        """
        # Skip verification if disabled or models not loaded
        if not self.enabled or not self.verification_pipeline or not self.bleu_scorer:
            return {"quality": "unknown", "score": None}
            
        # Skip verification for short texts
        if len(original_text.split()) < 5:
            return {"quality": "unknown", "score": None, "reason": "text_too_short"}
            
        perf_logger.start_timer("verify")
        
        try:
            # Back-translate to check quality
            back_translation = self.verification_pipeline(
                translated_text, 
                max_length=512
            )[0]['translation_text']
            
            # Calculate BLEU score between original and back-translated text
            bleu_score = self.bleu_scorer.sentence_score(back_translation, [original_text]).score
            
            # Determine quality based on threshold
            quality = "good" if bleu_score >= self.quality_threshold else "poor"
            
            duration = perf_logger.end_timer("verify")
            logger.debug(f"Translation verification completed in {duration:.3f}s, score: {bleu_score:.2f}, quality: {quality}")
            
            return {
                "quality": quality,
                "score": bleu_score,
                "back_translation": back_translation
            }
            
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return {"quality": "unknown", "score": None, "error": str(e)}

# Singleton instance
_instance = None

def get_verification_service():
    """Get the singleton VerificationService instance"""
    global _instance
    if _instance is None:
        _instance = VerificationService()
        if config.TRANSLATION_VERIFICATION:
            _instance.initialize()
    return _instance