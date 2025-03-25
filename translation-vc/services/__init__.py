"""
Services package initialization.
Provides access to service objects.
"""

# Import service getter functions for easier access
from .whisper_service import get_whisper_service
from .translation_service import get_translation_service
from .tts_service import get_tts_service
from .verification_service import get_verification_service
from .broadcaster_service import get_broadcaster_service
from .user_profile_service import get_user_profile_service

# Initialize all services
def initialize_services(force=False):
    """
    Initialize all services.
    
    Args:
        force: Whether to force initialization even if already initialized
        
    Returns:
        dict: Dictionary of service instances
    """
    whisper = get_whisper_service()
    translation = get_translation_service()
    tts = get_tts_service()
    verification = get_verification_service()
    broadcaster = get_broadcaster_service()
    user_profile = get_user_profile_service()
    
    # Force initialization if requested
    if force:
        whisper.initialize()
        translation.initialize()
        tts.initialize()
        if verification:
            verification.initialize()
    
    return {
        "whisper": whisper,
        "translation": translation,
        "tts": tts,
        "verification": verification,
        "broadcaster": broadcaster,
        "user_profile": user_profile
    }