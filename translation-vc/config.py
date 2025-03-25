# Configuration and constants for the multilingual voice chat application

import os
import torch

# Performance Constants
SAMPLE_RATE = 16000
USER_PROFILE_PATH = "user_profiles.json"
WHISPER_MODEL_SIZE = "base"  # Default model size
CHUNK_SIZE = 50  # Number of audio chunks to process at once
BATCH_SIZE = 16  # For batch processing where applicable
TRANSLATION_VERIFICATION = True  # Enable translation verification by default
ENABLE_MICROPHONE_BY_DEFAULT = False  # Microphone disabled by default
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# GPU Settings
# These are initialized in gpu_utils.py and then imported back here
DEVICE = "cpu"  # Will be set to "cuda" if available
GPU_ARCHITECTURE = None
ADA_OPTIMIZED = False
USE_FLASH_ATTENTION = False

# Language map
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese": "zh",
    "Arabic": "ar"
}

# Update function to modify global config variables at runtime
def update_config(key, value):
    """
    Update a configuration value at runtime
    Args:
        key: The config variable name (string)
        value: The new value to set
    """
    if key in globals():
        globals()[key] = value
        return True
    return False

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# These will be set by gpu_utils.py based on hardware detection
def update_gpu_settings(device, gpu_arch, ada_optimized, use_flash_attn):
    """Update GPU settings after hardware detection"""
    global DEVICE, GPU_ARCHITECTURE, ADA_OPTIMIZED, USE_FLASH_ATTENTION
    DEVICE = device
    GPU_ARCHITECTURE = gpu_arch
    ADA_OPTIMIZED = ada_optimized
    USE_FLASH_ATTENTION = use_flash_attn