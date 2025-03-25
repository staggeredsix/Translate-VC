"""
Audio processing utilities.
Handles audio file operations, format conversions, and audio analysis.
"""

import numpy as np
import soundfile as sf
import tempfile
import os
import logging
from .. import config

logger = logging.getLogger(__name__)

def normalize_audio(audio_data, target_type="float32"):
    """
    Normalize audio data to a specific type.
    
    Args:
        audio_data: Input audio as numpy array
        target_type: Target data type ('float32' or 'int16')
        
    Returns:
        numpy.ndarray: Normalized audio data
    """
    if audio_data.dtype == np.int16 and target_type == "float32":
        # Convert int16 to float32 (-1.0 to 1.0)
        return audio_data.astype(np.float32) / 32768.0
    elif audio_data.dtype == np.float32 and target_type == "int16":
        # Convert float32 to int16
        return (audio_data * 32767).astype(np.int16)
    elif audio_data.dtype != np.dtype(target_type):
        # General case conversion
        return audio_data.astype(target_type)
    return audio_data  # Already the right type

def save_audio_to_temp(audio_data, sample_rate=None):
    """
    Save audio data to a temporary file.
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate (default: uses config.SAMPLE_RATE)
        
    Returns:
        str: Path to the temporary file
    """
    if sample_rate is None:
        sample_rate = config.SAMPLE_RATE
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        sf.write(tmp.name, audio_data, sample_rate)
        return tmp.name

def load_audio_from_file(file_path):
    """
    Load audio data from a file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        tuple: (audio_data, sample_rate)
    """
    return sf.read(file_path)

def detect_voice_activity(audio_chunk, threshold=0.3):
    """
    Detect voice activity in an audio chunk.
    
    Args:
        audio_chunk: Audio data as numpy array
        threshold: Voice activity detection threshold (0.0-1.0)
        
    Returns:
        bool: True if voice activity detected, False otherwise
    """
    # Simple energy-based VAD
    audio_level = np.abs(audio_chunk).mean()
    return audio_level > threshold

def adaptive_vad_threshold(recent_levels, current_threshold):
    """
    Calculate an adaptive VAD threshold based on recent audio levels.
    
    Args:
        recent_levels: List of recent audio energy levels
        current_threshold: Current VAD threshold
        
    Returns:
        float: Updated threshold value
    """
    if not recent_levels or len(recent_levels) < 10:
        return current_threshold
        
    sorted_levels = sorted(recent_levels)
    noise_floor = sorted_levels[int(len(sorted_levels) * 0.1)]  # 10th percentile as noise floor
    
    # Set threshold at 2x the noise floor, within reasonable bounds
    adaptive_threshold = max(0.15, min(0.8, noise_floor * 2.0))
    
    # Smooth threshold changes
    new_threshold = current_threshold * 0.9 + adaptive_threshold * 0.1
    
    return new_threshold

def clean_temp_file(file_path):
    """
    Clean up a temporary file if it exists.
    
    Args:
        file_path: Path to the file to remove
    """
    if file_path and os.path.exists(file_path):
        try:
            os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to remove temporary file {file_path}: {e}")

def concatenate_audio_chunks(chunks):
    """
    Concatenate a list of audio chunks into a single array.
    
    Args:
        chunks: List of audio chunks (numpy arrays)
        
    Returns:
        numpy.ndarray: Concatenated audio data
    """
    if not chunks:
        return np.array([])
        
    return np.concatenate(chunks, axis=0)