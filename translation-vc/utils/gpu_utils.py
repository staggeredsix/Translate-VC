"""
GPU detection and optimization utilities.
Handles NVIDIA GPU detection and optimizations, especially for Ada Lovelace architecture.
"""

import os
import torch
import logging

from .. import config

logger = logging.getLogger(__name__)

def detect_gpu():
    """
    Detect GPU and its capabilities, with special handling for Ada Lovelace architecture.
    Updates the config with appropriate settings.
    
    Returns:
        tuple: (device, gpu_architecture, ada_optimized, flash_attention)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_architecture = None
    ada_optimized = False
    use_flash_attention = False
    
    if device == "cuda":
        # Set CUDA device to first available GPU
        torch.cuda.set_device(0)
        
        # Get GPU properties
        gpu_name = torch.cuda.get_device_name(0)
        gpu_capability = torch.cuda.get_device_capability(0)
        logger.info(f"Using GPU: {gpu_name} with compute capability {gpu_capability}")
        
        # Detect Ada Lovelace architecture (RTX 40 series)
        # Ada Lovelace uses compute capability 8.9
        if gpu_capability[0] >= 8 and gpu_capability[1] >= 9:
            gpu_architecture = "Ada Lovelace"
            ada_optimized = True
            use_flash_attention = True
            logger.info(f"Detected Ada Lovelace architecture - enabling specialized optimizations")
        elif "RTX 40" in gpu_name:
            gpu_architecture = "Ada Lovelace"
            ada_optimized = True
            use_flash_attention = True
            logger.info(f"Detected RTX 40 series GPU - enabling Ada optimizations")
        else:
            gpu_architecture = "Other NVIDIA"
            logger.info(f"Using standard CUDA optimizations")
    else:
        logger.warning("No GPU detected, using CPU. Performance will be significantly slower.")
    
    # Update the configuration
    config.update_gpu_settings(device, gpu_architecture, ada_optimized, use_flash_attention)
    
    return device, gpu_architecture, ada_optimized, use_flash_attention

def optimize_cuda():
    """
    Apply CUDA optimizations based on the detected GPU.
    More aggressive optimizations are applied for Ada Lovelace GPUs.
    """
    if config.DEVICE != "cuda":
        return
        
    # Standard CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 on Ampere+ GPUs
    torch.backends.cudnn.allow_tf32 = True
    
    # Ada-specific optimizations
    if config.ADA_OPTIMIZED:
        # Ada GPUs have better FP16 performance
        torch.set_float32_matmul_precision('high')
        
        # Special Ada optimizations
        os.environ["CUDA_MODULE_LOADING"] = "LAZY"  # Lazy-load CUDA modules
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8 API
        
        # Performance tuning for Ada architecture
        if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
            logger.info("Enabling automatic mixed precision for Ada GPUs")

def warm_up_gpu():
    """
    Pre-warm GPU to avoid cold start performance issues.
    Runs a small tensor operation to initialize CUDA kernels.
    """
    if config.DEVICE != "cuda":
        return
        
    try:
        # Run a small operation to initialize CUDA
        dummy_input = torch.zeros((1, 80), device=config.DEVICE)
        dummy_output = torch.nn.functional.relu(dummy_input)
        torch.cuda.synchronize()
        logger.debug("GPU warmed up successfully")
    except Exception as e:
        logger.warning(f"GPU warm-up failed: {e}")

def cleanup_gpu():
    """
    Clean up GPU resources to free memory.
    """
    if config.DEVICE != "cuda":
        return
        
    try:
        torch.cuda.empty_cache()
        logger.debug("GPU memory cache emptied")
    except Exception as e:
        logger.warning(f"GPU cleanup failed: {e}")

def gpu_memory_stats():
    """
    Get GPU memory usage statistics.
    
    Returns:
        dict: GPU memory statistics or None if no GPU
    """
    if config.DEVICE != "cuda":
        return None
        
    try:
        return {
            "allocated": torch.cuda.memory_allocated(0) / (1024 ** 3),  # GB
            "cached": torch.cuda.memory_reserved(0) / (1024 ** 3),  # GB
            "max_allocated": torch.cuda.max_memory_allocated(0) / (1024 ** 3),  # GB
        }
    except Exception as e:
        logger.warning(f"Failed to get GPU stats: {e}")
        return None