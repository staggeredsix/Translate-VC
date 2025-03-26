#!/usr/bin/env python
"""
Main entry point for the multilingual voice chat application.
"""

import os
import sys
import argparse
import logging

# Add the parent directory to the path so we can import the package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from multilingual_voice_chat.utils.logging_utils import setup_logging
from multilingual_voice_chat.utils.gpu_utils import detect_gpu, optimize_cuda
from multilingual_voice_chat.services import initialize_services
from multilingual_voice_chat.ui import VoiceChatUI
from multilingual_voice_chat import config

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multilingual Voice Chat")
    
    parser.add_argument("--share", action="store_true", help="Share the app publicly")
    parser.add_argument("--port", type=int, default=7860, help="Port to run the app on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", action="store_true", help="Log to file")
    parser.add_argument("--whisper-model", type=str, default=None, 
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--disable-verify", action="store_true", 
                        help="Disable translation verification")
    parser.add_argument("--enable-mic", action="store_true", 
                        help="Enable microphone by default")
    parser.add_argument("--host", type=str, default="127.0.0.1", 
                        help="Host to run the app on")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--log-file", action="store_true", help="Log to file")
    
    return parser.parse_args()

def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level=log_level, log_to_file=args.log_file)
    
    # Detect GPU capabilities
    logger.info("Detecting GPU capabilities...")
    detect_gpu()
    
    # Apply CUDA optimizations if GPU is available
    if config.DEVICE == "cuda":
        optimize_cuda()
    
    # Update config from arguments
    if args.whisper_model:
        config.update_config("WHISPER_MODEL_SIZE", args.whisper_model)
        
    if args.disable_verify:
        config.update_config("TRANSLATION_VERIFICATION", False)
        
    if args.enable_mic:
        config.update_config("ENABLE_MICROPHONE_BY_DEFAULT", True)
    
    # Initialize services
    logger.info("Initializing services...")
    initialize_services()
    
    # Create and launch UI
    logger.info("Starting application...")
    ui = VoiceChatUI()
    ui.launch(
        share=args.share,
        server_port=args.port
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)