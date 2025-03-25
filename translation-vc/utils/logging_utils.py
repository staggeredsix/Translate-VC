"""
Logging configuration for the application.
Sets up logging with proper formatting and levels.
"""

import logging
import os
import sys
from datetime import datetime

def setup_logging(log_level=logging.INFO, log_to_file=False):
    """
    Set up logging configuration.
    
    Args:
        log_level: The logging level (default: INFO)
        log_to_file: Whether to log to a file (default: False)
    
    Returns:
        logger: The configured root logger
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/voice_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers if any
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler with formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)
    
    return root_logger

def get_logger(name):
    """
    Get a logger with the specified name.
    
    Args:
        name: Name for the logger
    
    Returns:
        logger: The named logger
    """
    return logging.getLogger(name)

# Performance tracking utils
class PerformanceLogger:
    """
    Utility class for logging performance metrics.
    """
    def __init__(self, name):
        self.logger = logging.getLogger(f"perf.{name}")
        self.metrics = {}
    
    def start_timer(self, operation):
        """Start timing an operation"""
        import time
        self.metrics[operation] = {"start": time.time()}
    
    def end_timer(self, operation, log=True):
        """
        End timing an operation and optionally log the result
        
        Returns:
            float: The operation duration in seconds
        """
        import time
        if operation not in self.metrics:
            return None
            
        end_time = time.time()
        start_time = self.metrics[operation]["start"]
        duration = end_time - start_time
        
        self.metrics[operation]["end"] = end_time
        self.metrics[operation]["duration"] = duration
        
        if log:
            self.logger.debug(f"{operation} completed in {duration:.3f}s")
            
        return duration
    
    def log_metric(self, name, value):
        """Log a custom metric"""
        self.metrics[name] = value
        self.logger.debug(f"{name}: {value}")