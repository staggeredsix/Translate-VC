"""
Audio processor module.
Handles real-time audio processing and speech-to-text conversion.
"""

import time
import logging
import threading
import queue
import numpy as np
import torch
import sounddevice as sd
from typing import Callable, List, Optional

import config
from utils.audio_utils import normalize_audio, concatenate_audio_chunks, adaptive_vad_threshold
from utils.logging_utils import PerformanceLogger
from services.whisper_service import get_whisper_service
from services.broadcaster_service import get_broadcaster_service

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("audio_processor")

class AudioStreamProcessor:
    """
    Real-time audio processor with optimizations for Ada GPUs.
    
    Handles microphone input, voice activity detection, and speech recognition.
    """
    
    def __init__(self, session_id: str, target_lang: str, audio_callback: Callable, text_callback: Callable):
        """
        Initialize the audio processor.
        
        Args:
            session_id: Session ID
            target_lang: Target language for translation
            audio_callback: Function to receive audio data
            text_callback: Function to receive text
        """
        self.session_id = session_id
        self.target_lang = target_lang
        self.audio_callback = audio_callback
        self.text_callback = text_callback
        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None
        self.process_thread = None
        self.vad_threshold = 0.3  # Voice activity detection threshold
        self.silence_frames = 0   # Counter for silent frames
        self.min_speech_frames = 10  # Minimum number of speech frames to process
        self.microphone_enabled = config.ENABLE_MICROPHONE_BY_DEFAULT
        self.last_process_time = 0  # Track last processing time for throttling
        self.process_lock = threading.Lock()  # Lock for processing state
        self.recent_levels: List[float] = []  # Recent audio levels for adaptive VAD
        
    def audio_input_callback(self, indata, frames, time_info, status):
        """
        Callback for incoming audio data from sounddevice.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Time info dictionary
            status: Status flags
        """
        if status:
            logger.warning(f"Audio status: {status}")
            
        # Skip if microphone is disabled
        if not self.microphone_enabled:
            return
            
        self.audio_queue.put(indata.copy())

    def start(self) -> None:
        """Start audio processing"""
        if self.running:
            return
            
        self.running = True
        
        # Register with broadcaster service
        broadcaster = get_broadcaster_service()
        broadcaster.register(
            self.session_id, 
            self.target_lang, 
            self.audio_callback, 
            self.text_callback
        )
        
        # Start audio input stream with optimizations for Ada GPUs
        blocksize = 512 if config.ADA_OPTIMIZED else 1024  # Smaller blocks for Ada's faster processing
        
        self.stream = sd.InputStream(
            callback=self.audio_input_callback, 
            channels=1, 
            samplerate=config.SAMPLE_RATE,
            blocksize=blocksize,
            latency='low' if config.ADA_OPTIMIZED else 'high'  # Lower latency for Ada GPUs
        )
        self.stream.start()
        
        # Start processing thread with higher priority for Ada GPUs
        self.process_thread = threading.Thread(
            target=self.process_audio,
            name=f"AudioProcessor-{self.session_id[:8]}",
            daemon=True
        )
        self.process_thread.start()
        
        # Set thread priority on supported platforms
        try:
            import os
            if hasattr(os, "sched_setaffinity") and config.ADA_OPTIMIZED:
                import psutil
                # Try to set CPU affinity to performance cores on modern CPUs
                p = psutil.Process()
                # Use first half of cores (usually performance cores on hybrid architectures)
                cpu_count = psutil.cpu_count(logical=False)
                if cpu_count >= 4:
                    p.cpu_affinity(list(range(cpu_count // 2)))
        except:
            pass  # Ignore if not supported
            
        logger.info(f"Started audio processor for session {self.session_id[:8]}")
        
        # Initial microphone state notification
        self.text_callback(f"System: Microphone is {'enabled' if self.microphone_enabled else 'disabled'}")

    def stop(self) -> None:
        """Stop audio processing"""
        if not self.running:
            return
            
        self.running = False
        
        # Wait for processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=2.0)
            
        # Close stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
        # Unregister from broadcaster
        broadcaster = get_broadcaster_service()
        broadcaster.unregister(self.session_id)
        
        logger.info(f"Stopped audio processor for session {self.session_id[:8]}")
        
    def toggle_microphone(self) -> str:
        """
        Toggle microphone on/off.
        
        Returns:
            str: Status message
        """
        self.microphone_enabled = not self.microphone_enabled
        status = "enabled" if self.microphone_enabled else "disabled"
        logger.info(f"Microphone {status} for session {self.session_id[:8]}")
        
        # Clear the queue when disabling to avoid processing stale audio
        if not self.microphone_enabled:
            with self.audio_queue.mutex:
                self.audio_queue.queue.clear()
                
        self.text_callback(f"System: Microphone {status}")
        return f"Microphone {status}"

    def process_audio(self) -> None:
        """Process audio chunks from the queue with Ada optimizations"""
        buffer = []
        # Process cycle tracking for adaptive behavior
        cycle_times = []
        adaptive_chunk_size = config.CHUNK_SIZE
        
        while self.running:
            try:
                # Skip processing if microphone is disabled
                if not self.microphone_enabled:
                    time.sleep(0.1)  # Reduce CPU usage when disabled
                    continue
                    
                # Get audio chunk with timeout
                try:
                    chunk = self.audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                
                # Enhanced voice activity detection with noise floor calibration
                audio_level = np.abs(chunk).mean()
                
                # Track audio levels for adaptive threshold
                self.recent_levels.append(audio_level)
                self.recent_levels = self.recent_levels[-100:]  # Keep last 100 values
                
                # Dynamic VAD on Ada GPUs
                if config.ADA_OPTIMIZED and len(self.recent_levels) > 20:
                    self.vad_threshold = adaptive_vad_threshold(
                        self.recent_levels, 
                        self.vad_threshold
                    )
                
                # Check if current chunk contains speech
                if audio_level > self.vad_threshold:
                    buffer.append(chunk)
                    self.silence_frames = 0
                else:
                    self.silence_frames += 1
                    
                    # Still add some silence for context
                    if self.silence_frames < 10:
                        buffer.append(chunk)
                
                # Adapt chunk size based on processing speed on Ada GPUs
                if config.ADA_OPTIMIZED and len(cycle_times) >= 10:
                    avg_cycle_time = sum(cycle_times) / len(cycle_times)
                    if avg_cycle_time < 0.05:  # Very fast processing
                        adaptive_chunk_size = max(30, int(config.CHUNK_SIZE * 0.6))  # Process smaller chunks more frequently
                    elif avg_cycle_time > 0.2:  # Slower processing
                        adaptive_chunk_size = min(80, int(config.CHUNK_SIZE * 1.5))  # Process larger chunks less frequently
                
                # Process when buffer is large enough or we have enough silence after speech
                buffer_full = len(buffer) >= adaptive_chunk_size
                silence_after_speech = self.silence_frames >= 15 and len(buffer) > self.min_speech_frames
                time_since_last = time.time() - self.last_process_time > 1.0  # At least 1 second between long utterances
                
                if buffer_full or silence_after_speech or (time_since_last and len(buffer) > 30):
                    # Skip if buffer is too small (just noise)
                    if len(buffer) <= self.min_speech_frames:
                        buffer = []
                        continue
                    
                    # Only process if we can acquire the lock (avoid overlap)
                    if not self.process_lock.acquire(blocking=False):
                        continue
                        
                    try:
                        # Measure cycle time
                        cycle_start = time.time()
                        
                        audio_data = concatenate_audio_chunks(buffer)
                        buffer = []
                        self.silence_frames = 0
                        self.last_process_time = time.time()
                        
                        # Run in current thread for Ada GPUs, separate thread otherwise
                        if config.ADA_OPTIMIZED:
                            self.handle_translation(audio_data)
                        else:
                            threading.Thread(
                                target=self.handle_translation,
                                args=(audio_data,),
                                daemon=True
                            ).start()
                            
                        # Track cycle time
                        cycle_time = time.time() - cycle_start
                        cycle_times.append(cycle_time)
                        cycle_times = cycle_times[-10:]  # Keep last 10 times
                    finally:
                        self.process_lock.release()
                    
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                continue

    @torch.no_grad()
    def handle_translation(self, audio_data: np.ndarray) -> None:
        """
        Process audio data and send to broadcaster.
        
        Args:
            audio_data: Audio data as numpy array
        """
        try:
            perf_logger.start_timer("handle_translation")
            
            # Skip if microphone was disabled while processing
            if not self.microphone_enabled:
                return
                
            # Get speech-to-text service
            whisper_service = get_whisper_service()
            
            # Memory buffer for audio data on Ada GPUs
            if config.ADA_OPTIMIZED:
                # Skip temp file for faster processing on Ada GPUs
                audio_normalized = normalize_audio(audio_data, "float32")
                
                # Transcribe audio directly from memory
                text, source_lang = whisper_service.transcribe(
                    audio_normalized,
                    vad_filter=True
                )
            else:
                # Transcribe audio with standard settings
                text, source_lang = whisper_service.transcribe(
                    audio_data,
                    vad_filter=True
                )
            
            # Only broadcast if we have text and microphone is still enabled
            if text.strip() and self.microphone_enabled:
                # Get broadcaster service
                broadcaster = get_broadcaster_service()
                broadcaster.broadcast(text, source_lang, self.session_id)
                
            perf_logger.end_timer("handle_translation")
                
        except Exception as e:
            logger.error(f"Translation error: {e}")
            
    def set_target_language(self, target_lang: str) -> None:
        """
        Set the target language for this processor.
        
        Args:
            target_lang: Target language name
        """
        if target_lang != self.target_lang:
            logger.info(f"Changed target language for session {self.session_id[:8]} from {self.target_lang} to {target_lang}")
            self.target_lang = target_lang
            
            # Update broadcaster registration
            broadcaster = get_broadcaster_service()
            broadcaster.unregister(self.session_id)
            broadcaster.register(
                self.session_id, 
                self.target_lang, 
                self.audio_callback, 
                self.text_callback
            )