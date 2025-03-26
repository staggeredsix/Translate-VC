"""
Broadcaster service.
Manages the broadcasting of messages to connected sessions.
"""

import time
import logging
import threading
from typing import Dict, Any, Callable

import config
from utils.logging_utils import PerformanceLogger
from translation_service import get_translation_service
from tts_service import get_tts_service
from user_profile_service import get_user_profile_service

logger = logging.getLogger(__name__)
perf_logger = PerformanceLogger("broadcaster")

class BroadcasterService:
    """Service for broadcasting messages to connected sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def register(self, session_id: str, lang: str, audio_callback: Callable, text_callback: Callable) -> None:
        """
        Register a new client session.
        
        Args:
            session_id: Session ID
            lang: Target language
            audio_callback: Function to receive audio data
            text_callback: Function to receive text
        """
        with self.lock:
            self.sessions[session_id] = {
                "lang": lang, 
                "audio_cb": audio_callback, 
                "text_cb": text_callback
            }
        logger.info(f"Registered session {session_id[:8]} with language {lang}")
        
    def unregister(self, session_id: str) -> None:
        """
        Remove a client session.
        
        Args:
            session_id: Session ID to remove
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"Unregistered session {session_id[:8]}")
    
    def broadcast(self, text: str, source_lang: str, sender_id: str) -> None:
        """
        Broadcast a message to all connected sessions.
        
        Args:
            text: Text message to broadcast
            source_lang: Source language code
            sender_id: ID of the sending session
        """
        if not text.strip():
            return
            
        # Start broadcast timing
        perf_logger.start_timer("broadcast")
        broadcast_start = time.time()
            
        # Copy sessions to avoid lock during processing
        with self.lock:
            current_sessions = self.sessions.copy()
        
        # No sessions to broadcast to
        if not current_sessions:
            return
            
        # For Ada GPUs, we can process multiple sessions in parallel
        if config.ADA_OPTIMIZED and len(current_sessions) > 1:
            # Use multi-threading to parallelize broadcasts
            threads = []
            for sid, data in current_sessions.items():
                thread = threading.Thread(
                    target=self._process_session_broadcast,
                    args=(sid, data, text, source_lang, sender_id),
                    daemon=True
                )
                thread.start()
                threads.append(thread)
                
            # Wait for all threads to complete (with timeout)
            for thread in threads:
                thread.join(timeout=5.0)
                
            duration = time.time() - broadcast_start
            logger.debug(f"Parallel broadcast to {len(current_sessions)} sessions took {duration:.2f}s")
        else:
            # Sequential processing for non-Ada GPUs or single session
            for sid, data in current_sessions.items():
                self._process_session_broadcast(sid, data, text, source_lang, sender_id)
                
            duration = time.time() - broadcast_start
            logger.debug(f"Sequential broadcast to {len(current_sessions)} sessions took {duration:.2f}s")
        
        perf_logger.end_timer("broadcast")
                
    def _process_session_broadcast(self, sid: str, data: Dict[str, Any], text: str, source_lang: str, sender_id: str) -> None:
        """
        Process broadcast for a single session.
        
        Args:
            sid: Session ID to process
            data: Session data
            text: Original text message
            source_lang: Source language code
            sender_id: ID of the sender
        """
        try:
            perf_logger.start_timer(f"process_session_{sid[:8]}")
            target_lang = data["lang"]
            
            # Get translation service
            translation_service = get_translation_service()
            
            # Use cached translation when possible
            translated = translation_service.translate_cached(text, source_lang, config.LANG_MAP[target_lang])
            
            # Update transcript in user profile
            user_profile_service = get_user_profile_service()
            message = {
                "speaker": sender_id[:8], 
                "text": translated,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": time.time()
            }
            user_profile_service.add_transcript_message(sid, message)
            
            # Generate and send TTS audio
            self._generate_and_send_audio(sid, data, translated, sender_id)
            
            perf_logger.end_timer(f"process_session_{sid[:8]}")
                
        except Exception as e:
            logger.error(f"Broadcast error for session {sid}: {e}")
            
    def _generate_and_send_audio(self, sid: str, data: Dict[str, Any], translated_text: str, sender_id: str) -> None:
        """
        Generate TTS audio and send to client.
        
        Args:
            sid: Session ID to send to
            data: Session data
            translated_text: Translated text to synthesize
            sender_id: ID of the original sender
        """
        # Skip audio generation for empty translations
        if not translated_text.strip():
            return
            
        perf_logger.start_timer("tts_generation")
        
        try:
            # Get TTS service
            tts_service = get_tts_service()
            
            # Generate speech
            audio_data = tts_service.synthesize(translated_text)
            
            if audio_data:
                # Send audio and text to client
                data["audio_cb"](audio_data)
                data["text_cb"](f"{sender_id[:8]}: {translated_text}")
                
                perf_logger.end_timer("tts_generation")
            else:
                logger.warning(f"TTS failed to generate audio for session {sid}")
                # Still send text even if audio fails
                data["text_cb"](f"{sender_id[:8]}: {translated_text} [TTS failed]")
                
        except Exception as e:
            logger.error(f"TTS error for session {sid}: {e}")
            # Still send text even if audio fails
            data["text_cb"](f"{sender_id[:8]}: {translated_text} [TTS failed]")

# Singleton instance
_instance = None

def get_broadcaster_service():
    """Get the singleton BroadcasterService instance"""
    global _instance
    if _instance is None:
        _instance = BroadcasterService()
    return _instance