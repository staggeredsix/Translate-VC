"""
User profile service.
Manages user profiles, preferences and transcripts.
"""

import os
import json
import uuid
import logging
import threading
import time
from typing import Dict, Any, Optional

from .. import config

logger = logging.getLogger(__name__)

class UserProfileService:
    """Service for managing user profiles and preferences"""
    
    def __init__(self):
        self.profile_path = config.USER_PROFILE_PATH
        self.profiles = {}
        self.lock = threading.Lock()
        self.last_save = 0
        self.save_interval = 5  # seconds
        self.max_transcript_length = 100
    
    def load_profiles(self):
        """
        Load user profiles from disk.
        
        Returns:
            dict: User profiles
        """
        with self.lock:
            if os.path.exists(self.profile_path):
                try:
                    with open(self.profile_path, "r") as f:
                        self.profiles = json.load(f)
                    logger.info(f"Loaded {len(self.profiles)} user profiles")
                except json.JSONDecodeError:
                    logger.error("Failed to load user profiles, creating new file")
                    self.profiles = {}
            return self.profiles
    
    def save_profiles(self, force=False):
        """
        Save user profiles to disk.
        
        Args:
            force: Force save even if save interval hasn't elapsed
            
        Returns:
            bool: Success status
        """
        # Only save if interval has elapsed or force=True
        current_time = time.time()
        if not force and (current_time - self.last_save) < self.save_interval:
            return False
            
        with self.lock:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.profile_path) or '.', exist_ok=True)
                with open(self.profile_path, "w") as f:
                    json.dump(self.profiles, f)
                self.last_save = current_time
                logger.debug(f"Saved {len(self.profiles)} user profiles")
                return True
            except Exception as e:
                logger.error(f"Failed to save user profiles: {e}")
                return False
    
    def create_session(self, preferred_lang="English"):
        """
        Create a new user session.
        
        Args:
            preferred_lang: Preferred language for the user
            
        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())
        with self.lock:
            self.profiles[session_id] = {
                "preferred_lang": preferred_lang,
                "transcript": [],
                "created_at": time.time()
            }
        self.save_profiles()
        logger.info(f"Created new session: {session_id[:8]}")
        return session_id
    
    def get_session(self, session_id):
        """
        Get a user session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            dict: User session data or None if not found
        """
        with self.lock:
            return self.profiles.get(session_id)
    
    def update_preference(self, session_id, key, value):
        """
        Update a user preference.
        
        Args:
            session_id: Session ID
            key: Preference key
            value: Preference value
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if session_id not in self.profiles:
                return False
            self.profiles[session_id][key] = value
        
        # Only save on certain important preferences
        important_keys = ["preferred_lang"]
        if key in important_keys:
            self.save_profiles()
        return True
    
    def add_transcript_message(self, session_id, message):
        """
        Add a message to the user's transcript.
        
        Args:
            session_id: Session ID
            message: Message dictionary with text, speaker, etc.
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if session_id not in self.profiles:
                return False
                
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = time.time()
                
            # Add to transcript
            self.profiles[session_id]["transcript"].append(message)
            
            # Limit transcript length
            if len(self.profiles[session_id]["transcript"]) > self.max_transcript_length:
                self.profiles[session_id]["transcript"] = self.profiles[session_id]["transcript"][-self.max_transcript_length:]
                
        # Save periodically based on message count
        message_count = len(self.profiles[session_id]["transcript"])
        if message_count % 5 == 0:
            self.save_profiles()
            
        return True
    
    def get_transcript(self, session_id, limit=None):
        """
        Get the transcript for a session.
        
        Args:
            session_id: Session ID
            limit: Maximum number of messages to return (None for all)
            
        Returns:
            list: Transcript messages
        """
        with self.lock:
            if session_id not in self.profiles:
                return []
                
            transcript = self.profiles[session_id]["transcript"]
            if limit is not None:
                return transcript[-limit:]
            return transcript
    
    def clear_transcript(self, session_id):
        """
        Clear the transcript for a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if session_id not in self.profiles:
                return False
                
            self.profiles[session_id]["transcript"] = []
            
        self.save_profiles()
        return True
        
    def delete_session(self, session_id):
        """
        Delete a user session.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: Success status
        """
        with self.lock:
            if session_id not in self.profiles:
                return False
                
            del self.profiles[session_id]
            
        self.save_profiles()
        return True
        
    def clean_old_sessions(self, max_age_days=30):
        """
        Remove sessions older than the specified age.
        
        Args:
            max_age_days: Maximum age in days
            
        Returns:
            int: Number of sessions removed
        """
        max_age_seconds = max_age_days * 24 * 60 * 60
        current_time = time.time()
        sessions_to_delete = []
        
        with self.lock:
            for session_id, data in self.profiles.items():
                created_at = data.get("created_at", 0)
                if (current_time - created_at) > max_age_seconds:
                    sessions_to_delete.append(session_id)
                    
            # Delete the sessions
            for session_id in sessions_to_delete:
                del self.profiles[session_id]
                
        if sessions_to_delete:
            self.save_profiles()
            logger.info(f"Cleaned {len(sessions_to_delete)} old sessions")
            
        return len(sessions_to_delete)

# Singleton instance
_instance = None

def get_user_profile_service():
    """Get the singleton UserProfileService instance"""
    global _instance
    if _instance is None:
        _instance = UserProfileService()
        _instance.load_profiles()
    return _instance