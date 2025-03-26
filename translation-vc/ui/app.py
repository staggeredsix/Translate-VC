"""
Gradio UI components for the multilingual voice chat application.
"""

import gradio as gr
import logging
import torch

import config
from utils.gpu_utils import gpu_memory_stats
from services.user_profile_service import get_user_profile_service
from services.whisper_service import get_whisper_service
from services.translation_service import get_translation_service
from services.tts_service import get_tts_service
from services.verification_service import get_verification_service
from processors.audio_processor import AudioStreamProcessor

logger = logging.getLogger(__name__)

class VoiceChatUI:
    """Gradio UI for the multilingual voice chat application"""
    
    def __init__(self):
        self.app = None
        self.session_id = None
        self.audio_processor_state = None
        
        # Initialize services
        self.user_profile_service = get_user_profile_service()
        self.whisper_service = get_whisper_service()
        self.translation_service = get_translation_service()
        self.tts_service = get_tts_service()
    
    def build_ui(self):
        """Build the Gradio UI"""
        # Initialize user session
        self.session_id = self.user_profile_service.create_session()
        profile = self.user_profile_service.get_session(self.session_id)
        preferred_lang = profile.get("preferred_lang", "English") if profile else "English"
        
        # Create Gradio app
        with gr.Blocks(analytics_enabled=False, theme=gr.themes.Soft()) as app:
            gr.Markdown(f"## 🌎 Multilingual Real-Time Voice Chat (NVIDIA Ada GPU-Optimized)")
            
            # GPU status
            with gr.Row():
                gpu_info = f"🚀 Running on: {config.DEVICE.upper()}"
                if config.GPU_ARCHITECTURE:
                    gpu_info += f" - {config.GPU_ARCHITECTURE} ({torch.cuda.get_device_name(0)})"
                    if config.ADA_OPTIMIZED:
                        gpu_info += " ✓ Ada optimizations enabled"
                else:
                    gpu_info += f" - {torch.cuda.get_device_name(0) if config.DEVICE=='cuda' else 'CPU'}"
                    
                gr.Markdown(gpu_info)
            
            # Audio and text components
            output_audio = gr.Audio(label="Translated Audio Output", interactive=False)
            chat_box = gr.Textbox(
                label="Live Transcript (Multilingual Chat Log)", 
                lines=10, 
                interactive=False,
                max_lines=100
            )

            # State variables
            self.audio_processor_state = gr.State(None)

            # Settings panel
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    
                    lang_dropdown = gr.Dropdown(
                        choices=list(config.LANG_MAP.keys()), 
                        value=preferred_lang, 
                        label="Target Language"
                    )
                    
                    mic_enabled = gr.Checkbox(
                        value=config.ENABLE_MICROPHONE_BY_DEFAULT,
                        label="Enable Microphone",
                        info="Turn microphone on/off"
                    )
                    
                    vad_sensitivity = gr.Slider(
                        minimum=0.1, 
                        maximum=0.9, 
                        value=0.3, 
                        step=0.05, 
                        label="Voice Detection Sensitivity",
                        info="Higher values require louder speech"
                    )
                    
                    if config.ADA_OPTIMIZED:
                        # Ada-specific quality settings
                        audio_quality = gr.Radio(
                            choices=["Low", "Medium", "High", "Ultra"], 
                            value="Medium", 
                            label="Audio Quality",
                            info="Higher quality uses more GPU resources"
                        )
                    else:
                        audio_quality = gr.Radio(
                            choices=["Low", "Medium", "High"], 
                            value="Medium", 
                            label="Audio Quality"
                        )
                        
                    verify_translation = gr.Checkbox(
                        value=config.TRANSLATION_VERIFICATION,
                        label="Verify Translations",
                        info="Double-check and fix poor translations"
                    )
                
                # Chat and audio output panel
                with gr.Column(scale=2):
                    gr.Markdown("### Live Translation")
                    
                    # Controls row
                    with gr.Row():
                        stream_btn = gr.Button("🎙️ Start Stream", variant="primary", scale=1)
                        stop_btn = gr.Button("⏹️ Stop", variant="secondary", scale=1)
                        mic_toggle_btn = gr.Button("🔇/🔊 Toggle Mic", scale=1)
                        
                    # Status info
                    status_info = gr.Textbox(
                        label="Status", 
                        lines=3,
                        value=f"Ready. Session ID: {self.session_id[:8]}"
                    )
                    
                    # Add chat box and audio output
                    gr.Markdown("### Conversation")
                    app.append(chat_box)
                    app.append(output_audio)
            
            # Microphone input panel for single clips
            with gr.Row():
                gr.Markdown("### Record Single Translation")
                mic_input = gr.Audio(
                    source="microphone", 
                    type="numpy", 
                    label="Record Single Clip"
                )
            
            # Wire up event handlers
            self._connect_event_handlers(
                app=app,
                mic_input=mic_input,
                lang_dropdown=lang_dropdown,
                mic_enabled=mic_enabled,
                vad_sensitivity=vad_sensitivity,
                audio_quality=audio_quality,
                verify_translation=verify_translation,
                stream_btn=stream_btn,
                stop_btn=stop_btn,
                mic_toggle_btn=mic_toggle_btn,
                status_info=status_info,
                chat_box=chat_box,
                output_audio=output_audio
            )
            
            # Cleanup on close
            app.on_close(self._cleanup_resources)
            
            self.app = app
            return app
    
    def _connect_event_handlers(self, **components):
        """
        Connect event handlers to UI components
        
        Args:
            **components: UI components to connect
        """
        # Update chat text handler
        def update_chat(text):
            existing = components["chat_box"].value or ""
            return (existing + "\n" + text).strip()
            
        # Stream start handler
        def stream_start(state, target_lang, enable_mic):
            if state is not None:
                state.stop()
            
            processor = AudioStreamProcessor(
                self.session_id, 
                target_lang, 
                lambda a: components["output_audio"].update(value=a),
                update_chat
            )
            processor.microphone_enabled = enable_mic
            processor.start()
            
            return processor, f"Started streaming with target language: {target_lang}\nMicrophone is {'enabled' if enable_mic else 'disabled'}"
            
        # Stream stop handler
        def stream_stop(state):
            if state is not None:
                state.stop()
            return None, "Stopped streaming"
            
        # Microphone toggle handler
        def toggle_microphone(state):
            if state is None:
                return None, "Streaming not active"
            result = state.toggle_microphone()
            return state, result
            
        # Single audio translation handler
        def translate_audio(audio, target_lang):
            if not audio or not isinstance(audio, tuple) or len(audio) < 2:
                return None, f"No valid audio received"
                
            try:
                # Transcribe audio
                text, source_lang = self.whisper_service.transcribe(audio[1])
                
                if not text.strip():
                    return None, "No speech detected"
                    
                # Translate text
                translated_text = self.translation_service.translate(
                    text, 
                    source_lang, 
                    config.LANG_MAP[target_lang]
                )
                
                # Update user transcript
                self.user_profile_service.update_preference(self.session_id, "preferred_lang", target_lang)
                self.user_profile_service.add_transcript_message(
                    self.session_id,
                    {
                        "speaker": self.session_id[:8], 
                        "text": translated_text,
                        "original_text": text,
                        "source_lang": source_lang,
                        "target_lang": target_lang
                    }
                )
                
                # Generate speech
                audio_data = self.tts_service.synthesize(translated_text)
                
                return audio_data, (
                    f"Translated from {source_lang} to {config.LANG_MAP[target_lang]}:\n"
                    f"Original: {text}\n"
                    f"Translated: {translated_text}"
                )
                
            except Exception as e:
                logger.error(f"Translation error: {e}")
                return None, f"Error: {str(e)}"
                
        # VAD sensitivity update handler
        def update_vad(processor, sensitivity):
            if processor is not None:
                processor.vad_threshold = sensitivity
            return f"Updated voice detection sensitivity to {sensitivity}"
            
        # Language preference update handler
        def update_language(processor, lang):
            self.user_profile_service.update_preference(self.session_id, "preferred_lang", lang)
            
            if processor is not None:
                processor.set_target_language(lang)
            return f"Updated target language to {lang}"
            
        # Audio quality update handler
        def update_quality(quality):
            if quality == "Ultra" and config.ADA_OPTIMIZED:
                whisper_size = "medium"
            elif quality == "Low":
                whisper_size = "tiny"
            elif quality == "Medium":
                whisper_size = "base"
            else:  # High
                whisper_size = "small"
                
            self.whisper_service.update_model_size(whisper_size)
            return f"Updated audio quality to {quality} ({whisper_size} model)"
            
        # Translation verification toggle handler
        def toggle_verification(enable_verify):
            verification_service = get_verification_service()
            verification_service.enable(enable_verify)
            return f"Translation verification {'enabled' if enable_verify else 'disabled'}"
            
        # Microphone enable/disable handler
        def update_microphone(processor, enabled):
            if processor is not None:
                processor.microphone_enabled = enabled
                return f"Microphone {'enabled' if enabled else 'disabled'}"
            return "Streaming not active"
            
        # Connect event handlers to UI components
        components["mic_input"].change(
            fn=translate_audio, 
            inputs=[components["mic_input"], components["lang_dropdown"]], 
            outputs=[components["output_audio"], components["status_info"]]
        )
        
        components["stream_btn"].click(
            fn=stream_start, 
            inputs=[self.audio_processor_state, components["lang_dropdown"], components["mic_enabled"]], 
            outputs=[self.audio_processor_state, components["status_info"]]
        )
        
        components["stop_btn"].click(
            fn=stream_stop, 
            inputs=[self.audio_processor_state], 
            outputs=[self.audio_processor_state, components["status_info"]]
        )
        
        components["mic_toggle_btn"].click(
            fn=toggle_microphone,
            inputs=[self.audio_processor_state],
            outputs=[self.audio_processor_state, components["status_info"]]
        )
        
        components["vad_sensitivity"].change(
            fn=update_vad,
            inputs=[self.audio_processor_state, components["vad_sensitivity"]],
            outputs=[components["status_info"]]
        )
        
        components["lang_dropdown"].change(
            fn=update_language,
            inputs=[self.audio_processor_state, components["lang_dropdown"]],
            outputs=[components["status_info"]]
        )
        
        components["audio_quality"].change(
            fn=update_quality,
            inputs=[components["audio_quality"]],
            outputs=[components["status_info"]]
        )
        
        components["verify_translation"].change(
            fn=toggle_verification,
            inputs=[components["verify_translation"]],
            outputs=[components["status_info"]]
        )
        
        components["mic_enabled"].change(
            fn=update_microphone,
            inputs=[self.audio_processor_state, components["mic_enabled"]],
            outputs=[components["status_info"]]
        )
    
    def _cleanup_resources(self):
        """Cleanup resources when the app is closed"""
        from ..utils.gpu_utils import cleanup_gpu
        
        # Stop audio processor if running
        if self.audio_processor_state is not None:
            state_value = self.audio_processor_state.value
            if state_value is not None:
                state_value.stop()
        
        # Save user profiles
        self.user_profile_service.save_profiles(force=True)
        
        # Clean up GPU resources
        cleanup_gpu()
        
        logger.info("Application resources cleaned up")
    
    def launch(self, **kwargs):
        """
        Launch the Gradio app.
        
        Args:
            **kwargs: Arguments to pass to gr.Blocks.launch()
        """
        if self.app is None:
            self.build_ui()
            
        # Apply system optimizations before launch
        if config.ADA_OPTIMIZED:
            # Pre-warm models to avoid cold start
            logger.info("Pre-warming models for optimal performance...")
            from ..utils.gpu_utils import warm_up_gpu
            warm_up_gpu()
            
        # Default launch parameters
        default_kwargs = {
            "share": False,
            "enable_queue": True,
            "show_error": True,
            "max_threads": 8 if config.ADA_OPTIMIZED else 4,
            "quiet": False
        }
        
        # Override defaults with provided kwargs
        launch_kwargs = {**default_kwargs, **kwargs}
        
        # Launch the app
        return self.app.launch(**launch_kwargs)