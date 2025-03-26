

# Multilingual Voice Chat (NVIDIA Ada GPU-Optimized)

Real-time multilingual voice chat with speech recognition, translation, and text-to-speech. This application is optimized for NVIDIA Ada Lovelace GPUs (RTX 40 series) but works on any CUDA-capable GPU or even on CPU.

## Features

- Real-time speech-to-text using Whisper models
- Neural machine translation between multiple languages
- Text-to-speech synthesis for translated content
- Automatic translation verification and correction
- Manual microphone control
- Voice activity detection with adaptive thresholds
- Optimized for NVIDIA Ada Lovelace GPUs
- Gradio web interface

## Running with Docker (Recommended)

The easiest way to run this application is using Docker, which handles all dependencies and Python version requirements automatically.

1. Make sure you have Docker installed (and NVIDIA Container Toolkit for GPU support)
2. Run the application using the provided script:

```bash
chmod +x run_docker.sh
./run_docker.sh
```

3. Open your browser and go to http://localhost:7860

See [CONTAINERIZATION.md](CONTAINERIZATION.md) for more detailed instructions on running with Docker.

## Project Structure

The project has been organized into modular components:

```
multilingual_voice_chat/
├── config.py                # Configuration and constants
├── main.py                  # Entry point
├── requirements.txt         # Dependencies
├── models/                  # Directory for cached models
├── utils/                   # Utility functions
├── services/                # Core services (STT, TTS, Translation)
├── processors/              # Audio processing components
└── ui/                      # Gradio UI components
```

## Installation

1. Clone the repository:

```bash
git clone https://your-repository-url.git
cd multilingual-voice-chat
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

Run the application with default settings:

```bash
python main.py
```

### Command Line Options

- `--share`: Share the app publicly via Gradio
- `--port PORT`: Port to run the app on (default: 7860)
- `--host HOST`: Host to run the app on (default: 127.0.0.1)
- `--debug`: Enable debug logging
- `--log-file`: Log to file
- `--whisper-model {tiny,base,small,medium,large}`: Set Whisper model size
- `--disable-verify`: Disable translation verification
- `--enable-mic`: Enable microphone by default

Example with custom settings:

```bash
python main.py --port 8080 --whisper-model small --enable-mic
```

## GPU Optimization

The application automatically detects NVIDIA Ada Lovelace GPUs (RTX 40 series) and applies specialized optimizations:

- Automatic Mixed Precision (AMP) with float16
- Lazy CUDA module loading
- TensorFloat32 (TF32) matrix multiplication
- Parallel processing for multiple users
- Optimized batch processing settings
- Direct memory buffer processing (reducing disk I/O)

## Configuration

Main configuration options can be found in `config.py`. You can modify these settings directly or through command line arguments.

## Services

The application is built on several core services:

- **WhisperService**: Speech-to-text using Whisper models
- **TranslationService**: Neural machine translation
- **TTSService**: Text-to-speech synthesis
- **VerificationService**: Translation quality verification
- **BroadcasterService**: Message distribution to connected clients
- **UserProfileService**: User profile and preference management

## Extending

### Adding New Languages

To add support for a new language, update the `LANG_MAP` dictionary in `config.py`:

```python
LANG_MAP = {
    "English": "en",
    "Spanish": "es",
    # Add new languages here
    "Thai": "th",
}
```

### Using Different Models

You can change the models used for each service:

- **WhisperService**: Set the model size using `--whisper-model` or by modifying `WHISPER_MODEL_SIZE` in `config.py`
- **TranslationService**: Modify the `model_name` in `translation_service.py`
- **TTSService**: Change the model in `tts_service.py`

## License

[License details here]

## Acknowledgements

This project uses the following open source components:
- [Whisper](https://github.com/openai/whisper) - Speech recognition
- [MarianMT](https://huggingface.co/transformers/model_doc/marian.html) - Neural machine translation
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech synthesis
- [Gradio](https://gradio.app/) - Web interface