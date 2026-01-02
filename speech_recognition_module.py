"""
Speech Recognition Module using Distil-Whisper for real-time audio transcription.
Based on: https://huggingface.co/distil-whisper/distil-large-v3
"""

import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading
import queue
import time

# Global model cache
_whisper_model = None
_whisper_processor = None
_whisper_pipeline = None
_device = None
_torch_dtype = None


def initialize_whisper():
    """Initialize Distil-Whisper model (lazy loading)."""
    global _whisper_model, _whisper_processor, _whisper_pipeline, _device, _torch_dtype
    
    if _whisper_pipeline is not None:
        return _whisper_pipeline
    
    try:
        print("Loading Distil-Whisper model...")
        _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        _torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id = "distil-whisper/distil-large-v3"
        
        _whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=_torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        _whisper_model.to(_device)
        
        _whisper_processor = AutoProcessor.from_pretrained(model_id)
        
        _whisper_pipeline = pipeline(
            "automatic-speech-recognition",
            model=_whisper_model,
            tokenizer=_whisper_processor.tokenizer,
            feature_extractor=_whisper_processor.feature_extractor,
            max_new_tokens=256,  # Increased for longer transcriptions
            chunk_length_s=30,  # Process longer chunks for better context
            torch_dtype=_torch_dtype,
            device=_device,
        )
        
        print("✓ Distil-Whisper model loaded successfully!")
        return _whisper_pipeline
        
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        raise


def transcribe_audio(audio_data, sample_rate=16000):
    """
    Transcribe audio data to text.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: Sample rate of audio (default 16000 for Whisper)
        
    Returns:
        Transcribed text string
    """
    try:
        if _whisper_pipeline is None:
            initialize_whisper()
        
        # Ensure audio is in correct format
        if isinstance(audio_data, np.ndarray):
            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize if needed
            if audio_data.max() > 1.0 or audio_data.min() < -1.0:
                audio_data = audio_data / np.max(np.abs(audio_data))
        
        # Transcribe with better accuracy settings
        result = _whisper_pipeline(
            {"array": audio_data, "sampling_rate": sample_rate},
            return_timestamps=False,
            generate_kwargs={
                "language": "en",  # Specify English for better accuracy
                "task": "transcribe",
                "temperature": 0.0,  # Lower temperature for more consistent results
            }
        )
        
        text = result.get("text", "").strip()
        
        # Clean up common transcription errors
        if text:
            # Remove extra whitespace
            text = " ".join(text.split())
            # Capitalize first letter
            if len(text) > 0:
                text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
        
    except Exception as e:
        print(f"Error in transcription: {e}")
        return ""


def transcribe_audio_file(file_path):
    """
    Transcribe audio from a file.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Transcribed text string
    """
    try:
        if _whisper_pipeline is None:
            initialize_whisper()
        
        result = _whisper_pipeline(file_path, return_timestamps=False)
        text = result.get("text", "").strip()
        return text
        
    except Exception as e:
        print(f"Error transcribing file: {e}")
        return ""

