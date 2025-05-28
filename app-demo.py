from fastapi import FastAPI, UploadFile, HTTPException
import whisper
import spacy
import numpy as np
import torch
import io
import soundfile as sf
import os
import shutil # For safely handling temporary files

from TTS.api import TTS

# --- Load models once at application startup ---
try:
    print("Loading Whisper model...")
    whisper_model_global = whisper.load_model("small").to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Whisper model loaded on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded.")

    print("Loading TTS model...")
    # It's good practice to also move TTS model to GPU if available and it supports it
    tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC").to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TTS model loaded on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

except Exception as e:
    print(f"Error loading models: {e}")
    # Consider raising an error or exiting if models can't be loaded
    # For a FastAPI app, you might want to log this and let the app fail gracefully
    # raise e

app = FastAPI()

def speech_to_text(audio_file: UploadFile):
    """Convert speech to text using Whisper AI by saving to a temporary file."""
    # Ensure a directory for temporary files exists
    temp_dir = "temp_audio_files"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate a unique temporary file path
    temp_file_path = os.path.join(temp_dir, f"uploaded_audio_{os.getpid()}_{hash(audio_file.filename)}.wav")

    try:
        # Save the uploaded audio to the temporary file
        with open(temp_file_path, "wb") as f:
            # Use shutil.copyfileobj for efficient copying of file-like objects
            shutil.copyfileobj(audio_file.file, f) 

        print(f"Audio saved to temporary file: {temp_file_path}")

        # Transcribe audio directly from the file path
        # Whisper handles the loading, resampling, and memory management internally
        result = whisper_model_global.transcribe(temp_file_path)
        return result["text"]
    except Exception as e:
        print(f"Error during speech-to-text transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text processing failed: {e}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file removed: {temp_file_path}")
        # Optionally, clean up the temp_dir if it becomes empty
        # try:
        #     os.rmdir(temp_dir)
        # except OSError:
        #     pass # Directory not empty, ignore

def get_intent(text):
    """Classify user intent using spaCy"""
    doc = nlp(text)
    if "order" in text.lower():
        return "order_status"
    elif "hello" in text.lower() or "hi" in text.lower():
        return "greeting"
    elif "problem" in text.lower() or "issue" in text.lower():
        return "complaint"
    else:
        return "unknown_intent"

def text_to_speech(text):
    """Convert AI response to speech and return the path to the generated audio."""
    # Generate a unique filename for the response audio
    response_audio_filename = f"response_{os.getpid()}_{hash(text)}.wav"
    response_audio_path = os.path.join("response_audio", response_audio_filename) # Store in a sub-directory

    # Ensure the directory exists
    os.makedirs("response_audio", exist_ok=True)

    try:
        tts_model.tts_to_file(text=text, file_path=response_audio_path)
        return response_audio_path
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech processing failed: {e}")


@app.post("/process_call/")
async def process_call(audio_file: UploadFile):
    """API endpoint to process call audio"""
    if not audio_file.filename.endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Unsupported audio file format. Please upload WAV, MP3, OGG, or FLAC.")

    user_text = speech_to_text(audio_file) # Pass UploadFile object directly
    user_intent = get_intent(user_text)

    responses = {
        "greeting": "Hello! How can I assist you today?",
        "order_status": "Your order is being processed.",
        "complaint": "I'm sorry you're facing issues. Let me help!",
        "unknown_intent": "I'm here to assist you. Could you please clarify?"
    }
    bot_response = responses.get(user_intent, "I'm here to help!")

    response_audio_path = text_to_speech(bot_response)

    # In a real-world scenario, you might want to return the audio file itself
    # or a URL to it, not just the local path. For demonstration, returning path.
    return {"user_text": user_text, "bot_response": bot_response, "audio_file_path": response_audio_path}

# To serve the generated audio files (optional, for testing)
from fastapi.responses import FileResponse

@app.get("/audio/{file_path:path}")
async def get_audio_file(file_path: str):
    """Endpoint to serve generated audio files."""
    full_path = os.path.join("response_audio", file_path)
    if os.path.exists(full_path):
        return FileResponse(full_path)
    else:
        raise HTTPException(status_code=404, detail="Audio file not found.")