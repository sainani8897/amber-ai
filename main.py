from fastapi import FastAPI, UploadFile, HTTPException, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles # <--- NEW IMPORT
import whisper
import spacy
import torch
import os
import shutil
import uuid
import asyncio
import numpy as np
import io
import soundfile as sf
from starlette.websockets import WebSocketDisconnect
import time # For measuring processing time

from pydub import AudioSegment # For decoding browser audio

# LangChain imports for RAG
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from TTS.api import TTS

# --- Environment Variable for OpenAI API Key ---
# IMPORTANT: DO NOT HARDCODE YOUR API KEY IN PRODUCTION!
# For demonstration, you've hardcoded it here.
# For production, always use: OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
try:
    # Your hardcoded key (for testing ONLY, as you've provided it)
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # For production, uncomment the line below and remove the hardcoded key:
    # OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 
except KeyError:
    raise ValueError("OPENAI_API_KEY environment variable not set. Please set it before running the app.")

# Define your bot's name globally
BOT_NAME = "TechBot" 

# --- Global Model Loading ---
# These models are loaded once when the application starts to save memory and time.
try:
    print("Loading Whisper model...")
    # Consider "tiny.en" or "base.en" for faster processing if "small" is too slow for real-time feel
    whisper_model_global = whisper.load_model("small").to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Whisper model loaded on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy model loaded.")

    print("Loading TTS model...")
    tts_model_global = TTS("tts_models/en/ljspeech/tacotron2-DDC").to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"TTS model loaded on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    # --- LLM Component Initialization ---
    print("Initializing LLM components (OpenAIEmbeddings, ChromaDB, OpenAI LLM)...")
    PERSIST_DIRECTORY = 'chroma_db'
    TEMP_AUDIO_DIR = "temp_websocket_audio" # Directory for temporary audio files

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    
    if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
        raise RuntimeError(f"Chroma DB not found at {PERSIST_DIRECTORY}. Please run prepare_data.py first.")
    
    vectordb = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7) 
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    qa_template = f"""You are {BOT_NAME}, an AI assistant specialized in providing information about Tech Innovations Inc.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know and ask for clarification, do not try to make up an answer.
Keep the answer concise and relevant to the business data provided.

Context:
{{context}}

Question: {{question}}
Answer:""" 
    QA_CHAIN_PROMPT = PromptTemplate.from_template(qa_template)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False, 
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("LLM components initialized.")

except Exception as e:
    print(f"Error loading models or LLM components: {e}")
    raise RuntimeError(f"Failed to load one or more AI models or LLM components: {e}")

app = FastAPI()

# --- NEW: Serve static files (your frontend) ---
# Mounts the 'static' directory to be served at the '/static' URL path.
# E.g., http://localhost:8000/static/your_image.png
app.mount("/static", StaticFiles(directory="static"), name="static")

# This endpoint serves your index.html file when someone accesses the root URL.
# So, going to http://localhost:8000/ will load your frontend.
@app.get("/")
async def serve_index():
    return FileResponse("static/index.html", media_type="text/html")

# --- Utility Functions ---

def speech_to_text(audio_file: UploadFile):
    """Converts speech to text using Whisper AI, handling large files efficiently."""
    temp_upload_dir = "temp_uploaded_audio"
    os.makedirs(temp_upload_dir, exist_ok=True)
    unique_filename = f"uploaded_audio_{uuid.uuid4()}.wav"
    temp_file_path = os.path.join(temp_upload_dir, unique_filename)

    try:
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(audio_file.file, f) 
        print(f"Audio saved to temporary file: {temp_file_path}")
        result = whisper_model_global.transcribe(temp_file_path)
        return result["text"]
    except Exception as e:
        print(f"Error during speech-to-text transcription: {e}")
        raise HTTPException(status_code=500, detail=f"Speech-to-text processing failed: {e}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file removed: {temp_file_path}")

def get_intent(text):
    """Classifies user intent using spaCy."""
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
    """Converts AI response text to speech using Coqui TTS."""
    response_audio_dir = "response_audio_files"
    os.makedirs(response_audio_dir, exist_ok=True)
    unique_filename = f"response_{uuid.uuid4()}.wav"
    response_audio_path = os.path.join(response_audio_dir, unique_filename)

    try:
        tts_model_global.tts_to_file(text=text, file_path=response_audio_path)
        return response_audio_path
    except Exception as e:
        print(f"Error during text-to-speech conversion: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech processing failed: {e}")

# --- API Endpoint: Process Call (Unchanged) ---
@app.post("/process_call/")
async def process_call(audio_file: UploadFile):
    """
    API endpoint to process call audio:
    1. Converts speech to text.
    2. Uses LLM to answer business-related queries based on provided data.
    3. Converts the bot response to speech.
    """
    allowed_audio_types = [".wav", ".mp3", ".ogg", ".flac"]
    if not audio_file.filename.lower().endswith(tuple(allowed_audio_types)):
        raise HTTPException(status_code=400, detail=f"Unsupported audio file format. Please upload one of: {', '.join(allowed_audio_types)}")

    user_text = speech_to_text(audio_file)
    print(f"User Transcribed Text: {user_text}")

    try:
        llm_response = rag_chain.invoke(user_text) 
        bot_response_text = llm_response['result']
        print(f"LLM Generated Response: {bot_response_text}")

    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        bot_response_text = f"I'm sorry, I'm {BOT_NAME} and I'm having trouble connecting to my knowledge base right now. Please try again later."
        
    response_audio_path = text_to_speech(bot_response_text)

    return {
        "user_text": user_text,
        "bot_response": bot_response_text,
        "response_audio_file_path": response_audio_path
    }

# --- API Endpoint: Query Text Directly (Unchanged) ---
class TextQuery(BaseModel):
    query: str

@app.post("/query_text/")
async def query_text(request: TextQuery):
    """
    API endpoint to directly query the LLM with text input.
    1. Takes a text query.
    2. Uses LLM (RAG chain) to answer based on business data.
    3. Returns the text response.
    """
    user_query = request.query
    print(f"Received Text Query: {user_query}")

    try:
        llm_response = rag_chain.invoke(user_query)
        bot_response_text = llm_response['result']
        print(f"LLM Generated Response for Text Query: {bot_response_text}")

    except Exception as e:
        print(f"Error communicating with LLM for text query: {e}")
        raise HTTPException(status_code=500, detail=f"LLM text processing failed: {e}. Please try again later.")

    return {
        "user_query": user_query,
        "bot_response": bot_response_text
    }

# --- NEW API Endpoint: Live Audio WebSocket ---

# Whisper typically expects 16kHz, mono audio
WHISPER_TARGET_SAMPLE_RATE = 16000 
WHISPER_TARGET_CHANNELS = 1

@app.websocket("/ws/live_audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection accepted.")
    audio_buffer = bytearray() # Buffer to accumulate audio chunks
    temp_input_file_path = None # Initialize to None
    response_audio_path = None # Initialize for cleanup

    try:
        # Loop to receive messages (audio chunks or END_OF_AUDIO signal)
        while True:
            print("DEBUG: Server waiting for message from client...")
            message = await websocket.receive()
            print(f"DEBUG: Server received message type: {message['type']}")

            # In your app.py, inside the websocket_endpoint, within the while True loop:
            if message["type"] == "websocket.receive":
                if "bytes" in message and message["bytes"]:
                    audio_chunk = message["bytes"]
                    print(f"DEBUG: Server received audio chunk, size: {len(audio_chunk)} bytes")
                    audio_buffer.extend(audio_chunk)
                elif "text" in message:
                    text_message = message["text"]
                    print(f"DEBUG: Server received text message: '{text_message}'")
                    if text_message == "END_OF_AUDIO":
                        print("DEBUG: Received END_OF_AUDIO signal. Breaking from receive loop.")
                        break # <-- Break the loop to process audio
                    elif text_message == "PING": # <-- ADD THIS LINE
                        print("DEBUG: Received PING from client. Keeping connection alive.") # <-- ADD THIS LINE
                        # Optionally, send a PONG back if you want, but not strictly necessary for keep-alive
                        # await websocket.send_text("PONG")
                    else: # Handle other unexpected text messages
                        print(f"DEBUG: Server received unhandled text message: '{text_message}'")

            elif message["type"] == "websocket.disconnect":
                # If the client disconnected without sending END_OF_AUDIO,
                # this is an unexpected disconnect. Process what's buffered.
                print(f"DEBUG: Client disconnected during receive loop (code: {message.get('code')}, reason: '{message.get('reason')}').")
                break # Break the loop to process buffered audio

            else:
                print(f"DEBUG: Server received unhandled message type: {message['type']}")

        # --- Audio processing block starts here, AFTER the loop breaks ---
        # This block will execute when END_OF_AUDIO is received or client disconnects
        if audio_buffer:
            print("DEBUG: Processing accumulated audio buffer...")
            try:
                # Create directory for temporary audio files if it doesn't exist
                os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)
                
                # Write the collected audio buffer to a temporary file
                temp_input_file_name = f"ws_input_{uuid.uuid4()}.webm" # Assuming webm from browser
                temp_input_file_path = os.path.join(TEMP_AUDIO_DIR, temp_input_file_name)
                with open(temp_input_file_path, "wb") as f:
                    f.write(audio_buffer)
                print(f"DEBUG: Accumulated audio written to: {temp_input_file_path}")

                # Load the encoded audio using pydub and convert to target format
                start_time = time.time()
                audio_segment = AudioSegment.from_file(temp_input_file_path)
                audio_segment = audio_segment.set_frame_rate(WHISPER_TARGET_SAMPLE_RATE)
                audio_segment = audio_segment.set_channels(WHISPER_TARGET_CHANNELS)
                
                # Export as a temporary WAV file for Whisper
                temp_output_wav_name = f"ws_processed_{uuid.uuid4()}.wav"
                temp_output_wav_path = os.path.join(TEMP_AUDIO_DIR, temp_output_wav_name)
                audio_segment.export(temp_output_wav_path, format="wav")
                print("DEBUG: AudioSegment decoded and exported to WAV.")

                # Transcribe audio using Whisper
                transcribed_text = whisper_model_global.transcribe(temp_output_wav_path)["text"].strip()
                print(f"Transcribed Batch: '{transcribed_text}'")
                
                # Send user transcript back to client
                try:
                    await websocket.send_text(f"User: {transcribed_text}")
                except Exception as e:
                    print(f"WARNING: Could not send User transcript to client: {e}")

                # Get LLM response
                print(f"Sending to LLM: '{transcribed_text}'")
                llm_response = rag_chain.invoke(transcribed_text)
                bot_response_text = llm_response['result'].strip()
                print(f"Bot Response: '{bot_response_text}'")
                
                # Send bot response text
                try:
                    await websocket.send_text(f"TechBot: {bot_response_text}")
                except Exception as e:
                    print(f"WARNING: Could not send TechBot text to client: {e}")

                # Text-to-Speech for bot's response
                response_audio_path = text_to_speech(bot_response_text)
                print(f"DEBUG: TTS audio generated: {response_audio_path}")

                # Calculate and print processing time
                processing_time = time.time() - start_time
                print(f" > Processing time: {processing_time}")
                audio_duration = len(audio_segment) / 1000 # duration in seconds
                if audio_duration > 0:
                    real_time_factor = processing_time / audio_duration
                    print(f" > Real-time factor: {real_time_factor}")
                else:
                    print(" > Real-time factor: N/A (audio duration is zero)")

                with open(response_audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                    # Try to send the audio response
                    try:
                        await websocket.send_bytes(audio_bytes)
                        print(f"DEBUG: Server finished sending bot response audio for '{bot_response_text[:50]}...'")
                    except Exception as send_error:
                        print(f"ERROR: Failed to send bot audio to client: {send_error}")

                # Send SERVER_DONE_TURN signal
                try:
                    await websocket.send_text("SERVER_DONE_TURN")
                    print("DEBUG: Server sent 'SERVER_DONE_TURN' signal.")
                except Exception as send_error:
                    print(f"ERROR: Failed to send SERVER_DONE_TURN to client: {send_error}")

            except Exception as e:
                print(f"ERROR: An error occurred during audio processing or sending: {e}")
                # Try to send an error message to the client, if connection still open
                try:
                    await websocket.send_text(f"TechBot: Server error: {e}")
                except Exception as send_error:
                    print(f"Could not send general error message: {send_error}")

        else:
            print("No audio received from client for this turn, skipping processing.")
            # If no audio, still send SERVER_DONE_TURN to reset client UI
            try:
                await websocket.send_text("SERVER_DONE_TURN")
                print("DEBUG: Server sent 'SERVER_DONE_TURN' for empty audio.")
            except Exception as e:
                print(f"Could not send SERVER_DONE_TURN for empty audio: {e}")


    except WebSocketDisconnect as e:
        print(f"WebSocket disconnected unexpectedly (code: {e.code}, reason: '{e.reason}').")
        # If processing hadn't happened (e.g., client disconnected during receive loop)
        if audio_buffer: # If there's buffered audio, attempt to process it even on unexpected disconnect
            print("DEBUG: Audio buffered, attempting to process on unexpected disconnect.")
            # Re-call the processing block here if you want to ensure it runs
            # For this example, we'll let the main processing block handle it if it was reached.
            # If you want to ensure it runs even if the client disconnects before END_OF_AUDIO,
            # you'd move the entire processing block into this except. For now, it's outside.

    except Exception as e:
        print(f"An unexpected general error occurred in WebSocket: {e}")
        try:
            await websocket.send_text(f"TechBot: An unexpected error occurred. Please try again. Details: {e}")
        except Exception as send_error:
            print(f"Could not send general error message: {send_error}")

    finally:
        # Ensure temporary files are removed regardless of success or failure
        if temp_input_file_path and os.path.exists(temp_input_file_path):
            os.remove(temp_input_file_path)
            print(f"Removed temp input file: {temp_input_file_path}")
        if response_audio_path and os.path.exists(response_audio_path):
            os.remove(response_audio_path)
            print(f"Removed temp output WAV: {response_audio_path}")
        # The connection will remain open unless a true disconnect happened
        print("INFO: WebSocket connection remains open (unless explicit disconnect occurred).")

# --- Endpoint to Serve Generated Audio Files (Unchanged) ---
@app.get("/audio/{file_name}")
async def get_audio_file(file_name: str):
    """Serves generated audio files from the 'response_audio_files' directory."""
    file_path = os.path.join("response_audio_files", file_name)
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=file_name, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="Audio file not found.")