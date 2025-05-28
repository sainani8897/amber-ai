# AI Voice Assistant

This is a real-time, push-to-talk AI Voice Assistant that leverages OpenAI's Whisper for Speech-to-Text (STT), a Large Language Model (LLM) (e.g., GPT-3.5/4) for conversational responses, and Eleven Labs for Text-to-Speech (TTS). The application uses FastAPI for the backend WebSocket communication and a simple HTML/JavaScript frontend for the user interface.

## Features

-   **Real-time Voice Input:** Push-to-talk functionality for continuous audio streaming.
-   **Whisper STT:** Transcribes user speech to text.
-   **LLM Integration:** Processes transcribed text to generate intelligent conversational responses.
-   **Eleven Labs TTS:** Synthesizes realistic human-like speech from the bot's text responses.
-   **WebSocket Communication:** Efficient bidirectional communication between frontend and backend.
-   **Persistent Connection:** WebSocket remains open for multi-turn conversations.

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)
-   An OpenAI API Key
-   An Eleven Labs API Key

### 1. Clone the Repository

```bash
git clone [<your-repository-url>](https://github.com/sainani8897/amber-ai)
cd amber
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # On Windows
#source venv/bin/activate  # On macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the root directory of your project (same level as main.py) and add your OpenAI and Eleven Labs API keys:

```bash
OPENAI_API_KEY="your_openai_api_key_here"
```

### 5. Run the Application 

```bash
uvicorn app:app --reload
```
> **Note:** The `--reload` flag is useful during development as it automatically restarts the server on code changes.

### 6. Access the Frontend
- Open your web browser and navigate to: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)

## Usage

1. **Hold to Speak:** Click and hold the "Hold to Speak" button on the web page. The button will turn red and display "Recording...".
2. **Speak:** Clearly speak into your microphone while holding the button.
3. **Release:** Let go of the button when finished. The button will turn orange and display "Processing..." or "Thinking...".
4. **Listen:** The AI assistant will process your speech and respond with audio playback.
5. **Multi-turn Conversations:** The WebSocket connection stays open, allowing you to hold the button and speak again for additional turns in the conversation.


## Project Structure

```
.
├── app.py                  # FastAPI backend logic (WebSocket endpoint, LLM, TTS)
├── static/                 # Directory for static frontend files
│   └── index.html          # Frontend HTML/JavaScript for the voice assistant UI
├── requirements.txt        # Python dependencies
├── .env.example            # Example of environment variables
├── .gitignore              # Files/folders to ignore in Git
├── README.md               # Project documentation
├── chroma_db/              # Directory for Chroma vector database (if used)
├── temp_websocket_audio/   # Directory for temporary audio input files (created by app.py)
└── response_audio_files/   # Directory for temporary TTS output files (created by app.py)
```

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.

## License

This project is open-source and available under the MIT License.

