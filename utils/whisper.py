import whisper

def speech_to_text(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    return result["text"]