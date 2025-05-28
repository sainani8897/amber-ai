import torchaudio
from TTS.api import TTS

tts_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")

def text_to_speech(text):
    tts_model.tts_to_file(text=text, file_path="response.wav")
    return "response.wav"