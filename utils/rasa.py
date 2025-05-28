from rasa.nlu.model import Interpreter

interpreter = Interpreter.load("models/nlu")

def get_intent(text):
    response = interpreter.parse(text)
    return response["intent"]["name"]