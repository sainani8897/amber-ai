import spacy

nlp = spacy.load("en_core_web_sm")

def get_intent(text):
    doc = nlp(text)
    if "order" in text.lower():
        return "order_status"
    elif "hello" in text.lower() or "hi" in text.lower():
        return "greeting"
    elif "problem" in text.lower() or "issue" in text.lower():
        return "complaint"
    else:
        return "unknown_intent"

# Example usage
print(get_intent("I want to check my order status"))  # Output: order_status