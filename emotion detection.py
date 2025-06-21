from transformers import pipeline

pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")

def detect_emotion(text):
    result = pipe(text)
    return result[0]['label'], result[0]['score']
emotion, confidence = detect_emotion("I'will kill you.")
print(f"Emotion: {emotion}, Confidence: {confidence}")
