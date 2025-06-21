import openai
import speech_recognition as sr
import pyttsx3
import webbrowser
import joblib
from transformers import pipeline



# Load model once globally
emotion_pipe = pipeline("text-classification", model="michellejieli/emotion_text_classifier")

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Predefined music links
music = {
    "pakistani": "https://www.youtube.com/watch?v=zY99Cjy44JA&pp=ygUFbXVzaWM%3D",
    "whotalha": "https://www.youtube.com/watch?v=a5aOM1EpJmE&pp=ygULdGFsaGEgYW5tdW0%3D",
    "goat": "https://www.youtube.com/watch?v=M8vDwlHigJA&pp=ygUQc2lkaHUgbW9vc2Ugd2FsYQ%3D%3D",
    "sohigh": "https://youtu.be/GgmFC8y8q3k?si=Z6jMgURsvX5DSVXq"
}

emotion_music_links = {
    "joy": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",         # Happy - Pharrell Williams
    "sadness": "https://www.youtube.com/watch?v=ho9rZjlsyYY",     # Sad Piano Music
    "anger": "https://www.youtube.com/watch?v=UfcAVejslrU",       # Calm Down - Relax Music
    "fear": "https://www.youtube.com/watch?v=6zGQSWib32U",        # Reassuring calm music
    "love": "https://www.youtube.com/watch?v=450p7goxZqg",        # Perfect - Ed Sheeran
    "surprise": "https://www.youtube.com/watch?v=0KSOMA3QBU0",    # Something exciting
}

# Load ML model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()


def aiProcess(command):
    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a virtual assistant named Jarvis skilled in general tasks like Alexa and Google Cloud. Give short responses please."},
            {"role": "user", "content": command}
        ]
    )
    reply = response['choices'][0]['message']['content']
    speak(reply)

def predict_fake_news(news_text):
    vec = vectorizer.transform([news_text])
    result = model.predict(vec)[0]
    return "Fake News" if result == 1 else "Real News"


def detect_emotion(text):
    result = emotion_pipe(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

def emotion_detection_voice():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    print("🎙️ Listening for your emotions... Say 'stop' to quit.")

    while True:
        try:
            with mic as source:
                recognizer.adjust_for_ambient_noise(source)
                print("🟢 Speak now...")
                audio = recognizer.listen(source)

            print("🧠 Recognizing...")
            text = recognizer.recognize_google(audio)
            print(f"🗣️ You said: {text}")

            if text.lower() in ["stop", "exit", "quit", "bye"]:
                print("👋 Exiting emotion detection.")
                break

            emotion, confidence = detect_emotion(text)
            print(f"❤️ Detected Emotion: {emotion} ({confidence:.2f})")

            responses = {
                "joy": "😄 I'm glad you're feeling happy!",
                "sadness": "😔 I'm here if you need to talk.",
                "anger": "😠 Take a deep breath. Let's calm down.",
                "fear": "🫂 Don't worry, you're not alone.",
                "love": "❤️ A beautiful emotion indeed!",
                "surprise": "😲 Whoa! That sounds exciting!"
            }
            print(responses.get(emotion, "🤔 Interesting emotion."))

            # Play mood-based music
            if emotion in emotion_music_links:
                print(f"🎵 Playing music for your mood: {emotion}")
                webbrowser.open(emotion_music_links[emotion])

        except sr.UnknownValueError:
            print("❌ Could not understand audio.")
        except sr.RequestError as e:
            print(f"🛑 Error with the speech service: {e}")

def processCommand(command):
    command = command.lower()

    if "open google" in command:
        webbrowser.open("https://google.com")
    elif "open facebook" in command:
        webbrowser.open("https://facebook.com")
    elif "open youtube" in command:
        webbrowser.open("https://youtube.com")
    elif "open linkedin" in command:
        webbrowser.open("https://linkedin.com")
    elif command.startswith("play"):
        song_name = command.split("play", 1)[1].strip()
        link = music.get(song_name)
        if link:
            webbrowser.open(link)
        else:
            speak("Sorry, I couldn't find the song.")
    elif "check this news" in command:
        news_text = command.replace("check this news", "").strip()
        if news_text:
            result = predict_fake_news(news_text)
            speak(f"This news is likely {result}")
        else:
            speak("Please say the news after 'check this news'.")
    elif "stop listening" in command or "go to sleep" in command:
        speak("Going to sleep now. Say 'Jarvis' to wake me up.")
        return False  # signal to stop listening
    
    elif "emotion" in command:
     emotion_detection_voice()

    else:
        aiProcess(command)
    return True  # continue listening

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)
        try:
            command = recognizer.recognize_google(audio)
            print("You:", command)
            return command.lower()
        except:
            speak("Sorry, I didn't catch that.")
            return ""

if __name__ == "__main__":
    speak("Initializing Jarvis...")
    while True:
        trigger = listen()
        if "jarvis" in trigger:
            speak("Yes, how can I help you?")
            while True:
                command = listen()
                if not processCommand(command):
                    break
