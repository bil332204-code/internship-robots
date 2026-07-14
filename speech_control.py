import speech_recognition as sr

commands = {
    "wave": "Wave",
    "bow": "Bow",
    "dance": "Disco Dance",
    "jump": "Jump Jack",
    "kick": "Kick",
    "point": "Point",
    "happy": "Happy Hands",
}

def send_to_arc(action):
    with open("C:/Users/hp/Documents/internship-robots/gesture.txt", "w", encoding="utf-8") as f:
        f.write(action)
    print(f"Sent to JD: {action}")

recognizer = sr.Recognizer()

print("🎤 Calibrating microphone...")
with sr.Microphone() as source:
    recognizer.adjust_for_ambient_noise(source, duration=1)
print("✅ Ready! Speak now...")

while True:
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=3)

        command = recognizer.recognize_google(audio).lower()
        print(f"Heard: {command}")

        for keyword, action in commands.items():
            if keyword in command:
                send_to_arc(action)
                break

        if "exit" in command:
            print("Stopping...")
            break

    except sr.WaitTimeoutError:
        print("Listening again...")
    except sr.UnknownValueError:
        print("Could not understand, try again...")
    except Exception as e:
        print(f"Error: {e}")