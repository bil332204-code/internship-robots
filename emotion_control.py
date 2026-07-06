import cv2
from deepface import DeepFace
import time

emotion_actions = {
    "happy": "Happy Hands",
    "sad": "Thinking",
    "angry": "Kick",
    "surprise": "Jump Jack",
    "fear": "Bow",
    "neutral": "Wave",
    "disgust": "Situps",
}

def send_to_arc(action):
    with open("C:/Users/hp/Documents/internship-robots/gesture.txt", "w", encoding="utf-8") as f:
        f.write(action)
    print(f"Sent to JD: {action}")

cap = cv2.VideoCapture(0)
last_emotion = None
last_sent_time = 0
DELAY = 5  # seconds between actions

print("😊 Emotion Detection Started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        current_time = time.time()

        if emotion != last_emotion and (current_time - last_sent_time) > DELAY:
            action = emotion_actions.get(emotion, "Wave")
            send_to_arc(action)
            last_emotion = emotion
            last_sent_time = current_time
            print(f"Emotion: {emotion} → JD: {action}")

        cv2.putText(frame, f"Emotion: {emotion}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    except Exception as e:
        cv2.putText(frame, "No face detected", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Emotion Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()