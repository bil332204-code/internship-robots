"""
emotion_control.py — Emotion Mode

Watches your face via the webcam, detects the dominant emotion with
DeepFace, and requires 80% frame agreement over the last 10 frames
before sending the matching reaction to JD, with a 5-second cooldown.
"""

import cv2
from deepface import DeepFace
import time
from collections import Counter

CONFIRMATION_FRAMES = 10
COOLDOWN_SECONDS = 5
GESTURE_FILE = "gesture.txt"

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
    with open(GESTURE_FILE, "w", encoding="utf-8") as f:
        f.write(action)
    print(f"✅ Sent to JD: {action}")


def draw_ui(frame, label, confirmed_action, cooldown_remaining):
    h, w = frame.shape[:2]

    cv2.putText(frame, f"Detecting: {label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    if confirmed_action:
        cv2.putText(frame, f"JD: {confirmed_action}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    if cooldown_remaining > 0:
        bar_width = int((cooldown_remaining / COOLDOWN_SECONDS) * (w - 40))
        cv2.rectangle(frame, (20, h - 40), (20 + bar_width, h - 20), (0, 0, 255), -1)
        cv2.putText(frame, f"Next in: {cooldown_remaining:.1f}s", (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Ready!", (20, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame


def run_emotion():
    cap = cv2.VideoCapture(0)
    detection_buffer = []
    confirmed_action = None
    last_sent_time = 0

    print("😊 Emotion Mode Active! Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_time = time.time()
        cooldown_remaining = max(0, COOLDOWN_SECONDS - (current_time - last_sent_time))
        current_label = "None"

        if cooldown_remaining == 0:
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                emotion = result[0]['dominant_emotion']
                current_label = emotion
                detection_buffer.append(emotion)

                if len(detection_buffer) > CONFIRMATION_FRAMES:
                    detection_buffer.pop(0)

                if len(detection_buffer) == CONFIRMATION_FRAMES:
                    most_common = Counter(detection_buffer).most_common(1)[0]
                    if most_common[1] >= CONFIRMATION_FRAMES * 0.8:
                        confirmed_action = emotion_actions.get(most_common[0], "Wave")
                        send_to_arc(confirmed_action)
                        last_sent_time = current_time
                        detection_buffer = []
            except Exception:
                current_label = "No face"
        else:
            current_label = "Cooling down..."

        frame = draw_ui(frame, current_label, confirmed_action, cooldown_remaining)
        cv2.imshow("Emotion Control 😊", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_emotion()
