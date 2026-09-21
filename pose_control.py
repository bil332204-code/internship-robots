"""
pose_control.py — Pose Mode

Watches your body pose via the webcam, and requires 80% frame agreement
over the last 10 frames before sending the matching action to JD, with
a 5-second cooldown.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import Counter

CONFIRMATION_FRAMES = 10
COOLDOWN_SECONDS = 5
GESTURE_FILE = "gesture.txt"


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


def detect_pose(landmarks):
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]

    left_hand_up = left_wrist.y < left_shoulder.y
    right_hand_up = right_wrist.y < right_shoulder.y

    if left_hand_up and right_hand_up:
        return "Jump Jack"
    elif left_hand_up and not right_hand_up:
        return "Wave"
    elif right_hand_up and not left_hand_up:
        return "Happy Hands"
    return None


def run_pose():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, min_pose_detection_confidence=0.7)
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    detection_buffer = []
    confirmed_action = None
    last_sent_time = 0

    print("🕺 Pose Mode Active! Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        current_time = time.time()
        cooldown_remaining = max(0, COOLDOWN_SECONDS - (current_time - last_sent_time))
        current_label = "None"

        if result.pose_landmarks and cooldown_remaining == 0:
            landmarks = result.pose_landmarks[0]
            action = detect_pose(landmarks)
            if action:
                current_label = action
                detection_buffer.append(action)

                if len(detection_buffer) > CONFIRMATION_FRAMES:
                    detection_buffer.pop(0)

                if len(detection_buffer) == CONFIRMATION_FRAMES:
                    most_common = Counter(detection_buffer).most_common(1)[0]
                    if most_common[1] >= CONFIRMATION_FRAMES * 0.8:
                        confirmed_action = most_common[0]
                        send_to_arc(confirmed_action)
                        last_sent_time = current_time
                        detection_buffer = []
            else:
                detection_buffer = []
        elif cooldown_remaining > 0:
            current_label = "Cooling down..."

        frame = draw_ui(frame, current_label, confirmed_action, cooldown_remaining)
        cv2.imshow("Pose Control 🕺", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose()
