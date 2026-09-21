"""
gesture_control.py — Gesture Mode

Watches your hand via the webcam, recognizes a set of gestures, and
requires the SAME gesture to be seen for 80% of the last 10 frames
(confirmation buffer) before sending it to JD, with a 5-second cooldown
after each action -- this avoids JD reacting to a single flickery
misread frame.
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


def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks[0]
    index_tip = landmarks[8].y
    index_pip = landmarks[6].y
    middle_tip = landmarks[12].y
    middle_pip = landmarks[10].y
    ring_tip = landmarks[16].y
    ring_pip = landmarks[14].y
    pinky_tip = landmarks[20].y
    pinky_pip = landmarks[18].y
    thumb_tip_x = landmarks[4].x
    thumb_ip_x = landmarks[3].x

    fingers_up = [
        index_tip < index_pip,
        middle_tip < middle_pip,
        ring_tip < ring_pip,
        pinky_tip < pinky_pip,
    ]
    thumb_up = thumb_tip_x < thumb_ip_x
    count = sum(fingers_up)

    if count == 4 and not thumb_up:
        return "Wave"
    elif count == 4 and thumb_up:
        return "Jump Jack"
    elif thumb_up and count == 0:
        return "Happy Hands"
    elif count == 0 and not thumb_up:
        return "Bow"
    elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return "Disco Dance"
    elif fingers_up[3] and count == 1:
        return "Kick"
    elif fingers_up[0] and count == 1:
        return "Point"
    return None


def run_gesture():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)

    detection_buffer = []
    confirmed_action = None
    last_sent_time = 0

    print("🖐️ Gesture Mode Active! Press Q to quit.")

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

        if result.hand_landmarks and cooldown_remaining == 0:
            gesture = detect_gesture(result.hand_landmarks)
            if gesture:
                current_label = gesture
                detection_buffer.append(gesture)
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
        cv2.imshow("Gesture Control 🖐️", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_gesture()
