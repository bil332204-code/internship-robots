import socket
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess

def send_to_arc(action):
    with open("C:/Users/hp/Documents/internship-robots/gesture.txt", "w", encoding="utf-8") as f:
        f.write(action)
    print(f"Sent: {action}")

def detect_gesture(hand_landmarks):
    landmarks = hand_landmarks[0]
    
    # Finger tip & base landmarks
    thumb_tip = landmarks[4].y
    thumb_base = landmarks[2].y
    index_tip = landmarks[8].y
    index_base = landmarks[5].y
    middle_tip = landmarks[12].y
    middle_base = landmarks[9].y
    ring_tip = landmarks[16].y
    ring_base = landmarks[13].y
    pinky_tip = landmarks[20].y
    pinky_base = landmarks[17].y

    fingers_up = [
        index_tip < index_base,
        middle_tip < middle_base,
        ring_tip < ring_base,
        pinky_tip < pinky_base,
    ]
    thumb_up = thumb_tip < thumb_base

    count = sum(fingers_up)

    if count >= 4 and not thumb_up:
        return "Wave"           # Open hand
    elif thumb_up and count == 0:
        return "Happy Hands"    # Thumbs up
    elif count == 0 and not thumb_up:
        return "Bow"            # Fist
    elif fingers_up[0] and count == 1:
        return "Point"          # Index finger only
    elif fingers_up[0] and fingers_up[1] and not fingers_up[2] and not fingers_up[3]:
        return "Disco Dance"   # Peace sign
    elif count >= 3 and thumb_up:
        return "Jump Jack"     # All 5 fingers
    elif fingers_up[3] and count == 1:
        return "Kick"          # Pinky only
    return None
# Setup
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_action = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        gesture = detect_gesture(result.hand_landmarks)
        if gesture and gesture != last_action:
            send_to_arc(gesture)
            last_action = gesture
        if gesture:
            cv2.putText(frame, gesture, (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        last_action = None

    cv2.imshow("Gesture Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()