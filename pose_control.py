import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

def send_to_arc(action):
    with open("C:/Users/hp/Documents/internship-robots/gesture.txt", "w", encoding="utf-8") as f:
        f.write(action)
    print(f"Sent to JD: {action}")

def detect_pose(landmarks):
    nose = landmarks[0]
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

base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
last_action = None
last_sent_time = 0
DELAY = 3

print("🕺 Pose Detection Started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        action = detect_pose(landmarks)

        if action:
            current_time = time.time()
            if action != last_action and (current_time - last_sent_time) > DELAY:
                send_to_arc(action)
                last_action = action
                last_sent_time = current_time

            cv2.putText(frame, f"Pose: {action}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    else:
        last_action = None

    cv2.imshow("Pose Control", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()