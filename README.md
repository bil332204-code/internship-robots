# internship-robots
Exploring EZ-Robot JD, Alpha Mini, and drones during internship. Documenting experiments and code.
# 🤖 Gesture Controlled EZ-Robot JD

Control EZ-Robot JD using hand gestures via MediaPipe and Python.

## 🛠️ Tech Stack
- Python 3.x
- MediaPipe (Hand Detection)
- OpenCV (Camera Feed)
- FastAPI (Backend)
- ARC/Synthiam (Robot Control)

## 🖐️ Gesture Controls

| Gesture | Action |
|---------|--------|
| ✋ 4 Fingers (no thumb) | Wave |
| 🖐️ All 5 Fingers | Jump Jack |
| 👍 Thumbs Up | Happy Hands |
| ✊ Fist | Bow |
| ✌️ Peace Sign | Disco Dance |
| ☝️ Index Finger | Point |
| 🤙 Pinky Only | Kick |

## 🚀 How to Run

1. Connect JD robot via WiFi (EZ-B v4.x/2 network)
2. Open ARC → Connect JD → Run Python Script skill
3. Activate virtual environment:
```bash
my_venv\Scripts\activate
```
4. Run gesture detection:
```bash
python gesture_control.py
```
5. Show hand gestures to camera → JD moves!

## 👥 Team
- Bilal Ahmed — Gesture Detection & Backend

## 📅 Internship Project — 2026