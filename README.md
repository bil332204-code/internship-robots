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

There are two ways to run the whole project:

### Option A — Web Dashboard (recommended)
Everything lives on one webpage — camera modes stream live into the
browser, Ask JD/Speech/Find Object work by typing or tapping a mic
button, no terminal menu needed after startup.
```bash
python web_dashboard.py
```
Then open the printed link (`http://localhost:5000`, or the phone-friendly
`http://<your-ip>:5000` link) in a browser.

### Option B — Terminal Menu (classic)
```bash
python main_controller.py
```
Pick a number 0-8, each mode runs in its own console/native window.

Either way, first:
1. Connect JD robot via WiFi (EZ-B v4.x/2 network)
2. Open ARC → Connect JD → Run the Python/EZ-Script bridge skill
3. Activate your virtual environment and `pip install -r requirements.txt`

## 👥 Team
- Bilal Ahmed — Gesture Detection & Backend

## 📅 Internship Project — 2026