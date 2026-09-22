AEGIS-JD

Autonomous Engagement & Guardian Intelligence System

A unified, multi-modal control system for the EZ-Robot JD humanoid robot — built during a robotics internship at Al-Khidmat Foundation Pakistan. AEGIS-JD combines gesture, emotion, pose, and speech control, a conversational AI assistant, object detection, and a face-recognition security gate into a single application controlling one physical robot.

Team
Bilal Ahmed — Group Leader — Gesture/Emotion/Pose/Speech Control, Ask JD (co-built), security-gate integration, system merge
Sameen Fatima — AI Lead — Ask JD conversational assistant (co-built)
Ahmed, Asjid, Sajid — Perception Team — Object detection and face recognition
Overview

The project runs on a companion laptop that communicates with JD over WiFi through EZ-Robot's ARC software, using a file-based command bridge and EZ-Script. All computer vision and AI processing happens on the laptop; JD's mechanical build and ARC's scripting environment are treated as a fixed platform.

Features
🖐️ Gesture, Emotion, Pose & Speech Control

Four parallel input modes drive JD's actions:

Gesture — MediaPipe hand landmarks mapped to seven gestures
Emotion — DeepFace reads dominant facial emotion
Pose — MediaPipe pose landmarks read body posture
Speech — Faster-Whisper transcribes commands locally

Every mode requires a reading to hold across 80% of the last ten frames, followed by a five-second cooldown, before JD acts — preventing single-frame misfires. All four write to the same shared output file.

💬 Ask JD — Conversational AI

A voice-and-text assistant for JD. Speech is transcribed locally via Faster-Whisper; replies come from Ollama-hosted LLaMA models (a fast model for everyday exchanges, a deeper 8B model for involved questions). Up to 50 messages of history are retained for natural follow-ups. Accessible via console or browser chat. Scoped by system prompt to robotics/AI/CS topics.

🔍 Object Detection & Face Recognition
Object detection — YOLOv8 (yolov8n), fully local. "Find Object" starts with Gemini interpreting a request ("find my bottle"), then JD scans five head positions while YOLO checks each photo, points, and announces on a match.
Face recognition — DeepFace (Facenet) + cosine similarity, multiple embeddings per person stored in Firestore with a local offline cache.
🔒 Security Gate

Face recognition doesn't just answer "who is this?" — it gates who JD responds to at all. A person's face is checked against the Firestore embedding store before any gesture, emotion, pose, or speech input is acted on, with a hashed admin-password fallback for resilience.

Architecture
Input (webcam/mic)
      ↓
Perception (MediaPipe, DeepFace, YOLOv8, Faster-Whisper)
      ↓
Security Gate (face recognition authorization check)
      ↓
Decision (rule-based mapping + Ollama LLaMA + Gemini)
      ↓
Output (ARC / EZ-Script → JD's physical actions)
Tech Stack
Category	Technologies
Computer Vision	OpenCV, MediaPipe (Hand & Pose Landmarker)
Face & Emotion AI	DeepFace (Facenet), cosine similarity
Object Detection	YOLOv8 (Ultralytics, yolov8n)
Speech	Faster-Whisper (local)
Conversational AI	Ollama (LLaMA 3.2 / 3.1:8B), Google Gemini
Backend/Web	Flask, threading for concurrent camera streams
Data & Auth	Firebase/Firestore, SHA-256 admin password hash
Robot Control	EZ-Robot ARC + EZ-Script, file-based command bridge
Entry Points
main_controller.py — terminal menu
web_dashboard.py — browser dashboard streaming every camera mode live

Both sit behind the same security gate. Heavy dependencies (Whisper, YOLO, DeepFace, Ollama, Flask) load lazily, only when their mode is selected.

Results
Reliable real-time gesture/emotion/pose control with debounce logic
Two-mode voice interface (Speech Control + Ask JD), no cloud speech dependency
Working "Find Object" combining language understanding, local detection, and physical scanning
Functioning security gate — recognized individuals can trigger JD, unrecognized cannot
One unified system replacing five previously disconnected demos
Future Work
Calibrate face-recognition similarity threshold against real lighting conditions
Move the ARC bridge from file-based communication to a persistent socket/WebSocket connection
Add per-person permission levels (admin vs. guest actions)
Add offline fallbacks so "Find Object" degrades gracefully without internet access
