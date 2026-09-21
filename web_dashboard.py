"""
web_dashboard.py — JD Command Center (everything on the web)

Run this, open the printed link in a browser, and every mode lives on
its own page from there — no more terminal menu.

  /                dashboard home
  /gesture         live-streamed hand gesture detection
  /emotion         live-streamed facial emotion detection
  /pose            live-streamed body pose detection
  /speech          browser-mic voice commands -> JD actions
  /ask-jd          chat with JD (type or speak)
  /find-object     "find my bottle", "who is this" (type or speak)
  /train-faces     teach JD a new face, step by step

Camera modes (Gesture/Emotion/Pose) share ONE physical webcam, so only
one of them can be streaming at a time -- starting a second one while
another is active shows a friendly warning instead of a crash.

Voice features (Speech Control, Ask JD, Find Object) share ONE Whisper
model, loaded lazily on first use so the dashboard itself starts instantly.
"""

import io
import json
import os
import socket
import tempfile
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, render_template_string, request, jsonify, redirect

app = Flask(__name__)


def make_message_frame(text: str, color=(0, 0, 255)) -> bytes:
    """Renders a plain black frame with a short message on it, encoded
    as JPEG bytes ready to yield in a video stream. Used so camera
    failures show up as a real, visible message in the browser instead
    of an eternally blank/black <img> tag with no explanation."""
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # Basic manual word-wrap so long messages don't run off the frame.
    words = text.split(" ")
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if len(candidate) > 42:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)

    y = 150
    for line in lines[:4]:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y += 34

    ok, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes() if ok else b""

# Face database: create tables if needed, and pull in any pre-existing
# face_model.yml/labels.json as a one-time starting point so already
# trained people aren't lost. Cheap to run, safe to call every startup.
import face_db
face_db.init_db()
face_db.migrate_legacy_files_if_needed()

# ─────────────────────────────────────────────
# Shared Whisper model -- one instance, used by Speech Control, Ask JD,
# and Find Object, loaded the first time ANY of them is actually used.
# ─────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper():
    global _whisper_model
    with _whisper_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel
            print("🎙️ Loading Whisper model (first run may take a moment)...")
            _whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            print("✅ Whisper model ready!")
    return _whisper_model


def transcribe_uploaded_audio(audio_file_storage) -> str:
    """Shared helper: saves an uploaded browser recording and runs Whisper on it."""
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file_storage.save(tmp.name)
        tmp_path = tmp.name
    try:
        model = get_whisper()
        segments, _ = model.transcribe(tmp_path, beam_size=5, vad_filter=True)
        return " ".join(seg.text for seg in segments).strip()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "unknown — run 'ipconfig' (Windows) to find it manually"


# ─────────────────────────────────────────────
# Camera mode lock -- only one of Gesture/Emotion/Pose can hold the
# physical webcam at a time.
# ─────────────────────────────────────────────
camera_state = {"active_mode": None}
camera_lock = threading.Lock()


def try_claim_camera(mode_name: str) -> bool:
    with camera_lock:
        if camera_state["active_mode"] is not None and camera_state["active_mode"] != mode_name:
            return False
        camera_state["active_mode"] = mode_name
        return True


def release_camera(mode_name: str):
    with camera_lock:
        if camera_state["active_mode"] == mode_name:
            camera_state["active_mode"] = None


stream_flags = {"gesture": False, "emotion": False, "pose": False}

# ─────────────────────────────────────────────
# Face ID gate — nobody reaches the dashboard or any mode until JD
# recognizes their face (or they register as a new user). This is
# deliberately a simple global flag, not per-browser login: JD only has
# one camera pointed at whoever is physically standing in front of it,
# so there's only ever one "current user" at a time anyway.
# ─────────────────────────────────────────────
access_state = {"granted": False, "user": None}

PROTECTED_PREFIXES = [
    "/dashboard", "/gesture", "/emotion", "/pose",
    "/speech", "/ask-jd", "/find-object", "/train-faces",
]


@app.before_request
def enforce_face_gate():
    path = request.path
    if not any(path == p or path.startswith(p + "/") for p in PROTECTED_PREFIXES):
        return None  # gate page + its own routes + static files are always reachable

    if not access_state["granted"]:
        if request.method == "GET":
            return redirect("/")
        return jsonify({"error": "Access locked. Scan your face at the gate first."}), 403
    return None

# ─────────────────────────────────────────────
# Shared page shell — every page uses this so the site feels consistent
# ─────────────────────────────────────────────
BASE_STYLE = """
:root {
  --bg: #0f172a; --panel: #1e293b; --accent: #38bdf8; --accent2: #a78bfa;
  --text: #e2e8f0; --muted: #94a3b8; --rec: #ef4444; --good: #22c55e;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text); min-height: 100vh;
}
a { color: var(--accent); text-decoration: none; }
header.top {
  padding: 16px 20px; background: var(--panel); border-bottom: 1px solid #334155;
  display: flex; align-items: center; justify-content: space-between;
}
header.top h1 { margin: 0; font-size: 18px; }
header.top .back { font-size: 14px; color: var(--muted); }
.container { max-width: 720px; margin: 0 auto; padding: 24px 16px; }
.card {
  background: var(--panel); border: 1px solid #334155; border-radius: 14px;
  padding: 16px; margin-bottom: 14px;
}
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
@media (max-width: 520px) { .grid { grid-template-columns: 1fr; } }
.mode-card {
  display: block; background: var(--panel); border: 1px solid #334155; border-radius: 14px;
  padding: 18px; transition: border-color 0.15s ease, transform 0.1s ease;
}
.mode-card:hover { border-color: var(--accent); transform: translateY(-2px); }
.mode-card .emoji { font-size: 28px; }
.mode-card .title { font-weight: 600; margin-top: 8px; color: var(--text); }
.mode-card .desc { font-size: 13px; color: var(--muted); margin-top: 4px; }
button {
  border: none; border-radius: 999px; padding: 10px 20px; font-weight: 600;
  cursor: pointer; font-size: 14px;
}
button.primary { background: var(--accent); color: #04121f; }
button.danger { background: var(--rec); color: white; }
button:disabled { opacity: 0.45; cursor: not-allowed; }
.status { font-size: 13px; color: var(--muted); margin-top: 10px; min-height: 18px; }
.status.good { color: var(--good); }
.video-wrap {
  background: black; border-radius: 12px; overflow: hidden; display: flex;
  align-items: center; justify-content: center; min-height: 320px;
}
.video-wrap img { width: 100%; display: block; }
.chat-log {
  display: flex; flex-direction: column; gap: 10px; max-height: 50vh; overflow-y: auto;
  margin-bottom: 12px;
}
.bubble { max-width: 82%; padding: 10px 14px; border-radius: 16px; line-height: 1.4; font-size: 14px; white-space: pre-wrap; }
.user { align-self: flex-end; background: var(--accent); color: #04121f; border-bottom-right-radius: 4px; }
.jd { align-self: flex-start; background: #334155; border-bottom-left-radius: 4px; }
.jd.thinking { color: var(--muted); font-style: italic; }
.input-row { display: flex; gap: 8px; align-items: center; }
input[type="text"] {
  flex: 1; padding: 12px 14px; border-radius: 999px; border: none;
  background: #0b1220; color: var(--text); font-size: 14px; outline: none;
}
#mic-btn {
  width: 52px; height: 52px; min-width: 52px; border-radius: 50%; border: none;
  background: var(--accent2); color: #0b1220; font-size: 22px; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
}
#mic-btn.recording { background: var(--rec); animation: pulse 1s infinite; }
@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.6); }
  70% { box-shadow: 0 0 0 14px rgba(239,68,68,0); }
  100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
ul.people { padding-left: 20px; color: var(--muted); }
"""


def page_shell(title: str, body_html: str, script: str = "") -> str:
    user_badge = (
        f'<span style="color:var(--muted); font-size:13px; margin-right:12px;">👤 {access_state["user"]}</span>'
        f'<button onclick="fetch(\'/gate/lock\',{{method:\'POST\'}}).then(()=>location.href=\'/\')" '
        f'style="background:#334155; color:var(--muted); padding:6px 14px; font-size:12px;">🔒 Lock</button>'
        if access_state["granted"] else ""
    )
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — JD Command Center</title>
<style>{BASE_STYLE}</style>
</head>
<body>
<header class="top">
  <h1>🤖 {title}</h1>
  <div style="display:flex; align-items:center;">
    {user_badge}
    <a class="back" href="/dashboard" style="margin-left:12px;">← Dashboard</a>
  </div>
</header>
<div class="container">
{body_html}
</div>
<script>{script}</script>
</body>
</html>
"""


MODES = [
    ("gesture", "🖐️", "Gesture Control", "Hand signals -> action, live camera feed"),
    ("emotion", "😊", "Emotion Control", "Facial expression -> action, live camera feed"),
    ("pose", "🕺", "Pose Control", "Body pose -> action, live camera feed"),
    ("speech", "🎤", "Speech Control", "Say a command, JD performs it"),
    ("ask-jd", "💬", "Ask JD", "Chat with JD — type or speak"),
    ("find-object", "🔍", "Find Object", '"find my bottle", "who is this"'),
    ("train-faces", "👤", "Train New Faces", "Teach JD to recognize someone new"),
]


@app.route("/dashboard")
def dashboard():
    cards = "\n".join(
        f"""<a class="mode-card" href="/{slug}">
              <div class="emoji">{emoji}</div>
              <div class="title">{title}</div>
              <div class="desc">{desc}</div>
            </a>"""
        for slug, emoji, title, desc in MODES
    )
    body = f'<div class="grid">{cards}</div>'
    return page_shell("Command Center", body)


# =============================================================================
# FACE ID GATE — the new front door. JD looks at whoever's there; if it
# recognizes them, they get access to everything above. If not, it offers
# to register them as a new user using that same face.
# =============================================================================
GATE_STYLE = BASE_STYLE + """
body { display:flex; align-items:center; justify-content:center; }
.gate-box { max-width: 420px; width: 100%; padding: 20px; text-align: center; }
.gate-box .robot { font-size: 56px; }
.gate-box h1 { margin: 10px 0 4px; font-size: 22px; }
.gate-box p.sub { color: var(--muted); font-size: 14px; margin-bottom: 24px; }
#scan-btn {
  width: 100%; padding: 16px; font-size: 16px; background: var(--accent); color: #04121f;
}
#result { margin-top: 18px; font-size: 14px; min-height: 20px; }
#result.good { color: var(--good); }
#result.bad { color: var(--rec); }
#register-panel { display:none; text-align:left; margin-top: 20px; }
"""


@app.route("/")
def gate_page():
    if access_state["granted"]:
        return redirect("/dashboard")

    from face_trainer import PHOTOS_PER_PERSON

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>JD Security Gate</title>
<style>{GATE_STYLE}</style>
</head>
<body>
<div class="gate-box">
  <div class="robot">🔒🤖</div>
  <h1>JD Security Gate</h1>
  <p class="sub">JD needs to recognize your face before you can control anything.</p>

  <div class="card" style="text-align:left;">
    <div style="display:flex; gap:18px; justify-content:center;">
      <label style="display:flex; align-items:center; gap:6px; font-size:14px;">
        <input type="radio" name="cam-source" value="webcam" checked> Laptop webcam
      </label>
      <label style="display:flex; align-items:center; gap:6px; font-size:14px;">
        <input type="radio" name="cam-source" value="jd"> JD's camera
      </label>
    </div>
  </div>

  <div class="card">
    <div id="result">Starting camera scan automatically...</div>
    <button class="primary" id="scan-btn" onclick="scanFace()" style="margin-top:10px;">📸 Scan Now</button>
  </div>

  <div class="card" id="register-panel">
    <b>New here?</b>
    <p style="color:var(--muted); font-size:13px;">Register the face JD just saw so it recognizes you next time.</p>
    <div class="input-row">
      <input type="text" id="reg-name" placeholder="Your name" autocomplete="off" />
    </div>
    <div style="margin-top:12px; display:flex; gap:10px;">
      <button class="primary" id="reg-capture-btn" onclick="captureRegPhoto()">📸 Capture Photo</button>
      <button class="danger" id="reg-finish-btn" onclick="finishRegistration()" disabled>Finish &amp; Enter</button>
    </div>
    <div class="status" id="reg-status"></div>
  </div>

  <p style="margin-top:24px;">
    <a href="#" onclick="document.getElementById('admin-panel').style.display='block'; return false;"
       style="color:var(--muted); font-size:12px;">⚙️ Admin</a>
  </p>
  <div class="card" id="admin-panel" style="display:none; text-align:left;">
    <b>Admin login (bypass face recognition)</b>
    <p style="color:var(--muted); font-size:13px;">
      Use the shared admin password to get in without a face scan.
    </p>
    <div class="input-row">
      <input type="password" id="admin-login-password" placeholder="Admin password" autocomplete="off" />
    </div>
    <div style="margin-top:12px;">
      <button class="primary" onclick="adminLogin()">🔑 Log In as Admin</button>
    </div>
    <div class="status" id="admin-login-status"></div>

    <hr style="border-color:#334155; margin:20px 0;">

    <b>⚠️ Reset ALL registered faces</b>
    <p style="color:var(--muted); font-size:13px;">
      This deletes every registered person and their photos. Everyone will need
      to register from scratch. Requires the admin password.
    </p>
    <div class="input-row">
      <input type="text" id="admin-password" placeholder="Admin password" autocomplete="off" />
    </div>
    <div style="margin-top:12px;">
      <button class="danger" onclick="resetAllFaces()">🗑️ Wipe All Faces</button>
    </div>
    <div class="status" id="admin-status"></div>
  </div>
</div>

<script>
const PHOTOS_PER_PERSON = {PHOTOS_PER_PERSON};
let regPhotoCount = 0;
let autoScanTimer = null;
let autoScanPaused = false;  // paused while registration panel is open, so we
                              // don't fight the user for the camera mid-flow

function getCameraSource() {{
  return document.querySelector('input[name="cam-source"]:checked').value;
}}

function startAutoScan() {{
  stopAutoScan();
  autoScanTimer = setInterval(() => {{
    if (!autoScanPaused) scanFace(true);
  }}, 2500);
}}

function stopAutoScan() {{
  if (autoScanTimer) {{ clearInterval(autoScanTimer); autoScanTimer = null; }}
}}

async function scanFace(isAuto) {{
  const btn = document.getElementById('scan-btn');
  const result = document.getElementById('result');
  if (!isAuto) btn.disabled = true;
  result.className = ''; result.textContent = isAuto ? 'Looking for a face...' : 'Scanning... look at the camera.';

  try {{
    const res = await fetch('/gate/scan', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ camera_source: getCameraSource() }})
    }});
    const data = await res.json();

    if (data.status === 'recognized') {{
      stopAutoScan();
      result.className = 'good';
      result.textContent = `✅ Welcome back, ${{data.name}}! Opening dashboard...`;
      setTimeout(() => location.href = '/dashboard', 1200);
    }} else if (data.status === 'unknown') {{
      autoScanPaused = true;
      result.className = 'bad';
      result.textContent = data.message || "❌ Face not recognized.";
      document.getElementById('register-panel').style.display = 'block';
    }} else if (data.status === 'no_face') {{
      if (!isAuto) {{
        result.className = 'bad';
        result.textContent = '⚠️ ' + data.message;
      }} else {{
        result.textContent = 'No one in frame yet -- step in front of the camera.';
      }}
      btn.disabled = false;
    }} else {{
      result.className = 'bad';
      result.textContent = '⚠️ ' + (data.message || 'Something went wrong.');
      btn.disabled = false;
    }}
  }} catch (err) {{
    result.className = 'bad';
    result.textContent = 'Error: ' + err.message;
    btn.disabled = false;
  }}
}}

async function captureRegPhoto() {{
  const name = document.getElementById('reg-name').value.trim();
  const status = document.getElementById('reg-status');
  if (!name) {{ status.textContent = 'Enter your name first.'; return; }}

  document.getElementById('reg-capture-btn').disabled = true;
  status.textContent = 'Requesting photo...';
  try {{
    const res = await fetch('/gate/register/capture', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ name: name, photo_number: regPhotoCount + 1, camera_source: getCameraSource() }})
    }});
    const data = await res.json();
    if (data.ok) {{
      regPhotoCount += 1;
      status.textContent = `✅ ${{data.message}} (${{regPhotoCount}}/${{PHOTOS_PER_PERSON}})`;
      if (regPhotoCount >= PHOTOS_PER_PERSON) {{
        document.getElementById('reg-finish-btn').disabled = false;
      }}
    }} else {{
      status.textContent = '⚠️ ' + data.message;
    }}
  }} catch (err) {{
    status.textContent = 'Error: ' + err.message;
  }} finally {{
    if (regPhotoCount < PHOTOS_PER_PERSON) document.getElementById('reg-capture-btn').disabled = false;
  }}
}}

async function finishRegistration() {{
  const name = document.getElementById('reg-name').value.trim();
  const status = document.getElementById('reg-status');
  document.getElementById('reg-finish-btn').disabled = true;
  document.getElementById('reg-capture-btn').disabled = true;
  status.textContent = 'Training JD to recognize you...';
  try {{
    const res = await fetch('/gate/register/finish', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ name: name }})
    }});
    const data = await res.json();
    status.textContent = data.message;
    if (data.ok) setTimeout(() => location.href = '/dashboard', 1200);
  }} catch (err) {{
    status.textContent = 'Error: ' + err.message;
  }}
}}

async function resetAllFaces() {{
  const password = document.getElementById('admin-password').value;
  const status = document.getElementById('admin-status');
  if (!password) {{ status.textContent = 'Enter the admin password.'; return; }}
  if (!confirm('This deletes every registered face permanently. Are you sure?')) return;

  status.textContent = 'Wiping database...';
  try {{
    const res = await fetch('/admin/reset', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ password: password }})
    }});
    const data = await res.json();
    status.textContent = data.message;
    if (data.ok) setTimeout(() => location.reload(), 1500);
  }} catch (err) {{
    status.textContent = 'Error: ' + err.message;
  }}
}}

async function adminLogin() {{
  const password = document.getElementById('admin-login-password').value;
  const status = document.getElementById('admin-login-status');
  if (!password) {{ status.textContent = 'Enter the admin password.'; return; }}

  status.textContent = 'Checking...';
  try {{
    const res = await fetch('/gate/admin-login', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ password: password }})
    }});
    const data = await res.json();
    status.textContent = data.message;
    if (data.ok) {{
      stopAutoScan();
      setTimeout(() => location.href = '/dashboard', 800);
    }}
  }} catch (err) {{
    status.textContent = 'Error: ' + err.message;
  }}
}}

// Face recognition starts scanning on its own the moment this page
// loads -- no button click needed. The manual "Scan Now" button and
// admin-password panel both still work as fallbacks alongside it.
startAutoScan();
</script>
</body>
</html>
"""


@app.route("/gate/scan", methods=["POST"])
def gate_scan():
    try:
        from face_trainer import capture_snapshot
        from face_recognizer import recognize_people
    except Exception as e:
        return jsonify({"status": "error", "message": f"Setup problem: {e}"})

    data = request.get_json(silent=True) or {}
    camera_source = (data.get("camera_source") or "jd").strip().lower()

    if camera_source == "webcam":
        if not try_claim_camera("gate"):
            return jsonify({"status": "error", "message": "Laptop webcam is busy with another mode right now."})
        try:
            got_photo = capture_snapshot("webcam")
        finally:
            release_camera("gate")
        if not got_photo:
            return jsonify({"status": "error", "message": "Couldn't access the laptop webcam."})
    else:
        if not capture_snapshot("jd"):
            return jsonify({"status": "error", "message": "Timed out waiting for JD's camera. Is ARC_SCRIPT.py running?"})

    try:
        results = recognize_people("snapshot.jpg")
    except FileNotFoundError:
        # Nobody has ever registered (or the DB was just wiped) -- there's
        # no model to compare against yet. This is NOT a dead end: treat
        # it exactly like "unknown face" so the self-service registration
        # panel shows up, same as it would for any unrecognized face.
        return jsonify({
            "status": "unknown",
            "message": "No one is registered yet -- register below to get started.",
        })

    if not results:
        return jsonify({"status": "no_face", "message": "No face detected -- step in front of the camera and try again."})

    # Only the FIRST face matters here -- one person at the gate at a time.
    first_match = results[0]
    if first_match != "Unknown":
        access_state["granted"] = True
        access_state["user"] = first_match
        return jsonify({"status": "recognized", "name": first_match})

    return jsonify({"status": "unknown"})


@app.route("/gate/register/capture", methods=["POST"])
def gate_register_capture():
    from face_trainer import capture_one_photo

    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    photo_number = int(data.get("photo_number", 1))
    camera_source = (data.get("camera_source") or "jd").strip().lower()
    if not name:
        return jsonify({"ok": False, "message": "No name given."}), 400

    if camera_source == "webcam":
        if not try_claim_camera("gate"):
            return jsonify({"ok": False, "message": "Laptop webcam is busy with another mode right now."})
        try:
            result = capture_one_photo(name, photo_number, camera_source="webcam")
        finally:
            release_camera("gate")
    else:
        result = capture_one_photo(name, photo_number, camera_source="jd")

    return jsonify(result)


@app.route("/gate/register/finish", methods=["POST"])
def gate_register_finish():
    from face_trainer import register_person

    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "message": "No name given."}), 400

    try:
        result = register_person(name)
    except Exception as e:
        return jsonify({"ok": False, "message": f"Registration failed: {e}"}), 500

    if not result["ok"]:
        return jsonify(result)

    access_state["granted"] = True
    access_state["user"] = name
    return jsonify(result)


@app.route("/gate/admin-login", methods=["POST"])
def gate_admin_login():
    """Alternate login path: the shared admin password bypasses face
    recognition entirely. Either method independently grants access --
    this does NOT replace face recognition, it's a fallback (e.g. face
    recognition failing mid-demo, or a new laptop with no camera set up)."""
    data = request.get_json(force=True)
    password = data.get("password") or ""

    if not face_db.verify_admin_password(password):
        return jsonify({"ok": False, "message": "❌ Incorrect admin password."})

    access_state["granted"] = True
    access_state["user"] = "Admin"
    return jsonify({"ok": True, "message": "✅ Admin access granted."})


@app.route("/gate/lock", methods=["POST"])
def gate_lock():
    access_state["granted"] = False
    access_state["user"] = None
    return jsonify({"ok": True})


@app.route("/admin/reset", methods=["POST"])
def admin_reset():
    data = request.get_json(force=True)
    password = data.get("password") or ""

    result = face_db.reset_all_faces(password)
    if result["ok"]:
        import face_recognizer
        face_recognizer._reload_model()
        # Wiping faces means nobody is a verified user anymore either.
        access_state["granted"] = False
        access_state["user"] = None

    return jsonify(result)


# =============================================================================
# GESTURE MODE — live streamed hand gesture detection
# =============================================================================
@app.route("/gesture")
def gesture_page():
    body = """
<div class="card">
  <div class="video-wrap"><img id="feed" src="" /></div>
  <div style="margin-top:12px; display:flex; gap:10px;">
    <button class="primary" id="start-btn" onclick="startStream()">Start</button>
    <button class="danger" id="stop-btn" onclick="stopStream()" disabled>Stop</button>
  </div>
  <div class="status" id="status">Camera is off. Press Start.</div>
</div>
<div class="card">
  <b>Gestures:</b>
  <div style="color:var(--muted); font-size:13px; margin-top:6px; line-height:1.7;">
    ✋ 4 fingers -> Wave &nbsp;|&nbsp; 🖐️ 5 fingers + thumb -> Jump Jack &nbsp;|&nbsp;
    👍 Thumbs up -> Happy Hands &nbsp;|&nbsp; ✊ Fist -> Bow &nbsp;|&nbsp;
    ✌️ Peace -> Disco Dance &nbsp;|&nbsp; ☝️ Point finger -> Point &nbsp;|&nbsp;
    🤙 Pinky -> Kick
  </div>
</div>
"""
    script = """
const img = document.getElementById('feed');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

async function startStream() {
  const res = await fetch('/gesture/start', {method: 'POST'});
  const data = await res.json();
  if (!data.ok) { statusEl.textContent = data.message; return; }
  statusEl.textContent = 'Starting camera... first use can take up to 15 seconds.';
  statusEl.className = 'status';
  img.onerror = () => {
    statusEl.textContent = '⚠️ Camera stream failed to connect. Try Stop, then Start again.';
    statusEl.className = 'status bad';
  };
  img.onload = () => {
    statusEl.textContent = 'Streaming — show a gesture to the camera.';
    statusEl.className = 'status good';
  };
  img.src = '/gesture/feed?t=' + Date.now();
  startBtn.disabled = true; stopBtn.disabled = false;
}
async function stopStream() {
  await fetch('/gesture/stop', {method: 'POST'});
  img.src = '';
  startBtn.disabled = false; stopBtn.disabled = true;
  statusEl.textContent = 'Camera is off. Press Start.';
  statusEl.className = 'status';
}
window.addEventListener('beforeunload', () => { fetch('/gesture/stop', {method: 'POST'}); });
"""
    return page_shell("Gesture Control", body, script)


@app.route("/gesture/start", methods=["POST"])
def gesture_start():
    if not try_claim_camera("gesture"):
        return jsonify({"ok": False, "message": f"Camera is busy with {camera_state['active_mode']} mode. Stop that first."})
    stream_flags["gesture"] = True
    return jsonify({"ok": True})


@app.route("/gesture/stop", methods=["POST"])
def gesture_stop():
    stream_flags["gesture"] = False
    release_camera("gesture")
    return jsonify({"ok": True})


def gesture_frame_generator():
    from gesture_control import detect_gesture, CONFIRMATION_FRAMES, COOLDOWN_SECONDS
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp
    from collections import Counter

    base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = mp_vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = mp_vision.HandLandmarker.create_from_options(options)

    # CAP_DSHOW (DirectShow) instead of the default MSMF backend --
    # MSMF is known to be slow to open and occasionally fails silently
    # on many Windows laptop webcam drivers (this matches the
    # "cap_msmf.cpp OnReadSample() error" warnings seen in this
    # project's own logs). DSHOW opens faster and more reliably.
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               make_message_frame("Could not open the webcam. Is it in use by another app?") +
               b'\r\n')
        cap.release()
        stream_flags["gesture"] = False
        release_camera("gesture")
        return

    # Warm-up: a freshly opened webcam's first few frames are often
    # dark/stale/failed reads -- discard a handful before streaming.
    for _ in range(5):
        cap.read()

    detection_buffer = []
    confirmed_action = None
    last_sent_time = 0

    try:
        while stream_flags["gesture"]:
            ret, frame = cap.read()
            if not ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       make_message_frame("Lost the webcam feed. Try Stop, then Start again.") +
                       b'\r\n')
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
                            with open("gesture.txt", "w", encoding="utf-8") as f:
                                f.write(confirmed_action)
                            last_sent_time = current_time
                            detection_buffer = []
                else:
                    detection_buffer = []
            elif cooldown_remaining > 0:
                current_label = "Cooling down..."

            h, w = frame.shape[:2]
            cv2.putText(frame, f"Detecting: {current_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            if confirmed_action:
                cv2.putText(frame, f"JD: {confirmed_action}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            if cooldown_remaining > 0:
                bar_width = int((cooldown_remaining / COOLDOWN_SECONDS) * (w - 40))
                cv2.rectangle(frame, (20, h - 30), (20 + bar_width, h - 15), (0, 0, 255), -1)

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        stream_flags["gesture"] = False
        release_camera("gesture")


@app.route("/gesture/feed")
def gesture_feed():
    return Response(gesture_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================================================
# EMOTION MODE — live streamed facial emotion detection
# =============================================================================
@app.route("/emotion")
def emotion_page():
    body = """
<div class="card">
  <div class="video-wrap"><img id="feed" src="" /></div>
  <div style="margin-top:12px; display:flex; gap:10px;">
    <button class="primary" id="start-btn" onclick="startStream()">Start</button>
    <button class="danger" id="stop-btn" onclick="stopStream()" disabled>Stop</button>
  </div>
  <div class="status" id="status">Camera is off. Press Start.</div>
</div>
<div class="card">
  <b>Emotions:</b>
  <div style="color:var(--muted); font-size:13px; margin-top:6px; line-height:1.7;">
    😀 Happy -> Happy Hands &nbsp;|&nbsp; 😢 Sad -> Thinking &nbsp;|&nbsp;
    😠 Angry -> Kick &nbsp;|&nbsp; 😲 Surprise -> Jump Jack &nbsp;|&nbsp;
    😨 Fear -> Bow &nbsp;|&nbsp; 😐 Neutral -> Wave &nbsp;|&nbsp; 🤢 Disgust -> Situps
  </div>
</div>
"""
    script = """
const img = document.getElementById('feed');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

async function startStream() {
  const res = await fetch('/emotion/start', {method: 'POST'});
  const data = await res.json();
  if (!data.ok) { statusEl.textContent = data.message; return; }
  statusEl.textContent = 'Starting camera & emotion model... first use can take up to 30 seconds.';
  statusEl.className = 'status';
  img.onerror = () => {
    statusEl.textContent = '⚠️ Camera stream failed to connect. Try Stop, then Start again.';
    statusEl.className = 'status bad';
  };
  img.onload = () => {
    statusEl.textContent = 'Streaming — look at the camera.';
    statusEl.className = 'status good';
  };
  img.src = '/emotion/feed?t=' + Date.now();
  startBtn.disabled = true; stopBtn.disabled = false;
}
async function stopStream() {
  await fetch('/emotion/stop', {method: 'POST'});
  img.src = '';
  startBtn.disabled = false; stopBtn.disabled = true;
  statusEl.textContent = 'Camera is off. Press Start.';
  statusEl.className = 'status';
}
window.addEventListener('beforeunload', () => { fetch('/emotion/stop', {method: 'POST'}); });
"""
    return page_shell("Emotion Control", body, script)


@app.route("/emotion/start", methods=["POST"])
def emotion_start():
    if not try_claim_camera("emotion"):
        return jsonify({"ok": False, "message": f"Camera is busy with {camera_state['active_mode']} mode. Stop that first."})
    stream_flags["emotion"] = True
    return jsonify({"ok": True})


@app.route("/emotion/stop", methods=["POST"])
def emotion_stop():
    stream_flags["emotion"] = False
    release_camera("emotion")
    return jsonify({"ok": True})


def emotion_frame_generator():
    from deepface import DeepFace
    from emotion_control import emotion_actions, CONFIRMATION_FRAMES, COOLDOWN_SECONDS
    from collections import Counter

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               make_message_frame("Could not open the webcam. Is it in use by another app?") +
               b'\r\n')
        cap.release()
        stream_flags["emotion"] = False
        release_camera("emotion")
        return

    for _ in range(5):
        cap.read()

    detection_buffer = []
    confirmed_action = None
    last_sent_time = 0

    try:
        while stream_flags["emotion"]:
            ret, frame = cap.read()
            if not ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       make_message_frame("Lost the webcam feed. Try Stop, then Start again.") +
                       b'\r\n')
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
                            with open("gesture.txt", "w", encoding="utf-8") as f:
                                f.write(confirmed_action)
                            last_sent_time = current_time
                            detection_buffer = []
                except Exception as e:
                    # DeepFace downloads its emotion model weights from the
                    # internet the FIRST time it's ever used, if not already
                    # cached on disk. If this laptop is connected to JD's own
                    # WiFi network at that moment (no general internet, per
                    # this project's setup), that download hangs/fails and
                    # looks exactly like a frozen black screen with no clue
                    # why -- surface it clearly instead of hiding it as "No face".
                    error_text = str(e).lower()
                    if "connection" in error_text or "download" in error_text or "url" in error_text:
                        current_label = "No internet - can't download emotion model (1st run only)"
                    else:
                        current_label = "No face"
            else:
                current_label = "Cooling down..."

            h, w = frame.shape[:2]
            cv2.putText(frame, f"Detecting: {current_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            if confirmed_action:
                cv2.putText(frame, f"JD: {confirmed_action}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            if cooldown_remaining > 0:
                bar_width = int((cooldown_remaining / COOLDOWN_SECONDS) * (w - 40))
                cv2.rectangle(frame, (20, h - 30), (20 + bar_width, h - 15), (0, 0, 255), -1)

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        stream_flags["emotion"] = False
        release_camera("emotion")


@app.route("/emotion/feed")
def emotion_feed():
    return Response(emotion_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================================================
# POSE MODE — live streamed body pose detection
# =============================================================================
@app.route("/pose")
def pose_page():
    body = """
<div class="card">
  <div class="video-wrap"><img id="feed" src="" /></div>
  <div style="margin-top:12px; display:flex; gap:10px;">
    <button class="primary" id="start-btn" onclick="startStream()">Start</button>
    <button class="danger" id="stop-btn" onclick="stopStream()" disabled>Stop</button>
  </div>
  <div class="status" id="status">Camera is off. Press Start. Stand back so your shoulders and wrists are visible.</div>
</div>
<div class="card">
  <b>Poses:</b>
  <div style="color:var(--muted); font-size:13px; margin-top:6px; line-height:1.7;">
    🙌 Both hands up -> Jump Jack &nbsp;|&nbsp; 🤚 Left hand up -> Wave &nbsp;|&nbsp;
    ✋ Right hand up -> Happy Hands
  </div>
</div>
"""
    script = """
const img = document.getElementById('feed');
const statusEl = document.getElementById('status');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

async function startStream() {
  const res = await fetch('/pose/start', {method: 'POST'});
  const data = await res.json();
  if (!data.ok) { statusEl.textContent = data.message; return; }
  statusEl.textContent = 'Starting camera... first use can take up to 15 seconds.';
  statusEl.className = 'status';
  img.onerror = () => {
    statusEl.textContent = '⚠️ Camera stream failed to connect. Try Stop, then Start again.';
    statusEl.className = 'status bad';
  };
  img.onload = () => {
    statusEl.textContent = 'Streaming — step back so your upper body is visible.';
    statusEl.className = 'status good';
  };
  img.src = '/pose/feed?t=' + Date.now();
  startBtn.disabled = true; stopBtn.disabled = false;
}
async function stopStream() {
  await fetch('/pose/stop', {method: 'POST'});
  img.src = '';
  startBtn.disabled = false; stopBtn.disabled = true;
  statusEl.textContent = 'Camera is off. Press Start.';
  statusEl.className = 'status';
}
window.addEventListener('beforeunload', () => { fetch('/pose/stop', {method: 'POST'}); });
"""
    return page_shell("Pose Control", body, script)


@app.route("/pose/start", methods=["POST"])
def pose_start():
    if not try_claim_camera("pose"):
        return jsonify({"ok": False, "message": f"Camera is busy with {camera_state['active_mode']} mode. Stop that first."})
    stream_flags["pose"] = True
    return jsonify({"ok": True})


@app.route("/pose/stop", methods=["POST"])
def pose_stop():
    stream_flags["pose"] = False
    release_camera("pose")
    return jsonify({"ok": True})


def pose_frame_generator():
    from pose_control import detect_pose, CONFIRMATION_FRAMES, COOLDOWN_SECONDS
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import mediapipe as mp
    from collections import Counter

    base_options = mp_python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = mp_vision.PoseLandmarkerOptions(base_options=base_options, min_pose_detection_confidence=0.7)
    detector = mp_vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               make_message_frame("Could not open the webcam. Is it in use by another app?") +
               b'\r\n')
        cap.release()
        stream_flags["pose"] = False
        release_camera("pose")
        return

    for _ in range(5):
        cap.read()

    detection_buffer = []
    confirmed_action = None
    last_sent_time = 0

    try:
        while stream_flags["pose"]:
            ret, frame = cap.read()
            if not ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       make_message_frame("Lost the webcam feed. Try Stop, then Start again.") +
                       b'\r\n')
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
                            with open("gesture.txt", "w", encoding="utf-8") as f:
                                f.write(confirmed_action)
                            last_sent_time = current_time
                            detection_buffer = []
                else:
                    detection_buffer = []
            elif cooldown_remaining > 0:
                current_label = "Cooling down..."

            h, w = frame.shape[:2]
            cv2.putText(frame, f"Detecting: {current_label}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            if confirmed_action:
                cv2.putText(frame, f"JD: {confirmed_action}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
            if cooldown_remaining > 0:
                bar_width = int((cooldown_remaining / COOLDOWN_SECONDS) * (w - 40))
                cv2.rectangle(frame, (20, h - 30), (20 + bar_width, h - 15), (0, 0, 255), -1)

            ok, buffer = cv2.imencode('.jpg', frame)
            if not ok:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        cap.release()
        stream_flags["pose"] = False
        release_camera("pose")


@app.route("/pose/feed")
def pose_feed():
    return Response(pose_frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')


# =============================================================================
# SPEECH CONTROL — browser mic -> Whisper -> matched action -> JD
# =============================================================================
@app.route("/speech")
def speech_page():
    body = """
<div class="card">
  <div style="text-align:center; padding: 10px 0;">
    <button id="mic-btn" style="width:84px;height:84px;font-size:32px;">🎤</button>
  </div>
  <div class="status" id="status" style="text-align:center;">Tap the mic and say a command, e.g. "wave" or "sit down"</div>
</div>
<div class="card">
  <b>Recent commands</b>
  <div class="chat-log" id="log" style="margin-top:10px;"></div>
</div>
"""
    script = """
const micBtn = document.getElementById('mic-btn');
const statusEl = document.getElementById('status');
const logEl = document.getElementById('log');
let mediaRecorder, audioChunks = [], isRecording = false;

function addLog(text, cls) {
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + cls;
  bubble.textContent = text;
  logEl.appendChild(bubble);
  logEl.scrollTop = logEl.scrollHeight;
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();
  isRecording = true;
  micBtn.classList.add('recording');
  statusEl.textContent = 'Listening... tap again to stop';
}

function stopRecording() {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
  isRecording = false;
  micBtn.classList.remove('recording');
}

async function handleStop() {
  micBtn.disabled = true;
  statusEl.textContent = 'Thinking...';
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const form = new FormData();
  form.append('audio', blob, 'command.webm');
  try {
    const res = await fetch('/speech/command', { method: 'POST', body: form });
    const data = await res.json();
    if (data.heard) addLog('Heard: "' + data.heard + '"', 'user');
    addLog(data.message, 'jd');
    statusEl.textContent = 'Tap the mic to say another command';
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  } finally {
    micBtn.disabled = false;
  }
}

micBtn.addEventListener('click', () => { isRecording ? stopRecording() : startRecording(); });
"""
    return page_shell("Speech Control", body, script)


@app.route("/speech/command", methods=["POST"])
def speech_command():
    from speech_control import match_command

    if "audio" not in request.files:
        return jsonify({"heard": None, "message": "No audio received."}), 400

    text = transcribe_uploaded_audio(request.files["audio"])
    if not text:
        return jsonify({"heard": None, "message": "Didn't catch that — try again."})

    action = match_command(text)
    if action:
        with open("gesture.txt", "w", encoding="utf-8") as f:
            f.write(action)
        return jsonify({"heard": text, "message": f"✅ Sent to JD: {action}"})
    return jsonify({"heard": text, "message": "⚠️ No matching command recognized."})


# =============================================================================
# ASK JD — chat with JD, type or speak. Reuses voice_jd.py's brain
# (memory, follow-up expansion, deep-dive, safety filter) directly by
# importing it, instead of duplicating that logic here.
# =============================================================================
@app.route("/ask-jd")
def ask_jd_page():
    body = """
<div class="card">
  <div class="chat-log" id="log"></div>
  <div class="input-row">
    <button id="mic-btn">🎤</button>
    <input type="text" id="text-input" placeholder="Type a question..." autocomplete="off" />
    <button class="primary" id="send-btn">Send</button>
  </div>
  <div class="status" id="status"></div>
</div>
"""
    script = """
const logEl = document.getElementById('log');
const micBtn = document.getElementById('mic-btn');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
let mediaRecorder, audioChunks = [], isRecording = false;

function addBubble(text, cls) {
  const b = document.createElement('div');
  b.className = 'bubble ' + cls;
  b.textContent = text;
  logEl.appendChild(b);
  logEl.scrollTop = logEl.scrollHeight;
  return b;
}

async function sendQuestion(question) {
  addBubble(question, 'user');
  const thinking = addBubble('JD is thinking...', 'jd thinking');
  try {
    const res = await fetch('/ask-jd/ask', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ question })
    });
    const data = await res.json();
    thinking.remove();
    addBubble(data.error ? ('Error: ' + data.error) : data.answer, 'jd');
  } catch (err) {
    thinking.remove();
    addBubble('Could not reach JD: ' + err.message, 'jd');
  }
}

sendBtn.addEventListener('click', () => {
  const q = textInput.value.trim();
  if (!q) return;
  textInput.value = '';
  sendQuestion(q);
});
textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendBtn.click(); });

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();
  isRecording = true;
  micBtn.classList.add('recording');
  statusEl.textContent = 'Listening... tap mic again to stop';
}
function stopRecording() {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
  isRecording = false;
  micBtn.classList.remove('recording');
}
async function handleStop() {
  micBtn.disabled = true;
  statusEl.textContent = 'Transcribing...';
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const form = new FormData();
  form.append('audio', blob, 'q.webm');
  try {
    const res = await fetch('/ask-jd/transcribe', { method: 'POST', body: form });
    const data = await res.json();
    micBtn.disabled = false;
    statusEl.textContent = '';
    if (data.text) sendQuestion(data.text);
  } catch (err) {
    micBtn.disabled = false;
    statusEl.textContent = 'Error: ' + err.message;
  }
}
micBtn.addEventListener('click', () => { isRecording ? stopRecording() : startRecording(); });

addBubble("Hi! I'm JD. Ask me anything about robotics, AI, or tech — type or tap the mic.", 'jd');
"""
    return page_shell("Ask JD", body, script)


@app.route("/ask-jd/transcribe", methods=["POST"])
def ask_jd_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400
    text = transcribe_uploaded_audio(request.files["audio"])
    return jsonify({"text": text})


@app.route("/ask-jd/ask", methods=["POST"])
def ask_jd_ask():
    import voice_jd  # reuse its memory/safety/deep-dive logic directly

    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400

    try:
        answer = voice_jd.ask_ollama_with_memory(question)
    except Exception as e:
        # Most common cause: Ollama isn't running, or the model hasn't been
        # pulled. Without this try/except, an unhandled exception here makes
        # Flask return its default HTML error page -- the frontend then
        # tries to JSON-parse that HTML and fails with a confusing
        # "Unexpected token '<'" instead of a real error message.
        return jsonify({
            "error": f"Couldn't reach Ollama ({e}). Is 'ollama serve' running, and have "
                     f"you pulled the {voice_jd.FAST_MODEL} and {voice_jd.DEEP_MODEL} models?"
        }), 502

    reaction = voice_jd.pick_reaction(answer)
    voice_jd.send_to_arc(reaction)
    voice_jd.send_speech_to_arc(answer)

    return jsonify({"answer": answer, "reaction": reaction})


# =============================================================================
# FIND OBJECT — "find my bottle", "who is this", "find Ahmed"
# Each command can take a while (JD physically scans positions and
# speaks out loud), so the page sets clear expectations while waiting.
# =============================================================================
@app.route("/find-object")
def find_object_page():
    body = """
<div class="card">
  <div class="chat-log" id="log"></div>
  <div class="input-row">
    <button id="mic-btn">🎤</button>
    <input type="text" id="text-input" placeholder='e.g. "find my bottle"' autocomplete="off" />
    <button class="primary" id="send-btn">Send</button>
  </div>
  <div class="status" id="status">Commands can take 10-30 seconds — JD physically scans the room.</div>
</div>
"""
    script = """
const logEl = document.getElementById('log');
const micBtn = document.getElementById('mic-btn');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const statusEl = document.getElementById('status');
let mediaRecorder, audioChunks = [], isRecording = false;

function addBubble(text, cls) {
  const b = document.createElement('div');
  b.className = 'bubble ' + cls;
  b.textContent = text;
  logEl.appendChild(b);
  logEl.scrollTop = logEl.scrollHeight;
  return b;
}

async function sendCommand(command) {
  addBubble(command, 'user');
  const thinking = addBubble('JD is scanning... this can take a bit', 'jd thinking');
  sendBtn.disabled = true; micBtn.disabled = true;
  try {
    const res = await fetch('/find-object/command', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ command })
    });
    const data = await res.json();
    thinking.remove();
    addBubble(data.reply, 'jd');
  } catch (err) {
    thinking.remove();
    addBubble('Error: ' + err.message, 'jd');
  } finally {
    sendBtn.disabled = false; micBtn.disabled = false;
  }
}

sendBtn.addEventListener('click', () => {
  const c = textInput.value.trim();
  if (!c) return;
  textInput.value = '';
  sendCommand(c);
});
textInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendBtn.click(); });

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];
  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = handleStop;
  mediaRecorder.start();
  isRecording = true;
  micBtn.classList.add('recording');
  statusEl.textContent = 'Listening... tap mic again to stop';
}
function stopRecording() {
  mediaRecorder.stop();
  mediaRecorder.stream.getTracks().forEach(t => t.stop());
  isRecording = false;
  micBtn.classList.remove('recording');
}
async function handleStop() {
  micBtn.disabled = true;
  statusEl.textContent = 'Transcribing...';
  const blob = new Blob(audioChunks, { type: 'audio/webm' });
  const form = new FormData();
  form.append('audio', blob, 'q.webm');
  try {
    const res = await fetch('/find-object/transcribe', { method: 'POST', body: form });
    const data = await res.json();
    micBtn.disabled = false;
    statusEl.textContent = 'Commands can take 10-30 seconds — JD physically scans the room.';
    if (data.text) sendCommand(data.text);
  } catch (err) {
    micBtn.disabled = false;
    statusEl.textContent = 'Error: ' + err.message;
  }
}
micBtn.addEventListener('click', () => { isRecording ? stopRecording() : startRecording(); });

addBubble("Hi! I'm JD's object-finding assistant. Try \\"find my bottle\\" or \\"who is this\\".", 'jd');
"""
    return page_shell("Find Object", body, script)


@app.route("/find-object/transcribe", methods=["POST"])
def find_object_transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "No audio received"}), 400
    text = transcribe_uploaded_audio(request.files["audio"])
    return jsonify({"text": text})


@app.route("/find-object/command", methods=["POST"])
def find_object_command():
    data = request.get_json(force=True)
    command = (data.get("command") or "").strip()
    if not command:
        return jsonify({"reply": "I didn't catch a command."}), 400

    try:
        import object_finder
        reply = object_finder.run_object_finder_command(command)
    except Exception as e:
        reply = f"Something went wrong: {e}"
    return jsonify({"reply": reply})


# =============================================================================
# TRAIN NEW FACES — step-by-step wizard: enter a name, capture photos one
# at a time (each click requests one photo from JD via the trigger/ready
# protocol), then train the model.
# =============================================================================
@app.route("/train-faces")
def train_faces_page():
    from face_trainer import get_trained_people, PHOTOS_PER_PERSON

    people = get_trained_people()
    people_html = (
        "<ul class='people'>" + "".join(f"<li>{p}</li>" for p in people) + "</ul>"
        if people else "<p style='color:var(--muted); font-size:13px;'>No one trained yet.</p>"
    )

    body = f"""
<div class="card">
  <b>Currently trained:</b>
  {people_html}
</div>
<div class="card" id="name-card">
  <b>Add a new person</b>
  <div class="input-row" style="margin-top:10px;">
    <input type="text" id="name-input" placeholder="Person's name" autocomplete="off" />
    <button class="primary" id="begin-btn">Begin</button>
  </div>
</div>
<div class="card" id="capture-card" style="display:none;">
  <b id="capture-title"></b>
  <div class="status" id="capture-status" style="margin-top:8px;"></div>
  <div style="margin-top:12px; display:flex; gap:10px;">
    <button class="primary" id="capture-btn">📸 Capture Photo</button>
    <button class="danger" id="finish-btn">Finish &amp; Train Model</button>
  </div>
</div>
"""
    script = f"""
const PHOTOS_PER_PERSON = {PHOTOS_PER_PERSON};
let currentName = '';
let photoCount = 0;

const nameInput = document.getElementById('name-input');
const beginBtn = document.getElementById('begin-btn');
const nameCard = document.getElementById('name-card');
const captureCard = document.getElementById('capture-card');
const captureTitle = document.getElementById('capture-title');
const captureStatus = document.getElementById('capture-status');
const captureBtn = document.getElementById('capture-btn');
const finishBtn = document.getElementById('finish-btn');

beginBtn.addEventListener('click', () => {{
  const name = nameInput.value.trim();
  if (!name) return;
  currentName = name;
  photoCount = 0;
  nameCard.style.display = 'none';
  captureCard.style.display = 'block';
  captureTitle.textContent = 'Capturing photos for: ' + currentName;
  captureStatus.textContent = `0/${{PHOTOS_PER_PERSON}} photos captured. Face the camera and click Capture.`;
}});

captureBtn.addEventListener('click', async () => {{
  captureBtn.disabled = true;
  captureStatus.textContent = 'Requesting photo from JD...';
  try {{
    const res = await fetch('/train-faces/capture', {{
      method: 'POST', headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{ name: currentName, photo_number: photoCount + 1 }})
    }});
    const data = await res.json();
    if (data.ok) {{
      photoCount += 1;
      captureStatus.textContent = `✅ ${{data.message}} (${{photoCount}}/${{PHOTOS_PER_PERSON}})`;
      if (photoCount >= PHOTOS_PER_PERSON) {{
        captureBtn.disabled = true;
        captureStatus.textContent += ' — enough photos! Click Finish & Train.';
      }}
    }} else {{
      captureStatus.textContent = '⚠️ ' + data.message;
    }}
  }} catch (err) {{
    captureStatus.textContent = 'Error: ' + err.message;
  }} finally {{
    if (photoCount < PHOTOS_PER_PERSON) captureBtn.disabled = false;
  }}
}});

finishBtn.addEventListener('click', async () => {{
  finishBtn.disabled = true; captureBtn.disabled = true;
  captureStatus.textContent = 'Training model on all captured photos...';
  try {{
    const res = await fetch('/train-faces/finish', {{ method: 'POST' }});
    const data = await res.json();
    captureStatus.textContent = data.message;
    setTimeout(() => location.reload(), 2500);
  }} catch (err) {{
    captureStatus.textContent = 'Error: ' + err.message;
    finishBtn.disabled = false;
  }}
}});
"""
    return page_shell("Train New Faces", body, script)


@app.route("/train-faces/capture", methods=["POST"])
def train_faces_capture():
    from face_trainer import capture_one_photo

    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    photo_number = int(data.get("photo_number", 1))
    if not name:
        return jsonify({"ok": False, "message": "No name given."}), 400

    result = capture_one_photo(name, photo_number)
    return jsonify(result)


@app.route("/train-faces/finish", methods=["POST"])
def train_faces_finish():
    from face_trainer import train_model

    try:
        train_model()
    except Exception as e:
        return jsonify({"message": f"Training failed: {e}"}), 500
    return jsonify({"message": "✅ Model trained! Reloading page..."})


def preload_emotion_model():
    """Runs once, in a background thread, right when the server starts.
    DeepFace downloads its emotion-detection model weights from the
    internet the FIRST time it's ever called, if not already cached on
    disk -- doing that download the moment someone clicks "Start" on
    Emotion mode is exactly what makes it look like a frozen black
    screen. Warming it up here means that download (if needed) happens
    during server startup instead, with a clear message either way."""
    try:
        from deepface import DeepFace
        print("🧠 Preloading emotion detection model (first run may download weights)...")
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_frame, actions=['emotion'], enforce_detection=False)
        print("✅ Emotion model ready.")
    except Exception as e:
        print(
            f"⚠️  Couldn't preload the emotion model ({e}). Emotion mode will try "
            "again on first use, but may be slow or fail if there's no internet "
            "access right now (e.g. if this laptop is only connected to JD's own "
            "WiFi network)."
        )


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    local_ip = get_local_ip()
    print("=" * 55)
    print("  🤖 JD COMMAND CENTER is starting...")
    print("=" * 55)
    print(f"  On THIS laptop, open:    http://localhost:5000")
    print(f"  On a PHONE (same WiFi):  http://{local_ip}:5000")
    print("  Note: browser mic access over the phone URL needs the")
    print("  phone and laptop on the same network; if the mic button")
    print("  doesn't work from a phone, that's almost always why.")
    print("=" * 55)
    threading.Thread(target=preload_emotion_model, daemon=True).start()
    # threaded=True is required here -- the camera streaming routes are
    # long-running generators, so without threading they'd block every
    # other page/request while a stream is active.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)