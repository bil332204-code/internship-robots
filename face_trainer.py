"""
face_trainer.py
------------------
Captures training photos for a person (from either JD's own camera via
the ARC trigger/ready file bridge, or the laptop webcam directly), and
turns them into DeepFace embeddings pushed to Firestore via face_db.

This replaces the old LBPH version, and fixes the confirmed missing-
function bug from the previous session: web_dashboard.py's gate routes
call `capture_snapshot(camera_source)` with a "webcam" or "jd" source
argument, but no such function existed anywhere in this file before --
only `request_photo_and_wait()`, which only ever talked to JD's camera
and took no arguments. That's the function this file was missing.

Photos are still saved under faces/<PersonName>/ for reference (and so
a person can be re-processed without re-capturing), but there's no
more local .yml model file -- the embeddings themselves live in
Firestore, generated from these photos.
"""

import os
import time

import cv2

import face_db

TRIGGER_FILE = "trigger.txt"
READY_FILE = "ready.txt"
SNAPSHOT_FILE = "snapshot.jpg"

FACES_DIR = "faces"
PHOTOS_PER_PERSON = 12
READY_TIMEOUT_SECONDS = 15

HAARCASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_cascade.empty():
    raise FileNotFoundError(
        f"Could not load {HAARCASCADE_PATH} -- make sure haarcascade_frontalface_default.xml "
        "is in your robotics folder."
    )


# ─────────────────────────────────────────────
# Capturing a photo -- JD's camera OR the laptop webcam
# ─────────────────────────────────────────────

def request_photo_and_wait() -> bool:
    """JD-camera path: writes the trigger file, waits for ARC's
    EZ-Script loop to save a fresh snapshot.jpg and flip ready.txt
    back to "1". Unchanged from the original implementation."""
    with open(TRIGGER_FILE, "w") as f:
        f.write("1")
    waited = 0
    while waited < READY_TIMEOUT_SECONDS:
        if os.path.exists(READY_FILE):
            with open(READY_FILE, "r") as f:
                if f.read().strip() == "1":
                    with open(READY_FILE, "w") as f:
                        f.write("")
                    return True
        time.sleep(0.5)
        waited += 0.5
    return False


def _capture_from_webcam() -> bool:
    """Laptop-webcam path: grabs a single frame directly with OpenCV
    and writes it to SNAPSHOT_FILE, so the rest of the pipeline
    (face-check, save, embed) doesn't need to know which camera it
    came from. Caller is responsible for holding the shared camera
    lock (try_claim_camera in web_dashboard.py) before calling this --
    this function only owns the device for the single frame grab."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return False
    try:
        # Give the webcam a couple of frames to warm up exposure/focus
        # before the one we actually keep -- the very first frame from
        # a freshly opened camera is often dark/blurry.
        frame = None
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                return False
        if frame is None:
            return False
        cv2.imwrite(SNAPSHOT_FILE, frame)
        return True
    finally:
        cap.release()


def capture_snapshot(camera_source: str) -> bool:
    """Unified entry point the gate routes call: capture_snapshot("webcam")
    or capture_snapshot("jd"). Always leaves the result (or lack of one)
    in SNAPSHOT_FILE, matching the JD-camera protocol's existing contract."""
    camera_source = (camera_source or "jd").strip().lower()
    if camera_source == "webcam":
        return _capture_from_webcam()
    elif camera_source == "jd":
        return request_photo_and_wait()
    else:
        raise ValueError(f"Unknown camera_source: {camera_source!r} (expected 'webcam' or 'jd')")


# ─────────────────────────────────────────────
# Web-friendly single-photo capture (used by /gate/register/capture)
# ─────────────────────────────────────────────

def capture_one_photo(name: str, photo_number: int, camera_source: str = "jd") -> dict:
    """Requests ONE photo from the given camera source, checks a face
    is actually visible (cheap Haar-cascade pass, same first-pass
    filter used elsewhere in the project), and saves it under
    faces/<name>/. Returns a small status dict for the web page --
    the actual DeepFace embedding happens later, in register_person(),
    once all photos are captured (keeps each capture call fast)."""
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    if not capture_snapshot(camera_source):
        source_label = "JD's camera (is ARC_SCRIPT.py running?)" if camera_source != "webcam" else "the laptop webcam"
        return {"ok": False, "message": f"Couldn't get a photo from {source_label}. Try again."}

    image = cv2.imread(SNAPSHOT_FILE)
    if image is None:
        return {"ok": False, "message": "Couldn't read the captured photo. Try again."}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        return {"ok": False, "message": "No face detected in that photo -- face the camera directly and try again."}

    # Save the FULL color photo (not a cropped grayscale square like
    # the old LBPH version) -- DeepFace does its own detection/alignment
    # when generating the embedding, so it needs the original image.
    save_path = os.path.join(person_dir, f"{photo_number}.jpg")
    cv2.imwrite(save_path, image)

    return {"ok": True, "message": f"Saved photo {photo_number}/{PHOTOS_PER_PERSON}."}


def get_trained_people() -> list:
    """Returns names already registered, for display on the web page."""
    return [p["name"] for p in face_db.get_known_people()]


# ─────────────────────────────────────────────
# Turning saved photos into embeddings + pushing to Firestore
# ─────────────────────────────────────────────

def _generate_embeddings_for_person(name: str) -> list:
    """Runs DeepFace on every saved photo for this person and returns
    a list of embedding vectors (one per photo that had a usable face).
    Imported lazily so face_trainer.py doesn't require deepface just to
    capture/save photos (matches the "capture is fast, embedding is a
    separate step" design above)."""
    from face_recognizer import MODEL_NAME, DETECTOR_BACKEND
    from deepface import DeepFace

    person_dir = os.path.join(FACES_DIR, name)
    if not os.path.isdir(person_dir):
        return []

    embeddings = []
    for filename in sorted(os.listdir(person_dir)):
        path = os.path.join(person_dir, filename)
        try:
            faces = DeepFace.represent(
                img_path=path,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False,
                align=True,
            )
        except Exception as e:
            print(f"⚠️  Skipping {path}, DeepFace couldn't process it: {e}")
            continue

        if not faces:
            continue
        # One person, one face expected per training photo -- take the
        # highest-confidence detection if DeepFace found more than one.
        best = max(faces, key=lambda f: f.get("face_confidence", 0))
        embeddings.append(best["embedding"])

    return embeddings


def register_person(name: str) -> dict:
    """Web-facing finalize step (called from /gate/register/finish):
    turns this person's captured photos into embeddings and pushes
    them to Firestore. Safe to call again later for the same person
    (e.g. adding more photos in different lighting) -- embeddings are
    appended, never overwritten, by face_db.add_embeddings_for_person."""
    if not name:
        return {"ok": False, "message": "No name given."}

    embeddings = _generate_embeddings_for_person(name)
    if not embeddings:
        return {"ok": False, "message": "Couldn't generate any usable face embeddings from those photos. Try capturing again with better lighting."}

    try:
        face_db.add_embeddings_for_person(name, embeddings)
    except Exception as e:
        return {"ok": False, "message": f"Saving to Firestore failed: {e}"}

    # A running server keeps its own in-memory copy of known embeddings
    # for speed -- invalidate it so this new person is recognized
    # immediately, not just after a restart.
    import face_recognizer
    face_recognizer._reload_model()

    return {"ok": True, "message": f"✅ Registered! JD now recognizes you as {name} ({len(embeddings)} photo(s) processed)."}


def train_model():
    """Back-compat / standalone-CLI entry point: processes EVERY
    person's folder under faces/ and (re-)uploads their embeddings.
    Used by the terminal flow below (run_face_trainer) and any script
    that still calls this name directly. The web gate calls
    register_person(name) directly instead, since it already knows
    exactly which person just finished capturing."""
    if not os.path.isdir(FACES_DIR):
        print("No training photos found. Capture some faces first.")
        return

    people = [d for d in sorted(os.listdir(FACES_DIR)) if os.path.isdir(os.path.join(FACES_DIR, d))]
    if not people:
        print("No training photos found. Capture some faces first.")
        return

    for name in people:
        result = register_person(name)
        print(("✅ " if result["ok"] else "❌ ") + result["message"])


# ─────────────────────────────────────────────
# Terminal / standalone flow (main_controller.py fallback usage)
# ─────────────────────────────────────────────

def capture_faces_for_person(name: str):
    person_dir = os.path.join(FACES_DIR, name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"\nCapturing {PHOTOS_PER_PERSON} photos for '{name}'.")
    print("Make sure they're facing the camera, and move slightly between shots")
    print("(different angle/expression each time helps accuracy).")

    saved = 0
    attempt = 0
    while saved < PHOTOS_PER_PERSON and attempt < PHOTOS_PER_PERSON * 3:
        attempt += 1
        input(f"  Press Enter to capture photo {saved + 1}/{PHOTOS_PER_PERSON}...")

        result = capture_one_photo(name, saved + 1, camera_source="jd")
        if not result["ok"]:
            print(f"  {result['message']}")
            continue

        saved += 1
        print(f"  Saved ({saved}/{PHOTOS_PER_PERSON}).")

    print(f"Done capturing for '{name}'.")


def run_face_trainer():
    os.makedirs(FACES_DIR, exist_ok=True)
    print("=== JD Face Trainer ===")
    print("Add group members one at a time. Type 'done' when finished.\n")

    while True:
        name = input("Enter person's name (or 'done' to finish): ").strip()
        if name.lower() == "done":
            break
        if not name:
            continue
        capture_faces_for_person(name)

    train_model()


if __name__ == "__main__":
    run_face_trainer()
