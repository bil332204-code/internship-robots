"""
face_recognizer.py
---------------------
Recognizes people in a photo using DeepFace face embeddings + cosine
similarity, comparing against every embedding stored in Firestore
(via face_db.get_all_embeddings()). Replaces the old LBPH
predict()-based version.

IMPORTANT SCALE DIFFERENCE FROM THE OLD SYSTEM:
LBPH's `confidence` was a DISTANCE score -- LOWER meant a better match,
typically compared against a threshold like 70. DeepFace cosine
SIMILARITY is the opposite: HIGHER means a better match, on a
roughly 0-1 scale. These are NOT interchangeable numbers -- do not
reuse the old CONFIDENCE_THRESHOLD value here.

SIMILARITY_THRESHOLD below is a reasonable starting point for
Facenet + cosine similarity (DeepFace's own `verify()` uses an
equivalent ~0.60 similarity cutoff for this model), but it has NOT
been calibrated against this project's actual camera/lighting yet --
see PROJECT_HANDOFF.md Section 4, item 1. Tune it up (stricter) if
strangers get matched, or down (looser) if registered people keep
reading as Unknown.
"""

import os

import cv2
import numpy as np
from deepface import DeepFace

import face_db

# Facenet was chosen because it's a good accuracy/speed balance for a
# live demo on a laptop CPU (no GPU assumed) and DeepFace ships it
# without extra downloads beyond the model weights it fetches once.
MODEL_NAME = "Facenet"

# "opencv" detector backend deliberately matches the rest of the
# project -- it's why opencv-contrib-python is pinned to 4.10.0.84 in
# requirements.txt (newer versions are missing the bundled Haar
# cascade data DeepFace's opencv backend needs).
DETECTOR_BACKEND = "opencv"

SIMILARITY_THRESHOLD = 0.60  # cosine similarity, higher = better match. NEEDS CALIBRATION.

_embeddings_cache = None  # {name: [np.array, ...]}


def _get_known_embeddings():
    """Lazily loaded, process-wide cache of every known person's
    embeddings, so we're not hitting Firestore on every single frame.
    Call _reload_model() after registering/resetting to force a
    refresh."""
    global _embeddings_cache
    if _embeddings_cache is None:
        raw = face_db.get_all_embeddings()  # {name: [ [floats], ... ]}
        if not raw:
            raise FileNotFoundError(
                "No trained face embeddings yet. Register at least one person first."
            )
        _embeddings_cache = {
            name: [np.array(vec, dtype=np.float32) for vec in vectors]
            for name, vectors in raw.items()
        }
    return _embeddings_cache


def _reload_model():
    """Forces the next recognition call to re-read embeddings from
    Firestore -- call this after training/registering someone new or
    resetting all faces. Name kept from the old LBPH version so
    web_dashboard.py doesn't need to change its call sites."""
    global _embeddings_cache
    _embeddings_cache = None


def get_known_names() -> list:
    """Returns the list of names the system has been trained on."""
    return [p["name"] for p in face_db.get_known_people()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _best_match(embedding: np.ndarray, known: dict):
    """Returns (name, similarity) for the closest known person, or
    ("Unknown", best_similarity) if nothing clears the threshold.
    Compares against EVERY stored photo/embedding for each person
    (not just one averaged vector) and keeps that person's single
    best score."""
    best_name = "Unknown"
    best_score = -1.0
    for name, vectors in known.items():
        for vec in vectors:
            score = _cosine_similarity(embedding, vec)
            if score > best_score:
                best_score = score
                best_name = name if score >= SIMILARITY_THRESHOLD else "Unknown"
    return best_name, best_score


def recognize_people(image_path: str) -> list:
    """Detects and recognizes faces in the given photo. Returns a list
    of names, e.g. ["Ahmed", "Ali"], or [] if no faces were found, or
    "Unknown" entries for unrecognized faces."""
    results, _, _ = recognize_people_with_positions(image_path)
    return [r["name"] for r in results]


def recognize_people_with_positions(image_path: str):
    """Like recognize_people, but also returns each face's center
    position (needed for PID fine-centering, same contract as
    detect_with_positions in yolo_detector.py). Returns
    (results_list, image_width, image_height).

    results_list is a list of dicts:
        {"name": "Ahmed", "similarity": 0.74, "center_x": 142.5, "center_y": 88.0}
    """
    known = _get_known_embeddings()  # raises FileNotFoundError if nobody registered

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Couldn't read image: {image_path}")
    image_height, image_width = image.shape[:2]

    try:
        faces = DeepFace.represent(
            img_path=image_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True,
        )
    except Exception as e:
        print(f"⚠️  DeepFace.represent failed: {e}")
        return [], image_width, image_height

    # enforce_detection=False means DeepFace still returns one result
    # even with no real face (a low-confidence guess on the whole
    # frame) -- filter those out using face_confidence when available.
    results = []
    for face in faces:
        confidence = face.get("face_confidence", 1.0)
        if confidence is not None and confidence <= 0:
            continue

        embedding = np.array(face["embedding"], dtype=np.float32)
        name, similarity = _best_match(embedding, known)

        area = face.get("facial_area", {})
        x = area.get("x", 0)
        y = area.get("y", 0)
        w = area.get("w", 0)
        h = area.get("h", 0)

        results.append({
            "name": name,
            "similarity": round(similarity, 3),
            "center_x": x + w / 2,
            "center_y": y + h / 2,
        })

    return results, image_width, image_height


if __name__ == "__main__":
    # Quick manual test: python face_recognizer.py
    result = recognize_people("snapshot.jpg")
    print("Recognized:", result)
