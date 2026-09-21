"""
face_db.py — Face recognition database (Firestore)
----------------------------------------------------
Single source of truth for "who does JD recognize". Replaces the old
SQLite/LBPH version: instead of one averaged face per person, each
person document stores a LIST of DeepFace embeddings (one per training
photo), so recognition can match against multiple angles/lighting
conditions instead of a single "best" photo.

Firestore layout:
  people/<name>            -> { name, embeddings: [ {vector, added_at}, ... ],
                                 registered_at, updated_at }
  admin_config/admin       -> { password_hash }   (SHA-256 hex, never plaintext)

Embeddings are stored as {"vector": [...]} maps rather than raw nested
arrays -- Firestore arrays-of-arrays are finicky across SDK versions,
arrays-of-maps are not.

Offline behavior: every successful Firestore read refreshes a local
JSON cache (face_embeddings_cache.json). If Firestore is briefly
unreachable, reads fall back to that cache so the gate doesn't go
completely dark mid-demo. Writes are NOT queued/retried offline --
registering a new face still requires a live connection.
"""

import hashlib
import json
import os
import time

import firebase_config
from firebase_config import FirebaseNotConfiguredError

PEOPLE_COLLECTION = "people"
ADMIN_COLLECTION = "admin_config"
ADMIN_DOC_ID = "admin"

CACHE_FILE = "face_embeddings_cache.json"

# Legacy LBPH artifacts this rebuild is retiring -- only read here to
# power the one-time migration notice below.
LEGACY_MODEL_FILE = "face_model.yml"
LEGACY_LABELS_FILE = "labels.json"


# ─────────────────────────────────────────────
# Startup / compatibility entry points
# (web_dashboard.py calls both of these once, at import time)
# ─────────────────────────────────────────────

def init_db():
    """Back-compat name from the old SQLite version. There are no
    tables to create in Firestore, so this just verifies the app CAN
    reach Firebase and prints a clear, non-fatal warning if not --
    the dashboard should still start even with Firebase unconfigured,
    so the rest of the site is browsable; only the gate itself will
    fail (with a clear message) until credentials are added."""
    if not firebase_config.is_configured():
        print(
            "⚠️  Firebase isn't configured yet (FIREBASE_CREDENTIALS_PATH missing "
            "or invalid in .env). The face gate will not work until this is fixed. "
            "See .env.example."
        )
        return
    try:
        firebase_config.get_db()
        print("✅ Firestore connected.")
    except FirebaseNotConfiguredError as e:
        print(f"⚠️  {e}")


def migrate_legacy_files_if_needed():
    """One-time notice: if old LBPH artifacts are sitting on disk and
    Firestore has no registered people yet, explain that they can't be
    auto-converted (LBPH never stored embeddings, just a trained
    classifier) and that people need to be re-registered through the
    new face-embedding flow."""
    has_legacy = os.path.isfile(LEGACY_MODEL_FILE) or os.path.isfile(LEGACY_LABELS_FILE)
    if not has_legacy:
        return

    try:
        people = get_known_people()
    except FirebaseNotConfiguredError:
        return  # can't check Firestore yet -- nothing useful to say

    if not people:
        print(
            "ℹ️  Found legacy face_model.yml/labels.json from the old LBPH system, "
            "but no one is registered in Firestore yet. These files CANNOT be "
            "auto-converted into face embeddings -- LBPH never stored anything "
            "usable for that. Please re-register each person through the new "
            "'Train Faces' flow. The old files are left on disk for now; safe to "
            "delete once everyone's re-registered."
        )


# ─────────────────────────────────────────────
# Local offline cache
# ─────────────────────────────────────────────

def _load_cache() -> dict:
    if not os.path.isfile(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(people_dict: dict):
    """people_dict: {name: [embedding_vector, ...]}"""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(people_dict, f)
    except OSError as e:
        print(f"⚠️  Couldn't write local face cache: {e}")


# ─────────────────────────────────────────────
# People / embeddings
# ─────────────────────────────────────────────

def get_known_people() -> list:
    """Returns [{"name": ...}, ...] for every registered person.
    Falls back to the local cache if Firestore is unreachable."""
    try:
        db = firebase_config.get_db()
        docs = db.collection(PEOPLE_COLLECTION).stream()
        names = [doc.id for doc in docs]
        return [{"name": n} for n in names]
    except FirebaseNotConfiguredError:
        raise
    except Exception as e:
        print(f"⚠️  Firestore unreachable ({e}), using local cache.")
        cache = _load_cache()
        return [{"name": n} for n in cache.keys()]


def get_all_embeddings() -> dict:
    """Returns {name: [embedding_vector, ...]} for every registered
    person -- this is what face_recognizer.py compares a live photo
    against. Refreshes the local cache on success; falls back to the
    cache on failure."""
    try:
        db = firebase_config.get_db()
        docs = db.collection(PEOPLE_COLLECTION).stream()
        result = {}
        for doc in docs:
            data = doc.to_dict() or {}
            vectors = [e["vector"] for e in data.get("embeddings", []) if "vector" in e]
            if vectors:
                result[doc.id] = vectors
        _save_cache(result)
        return result
    except FirebaseNotConfiguredError:
        raise
    except Exception as e:
        print(f"⚠️  Firestore unreachable ({e}), using local cache.")
        return _load_cache()


def add_embeddings_for_person(name: str, embeddings: list):
    """Appends new embedding vectors to a person's document, creating
    it if needed. APPENDS rather than overwrites -- registering the
    same person again (e.g. adding photos in better lighting later)
    adds to their existing set instead of destroying it, and never
    touches any other person's data."""
    if not name:
        raise ValueError("name is required")
    if not embeddings:
        return

    db = firebase_config.get_db()
    doc_ref = db.collection(PEOPLE_COLLECTION).document(name)
    snapshot = doc_ref.get()
    now = time.time()

    new_entries = [{"vector": list(vec), "added_at": now} for vec in embeddings]

    if snapshot.exists:
        existing = (snapshot.to_dict() or {}).get("embeddings", [])
        doc_ref.update({
            "embeddings": existing + new_entries,
            "updated_at": now,
        })
    else:
        doc_ref.set({
            "name": name,
            "embeddings": new_entries,
            "registered_at": now,
            "updated_at": now,
        })


def remove_person(name: str):
    db = firebase_config.get_db()
    db.collection(PEOPLE_COLLECTION).document(name).delete()


def reset_all_faces(password: str) -> dict:
    """Wipes every registered person. Requires the admin password --
    this is the one destructive, all-users-affecting action in the
    system, so it's gated even though the file-bridge/webcam parts of
    this app have no other concept of authorization."""
    if not verify_admin_password(password):
        return {"ok": False, "message": "❌ Incorrect admin password."}

    try:
        db = firebase_config.get_db()
        docs = db.collection(PEOPLE_COLLECTION).stream()
        count = 0
        for doc in docs:
            doc.reference.delete()
            count += 1
        _save_cache({})
        return {"ok": True, "message": f"✅ Wiped {count} registered face(s)."}
    except FirebaseNotConfiguredError as e:
        return {"ok": False, "message": str(e)}
    except Exception as e:
        return {"ok": False, "message": f"❌ Reset failed: {e}"}


# ─────────────────────────────────────────────
# Admin password (SHA-256 hash, never plaintext)
# ─────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _get_stored_hash():
    """Returns the Firestore-stored hash, or None if no password has
    ever been set (fresh install)."""
    db = firebase_config.get_db()
    snapshot = db.collection(ADMIN_COLLECTION).document(ADMIN_DOC_ID).get()
    if snapshot.exists:
        return (snapshot.to_dict() or {}).get("password_hash")
    return None


def verify_admin_password(password: str) -> bool:
    """Checks against the Firestore-stored hash if one has ever been
    set. If none has been set yet (fresh install, nobody has called
    set_admin_password), falls back to ADMIN_PASSWORD_FALLBACK from
    .env, so a fresh install has a working admin login on day one."""
    if not password:
        return False
    try:
        stored_hash = _get_stored_hash()
    except FirebaseNotConfiguredError:
        stored_hash = None
    except Exception as e:
        print(f"⚠️  Couldn't reach Firestore to verify admin password ({e}).")
        stored_hash = None

    if stored_hash:
        return _hash_password(password) == stored_hash

    fallback = os.environ.get("ADMIN_PASSWORD_FALLBACK")
    return bool(fallback) and password == fallback


def set_admin_password(new_password: str, current_password: str = None) -> dict:
    """Sets/changes the admin password. If a password already exists
    in Firestore, current_password must correctly verify first -- this
    is the "password-gated reset" requirement: you can't silently
    overwrite the admin password without proving you already know it
    (or the .env fallback), even though there's no per-user login
    system to hang normal auth off of."""
    try:
        existing_hash = _get_stored_hash()
    except FirebaseNotConfiguredError as e:
        return {"ok": False, "message": str(e)}

    if existing_hash and not verify_admin_password(current_password or ""):
        return {"ok": False, "message": "❌ Current admin password is incorrect."}

    db = firebase_config.get_db()
    db.collection(ADMIN_COLLECTION).document(ADMIN_DOC_ID).set({
        "password_hash": _hash_password(new_password),
        "updated_at": time.time(),
    })
    return {"ok": True, "message": "✅ Admin password updated."}
