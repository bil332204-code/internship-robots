"""
firebase_config.py
-------------------
Centralized Firebase Admin SDK initialization, used by face_db.py (and
anything else that eventually needs Firestore). Everything else in the
project should call get_db() rather than touching firebase_admin
directly -- that keeps init-once behavior and error messages consistent
in one place instead of scattered across files.

Setup (one-time, per machine):
  1. In the Firebase Console, create a project and enable Firestore.
  2. Project Settings -> Service Accounts -> "Generate new private key".
     This downloads a JSON file -- keep it OUTSIDE the git repo.
  3. In .env, set FIREBASE_CREDENTIALS_PATH to the full path of that file.

Deliberately NOT using Firebase's client SDK / API keys here -- the
Admin SDK + service account is the correct choice for a trusted backend
process like this Flask app, and it bypasses Firestore Security Rules
by design (the rules exist to protect against other, untrusted clients,
e.g. if a mobile app were ever added later -- see firestore.rules).
"""

import os
import threading

from dotenv import load_dotenv

load_dotenv()  # reads FIREBASE_CREDENTIALS_PATH (and everything else) from .env

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseNotConfiguredError(Exception):
    """Raised when Firestore is needed but hasn't been set up yet on this
    machine. Callers should catch this and show a clear, actionable
    message instead of letting a raw stack trace surface -- this is the
    #1 thing a teammate on a fresh laptop will hit."""
    pass


_db = None
_init_lock = threading.Lock()


def _credentials_path() -> str:
    return (os.environ.get("FIREBASE_CREDENTIALS_PATH") or "").strip()


def is_configured() -> bool:
    """True if a credentials path is set AND the file actually exists.
    Safe to call anytime -- never raises."""
    path = _credentials_path()
    return bool(path) and os.path.isfile(path)


def get_db():
    """Returns a Firestore client, initializing the Admin SDK exactly
    once per process. Raises FirebaseNotConfiguredError (never a raw
    firebase_admin/google-cloud stack trace) if credentials aren't set
    up yet."""
    global _db
    if _db is not None:
        return _db

    with _init_lock:
        if _db is not None:  # re-check after acquiring the lock
            return _db

        path = _credentials_path()
        if not path:
            raise FirebaseNotConfiguredError(
                "FIREBASE_CREDENTIALS_PATH is not set. Add it to your .env, "
                "pointing at the service account JSON key file you downloaded "
                "from the Firebase Console (Project Settings -> Service Accounts)."
            )
        if not os.path.isfile(path):
            raise FirebaseNotConfiguredError(
                f"FIREBASE_CREDENTIALS_PATH is set to '{path}', but no file "
                "exists there. Double-check the path in your .env."
            )

        if not firebase_admin._apps:
            cred = credentials.Certificate(path)
            firebase_admin.initialize_app(cred)

        _db = firestore.client()
        return _db


def _reset_for_tests():
    """Test-only hook: forces the next get_db() call to re-initialize.
    Not used by the running app."""
    global _db
    _db = None
