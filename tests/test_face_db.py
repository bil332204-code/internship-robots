"""
tests/test_face_db.py
----------------------
Verifies face_db.py's logic against a hand-built mock Firestore client,
since real Firebase credentials aren't available in this sandbox.
Mirrors the test coverage the handoff document describes: add/append,
no-cross-contamination between people, password hashing, offline
cache fallback, and password-gated admin reset.

Run with:  python -m pytest tests/test_face_db.py -v
       or: python tests/test_face_db.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import face_db
import firebase_config


# ─────────────────────────────────────────────
# A minimal in-memory stand-in for a Firestore client, covering just
# the surface face_db.py actually calls: collection().document(id),
# .get()/.set()/.update()/.delete(), and collection().stream().
# ─────────────────────────────────────────────

class _MockSnapshot:
    def __init__(self, doc_id, data, reference=None):
        self.id = doc_id
        self._data = data
        self.exists = data is not None
        self.reference = reference  # real Firestore DocumentSnapshot has this too

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _MockDocRef:
    def __init__(self, store, collection_name, doc_id):
        self._store = store
        self._collection_name = collection_name
        self.id = doc_id

    def get(self):
        data = self._store.get(self._collection_name, {}).get(self.id)
        return _MockSnapshot(self.id, data, reference=self)

    def set(self, data):
        self._store.setdefault(self._collection_name, {})[self.id] = dict(data)

    def update(self, data):
        existing = self._store.setdefault(self._collection_name, {}).setdefault(self.id, {})
        existing.update(data)

    def delete(self):
        self._store.get(self._collection_name, {}).pop(self.id, None)

    @property
    def reference(self):
        return self


class _MockCollection:
    def __init__(self, store, name):
        self._store = store
        self._name = name

    def document(self, doc_id):
        return _MockDocRef(self._store, self._name, doc_id)

    def stream(self):
        docs = self._store.get(self._name, {})
        return [
            _MockSnapshot(doc_id, data, reference=_MockDocRef(self._store, self._name, doc_id))
            for doc_id, data in list(docs.items())
        ]


class MockFirestoreClient:
    def __init__(self):
        self._store = {}

    def collection(self, name):
        return _MockCollection(self._store, name)


class FaceDbTests(unittest.TestCase):
    def setUp(self):
        self.mock_db = MockFirestoreClient()
        firebase_config._db = self.mock_db  # inject the mock, skip real init
        self._orig_is_configured = firebase_config.is_configured
        firebase_config.is_configured = lambda: True
        # Isolate the offline cache file per test run
        self.cache_path = "test_face_embeddings_cache.json"
        face_db.CACHE_FILE = self.cache_path
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def tearDown(self):
        firebase_config._db = None
        firebase_config.is_configured = self._orig_is_configured
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def test_add_and_append_embeddings(self):
        face_db.add_embeddings_for_person("Ahmed", [[0.1, 0.2, 0.3]])
        face_db.add_embeddings_for_person("Ahmed", [[0.4, 0.5, 0.6]])

        all_emb = face_db.get_all_embeddings()
        self.assertIn("Ahmed", all_emb)
        self.assertEqual(len(all_emb["Ahmed"]), 2, "second call should APPEND, not overwrite")

    def test_no_cross_contamination_between_people(self):
        face_db.add_embeddings_for_person("Ahmed", [[0.1, 0.2]])
        face_db.add_embeddings_for_person("Asjid", [[0.9, 0.8]])

        all_emb = face_db.get_all_embeddings()
        self.assertEqual(len(all_emb["Ahmed"]), 1)
        self.assertEqual(len(all_emb["Asjid"]), 1)
        self.assertNotEqual(all_emb["Ahmed"], all_emb["Asjid"])

    def test_password_hashing_never_plaintext(self):
        face_db.set_admin_password("supersecret")
        stored = self.mock_db._store["admin_config"]["admin"]["password_hash"]
        self.assertNotEqual(stored, "supersecret")
        self.assertEqual(len(stored), 64)  # sha256 hex digest length

        self.assertTrue(face_db.verify_admin_password("supersecret"))
        self.assertFalse(face_db.verify_admin_password("wrong_password"))

    def test_admin_password_fallback_when_unset(self):
        os.environ["ADMIN_PASSWORD_FALLBACK"] = "fallback123"
        try:
            self.assertTrue(face_db.verify_admin_password("fallback123"))
            self.assertFalse(face_db.verify_admin_password("nope"))
        finally:
            del os.environ["ADMIN_PASSWORD_FALLBACK"]

    def test_password_gated_change(self):
        face_db.set_admin_password("first_password")
        # Wrong current password should be rejected
        result = face_db.set_admin_password("new_password", current_password="wrong")
        self.assertFalse(result["ok"])
        # Correct current password should succeed
        result = face_db.set_admin_password("new_password", current_password="first_password")
        self.assertTrue(result["ok"])
        self.assertTrue(face_db.verify_admin_password("new_password"))

    def test_offline_cache_fallback(self):
        face_db.add_embeddings_for_person("Ahmed", [[0.1, 0.2, 0.3]])
        face_db.get_all_embeddings()  # populates the local cache file
        self.assertTrue(os.path.exists(self.cache_path))

        # Simulate Firestore being unreachable
        def _broken_collection(name):
            raise ConnectionError("simulated network failure")
        self.mock_db.collection = _broken_collection

        cached = face_db.get_all_embeddings()
        self.assertIn("Ahmed", cached, "should fall back to local cache when Firestore is unreachable")

    def test_reset_requires_correct_password(self):
        face_db.set_admin_password("adminpass")
        face_db.add_embeddings_for_person("Ahmed", [[0.1, 0.2]])

        result = face_db.reset_all_faces("wrong")
        self.assertFalse(result["ok"])
        self.assertIn("Ahmed", face_db.get_all_embeddings())

        result = face_db.reset_all_faces("adminpass")
        self.assertTrue(result["ok"])
        self.assertEqual(face_db.get_all_embeddings(), {})


if __name__ == "__main__":
    unittest.main()
