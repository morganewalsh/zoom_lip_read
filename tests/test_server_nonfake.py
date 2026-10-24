"""Non-fake `/transcribe_chunk` envelope and temp-cleanup tests.

Covers (via an INJECTED transcribe callable so no real model/GPU/media is
needed, R10.4 / R10.5):
- Non-fake injected test (R5.6, R5.10): `create_app(transcribe=fake)` with
  FAKE_MODE off; posting a small dummy file echoes the request's `chunk_index`,
  returns the injected `segments` list verbatim, and the upload is written with
  its extension preserved (the fake records the path it was handed).
- Edge case (R5.7): the temporary directory and uploaded file are gone after the
  request completes, both when the injected transcribe RETURNS and when it
  RAISES.
- Form-field defaults applied when omitted (R5.4, R5.10): `chunk_start` defaults
  to 0.0 and `mode` to "auto".
- No heavy ML module (torch/cv2/mediapipe/faster_whisper) is imported.

These tests need FastAPI's TestClient. When it is unavailable they are skipped
cleanly so the suite still runs on a bare machine (numpy only).

Run: python -m tests.test_server_nonfake   (from the project root)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read import server  # noqa: E402  (light import, no heavy deps)

# TestClient availability gate: skip the whole module when fastapi's TestClient
# (which also needs httpx/starlette) cannot be imported.
try:
    from fastapi.testclient import TestClient

    HAS_TESTCLIENT = True
except Exception:  # pragma: no cover - bare machine without fastapi
    HAS_TESTCLIENT = False

PASSED = 0
FAILED = 0

HEAVY_MODULES = ("torch", "cv2", "mediapipe", "faster_whisper")


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  - {name}")
    else:
        FAILED += 1
        print(f"  FAIL- {name}")


class _EnvGuard:
    """Set/clear FAKE_MODE for the duration of a block, then restore."""

    def __init__(self, value):
        self._value = value
        self._saved = None
        self._had = False

    def __enter__(self):
        self._had = "FAKE_MODE" in os.environ
        self._saved = os.environ.get("FAKE_MODE")
        if self._value is None:
            os.environ.pop("FAKE_MODE", None)
        else:
            os.environ["FAKE_MODE"] = self._value
        return self

    def __exit__(self, *exc):
        if self._had:
            os.environ["FAKE_MODE"] = self._saved
        else:
            os.environ.pop("FAKE_MODE", None)
        return False


# --- Non-fake injected test: envelope + suffix + cleanup (R5.6, R5.7, R5.10) --


def test_nonfake_injected_envelope_and_cleanup():
    """FAKE_MODE off + injected transcribe: echoes chunk_index, returns the
    injected segments verbatim, preserves the upload extension, and leaves no
    temp dir/file behind (R5.6, R5.7, R5.10, R10.4)."""
    # The injected transcribe stands in for the real backend. It records the
    # path it was handed, that the uploaded file EXISTS at call time, and the
    # Segment_Records to return. Asserting existence at call time + absence
    # afterwards proves the temp dir lives exactly for the request (R5.7).
    recorded = {}

    returned_segments = [
        {"start": 0.0, "end": 1.0, "text": "hi there", "source": "audio"},
        {"start": 1.0, "end": 2.0, "text": "", "source": "lip", "unclear": True},
    ]

    def fake_transcribe(path, chunk_start=0.0, mode="auto"):
        recorded["path"] = path
        recorded["exists_at_call"] = os.path.exists(path)
        recorded["chunk_start"] = chunk_start
        recorded["mode"] = mode
        # Echo back the handed-over values so the envelope can be verified too.
        return returned_segments

    with _EnvGuard("0"):
        client = TestClient(server.create_app(transcribe=fake_transcribe))
        r = client.post(
            "/transcribe_chunk",
            files={"file": ("clip.mp4", b"dummy-bytes", "application/octet-stream")},
            data={"session_id": "s1", "chunk_index": "7", "chunk_start": "4.5", "mode": "lip_only"},
        )

    body = r.json()
    check("echoes chunk_index", body.get("chunk_index") == 7)
    check("returns injected segments verbatim", body.get("segments") == returned_segments)
    check("transcribe received the request's chunk_start", recorded.get("chunk_start") == 4.5)
    check("transcribe received the request's mode", recorded.get("mode") == "lip_only")

    # Suffix preserved: ".mp4" filename -> upload.mp4 (R5.6).
    path = recorded.get("path", "")
    check("upload path preserves extension", os.path.basename(path) == "upload.mp4")
    check("uploaded file existed during the call", recorded.get("exists_at_call") is True)
    # Temp dir/upload removed after the request (R5.7).
    check("uploaded file removed after request", not os.path.exists(path))
    check("temp dir removed after request",
          not os.path.exists(os.path.dirname(path)) if path else False)

    # No heavy ML module was imported by the injected non-fake path (R10.5).
    loaded = [m for m in HEAVY_MODULES if m in sys.modules]
    check("no heavy model module loaded in non-fake injected path", loaded == [])


# --- Edge case: cleanup also happens when transcribe RAISES (R5.7) ---------


def test_nonfake_cleanup_when_transcribe_raises():
    """The temp dir/upload is removed even when the injected transcribe raises,
    because the TemporaryDirectory context manager cleans up on exception (R5.7)."""
    recorded = {}

    def boom_transcribe(path, chunk_start=0.0, mode="auto"):
        recorded["path"] = path
        recorded["exists_at_call"] = os.path.exists(path)
        raise RuntimeError("boom")

    with _EnvGuard("0"):
        # raise_server_exceptions=False so the 500 is returned rather than
        # re-raised into the test; we only care that cleanup happened.
        client = TestClient(
            server.create_app(transcribe=boom_transcribe),
            raise_server_exceptions=False,
        )
        r = client.post(
            "/transcribe_chunk",
            files={"file": ("clip.webm", b"dummy-bytes", "application/octet-stream")},
            data={"session_id": "s1", "chunk_index": "1"},
        )

    check("request errored (status 500) when transcribe raised", r.status_code == 500)
    path = recorded.get("path", "")
    check("uploaded file existed during the failing call", recorded.get("exists_at_call") is True)
    check("uploaded file removed after a failing request", not os.path.exists(path))
    check("temp dir removed after a failing request",
          not os.path.exists(os.path.dirname(path)) if path else False)


# --- Form-field defaults applied when omitted (R5.4, R5.10) ----------------


def test_nonfake_form_field_defaults():
    """When `chunk_start` and `mode` are omitted, the server applies the
    defaults 0.0 and "auto" (R5.4). A filename with no extension defaults the
    upload to ".webm" (R5.6)."""
    recorded = {}

    def fake_transcribe(path, chunk_start=0.0, mode="auto"):
        recorded["path"] = path
        recorded["chunk_start"] = chunk_start
        recorded["mode"] = mode
        return []

    with _EnvGuard("0"):
        client = TestClient(server.create_app(transcribe=fake_transcribe))
        r = client.post(
            "/transcribe_chunk",
            files={"file": ("noext", b"dummy-bytes", "application/octet-stream")},
            data={"session_id": "s1", "chunk_index": "2"},  # no chunk_start/mode
        )

    body = r.json()
    check("defaults: echoes chunk_index", body.get("chunk_index") == 2)
    check("defaults: chunk_start defaults to 0.0", recorded.get("chunk_start") == 0.0)
    check("defaults: mode defaults to 'auto'", recorded.get("mode") == "auto")
    # Extensionless filename -> ".webm" default (R5.6).
    check("defaults: extensionless upload becomes upload.webm",
          os.path.basename(recorded.get("path", "")) == "upload.webm")


if __name__ == "__main__":
    print(f"fastapi TestClient available: {HAS_TESTCLIENT}")
    if not HAS_TESTCLIENT:
        print("  (fastapi TestClient unavailable: non-fake server tests skipped)")
        print(f"\n{PASSED} passed, {FAILED} failed")
        sys.exit(0)

    print("test_nonfake_injected_envelope_and_cleanup")
    test_nonfake_injected_envelope_and_cleanup()
    print("test_nonfake_cleanup_when_transcribe_raises")
    test_nonfake_cleanup_when_transcribe_raises()
    print("test_nonfake_form_field_defaults")
    test_nonfake_form_field_defaults()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
