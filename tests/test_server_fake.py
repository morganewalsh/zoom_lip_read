"""FAKE_MODE server tests + /health and chunk-offset property tests.

Covers:
- FAKE_MODE test (R10.3): with FAKE_MODE on, `GET /health` reports
  status == "ok" and fake_mode == True, and `POST /transcribe_chunk` returns
  the canned two-segment response offset by `chunk_start` without loading any
  model.
- Property 7: /health reports status "ok" and FAKE_MODE mirrors the
  environment (true iff `FAKE_MODE` unset or "1").
- Property 6 (fake half): the chunk-start offset is additive -- both canned
  segments' start/end equal base + chunk_start within 2-dp rounding.

These tests need FastAPI's TestClient. When it is unavailable they are skipped
cleanly so the suite still runs on a bare machine (numpy only). No heavy ML
dependency (torch/cv2/mediapipe/faster_whisper) is ever imported.

Run: python -m tests.test_server_fake   (from the project root)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read import server  # noqa: E402  (light import, no heavy deps)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - exercised on hosts without hypothesis
    HAS_HYPOTHESIS = False

# TestClient availability gate: skip the whole module when fastapi's TestClient
# (which also needs httpx/starlette) cannot be imported.
try:
    from fastapi.testclient import TestClient

    HAS_TESTCLIENT = True
except Exception:  # pragma: no cover - bare machine without fastapi
    HAS_TESTCLIENT = False

PASSED = 0
FAILED = 0

# Base times from server._canned_segments with chunk_start == 0.
BASE = [
    {"start": 0.0, "end": 2.6, "text": "let's meet friday", "source": "audio"},
    {"start": 2.6, "end": 4.7, "text": "see you then", "source": "lip", "unclear": False},
]

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


def _client():
    # Build a fresh app so middleware/routes are clean per use.
    return TestClient(server.create_app())


# --- FAKE_MODE end-to-end test (R10.3) -------------------------------------


def test_fake_mode_endpoints():
    with _EnvGuard("1"):
        client = _client()

        # /health
        r = client.get("/health")
        body = r.json()
        check("health status ok", body.get("status") == "ok")
        check("health fake_mode true", body.get("fake_mode") is True)

        # /transcribe_chunk returns the canned two-segment response offset by
        # chunk_start.
        cs = 10.0
        r = client.post(
            "/transcribe_chunk",
            files={"file": ("clip.webm", b"not-real-media", "application/octet-stream")},
            data={"session_id": "s1", "chunk_index": "3", "chunk_start": str(cs), "mode": "auto"},
        )
        body = r.json()
        check("echoes chunk_index", body.get("chunk_index") == 3)
        segs = body.get("segments", [])
        check("returns exactly two canned segments", len(segs) == 2)
        if len(segs) == 2:
            check("segment 0 is audio offset by chunk_start",
                  segs[0]["source"] == "audio"
                  and segs[0]["text"] == "let's meet friday"
                  and abs(segs[0]["start"] - (0.0 + cs)) < 1e-9
                  and abs(segs[0]["end"] - (2.6 + cs)) < 1e-9)
            check("segment 1 is lip offset by chunk_start with unclear False",
                  segs[1]["source"] == "lip"
                  and segs[1]["text"] == "see you then"
                  and segs[1]["unclear"] is False
                  and abs(segs[1]["start"] - (2.6 + cs)) < 1e-9
                  and abs(segs[1]["end"] - (4.7 + cs)) < 1e-9)

        # No model loaded: none of the heavy ML modules were imported.
        loaded = [m for m in HEAVY_MODULES if m in sys.modules]
        check("no heavy model module loaded in fake mode", loaded == [])


# --- Property 7: /health mirrors FAKE_MODE env -----------------------------


def _check_health_for_env(env_value):
    # Feature: colab-integration, Property 7: /health reports status "ok" and
    # FAKE_MODE mirrors the environment.
    expected_fake = env_value is None or env_value == "1"
    with _EnvGuard(env_value):
        client = _client()
        body = client.get("/health").json()
        assert body["status"] == "ok", body
        assert body["fake_mode"] is expected_fake, (env_value, body["fake_mode"], expected_fake)


def test_health_mirrors_env_table():
    cases = [None, "1", "0", "", "2", "true", "TRUE", "yes"]
    ok = True
    for env_value in cases:
        try:
            _check_health_for_env(env_value)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample FAKE_MODE={env_value!r}: {exc}")
    check("Property 7: /health status ok + fake_mode mirrors env (table)", ok)


# --- Property 6 (fake half): chunk-start offset is additive ----------------


def _check_offset_additive(chunk_start):
    # Feature: colab-integration, Property 6 (fake half): chunk-start offset is
    # additive -- each canned segment's start/end equals base + chunk_start
    # within 2-dp rounding.
    segs = server._canned_segments(chunk_start)
    assert len(segs) == 2, segs
    for base, seg in zip(BASE, segs):
        assert round(seg["start"], 2) == round(base["start"] + chunk_start, 2), (base, seg)
        assert round(seg["end"], 2) == round(base["end"] + chunk_start, 2), (base, seg)


def test_offset_additive_table():
    cases = [0.0, 0.5, 1.0, 2.6, 10.0, 123.45, 999.99]
    ok = True
    for cs in cases:
        try:
            _check_offset_additive(cs)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample chunk_start={cs}: {exc}")
    check("Property 6 (fake half): canned offset additive (table)", ok)


# --- Hypothesis-driven variants --------------------------------------------

if HAS_HYPOTHESIS:

    @settings(max_examples=200)
    @given(st.sampled_from([None, "1", "0", "", "2", "x", "10", "11"]))
    def test_health_mirrors_env_hypothesis(env_value):
        _check_health_for_env(env_value)

    @settings(max_examples=200)
    @given(st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    def test_offset_additive_hypothesis(chunk_start):
        _check_offset_additive(chunk_start)


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print(f"fastapi TestClient available: {HAS_TESTCLIENT}")
    if not HAS_TESTCLIENT:
        # Property 6 (fake half) needs only the pure helper, so run it regardless.
        print("test_offset_additive_table"); test_offset_additive_table()
        if HAS_HYPOTHESIS:
            print("test_offset_additive_hypothesis"); test_offset_additive_hypothesis()
            check("Property 6 (fake half): canned offset additive (hypothesis)", True)
        print("  (fastapi TestClient unavailable: skipped /health + endpoint tests)")
        print(f"\n{PASSED} passed, {FAILED} failed")
        sys.exit(1 if FAILED else 0)

    print("test_fake_mode_endpoints"); test_fake_mode_endpoints()
    print("test_health_mirrors_env_table"); test_health_mirrors_env_table()
    print("test_offset_additive_table"); test_offset_additive_table()
    if HAS_HYPOTHESIS:
        print("test_health_mirrors_env_hypothesis"); test_health_mirrors_env_hypothesis()
        check("Property 7: /health mirrors env (hypothesis)", True)
        print("test_offset_additive_hypothesis"); test_offset_additive_hypothesis()
        check("Property 6 (fake half): canned offset additive (hypothesis)", True)
    else:
        print("  (hypothesis not installed: ran table-driven cases)")
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
