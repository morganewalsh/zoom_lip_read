"""Property test for ``AutoAvsrLipReader.read_segment`` never raising.

Covers Property 2: ``read_segment`` must swallow any internal failure and
return the empty string, for failures injected at any internal stage
(``_ensure_model``, frame reading, landmark detection, ``_decode``). No torch,
cv2, mediapipe, or real media is required: each stage is monkeypatched on a
real instance to raise, and we assert the public method returns "" and
propagates nothing.

Run: python -m tests.test_avsr_read_segment   (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read.avsr_lip_reader import AutoAvsrLipReader  # noqa: E402

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - exercised on hosts without hypothesis
    HAS_HYPOTHESIS = False

PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  - {name}")
    else:
        FAILED += 1
        print(f"  FAIL- {name}")


class _Boom(Exception):
    """Distinct exception injected into an internal stage."""


# The internal stages read_segment drives, in order. Injecting a failure at any
# one of them (and letting earlier stages succeed as no-ops) must still yield "".
_STAGES = ["_ensure_model", "_read_frames", "_landmarks_detector", "_decode"]


def _reader_failing_at(stage):
    """Return a reader whose stages are stubbed, with ``stage`` raising ``_Boom``.

    Every stage is replaced so no heavy dependency, ffmpeg binary, or real media
    is touched. Stages before the failing one are harmless no-ops; the chosen
    stage raises. ``_landmarks_detector`` is exercised via ``_crops``.
    """
    reader = AutoAvsrLipReader(repo="x", ckpt="y", face_tflite="z")

    def boom(*args, **kwargs):
        raise _Boom(stage)

    # Default no-op stubs so read_segment can reach the targeted stage.
    reader._ensure_model = lambda *a, **k: None
    reader._clip_duration = lambda *a, **k: 10.0
    reader._extract_segment = lambda *a, **k: "seg.mp4"
    # _crops internally calls _read_frames, _landmarks_detector, _video_process.
    reader._read_frames = lambda *a, **k: ([object()], 25.0)
    reader._landmarks_detector = lambda *a, **k: [object()]
    reader._video_process = lambda *a, **k: [object()]
    reader._decode = lambda *a, **k: "decoded text"

    # Now make the targeted stage raise.
    setattr(reader, stage, boom)
    return reader


def _assert_never_raises(stage, start, end):
    # Feature: colab-integration, Property 2: read_segment never raises and
    # returns "" on any internal failure.
    reader = _reader_failing_at(stage)
    result = reader.read_segment("video.mp4", start, end)
    assert result == "", f"stage {stage!r} ({start},{end}): expected '', got {result!r}"


def _assert_success_passthrough(start, end):
    # Sanity: when nothing fails, read_segment returns the decoded text, proving
    # the empty-string result is caused by the injected failure, not by the stubs.
    reader = AutoAvsrLipReader(repo="x", ckpt="y", face_tflite="z")
    reader._ensure_model = lambda *a, **k: None
    reader._clip_duration = lambda *a, **k: 10.0
    reader._extract_segment = lambda *a, **k: "seg.mp4"
    reader._crops = lambda *a, **k: [object()]
    reader._decode = lambda *a, **k: "decoded text"
    assert reader.read_segment("video.mp4", start, end) == "decoded text"


def test_read_segment_never_raises():
    if HAS_HYPOTHESIS:

        @settings(max_examples=200)
        @given(
            st.sampled_from(_STAGES),
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        )
        def run(stage, a, b):
            start, end = min(a, b), max(a, b)
            _assert_never_raises(stage, start, end)

        run()
        check("Property 2: read_segment returns '' on any injected internal failure (hypothesis)", True)
    else:
        cases = [(0.0, 0.5), (1.0, 5.0), (0.0, 0.0), (3.2, 3.2), (10.0, 90.0)]
        ok = True
        for stage in _STAGES:
            for start, end in cases:
                try:
                    _assert_never_raises(stage, start, end)
                except AssertionError as exc:
                    ok = False
                    print(f"    counterexample: {exc}")
        check("Property 2: read_segment returns '' on any injected internal failure (table)", ok)


def test_read_segment_success_passthrough():
    cases = [(0.0, 0.5), (1.0, 5.0), (10.0, 90.0)]
    ok = True
    for start, end in cases:
        try:
            _assert_success_passthrough(start, end)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample: {exc}")
    check("read_segment returns decoded text when no stage fails", ok)


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print("test_read_segment_never_raises"); test_read_segment_never_raises()
    print("test_read_segment_success_passthrough"); test_read_segment_success_passthrough()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
