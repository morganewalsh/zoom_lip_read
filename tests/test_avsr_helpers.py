"""Property tests for the pure helpers in ``zoom_lip_read.avsr_lip_reader``.

These cover the extractable, heavy-dependency-free logic: ``_scaled_size``,
``_pad``, and ``_env_path``. They run on a plain machine with only the standard
library (hypothesis is used when importable, otherwise a table-driven fallback
exercises the same properties).

Run: python -m tests.test_avsr_helpers   (from the project root)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read.avsr_lip_reader import (  # noqa: E402
    MAX_SIDE,
    PAD_TO_SEC,
    _env_path,
    _pad,
    _scaled_size,
)

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


# --- Shared property bodies (assert-based so hypothesis can drive them) ------


def _assert_scaled_size(h, w):
    # Feature: colab-integration, Property 1: Frame downscaling caps the larger
    # side at 960 and preserves aspect ratio.
    out_h, out_w = _scaled_size(h, w)
    if max(h, w) <= MAX_SIDE:
        assert (out_h, out_w) == (h, w), f"expected unchanged for ({h},{w}), got ({out_h},{out_w})"
        return
    # Larger side is capped at 960 within one pixel of integer rounding.
    assert max(out_h, out_w) <= MAX_SIDE, f"larger side {max(out_h, out_w)} exceeds {MAX_SIDE}"
    assert abs(max(out_h, out_w) - MAX_SIDE) <= 1, f"larger side {max(out_h, out_w)} not near {MAX_SIDE}"
    # Aspect ratio preserved within one pixel of rounding (truncating toward
    # zero, matching the reference `int(dim * scale)`; the smaller side may
    # legitimately round down to 0 for extreme aspect ratios).
    scale = MAX_SIDE / max(h, w)
    assert abs(out_h - int(h * scale)) <= 1
    assert abs(out_w - int(w * scale)) <= 1


def _assert_pad(start, end, total):
    # Feature: colab-integration, Property 12: Short-segment padding is centered
    # on the midpoint and clamped to the clip.
    s, e = _pad(start, end, total)
    # Result stays within [0, total].
    assert 0.0 <= s <= total + 1e-9, f"start {s} out of [0,{total}]"
    assert -1e-9 <= e <= total + 1e-9, f"end {e} out of [0,{total}]"
    assert s <= e + 1e-9
    if end - start < PAD_TO_SEC:
        mid = (start + end) / 2
        assert abs(s - max(0.0, mid - 1.0)) < 1e-9
        assert abs(e - min(total, mid + 1.0)) < 1e-9
    else:
        assert (s, e) == (start, end)


def _assert_env_path(name, default, value):
    # Feature: colab-integration, Property 3: Environment path resolution falls
    # back to the default.
    saved = os.environ.get(name)
    try:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
        result = _env_path(name, default)
        if value:  # non-empty string
            assert result == value, f"expected {value!r}, got {result!r}"
        else:  # unset or empty -> default
            assert result == default, f"expected default {default!r}, got {result!r}"
    finally:
        if saved is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = saved


# --- Property 1: _scaled_size -----------------------------------------------


def test_scaled_size():
    if HAS_HYPOTHESIS:

        @settings(max_examples=300)
        @given(
            st.integers(min_value=1, max_value=4000),
            st.integers(min_value=1, max_value=4000),
        )
        def run(h, w):
            _assert_scaled_size(h, w)

        run()
        check("Property 1: _scaled_size caps larger side at 960, preserves ratio (hypothesis)", True)
    else:
        cases = [
            (1, 1), (960, 960), (960, 1), (1, 960), (500, 800),
            (961, 480), (2000, 1000), (4000, 4000), (4000, 1),
            (1, 4000), (1920, 1080), (1080, 1920), (3000, 1500),
        ]
        ok = True
        for h, w in cases:
            try:
                _assert_scaled_size(h, w)
            except AssertionError as exc:
                ok = False
                print(f"    counterexample ({h},{w}): {exc}")
        check("Property 1: _scaled_size caps larger side at 960, preserves ratio (table)", ok)


# --- Property 12: _pad ------------------------------------------------------


def test_pad():
    if HAS_HYPOTHESIS:

        @settings(max_examples=300)
        @given(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        )
        def run(a, b, c):
            start, end = min(a, b), max(a, b)
            total = max(end, c)  # ensure 0 <= start <= end <= total
            _assert_pad(start, end, total)

        run()
        check("Property 12: _pad centers short segments on midpoint, clamps to clip (hypothesis)", True)
    else:
        cases = [
            (0.0, 0.0, 0.0),
            (1.0, 1.5, 10.0),
            (0.0, 0.5, 10.0),
            (0.0, 0.2, 1.0),
            (1.0, 5.0, 10.0),
            (0.0, 2.0, 2.0),
            (4.0, 4.5, 5.0),
            (9.5, 9.8, 10.0),
            (0.0, 1.9999, 100.0),
            (3.0, 3.0, 3.0),
        ]
        ok = True
        for start, end, total in cases:
            try:
                _assert_pad(start, end, total)
            except AssertionError as exc:
                ok = False
                print(f"    counterexample ({start},{end},{total}): {exc}")
        check("Property 12: _pad centers short segments on midpoint, clamps to clip (table)", ok)


# --- Property 3: _env_path --------------------------------------------------

_ENV_NAMES = [
    ("AUTO_AVSR_REPO", "/content/auto_avsr"),
    ("AVSR_CKPT", "/content/auto_avsr/tutorials/vsr_trlrs2lrs3vox2avsp_base.pth"),
    ("FACE_TFLITE", "/content/auto_avsr/tutorials/blaze_face_short_range.tflite"),
]


def test_env_path():
    if HAS_HYPOTHESIS:

        @settings(max_examples=200)
        @given(st.sampled_from(_ENV_NAMES), st.one_of(st.none(), st.text()))
        def run(name_default, value):
            name, default = name_default
            _assert_env_path(name, default, value)

        run()
        check("Property 3: _env_path falls back to default on unset/empty (hypothesis)", True)
    else:
        value_cases = [None, "", "/custom/path", "relative/path", "x"]
        ok = True
        for name, default in _ENV_NAMES:
            for value in value_cases:
                try:
                    _assert_env_path(name, default, value)
                except AssertionError as exc:
                    ok = False
                    print(f"    counterexample ({name}, value={value!r}): {exc}")
        check("Property 3: _env_path falls back to default on unset/empty (table)", ok)


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print("test_scaled_size"); test_scaled_size()
    print("test_pad"); test_pad()
    print("test_env_path"); test_env_path()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
