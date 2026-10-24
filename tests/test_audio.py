"""Tests for the ffmpeg-based audio loader and `has_audio`.

Covers:
- Property 13: int16 -> float32 decoding normalizes samples into [-1.0, 1.0)
- Property 11: audio usable iff decoded waveform exceeds 0.2 seconds
- Edge case: empty ffmpeg stdout yields a length-0 float32 array (R3.4)

Uses hypothesis when importable, otherwise falls back to table-driven cases, so
the suite runs with only numpy. No real ffmpeg binary is required: subprocess /
load_audio are monkeypatched. Any real-ffmpeg assertion is gated behind the
binary being present.

Run: python -m tests.test_audio   (from the project root)
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read import audio as audio_mod  # noqa: E402

SAMPLE_RATE = 16000

try:  # Use hypothesis when available; otherwise fall back to table-driven cases.
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAVE_HYPOTHESIS = True
except Exception:  # pragma: no cover - exercised only when hypothesis is absent
    HAVE_HYPOTHESIS = False


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


class _FakeCompleted:
    def __init__(self, stdout: bytes):
        self.stdout = stdout
        self.stderr = b""
        self.returncode = 0


def _patch_subprocess(monkeyish, stdout_bytes: bytes):
    """Install a fake subprocess module into audio.load_audio's lazy import.

    `monkeyish` is either a pytest monkeypatch or our tiny shim (see below).
    """
    fake = types.ModuleType("subprocess")

    def run(cmd, capture_output=False, **kwargs):  # noqa: ANN001
        return _FakeCompleted(stdout_bytes)

    fake.run = run
    monkeyish.setitem(sys.modules, "subprocess", fake)


class _Monkey:
    """Minimal monkeypatch shim so the table-driven path works without pytest."""

    def __init__(self):
        self._undo = []

    def setitem(self, dic, key, value):
        had = key in dic
        old = dic.get(key)
        dic[key] = value
        self._undo.append(lambda: (dic.__setitem__(key, old) if had else dic.pop(key, None)))

    def setattr(self, obj, name, value):
        had = hasattr(obj, name)
        old = getattr(obj, name, None)
        setattr(obj, name, value)
        self._undo.append(lambda: (setattr(obj, name, old) if had else delattr(obj, name)))

    def undo(self):
        while self._undo:
            self._undo.pop()()


def _int16_to_bytes(values) -> bytes:
    return np.asarray(values, dtype="<i2").tobytes()


# --- Property 13 core assertion -------------------------------------------- #


def _assert_decode_normalizes(int16_values):
    # Feature: colab-integration, Property 13: int16->float32 decoding normalizes
    # samples into [-1.0, 1.0)
    mk = _Monkey()
    try:
        _patch_subprocess(mk, _int16_to_bytes(int16_values))
        out = audio_mod.load_audio("whatever.webm", SAMPLE_RATE)

        expected = np.asarray(int16_values, dtype=np.int16).astype(np.float32) / 32768.0
        assert out.dtype == np.float32, out.dtype
        assert len(out) == len(int16_values), (len(out), len(int16_values))
        assert np.array_equal(out, expected), "elementwise samples / 32768.0"
        if len(out):
            # int16 range is [-32768, 32767] -> normalized range is [-1.0, 1.0)
            assert out.min() >= -1.0, out.min()
            assert out.max() < 1.0, out.max()
    finally:
        mk.undo()


# --- Property 11 core assertion -------------------------------------------- #


def _assert_has_audio_threshold(n: int):
    # Feature: colab-integration, Property 11: audio usable iff decoded waveform
    # exceeds 0.2 seconds
    mk = _Monkey()
    try:
        mk.setattr(
            audio_mod,
            "load_audio",
            lambda path, sample_rate=SAMPLE_RATE: np.zeros(n, dtype=np.float32),
        )
        result = audio_mod.has_audio("whatever.webm", SAMPLE_RATE)
        expected = n > SAMPLE_RATE * 0.2
        assert result is expected or result == expected, (n, result, expected)
    finally:
        mk.undo()


# --- Hypothesis-driven variants -------------------------------------------- #

if HAVE_HYPOTHESIS:

    @settings(max_examples=200)
    @given(st.lists(st.integers(min_value=-32768, max_value=32767), max_size=256))
    def test_decode_normalizes_hypothesis(values):
        _assert_decode_normalizes(values)

    @settings(max_examples=200)
    @given(st.integers(min_value=0, max_value=4 * SAMPLE_RATE))
    def test_has_audio_threshold_hypothesis(n):
        _assert_has_audio_threshold(n)


# --- Table-driven variants (always run, also used by __main__) -------------- #


def test_decode_normalizes_table():
    cases = [
        [],
        [0],
        [1, -1],
        [32767, -32768, 0, 100, -100],
        [-32768] * 8,
        [32767] * 8,
        list(range(-32768, 32768, 4096)),
    ]
    for values in cases:
        _assert_decode_normalizes(values)
    check("Property 13: decode normalizes int16 -> [-1.0, 1.0)", True)


def test_decode_range_bounds():
    # Extremes: max int16 stays < 1.0, min int16 == -1.0 exactly.
    mk = _Monkey()
    try:
        _patch_subprocess(mk, _int16_to_bytes([32767, -32768]))
        out = audio_mod.load_audio("x.webm", SAMPLE_RATE)
        check("max int16 normalizes to < 1.0", out[0] < 1.0)
        check("min int16 normalizes to -1.0", out[1] == -1.0)
    finally:
        mk.undo()


def test_empty_stdout_yields_length_zero_float32():
    # Edge case R3.4: empty ffmpeg stdout -> length-0 float32 array.
    mk = _Monkey()
    try:
        _patch_subprocess(mk, b"")
        out = audio_mod.load_audio("silent.webm", SAMPLE_RATE)
        check("empty stdout -> length 0", len(out) == 0)
        check("empty stdout -> float32 dtype", out.dtype == np.float32)
        check("empty stdout -> 1-D array", out.ndim == 1)
    finally:
        mk.undo()


def test_has_audio_threshold_table():
    boundary = int(SAMPLE_RATE * 0.2)  # 3200
    cases = [0, 1, boundary - 1, boundary, boundary + 1, SAMPLE_RATE, 4 * SAMPLE_RATE]
    for n in cases:
        _assert_has_audio_threshold(n)
    # Explicitly pin the biconditional at the boundary.
    check("Property 11: has_audio False at exactly 0.2s", not _has_audio_for(boundary))
    check("Property 11: has_audio True just above 0.2s", _has_audio_for(boundary + 1))
    check("Property 11: has_audio False below 0.2s", not _has_audio_for(boundary - 1))


def _has_audio_for(n: int) -> bool:
    mk = _Monkey()
    try:
        mk.setattr(
            audio_mod,
            "load_audio",
            lambda path, sample_rate=SAMPLE_RATE: np.zeros(n, dtype=np.float32),
        )
        return audio_mod.has_audio("x.webm", SAMPLE_RATE)
    finally:
        mk.undo()


def test_real_ffmpeg_decode_gated():
    # Gate any real-ffmpeg assertion behind the binary being present.
    if shutil.which("ffmpeg") is None:
        check("real ffmpeg decode (skipped: ffmpeg not on PATH)", True)
        return
    # ffmpeg present: generate 1s of silence and confirm it decodes to ~16000
    # float32 samples. Uses ffmpeg's lavfi source so no media file is needed.
    import subprocess
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        wav = str(Path(d) / "tone.wav")
        gen = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-ar", str(SAMPLE_RATE), "-ac", "1", wav],
            capture_output=True,
        )
        if gen.returncode != 0:
            check("real ffmpeg decode (skipped: generation failed)", True)
            return
        out = audio_mod.load_audio(wav, SAMPLE_RATE)
        check("real ffmpeg decode -> float32", out.dtype == np.float32)
        check("real ffmpeg decode ~1s of samples", abs(len(out) - SAMPLE_RATE) <= SAMPLE_RATE // 10)
        check("real ffmpeg decode within [-1.0, 1.0)", out.size and out.min() >= -1.0 and out.max() < 1.0)
        check("real ffmpeg: has_audio True for 1s clip", audio_mod.has_audio(wav, SAMPLE_RATE))


if __name__ == "__main__":
    print("test_decode_normalizes_table"); test_decode_normalizes_table()
    print("test_decode_range_bounds"); test_decode_range_bounds()
    print("test_empty_stdout_yields_length_zero_float32"); test_empty_stdout_yields_length_zero_float32()
    print("test_has_audio_threshold_table"); test_has_audio_threshold_table()
    print("test_real_ffmpeg_decode_gated"); test_real_ffmpeg_decode_gated()
    if HAVE_HYPOTHESIS:
        print("test_decode_normalizes_hypothesis"); test_decode_normalizes_hypothesis()
        print("test_has_audio_threshold_hypothesis"); test_has_audio_threshold_hypothesis()
        check("hypothesis property runs", True)
    else:
        print("  (hypothesis not installed: ran table-driven cases)")
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
