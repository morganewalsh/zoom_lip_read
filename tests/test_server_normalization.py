"""Normalization-before-transcription tests for `server._default_transcribe`.

`_default_transcribe` opens a `tempfile.TemporaryDirectory()`, converts the
upload to `chunk.mp4` via `_to_mp4_25fps`, and runs `_clip_duration`,
`load_audio`, `has_audio`, and every `read_segment` against that normalized mp4
so the audio and lip paths share one timeline (R5.12, R5.13). On a normalization
failure it degrades to the raw upload without raising (R5.14).

These tests drive the FULL path in-memory by injecting a fake `_to_mp4_25fps`
(on the server module) that records its args and writes a stub file, plus fakes
for the lazily-imported `_clip_duration`/`load_audio`/`has_audio`/
`AutoAvsrLipReader` on their SOURCE modules (so the `from .x import y` inside
`_default_transcribe` picks up the fakes). No torch, cv2, mediapipe,
faster-whisper, ffmpeg binary, or real media is touched (R10.5).

Covers:
- Property 14: Normalization precedes transcription on a single timeline -- the
  normalizer runs exactly once, and every downstream stage (`_clip_duration`,
  `load_audio`, `has_audio`, `read_segment`) receives the normalized `chunk.mp4`
  path, not the raw upload.
- Failure case (R5.14): when the injected `_to_mp4_25fps` raises, every
  downstream stage receives the RAW upload path and `_default_transcribe`
  returns without raising.

Run: python -m tests.test_server_normalization   (from the project root)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zoom_lip_read import server  # noqa: E402  (light import, no heavy deps)

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - exercised on hosts without hypothesis
    HAS_HYPOTHESIS = False

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


# --------------------------------------------------------------------------- #
# Fakes + harness that records the PATH handed to each downstream stage
# --------------------------------------------------------------------------- #


class _FakeTranscriber:
    """Records the path it was handed; returns no words (keeps the path light)."""

    paths: list = []

    def __init__(self, config):
        self.config = config

    def transcribe(self, path):
        _FakeTranscriber.paths.append(path)
        return [], "en", 0.0


class _FakeDetector:
    """Returns no flagged spans so `auto` collapses to one whole-clip lip span."""

    def __init__(self, config, sample_rate=16000):
        self.config = config
        self.sample_rate = sample_rate

    def detect(self, waveform):
        return []


class _FakeLipReader:
    """NullLipReader-style stand-in; records the path handed to read_segment."""

    paths: list = []

    def __init__(self, *a, **k):
        pass

    def read_segment(self, video_path, start, end):
        _FakeLipReader.paths.append(video_path)
        return ""


class _Harness:
    """Inject `_to_mp4_25fps` (server module) + the lazily-imported downstream
    names on their SOURCE modules, recording the path each stage receives.

    `normalize` controls the injected `_to_mp4_25fps`:
      - "ok":   records (src, dst) and writes a stub file at dst (success path)
      - "raise": raises, so `_default_transcribe` must degrade to the raw upload
    """

    def __init__(self, *, normalize="ok", has_audio=True, duration=10.0):
        self.normalize = normalize
        self.has_audio = has_audio
        self.duration = duration
        # Records, visible to the test after the block.
        self.norm_calls: list = []        # list of (src, dst)
        self.clip_duration_paths: list = []
        self.load_audio_paths: list = []
        self.has_audio_paths: list = []
        self._saved = {}

    def __enter__(self):
        import zoom_lip_read.audio as audio_mod
        import zoom_lip_read.transcriber as transcriber_mod
        import zoom_lip_read.noise_detector as noise_mod
        import zoom_lip_read.avsr_lip_reader as avsr_mod

        self._audio_mod = audio_mod
        self._transcriber_mod = transcriber_mod
        self._noise_mod = noise_mod
        self._avsr_mod = avsr_mod

        # Save originals.
        self._saved["_to_mp4_25fps"] = server._to_mp4_25fps
        self._saved["_clip_duration"] = server._clip_duration
        self._saved["load_audio"] = audio_mod.load_audio
        self._saved["has_audio"] = audio_mod.has_audio
        self._saved["WhisperTranscriber"] = transcriber_mod.WhisperTranscriber
        self._saved["NoiseDetector"] = noise_mod.NoiseDetector
        self._saved["AutoAvsrLipReader"] = avsr_mod.AutoAvsrLipReader

        _FakeTranscriber.paths = []
        _FakeLipReader.paths = []

        waveform = np.zeros(8000, dtype=np.float32)

        def fake_norm(src, dst):
            self.norm_calls.append((str(src), str(dst)))
            if self.normalize == "raise":
                raise RuntimeError("ffmpeg normalization failed")
            # Success: create a stub file at dst so it looks like a real mp4.
            with open(dst, "wb") as f:
                f.write(b"\x00\x00\x00\x18ftypmp42")

        def fake_clip_duration(path):
            self.clip_duration_paths.append(str(path))
            return self.duration

        def fake_load_audio(path, sr=16000):
            self.load_audio_paths.append(str(path))
            return waveform

        def fake_has_audio(path, sr=16000):
            self.has_audio_paths.append(str(path))
            return self.has_audio

        # Patch.
        server._to_mp4_25fps = fake_norm
        server._clip_duration = fake_clip_duration
        audio_mod.load_audio = fake_load_audio
        audio_mod.has_audio = fake_has_audio
        transcriber_mod.WhisperTranscriber = _FakeTranscriber
        noise_mod.NoiseDetector = _FakeDetector
        avsr_mod.AutoAvsrLipReader = _FakeLipReader
        return self

    def __exit__(self, *exc):
        server._to_mp4_25fps = self._saved["_to_mp4_25fps"]
        server._clip_duration = self._saved["_clip_duration"]
        self._audio_mod.load_audio = self._saved["load_audio"]
        self._audio_mod.has_audio = self._saved["has_audio"]
        self._transcriber_mod.WhisperTranscriber = self._saved["WhisperTranscriber"]
        self._noise_mod.NoiseDetector = self._saved["NoiseDetector"]
        self._avsr_mod.AutoAvsrLipReader = self._saved["AutoAvsrLipReader"]
        return False

    @property
    def lip_paths(self):
        return list(_FakeLipReader.paths)

    @property
    def transcriber_paths(self):
        return list(_FakeTranscriber.paths)


# --------------------------------------------------------------------------- #
# Property 14: normalization precedes transcription on a single timeline
# --------------------------------------------------------------------------- #


def _assert_normalized_timeline(raw_path, chunk_start, mode, *, has_audio):
    # Feature: colab-integration, Property 14: Normalization precedes
    # transcription on a single timeline.
    with _Harness(normalize="ok", has_audio=has_audio, duration=10.0) as h:
        recs = server._default_transcribe(raw_path, chunk_start=chunk_start, mode=mode)

    # Normalizer ran exactly once.
    assert len(h.norm_calls) == 1, h.norm_calls
    src, dst = h.norm_calls[0]
    # It was asked to convert the RAW upload into a chunk.mp4.
    assert src == str(raw_path), (src, raw_path)
    assert os.path.basename(dst) == "chunk.mp4", dst

    normalized = dst

    # Every downstream stage saw the normalized chunk.mp4, never the raw upload.
    assert h.clip_duration_paths == [normalized], h.clip_duration_paths
    assert h.load_audio_paths == [normalized], h.load_audio_paths
    # has_audio is consulted for lip_only / usable-audio gating; whenever it runs
    # it must see the normalized media (not the raw upload).
    for p in h.has_audio_paths:
        assert p == normalized, (p, normalized)
    for p in h.transcriber_paths:
        assert p == normalized, (p, normalized)
    for p in h.lip_paths:
        assert p == normalized, (p, normalized)

    # The raw upload path never leaked into any downstream stage.
    all_downstream = (
        h.clip_duration_paths
        + h.load_audio_paths
        + h.has_audio_paths
        + h.transcriber_paths
        + h.lip_paths
    )
    assert str(raw_path) not in all_downstream, all_downstream
    return recs


def test_property14_normalized_timeline_table():
    ok = True
    cases = [
        ("upload.webm", 0.0, "auto", True),
        ("upload.mp4", 2.6, "lip_only", True),
        ("upload.webm", 1.5, "audio_only", True),
        ("recording.mkv", 10.0, "auto", False),   # no usable audio -> lip span
        ("c.webm", 0.0, "lip_only", False),
    ]
    for raw, cs, mode, has_audio in cases:
        try:
            _assert_normalized_timeline(raw, cs, mode, has_audio=has_audio)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample raw={raw} cs={cs} mode={mode} audio={has_audio}: {exc}")
    check("Property 14: normalization precedes transcription on one timeline (table)", ok)


# --------------------------------------------------------------------------- #
# Failure case (R5.14): normalization raises -> raw upload + no raise
# --------------------------------------------------------------------------- #


def _assert_degrade_to_raw(raw_path, chunk_start, mode, *, has_audio):
    # Feature: colab-integration, Property 14: Normalization precedes
    # transcription on a single timeline (failure path, R5.14).
    with _Harness(normalize="raise", has_audio=has_audio, duration=10.0) as h:
        # Must NOT raise even though _to_mp4_25fps raised.
        recs = server._default_transcribe(raw_path, chunk_start=chunk_start, mode=mode)

    # Normalizer was still attempted exactly once.
    assert len(h.norm_calls) == 1, h.norm_calls

    # Every downstream stage degraded to the RAW upload path.
    assert h.clip_duration_paths == [str(raw_path)], h.clip_duration_paths
    assert h.load_audio_paths == [str(raw_path)], h.load_audio_paths
    for p in h.has_audio_paths:
        assert p == str(raw_path), (p, raw_path)
    for p in h.transcriber_paths:
        assert p == str(raw_path), (p, raw_path)
    for p in h.lip_paths:
        assert p == str(raw_path), (p, raw_path)

    # The normalized chunk.mp4 never reached a downstream stage.
    all_downstream = (
        h.clip_duration_paths
        + h.load_audio_paths
        + h.has_audio_paths
        + h.transcriber_paths
        + h.lip_paths
    )
    assert not any(os.path.basename(p) == "chunk.mp4" for p in all_downstream), all_downstream
    return recs


def test_property14_degrade_to_raw_on_failure_table():
    ok = True
    cases = [
        ("upload.webm", 0.0, "auto", True),
        ("upload.mp4", 2.6, "lip_only", True),
        ("upload.webm", 1.5, "audio_only", True),
        ("recording.mkv", 10.0, "auto", False),
        ("c.webm", 5.0, "lip_only", False),
    ]
    for raw, cs, mode, has_audio in cases:
        try:
            recs = _assert_degrade_to_raw(raw, cs, mode, has_audio=has_audio)
            # A result list is still returned (request completes, R5.14).
            assert isinstance(recs, list), recs
        except AssertionError as exc:
            ok = False
            print(f"    counterexample raw={raw} cs={cs} mode={mode} audio={has_audio}: {exc}")
        except Exception as exc:  # pragma: no cover - a raise here is a failure
            ok = False
            print(f"    _default_transcribe RAISED (should degrade) raw={raw}: {exc!r}")
    check("Property 14: normalization failure degrades to raw upload, no raise (table)", ok)


# --------------------------------------------------------------------------- #
# No heavy module leaked while exercising the normalization path
# --------------------------------------------------------------------------- #


def test_no_heavy_modules_loaded():
    loaded = [m for m in HEAVY_MODULES if m in sys.modules]
    check("no heavy model module loaded by normalization tests", loaded == [])


# --------------------------------------------------------------------------- #
# Hypothesis-driven variants
# --------------------------------------------------------------------------- #

if HAS_HYPOTHESIS:

    _MODES = st.sampled_from(["auto", "lip_only", "audio_only"])
    _NAMES = st.sampled_from(["upload.webm", "upload.mp4", "clip.mkv", "a.ogg", "noext"])
    _CS = st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False)

    @settings(max_examples=150, deadline=None)
    @given(_NAMES, _CS, _MODES, st.booleans())
    def test_property14_normalized_timeline_hypothesis(name, cs, mode, has_audio):
        _assert_normalized_timeline(name, cs, mode, has_audio=has_audio)

    @settings(max_examples=150, deadline=None)
    @given(_NAMES, _CS, _MODES, st.booleans())
    def test_property14_degrade_to_raw_hypothesis(name, cs, mode, has_audio):
        recs = _assert_degrade_to_raw(name, cs, mode, has_audio=has_audio)
        assert isinstance(recs, list)


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print("test_property14_normalized_timeline_table")
    test_property14_normalized_timeline_table()
    print("test_property14_degrade_to_raw_on_failure_table")
    test_property14_degrade_to_raw_on_failure_table()
    if HAS_HYPOTHESIS:
        print("test_property14_normalized_timeline_hypothesis")
        test_property14_normalized_timeline_hypothesis()
        check("Property 14: normalized timeline (hypothesis)", True)
        print("test_property14_degrade_to_raw_hypothesis")
        test_property14_degrade_to_raw_hypothesis()
        check("Property 14: degrade to raw on failure (hypothesis)", True)
    else:
        print("  (hypothesis not installed: ran table-driven cases)")
    print("test_no_heavy_modules_loaded")
    test_no_heavy_modules_loaded()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
