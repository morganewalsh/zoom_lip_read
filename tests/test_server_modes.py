"""Mode, partition, and offset property tests for `server._default_transcribe`.

The real backend-style entry (`_default_transcribe`) lazily imports its heavy
machinery -- the ffmpeg audio loader, faster-whisper, the noise detector, and
the auto_avsr lip reader -- at call time. These tests drive the FULL mode logic
in-memory by monkeypatching those lazily-imported names on their SOURCE modules
(so the `from .x import y` inside `_default_transcribe` picks up the fakes), plus
`server._clip_duration`. No torch, cv2, mediapipe, faster-whisper, ffmpeg binary,
or real media is touched (R10.5).

Covers:
- Property 8: Transcription modes are mutually exclusive in which stage runs --
  `audio_only` emits no lip segment and never calls the lip reader; `lip_only`
  never calls the transcriber and yields exactly one lip record
  `[chunk_start, round(duration + chunk_start, 2)]`.
- Property 9: `auto` mode partitions words by span midpoint and emits one lip
  segment per span -- a word is audio content iff its midpoint lies outside
  every span; lip-segment count equals span count.
- Property 10: Absent/unreadable audio collapses the chunk to a single
  full-length lip span -- a single lip record `[chunk_start,
  round(d + chunk_start, 2)]`.
- Property 6 (non-fake half): the chunk-start offset is additive across every
  returned segment within 2-dp rounding.

Run: python -m tests.test_server_modes   (from the project root)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from zoom_lip_read import server  # noqa: E402  (light import, no heavy deps)
from zoom_lip_read.types import Word  # noqa: E402
from zoom_lip_read.types import FlaggedSpan  # noqa: E402

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
# Fakes + harness that patches the lazily-imported names in `_default_transcribe`
# --------------------------------------------------------------------------- #


@dataclass
class _Calls:
    """Records whether each heavy stage was invoked, for mutual-exclusion checks."""

    transcriber_built: int = 0
    transcriber_calls: int = 0
    detector_calls: int = 0
    lip_reads: int = 0


class _FakeTranscriber:
    """Stands in for `WhisperTranscriber`; returns the pre-seeded word list."""

    words: list = []
    calls: _Calls | None = None

    def __init__(self, config):
        if _FakeTranscriber.calls is not None:
            _FakeTranscriber.calls.transcriber_built += 1
        self.config = config

    def transcribe(self, path):
        if _FakeTranscriber.calls is not None:
            _FakeTranscriber.calls.transcriber_calls += 1
        # Matches the real contract: (words, language, duration).
        return list(_FakeTranscriber.words), "en", 0.0


class _FakeDetector:
    """Stands in for `NoiseDetector`; returns the pre-seeded flagged spans."""

    spans: list = []  # list of (start, end) tuples
    calls: _Calls | None = None

    def __init__(self, config, sample_rate=16000):
        self.config = config
        self.sample_rate = sample_rate

    def detect(self, waveform):
        if _FakeDetector.calls is not None:
            _FakeDetector.calls.detector_calls += 1
        return [FlaggedSpan(a, b, reason="noisy") for a, b in _FakeDetector.spans]


class _FakeLipReader:
    """NullLipReader-style stand-in for `AutoAvsrLipReader` (never raises)."""

    text_for = staticmethod(lambda a, b: "")  # default: empty => unclear
    calls: _Calls | None = None

    def __init__(self, *a, **k):
        pass

    def read_segment(self, video_path, start, end):
        if _FakeLipReader.calls is not None:
            _FakeLipReader.calls.lip_reads += 1
        return _FakeLipReader.text_for(start, end)


class _Harness:
    """Patch every lazily-imported name used by `_default_transcribe`.

    `_default_transcribe` does `from .audio import load_audio, has_audio`,
    `from .transcriber import WhisperTranscriber`, etc. at call time, so patching
    the attributes on the SOURCE modules makes those imports resolve to our
    fakes. `server._clip_duration` is patched directly on the server module.
    """

    def __init__(self, *, words, spans, duration, has_audio, lip_text=None):
        self.words = words
        self.spans = spans
        self.duration = duration
        self.has_audio = has_audio
        self.lip_text = lip_text or (lambda a, b: "")
        self.calls = _Calls()
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
        self._saved["load_audio"] = audio_mod.load_audio
        self._saved["has_audio"] = audio_mod.has_audio
        self._saved["WhisperTranscriber"] = transcriber_mod.WhisperTranscriber
        self._saved["NoiseDetector"] = noise_mod.NoiseDetector
        self._saved["AutoAvsrLipReader"] = avsr_mod.AutoAvsrLipReader
        self._saved["_clip_duration"] = server._clip_duration

        # Seed fakes.
        _FakeTranscriber.words = self.words
        _FakeTranscriber.calls = self.calls
        _FakeDetector.spans = self.spans
        _FakeDetector.calls = self.calls
        _FakeLipReader.text_for = staticmethod(self.lip_text)
        _FakeLipReader.calls = self.calls

        # A tiny non-empty waveform is enough: the detector is faked. Use a
        # length proportional to has_audio only for realism; detector ignores it.
        waveform = np.zeros(8000, dtype=np.float32)

        # Patch.
        audio_mod.load_audio = lambda path, sr=16000: waveform
        audio_mod.has_audio = lambda path, sr=16000: self.has_audio
        transcriber_mod.WhisperTranscriber = _FakeTranscriber
        noise_mod.NoiseDetector = _FakeDetector
        avsr_mod.AutoAvsrLipReader = _FakeLipReader
        server._clip_duration = lambda path: self.duration
        return self

    def __exit__(self, *exc):
        self._audio_mod.load_audio = self._saved["load_audio"]
        self._audio_mod.has_audio = self._saved["has_audio"]
        self._transcriber_mod.WhisperTranscriber = self._saved["WhisperTranscriber"]
        self._noise_mod.NoiseDetector = self._saved["NoiseDetector"]
        self._avsr_mod.AutoAvsrLipReader = self._saved["AutoAvsrLipReader"]
        server._clip_duration = self._saved["_clip_duration"]
        _FakeTranscriber.calls = None
        _FakeDetector.calls = None
        _FakeLipReader.calls = None
        return False


def _words(pairs):
    """Build Word list from (start, end) pairs with generated text."""
    return [Word(text=f"w{i}", start=s, end=e) for i, (s, e) in enumerate(pairs)]


# --------------------------------------------------------------------------- #
# Property 8: modes are mutually exclusive in which stage they run
# --------------------------------------------------------------------------- #


def _assert_audio_only(words_pairs, chunk_start):
    # Feature: colab-integration, Property 8: Transcription modes are mutually
    # exclusive -- audio_only emits no lip segment and never invokes the lip
    # reader.
    with _Harness(
        words=_words(words_pairs), spans=[(1.0, 2.0)], duration=10.0, has_audio=True
    ) as h:
        recs = server._default_transcribe("clip.mp4", chunk_start=chunk_start, mode="audio_only")
    assert all(r["source"] == "audio" for r in recs), recs
    assert h.calls.lip_reads == 0, f"lip reader invoked in audio_only: {h.calls}"
    assert h.calls.transcriber_calls == 1, h.calls


def _assert_lip_only(duration, chunk_start):
    # Feature: colab-integration, Property 8: lip_only never invokes the
    # transcriber and yields exactly one lip record spanning the whole clip.
    with _Harness(
        words=_words([(0.0, 1.0)]), spans=[(1.0, 2.0)], duration=duration, has_audio=True
    ) as h:
        recs = server._default_transcribe("clip.mp4", chunk_start=chunk_start, mode="lip_only")
    assert len(recs) == 1, recs
    rec = recs[0]
    assert rec["source"] == "lip", rec
    assert rec["start"] == round(0.0 + chunk_start, 2), rec
    assert rec["end"] == round(duration + chunk_start, 2), rec
    assert h.calls.transcriber_calls == 0, f"transcriber invoked in lip_only: {h.calls}"
    assert h.calls.transcriber_built == 0, h.calls
    assert h.calls.lip_reads == 1, h.calls


def test_property8_mode_exclusivity_table():
    ok = True
    audio_cases = [
        ([(0.0, 0.5), (0.6, 1.0), (5.0, 5.5)], 0.0),
        ([(0.0, 0.5)], 3.25),
        ([], 10.0),
    ]
    for pairs, cs in audio_cases:
        try:
            _assert_audio_only(pairs, cs)
        except AssertionError as exc:
            ok = False
            print(f"    audio_only counterexample pairs={pairs} cs={cs}: {exc}")
    for duration, cs in [(10.0, 0.0), (4.7, 2.6), (0.0, 1.0), (123.45, 999.99)]:
        try:
            _assert_lip_only(duration, cs)
        except AssertionError as exc:
            ok = False
            print(f"    lip_only counterexample d={duration} cs={cs}: {exc}")
    check("Property 8: modes mutually exclusive in which stage runs (table)", ok)


# --------------------------------------------------------------------------- #
# Property 9: auto partitions words by span midpoint, one lip segment per span
# --------------------------------------------------------------------------- #


def _assert_auto_partition(words_pairs, spans, chunk_start):
    # Feature: colab-integration, Property 9: auto mode partitions words by span
    # midpoint and emits one lip segment per span.
    words = _words(words_pairs)
    with _Harness(
        words=words, spans=spans, duration=20.0, has_audio=True
    ) as h:
        recs = server._default_transcribe("clip.mp4", chunk_start=chunk_start, mode="auto")

    lip_recs = [r for r in recs if r["source"] == "lip"]
    audio_recs = [r for r in recs if r["source"] == "audio"]

    # One lip segment per span.
    assert len(lip_recs) == len(spans), (len(lip_recs), len(spans))
    assert h.calls.lip_reads == len(spans), h.calls
    assert h.calls.detector_calls == 1, h.calls

    # A word is audio content iff its midpoint lies outside EVERY span.
    def outside_all(w):
        mid = (w.start + w.end) / 2
        return not any(a <= mid <= b for a, b in spans)

    kept = [w for w in words if outside_all(w)]
    # Every kept word's text should appear in some audio record; every dropped
    # word's text should appear in none.
    audio_text = " ".join(r["text"] for r in audio_recs)
    for w in words:
        present = w.text in audio_text.split()
        assert present == outside_all(w), (w.text, present, outside_all(w))

    # Records sorted by start.
    starts = [r["start"] for r in recs]
    assert starts == sorted(starts), starts
    return kept


def test_property9_auto_partition_table():
    ok = True
    cases = [
        # (word pairs, spans, chunk_start). Spans are non-empty here: the
        # empty-detector auto fallback to a single whole-clip span is R7.6
        # behavior and is covered by Property 10, not this partition property.
        ([(0.0, 0.4), (1.0, 1.4), (5.0, 5.4)], [(0.9, 1.5)], 0.0),
        ([(0.0, 0.4), (2.0, 2.4), (8.0, 8.4)], [(1.5, 2.5), (7.5, 9.0)], 2.0),
        ([(0.0, 0.2)], [(0.0, 10.0)], 1.0),  # word swallowed by a full span
    ]
    for pairs, spans, cs in cases:
        try:
            _assert_auto_partition(pairs, spans, cs)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample pairs={pairs} spans={spans} cs={cs}: {exc}")
    check("Property 9: auto partitions by midpoint, one lip seg per span (table)", ok)


# --------------------------------------------------------------------------- #
# Property 10: absent/unreadable audio -> single full-length lip span
# --------------------------------------------------------------------------- #


def _assert_absent_audio(duration, chunk_start, *, mode, has_audio, spans):
    # Feature: colab-integration, Property 10: Absent/unreadable audio collapses
    # the chunk to a single full-length lip span.
    with _Harness(
        words=_words([(0.0, 1.0)]), spans=spans, duration=duration, has_audio=has_audio
    ) as h:
        recs = server._default_transcribe("clip.mp4", chunk_start=chunk_start, mode=mode)
    assert len(recs) == 1, recs
    rec = recs[0]
    assert rec["source"] == "lip", rec
    assert rec["start"] == round(0.0 + chunk_start, 2), rec
    assert rec["end"] == round(duration + chunk_start, 2), rec
    assert rec["unclear"] == (rec["text"] == ""), rec


def test_property10_absent_audio_table():
    ok = True
    # No usable audio: both auto and audio_only hit the no-audio branch first.
    for mode in ("auto", "audio_only", "lip_only"):
        for duration, cs in [(10.0, 0.0), (4.7, 2.6), (0.0, 5.0)]:
            try:
                _assert_absent_audio(duration, cs, mode=mode, has_audio=False, spans=[(1.0, 2.0)])
            except AssertionError as exc:
                ok = False
                print(f"    no-audio counterexample mode={mode} d={duration} cs={cs}: {exc}")
    # Unreadable audio in auto: detector yields no spans -> whole-clip fallback.
    for duration, cs in [(10.0, 0.0), (8.0, 1.5)]:
        try:
            _assert_absent_audio(duration, cs, mode="auto", has_audio=True, spans=[])
        except AssertionError as exc:
            ok = False
            print(f"    unreadable-auto counterexample d={duration} cs={cs}: {exc}")
    check("Property 10: absent/unreadable audio -> one full-length lip span (table)", ok)


# --------------------------------------------------------------------------- #
# Property 6 (non-fake half): chunk-start offset additive over every segment
# --------------------------------------------------------------------------- #


def _assert_offset_additive(chunk_start):
    # Feature: colab-integration, Property 6 (non-fake half): the chunk-start
    # offset is additive across every returned segment within 2-dp rounding.
    words_pairs = [(0.0, 0.4), (5.0, 5.4)]
    spans = [(2.0, 3.0)]
    duration = 10.0
    with _Harness(words=_words(words_pairs), spans=spans, duration=duration, has_audio=True):
        base = server._default_transcribe("clip.mp4", chunk_start=0.0, mode="auto")
    with _Harness(words=_words(words_pairs), spans=spans, duration=duration, has_audio=True):
        shifted = server._default_transcribe("clip.mp4", chunk_start=chunk_start, mode="auto")

    assert len(base) == len(shifted), (len(base), len(shifted))
    for b, s in zip(base, shifted):
        assert s["source"] == b["source"], (b, s)
        assert round(s["start"], 2) == round(b["start"] + chunk_start, 2), (b, s)
        assert round(s["end"], 2) == round(b["end"] + chunk_start, 2), (b, s)


def test_property6_offset_additive_table():
    ok = True
    for cs in [0.0, 0.5, 1.0, 2.6, 10.0, 123.45, 999.99]:
        try:
            _assert_offset_additive(cs)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample chunk_start={cs}: {exc}")
    check("Property 6 (non-fake half): offset additive over segments (table)", ok)


# --------------------------------------------------------------------------- #
# No heavy module leaked while exercising the mode logic
# --------------------------------------------------------------------------- #


def test_no_heavy_modules_loaded():
    loaded = [m for m in HEAVY_MODULES if m in sys.modules]
    check("no heavy model module loaded by mode tests", loaded == [])


# --------------------------------------------------------------------------- #
# Hypothesis-driven variants
# --------------------------------------------------------------------------- #

if HAS_HYPOTHESIS:

    _ts = st.floats(min_value=0.0, max_value=30.0, allow_nan=False, allow_infinity=False)

    def _word_pairs_strategy():
        return st.lists(
            st.tuples(_ts, st.floats(min_value=0.0, max_value=2.0, allow_nan=False)),
            max_size=6,
        ).map(lambda lst: [(s, s + d) for s, d in lst])

    def _spans_strategy():
        return st.lists(
            st.tuples(_ts, st.floats(min_value=0.1, max_value=5.0, allow_nan=False)),
            max_size=4,
        ).map(lambda lst: [(s, s + d) for s, d in lst])

    @settings(max_examples=150, deadline=None)
    @given(_word_pairs_strategy(),
           _spans_strategy().filter(lambda s: len(s) > 0),
           st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False))
    def test_property9_auto_partition_hypothesis(pairs, spans, cs):
        # Non-empty spans only: the empty-detector auto fallback (R7.6) is a
        # Property 10 concern, so the partition invariant assumes real spans.
        _assert_auto_partition(pairs, spans, cs)

    @settings(max_examples=150, deadline=None)
    @given(st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False))
    def test_property6_offset_additive_hypothesis(cs):
        _assert_offset_additive(cs)

    @settings(max_examples=100, deadline=None)
    @given(st.floats(min_value=0.0, max_value=1e3, allow_nan=False, allow_infinity=False),
           st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False))
    def test_property10_absent_audio_hypothesis(duration, cs):
        _assert_absent_audio(duration, cs, mode="lip_only", has_audio=True, spans=[(1.0, 2.0)])


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print("test_property8_mode_exclusivity_table"); test_property8_mode_exclusivity_table()
    print("test_property9_auto_partition_table"); test_property9_auto_partition_table()
    print("test_property10_absent_audio_table"); test_property10_absent_audio_table()
    print("test_property6_offset_additive_table"); test_property6_offset_additive_table()
    if HAS_HYPOTHESIS:
        print("test_property9_auto_partition_hypothesis"); test_property9_auto_partition_hypothesis()
        check("Property 9: auto partition (hypothesis)", True)
        print("test_property6_offset_additive_hypothesis"); test_property6_offset_additive_hypothesis()
        check("Property 6 (non-fake half): offset additive (hypothesis)", True)
        print("test_property10_absent_audio_hypothesis"); test_property10_absent_audio_hypothesis()
        check("Property 10: absent audio (hypothesis)", True)
    else:
        print("  (hypothesis not installed: ran table-driven cases)")
    print("test_no_heavy_modules_loaded"); test_no_heavy_modules_loaded()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
