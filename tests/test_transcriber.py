"""Tests for WhisperTranscriber wiring.

Covers:
- Task 2.1 / R3.2: `transcribe` passes a numpy ndarray waveform (not a path
  string) as the first positional arg to the underlying whisper model.

No ffmpeg or faster-whisper is required: a fake model is injected via
`transcriber._model` and `zoom_lip_read.audio.load_audio` is monkeypatched to
return a small in-memory ndarray.

Run: python -m tests.test_transcriber   (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read import audio as audio_mod  # noqa: E402
from zoom_lip_read.config import WhisperConfig  # noqa: E402
from zoom_lip_read.transcriber import WhisperTranscriber  # noqa: E402

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


class _Monkey:
    """Minimal monkeypatch shim so the test runs without pytest."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        had = hasattr(obj, name)
        old = getattr(obj, name, None)
        setattr(obj, name, value)
        self._undo.append(lambda: (setattr(obj, name, old) if had else delattr(obj, name)))

    def undo(self):
        while self._undo:
            self._undo.pop()()


class _FakeInfo:
    language = "en"
    duration = 1.0


class _FakeModel:
    """Records the first positional arg handed to `transcribe`."""

    def __init__(self):
        self.first_arg = None
        self.calls = 0

    def transcribe(self, audio, **kwargs):  # noqa: ANN001
        self.first_arg = audio
        self.calls += 1
        # Return an empty segment generator plus a fake info object.
        return iter(()), _FakeInfo()


def test_transcribe_passes_ndarray_not_path():
    # Task 2.1 / R3.2: the model must receive a numpy ndarray waveform, not a path.
    mk = _Monkey()
    try:
        waveform = np.asarray([0.0, 0.5, -0.5, 0.25], dtype=np.float32)
        mk.setattr(
            audio_mod,
            "load_audio",
            lambda path, sample_rate=16000: waveform,
        )

        transcriber = WhisperTranscriber(WhisperConfig())
        fake_model = _FakeModel()
        transcriber._model = fake_model  # inject so _load_model short-circuits

        words, language, duration = transcriber.transcribe("some/video.mp4")

        check("model.transcribe was called once", fake_model.calls == 1)
        check(
            "first positional arg is a numpy ndarray",
            isinstance(fake_model.first_arg, np.ndarray),
        )
        check(
            "first positional arg is not a path/str",
            not isinstance(fake_model.first_arg, (str, Path)),
        )
        check(
            "ndarray is the loaded waveform",
            np.array_equal(fake_model.first_arg, waveform),
        )
        # Sanity: the rest of the method still returns its triple.
        check("returns empty words for empty segments", words == [])
        check("returns detected language", language == "en")
        check("returns duration float", abs(duration - 1.0) < 1e-9)
    finally:
        mk.undo()


if __name__ == "__main__":
    print("test_transcribe_passes_ndarray_not_path"); test_transcribe_passes_ndarray_not_path()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
