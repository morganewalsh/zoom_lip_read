"""Logic tests that don't require faster-whisper or a real lip model.

Run: python -m tests.test_logic   (from the project root)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- Stub out librosa so NoiseDetector can run without the real package. ---
def _install_fake_librosa():
    if "librosa" in sys.modules:
        return
    fake = types.ModuleType("librosa")
    feat = types.ModuleType("librosa.feature")

    def spectral_flatness(y=None, n_fft=2048, hop_length=512, **kwargs):
        # Approximate flatness: geometric mean / arithmetic mean of power spectrum.
        if y is None or len(y) < 2:
            return np.array([[1.0]])
        spec = np.abs(np.fft.rfft(y)) ** 2 + 1e-10
        gmean = np.exp(np.mean(np.log(spec)))
        amean = np.mean(spec)
        return np.array([[float(gmean / amean)]])

    feat.spectral_flatness = spectral_flatness
    fake.feature = feat
    sys.modules["librosa"] = fake
    sys.modules["librosa.feature"] = feat


_install_fake_librosa()

from zoom_lip_read.config import NoiseDetectorConfig, PipelineConfig  # noqa: E402
from zoom_lip_read.noise_detector import NoiseDetector  # noqa: E402
from zoom_lip_read.pipeline import TranscriptionPipeline  # noqa: E402
from zoom_lip_read.lip_reader import NullLipReader  # noqa: E402
from zoom_lip_read.types import Word  # noqa: E402

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


def make_signal(sr=16000):
    """10s signal: silence 2-3s, white noise 5-6.5s, tone elsewhere."""
    dur = 10.0
    t = np.arange(int(dur * sr)) / sr
    sig = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)  # voiced-ish tone
    # Silence window 2.0 - 3.0s
    sig[int(2.0 * sr):int(3.0 * sr)] = 0.0
    # Loud white noise 5.0 - 6.5s
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(int(1.5 * sr)).astype(np.float32) * 0.2
    sig[int(5.0 * sr):int(6.5 * sr)] = noise
    return sig


def test_detector_flags_silence_and_noise():
    sr = 16000
    sig = make_signal(sr)
    det = NoiseDetector(NoiseDetectorConfig(), sample_rate=sr)
    spans = det.detect(sig)

    print("  spans:", [(round(s.start, 2), round(s.end, 2), s.reason) for s in spans])

    def covers(center):
        return any(s.start <= center <= s.end for s in spans)

    check("flags silence region (~2.5s)", covers(2.5))
    check("flags noise region (~5.75s)", covers(5.75))
    check("does not flag tonal region (~8s)", not covers(8.0))


def test_merge_adjacent_windows():
    cfg = NoiseDetectorConfig(merge_gap_seconds=0.5, min_span_seconds=0.0)
    det = NoiseDetector(cfg, sample_rate=16000)
    flagged = [
        {"start": 0.0, "end": 0.5, "reason": "a"},
        {"start": 0.5, "end": 1.0, "reason": "a"},
        {"start": 1.0, "end": 1.5, "reason": "b"},
        # gap > 0.5s here -> new span
        {"start": 3.0, "end": 3.5, "reason": "c"},
    ]
    spans = det._merge_windows(flagged)
    check("merges contiguous windows into one span", len(spans) == 2)
    check("first merged span is 0.0-1.5", abs(spans[0].start) < 1e-9 and abs(spans[0].end - 1.5) < 1e-9)
    check("second span is 3.0-3.5", abs(spans[1].start - 3.0) < 1e-9)


def test_min_span_filter():
    cfg = NoiseDetectorConfig(merge_gap_seconds=0.0, min_span_seconds=0.6)
    det = NoiseDetector(cfg, sample_rate=16000)
    spans = det._postprocess(
        [type("S", (), {"start": 0.0, "end": 0.4, "reason": ""})()], total_duration=10.0
    )
    check("drops spans shorter than min_span_seconds", len(spans) == 0)


def test_pipeline_assembly_replaces_words():
    """Whisper words inside a flagged span are replaced by lip-read text."""
    config = PipelineConfig()
    pipe = TranscriptionPipeline(lip_reader=NullLipReader("hello"), config=config)

    words = [
        Word("good", 0.0, 0.4),
        Word("morning", 0.4, 0.9),
        Word("GARBLED1", 2.1, 2.4),   # inside flagged span 2.0-3.0
        Word("GARBLED2", 2.5, 2.9),   # inside flagged span
        Word("everyone", 3.2, 3.8),
    ]
    from zoom_lip_read.types import FlaggedSpan
    spans = [FlaggedSpan(2.0, 3.0, reason="low-rms")]

    segments = pipe._assemble_segments(words, spans, video_path="x.mp4")
    sources = [s.source for s in segments]
    print("  segments:", [(s.source, round(s.start, 2), round(s.end, 2), s.text) for s in segments])

    check("produces audio, lip, audio in order", sources == ["audio", "lip", "audio"])
    lip_seg = [s for s in segments if s.source == "lip"][0]
    check("lip segment carries lip-read text", lip_seg.text == "hello")
    check("garbled whisper words removed from audio segments",
          all("GARBLED" not in s.text for s in segments if s.source == "audio"))

    from zoom_lip_read.types import TranscriptResult
    res = TranscriptResult(segments=segments, flagged_spans=spans)
    print("  text:", res.text)
    check("rendered text wraps lip span", "[lip-read: hello]" in res.text)
    check("rendered text keeps audio words", "good morning" in res.text and "everyone" in res.text)


def test_config_roundtrip(tmp="_cfg_tmp.json"):
    cfg = PipelineConfig()
    cfg.noise.rms_silence_threshold = 0.02
    cfg.to_json_file(tmp)
    loaded = PipelineConfig.from_json_file(tmp)
    check("config roundtrips threshold", abs(loaded.noise.rms_silence_threshold - 0.02) < 1e-9)
    check("config ignores unknown keys",
          PipelineConfig.from_dict({"noise": {"bogus": 1}, "whisper": {}}).noise.window_seconds == 0.5)
    Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    print("test_detector_flags_silence_and_noise"); test_detector_flags_silence_and_noise()
    print("test_merge_adjacent_windows"); test_merge_adjacent_windows()
    print("test_min_span_filter"); test_min_span_filter()
    print("test_pipeline_assembly_replaces_words"); test_pipeline_assembly_replaces_words()
    print("test_config_roundtrip"); test_config_roundtrip()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
