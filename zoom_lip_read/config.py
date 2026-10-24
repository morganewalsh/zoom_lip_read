"""Configuration for the transcription pipeline.

Detector thresholds live here so they can be tuned without touching logic.
Load overrides from a JSON/dict via `PipelineConfig.from_dict` / `from_json_file`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from pathlib import Path
from typing import Any


@dataclass
class NoiseDetectorConfig:
    """Thresholds and knobs for the windowed audio reliability detector.

    The detector slides a window across the audio and computes, per window:
      - RMS loudness (how loud the window is)
      - spectral flatness (how noise-like vs. tonal/voiced the window is;
        close to 1.0 == white-noise-like, close to 0.0 == tonal/voiced)

    A window is flagged as "speech likely unreliable" when it is either too
    quiet to contain intelligible speech, OR too noise-like (high flatness).
    """

    # Window / hop size for analysis, in seconds.
    window_seconds: float = 0.5
    hop_seconds: float = 0.5  # non-overlapping 0.5s windows by default

    # STFT frame used for spectral flatness within each window.
    n_fft: int = 2048
    hop_length: int = 512

    # --- Flagging thresholds (the main tuning knobs) ---

    # Windows quieter than this RMS are flagged (likely dropout / too quiet).
    # RMS is computed on amplitude normalized to [-1, 1].
    rms_silence_threshold: float = 0.01

    # Windows with spectral flatness ABOVE this are flagged (noise-like).
    # Range 0..1. Voiced speech is typically well below ~0.3.
    flatness_noise_threshold: float = 0.45

    # A loud-but-flat window (loud noise over speech) is flagged when it is
    # both louder than this RMS and flatter than flatness_noise_threshold.
    rms_loud_noise_floor: float = 0.02

    # --- Span post-processing ---

    # Merge flagged windows separated by a gap no larger than this (seconds).
    merge_gap_seconds: float = 0.5

    # Drop merged spans shorter than this (seconds). Set to 0 to keep all.
    min_span_seconds: float = 0.5

    # Pad each merged span by this many seconds on each side (clamped to clip).
    span_pad_seconds: float = 0.0


@dataclass
class WhisperConfig:
    """Settings for the faster-whisper transcription step."""

    model_size: str = "base"
    device: str = "auto"  # "cpu", "cuda", or "auto"
    compute_type: str = "default"  # e.g. "int8", "float16", "default"
    language: str | None = None  # None = autodetect
    beam_size: int = 5
    vad_filter: bool = False  # let our own detector decide, by default


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""

    noise: NoiseDetectorConfig = field(default_factory=NoiseDetectorConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)

    # Sample rate to resample audio to for analysis + whisper.
    sample_rate: int = 16000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_file(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PipelineConfig":
        data = dict(data or {})
        noise = _build(NoiseDetectorConfig, data.pop("noise", {}))
        whisper = _build(WhisperConfig, data.pop("whisper", {}))
        known = {f.name for f in fields(cls)} - {"noise", "whisper"}
        extra = {k: v for k, v in data.items() if k in known}
        return cls(noise=noise, whisper=whisper, **extra)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _build(dc_type, data: dict[str, Any]):
    """Construct a dataclass from a dict, ignoring unknown keys."""
    if not data:
        return dc_type()
    valid = {f.name for f in fields(dc_type)}
    return dc_type(**{k: v for k, v in data.items() if k in valid})
