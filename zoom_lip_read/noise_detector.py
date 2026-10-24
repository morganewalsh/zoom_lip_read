"""Windowed audio reliability detector.

Slides fixed-size windows over the audio and, per window, computes RMS loudness
and spectral flatness via librosa. Windows where speech is likely unreliable are
flagged, then adjacent flagged windows are merged into spans.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .config import NoiseDetectorConfig
from .types import FlaggedSpan


class NoiseDetector:
    """Flags audio windows where Whisper output is likely unreliable."""

    def __init__(self, config: NoiseDetectorConfig, sample_rate: int = 16000):
        self.config = config
        self.sample_rate = sample_rate

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect(self, waveform: np.ndarray) -> List[FlaggedSpan]:
        """Return merged, filtered spans where audio is unreliable."""
        windows = self._analyze_windows(waveform)
        flagged = [w for w in windows if w["flagged"]]
        spans = self._merge_windows(flagged)
        spans = self._postprocess(spans, total_duration=len(waveform) / self.sample_rate)
        return spans

    # ------------------------------------------------------------------ #
    # Per-window analysis
    # ------------------------------------------------------------------ #
    def _analyze_windows(self, waveform: np.ndarray) -> List[dict]:
        import librosa

        cfg = self.config
        sr = self.sample_rate
        win_len = max(1, int(round(cfg.window_seconds * sr)))
        hop_len = max(1, int(round(cfg.hop_seconds * sr)))
        n = len(waveform)

        results: List[dict] = []
        if n == 0:
            return results

        for start_sample in range(0, n, hop_len):
            end_sample = min(start_sample + win_len, n)
            frame = waveform[start_sample:end_sample]
            if frame.size == 0:
                continue

            start_t = start_sample / sr
            end_t = end_sample / sr

            # RMS loudness over the raw window amplitude.
            rms = float(np.sqrt(np.mean(np.square(frame)))) if frame.size else 0.0

            # Spectral flatness: geometric-mean / arithmetic-mean of the power
            # spectrum. ~1.0 => noise-like, ~0.0 => tonal/voiced.
            n_fft = min(cfg.n_fft, _next_pow2(frame.size))
            if frame.size < 2:
                flatness = 1.0
            else:
                flat = librosa.feature.spectral_flatness(
                    y=frame,
                    n_fft=n_fft,
                    hop_length=cfg.hop_length,
                )
                flatness = float(np.mean(flat)) if flat.size else 1.0

            flagged, reason = self._is_unreliable(rms, flatness)
            results.append(
                {
                    "start": start_t,
                    "end": end_t,
                    "rms": rms,
                    "flatness": flatness,
                    "flagged": flagged,
                    "reason": reason,
                }
            )
        return results

    def _is_unreliable(self, rms: float, flatness: float) -> tuple[bool, str]:
        """Decide whether a window should be flagged, with a reason string."""
        cfg = self.config
        reasons: List[str] = []

        # Too quiet to carry intelligible speech.
        if rms < cfg.rms_silence_threshold:
            reasons.append(f"low-rms({rms:.4f}<{cfg.rms_silence_threshold})")

        # Noise-like spectrum, and loud enough that it's masking (not silence).
        if flatness > cfg.flatness_noise_threshold and rms >= cfg.rms_loud_noise_floor:
            reasons.append(
                f"noisy(flatness={flatness:.3f}>{cfg.flatness_noise_threshold})"
            )

        return (len(reasons) > 0, ", ".join(reasons))

    # ------------------------------------------------------------------ #
    # Span merging + post-processing
    # ------------------------------------------------------------------ #
    def _merge_windows(self, flagged: List[dict]) -> List[FlaggedSpan]:
        """Merge adjacent/near-adjacent flagged windows into spans."""
        if not flagged:
            return []

        gap = self.config.merge_gap_seconds
        spans: List[FlaggedSpan] = []
        cur_start = flagged[0]["start"]
        cur_end = flagged[0]["end"]
        cur_reasons = {flagged[0]["reason"]}

        for w in flagged[1:]:
            if w["start"] - cur_end <= gap + 1e-9:
                cur_end = max(cur_end, w["end"])
                if w["reason"]:
                    cur_reasons.add(w["reason"])
            else:
                spans.append(
                    FlaggedSpan(cur_start, cur_end, reason="; ".join(sorted(cur_reasons)))
                )
                cur_start, cur_end = w["start"], w["end"]
                cur_reasons = {w["reason"]}

        spans.append(
            FlaggedSpan(cur_start, cur_end, reason="; ".join(sorted(cur_reasons)))
        )
        return spans

    def _postprocess(
        self, spans: List[FlaggedSpan], total_duration: float
    ) -> List[FlaggedSpan]:
        cfg = self.config
        out: List[FlaggedSpan] = []
        for s in spans:
            start = max(0.0, s.start - cfg.span_pad_seconds)
            end = min(total_duration, s.end + cfg.span_pad_seconds)
            span = FlaggedSpan(start, end, reason=s.reason)
            if span.duration + 1e-9 >= cfg.min_span_seconds:
                out.append(span)

        # Padding may cause overlaps; coalesce once more.
        return _coalesce(out)


def _coalesce(spans: List[FlaggedSpan]) -> List[FlaggedSpan]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: s.start)
    merged = [spans[0]]
    for s in spans[1:]:
        last = merged[-1]
        if s.start <= last.end + 1e-9:
            reason = "; ".join(r for r in {last.reason, s.reason} if r)
            merged[-1] = FlaggedSpan(last.start, max(last.end, s.end), reason=reason)
        else:
            merged.append(s)
    return merged


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return max(2, p)
