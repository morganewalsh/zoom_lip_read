"""Audio loading helpers.

Handles pulling a mono waveform out of a video/audio clip at a target sample
rate by invoking the ``ffmpeg`` binary directly. Decoding through ffmpeg (rather
than librosa/faster-whisper's own file loaders) avoids breakage from newer PyAV
releases, and works for any video container (.mp4/.mkv/.webm/...) as long as the
``ffmpeg`` binary is available on PATH at call time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Load `path` as a mono float32 waveform resampled to `sample_rate`.

    Decodes via the ``ffmpeg`` binary to signed 16-bit little-endian PCM and
    normalizes samples into the range [-1.0, 1.0). Returns a 1-D numpy array;
    an empty length-0 float32 array is returned when the input has no decodable
    audio.
    """
    import subprocess  # imported lazily so importing the package needs no ffmpeg

    raw = subprocess.run(
        ["ffmpeg", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
        capture_output=True,
    ).stdout
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def has_audio(path: str | Path, sample_rate: int = 16000) -> bool:
    """Return True when `path` decodes to more than 0.2 seconds of audio."""
    return len(load_audio(path, sample_rate)) > sample_rate * 0.2
