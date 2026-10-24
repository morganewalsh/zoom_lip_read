"""Lip-reading interface used as a fallback over unreliable audio spans.

The pipeline only depends on the `read_segment` method. Plug in a real
visual-speech-recognition model by subclassing `LipReader` (or providing any
object with a compatible `read_segment`) and passing it to the pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class LipReaderProtocol(Protocol):
    def read_segment(self, video_path: str | Path, start: float, end: float) -> str:
        ...


class LipReader:
    """Base lip reader.

    `read_segment` should run visual speech recognition on the video frames
    between `start` and `end` (seconds) and return the recognized text.

    This base implementation is a stub: it returns an empty string and is meant
    to be subclassed with a concrete model (e.g. AV-HuBERT, VSR transformer).
    Override `read_segment` with your model's inference.
    """

    def read_segment(self, video_path: str | Path, start: float, end: float) -> str:
        raise NotImplementedError(
            "LipReader.read_segment must be implemented by a concrete lip-reading "
            "model. Subclass LipReader or pass a compatible reader to the pipeline."
        )


class NullLipReader(LipReader):
    """A no-op reader for testing/dry-runs.

    Returns a placeholder so the pipeline can run end-to-end without a real
    lip-reading model installed.
    """

    def __init__(self, placeholder: str = "unintelligible"):
        self.placeholder = placeholder

    def read_segment(self, video_path: str | Path, start: float, end: float) -> str:
        return self.placeholder
