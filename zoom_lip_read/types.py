"""Shared data types for the transcription pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Literal

Source = Literal["audio", "lip"]


@dataclass
class Word:
    """A single word with its time range, as produced by Whisper or lip-reading."""

    text: str
    start: float
    end: float
    probability: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FlaggedSpan:
    """A contiguous time range where audio speech is considered unreliable.

    Produced by the noise detector after merging adjacent flagged windows.
    """

    start: float
    end: float
    # Diagnostic info: why the span was flagged (per-window metrics).
    reason: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, start: float, end: float) -> bool:
        """True if [start, end) overlaps this span."""
        return start < self.end and end > self.start

    def to_dict(self) -> dict:
        return {"start": self.start, "end": self.end, "reason": self.reason}


@dataclass
class Segment:
    """A contiguous chunk of final transcript from a single source."""

    source: Source
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class TranscriptResult:
    """Final pipeline output: ordered segments plus convenience renderings."""

    segments: List[Segment] = field(default_factory=list)
    flagged_spans: List[FlaggedSpan] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None

    @property
    def text(self) -> str:
        """Plain-text transcript. Lip-read spans are wrapped in [lip-read: ...]."""
        parts: List[str] = []
        for seg in self.segments:
            chunk = seg.text.strip()
            if not chunk:
                continue
            if seg.source == "lip":
                parts.append(f"[lip-read: {chunk}]")
            else:
                parts.append(chunk)
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "duration": self.duration,
            "text": self.text,
            "flagged_spans": [s.to_dict() for s in self.flagged_spans],
            "segments": [s.to_dict() for s in self.segments],
        }
