"""Rendering helpers for pipeline results (text + JSON)."""

from __future__ import annotations

import json
from pathlib import Path

from .types import TranscriptResult


def to_text(result: TranscriptResult) -> str:
    """Plain-text transcript with lip-read spans wrapped in [lip-read: ...]."""
    return result.text


def to_json(result: TranscriptResult, indent: int = 2) -> str:
    """Structured JSON: per-segment source + time range, plus flagged spans."""
    return json.dumps(result.to_dict(), indent=indent, ensure_ascii=False)


def to_segment_records(result: TranscriptResult) -> list[dict]:
    """Server-shaped records: exactly {start, end, text, source} per segment,
    plus `unclear` for lip segments. (R6.1-6.4)

    Unlike `Segment.to_dict`, this never leaks the `words` list, keeping the
    wire shape exact. For lip segments, `unclear` is the empty-text
    biconditional: True iff the (stripped) text is empty.
    """
    records: list[dict] = []
    for seg in result.segments:
        rec = {
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
            "source": seg.source,
        }
        if seg.source == "lip":
            rec["unclear"] = seg.text.strip() == ""
        records.append(rec)
    return records


def write_text(result: TranscriptResult, path: str | Path) -> None:
    Path(path).write_text(to_text(result) + "\n", encoding="utf-8")


def write_json(result: TranscriptResult, path: str | Path, indent: int = 2) -> None:
    Path(path).write_text(to_json(result, indent=indent) + "\n", encoding="utf-8")
