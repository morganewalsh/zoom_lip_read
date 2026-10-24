"""End-to-end transcription pipeline.

Steps:
  1. Load audio from the video clip.
  2. Transcribe with faster-whisper (word-level timestamps).
  3. Detect unreliable audio spans (RMS + spectral flatness, 0.5s windows).
  4. For each flagged span, call LipReader.read_segment and replace the
     overlapping Whisper words with the lip-read text.
  5. Assemble ordered segments tagged with source ("audio" | "lip").
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .audio import load_audio
from .config import PipelineConfig
from .lip_reader import LipReader
from .noise_detector import NoiseDetector
from .transcriber import WhisperTranscriber
from .types import FlaggedSpan, Segment, TranscriptResult, Word


class TranscriptionPipeline:
    def __init__(
        self,
        lip_reader: LipReader,
        config: Optional[PipelineConfig] = None,
        transcriber: Optional[WhisperTranscriber] = None,
    ):
        self.config = config or PipelineConfig()
        self.lip_reader = lip_reader
        self.transcriber = transcriber or WhisperTranscriber(self.config.whisper)
        self.detector = NoiseDetector(self.config.noise, self.config.sample_rate)

    def run(self, video_path: str | Path) -> TranscriptResult:
        video_path = str(video_path)

        # 1. Audio
        waveform = load_audio(video_path, self.config.sample_rate)
        clip_duration = len(waveform) / self.config.sample_rate

        # 2. Whisper
        words, language, whisper_duration = self.transcriber.transcribe(video_path)
        duration = max(clip_duration, whisper_duration)

        # 3. Noise detection
        flagged_spans = self.detector.detect(waveform)

        # 4 + 5. Merge whisper words with lip-read spans into ordered segments
        segments = self._assemble_segments(words, flagged_spans, video_path)

        return TranscriptResult(
            segments=segments,
            flagged_spans=flagged_spans,
            language=language or None,
            duration=duration,
        )

    # ------------------------------------------------------------------ #
    def _assemble_segments(
        self,
        words: List[Word],
        flagged_spans: List[FlaggedSpan],
        video_path: str,
    ) -> List[Segment]:
        """Walk the timeline, emitting audio segments between flagged spans and
        lip segments for each flagged span (in time order)."""
        segments: List[Segment] = []

        # A word belongs to a flagged span if it overlaps it. Words are assigned
        # to the first span they overlap; everything else stays "audio".
        def word_overlaps_any(w: Word) -> Optional[FlaggedSpan]:
            mid = (w.start + w.end) / 2.0
            for span in flagged_spans:
                # Use midpoint containment OR real overlap to avoid double-count.
                if span.start <= mid < span.end or span.overlaps(w.start, w.end):
                    return span
            return None

        # Partition words.
        audio_words: List[Word] = []
        span_words: dict[int, List[Word]] = {id(s): [] for s in flagged_spans}
        for w in words:
            span = word_overlaps_any(w)
            if span is None:
                audio_words.append(w)
            else:
                span_words[id(span)].append(w)

        # Build timeline events: audio-word runs + lip spans, sorted by start.
        events: List[tuple[float, str, object]] = []

        for w in audio_words:
            events.append((w.start, "audio", w))

        for span in flagged_spans:
            events.append((span.start, "lip", span))

        events.sort(key=lambda e: (e[0], 0 if e[1] == "lip" else 1))

        # Group consecutive audio words into audio segments; lip spans each
        # become their own segment.
        pending_audio: List[Word] = []

        def flush_audio():
            if not pending_audio:
                return
            seg = Segment(
                source="audio",
                start=pending_audio[0].start,
                end=pending_audio[-1].end,
                text=" ".join(w.text for w in pending_audio).strip(),
                words=list(pending_audio),
            )
            segments.append(seg)
            pending_audio.clear()

        for _, kind, obj in events:
            if kind == "audio":
                pending_audio.append(obj)  # type: ignore[arg-type]
            else:  # lip span
                flush_audio()
                span: FlaggedSpan = obj  # type: ignore[assignment]
                lip_text = (self.lip_reader.read_segment(video_path, span.start, span.end) or "").strip()
                replaced = span_words.get(id(span), [])
                seg = Segment(
                    source="lip",
                    start=span.start,
                    end=span.end,
                    text=lip_text,
                    words=[Word(text=lip_text, start=span.start, end=span.end)] if lip_text else [],
                )
                segments.append(seg)
                # Note: `replaced` whisper words are intentionally dropped in
                # favor of the lip-read text for this span.
                _ = replaced

        flush_audio()

        segments.sort(key=lambda s: s.start)
        return segments
