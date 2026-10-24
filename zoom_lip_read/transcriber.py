"""faster-whisper transcription with word-level timestamps."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .config import WhisperConfig
from .types import Word


class WhisperTranscriber:
    """Thin wrapper around faster-whisper that returns word-level timing."""

    def __init__(self, config: WhisperConfig):
        self.config = config
        self._model = None  # lazily loaded

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.config.model_size,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: str | Path) -> Tuple[List[Word], str, float]:
        """Transcribe `audio_path`.

        Returns (words, detected_language, audio_duration).
        Each Word has start/end timestamps in seconds.
        """
        from .audio import load_audio

        model = self._load_model()

        waveform = load_audio(audio_path, 16000)

        segments, info = model.transcribe(
            waveform,
            word_timestamps=True,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
        )

        words: List[Word] = []
        for seg in segments:  # generator; iterate to realize transcription
            for w in (seg.words or []):
                words.append(
                    Word(
                        text=w.word.strip(),
                        start=float(w.start),
                        end=float(w.end),
                        probability=float(getattr(w, "probability", None) or 0.0),
                    )
                )

        words.sort(key=lambda w: w.start)
        language = getattr(info, "language", None) or (self.config.language or "")
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        return words, language, duration
