"""Meeting-clip transcription pipeline with audio noise detection and lip-reading fallback."""

from .config import PipelineConfig, NoiseDetectorConfig, WhisperConfig
from .pipeline import TranscriptionPipeline
from .types import Segment, Word, FlaggedSpan, TranscriptResult
from .lip_reader import LipReader

__all__ = [
    "PipelineConfig",
    "NoiseDetectorConfig",
    "WhisperConfig",
    "TranscriptionPipeline",
    "Segment",
    "Word",
    "FlaggedSpan",
    "TranscriptResult",
    "LipReader",
]
