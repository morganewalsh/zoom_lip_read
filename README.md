# zoom_lip_read

Transcription pipeline for meeting clips. It transcribes audio with
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (word-level
timestamps), detects spans where the audio is unreliable, and falls back to a
lip-reading model over those spans.

## What it does

1. **Transcribe** the clip's audio with faster-whisper, keeping word-level
   timestamps.
2. **Detect unreliable audio** in 0.5s windows using RMS loudness and
   librosa spectral flatness. A window is flagged when it is too quiet
   (dropout) or too noise-like while loud (noise masking speech). Adjacent
   flagged windows are merged into spans.
3. **Lip-read** each flagged span via `LipReader.read_segment(video, start, end)`
   and replace the Whisper words in that span with the lip-read text.
4. **Output** the final transcript as plain text (lip spans wrapped in
   `[lip-read: ...]`) and as JSON, where every segment carries its `source`
   (`"audio"` or `"lip"`) and time range.

## Install

```bash
pip install -r requirements.txt
```

`librosa` needs `ffmpeg` available to decode video containers (`.mp4`, `.mkv`).

## Usage

CLI:

```bash
# Dry run with a stub lip reader (no lip model needed):
python -m zoom_lip_read.cli meeting.mp4 --dry-run \
    --out-text transcript.txt --out-json transcript.json

# Emit a default config file you can edit, then run with it:
python -m zoom_lip_read.cli meeting.mp4 --emit-config config.json
python -m zoom_lip_read.cli meeting.mp4 --config config.json --dry-run
```

Library:

```python
from zoom_lip_read import TranscriptionPipeline, PipelineConfig, LipReader

class MyLipReader(LipReader):
    def read_segment(self, video_path, start, end):
        # run your visual-speech-recognition model on frames [start, end)
        return "recovered text"

config = PipelineConfig.from_json_file("config.json")  # or PipelineConfig()
pipeline = TranscriptionPipeline(lip_reader=MyLipReader(), config=config)

result = pipeline.run("meeting.mp4")
print(result.text)                 # plain text, lip spans wrapped
import json; print(json.dumps(result.to_dict(), indent=2))  # structured JSON
```

## Plugging in a lip-reading model

The pipeline only depends on a `read_segment(video_path, start, end) -> str`
method. Subclass `zoom_lip_read.LipReader` (or pass any object with that
method). `NullLipReader` is provided as a stub for dry runs. To wire a real
model into the CLI, edit `resolve_lip_reader()` in `zoom_lip_read/cli.py`.

## Tuning the detector

All detector thresholds live in `zoom_lip_read/config.py`
(`NoiseDetectorConfig`) and can be overridden via a JSON config file. Key knobs:

| Field | Meaning |
|-------|---------|
| `window_seconds` / `hop_seconds` | Analysis window size / stride (default 0.5s, non-overlapping). |
| `rms_silence_threshold` | Flag windows quieter than this (dropout / too quiet). |
| `flatness_noise_threshold` | Flag windows flatter (more noise-like) than this (0..1). |
| `rms_loud_noise_floor` | A window is flagged as noise only if it is also at least this loud. |
| `merge_gap_seconds` | Merge flagged windows separated by a gap no larger than this. |
| `min_span_seconds` | Drop merged spans shorter than this. |
| `span_pad_seconds` | Pad each merged span on both sides. |

Example `config.json`:

```json
{
  "sample_rate": 16000,
  "noise": {
    "window_seconds": 0.5,
    "rms_silence_threshold": 0.01,
    "flatness_noise_threshold": 0.45,
    "merge_gap_seconds": 0.5,
    "min_span_seconds": 0.5
  },
  "whisper": {
    "model_size": "base",
    "device": "auto",
    "compute_type": "default"
  }
}
```

## Output JSON shape

```json
{
  "language": "en",
  "duration": 10.0,
  "text": "good morning [lip-read: hello] everyone",
  "flagged_spans": [{"start": 2.0, "end": 3.0, "reason": "low-rms(...)"}],
  "segments": [
    {"source": "audio", "start": 0.0, "end": 0.9, "text": "good morning", "words": [...]},
    {"source": "lip",   "start": 2.0, "end": 3.0, "text": "hello", "words": [...]},
    {"source": "audio", "start": 3.2, "end": 3.8, "text": "everyone", "words": [...]}
  ]
}
```

## Tests

Logic tests stub out librosa/whisper so they run without the heavy models:

```bash
python -m tests.test_logic
```
