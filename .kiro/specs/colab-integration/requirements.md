# Requirements Document

## Introduction

This feature integrates the proven Colab prototype code (`colab_reference/backend.py` and `colab_reference/server.py`) into the existing `zoom_lip_read` Python package. The Colab code provides two capabilities the package currently lacks: a real visual-speech-recognition (VSR) lip reader built on the `auto_avsr` repository, and a FastAPI HTTP service for chunked transcription requests.

The integration preserves the existing package structure, data types, configuration system, and all current tests. New capabilities are added as new modules (`avsr_lip_reader.py`, `server.py`), the ffmpeg-based audio loader is adopted as the single audio source, and the CLI is updated to select the real lip reader when the `auto_avsr` repository is available. The existing windowed noise detector (`NoiseDetector`) is retained; only its audio source changes.

Because the reference code targets a Colab/Linux GPU environment (`/content/...` paths, ffmpeg/ffprobe on PATH, mediapipe, torch, faster-whisper, GPU), all new behavior must be structured so the package still imports and tests still run on a plain Windows machine without those heavy dependencies, the real model, GPU, or real media.

## Glossary

- **Package**: The existing `zoom_lip_read` Python package.
- **Reference_Backend**: The prototype module `colab_reference/backend.py`.
- **Reference_Server**: The prototype module `colab_reference/server.py`.
- **AutoAvsrLipReader**: The new concrete `LipReader` subclass that performs real VSR using the `auto_avsr` model. Lives in `zoom_lip_read/avsr_lip_reader.py`.
- **LipReader**: The existing abstract base class in `zoom_lip_read/lip_reader.py` whose only contract the pipeline depends on is `read_segment(video_path, start, end) -> str`.
- **NullLipReader**: The existing stub `LipReader` subclass that returns a fixed placeholder without a real model.
- **NoiseDetector**: The existing windowed audio-reliability detector in `zoom_lip_read/noise_detector.py` (RMS threshold + spectral flatness over non-overlapping windows).
- **WhisperTranscriber**: The existing faster-whisper wrapper in `zoom_lip_read/transcriber.py`.
- **TranscriptionPipeline**: The existing end-to-end pipeline class in `zoom_lip_read/pipeline.py`.
- **Ffmpeg_Audio_Loader**: A new function that decodes any video/audio file to a 16 kHz mono float32 numpy array by invoking the `ffmpeg` binary (ported from Reference_Backend's `load_audio`).
- **Transcription_Server**: The new FastAPI application in `zoom_lip_read/server.py`.
- **FAKE_MODE**: Environment-variable-controlled server mode that returns a canned example response without loading any model.
- **Transcription_Mode**: The per-request mode selecting which internal path runs: `auto`, `lip_only`, or `audio_only`.
- **Mp4_Normalizer**: A function that converts an arbitrary uploaded media file to a 25 fps MP4 (video libx264, audio 16 kHz mono AAC), ported from Reference_Backend `to_mp4_25fps`.
- **Segment_Record**: The JSON object representing one output segment: `{"start", "end", "text", "source"}` plus `"unclear"` for lip segments.
- **Flagged_Span**: A contiguous time range (seconds) where audio speech is considered unreliable, produced by NoiseDetector.
- **AUTO_AVSR_REPO**: Environment variable for the `auto_avsr` repository path. Default `/content/auto_avsr`.
- **AVSR_CKPT**: Environment variable for the VSR model checkpoint path. Default `/content/auto_avsr/tutorials/vsr_trlrs2lrs3vox2avsp_base.pth`.
- **FACE_TFLITE**: Environment variable for the face-detector TFLite model path. Default `/content/auto_avsr/tutorials/blaze_face_short_range.tflite`.

## Requirements

### Requirement 1: Real lip reader module

**User Story:** As a developer deploying on a GPU host, I want a real VSR lip reader that drops into the existing pipeline, so that unreliable audio spans are transcribed from lip movements instead of a placeholder.

#### Acceptance Criteria

1. THE Package SHALL provide a module `zoom_lip_read/avsr_lip_reader.py` that defines a class `AutoAvsrLipReader` subclassing `LipReader`.
2. THE AutoAvsrLipReader SHALL implement a method with signature `read_segment(video_path, start, end) -> str` matching the existing LipReader contract that TranscriptionPipeline depends on.
3. WHEN reading frames from a segment, THE AutoAvsrLipReader SHALL resample to a target of 25 frames per second using the same next-timestamp sampling as Reference_Backend `read_frames`.
4. WHEN a source frame's larger dimension exceeds 960 pixels, THE AutoAvsrLipReader SHALL downscale that frame so its larger dimension equals 960 pixels while preserving aspect ratio, and SHALL leave frames whose larger dimension is 960 pixels or fewer unscaled.
5. THE AutoAvsrLipReader SHALL detect face landmarks using the same detection behavior as Reference_Backend `TasksLandmarksDetector`, selecting the detection with the largest bounding-box area per frame.
6. THE AutoAvsrLipReader SHALL construct, load, and invoke the model using the same model-loading and decoding logic as Reference_Backend `LipReader`, and SHALL return decoded text stripped of surrounding whitespace and lowercased.
7. WHERE a CUDA device is available, THE AutoAvsrLipReader SHALL attempt to load and run the model on the GPU.
8. IF moving the model to the GPU or decoding on the GPU raises an error, THEN THE AutoAvsrLipReader SHALL run the model on the CPU and continue without raising.
9. IF a CUDA device is not available, THEN THE AutoAvsrLipReader SHALL load and run the model on the CPU.
10. WHERE the requested segment duration (end minus start) is less than 2.0 seconds, THE AutoAvsrLipReader SHALL set the segment to start = max(0.0, midpoint - 1.0) and end = min(clip_duration, midpoint + 1.0), where midpoint = (start + end) / 2.
11. IF reading, landmark detection, or decoding a segment raises any exception, THEN THE AutoAvsrLipReader SHALL return an empty string.
12. THE AutoAvsrLipReader SHALL NOT propagate exceptions out of `read_segment` for any input.
13. WHEN constructing an instance, THE AutoAvsrLipReader SHALL read the repository path from the `AUTO_AVSR_REPO` environment variable, defaulting to `/content/auto_avsr` when the variable is unset or empty.
14. WHEN constructing an instance, THE AutoAvsrLipReader SHALL read the checkpoint path from the `AVSR_CKPT` environment variable, defaulting to `/content/auto_avsr/tutorials/vsr_trlrs2lrs3vox2avsp_base.pth` when the variable is unset or empty.
15. WHEN constructing an instance, THE AutoAvsrLipReader SHALL read the face-detector path from the `FACE_TFLITE` environment variable, defaulting to `/content/auto_avsr/tutorials/blaze_face_short_range.tflite` when the variable is unset or empty.
16. WHEN the module `zoom_lip_read/avsr_lip_reader.py` is imported, THE AutoAvsrLipReader SHALL NOT import torch, cv2, mediapipe, or auto_avsr modules, deferring those imports until an instance is constructed or a segment is read, so that the module imports successfully when those packages are absent.

### Requirement 2: CLI lip reader selection

**User Story:** As a user running the CLI, I want the tool to automatically use the real lip reader when the model is present and fall back to the stub otherwise, so that I can run dry runs anywhere and real transcription on the GPU host.

#### Acceptance Criteria

1. WHERE the directory named by AUTO_AVSR_REPO exists, THE Package SHALL select AutoAvsrLipReader in `resolve_lip_reader`.
2. WHERE the directory named by AUTO_AVSR_REPO does not exist, THE Package SHALL select NullLipReader in `resolve_lip_reader`.
3. WHEN the `--dry-run` flag is provided, THE Package SHALL select NullLipReader regardless of AUTO_AVSR_REPO.

### Requirement 3: Ffmpeg-based audio loading

**User Story:** As a developer, I want all audio decoding to go through an ffmpeg helper that returns a numpy waveform, so that newer PyAV releases do not break faster-whisper's and librosa's own file loaders.

#### Acceptance Criteria

1. THE Package SHALL provide an Ffmpeg_Audio_Loader function, ported from Reference_Backend `load_audio`, that decodes a file path to a mono float32 numpy array sampled at 16 kHz, with sample values normalized to the range [-1.0, 1.0), using the `ffmpeg` binary.
2. WHEN WhisperTranscriber transcribes a clip, THE WhisperTranscriber SHALL pass a numpy waveform produced by Ffmpeg_Audio_Loader to faster-whisper instead of a file path.
3. WHEN NoiseDetector needs the waveform for a clip, THE Package SHALL supply a waveform produced by Ffmpeg_Audio_Loader instead of calling `librosa.load` on a file path.
4. IF the `ffmpeg` invocation produces no audio sample bytes for the input file, THEN THE Ffmpeg_Audio_Loader SHALL return an empty numpy array.
5. THE Ffmpeg_Audio_Loader SHALL import its ffmpeg invocation dependencies lazily so that importing the Package does not require the `ffmpeg` binary to be present.

### Requirement 4: Noise detector reconciliation (decision point)

**User Story:** As a maintainer, I want a clear, single rule for how the existing noise detector relates to the reference backend's detector, so that behavior and tests stay predictable.

#### Acceptance Criteria

1. THE Package SHALL retain the existing NoiseDetector class, NoiseDetectorConfig dataclass, and its RMS-threshold-plus-spectral-flatness windowing behavior as the detector used by TranscriptionPipeline.
2. THE Package SHALL change only the NoiseDetector audio source to the waveform produced by Ffmpeg_Audio_Loader, and SHALL NOT replace the existing detector with Reference_Backend's `detect_noisy_spans` dB/flatness approach.
3. THE Package SHALL keep the existing NoiseDetector public API and span-merging behavior so that `tests/test_logic.py` continues to pass unchanged.

> Decision point resolved: the existing detector is kept (audio source swapped only). "Match backend exactly" applies to the lip reader and the server contract, not to the detector. The reference backend's dB-based flatness detector is intentionally not adopted.

### Requirement 5: Transcription HTTP server

**User Story:** As a client application, I want an HTTP endpoint that accepts a media chunk and returns timed transcript segments, so that I can transcribe meeting audio chunk by chunk.

#### Acceptance Criteria

1. THE Package SHALL provide a module `zoom_lip_read/server.py` exposing a FastAPI application instance as Transcription_Server.
2. THE Transcription_Server SHALL expose `GET /health` returning a JSON object whose status field equals `"ok"` and whose `fake_mode` field equals the current boolean FAKE_MODE state.
3. THE Transcription_Server SHALL treat FAKE_MODE as enabled when the `FAKE_MODE` environment variable is unset or equals `"1"`, and as disabled for any other value.
4. THE Transcription_Server SHALL expose `POST /transcribe_chunk` accepting multipart form fields `file`, `session_id`, `chunk_index`, `chunk_start` (defaulting to 0.0), and `mode` (defaulting to `auto`).
5. WHERE FAKE_MODE is enabled, THE Transcription_Server SHALL return the canned example response without loading any model, containing exactly two segments offset by `chunk_start`: an audio segment and a lip segment.
6. WHERE FAKE_MODE is disabled, THE Transcription_Server SHALL write the uploaded file into a temporary directory preserving its file extension (defaulting to `.webm` when absent), run transcription with the request's `chunk_start` and `mode`, and return the resulting segments.
7. WHEN a `POST /transcribe_chunk` request completes, whether it succeeds or fails, THE Transcription_Server SHALL delete the temporary directory and its uploaded file.
8. THE Transcription_Server SHALL process at most one non-fake transcription request at a time using a lock.
9. THE Transcription_Server SHALL offset every returned segment's `start` and `end` by the request's `chunk_start`.
10. THE Transcription_Server SHALL return a JSON object containing the request's `chunk_index` and the list of Segment_Records.
11. THE Transcription_Server SHALL produce the same response contract defined in Requirement 6 regardless of whether the internal path uses TranscriptionPipeline or a backend-style function.
12. WHERE FAKE_MODE is disabled, THE Transcription_Server SHALL normalize the uploaded media to a 25 fps MP4 before transcription using Mp4_Normalizer, mapping the first video stream and the first audio stream if present, re-encoding video with libx264 and audio to 16 kHz mono AAC, matching Reference_Backend `to_mp4_25fps`.
13. THE Transcription_Server SHALL run both the audio transcription path and the lip-reading path against the same normalized 25 fps MP4, so that all returned segment timestamps derive from a single consistent timeline.
14. IF the frame-rate normalization step fails for an uploaded chunk, THEN THE Transcription_Server SHALL complete the request without raising, returning whatever segments the downstream paths produce from the available input.

### Requirement 6: Output segment format

**User Story:** As a client application, I want each segment in a stable, documented shape, so that I can render audio and lip-read segments consistently.

#### Acceptance Criteria

1. THE Package SHALL represent each audio segment as a Segment_Record with exactly the keys `start`, `end`, `text`, and `source` set to `"audio"`.
2. THE Package SHALL represent each lip-read segment as a Segment_Record with exactly the keys `start`, `end`, `text`, `source` set to `"lip"`, and `unclear`.
3. WHEN a lip-read segment's text is empty, THE Package SHALL set that segment's `unclear` field to `true`.
4. WHEN a lip-read segment's text is non-empty, THE Package SHALL set that segment's `unclear` field to `false`.
5. THE Package SHALL retain the existing plain-text export that wraps lip-read spans as `[lip-read: ...]`.

### Requirement 7: Transcription modes

**User Story:** As a client application, I want to choose whether a chunk is transcribed from audio, from lips, or automatically, so that I can control behavior per chunk.

#### Acceptance Criteria

1. WHERE Transcription_Mode is `audio_only`, THE Package SHALL produce only audio-sourced segments and SHALL NOT perform lip reading.
2. WHERE Transcription_Mode is `lip_only`, THE Package SHALL produce exactly one lip-sourced segment spanning 0.0 to the chunk duration and SHALL NOT perform audio transcription.
3. WHERE Transcription_Mode is `auto`, THE Package SHALL perform audio transcription, keep as audio segments only those words whose time midpoint falls outside every Flagged_Span, and produce one lip-read segment per Flagged_Span.
4. IF the input chunk contains no usable audio, THEN in `auto` or `audio_only` mode THE Package SHALL treat the whole chunk as a single lip-read span spanning 0.0 to the chunk duration.
5. THE Package SHALL treat audio as usable only when the decoded 16 kHz mono waveform has a duration greater than 0.2 seconds, matching Reference_Backend `has_audio`.
6. IF the audio is unreadable by the noise detector in `auto` mode, THEN THE Package SHALL treat the whole chunk as a single lip-read span spanning 0.0 to the chunk duration.

### Requirement 8: Whisper model size for the integrated path

**User Story:** As a maintainer, I want the integrated server/backend path's Whisper model size documented, so that behavior matches the validated prototype.

#### Acceptance Criteria

1. THE Package SHALL document that the existing WhisperConfig default model size remains `base` for the existing pipeline and CLI paths.
2. WHERE the integrated server path mirrors Reference_Backend, THE Package SHALL use the Whisper model size `small.en` for that path, matching the validated prototype.

### Requirement 9: Dependencies

**User Story:** As a developer setting up the project, I want the dependency list to include the server and model packages with the correct variants, so that installation does not clash with `auto_avsr`.

#### Acceptance Criteria

1. THE Package SHALL add `fastapi`, `uvicorn[standard]`, and `python-multipart` to `requirements.txt`.
2. THE Package SHALL add `pytorch-lightning` to `requirements.txt` and SHALL NOT add the `lightning` package, because `lightning` clashes with `auto_avsr`'s `lightning.py`.
3. THE Package SHALL list `av` and `mediapipe` without pinned versions.

### Requirement 10: Importability and testability without heavy dependencies

**User Story:** As a contributor on a plain Windows machine, I want to import the package and run the test suite without the real model, GPU, ffmpeg media, mediapipe, torch, or faster-whisper, so that development and CI stay fast and portable.

#### Acceptance Criteria

1. THE Package SHALL allow importing `zoom_lip_read`, `zoom_lip_read.server`, and `zoom_lip_read.avsr_lip_reader` without `torch`, `cv2`, `mediapipe`, `faster_whisper`, or the `ffmpeg` binary being installed.
2. THE Package SHALL keep `tests/test_logic.py` passing unchanged.
3. THE Package SHALL add a server test that exercises `GET /health` and `POST /transcribe_chunk` in FAKE_MODE without loading any model.
4. THE Package SHALL add a server test that exercises the non-fake path using a NullLipReader-backed transcription so that no real model, GPU, mediapipe, torch, faster-whisper, or real media is required.
5. THE added tests SHALL NOT execute any code path that requires the real VSR model, GPU, real ffmpeg media, mediapipe, torch, or faster-whisper.
