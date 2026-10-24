# Implementation Plan: Colab Integration

## Overview

This plan ports the validated Colab prototype (`colab_reference/backend.py`, `colab_reference/server.py`) into the existing `zoom_lip_read` package. It proceeds bottom-up: first the shared ffmpeg audio source and its wiring, then the real lip reader with its extractable pure helpers, then the CLI selection and the server-shaped serializer, then the FastAPI server and its backend-style transcribe entry, and finally the dependency manifest. Testing is interleaved: pure helpers and injectable seams are exercised with property-based tests (hypothesis where available, otherwise table-driven) and the server is driven through FastAPI's `TestClient`, so the whole suite runs on Windows with only `numpy` (+ `fastapi` TestClient, + `hypothesis` if present). Heavy-dependency paths (real model, GPU, mediapipe, torch, faster-whisper, real ffmpeg media) are gated or marked manual because they cannot run in CI.

Every task ends by running the relevant tests plus the existing `tests/test_logic.py` to confirm no regression.

## Tasks

- [x] 1. Replace the audio loader with the ffmpeg decoder and add `has_audio`
  - In `zoom_lip_read/audio.py`, replace the librosa body of `load_audio(path, sample_rate=16000)` with the ported ffmpeg decoder: lazy `import subprocess`; run `["ffmpeg", "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(sample_rate), "-"]` with `capture_output=True`; return `np.zeros(0, dtype=np.float32)` when stdout is empty; otherwise `np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0`
  - Keep the function name, signature, and `np` imported at module top (numpy is always available); only `subprocess` is lazy
  - Add `has_audio(path, sample_rate=16000) -> bool` returning `len(load_audio(path, sample_rate)) > sample_rate * 0.2`
  - Do NOT touch `NoiseDetector`, `NoiseDetectorConfig`, or the pipeline's `load_audio(video_path, self.config.sample_rate)` call — the detector source changes for free
  - _Requirements: 3.1, 3.4, 3.5, 4.1, 4.2, 4.3, 7.5_

  - [x] 1.1 Write property/unit tests for the ffmpeg loader and `has_audio`
    - **Property 13: int16->float32 decoding normalizes samples into [-1.0, 1.0)** — monkeypatch `subprocess.run` to return crafted little-endian int16 bytes; assert elementwise `samples / 32768.0` and range `[-1.0, 1.0)`
    - **Property 11: audio usable iff decoded waveform exceeds 0.2 seconds** — monkeypatch `load_audio` to return length-`n` arrays; assert `has_audio` is `True` exactly when `n > 16000 * 0.2`
    - Edge case: empty `subprocess` stdout yields a length-0 float32 array (R3.4)
    - Use hypothesis if importable, else table-driven; gate any real-ffmpeg assertion behind ffmpeg presence
    - **Validates: Requirements 3.1, 3.4, 7.5**

- [x] 2. Wire `WhisperTranscriber` to pass a numpy waveform to faster-whisper
  - In `zoom_lip_read/transcriber.py`, change `transcribe` to `from .audio import load_audio`, build `waveform = load_audio(audio_path, 16000)`, and pass `waveform` (not `str(audio_path)`) as the first arg to `model.transcribe(...)`, keeping `word_timestamps=True`, `language`, `beam_size`, `vad_filter`
  - Preserve the rest of the method (Word building, sort, language/duration extraction); duration staying correct is handled by the pipeline's `max(clip_duration, whisper_duration)`
  - _Requirements: 3.2_

  - [x] 2.1 Write a unit test that the transcriber passes an ndarray, not a path
    - Inject a fake whisper model (via `transcriber._model`) that records its first positional arg; assert it received an `np.ndarray`
    - Monkeypatch `zoom_lip_read.audio.load_audio` to return a small `np.ndarray` so no ffmpeg/faster-whisper is needed
    - _Requirements: 3.2_

- [x] 3. Checkpoint - audio source swapped, existing tests green
  - Run `python -m tests.test_logic` and the new tests from tasks 1-2. Ensure all tests pass, ask the user if questions arise.

- [x] 4. Create `avsr_lip_reader.py` with extractable pure helpers
  - Create `zoom_lip_read/avsr_lip_reader.py` importing ONLY stdlib + typing + `from .lip_reader import LipReader` at module top (no torch/cv2/mediapipe/auto_avsr) (R1.16)
  - Define module constants `TARGET_FPS = 25`, `MAX_SIDE = 960`, `PAD_TO_SEC = 2.0`
  - Implement pure helper `_env_path(name, default) -> str`: returns `os.environ[name]` when set and non-empty, else `default` (R1.13-1.15)
  - Implement pure helper `_scaled_size(h, w) -> tuple[int, int]`: when `max(h, w) <= 960` return `(h, w)`; else `scale = MAX_SIDE / max(h, w)` and return the rounded `(int(h*scale), int(w*scale))` preserving aspect ratio (R1.4)
  - Implement pure helper `_pad(start, end, total) -> tuple[float, float]`: when `end - start < PAD_TO_SEC`, `mid = (start+end)/2`, return `(max(0.0, mid - 1.0), min(total, mid + 1.0))`; else return `(start, end)` (R1.10)
  - _Requirements: 1.4, 1.10, 1.13, 1.14, 1.15, 1.16_

  - [x] 4.1 Write property tests for `_scaled_size`, `_pad`, `_env_path`
    - **Property 1: Frame downscaling caps the larger side at 960 and preserves aspect ratio** — for `1 <= h,w <= 4000`: unchanged when `max <= 960`, else `max(_scaled_size) == 960` within one pixel of rounding
    - **Property 12: Short-segment padding is centered on the midpoint and clamped to the clip** — for `0 <= start <= end <= total`: result within `[0, total]`; the midpoint-centered formula when `end-start < 2.0`, else unchanged
    - **Property 3: Environment path resolution falls back to the default** — unset/empty -> default, non-empty -> value, for each of `AUTO_AVSR_REPO`/`AVSR_CKPT`/`FACE_TFLITE`
    - Use hypothesis if importable, else table-driven; monkeypatch `os.environ` entries and restore
    - **Validates: Requirements 1.4, 1.10, 1.13, 1.14, 1.15**

- [x] 5. Implement `AutoAvsrLipReader` construction and lazy model loading
  - In `avsr_lip_reader.py`, add `class AutoAvsrLipReader(LipReader)` with `__init__(self, repo=None, ckpt=None, face_tflite=None)` resolving each path via `_env_path` with the Colab defaults; initialize `_model/_video_process/_video_transform/_landmarks_detector/_device = None` (no heavy work in `__init__`) (R1.1, R1.13-1.15)
  - Add `_ensure_model(self)` performing lazy, heavy construction (return early if `_model` set): insert `self.repo` into `sys.path`; `import torch`; `from lightning import ModelModule`; `from datamodule.transforms import VideoTransform`; `from preparation.detectors.mediapipe.video_process import VideoProcess`; `import mediapipe as mp` and select `LandmarksDetector` (solutions API) vs `_TasksLandmarksDetector(self.face_tflite)` via `hasattr(mp, "solutions")`; load checkpoint with `torch.load(..., map_location=lambda s,l: s)`; build `ModelModule(argparse.Namespace(modality="video"))`, `load_state_dict`, `eval`; default `_device = cpu`; if `torch.cuda.is_available()` try `.to("cuda")` and set cuda device, catching exceptions to stay on CPU without raising (R1.6, R1.7, R1.8, R1.9, R1.16)
  - Add `_TasksLandmarksDetector` ported from `colab_reference/backend.py` `TasksLandmarksDetector` (lazy mediapipe tasks imports, `min_detection_confidence=0.5`, largest-bbox-area detection per frame, first 4 keypoints scaled to pixels, `None` when no face, assert at least one face) (R1.5)
  - _Requirements: 1.1, 1.5, 1.6, 1.7, 1.8, 1.9, 1.13, 1.14, 1.15, 1.16_

- [x] 6. Implement `AutoAvsrLipReader` frame reading, decode, and extraction helpers
  - Add `_read_frames(path)` ported from backend `read_frames`: lazy `import cv2`, open with `cv2.VideoCapture`, read `src_fps`, resample to 25fps via the next-timestamp accumulator (`next_t += 1/25`, skip `t < next_t`), downscale via `_scaled_size` with `INTER_AREA`, BGR->RGB, stack to numpy, raise if zero frames (R1.3, R1.4)
  - Add `_decode(crops)`: lazy `import torch`; `torch.tensor(crops).permute(0,3,1,2)`, apply `_video_transform`, `with torch.no_grad()` run on `_device`; on CUDA `RuntimeError` move model to CPU and retry; return `(text or "").strip().lower()` (R1.6, R1.8)
  - Add `_crops(seg_mp4)` combining `_read_frames` + `_landmarks_detector` + `_video_process`
  - Add `_clip_duration(path)`: lazy `import subprocess`, run `ffprobe -v error -show_entries format=duration -of csv=p=0 {path}`, return `float(out)` or `0.0` on failure
  - Add `_extract_segment(video, s, e)`: lazy `import subprocess`; inside a per-call `tempfile.TemporaryDirectory()`, run `ffmpeg -y -ss {s} -to {e} -i {video} -r 25 -an -c:v libx264 -preset veryfast {seg}`; return the segment path (managed by the caller's temp dir)
  - _Requirements: 1.3, 1.4, 1.6, 1.8_

- [x] 7. Implement `AutoAvsrLipReader.read_segment` (3-arg contract, never raises)
  - Add `read_segment(self, video_path, start, end) -> str`: wrap the whole body in `try/except Exception: return ""`; inside create a `tempfile.TemporaryDirectory()`, call `_ensure_model()`, compute `total = _clip_duration(video_path)`, `s, e = _pad(float(start), float(end), total)`, `seg = _extract_segment(...)`, `crops = _crops(seg)`, `return _decode(crops)`
  - Confirm arity matches the base `LipReader.read_segment(video_path, start, end)` so it is a drop-in for `TranscriptionPipeline` (R1.2)
  - _Requirements: 1.2, 1.8, 1.10, 1.11, 1.12_

  - [x] 7.1 Write a property test that `read_segment` never raises
    - **Property 2: read_segment never raises and returns "" on any internal failure** — subclass/monkeypatch so each internal stage (`_ensure_model`, `_read_frames`, landmark detection, `_decode`) raises an injected exception; assert `read_segment` returns `""` and propagates nothing; no torch needed
    - **Validates: Requirements 1.8, 1.11, 1.12**

- [x] 8. Update CLI lip-reader selection
  - In `zoom_lip_read/cli.py`, rewrite `resolve_lip_reader(dry_run)`: return `NullLipReader()` when `dry_run`; else resolve `repo = os.environ.get("AUTO_AVSR_REPO") or "/content/auto_avsr"`; when `os.path.isdir(repo)` lazily `from .avsr_lip_reader import AutoAvsrLipReader` and return it; otherwise return `NullLipReader()`
  - Remove the old `raise SystemExit(...)` so dry runs and repo-absent hosts both fall back gracefully; keep `AutoAvsrLipReader` imported lazily inside the function
  - _Requirements: 2.1, 2.2, 2.3_

  - [x] 8.1 Write a property test for CLI selection truth table
    - **Property 4: CLI lip-reader selection follows the dry-run/repo truth table** — for `dry_run x repo_exists`: `NullLipReader` when dry_run; else `AutoAvsrLipReader` iff the repo dir exists, `NullLipReader` otherwise
    - Monkeypatch `os.path.isdir` and `AUTO_AVSR_REPO`; assert returned instance types (no heavy construction happens in `__init__`)
    - **Validates: Requirements 2.1, 2.2, 2.3**

- [x] 9. Add the server-shaped serializer in `output.py`
  - In `zoom_lip_read/output.py`, add `to_segment_records(result) -> list[dict]`: for each segment emit exactly `{start, end, text, source}`; for `source == "lip"` add `unclear = (seg.text.strip() == "")`
  - Do NOT change `to_text`/`to_json`, the `[lip-read: ...]` wrapping, or `Segment.to_dict` (which keeps `words` for the CLI's rich JSON) (R6.5)
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [x] 9.1 Write a property test for exact key sets and the unclear biconditional
    - **Property 5: Segment_Record key sets are exact and unclear is the empty-text biconditional** — audio record key set is exactly `{start, end, text, source}` with `source=="audio"`; lip record is exactly `{start, end, text, source, unclear}` with `unclear == (text == "")`
    - Build `TranscriptResult`s with generated audio/lip `Segment`s; assert key sets and the biconditional
    - **Validates: Requirements 6.1, 6.2, 6.3, 6.4, 5.11**

- [x] 10. Checkpoint - lip reader, CLI, serializer complete
  - Run `python -m tests.test_logic` and the new tests from tasks 1-9. Ensure all tests pass, ask the user if questions arise.

- [x] 11. Implement the FastAPI server shell and FAKE_MODE path
  - Create `zoom_lip_read/server.py` importing only stdlib + typing at module top (FastAPI imported inside `create_app`) so importing `server` stays light (R10.1)
  - Add `fake_mode_enabled() -> bool` returning `os.environ.get("FAKE_MODE", "1") == "1"` (R5.3)
  - Add `_canned_segments(chunk_start) -> list[dict]`: exactly two segments offset by `chunk_start` — audio `"let's meet friday"` `[t, t+2.6]`, lip `"see you then"` `[t+2.6, t+4.7]` with `unclear=False` (R5.5)
  - Add `create_app(transcribe=None)` factory: build `FastAPI`, add open CORS (all origins/methods/headers), create a module-level `threading.Lock`, set `_transcribe = transcribe or _default_transcribe`; define `GET /health` returning `{"status": "ok", "fake_mode": fake_mode_enabled()}` (R5.2); define `POST /transcribe_chunk` with form fields `file`, `session_id`, `chunk_index`, `chunk_start=0.0`, `mode="auto"` (R5.4) that, in fake mode, returns `{"chunk_index": chunk_index, "segments": _canned_segments(chunk_start)}` (R5.5)
  - Add module-level `app = create_app()` (R5.1)
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 10.1_

  - [x] 11.1 Write the FAKE_MODE server test and /health property test
    - FAKE_MODE test (R10.3): with `TestClient` and FAKE_MODE on, assert `GET /health` -> `status=="ok"`, `fake_mode==true`; assert `POST /transcribe_chunk` returns the canned two-segment response offset by `chunk_start`; no model loaded
    - **Property 7: /health reports status "ok" and FAKE_MODE mirrors the environment** — flip `FAKE_MODE` env values; assert `status=="ok"` and `fake_mode` true iff unset or `"1"`
    - **Property 6 (fake half): Chunk-start offset is additive** — for `c >= 0`, both canned segment `start`/`end` equal base + `c` within 2-dp rounding
    - Skip these tests when `fastapi`'s `TestClient` is unavailable
    - **Validates: Requirements 5.2, 5.3, 5.5, 5.9**

- [x] 12. Implement the non-fake request handling in `/transcribe_chunk`
  - In `create_app`'s `POST /transcribe_chunk`, add the non-fake branch: compute `suffix = os.path.splitext(file.filename or "")[1] or ".webm"` (R5.6); acquire the `threading.Lock` (R5.8); inside a `tempfile.TemporaryDirectory()` write the upload to `upload{suffix}`, call `_transcribe(path, chunk_start=chunk_start, mode=mode)`, and return `{"chunk_index": chunk_index, "segments": segments}` (R5.10); the context manager guarantees temp cleanup on success or exception (R5.7)
  - _Requirements: 5.6, 5.7, 5.8, 5.10_

  - [x] 12.1 Write non-fake envelope and cleanup tests via injected transcribe
    - Non-fake injected test (R10.4): `create_app(transcribe=fake)` with `FAKE_MODE=0`, where `fake` is a `NullLipReader`-backed/direct callable returning `Segment_Record`s; post a small dummy file; assert echoed `chunk_index`, the `segments` list, preserved extension, and that no heavy module (torch/cv2/mediapipe/faster_whisper) was imported
    - Edge case (R5.7): temp dir/upload removed after the request both when the injected transcribe returns and when it raises
    - Form-field defaults applied when omitted (R5.4, R5.10)
    - Skip when `TestClient` unavailable
    - **Validates: Requirements 5.4, 5.6, 5.7, 5.10**

- [x] 13. Implement `_default_transcribe` and `_clip_duration` backend-style entry
  - In `server.py`, add `_clip_duration(path)` (ffprobe helper mirroring `_clip_duration` in the lip reader, lazy `import subprocess`, `0.0` on failure)
  - Add `_group_words(words, offset)` porting backend `group_words`: start a new audio segment when the gap to the previous word exceeds `1.0s`, else extend; round `start`/`end` to 2 dp after adding `offset` (R5.9, R6.1)
  - Add `_in_span(word, spans)` porting backend `in_span`: a word belongs to a span when `(start+end)/2` lies within any `[a, b]` (R7.3)
  - Add `_default_transcribe(path, chunk_start=0.0, mode="auto") -> list[dict]`: lazy-import `load_audio`/`has_audio`, `PipelineConfig`/`WhisperConfig`, `WhisperTranscriber`, `NoiseDetector`, `AutoAvsrLipReader`; `total = _clip_duration(path)`; `waveform = load_audio(path, 16000)`; `cfg = PipelineConfig(whisper=WhisperConfig(model_size="small.en"))` (R8.2); branch by mode — `lip_only` or no `has_audio` -> `spans=[(0.0, total)], words=[]` (R7.2, R7.4, R7.5); else `words = WhisperTranscriber(cfg.whisper).transcribe(path)[0]`, `audio_only` -> `spans=[]` (R7.1), `auto` -> flagged spans from `NoiseDetector(...).detect(waveform)` with the no-usable-audio fallback to `[(0.0, total)]` (R7.3, R7.6); build audio records via `_group_words([w not in any span], chunk_start)`, append one lip record per span (`unclear = text == ""`, offset+rounded) (R6.2-6.4, R7.3), `sort` by `start` (R5.9); return records
  - Ensure `_default_transcribe` is only *called* from the non-fake branch so importing `server` triggers no heavy imports (R10.1)
  - _Requirements: 5.6, 5.9, 5.11, 6.1, 6.2, 6.3, 6.4, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.2_

  - [x] 13.1 Write mode, partition, and offset property tests with injected fakes
    - Drive `_default_transcribe` with a fake whisper (returns generated `Word`s), a `NullLipReader`-style lip reader, and fake `load_audio`/`_clip_duration` so the full mode logic runs in-memory
    - **Property 8: Transcription modes are mutually exclusive in which stage they run** — `audio_only` produces no `lip` segment and never invokes the lip reader; `lip_only` never invokes the transcriber and yields exactly one lip record `[chunk_start, round(duration+chunk_start, 2)]`
    - **Property 9: auto mode partitions words by span midpoint and emits one lip segment per span** — a word is audio content iff its midpoint lies outside every span; lip-segment count equals span count
    - **Property 10: Absent/unreadable audio collapses the chunk to a single full-length lip span** — single lip record `[chunk_start, round(d+chunk_start, 2)]`
    - **Property 6 (non-fake half): chunk-start offset additive** across every returned segment within 2-dp rounding
    - **Validates: Requirements 5.9, 7.1, 7.2, 7.3, 7.4, 7.6**

- [x] 14. Update `requirements.txt`
  - Add `fastapi`, `uvicorn[standard]`, `python-multipart` (R9.1); add `pytorch-lightning` and do NOT add a bare `lightning` line (R9.2); add unpinned `av` and `mediapipe` (R9.3); keep existing pinned entries
  - _Requirements: 9.1, 9.2, 9.3_

  - [x] 14.1 Write a manifest test
    - Assert `requirements.txt` contains `fastapi`, `uvicorn[standard]`, `python-multipart`, `pytorch-lightning`, has no standalone `lightning` line, and lists unpinned `av` and `mediapipe`
    - Also assert `WhisperConfig().model_size == "base"` (R8.1) via a direct construction check
    - **Validates: Requirements 8.1, 9.1, 9.2, 9.3**

- [x] 15. Add import-safety tests for heavy-dependency isolation
  - Add a test that installs a `sys.meta_path` blocker raising `ImportError` for `torch`, `cv2`, `mediapipe`, `faster_whisper`, then imports `zoom_lip_read`, `zoom_lip_read.server`, and `zoom_lip_read.avsr_lip_reader` successfully and asserts those heavy modules are absent from `sys.modules` afterward
  - _Requirements: 1.16, 3.5, 10.1_

- [x] 16. Add gated host-only integration tests (manual / CI-skipped)
  - [x] 16.1 Add host-only smoke tests gated on the `auto_avsr` repo + GPU + ffmpeg media
    - Mark with a skip guard (skip unless `AUTO_AVSR_REPO` dir exists and torch/cv2/mediapipe import): real `_read_frames` frame-count check (R1.3), mediapipe largest-bbox selection (R1.5), real model decode to expected text (R1.6), GPU vs CPU selection (R1.7, R1.9), real ffmpeg decode of a known clip (R3.1 decode half)
    - These cannot run in CI; they document the stubbed paths
    - **Validates: Requirements 1.3, 1.5, 1.6, 1.7, 1.9, 3.1**

- [x] 17. Final checkpoint - full suite green on Windows
  - Run `python -m tests.test_logic` plus all new tests (Windows, numpy + fastapi TestClient + hypothesis if present). Ensure all tests pass and all heavy-dep/host-only tests skip cleanly; ask the user if questions arise.

- [x] 18. Add 25fps MP4 normalization to the server's non-fake path
  - In `zoom_lip_read/server.py`, add `_to_mp4_25fps(src, dst)` (lazy `import subprocess`) ported from Reference_Backend `to_mp4_25fps`: run `ffmpeg -y -i {src} -map 0:v:0 -map 0:a:0? -r 25 -c:v libx264 -preset veryfast -c:a aac -ar 16000 -ac 1 {dst}` with `check=True`
  - Change `_default_transcribe` to open a `tempfile.TemporaryDirectory()`, convert the input to `chunk.mp4` via `_to_mp4_25fps`, and run `_clip_duration`, `load_audio`, `has_audio`, and every `read_segment` call against that mp4 so the audio and lip paths share one timeline (R5.12, R5.13)
  - Wrap the normalization call so a failure falls back to the raw upload path and never raises out of the request (R5.14)
  - Keep all heavy/binary work lazy so importing `server` stays light (R10.1); run `python -m tests.test_server_modes`, `tests.test_server_fake`, `tests.test_server_nonfake`, `tests.test_import_safety`, and `python -m tests.test_logic`
  - _Requirements: 5.12, 5.13, 5.14_

  - [x] 18.1 Test normalization-before-transcription with injected fakes
    - **Property 14: Normalization precedes transcription on a single timeline** — inject a fake `_to_mp4_25fps` that records its args and creates a stub file, plus fakes for `_clip_duration`/`load_audio`/`has_audio`/`AutoAvsrLipReader` on their source modules; assert the normalizer runs exactly once and every downstream stage receives the normalized `chunk.mp4` path, not the raw upload
    - Failure case (R5.14): when the injected `_to_mp4_25fps` raises, assert downstream stages receive the raw upload path and `_default_transcribe` returns without raising
    - Use hypothesis if importable, else table-driven; skip-safe with only numpy; no heavy module imported
    - **Validates: Requirements 5.12, 5.13, 5.14**

## Notes

- Tasks marked with `*` are optional (test sub-tasks and host-only smoke tests) and can be skipped for a faster MVP, but the core wiring tasks must be implemented.
- `tests/test_logic.py` must remain unchanged and keep passing throughout (R10.2, R4.1-4.3).
- Property-based tests use `hypothesis` when importable, falling back to table-driven cases otherwise, so the suite runs with only numpy + fastapi `TestClient`.
- Heavy imports (torch, cv2, mediapipe, auto_avsr, faster_whisper) and the `ffmpeg`/`ffprobe` binaries are only touched inside methods exercised on the GPU host, never at module import time (R1.16, R3.5, R10.1).
- Each property test should carry a comment `# Feature: colab-integration, Property {n}: {property_text}` referencing its design property.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1", "4"] },
    { "id": 1, "tasks": ["1.1", "2", "4.1", "5", "9"] },
    { "id": 2, "tasks": ["2.1", "6", "8", "9.1"] },
    { "id": 3, "tasks": ["7", "8.1", "11", "14"] },
    { "id": 4, "tasks": ["7.1", "11.1", "12", "14.1", "15"] },
    { "id": 5, "tasks": ["12.1", "13"] },
    { "id": 6, "tasks": ["13.1", "16.1"] },
    { "id": 7, "tasks": ["18", "18.1"] }
  ]
}
```
