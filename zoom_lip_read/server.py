"""FastAPI transcription server (ported from `colab_reference/server.py`).

Importing this module stays light: only stdlib + typing are imported at module
top. FastAPI (and anything heavy) is imported lazily inside `create_app`, and
the real transcription entry (`_default_transcribe`, implemented in a later
task) lazily imports its own heavy dependencies and is only ever *called* from
the non-fake branch. Importing `zoom_lip_read.server` therefore triggers no
heavy imports (R10.1).
"""

# NOTE: deliberately NOT using `from __future__ import annotations`. FastAPI
# reads the route handler's parameter annotations (UploadFile/Form/File) at
# runtime to build its request model; stringized annotations can trip
# forward-ref resolution on some fastapi/pydantic versions. Keeping real
# annotation objects matches the reference server and avoids that pitfall.

import os
import tempfile
import threading
from typing import Callable, Optional

# Module-level lock so at most one non-fake transcription runs at a time (R5.8).
_LOCK = threading.Lock()


def fake_mode_enabled() -> bool:
    """FAKE_MODE is on when `FAKE_MODE` is unset or equals "1" (R5.3)."""
    return os.environ.get("FAKE_MODE", "1") == "1"


def _canned_segments(chunk_start: float) -> list[dict]:
    """The canned two-segment fake response, offset by `chunk_start` (R5.5)."""
    t = chunk_start
    return [
        {"start": t, "end": t + 2.6, "text": "let's meet friday", "source": "audio"},
        {"start": t + 2.6, "end": t + 4.7, "text": "see you then", "source": "lip", "unclear": False},
    ]


def _clip_duration(path) -> float:
    """Clip duration in seconds via ffprobe, 0.0 on any failure (R5.9).

    Mirrors `AutoAvsrLipReader._clip_duration`. `subprocess` is imported lazily
    so importing this module needs neither the `ffprobe` binary nor subprocess
    to be exercised; the function is only ever called from the heavy, non-fake
    path (R10.1).
    """
    import subprocess  # lazy: no ffprobe required to import this module

    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return 0.0


def _to_mp4_25fps(src, dst) -> None:
    """Normalize `src` to a 25 fps mp4 at `dst` via ffmpeg (R5.12).

    Ports Reference_Backend `to_mp4_25fps`. Maps the first video stream and the
    first audio stream if present, re-encoding video with libx264 and audio to
    16 kHz mono AAC. `subprocess` is imported lazily so importing this module
    needs neither the `ffmpeg` binary nor subprocess to be exercised; the
    function is only ever called from the heavy, non-fake path (R10.1).
    """
    import subprocess  # lazy: no ffmpeg required to import this module

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-r",
            "25",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _group_words(words, offset: float) -> list[dict]:
    """Group consecutive Whisper words into audio Segment_Records (R5.9, R6.1).

    Ports Reference_Backend `group_words`. A new audio segment starts whenever
    the gap from the previous word's end to this word's start exceeds 1.0s;
    otherwise the word extends the current segment. Each segment's `start`/`end`
    are offset by `offset` (the request's `chunk_start`) and rounded to 2 dp.

    `words` are the package's `Word` dataclass instances (`.text`/`.start`/
    `.end`); the records produced are `{start, end, text, source: "audio"}`.
    """
    segs: list[dict] = []
    for w in words:
        if segs and w.start - segs[-1]["end"] <= 1.0:
            segs[-1]["text"] += " " + w.text
            segs[-1]["end"] = w.end
        else:
            segs.append(
                {"start": w.start, "end": w.end, "text": w.text, "source": "audio"}
            )
    for s in segs:
        s["start"] = round(s["start"] + offset, 2)
        s["end"] = round(s["end"] + offset, 2)
    return segs


def _in_span(word, spans) -> bool:
    """True when `word`'s time midpoint falls within any span (R7.3).

    Ports Reference_Backend `in_span`. `word` is a `Word` dataclass; `spans` is
    an iterable of `(a, b)` tuples. A word belongs to a span when
    `(word.start + word.end) / 2` lies within `[a, b]`.
    """
    mid = (word.start + word.end) / 2
    return any(a <= mid <= b for a, b in spans)


def _default_transcribe(path, chunk_start: float = 0.0, mode: str = "auto") -> list[dict]:
    """Real transcription entry (backend-style), ported from Reference_Backend
    `transcribe`.

    All heavy machinery (ffmpeg audio loading, faster-whisper, the noise
    detector's librosa path, and the auto_avsr lip reader) is imported lazily
    here. This function is only ever *called* from the non-fake branch of
    `POST /transcribe_chunk`; module-level `create_app()` never calls it, so
    importing `zoom_lip_read.server` triggers no heavy imports (R10.1).

    Modes (R7.1-R7.6):
      - `lip_only` (or no usable audio): one lip segment spanning the whole clip.
      - `audio_only`: audio segments only; no spans, so no lip reading.
      - `auto`: audio transcription, keep words whose midpoint lies outside every
        flagged span, and one lip segment per flagged span. Unreadable audio
        collapses to a single full-length lip span.
    """
    # Lazy imports: deferred so importing the server stays light (R10.1).
    from .audio import load_audio, has_audio
    from .config import PipelineConfig, WhisperConfig
    from .transcriber import WhisperTranscriber
    from .noise_detector import NoiseDetector
    from .avsr_lip_reader import AutoAvsrLipReader

    # The integrated server path mirrors the validated prototype: small.en (R8.2).
    cfg = PipelineConfig(whisper=WhisperConfig(model_size="small.en"))

    # Normalize the upload to a single 25 fps mp4 up front so the audio and lip
    # paths share one consistent timeline (R5.12, R5.13). A normalization
    # failure must not raise out of the request (R5.14): fall back to the raw
    # upload so downstream best-effort paths still run.
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = os.path.join(tmp, "chunk.mp4")
        try:
            _to_mp4_25fps(path, mp4)  # R5.12 ffmpeg 25fps + 16k mono aac
            media = mp4
        except Exception:
            media = path  # R5.14 degrade, do not raise

        total = _clip_duration(media)  # ffprobe on the normalized mp4
        waveform = load_audio(media, 16000)

        if mode == "lip_only" or not has_audio(media):
            # One full-clip lip span, no audio transcription (R7.2, R7.4, R7.5).
            spans = [(0.0, total)]
            words = []
        else:
            words = WhisperTranscriber(cfg.whisper).transcribe(media)[0]
            if mode == "audio_only":
                # Audio segments only; no lip reading (R7.1).
                spans = []
            else:
                # auto: flag unreliable spans; unreadable audio -> whole-clip span.
                flagged = NoiseDetector(cfg.noise, cfg.sample_rate).detect(waveform)
                spans = [(s.start, s.end) for s in flagged]
                if not spans:
                    # No usable/flagged audio: treat the whole chunk as one lip
                    # span (R7.6). An empty waveform yields no flagged spans, so
                    # this also covers the unreadable-in-auto case.
                    spans = [(0.0, total)]

        # Audio records: words whose midpoint lies outside every span (R7.3).
        records = _group_words(
            [w for w in words if not _in_span(w, spans)], chunk_start
        )

        # One lip record per span (R6.2-R6.4, R7.3).
        lip_reader = AutoAvsrLipReader()
        for a, b in spans:
            text = lip_reader.read_segment(media, a, b)  # same normalized timeline (R5.13)
            records.append(
                {
                    "start": round(a + chunk_start, 2),
                    "end": round(b + chunk_start, 2),
                    "text": text,
                    "source": "lip",
                    "unclear": text == "",
                }
            )

    records.sort(key=lambda s: s["start"])  # chronological order (R5.9)
    return records


def create_app(transcribe: Optional[Callable[..., list[dict]]] = None):
    """Build and return the FastAPI application.

    FastAPI is imported here (not at module top) so importing `server` stays
    light. `transcribe` is the injectable non-fake transcription callable; it
    defaults to `_default_transcribe` (R5.1, R5.2, R5.4, R5.5).
    """
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _transcribe = transcribe or _default_transcribe

    @app.get("/health")
    def health():
        return {"status": "ok", "fake_mode": fake_mode_enabled()}

    @app.post("/transcribe_chunk")
    def transcribe_chunk(
        file: UploadFile = File(...),
        session_id: str = Form(...),
        chunk_index: int = Form(...),
        chunk_start: float = Form(0.0),
        mode: str = Form("auto"),
    ):
        if fake_mode_enabled():
            return {
                "chunk_index": chunk_index,
                "segments": _canned_segments(chunk_start),
            }
        # Non-fake request handling. Preserve the upload's extension, defaulting
        # to ".webm" when the client sent no filename/extension (R5.6).
        suffix = os.path.splitext(file.filename or "")[1] or ".webm"
        # Process at most one non-fake transcription at a time (R5.8).
        with _LOCK:
            # The TemporaryDirectory context manager guarantees the temp dir and
            # the uploaded file are removed whether transcription succeeds or
            # raises (R5.7).
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "upload" + suffix)
                with open(path, "wb") as f:
                    f.write(file.file.read())
                segments = _transcribe(path, chunk_start=chunk_start, mode=mode)
        # Return the request's chunk_index alongside the Segment_Records (R5.10).
        return {"chunk_index": chunk_index, "segments": segments}

    return app


# Module-level application instance (R5.1). create_app() must not call the heavy
# transcribe path, so this import stays light.
#
# FastAPI is a declared server dependency, not one of the heavy ML deps that
# R10.1 requires importing to tolerate (torch/cv2/mediapipe/faster_whisper).
# Still, we guard the module-level build so that importing `zoom_lip_read.server`
# succeeds even on a bare machine without fastapi installed: the import-safety
# tests and plain `import zoom_lip_read.server` then pass, and `app` is simply
# None until fastapi is present. Deployments install fastapi and get a real app.
try:
    app = create_app()
except ImportError:  # fastapi (or a transitive dep) not installed
    app = None
