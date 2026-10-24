"""Real visual-speech-recognition lip reader backed by the ``auto_avsr`` repo.

This module is a drop-in ``LipReader`` for :class:`~zoom_lip_read.pipeline.TranscriptionPipeline`.
It ports the validated Colab prototype (``colab_reference/backend.py``) but is
structured so that importing the module requires only the standard library and
``numpy``-free typing: torch, cv2, mediapipe, and the ``auto_avsr`` modules are
imported lazily, inside the methods that need them, so the package stays
importable on a plain Windows machine without those heavy dependencies (R1.16).

The pure helpers (``_env_path``, ``_scaled_size``, ``_pad``) carry no heavy
dependencies and are unit/property tested in ``tests``.
"""

from __future__ import annotations

import os
from typing import Tuple

from .lip_reader import LipReader

# Resampling / sizing constants (ported from Reference_Backend).
TARGET_FPS = 25
MAX_SIDE = 960
PAD_TO_SEC = 2.0


def _env_path(name: str, default: str) -> str:
    """Return ``os.environ[name]`` when set and non-empty, else ``default``.

    Used to resolve the ``AUTO_AVSR_REPO`` / ``AVSR_CKPT`` / ``FACE_TFLITE``
    paths with the Colab defaults (R1.13-1.15).
    """
    value = os.environ.get(name)
    if value:
        return value
    return default


def _scaled_size(h: int, w: int) -> Tuple[int, int]:
    """Cap the larger side at ``MAX_SIDE`` while preserving aspect ratio (R1.4).

    When ``max(h, w) <= MAX_SIDE`` the frame is left unscaled. Otherwise the
    dimensions are scaled by ``MAX_SIDE / max(h, w)`` and rounded toward zero
    (matching ``int(...)`` truncation in the reference ``read_frames``).
    """
    if max(h, w) <= MAX_SIDE:
        return (h, w)
    scale = MAX_SIDE / max(h, w)
    return (int(h * scale), int(w * scale))


def _pad(start: float, end: float, total: float) -> Tuple[float, float]:
    """Widen short segments to ``PAD_TO_SEC`` centered on their midpoint (R1.10).

    When the segment is shorter than ``PAD_TO_SEC`` seconds, re-center a
    ``PAD_TO_SEC``-wide window on the midpoint and clamp it to ``[0.0, total]``.
    Longer segments are returned unchanged.
    """
    if end - start < PAD_TO_SEC:
        mid = (start + end) / 2
        return (max(0.0, mid - 1.0), min(total, mid + 1.0))
    return (start, end)

# --- Colab default paths (overridable via environment variables). ---
_DEFAULT_REPO = "/content/auto_avsr"
_DEFAULT_CKPT = "/content/auto_avsr/tutorials/vsr_trlrs2lrs3vox2avsp_base.pth"
_DEFAULT_FACE_TFLITE = "/content/auto_avsr/tutorials/blaze_face_short_range.tflite"


class _TasksLandmarksDetector:
    """Face-landmark detector using the mediapipe *tasks* API (R1.5).

    Ported from Reference_Backend ``TasksLandmarksDetector``. Heavy mediapipe
    imports happen inside ``__init__`` so the enclosing module stays importable
    without mediapipe (R1.16).
    """

    def __init__(self, model_path: str = _DEFAULT_FACE_TFLITE):
        import numpy as np  # local to avoid module-level heavy-ish import order issues
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._np = np
        self.mp = mp
        self.detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                min_detection_confidence=0.5,
            )
        )

    def __call__(self, video_frames):
        np = self._np
        landmarks = []
        for frame in video_frames:
            h, w = frame.shape[:2]
            image = self.mp.Image(
                image_format=self.mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(frame),
            )
            result = self.detector.detect(image)
            if not result.detections:
                landmarks.append(None)
                continue
            best = max(
                result.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
            )
            landmarks.append(
                np.array([[int(k.x * w), int(k.y * h)] for k in best.keypoints[:4]])
            )
        assert any(l is not None for l in landmarks), (
            "Cannot detect a face in any frame of the video"
        )
        return landmarks


class AutoAvsrLipReader(LipReader):
    """Real VSR lip reader backed by the ``auto_avsr`` repo.

    Heavy construction (torch, mediapipe, auto_avsr modules) is deferred to
    :meth:`_ensure_model` so the module and class import cleanly on a plain
    machine (R1.16). Path configuration is resolved from the environment with
    the Colab defaults (R1.13-1.15).
    """

    def __init__(self, repo=None, ckpt=None, face_tflite=None):
        self.repo = repo if repo else _env_path("AUTO_AVSR_REPO", _DEFAULT_REPO)
        self.ckpt = ckpt if ckpt else _env_path("AVSR_CKPT", _DEFAULT_CKPT)
        self.face_tflite = (
            face_tflite if face_tflite else _env_path("FACE_TFLITE", _DEFAULT_FACE_TFLITE)
        )
        self._model = None
        self._video_process = None
        self._video_transform = None
        self._landmarks_detector = None
        self._device = None

    def _ensure_model(self):
        """Lazily build and load the VSR model (R1.6, R1.7, R1.8, R1.9, R1.16).

        Returns early if already loaded. Inserts ``self.repo`` onto ``sys.path``
        so the ``auto_avsr`` modules resolve, selects the mediapipe landmark
        detector variant, loads the checkpoint, and chooses CPU/GPU, falling
        back to CPU without raising if the GPU move fails.
        """
        if self._model is not None:
            return

        import sys
        import argparse

        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)

        import torch
        from lightning import ModelModule
        from datamodule.transforms import VideoTransform
        from preparation.detectors.mediapipe.video_process import VideoProcess
        import mediapipe as mp

        if hasattr(mp, "solutions"):
            from preparation.detectors.mediapipe.detector import LandmarksDetector

            self._landmarks_detector = LandmarksDetector()
        else:
            self._landmarks_detector = _TasksLandmarksDetector(self.face_tflite)

        self._video_process = VideoProcess(convert_gray=False)
        self._video_transform = VideoTransform(subset="test")

        ckpt = torch.load(self.ckpt, map_location=lambda storage, loc: storage)
        model = ModelModule(argparse.Namespace(modality="video"))
        model.model.load_state_dict(ckpt)
        model.eval()
        self._model = model

        self._device = torch.device("cpu")
        if torch.cuda.is_available():
            try:
                self._model.to("cuda")
                self._device = torch.device("cuda")
            except Exception as e:  # pragma: no cover - requires GPU
                print("GPU move failed, staying on CPU:", e)

    # ------------------------------------------------------------------ #
    # Frame reading, decode, and extraction helpers (Task 6).
    # ------------------------------------------------------------------ #
    def _read_frames(self, path):
        """Read and resample a clip to 25fps RGB frames (R1.3, R1.4).

        Ported from Reference_Backend ``read_frames``: opens the clip with
        OpenCV, resamples to ``TARGET_FPS`` with a next-timestamp accumulator,
        downscales frames whose larger side exceeds ``MAX_SIDE`` via
        ``_scaled_size`` with ``INTER_AREA``, converts BGR->RGB, and stacks to a
        numpy array. Raises if no frames could be read.
        """
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(path)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        frames, i, next_t = [], 0, 0.0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            t = i / src_fps
            i += 1
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / TARGET_FPS
            h, w = bgr.shape[:2]
            new_h, new_w = _scaled_size(h, w)
            if (new_h, new_w) != (h, w):
                bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"Could not read any frames from {path}")
        return np.stack(frames), src_fps

    def _decode(self, crops):
        """Run the VSR model on preprocessed crops and return text (R1.6, R1.8).

        Builds the input tensor, applies the video transform, runs with grads
        disabled on ``_device``; on a CUDA ``RuntimeError`` it moves the model
        to CPU and retries. Returns decoded text stripped and lowercased.
        """
        import torch

        x = torch.tensor(crops).permute(0, 3, 1, 2)
        x = self._video_transform(x)
        with torch.no_grad():
            try:
                text = self._model(x.to(self._device))
            except RuntimeError as e:
                if self._device.type == "cuda":  # device mismatch -> fall back to CPU
                    print("GPU decode failed, falling back to CPU:", e)
                    self._model.to("cpu")
                    self._device = torch.device("cpu")
                    text = self._model(x)
                else:
                    raise
        return (text or "").strip().lower()

    def _crops(self, seg_mp4):
        """Produce model-ready crops for a segment clip.

        Combines ``_read_frames`` + ``_landmarks_detector`` + ``_video_process``
        (mirrors Reference_Backend ``LipReader.crops``).
        """
        video, _src_fps = self._read_frames(seg_mp4)
        landmarks = self._landmarks_detector(video)
        return self._video_process(video, landmarks)

    def _clip_duration(self, path):
        """Return the clip duration in seconds via ffprobe, or 0.0 on failure."""
        import subprocess

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

    def _extract_segment(self, video, s, e):
        """Cut ``[s, e]`` from ``video`` to a 25fps silent H.264 segment.

        Uses a per-call ``tempfile.TemporaryDirectory`` to host the segment and
        returns its path. The caller is responsible for managing the lifetime of
        the surrounding temp dir; this helper returns the produced segment path.
        """
        import subprocess
        import os
        import tempfile

        tmp = tempfile.TemporaryDirectory()
        seg = os.path.join(tmp.name, f"seg_{float(s):.2f}.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                f"{float(s):.2f}",
                "-to",
                f"{float(e):.2f}",
                "-i",
                str(video),
                "-r",
                "25",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                seg,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Keep the temp dir alive for the lifetime of the returned path by
        # anchoring it on the instance-agnostic path; the caller's temp dir
        # (read_segment, Task 7) manages overall cleanup.
        self._last_segment_tmp = tmp
        return seg

    # ------------------------------------------------------------------ #
    # Public contract (Task 7).
    # ------------------------------------------------------------------ #
    def read_segment(self, video_path, start, end) -> str:
        """Transcribe lip movements in ``[start, end]`` and return text (R1.2).

        This is the single method :class:`~zoom_lip_read.pipeline.TranscriptionPipeline`
        depends on; its 3-argument signature ``(video_path, start, end)`` matches
        the base :meth:`LipReader.read_segment`, so the reader is a drop-in.

        The entire body is wrapped so that reading, landmark detection, decoding,
        or any other internal failure (including the GPU fallbacks inside
        ``_decode``) results in an empty string rather than a propagated
        exception (R1.8, R1.11, R1.12). A per-call ``TemporaryDirectory`` scopes
        any intermediate files to this invocation.
        """
        import tempfile

        try:
            with tempfile.TemporaryDirectory():
                self._ensure_model()
                total = self._clip_duration(video_path)
                s, e = _pad(float(start), float(end), total)
                seg = self._extract_segment(video_path, s, e)
                crops = self._crops(seg)
                return self._decode(crops)
        except Exception as e:
            # Still never raises, but print why so failures are visible in the server log.
            print(f"lip read failed for {start:.2f}-{end:.2f}s: {e!r}")
            return ""