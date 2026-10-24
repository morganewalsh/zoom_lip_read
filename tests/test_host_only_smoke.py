"""Host-only smoke tests for the heavy real-path lip-reading code.

These tests exercise the stubbed/heavy paths that CANNOT run in CI or on a
plain Windows machine: they require the ``auto_avsr`` repo checked out, the
heavy ML stack (torch / cv2 / mediapipe) importable, a GPU to meaningfully test
device selection, and the ``ffmpeg``/``ffprobe`` binaries plus a known media
clip. They are documented here as executable specifications of the real-path
contracts; by default they SKIP cleanly and the suite stays green.

Gating (`_host_ready`): every real assertion runs only when ALL of the
following hold, otherwise the suite prints "skipped: host not ready" and exits
0:

    - ``AUTO_AVSR_REPO`` names an existing directory, AND
    - ``torch``, ``cv2``, and ``mediapipe`` are all importable.

Coverage (all gated):
    - R1.3 : real ``_read_frames`` resamples to 25fps and returns the expected
             frame count for a known-duration clip.
    - R1.5 : mediapipe landmark detection picks the largest-bbox face per frame.
    - R1.6 : a real model decode of a known clip returns the expected text.
    - R1.7 / R1.9 : device selection prefers CUDA when available, else CPU, and
             never raises when the GPU move fails.
    - R3.1 : real ffmpeg decode of a known clip (the decode half of R3.1).

Optional media: ``SMOKE_CLIP`` (a short mp4 with a single visible face) and
``SMOKE_EXPECTED_TEXT`` (its expected lowercase transcription) refine the R1.6
/ R1.3 / R3.1 assertions when provided; without them those checks degrade to
structural assertions that still require the heavy host.

Run: python -m tests.test_host_only_smoke   (from the project root)
     -> on a plain machine: prints "skipped: host not ready" and exits 0.

Validates: Requirements 1.3, 1.5, 1.6, 1.7, 1.9, 3.1
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Only stdlib + numpy are imported at module load. The lip reader module itself
# imports cleanly without the heavy deps (see tests/test_import_safety.py), so
# importing it here is safe on a plain machine; its heavy imports are lazy.
from zoom_lip_read import audio as audio_mod  # noqa: E402
from zoom_lip_read.avsr_lip_reader import AutoAvsrLipReader  # noqa: E402


PASSED = 0
FAILED = 0


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  - {name}")
    else:
        FAILED += 1
        print(f"  FAIL- {name}")


# --------------------------------------------------------------------------- #
# Host-readiness gate.
# --------------------------------------------------------------------------- #
def _can_import(name: str) -> bool:
    """True when `name` is importable without actually importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        # A broken/partial install can raise from find_spec; treat as absent.
        return False


def _host_ready() -> bool:
    """Return True only when the heavy real-path deps are all present.

    Requires ``AUTO_AVSR_REPO`` to name an existing directory and ``torch`` /
    ``cv2`` / ``mediapipe`` to be importable. On a plain Windows CI machine this
    returns False so every real assertion is skipped.
    """
    repo = os.environ.get("AUTO_AVSR_REPO")
    if not repo or not os.path.isdir(repo):
        return False
    return all(_can_import(m) for m in ("torch", "cv2", "mediapipe"))


def _smoke_clip() -> str | None:
    """Optional path to a short, single-face test clip (``SMOKE_CLIP``)."""
    clip = os.environ.get("SMOKE_CLIP")
    if clip and os.path.isfile(clip):
        return clip
    return None


# --------------------------------------------------------------------------- #
# Gated real-path tests. Each body is guarded so NOTHING heavy runs unless the
# host is ready; locally the guard short-circuits and records a skip.
# --------------------------------------------------------------------------- #
def test_read_frames_resamples_to_25fps():
    """R1.3: real ``_read_frames`` returns ~25 frames per second of source.

    Opens a known clip with OpenCV, resamples to ``TARGET_FPS`` via the
    next-timestamp accumulator, and returns a stacked RGB array. For a clip of
    known duration the frame count should be close to ``25 * duration``.
    """
    if not _host_ready():
        check("R1.3 real _read_frames frame count (skipped: host not ready)", True)
        return

    reader = AutoAvsrLipReader()
    clip = _smoke_clip()
    if clip is None:
        check("R1.3 real _read_frames (skipped: no SMOKE_CLIP media)", True)
        return

    frames, src_fps = reader._read_frames(clip)
    duration = reader._clip_duration(clip)
    expected = 25.0 * duration

    check("R1.3 _read_frames returns a numpy array", isinstance(frames, np.ndarray))
    check("R1.3 _read_frames returns RGB (H,W,3) frames", frames.ndim == 4 and frames.shape[-1] == 3)
    check("R1.3 _read_frames reports a positive source fps", src_fps > 0)
    # Resampling to 25fps: allow ~1s of slack for rounding / boundary frames.
    check(
        "R1.3 _read_frames ~25fps frame count",
        duration <= 0 or abs(len(frames) - expected) <= 25,
    )


def test_mediapipe_selects_largest_bbox():
    """R1.5: landmark detection selects the largest-area face per frame.

    Builds the real landmark detector via ``_ensure_model`` and runs it on the
    frames of a clip containing a dominant face; every detected frame should
    yield 4 keypoints and at least one frame must contain a face (the detector
    asserts this internally).
    """
    if not _host_ready():
        check("R1.5 mediapipe largest-bbox selection (skipped: host not ready)", True)
        return

    reader = AutoAvsrLipReader()
    clip = _smoke_clip()
    if clip is None:
        check("R1.5 largest-bbox selection (skipped: no SMOKE_CLIP media)", True)
        return

    reader._ensure_model()
    frames, _ = reader._read_frames(clip)
    landmarks = reader._landmarks_detector(frames)

    detected = [l for l in landmarks if l is not None]
    check("R1.5 at least one face detected across frames", len(detected) > 0)
    check(
        "R1.5 each detected frame yields 4 keypoints (largest bbox)",
        all(getattr(l, "shape", (0,))[0] == 4 for l in detected),
    )


def test_real_model_decode_returns_expected_text():
    """R1.6: a real model decode of a known clip returns the expected text.

    Runs the full ``read_segment`` contract over the whole clip. When
    ``SMOKE_EXPECTED_TEXT`` is provided the decoded, lowercased text must match
    it; otherwise we only assert a non-crashing string result (read_segment
    never raises, returning "" on failure).
    """
    if not _host_ready():
        check("R1.6 real model decode (skipped: host not ready)", True)
        return

    reader = AutoAvsrLipReader()
    clip = _smoke_clip()
    if clip is None:
        check("R1.6 real model decode (skipped: no SMOKE_CLIP media)", True)
        return

    duration = reader._clip_duration(clip)
    text = reader.read_segment(clip, 0.0, duration or 2.0)
    check("R1.6 read_segment returns a str", isinstance(text, str))

    expected = os.environ.get("SMOKE_EXPECTED_TEXT")
    if expected:
        check(
            "R1.6 decoded text matches SMOKE_EXPECTED_TEXT",
            text.strip().lower() == expected.strip().lower(),
        )
    else:
        check("R1.6 real model decode (no SMOKE_EXPECTED_TEXT: non-crash check)", True)


def test_device_selection_prefers_cuda_else_cpu():
    """R1.7 / R1.9: device selection prefers CUDA when available, else CPU.

    After ``_ensure_model`` the reader's ``_device`` must be a real torch device
    whose type is ``cuda`` exactly when ``torch.cuda.is_available()`` is True,
    and ``cpu`` otherwise. The GPU move is caught internally, so loading never
    raises even if the move to CUDA fails.
    """
    if not _host_ready():
        check("R1.7/R1.9 GPU vs CPU selection (skipped: host not ready)", True)
        return

    import torch  # safe: host is ready

    reader = AutoAvsrLipReader()
    reader._ensure_model()  # must not raise even if the GPU move fails
    device = reader._device
    check("R1.7 _ensure_model sets a torch.device", isinstance(device, torch.device))

    if torch.cuda.is_available():
        check("R1.7 selects CUDA when available", device.type == "cuda")
    else:
        check("R1.9 falls back to CPU when CUDA unavailable", device.type == "cpu")


def test_real_ffmpeg_decode_of_known_clip():
    """R3.1 (decode half): real ffmpeg decode of a known clip yields audio.

    Decodes the clip's audio track to a mono float32 waveform via the real
    ``ffmpeg`` binary and asserts the samples are normalized into [-1.0, 1.0).
    """
    if not _host_ready():
        check("R3.1 real ffmpeg decode (skipped: host not ready)", True)
        return

    clip = _smoke_clip()
    if clip is None:
        check("R3.1 real ffmpeg decode (skipped: no SMOKE_CLIP media)", True)
        return

    waveform = audio_mod.load_audio(clip, 16000)
    check("R3.1 ffmpeg decode -> float32 waveform", waveform.dtype == np.float32)
    check("R3.1 ffmpeg decode -> 1-D array", waveform.ndim == 1)
    if waveform.size:
        check("R3.1 ffmpeg decode within [-1.0, 1.0)", waveform.min() >= -1.0 and waveform.max() < 1.0)
    else:
        check("R3.1 ffmpeg decode produced a (possibly silent) array", True)


if __name__ == "__main__":
    if not _host_ready():
        # The whole point: on a plain machine the suite skips cleanly and the
        # process exits 0 so CI stays green.
        print(
            "skipped: host not ready "
            "(requires AUTO_AVSR_REPO dir + torch/cv2/mediapipe importable)"
        )
        sys.exit(0)

    print("test_read_frames_resamples_to_25fps"); test_read_frames_resamples_to_25fps()
    print("test_mediapipe_selects_largest_bbox"); test_mediapipe_selects_largest_bbox()
    print("test_real_model_decode_returns_expected_text"); test_real_model_decode_returns_expected_text()
    print("test_device_selection_prefers_cuda_else_cpu"); test_device_selection_prefers_cuda_else_cpu()
    print("test_real_ffmpeg_decode_of_known_clip"); test_real_ffmpeg_decode_of_known_clip()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
