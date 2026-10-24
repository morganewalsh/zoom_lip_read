import argparse, os, sys, subprocess, tempfile, threading
import cv2, numpy as np, torch, librosa

REPO = "/content/auto_avsr"
TUT = f"{REPO}/tutorials"
CKPT = f"{TUT}/vsr_trlrs2lrs3vox2avsp_base.pth"
TFLITE = f"{TUT}/blaze_face_short_range.tflite"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

TARGET_FPS = 25
MAX_SIDE = 960

# ---- tune on a real noisy clip (see tuning cell) ----
NOISE_CFG = {
    "window_sec": 0.5,
    "flatness_thr": 0.20,    # higher = more noise-like (fan/white noise high, clean speech low)
    "rms_floor_db": -50.0,   # below this = silence
    "flag_silence": False,   # True = also lip-read silent windows (audio dropout). Off by default:
                             #   a closed mouth during real silence can produce made-up words.
    "min_span_sec": 1.0,
    "pad_to_sec": 2.0,       # lip-read windows shorter than this are widened
}
WHISPER_SIZE = "small.en"    # "base.en" is faster

_lock = threading.Lock()


# ---------------- lip reading (from teammate notebook) ----------------
def read_frames(path):
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
        scale = MAX_SIDE / max(h, w)
        if scale < 1:
            bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frames from {path}")
    return np.stack(frames), src_fps


class TasksLandmarksDetector:
    def __init__(self, model_path=TFLITE):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(base_options=mp_python.BaseOptions(model_asset_path=model_path),
                                       min_detection_confidence=0.5))

    def __call__(self, video_frames):
        landmarks = []
        for frame in video_frames:
            h, w = frame.shape[:2]
            image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
            result = self.detector.detect(image)
            if not result.detections:
                landmarks.append(None)
                continue
            best = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
            landmarks.append(np.array([[int(k.x * w), int(k.y * h)] for k in best.keypoints[:4]]))
        assert any(l is not None for l in landmarks), "Cannot detect a face in any frame of the video"
        return landmarks


class LipReader:
    def __init__(self, ckpt_path=CKPT):
        from lightning import ModelModule
        from datamodule.transforms import VideoTransform
        from preparation.detectors.mediapipe.video_process import VideoProcess
        import mediapipe as mp
        if hasattr(mp, "solutions"):
            from preparation.detectors.mediapipe.detector import LandmarksDetector
            self.landmarks_detector = LandmarksDetector()
        else:
            self.landmarks_detector = TasksLandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform(subset="test")
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        self.modelmodule = ModelModule(argparse.Namespace(modality="video"))
        self.modelmodule.model.load_state_dict(ckpt)
        self.modelmodule.eval()
        self.device = torch.device("cpu")
        if DEVICE.type == "cuda":
            try:
                self.modelmodule.to(DEVICE)
                self.device = DEVICE
            except Exception as e:
                print("GPU move failed, staying on CPU:", e)

    def crops(self, path):
        video, src_fps = read_frames(path)
        landmarks = self.landmarks_detector(video)
        return self.video_process(video, landmarks), src_fps

    def decode(self, crops):
        x = torch.tensor(crops).permute(0, 3, 1, 2)
        x = self.video_transform(x)
        with torch.no_grad():
            try:
                return self.modelmodule(x.to(self.device))
            except RuntimeError as e:
                if self.device.type == "cuda":          # device mismatch inside the repo -> fall back
                    print("GPU decode failed, falling back to CPU:", e)
                    self.modelmodule.to("cpu"); self.device = torch.device("cpu")
                    return self.modelmodule(x)
                raise

    def read(self, path):
        try:
            crops, _ = self.crops(path)
            return (self.decode(crops) or "").strip().lower()
        except Exception as e:                           # e.g. no face found
            print("lip read failed:", e)
            return ""

    def read_segment(self, video_path, start, end, total, workdir):
        pad = NOISE_CFG["pad_to_sec"]
        if end - start < pad:
            mid = (start + end) / 2
            start, end = max(0.0, mid - pad / 2), min(total, mid + pad / 2)
        seg = os.path.join(workdir, f"seg_{start:.2f}.mp4")
        run(["ffmpeg", "-y", "-ss", f"{start:.2f}", "-to", f"{end:.2f}", "-i", video_path,
             "-r", "25", "-an", "-c:v", "libx264", "-preset", "veryfast", seg])
        return self.read(seg)


# ---------------- audio side ----------------
def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_audio(path):
    """Any video/audio file -> 16 kHz mono float32 numpy array, via ffmpeg (empty if no audio)."""
    raw = subprocess.run(["ffmpeg", "-i", path, "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def has_audio(path):
    return len(load_audio(path)) > 16000 * 0.2


def to_mp4_25fps(src, dst):
    run(["ffmpeg", "-y", "-i", src, "-map", "0:v:0", "-map", "0:a:0?", "-r", "25",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ar", "16000", "-ac", "1", dst])


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


class AudioASR:
    def __init__(self):
        from faster_whisper import WhisperModel
        ct = "float16" if DEVICE.type == "cuda" else "int8"
        self.model = WhisperModel(WHISPER_SIZE, device=DEVICE.type, compute_type=ct)

    def words(self, path):
        # numpy audio, not a file path: faster-whisper's own file loader breaks on newer
        # PyAV ("unexpected keyword argument 'metadata_errors'")
        segs, _ = self.model.transcribe(load_audio(path), language="en", word_timestamps=True)
        return [{"start": float(w.start), "end": float(w.end), "word": w.word.strip()}
                for s in segs for w in (s.words or [])]


def window_stats(audio_path):
    y, sr = load_audio(audio_path), 16000
    hop = int(NOISE_CFG["window_sec"] * sr)
    stats = []
    for i in range(0, max(len(y), 1), hop):
        w = y[i:i + hop]
        if len(w) < hop // 4:
            break
        rms = float(np.sqrt(np.mean(w ** 2)) + 1e-10)
        flat = float(np.mean(librosa.feature.spectral_flatness(y=w, n_fft=512, hop_length=256)))
        stats.append({"start": i / sr, "end": (i + len(w)) / sr, "rms_db": 20 * np.log10(rms), "flatness": flat})
    return stats


def detect_noisy_spans(audio_path):
    try:
        stats = window_stats(audio_path)
    except Exception as e:
        print("audio unreadable, treating whole chunk as unreliable:", e)
        return None
    flagged = []
    for s in stats:
        silent = s["rms_db"] < NOISE_CFG["rms_floor_db"]
        noisy = (not silent) and s["flatness"] > NOISE_CFG["flatness_thr"]
        if noisy or (silent and NOISE_CFG["flag_silence"]):
            flagged.append([s["start"], s["end"]])
    merged = []
    for a, b in flagged:
        if merged and a - merged[-1][1] < 1e-3:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged if b - a >= NOISE_CFG["min_span_sec"]]


# ---------------- main entry ----------------
_lip, _asr = None, None

def load_models():
    global _lip, _asr
    if _lip is None:
        print("Loading lip reader..."); _lip = LipReader()
    if _asr is None:
        print("Loading Whisper..."); _asr = AudioASR()
    print("Lip reader on", _lip.device, "| Whisper on", DEVICE)


def group_words(words, offset):
    segs = []
    for w in words:
        if segs and w["start"] - segs[-1]["end"] <= 1.0:
            segs[-1]["text"] += " " + w["word"]; segs[-1]["end"] = w["end"]
        else:
            segs.append({"start": w["start"], "end": w["end"], "text": w["word"], "source": "audio"})
    for s in segs:
        s["start"] = round(s["start"] + offset, 2); s["end"] = round(s["end"] + offset, 2)
    return segs


def transcribe(input_path, chunk_start=0.0, mode="auto"):
    """Temp files are deleted when this returns."""
    load_models()
    with _lock, tempfile.TemporaryDirectory() as tmp:
        mp4 = os.path.join(tmp, "chunk.mp4")
        to_mp4_25fps(input_path, mp4)
        total = duration(mp4)
        if mode == "lip_only" or not has_audio(mp4):
            spans, words = [(0.0, total)], []
        else:
            words = _asr.words(mp4)
            spans = [] if mode == "audio_only" else detect_noisy_spans(mp4)
            if spans is None:
                spans = [(0.0, total)]

        def in_span(w):
            mid = (w["start"] + w["end"]) / 2
            return any(a <= mid <= b for a, b in spans)

        segments = group_words([w for w in words if not in_span(w)], chunk_start)
        for a, b in spans:
            text = _lip.read_segment(mp4, a, b, total, tmp)
            segments.append({"start": round(a + chunk_start, 2), "end": round(b + chunk_start, 2),
                             "text": text, "source": "lip", "unclear": text == ""})
        segments.sort(key=lambda s: s["start"])
        return segments