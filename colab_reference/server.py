import os, tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import backend

FAKE_MODE = os.environ.get("FAKE_MODE", "1") == "1"
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok", "fake_mode": FAKE_MODE}


@app.post("/transcribe_chunk")
def transcribe_chunk(file: UploadFile = File(...), session_id: str = Form(...),
                     chunk_index: int = Form(...), chunk_start: float = Form(0.0),
                     mode: str = Form("auto")):
    if FAKE_MODE:
        t = chunk_start
        return {"chunk_index": chunk_index, "segments": [
            {"start": t, "end": t + 2.6, "text": "let's meet friday", "source": "audio"},
            {"start": t + 2.6, "end": t + 4.7, "text": "see you then", "source": "lip", "unclear": False},
        ]}
    suffix = os.path.splitext(file.filename or "")[1] or ".webm"
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "upload" + suffix)
        with open(path, "wb") as f:
            f.write(file.file.read())
        segments = backend.transcribe(path, chunk_start=chunk_start, mode=mode)
    return {"chunk_index": chunk_index, "segments": segments}