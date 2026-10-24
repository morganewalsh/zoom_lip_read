"""Manifest test for `requirements.txt` and the default Whisper model size.

Covers:
- Requirement 8.1: the existing WhisperConfig default model size remains `base`
  for the existing pipeline/CLI paths (asserted via direct construction).
- Requirement 9.1: `fastapi`, `uvicorn[standard]`, and `python-multipart` are
  listed.
- Requirement 9.2: `pytorch-lightning` is listed and there is NO standalone
  `lightning` line (the bare `lightning` package clashes with auto_avsr's
  `lightning.py`). The check distinguishes `pytorch-lightning` from `lightning`.
- Requirement 9.3: `av` and `mediapipe` are listed without pinned versions.

No heavy dependencies are touched: this reads the manifest file and constructs
a dataclass.

Run: python -m tests.test_requirements_manifest   (from the project root)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read.config import WhisperConfig  # noqa: E402

PASSED = 0
FAILED = 0

# Characters that introduce a version pin / constraint in a requirements line.
_PIN_CHARS = ("=", "<", ">", "~", "!")


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  - {name}")
    else:
        FAILED += 1
        print(f"  FAIL- {name}")


def _read_requirement_lines():
    """Return the non-empty, non-comment requirement lines from requirements.txt."""
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip any inline comment.
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _package_name(line: str) -> str:
    """Extract the bare package name (with any extras) from a requirement line.

    e.g. ``uvicorn[standard]>=1`` -> ``uvicorn[standard]``;
         ``pytorch-lightning`` -> ``pytorch-lightning``.
    """
    # Split off the version constraint at the first pin char.
    name = re.split(r"[=<>~!]", line, maxsplit=1)[0].strip()
    return name


def _base_name(line: str) -> str:
    """Extract the package name without extras or version, e.g. ``uvicorn``."""
    name = _package_name(line)
    return name.split("[", 1)[0].strip()


def _is_pinned(line: str) -> bool:
    """True when the requirement line carries any version specifier."""
    # Only look past the extras bracket, so `uvicorn[standard]` is not a pin.
    after_extras = re.sub(r"\[[^\]]*\]", "", line)
    return any(c in after_extras for c in _PIN_CHARS)


def test_requirements_manifest():
    lines = _read_requirement_lines()
    names = [_package_name(l) for l in lines]
    base_names = [_base_name(l) for l in lines]

    # R9.1: server dependencies present.
    check("R9.1: fastapi listed", "fastapi" in base_names)
    check("R9.1: uvicorn[standard] listed", "uvicorn[standard]" in names)
    check("R9.1: python-multipart listed", "python-multipart" in base_names)

    # R9.2: pytorch-lightning present; no standalone `lightning` line.
    check("R9.2: pytorch-lightning listed", "pytorch-lightning" in base_names)
    # Distinguish a bare `lightning` entry from `pytorch-lightning`: match the
    # exact base name, not a substring, so `pytorch-lightning` does not count.
    has_bare_lightning = any(b == "lightning" for b in base_names)
    check("R9.2: no standalone `lightning` line", not has_bare_lightning)

    # R9.3: av and mediapipe listed without pinned versions.
    def unpinned(pkg: str) -> bool:
        matches = [l for l in lines if _base_name(l) == pkg]
        return bool(matches) and all(not _is_pinned(l) for l in matches)

    check("R9.3: av listed", "av" in base_names)
    check("R9.3: av unpinned", unpinned("av"))
    check("R9.3: mediapipe listed", "mediapipe" in base_names)
    check("R9.3: mediapipe unpinned", unpinned("mediapipe"))

    # Keep the existing pinned entries intact (sanity on a couple of them).
    check("existing faster-whisper entry kept", any(b == "faster-whisper" for b in base_names))
    check("existing numpy entry kept", any(b == "numpy" for b in base_names))


def test_whisper_default_model_size():
    # R8.1: the existing WhisperConfig default model size remains `base`.
    check("R8.1: WhisperConfig().model_size == 'base'", WhisperConfig().model_size == "base")


if __name__ == "__main__":
    print("test_requirements_manifest"); test_requirements_manifest()
    print("test_whisper_default_model_size"); test_whisper_default_model_size()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
