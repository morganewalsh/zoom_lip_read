"""Import-safety tests for heavy-dependency isolation.

Confirms that the package and its server / lip-reader modules import cleanly on
a plain machine where the heavy ML dependencies (torch, cv2, mediapipe,
faster_whisper) are unavailable. Heavy imports live inside the methods that use
them, never at module import time (R1.16, R3.5, R10.1).

The test installs a `sys.meta_path` finder that raises `ImportError` for the
blocked names (and any of their submodules), removes any already-imported
blocked modules, then imports:

    - zoom_lip_read
    - zoom_lip_read.server
    - zoom_lip_read.avsr_lip_reader

and asserts none of the blocked modules landed in `sys.modules` afterward.
`sys.modules` / `sys.meta_path` are restored when done.

Run: python -m tests.test_import_safety   (from the project root)
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Heavy modules that must NOT be pulled in by importing the light modules.
BLOCKED = ("torch", "cv2", "mediapipe", "faster_whisper")

# Target modules that must import without touching the blocked dependencies.
TARGETS = ("zoom_lip_read", "zoom_lip_read.server", "zoom_lip_read.avsr_lip_reader")


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


def _is_blocked(name: str) -> bool:
    """True when `name` is a blocked module or a submodule of one."""
    return any(name == b or name.startswith(b + ".") for b in BLOCKED)


class _BlockingFinder:
    """A meta_path finder that refuses to import the blocked modules.

    Its `find_spec` raises `ImportError` for any blocked name (or submodule), so
    even a lazy `import torch` deep inside a method would fail. For every other
    name it returns `None`, delegating to the remaining finders on `sys.meta_path`.
    """

    def find_spec(self, fullname, path=None, target=None):
        if _is_blocked(fullname):
            raise ImportError(f"import of {fullname!r} is blocked by test harness")
        return None

    # Legacy import protocol, in case anything falls back to it.
    def find_module(self, fullname, path=None):
        if _is_blocked(fullname):
            raise ImportError(f"import of {fullname!r} is blocked by test harness")
        return None


def test_light_modules_import_without_heavy_deps():
    saved_meta_path = list(sys.meta_path)
    saved_modules = dict(sys.modules)

    try:
        # Drop any already-imported blocked modules and target modules so the
        # imports below actually execute under the blocker.
        for name in list(sys.modules):
            if _is_blocked(name) or name == "zoom_lip_read" or name.startswith("zoom_lip_read."):
                del sys.modules[name]

        # Install the blocker at the front so it intercepts first.
        sys.meta_path.insert(0, _BlockingFinder())

        # Sanity: the blocker actually blocks a direct heavy import.
        blocked_ok = True
        for b in BLOCKED:
            try:
                importlib.import_module(b)
                blocked_ok = False  # should have raised
            except ImportError:
                pass
            except Exception:
                blocked_ok = False
        check("blocker raises ImportError for heavy modules", blocked_ok)
        # Remove any partial entries the sanity probe may have created.
        for name in list(sys.modules):
            if _is_blocked(name):
                del sys.modules[name]

        # Each target must import successfully under the blocker...
        for target in TARGETS:
            try:
                importlib.import_module(target)
                imported = True
            except Exception as e:  # noqa: BLE001 - surface the failure in the check name
                imported = False
                print(f"       import of {target} raised: {e!r}")
            check(f"imports {target} without heavy deps", imported)

            # ...and must not have dragged in any blocked module.
            present = [b for b in BLOCKED if b in sys.modules]
            check(f"no heavy modules in sys.modules after importing {target}", not present)

        # Final global assertion across all three imports.
        leaked = [b for b in BLOCKED if b in sys.modules]
        check("no heavy modules present after all imports", not leaked)
    finally:
        # Restore meta_path and sys.modules exactly.
        sys.meta_path[:] = saved_meta_path
        sys.modules.clear()
        sys.modules.update(saved_modules)


if __name__ == "__main__":
    print("test_light_modules_import_without_heavy_deps")
    test_light_modules_import_without_heavy_deps()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
