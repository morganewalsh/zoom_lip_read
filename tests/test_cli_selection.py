"""Property test for CLI lip-reader selection (``resolve_lip_reader``).

Covers the dry-run/repo truth table. The test monkeypatches ``os.path.isdir``
and the ``AUTO_AVSR_REPO`` environment variable and asserts the *type* of the
returned reader. No heavy construction happens in ``AutoAvsrLipReader.__init__``
(torch/cv2/mediapipe are imported lazily), so this runs on a plain machine with
only the standard library (hypothesis is used when importable, otherwise a
table-driven fallback exercises the same property).

Run: python -m tests.test_cli_selection   (from the project root)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read import cli  # noqa: E402
from zoom_lip_read.avsr_lip_reader import AutoAvsrLipReader  # noqa: E402
from zoom_lip_read.lip_reader import NullLipReader  # noqa: E402

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - exercised on hosts without hypothesis
    HAS_HYPOTHESIS = False

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


def _assert_selection(dry_run, repo_exists):
    # Feature: colab-integration, Property 4: CLI lip-reader selection follows
    # the dry-run/repo truth table -- NullLipReader when dry_run; else
    # AutoAvsrLipReader iff the repo dir exists, NullLipReader otherwise.
    saved_env = os.environ.get("AUTO_AVSR_REPO")
    saved_isdir = os.path.isdir
    try:
        os.environ["AUTO_AVSR_REPO"] = "/some/repo"
        os.path.isdir = lambda path: repo_exists  # type: ignore[assignment]
        reader = cli.resolve_lip_reader(dry_run)
        if dry_run:
            assert isinstance(reader, NullLipReader), (
                f"dry_run should give NullLipReader, got {type(reader).__name__}"
            )
        elif repo_exists:
            assert isinstance(reader, AutoAvsrLipReader), (
                f"repo present should give AutoAvsrLipReader, got {type(reader).__name__}"
            )
        else:
            assert isinstance(reader, NullLipReader), (
                f"repo absent should give NullLipReader, got {type(reader).__name__}"
            )
            # Absent-repo fallback must not be the real reader.
            assert not isinstance(reader, AutoAvsrLipReader)
    finally:
        os.path.isdir = saved_isdir  # type: ignore[assignment]
        if saved_env is None:
            os.environ.pop("AUTO_AVSR_REPO", None)
        else:
            os.environ["AUTO_AVSR_REPO"] = saved_env


def test_selection_truth_table():
    if HAS_HYPOTHESIS:

        @settings(max_examples=50)
        @given(st.booleans(), st.booleans())
        def run(dry_run, repo_exists):
            _assert_selection(dry_run, repo_exists)

        run()
        check("Property 4: resolve_lip_reader follows dry-run/repo truth table (hypothesis)", True)
    else:
        ok = True
        for dry_run in (True, False):
            for repo_exists in (True, False):
                try:
                    _assert_selection(dry_run, repo_exists)
                except AssertionError as exc:
                    ok = False
                    print(f"    counterexample (dry_run={dry_run}, repo_exists={repo_exists}): {exc}")
        check("Property 4: resolve_lip_reader follows dry-run/repo truth table (table)", ok)


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print("test_selection_truth_table"); test_selection_truth_table()
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
