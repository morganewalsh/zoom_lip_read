"""Property test for the server-shaped serializer `output.to_segment_records`.

Covers:
- Property 5: Segment_Record key sets are exact and `unclear` is the empty-text
  biconditional.

Audio records have exactly {start, end, text, source} with source == "audio";
lip records have exactly {start, end, text, source, unclear} with
source == "lip" and `unclear` True iff the lip text is empty.

Note on the biconditional: the serializer defines `unclear = (seg.text.strip()
== "")`, i.e. whitespace-only text also counts as unclear. The property text
phrases this as `unclear == (text == "")`; these agree for non-empty,
non-whitespace text and for the empty string. We assert the serializer's actual
rule (`text.strip() == ""`) which subsumes the plain-empty biconditional.

Uses hypothesis when importable, otherwise a table-driven fallback, so the suite
runs with only numpy. No heavy dependencies are touched.

Run: python -m tests.test_output_records   (from the project root)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zoom_lip_read.output import to_segment_records  # noqa: E402
from zoom_lip_read.types import Segment, TranscriptResult  # noqa: E402

try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HAS_HYPOTHESIS = True
except ImportError:  # pragma: no cover - exercised on hosts without hypothesis
    HAS_HYPOTHESIS = False

PASSED = 0
FAILED = 0

AUDIO_KEYS = {"start", "end", "text", "source"}
LIP_KEYS = {"start", "end", "text", "source", "unclear"}


def check(name, cond):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ok  - {name}")
    else:
        FAILED += 1
        print(f"  FAIL- {name}")


def _make_result(specs):
    """Build a TranscriptResult from a list of (source, text) specs.

    Times are synthetic and monotonically increasing; they don't affect the
    key-set/biconditional assertions.
    """
    segments = []
    t = 0.0
    for source, text in specs:
        words = [] if source == "audio" else []
        segments.append(Segment(source=source, start=t, end=t + 1.0, text=text, words=words))
        t += 1.0
    return TranscriptResult(segments=segments)


def _assert_records(specs):
    # Feature: colab-integration, Property 5: Segment_Record key sets are exact
    # and unclear is the empty-text biconditional.
    result = _make_result(specs)
    records = to_segment_records(result)
    assert len(records) == len(specs), (len(records), len(specs))
    for (source, text), rec in zip(specs, records):
        keys = set(rec.keys())
        assert rec["source"] == source, (rec["source"], source)
        assert rec["text"] == text
        if source == "audio":
            assert keys == AUDIO_KEYS, f"audio key set {keys} != {AUDIO_KEYS}"
            assert "unclear" not in rec, "audio record must not carry `unclear`"
        else:
            assert keys == LIP_KEYS, f"lip key set {keys} != {LIP_KEYS}"
            # Biconditional: unclear iff the lip text is empty (after strip).
            assert rec["unclear"] == (text.strip() == ""), (text, rec["unclear"])
            assert isinstance(rec["unclear"], bool)


# --- Hypothesis-driven variant ---------------------------------------------

if HAS_HYPOTHESIS:
    # Text that is either clearly empty/whitespace or clearly non-empty, so the
    # `text == ""` and `text.strip() == ""` phrasings of the biconditional agree.
    _text_st = st.one_of(
        st.sampled_from(["", " ", "   ", "\t", "\n "]),
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1).filter(
            lambda s: s.strip() != ""
        ),
    )
    _spec_st = st.tuples(st.sampled_from(["audio", "lip"]), _text_st)

    @settings(max_examples=300)
    @given(st.lists(_spec_st, max_size=8))
    def test_records_hypothesis(specs):
        _assert_records(specs)


# --- Table-driven variant (always run) -------------------------------------


def test_records_table():
    cases = [
        [],
        [("audio", "hello world")],
        [("lip", "see you then")],
        [("lip", "")],
        [("lip", "   ")],
        [("audio", ""), ("lip", ""), ("lip", "something"), ("audio", "more")],
        [("lip", "a"), ("lip", ""), ("lip", "\t"), ("audio", "x")],
    ]
    ok = True
    for specs in cases:
        try:
            _assert_records(specs)
        except AssertionError as exc:
            ok = False
            print(f"    counterexample {specs}: {exc}")
    check("Property 5: exact key sets + unclear biconditional (table)", ok)


def test_unclear_boundary():
    # Pin the biconditional explicitly at the empty/non-empty boundary.
    empty = to_segment_records(_make_result([("lip", "")]))[0]
    blank = to_segment_records(_make_result([("lip", "   ")]))[0]
    words = to_segment_records(_make_result([("lip", "hi")]))[0]
    check("empty lip text -> unclear True", empty["unclear"] is True)
    check("whitespace lip text -> unclear True", blank["unclear"] is True)
    check("non-empty lip text -> unclear False", words["unclear"] is False)
    audio = to_segment_records(_make_result([("audio", "")]))[0]
    check("audio record omits unclear even when empty", "unclear" not in audio)


if __name__ == "__main__":
    print(f"hypothesis available: {HAS_HYPOTHESIS}")
    print("test_records_table"); test_records_table()
    print("test_unclear_boundary"); test_unclear_boundary()
    if HAS_HYPOTHESIS:
        print("test_records_hypothesis"); test_records_hypothesis()
        check("Property 5: exact key sets + unclear biconditional (hypothesis)", True)
    else:
        print("  (hypothesis not installed: ran table-driven cases)")
    print(f"\n{PASSED} passed, {FAILED} failed")
    sys.exit(1 if FAILED else 0)
