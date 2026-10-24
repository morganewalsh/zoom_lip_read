"""Command-line entrypoint.

Usage:
    python -m zoom_lip_read.cli INPUT.mp4 \
        [--config config.json] \
        [--out-text transcript.txt] [--out-json transcript.json] \
        [--dry-run]

--dry-run uses a NullLipReader so you can exercise the pipeline without a real
lip-reading model. Replace it by importing your own LipReader subclass.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .config import PipelineConfig
from .lip_reader import LipReader, NullLipReader
from .output import to_json, to_text, write_json, write_text
from .pipeline import TranscriptionPipeline


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Meeting-clip transcription with lip-read fallback.")
    p.add_argument("input", help="Path to the meeting video/audio clip.")
    p.add_argument("--config", help="Path to a JSON config file (detector thresholds, etc.).")
    p.add_argument("--out-text", help="Write plain-text transcript to this path.")
    p.add_argument("--out-json", help="Write JSON transcript to this path.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a stub lip reader so the pipeline runs without a real model.",
    )
    p.add_argument(
        "--emit-config",
        metavar="PATH",
        help="Write the effective (default or loaded) config to PATH and exit.",
    )
    return p


def load_config(path: str | None) -> PipelineConfig:
    if path:
        return PipelineConfig.from_json_file(path)
    return PipelineConfig()


def resolve_lip_reader(dry_run: bool) -> LipReader:
    """Select the lip reader for a run (R2.1-2.3).

    A ``--dry-run`` always uses :class:`NullLipReader`. Otherwise the real
    :class:`AutoAvsrLipReader` is selected when the ``AUTO_AVSR_REPO`` directory
    exists (defaulting to ``/content/auto_avsr``); if the repo is absent we fall
    back to :class:`NullLipReader` so the pipeline still runs on hosts without
    the model. ``AutoAvsrLipReader`` is imported lazily so importing the CLI does
    not pull in the heavy VSR dependencies.
    """
    if dry_run:
        return NullLipReader()
    repo = os.environ.get("AUTO_AVSR_REPO") or "/content/auto_avsr"
    if os.path.isdir(repo):
        from .avsr_lip_reader import AutoAvsrLipReader

        return AutoAvsrLipReader()
    return NullLipReader()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    config = load_config(args.config)

    if args.emit_config:
        config.to_json_file(args.emit_config)
        print(f"Wrote config to {args.emit_config}")
        return 0

    if not Path(args.input).exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    lip_reader = resolve_lip_reader(args.dry_run)
    pipeline = TranscriptionPipeline(lip_reader=lip_reader, config=config)
    result = pipeline.run(args.input)

    if args.out_text:
        write_text(result, args.out_text)
        print(f"Wrote text transcript to {args.out_text}")
    if args.out_json:
        write_json(result, args.out_json)
        print(f"Wrote JSON transcript to {args.out_json}")

    if not args.out_text and not args.out_json:
        print("=== TRANSCRIPT ===")
        print(to_text(result))
        print("\n=== JSON ===")
        print(to_json(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
