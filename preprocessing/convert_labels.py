#!/usr/bin/env python
"""Batch-convert SymLG (.lg) label files to LaTeX (.tex)."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.lg_to_latex import convert_lg_dir, convert_lg_file, DEFAULT_MAP
from preprocessing.batch_utils import default_workers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert SymLG (.lg) files to LaTeX text files."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input file or directory",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output .tex file (single input) or directory (batch)",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=DEFAULT_MAP,
        help="Symbol map CSV (default: preprocessing/symLG_map.csv)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recurse into subdirectories",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help=f"Parallel worker processes (default: {default_workers()})",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    args = parser.parse_args()

    recursive = not args.no_recursive
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if input_path.is_file():
        ok, msg = convert_lg_file(input_path, output_path, map_path=args.map)
        if ok:
            print(f"Wrote {output_path}")
            print(msg)
        else:
            raise SystemExit(f"Failed: {msg}")
        return

    if not input_path.is_dir():
        raise SystemExit(f"Input not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    ok, fail, errors = convert_lg_dir(
        input_path,
        output_path,
        map_path=args.map,
        recursive=recursive,
        workers=args.workers,
        show_progress=not args.no_progress,
    )

    print(f"Done: {ok} converted, {fail} failed -> {output_path}")
    if errors:
        print("First errors:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
