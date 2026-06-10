#!/usr/bin/env python
"""Print equations whose tokens are all in dictionary.txt (no OOV)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator, Optional, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MAX_LEN = 100

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from read_latex import (  # noqa: E402
    iter_csv_records,
    iter_formula_lines,
    split_equation_line,
    strip_trailing_punctuation,
    tokenize_latex,
)


def load_dictionary(path: Path) -> Set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def is_legal(tokens: list, dictionary: Set[str], max_len: int) -> bool:
    return (
        bool(tokens)
        and len(tokens) <= max_len
        and all(token in dictionary for token in tokens)
    )


def iter_legal_equations(
    source: str,
    data_path: Path,
    dictionary: Set[str],
    split: bool,
    max_len: int,
    scan_limit: Optional[int] = None,
) -> Iterator[Tuple[str, str]]:
    """Yield (label, equation_text) for each legal equation."""
    scanned = 0
    if source == "formulas":
        for line in iter_formula_lines(data_path):
            scanned += 1
            if scan_limit is not None and scanned > scan_limit:
                return
            parts = split_equation_line(line) if split else [line]
            for part in parts:
                tokens = strip_trailing_punctuation(tokenize_latex(part))
                if is_legal(tokens, dictionary, max_len):
                    yield f"line {scanned}", " ".join(tokens)
        return

    for record in iter_csv_records(data_path):
        scanned += 1
        if scan_limit is not None and scanned > scan_limit:
            return
        parts = split_equation_line(record.latex) if split else [record.latex]
        for part in parts:
            tokens = strip_trailing_punctuation(tokenize_latex(part))
            if is_legal(tokens, dictionary, max_len):
                yield record.image_filename, " ".join(tokens)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print legal equations (all tokens in dictionary.txt)"
    )
    parser.add_argument(
        "--source",
        choices=("formulas", "csv"),
        default="formulas",
        help="formulas: final_png_formulas.txt; csv: printed_mathematical_expressions_train",
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=200000,
        help="Number of legal equations to print (default: 20, 0 = all found within --limit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max source rows/lines to scan",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split on , ; \\; \\quad \\qquad before checking legality",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"Max token length per equation (default: {DEFAULT_MAX_LEN})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory with dictionary.txt and source files",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional output file (one equation per block)",
    )
    args = parser.parse_args()

    dict_path = args.data_dir / "dictionary.txt"
    if not dict_path.is_file():
        raise SystemExit(f"dictionary.txt not found: {dict_path}")

    if args.source == "formulas":
        data_path = args.data_dir / "final_png_formulas.txt"
    else:
        data_path = args.data_dir / "printed_mathematical_expressions_train"
    if not data_path.is_file():
        raise SystemExit(f"Source file not found: {data_path}")

    dictionary = load_dictionary(dict_path)
    lines_out = []
    printed = 0
    scanned_hint = 0

    for label, equation in iter_legal_equations(
        args.source,
        data_path,
        dictionary,
        args.split,
        args.max_len,
        args.limit,
    ):
        printed += 1
        block = f"[{printed}] {label}\n{equation}\n"
        lines_out.append(block)
        if args.num > 0 and printed >= args.num:
            break

    output_text = "\n".join(lines_out)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
        print(f"Wrote {printed} legal equation(s) to {args.output.resolve()}")
    else:
        if output_text:
            print(output_text, end="")
        else:
            print("No legal equations found.")

    print(
        f"Summary: printed {printed} legal equation(s) "
        f"(dictionary size: {len(dictionary)}, max_len={args.max_len}, split={args.split})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
