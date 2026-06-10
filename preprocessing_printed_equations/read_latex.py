#!/usr/bin/env python
"""Read LaTeX labels from the printed-equations dataset."""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_PATH = DATA_DIR / "printed_mathematical_expressions_train"
FORMULAS_PATH = DATA_DIR / "final_png_formulas.txt"

# Same-line expression separators (space-tokenized).
SPLIT_TOKENS = frozenset({",", ";", "\\;", "\\quad", "\\qquad"})


@dataclass(frozen=True)
class PrintedEquationRecord:
    image_filename: str
    latex: str


def iter_csv_records(path: Path = CSV_PATH) -> Iterator[PrintedEquationRecord]:
    """Yield (image_filename, raw LaTeX) from the CSV export."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image = (row.get("image_filename") or "").strip()
            latex = (row.get("latex") or "").strip()
            if image and latex:
                yield PrintedEquationRecord(image_filename=image, latex=latex)


def load_csv_records(path: Path = CSV_PATH, limit: Optional[int] = None) -> List[PrintedEquationRecord]:
    records: List[PrintedEquationRecord] = []
    for record in iter_csv_records(path):
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def iter_formula_lines(path: Path = FORMULAS_PATH) -> Iterator[str]:
    """Yield one space-tokenized LaTeX line per row."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield text


def load_formula_lines(path: Path = FORMULAS_PATH, limit: Optional[int] = None) -> List[str]:
    lines: List[str] = []
    for line in iter_formula_lines(path):
        lines.append(line)
        if limit is not None and len(lines) >= limit:
            break
    return lines


def tokenize_latex(text: str) -> List[str]:
    return text.strip().split()


_TRAILING_PUNCT = frozenset({".", ","})


def strip_trailing_punctuation(tokens: List[str]) -> List[str]:
    """Remove trailing '.' or ',' tokens."""
    end = len(tokens)
    while end > 0 and tokens[end - 1] in _TRAILING_PUNCT:
        end -= 1
    return tokens[:end]


def normalize_equation_line(line: str) -> str:
    """Drop trailing '.' and ',' then rejoin tokens."""
    tokens = strip_trailing_punctuation(tokenize_latex(line))
    return " ".join(tokens)


_OPEN_TOKENS = frozenset({"(", "[", "{", "\\{", "\\left(", "\\left[", "\\left\\{"})
_CLOSE_TOKENS = frozenset({")", "]", "}", "\\}", "\\right)", "\\right]", "\\right\\}"})


def _delimiter_depth(token: str, depth: int) -> int:
    if token in _OPEN_TOKENS:
        return depth + 1
    if token in _CLOSE_TOKENS:
        return max(0, depth - 1)
    return depth


def split_equations(tokens: List[str]) -> List[str]:
    """Split a token list into sub-equations on comma/spacing separator tokens.

    Separators are ignored inside balanced (), [], or {} groups.
    """
    parts: List[List[str]] = [[]]
    depth = 0
    for token in tokens:
        if token in SPLIT_TOKENS and depth == 0:
            parts.append([])
            continue
        parts[-1].append(token)
        depth = _delimiter_depth(token, depth)

    results: List[str] = []
    for part in parts:
        if not part:
            continue
        results.append(" ".join(part))
    return results


def split_equation_line(line: str) -> List[str]:
    """Split one space-tokenized LaTeX line into sub-equations."""
    tokens = tokenize_latex(line)
    if not tokens:
        return []
    return split_equations(tokens)


def _print_samples(records: List[PrintedEquationRecord], n: int) -> None:
    for i, record in enumerate(records[:n], start=1):
        print(f"[{i}] {record.image_filename}")
        print(record.latex)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read LaTeX from preprocessing_printed_equations/data/"
    )
    parser.add_argument(
        "--source",
        choices=("csv", "formulas"),
        default="csv",
        help="csv: image_filename + raw latex; formulas: one tokenized line per row",
    )
    parser.add_argument("-n", "--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows to load before sampling (default: stream until enough for -n)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the dataset files",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split each line into sub-equations on , ; \\; \\quad \\qquad",
    )
    args = parser.parse_args()

    if args.source == "csv":
        csv_path = args.data_dir / "printed_mathematical_expressions_train"
        if not csv_path.is_file():
            raise SystemExit(f"CSV not found: {csv_path}")

        pool: List[PrintedEquationRecord] = []
        rng = random.Random(args.seed)
        for record in iter_csv_records(csv_path):
            if args.limit is not None and len(pool) >= max(args.num_samples, args.limit):
                break
            if len(pool) < args.num_samples:
                pool.append(record)
            else:
                j = rng.randint(0, len(pool))
                if j < args.num_samples:
                    pool[j] = record

        print(f"Loaded {len(pool)} sample(s) from {csv_path}")
        if args.split:
            for i, record in enumerate(pool[: args.num_samples], start=1):
                print(f"[{i}] {record.image_filename}")
                print("  full:", record.latex)
                for j, part in enumerate(split_equation_line(record.latex), start=1):
                    print(f"  [{j}] {part}")
                print()
        else:
            _print_samples(pool, args.num_samples)
        return

    formulas_path = args.data_dir / "final_png_formulas.txt"
    if not formulas_path.is_file():
        raise SystemExit(f"Formulas file not found: {formulas_path}")

    pool_lines: List[str] = []
    rng = random.Random(args.seed)
    for line in iter_formula_lines(formulas_path):
        if args.limit is not None and len(pool_lines) >= max(args.num_samples, args.limit):
            break
        if len(pool_lines) < args.num_samples:
            pool_lines.append(line)
        else:
            j = rng.randint(0, len(pool_lines))
            if j < args.num_samples:
                pool_lines[j] = line

    print(f"Loaded {len(pool_lines)} sample(s) from {formulas_path}")
    for i, line in enumerate(pool_lines[: args.num_samples], start=1):
        print(f"[{i}] {line}")
        if args.split:
            for j, part in enumerate(split_equation_line(line), start=1):
                print(f"  [{j}] {part}")
        print()


if __name__ == "__main__":
    main()
