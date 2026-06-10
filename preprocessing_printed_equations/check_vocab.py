#!/usr/bin/env python
"""Check LaTeX token vocabulary against dictionary.txt."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

DATA_DIR = Path(__file__).resolve().parent / "data"


def has_oov(tokens: List[str], dictionary: Set[str]) -> bool:
    return any(token not in dictionary for token in tokens)


def load_dictionary(path: Path) -> Set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def tokenize(text: str) -> List[str]:
    return text.strip().split()


def analyze_tokens(
    tokens: Iterable[str],
    dictionary: Set[str],
) -> Tuple[Counter[str], int, int]:
    freq: Counter[str] = Counter()
    total = 0
    oov_instances = 0
    for token in tokens:
        freq[token] += 1
        total += 1
        if token not in dictionary:
            oov_instances += 1
    return freq, total, oov_instances


def report(
    name: str,
    freq: Counter[str],
    total_tokens: int,
    oov_instances: int,
    oov_lines: int,
    total_lines: int,
    dictionary: Set[str],
    top_k: int,
) -> None:
    data_vocab = set(freq.keys())
    in_dict = data_vocab & dictionary
    oov_vocab = data_vocab - dictionary
    unused = dictionary - data_vocab

    print(f"=== {name} vs dictionary.txt ===")
    print(f"Dictionary size:     {len(dictionary)}")
    print(f"Data vocab size:     {len(data_vocab)}")
    print(f"Tokens in both:      {len(in_dict)}")
    print(f"OOV token types:     {len(oov_vocab)}")
    print(f"Dict tokens unused:  {len(unused)}")
    print(f"Lines scanned:       {total_lines:,}")
    print(f"Lines with OOV:      {oov_lines:,} ({100 * oov_lines / max(total_lines, 1):.2f}%)")
    print(f"Total token count:   {total_tokens:,}")
    print(
        f"OOV token instances: {oov_instances:,} "
        f"({100 * oov_instances / max(total_tokens, 1):.2f}%)"
    )
    print()
    print(f"Top {top_k} OOV tokens:")
    oov_freq = Counter({k: v for k, v in freq.items() if k not in dictionary})
    for tok, count in oov_freq.most_common(top_k):
        print(f"  {repr(tok):30} {count:>10,}")
    print()
    print("Unused dictionary tokens:")
    print("  " + ", ".join(sorted(unused)))


def check_formulas(path: Path, dictionary: Set[str], limit: Optional[int], top_k: int) -> None:
    freq: Counter[str] = Counter()
    total_lines = 0
    oov_lines = 0
    total_tokens = 0
    oov_instances = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            total_lines += 1
            if limit is not None and total_lines > limit:
                break
            tokens = tokenize(line)
            if not tokens:
                continue
            if any(t not in dictionary for t in tokens):
                oov_lines += 1
            for token in tokens:
                freq[token] += 1
                total_tokens += 1
                if token not in dictionary:
                    oov_instances += 1

    report(path.name, freq, total_tokens, oov_instances, oov_lines, total_lines, dictionary, top_k)


def check_csv(path: Path, dictionary: Set[str], limit: Optional[int], top_k: int) -> None:
    freq: Counter[str] = Counter()
    total_lines = 0
    oov_lines = 0
    total_tokens = 0
    oov_instances = 0

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_lines += 1
            if limit is not None and total_lines > limit:
                break
            tokens = tokenize(row.get("latex") or "")
            if not tokens:
                continue
            if any(t not in dictionary for t in tokens):
                oov_lines += 1
            for token in tokens:
                freq[token] += 1
                total_tokens += 1
                if token not in dictionary:
                    oov_instances += 1

    report(path.name, freq, total_tokens, oov_instances, oov_lines, total_lines, dictionary, top_k)


def print_in_vocab_equations(
    path: Path,
    dictionary: Set[str],
    num_print: int,
    scan_limit: Optional[int],
    split: bool,
    source: str,
) -> None:
    import sys

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from read_latex import (
        iter_csv_records,
        iter_formula_lines,
        split_equation_line,
        tokenize_latex,
    )

    printed = 0
    scanned = 0
    matched = 0

    def check_and_print(label: str, text: str) -> bool:
        nonlocal printed, matched
        candidates = split_equation_line(text) if split else [text]
        for part in candidates:
            tokens = tokenize_latex(part)
            if not tokens or has_oov(tokens, dictionary):
                continue
            matched += 1
            printed += 1
            print(f"[{printed}] {label}")
            print(part)
            print()
            if printed >= num_print:
                return True
        return False

    if source == "formulas":
        for line in iter_formula_lines(path):
            scanned += 1
            if scan_limit is not None and scanned > scan_limit:
                break
            if check_and_print(f"line {scanned}", line):
                break
    else:
        for record in iter_csv_records(path):
            scanned += 1
            if scan_limit is not None and scanned > scan_limit:
                break
            if check_and_print(record.image_filename, record.latex):
                break

    print(
        f"Found {matched} in-vocab equation(s); printed {printed} "
        f"(scanned {scanned} {source} row(s))"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check LaTeX vocab against dictionary.txt")
    parser.add_argument(
        "--source",
        choices=("formulas", "csv", "both"),
        default="formulas",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Max lines/rows to scan")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument(
        "--print-in-vocab",
        type=int,
        default=None,
        metavar="N",
        help="Print N equations whose tokens are all in dictionary.txt",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="With --print-in-vocab, check sub-equations after comma/spacing split",
    )
    args = parser.parse_args()

    dictionary = load_dictionary(args.data_dir / "dictionary.txt")
    if args.print_in_vocab is not None:
        if args.source == "both":
            raise SystemExit("--print-in-vocab requires --source formulas or csv")
        data_path = (
            args.data_dir / "final_png_formulas.txt"
            if args.source == "formulas"
            else args.data_dir / "printed_mathematical_expressions_train"
        )
        print_in_vocab_equations(
            data_path,
            dictionary,
            args.print_in_vocab,
            args.limit,
            args.split,
            args.source,
        )
        return

    if args.source in ("formulas", "both"):
        check_formulas(args.data_dir / "final_png_formulas.txt", dictionary, args.limit, args.top_k)
        if args.source == "both":
            print()
    if args.source in ("csv", "both"):
        check_csv(args.data_dir / "printed_mathematical_expressions_train", dictionary, args.limit, args.top_k)


if __name__ == "__main__":
    main()
