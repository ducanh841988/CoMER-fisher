"""Analyze LaTeX labels (space-tokenized) for train / val / test splits."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

SPLITS = ("train", "val", "test")


@dataclass
class LengthStats:
    count: int = 0
    min: int = 0
    max: int = 0
    mean: float = 0.0
    median: float = 0.0
    total_tokens: int = 0


@dataclass
class SplitAnalysis:
    name: str
    num_files: int = 0
    empty_files: int = 0
    length: LengthStats = field(default_factory=LengthStats)
    vocab_size: int = 0
    token_frequency: Counter[str] = field(default_factory=Counter)


def tokenize_latex(text: str) -> List[str]:
    """Split LaTeX into space-separated tokens (CoMER caption format)."""
    return text.strip().split()


def iter_tex_files(root: Path, split: str) -> Iterable[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        return
    yield from sorted(split_dir.rglob("*.tex"))


def analyze_split(root: Path, split: str) -> SplitAnalysis:
    lengths: List[int] = []
    freq: Counter[str] = Counter()
    num_files = 0
    empty_files = 0

    for tex_path in iter_tex_files(root, split):
        num_files += 1
        tokens = tokenize_latex(tex_path.read_text(encoding="utf-8", errors="replace"))
        if not tokens:
            empty_files += 1
            continue
        lengths.append(len(tokens))
        freq.update(tokens)

    length_stats = LengthStats()
    if lengths:
        length_stats = LengthStats(
            count=len(lengths),
            min=min(lengths),
            max=max(lengths),
            mean=round(statistics.mean(lengths), 2),
            median=round(statistics.median(lengths), 2),
            total_tokens=sum(lengths),
        )

    return SplitAnalysis(
        name=split,
        num_files=num_files,
        empty_files=empty_files,
        length=length_stats,
        vocab_size=len(freq),
        token_frequency=freq,
    )


def analyze_dataset(
    root: Path,
    splits: Iterable[str] = SPLITS,
) -> Dict[str, SplitAnalysis]:
    return {split: analyze_split(root, split) for split in splits}


def combined_vocab(analyses: Dict[str, SplitAnalysis]) -> Set[str]:
    vocab: Set[str] = set()
    for data in analyses.values():
        vocab.update(data.token_frequency.keys())
    return vocab


def combined_frequency(analyses: Dict[str, SplitAnalysis]) -> Counter[str]:
    freq: Counter[str] = Counter()
    for data in analyses.values():
        freq.update(data.token_frequency)
    return freq


def _print_split_report(analysis: SplitAnalysis, top_k: int = 10) -> None:
    print(f"=== {analysis.name} ===")
    print(f"  files:        {analysis.num_files}")
    print(f"  empty:        {analysis.empty_files}")
    print(f"  vocab size:   {analysis.vocab_size}")
    ls = analysis.length
    if ls.count:
        print(f"  length min:   {ls.min}")
        print(f"  length max:   {ls.max}")
        print(f"  length mean:  {ls.mean}")
        print(f"  length median:{ls.median}")
        print(f"  total tokens: {ls.total_tokens}")
    else:
        print("  length:       (no non-empty files)")
    print(f"  top {top_k} tokens:")
    for token, count in analysis.token_frequency.most_common(top_k):
        print(f"    {token!r:20} {count}")
    print()


def save_json(
    analyses: Dict[str, SplitAnalysis],
    path: Path,
    top_k: Optional[int],
) -> None:
    all_vocab = combined_vocab(analyses)
    all_freq = combined_frequency(analyses)
    payload: Dict[str, object] = {
        "combined": {
            "vocab_size": len(all_vocab),
            "total_tokens": sum(all_freq.values()),
            "token_frequency": dict(
                all_freq.most_common(top_k) if top_k else all_freq.most_common()
            ),
        },
        "splits": {},
    }
    for split, data in analyses.items():
        split_payload = asdict(data)
        split_payload["token_frequency"] = dict(
            data.token_frequency.most_common(top_k)
            if top_k
            else data.token_frequency.most_common()
        )
        payload["splits"][split] = split_payload

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_vocab_frequency_csv(
    analyses: Dict[str, SplitAnalysis],
    path: Path,
    top_k: Optional[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "token", "frequency"])
        for split, data in analyses.items():
            items = (
                data.token_frequency.most_common(top_k)
                if top_k
                else data.token_frequency.most_common()
            )
            for token, count in items:
                writer.writerow([split, token, count])

        writer.writerow([])
        writer.writerow(["combined", "", ""])
        for token, count in (
            combined_frequency(analyses).most_common(top_k)
            if top_k
            else combined_frequency(analyses).most_common()
        ):
            writer.writerow(["combined", token, count])


def save_vocab_txt(analyses: Dict[str, SplitAnalysis], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in analyses.items():
        vocab_path = out_dir / f"vocab_{split}.txt"
        tokens = sorted(data.token_frequency.keys())
        vocab_path.write_text("\n".join(tokens) + ("\n" if tokens else ""), encoding="utf-8")

    all_tokens = sorted(combined_vocab(analyses))
    (out_dir / "vocab_all.txt").write_text(
        "\n".join(all_tokens) + ("\n" if all_tokens else ""), encoding="utf-8"
    )


def save_full_frequency_csv(analyses: Dict[str, SplitAnalysis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "token", "frequency"])
        for split, data in analyses.items():
            for token, count in data.token_frequency.most_common():
                writer.writerow([split, token, count])
        for token, count in combined_frequency(analyses).most_common():
            writer.writerow(["combined", token, count])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze LaTeX token length, vocabulary, and frequencies per split."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "lg",
        help="Root with train/, val/, test/ subfolders of .tex files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "analysis",
        help="Directory for JSON/CSV/vocab reports",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Limit frequency rows in summary JSON/CSV (0 = save all)",
    )
    args = parser.parse_args()

    root = args.input.resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root not found: {root}")

    top_k = args.top_k if args.top_k > 0 else None
    analyses = analyze_dataset(root)

    for split in SPLITS:
        if (root / split).is_dir():
            _print_split_report(analyses[split])
        else:
            print(f"=== {split} === (missing)\n")

    all_vocab = combined_vocab(analyses)
    print("=== combined (train + val + test) ===")
    print(f"  vocab size:   {len(all_vocab)}")
    print(f"  total tokens: {sum(combined_frequency(analyses).values())}")
    print()

    out = args.output.resolve()
    save_json(analyses, out / "latex_stats.json", top_k=top_k)
    save_vocab_frequency_csv(analyses, out / "token_frequency.csv", top_k=top_k)
    save_full_frequency_csv(analyses, out / "token_frequency_full.csv")
    save_vocab_txt(analyses, out)

    print(f"Saved reports to: {out}")
    print("  latex_stats.json")
    print("  token_frequency.csv")
    print("  token_frequency_full.csv")
    print("  vocab_train.txt / vocab_val.txt / vocab_test.txt / vocab_all.txt")


if __name__ == "__main__":
    main()
