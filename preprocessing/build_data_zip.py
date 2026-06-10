"""Pack unified pipeline output into CoMER data.zip format."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.paths import (
    DATASET_OUTPUT_LAYOUT,
    discover_pack_sources,
    migrate_dataset_layout,
)


def _load_vocab(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    words: Set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            token = line.strip()
            if token:
                words.add(token)
    return words


def _filter_caption_lines(
    caption_path: Path, vocab: Optional[Set[str]]
) -> tuple:
    """Return kept lines and count of skipped OOV samples."""
    lines = caption_path.read_text(encoding="utf-8").splitlines(keepends=True)
    if vocab is None:
        return lines, 0
    kept: List[str] = []
    skipped = 0
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            skipped += 1
            continue
        tokens = parts[1:]
        if any(t not in vocab for t in tokens):
            skipped += 1
            continue
        kept.append(line if line.endswith("\n") else line + "\n")
    return kept, skipped


def build_data_zip(
    dataset_root: Path,
    output_zip: Path,
    sources: Optional[Iterable[Tuple[Path, str]]] = None,
    dictionary: Optional[Path] = None,
) -> dict:
    """Zip ``data/{train,val,2019,2023}/`` (same layout as on disk)."""
    dataset_root = dataset_root.resolve()
    pack_list = list(sources) if sources is not None else discover_pack_sources(dataset_root)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    vocab = _load_vocab(dictionary)
    stats: Dict[str, object] = {"folders": {}, "images": 0, "skipped_oov": 0}

    if not pack_list:
        return stats

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for split_dir, arc_prefix in pack_list:
            caption_path = split_dir / "caption.txt"
            img_dir = split_dir / "img"
            if not caption_path.is_file():
                continue

            lines, skipped = _filter_caption_lines(caption_path, vocab)
            stats["skipped_oov"] = int(stats["skipped_oov"]) + skipped
            if not lines:
                continue

            allowed_ids = {line.split()[0] for line in lines if line.strip()}
            archive.writestr(f"{arc_prefix}/caption.txt", "".join(lines))

            written = 0
            if img_dir.is_dir():
                for bmp_path in sorted(img_dir.glob("*.bmp")):
                    if bmp_path.stem not in allowed_ids:
                        continue
                    archive.write(bmp_path, f"{arc_prefix}/img/{bmp_path.name}")
                    written += 1

            stats["folders"][arc_prefix] = {"captions": len(lines), "images": written}
            stats["images"] = int(stats["images"]) + written

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack dataset into CoMER data.zip (paths match on-disk layout)."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "dataset",
        help="Root containing data/train, data/val, data/2019, data/2023",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data.zip",
        help="Output zip path",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default=None,
        help="Optional comer/datamodule/dictionary.txt to drop OOV samples",
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        default=True,
        help="Move legacy train/val/test/* into data/* before packing (default: on)",
    )
    parser.add_argument(
        "--no-migrate",
        action="store_false",
        dest="migrate",
        help="Do not move legacy folders; require data/* layout",
    )
    parser.add_argument(
        "--migrate-only",
        action="store_true",
        help="Only run layout migration, do not create zip",
    )
    args = parser.parse_args()

    dataset_root = args.dataset.resolve()
    if args.migrate:
        moved = migrate_dataset_layout(dataset_root)
        for line in moved:
            print(f"Migrated: {line}")

    if args.migrate_only:
        return

    sources = discover_pack_sources(dataset_root)
    if not sources:
        expected = ", ".join(DATASET_OUTPUT_LAYOUT)
        raise SystemExit(
            f"No caption.txt under {dataset_root}\n"
            f"Expected folders: {expected}\n"
            f"Run: bash preprocessing/make_dataset.sh\n"
            f"Or migrate legacy output: bash preprocessing/make_dataset.sh --zip --migrate-only"
        )

    stats = build_data_zip(
        dataset_root=dataset_root,
        output_zip=args.output,
        sources=sources,
        dictionary=args.dictionary.resolve() if args.dictionary else None,
    )
    if not stats["folders"]:
        raise SystemExit("No samples packed (empty captions or missing images).")

    print(f"Wrote {args.output.resolve()}")
    for folder, counts in stats["folders"].items():
        print(f"  {folder}: captions={counts['captions']} images={counts['images']}")
    if stats["skipped_oov"]:
        print(f"  skipped_oov={stats['skipped_oov']}")


if __name__ == "__main__":
    main()
