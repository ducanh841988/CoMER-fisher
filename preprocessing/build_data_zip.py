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

from preprocessing.paths import COMER_ZIP_FOLDER_MAP, DATASET_OUTPUT_LAYOUT

# (path under output/dataset, name inside data.zip)
DEFAULT_PACK_PAIRS: Tuple[Tuple[str, str], ...] = tuple(
    (src, COMER_ZIP_FOLDER_MAP[src])
    for src in DATASET_OUTPUT_LAYOUT
    if src in COMER_ZIP_FOLDER_MAP
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
) -> tuple[List[str], int]:
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
    pack_pairs: Iterable[Tuple[str, str]] = DEFAULT_PACK_PAIRS,
    dictionary: Optional[Path] = None,
) -> dict:
    """Zip dataset folders into CoMER ``data/{train,2014,2019,2023}/`` layout."""
    dataset_root = dataset_root.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    vocab = _load_vocab(dictionary)
    stats: Dict[str, object] = {"folders": {}, "images": 0, "skipped_oov": 0}

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for src_rel, zip_name in pack_pairs:
            split_dir = dataset_root / src_rel
            if not split_dir.is_dir():
                continue
            caption_path = split_dir / "caption.txt"
            img_dir = split_dir / "img"
            if not caption_path.is_file():
                continue

            lines, skipped = _filter_caption_lines(caption_path, vocab)
            stats["skipped_oov"] = int(stats["skipped_oov"]) + skipped
            if not lines:
                continue

            allowed_ids = {line.split()[0] for line in lines if line.strip()}
            archive.writestr(f"data/{zip_name}/caption.txt", "".join(lines))

            written = 0
            if img_dir.is_dir():
                for bmp_path in sorted(img_dir.glob("*.bmp")):
                    if bmp_path.stem not in allowed_ids:
                        continue
                    arcname = f"data/{zip_name}/img/{bmp_path.name}"
                    archive.write(bmp_path, arcname)
                    written += 1

            stats["folders"][zip_name] = {"captions": len(lines), "images": written}
            stats["images"] = int(stats["images"]) + written

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack preprocessing/output/dataset into CoMER data.zip."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "dataset",
        help="Pipeline output (train/, val/, test/2019/, test/2023/)",
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
    args = parser.parse_args()

    stats = build_data_zip(
        dataset_root=args.dataset,
        output_zip=args.output,
        dictionary=args.dictionary.resolve() if args.dictionary else None,
    )
    print(f"Wrote {args.output.resolve()}")
    for folder, counts in stats["folders"].items():
        print(f"  data/{folder}: captions={counts['captions']} images={counts['images']}")
    if stats["skipped_oov"]:
        print(f"  skipped_oov={stats['skipped_oov']}")


if __name__ == "__main__":
    main()
