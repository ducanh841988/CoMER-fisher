"""Discover SymLG samples and map matching INKML / IMG files."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SPLITS = ("train", "val", "test")
DEFAULT_SPLIT_MAP = {
    "train": "train",
    "val": "val",
    "test": "test",
}
DATA_PREFIX = "data"
# SymLG test subfolders -> folder name under data/
TEST_FOLDER_MAP = {
    "CROHME2019_test": "2019",
    "CROHME2023_test": "2023",
}
# Relative paths under preprocessing/output/dataset/ (matches data.zip layout)
DATASET_OUTPUT_LAYOUT = (
    f"{DATA_PREFIX}/train",
    f"{DATA_PREFIX}/val",
    f"{DATA_PREFIX}/2019",
    f"{DATA_PREFIX}/2023",
)
# Legacy on-disk paths from older pipeline runs -> canonical layout
LEGACY_DATASET_PATHS: Tuple[Tuple[str, str], ...] = (
    ("train", f"{DATA_PREFIX}/train"),
    ("val", f"{DATA_PREFIX}/val"),
    ("test/2019", f"{DATA_PREFIX}/2019"),
    ("test/2023", f"{DATA_PREFIX}/2023"),
)


def discover_pack_sources(dataset_root: Path) -> List[Tuple[Path, str]]:
    """Find ``data/{train,val,2019,2023}`` folders (same paths used in zip)."""
    root = dataset_root.resolve()
    found: List[Tuple[Path, str]] = []
    for folder in DATASET_OUTPUT_LAYOUT:
        src_dir = root / folder
        if (src_dir / "caption.txt").is_file():
            found.append((src_dir, folder))
    return found


def migrate_dataset_layout(dataset_root: Path, dry_run: bool = False) -> List[str]:
    """Move legacy ``train/``, ``val/``, ``test/*`` into ``data/*`` (one-time)."""
    root = dataset_root.resolve()
    actions: List[str] = []
    for src_rel, dst_rel in LEGACY_DATASET_PATHS:
        src = root / src_rel
        dst = root / dst_rel
        if not src.is_dir():
            continue
        if not ((src / "caption.txt").is_file() or (src / "img").is_dir()):
            continue
        if dst.exists():
            raise SystemExit(
                f"Cannot move {src_rel} -> {dst_rel}: destination already exists.\n"
                f"Remove or merge {dst} manually, then retry."
            )
        actions.append(f"{src_rel} -> {dst_rel}")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
    if not dry_run:
        test_parent = root / "test"
        if test_parent.is_dir() and not any(test_parent.iterdir()):
            test_parent.rmdir()
    return actions


IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg")
INKML_SUFFIX = ".inkml"
LG_SUFFIX = ".lg"


@dataclass(frozen=True)
class SampleRecord:
    """One training sample keyed by a SymLG file."""

    split: str
    output_folder: str
    sample_id: str
    rel_stem: Path
    lg_path: Path
    img_path: Optional[Path]
    inkml_path: Optional[Path]


@dataclass
class _SidecarIndex:
    """Lookup by relative path (no suffix) or by filename stem."""

    by_rel_stem: Dict[Path, Path]
    by_stem: Dict[str, Path]


def _test_year_folder(rel_stem: Path) -> str:
    if not rel_stem.parts:
        return "unknown"
    top = rel_stem.parts[0]
    if top in TEST_FOLDER_MAP:
        return TEST_FOLDER_MAP[top]
    if "2023" in top:
        return "2023"
    if "2019" in top:
        return "2019"
    return "unknown"


def resolve_output_folder(
    split: str,
    rel_stem: Path,
    split_map: Optional[Dict[str, str]] = None,
) -> str:
    """Map a sample to ``data/train``, ``data/val``, ``data/2019``, or ``data/2023``."""
    split_map = split_map or DEFAULT_SPLIT_MAP
    if split == "test":
        return f"{DATA_PREFIX}/{_test_year_folder(rel_stem)}"
    name = split_map.get(split, split)
    return f"{DATA_PREFIX}/{name}"


def flat_sample_id(rel_stem: Path) -> str:
    """Flat id for CoMER caption.txt / img/{id}.bmp (unique across subfolders)."""
    return str(rel_stem).replace("\\", "/").replace("/", "__")


def _build_sidecar_index(
    split_root: Path,
    *,
    inkml: bool,
) -> _SidecarIndex:
    """Walk ``split_root`` once and index files by rel path and stem."""
    by_rel: Dict[Path, Path] = {}
    by_stem: Dict[str, Path] = {}
    if not split_root.is_dir():
        return _SidecarIndex(by_rel, by_stem)

    if inkml:
        patterns = (f"*{INKML_SUFFIX}",)
    else:
        patterns = tuple(f"*{ext}" for ext in IMAGE_EXTENSIONS)

    for pattern in patterns:
        for path in sorted(split_root.rglob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(split_root).with_suffix("")
            by_rel.setdefault(rel, path)
            by_stem.setdefault(rel.name, path)

    return _SidecarIndex(by_rel, by_stem)


def resolve_from_index(
    index: _SidecarIndex,
    rel_stem: Path,
    stem: str,
) -> Optional[Path]:
    direct = index.by_rel_stem.get(rel_stem)
    if direct is not None:
        return direct
    return index.by_stem.get(stem)


def find_by_stem(root: Path, stem: str, suffix: str) -> Optional[Path]:
    """Find first file with ``stem`` + ``suffix`` anywhere under ``root``."""
    index = _build_sidecar_index(
        root,
        inkml=suffix == INKML_SUFFIX,
    )
    return index.by_stem.get(stem)


def find_image_file(root: Path, stem: str) -> Optional[Path]:
    index = _build_sidecar_index(root, inkml=False)
    return index.by_stem.get(stem)


def resolve_sidecar(
    split_root: Path,
    rel_stem: Path,
    stem: str,
    *,
    inkml: bool = False,
    index: Optional[_SidecarIndex] = None,
) -> Optional[Path]:
    """Resolve INKML or image path: same relative path first, then search by stem."""
    if index is None:
        index = _build_sidecar_index(split_root, inkml=inkml)
    return resolve_from_index(index, rel_stem, stem)


def iter_lg_files(lg_split_root: Path) -> Iterable[Path]:
    if not lg_split_root.is_dir():
        return
    yield from sorted(lg_split_root.rglob(f"*{LG_SUFFIX}"))


def discover_samples(
    data_root: Path,
    splits: Iterable[str] = SPLITS,
    split_map: Optional[Dict[str, str]] = None,
    max_samples: Optional[int] = None,
) -> List[SampleRecord]:
    """Index all ``.lg`` files and map matching IMG (preferred) or INKML paths."""
    split_map = split_map or DEFAULT_SPLIT_MAP
    sym_root = data_root / "SymLG"
    inkml_root = data_root / "INKML"
    img_root = data_root / "IMG"

    records: List[SampleRecord] = []
    seen_ids: set[str] = set()

    for split in splits:
        lg_split = sym_root / split
        if not lg_split.is_dir():
            continue
        inkml_split = inkml_root / split
        img_split = img_root / split

        img_index = _build_sidecar_index(img_split, inkml=False)
        inkml_index = _build_sidecar_index(inkml_split, inkml=True)

        for lg_path in iter_lg_files(lg_split):
            rel_stem = lg_path.relative_to(lg_split).with_suffix("")
            stem = rel_stem.name
            sample_id = flat_sample_id(rel_stem)
            if sample_id in seen_ids:
                continue
            seen_ids.add(sample_id)

            img_path = resolve_from_index(img_index, rel_stem, stem)
            inkml_path = resolve_from_index(inkml_index, rel_stem, stem)

            output_folder = resolve_output_folder(split, rel_stem, split_map)
            records.append(
                SampleRecord(
                    split=split,
                    output_folder=output_folder,
                    sample_id=sample_id,
                    rel_stem=rel_stem,
                    lg_path=lg_path,
                    img_path=img_path,
                    inkml_path=inkml_path,
                )
            )
            if max_samples is not None and len(records) >= max_samples:
                return records

    return records
