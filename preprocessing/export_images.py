"""Export images: copy from IMG folder or render from INKML."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.batch_utils import default_workers, run_parallel, summarize_results
from preprocessing.image_crop import crop_to_content, scale_to_max_size
from preprocessing.inkml_to_image import (
    DEFAULT_MAX_IMAGE_SIZE,
    render_inkml_to_image,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional at runtime
    tqdm = None

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = (".png", ".bmp", ".jpg", ".jpeg")


def find_image(stem: str, search_dir: Path) -> Optional[Path]:
    """Find an image file matching ``stem`` anywhere under ``search_dir``."""
    if not search_dir.is_dir():
        return None
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(search_dir.rglob(f"{stem}{ext}"))
        if matches:
            return matches[0]
    return None


def _discover_split_dirs(root: Path, splits: Iterable[str]) -> List[Tuple[str, Path]]:
    """Return ``(split_name, directory)`` pairs under ``root``."""
    found = [(split, root / split) for split in splits if (root / split).is_dir()]
    if found:
        return found
    if any(root.rglob("*.inkml")) or any(
        root.rglob(f"*{ext}") for ext in IMAGE_EXTENSIONS
    ):
        return [("custom", root)]
    return []


def iter_inkml_files(inkml_root: Path, split: str) -> Iterable[Tuple[Path, Path]]:
    """Yield (inkml_path, relative_parent_dir) under one split."""
    split_dir = inkml_root / split if (inkml_root / split).is_dir() else inkml_root
    if split != "custom" and not (inkml_root / split).is_dir():
        return
    if not split_dir.is_dir():
        return
    base = split_dir if split == "custom" else inkml_root / split
    for inkml_path in sorted(base.rglob("*.inkml")):
        rel_parent = inkml_path.parent.relative_to(base)
        yield inkml_path, rel_parent


def iter_image_files(img_root: Path, split: str) -> Iterable[Tuple[Path, Path]]:
    """Yield (image_path, relative_parent_dir) under one split."""
    split_dir = img_root / split if (img_root / split).is_dir() else img_root
    if split != "custom" and not (img_root / split).is_dir():
        return
    if not split_dir.is_dir():
        return
    base = split_dir if split == "custom" else img_root / split
    seen: set[Path] = set()
    for ext in IMAGE_EXTENSIONS:
        for img_path in sorted(base.rglob(f"*{ext}")):
            if img_path in seen:
                continue
            seen.add(img_path)
            rel_parent = img_path.parent.relative_to(base)
            yield img_path, rel_parent


def export_one(
    output_path: Path,
    inkml_path: Optional[Path],
    img_path: Optional[Path],
    dpi: int,
    line_width: float,
    padding: float,
    crop: bool = True,
    max_image_size: float = DEFAULT_MAX_IMAGE_SIZE,
) -> Tuple[bool, str]:
    """Copy ``img_path`` or render ``inkml_path`` to ``output_path``."""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        margin = max(0, int(round(padding)))
        if img_path is not None and img_path.is_file():
            if crop:
                crop_to_content(
                    img_path,
                    output_path,
                    margin=margin,
                    max_image_size=max_image_size,
                )
            else:
                shutil.copy2(img_path, output_path)
                scale_to_max_size(output_path, max_image_size)
            return True, "copied+cropped" if crop else "copied"
        if inkml_path is not None and inkml_path.is_file():
            render_inkml_to_image(
                inkml_path,
                output_path,
                dpi=dpi,
                line_width=line_width,
                padding=padding,
                max_image_size=max_image_size,
            )
            return True, "rendered"
        if inkml_path is not None:
            return False, f"no image or inkml source for {inkml_path.stem}"
        return False, f"no image source for {output_path.stem}"
    except Exception as exc:
        return False, str(exc)


def _export_worker(args: Tuple) -> Tuple[bool, str]:
    return export_one(*args)


def _progress(iterable, desc: str, show_progress: bool):
    if show_progress and tqdm is not None:
        return tqdm(iterable, desc=desc, unit="file")
    return iterable


def build_tasks_from_inkml(
    inkml_root: Path,
    img_root: Optional[Path],
    output_root: Path,
    splits: Iterable[str],
    dpi: int,
    line_width: float,
    padding: float,
    show_progress: bool = True,
    crop: bool = True,
    inkml_only: bool = False,
    max_image_size: float = DEFAULT_MAX_IMAGE_SIZE,
) -> List[Tuple]:
    tasks: List[Tuple] = []
    split_dirs = _discover_split_dirs(inkml_root, splits)
    img_candidates = (
        []
        if inkml_only
        else (_discover_split_dirs(img_root, splits) if img_root else [])
    )

    inkml_items: List[Tuple[str, Path, Path]] = []
    for split, _ in split_dirs:
        for inkml_path, rel_parent in iter_inkml_files(inkml_root, split):
            inkml_items.append((split, inkml_path, rel_parent))

    for split, inkml_path, rel_parent in _progress(
        inkml_items, "Scanning INKML", show_progress
    ):
        img_split = None
        if img_candidates:
            img_split = next((p for s, p in img_candidates if s == split), None)
            if img_split is None:
                img_split = img_candidates[0][1]
        stem = inkml_path.stem
        out_path = output_root / split / rel_parent / f"{stem}.png"
        if split == "custom":
            out_path = output_root / rel_parent / f"{stem}.png"
        img_path = find_image(stem, img_split) if img_split else None
        tasks.append(
            (out_path, inkml_path, img_path, dpi, line_width, padding, crop, max_image_size)
        )
    return tasks


def build_tasks_from_img_only(
    img_root: Path,
    output_root: Path,
    splits: Iterable[str],
    dpi: int,
    line_width: float,
    padding: float,
    show_progress: bool = True,
    crop: bool = True,
    max_image_size: float = DEFAULT_MAX_IMAGE_SIZE,
) -> List[Tuple]:
    tasks: List[Tuple] = []
    seen: set[Path] = set()
    img_items: List[Tuple[str, Path, Path]] = []
    for split, _ in _discover_split_dirs(img_root, splits):
        for img_path, rel_parent in iter_image_files(img_root, split):
            img_items.append((split, img_path, rel_parent))

    for split, img_path, rel_parent in _progress(
        img_items, "Scanning IMG", show_progress
    ):
        out_path = output_root / split / rel_parent / f"{img_path.stem}.png"
        if split == "custom":
            out_path = output_root / rel_parent / f"{img_path.stem}.png"
        if out_path in seen:
            continue
        seen.add(out_path)
        tasks.append(
            (out_path, None, img_path, dpi, line_width, padding, crop, max_image_size)
        )
    return tasks


def export_images(
    output_root: Path,
    inkml_root: Optional[Path] = None,
    img_root: Optional[Path] = None,
    splits: Iterable[str] = SPLITS,
    workers: Optional[int] = None,
    show_progress: bool = True,
    img_only: bool = False,
    inkml_only: bool = False,
    dpi: int = 150,
    line_width: float = 2.0,
    padding: float = 5.0,
    crop: bool = True,
    max_image_size: float = DEFAULT_MAX_IMAGE_SIZE,
) -> Tuple[int, int, List[str]]:
    """Export images to ``output_root``. Returns (ok, fail, errors)."""
    if img_only and inkml_only:
        raise ValueError("img_only and inkml_only cannot be used together")
    if img_only:
        if img_root is None:
            raise ValueError("img_root is required for img_only mode")
        tasks = build_tasks_from_img_only(
            img_root,
            output_root,
            splits,
            dpi,
            line_width,
            padding,
            show_progress=show_progress,
            crop=crop,
            max_image_size=max_image_size,
        )
    else:
        if inkml_root is None:
            raise ValueError("inkml_root is required unless img_only is set")
        tasks = build_tasks_from_inkml(
            inkml_root,
            img_root,
            output_root,
            splits,
            dpi,
            line_width,
            padding,
            show_progress=show_progress,
            crop=crop,
            inkml_only=inkml_only,
            max_image_size=max_image_size,
        )

    if show_progress and tqdm is not None:
        tqdm.write(f"Exporting {len(tasks)} images with {workers or default_workers()} workers")

    results = run_parallel(
        tasks,
        _export_worker,
        workers=workers or default_workers(),
        desc="Exporting images",
        show_progress=show_progress,
    )
    return summarize_results(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy images from IMG or render from INKML into output folders."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "img",
        help="Output root (creates train/val/test subfolders)",
    )
    parser.add_argument(
        "--inkml-root",
        type=Path,
        help="TC11 INKML root (e.g. DATA/INKML). Uses render when no PNG is found.",
    )
    parser.add_argument(
        "--img-root",
        type=Path,
        help="TC11 IMG root (e.g. DATA/IMG). PNG is copied when found by filename.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        help="TC11_CROHME23 root; sets --inkml-root and --img-root automatically",
    )
    parser.add_argument(
        "--img-only",
        action="store_true",
        help="Copy all images from IMG only (no INKML rendering)",
    )
    parser.add_argument(
        "--inkml-only",
        action="store_true",
        help="Render all images from INKML only (skip IMG lookup/copy)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help=f"Parallel workers (default: {default_workers()})",
    )
    parser.add_argument("--dpi", type=int, default=150, help="DPI for INKML rendering")
    parser.add_argument("--line-width", type=float, default=2.0, help="Stroke width")
    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Margin for INKML render and IMG crop (default: 5)",
    )
    parser.add_argument(
        "--max-image-size",
        type=float,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help="Max INKML equation size in ink units before scale-down (default: 1000)",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Copy IMG files as-is without cropping blank margins",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    data_root = args.data_root
    inkml_root = args.inkml_root
    img_root = args.img_root
    if data_root:
        data_root = data_root.resolve()
        inkml_root = inkml_root or data_root / "INKML"
        if not args.inkml_only:
            img_root = img_root or data_root / "IMG"

    if args.img_only and args.inkml_only:
        raise SystemExit("Use either --img-only or --inkml-only, not both")

    if not args.img_only and inkml_root is None:
        raise SystemExit("Provide --inkml-root or --data-root (or use --img-only with --img-root)")

    output_root = args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ok, fail, errors = export_images(
        output_root=output_root,
        inkml_root=inkml_root.resolve() if inkml_root else None,
        img_root=img_root.resolve() if img_root else None,
        workers=args.workers,
        show_progress=not args.no_progress,
        img_only=args.img_only,
        inkml_only=args.inkml_only,
        dpi=args.dpi,
        line_width=args.line_width,
        padding=args.padding,
        crop=not args.no_crop,
        max_image_size=args.max_image_size,
    )

    print(f"Done: {ok} exported, {fail} failed -> {output_root}")
    if errors:
        print("First errors:")
        for err in errors[:10]:
            print(f"  {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()
