"""Unified preprocessing: SymLG → LaTeX + image export → CoMER caption.txt."""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.batch_utils import default_workers, run_parallel, summarize_results
from preprocessing.image_crop import crop_to_content, scale_to_max_size
from preprocessing.inkml_to_image import DEFAULT_MAX_IMAGE_SIZE, render_inkml_to_image
from preprocessing.lg_to_latex import DEFAULT_MAP, latex_from_lg
from preprocessing.paths import (
    DATASET_OUTPUT_LAYOUT,
    DEFAULT_SPLIT_MAP,
    SPLITS,
    SampleRecord,
    discover_samples,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass
class SplitStats:
    total: int = 0
    ok: int = 0
    fail: int = 0
    no_visual: int = 0
    used_img: int = 0
    used_inkml: int = 0


@dataclass
class PipelineReport:
    folder_stats: Dict[str, SplitStats] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def _png_to_bmp(png_path: Path, bmp_path: Path) -> None:
    with Image.open(png_path) as img:
        img.convert("RGB").save(bmp_path, format="BMP")


def _export_visual(
    record: SampleRecord,
    png_path: Path,
    dpi: int,
    line_width: float,
    padding: float,
    crop: bool,
    max_image_size: float,
) -> Tuple[bool, str, str]:
    """Write PNG for one sample. Returns (ok, message, source_kind)."""
    if record.img_path is not None and record.img_path.is_file():
        margin = max(0, int(round(padding)))
        if crop:
            crop_to_content(
                record.img_path,
                png_path,
                margin=margin,
                max_image_size=max_image_size,
            )
        else:
            shutil.copy2(record.img_path, png_path)
            scale_to_max_size(png_path, max_image_size)
        return True, "img", "img"

    if record.inkml_path is not None and record.inkml_path.is_file():
        render_inkml_to_image(
            record.inkml_path,
            png_path,
            dpi=dpi,
            line_width=line_width,
            padding=padding,
            max_image_size=max_image_size,
        )
        return True, "inkml", "inkml"

    return False, "no IMG or INKML match", ""


def _record_to_dict(record: SampleRecord) -> dict:
    return {
        "split": record.split,
        "output_folder": record.output_folder,
        "sample_id": record.sample_id,
        "rel_stem": str(record.rel_stem),
        "lg_path": str(record.lg_path),
        "img_path": str(record.img_path) if record.img_path else None,
        "inkml_path": str(record.inkml_path) if record.inkml_path else None,
    }


def _dict_to_record(data: dict) -> SampleRecord:
    return SampleRecord(
        split=data["split"],
        output_folder=data["output_folder"],
        sample_id=data["sample_id"],
        rel_stem=Path(data["rel_stem"]),
        lg_path=Path(data["lg_path"]),
        img_path=Path(data["img_path"]) if data["img_path"] else None,
        inkml_path=Path(data["inkml_path"]) if data["inkml_path"] else None,
    )


def _process_worker(args: Tuple) -> Tuple[bool, str, str, str, str, str]:
    """Returns (ok, msg, split, output_folder, sample_id, caption_line)."""
    (
        record_dict,
        output_root,
        map_path,
        dpi,
        line_width,
        padding,
        crop,
        max_image_size,
        write_tex,
    ) = args

    record = _dict_to_record(record_dict)
    map_path = Path(map_path) if map_path else DEFAULT_MAP
    output_root = Path(output_root)

    latex, err = latex_from_lg(record.lg_path, map_path=map_path)
    if latex is None:
        return False, err or "latex failed", record.split, record.output_folder, record.sample_id, ""

    out_dir = output_root / record.output_folder
    bmp_path = out_dir / "img" / f"{record.sample_id}.bmp"
    png_path = bmp_path.with_suffix(".png")
    bmp_path.parent.mkdir(parents=True, exist_ok=True)

    ok, msg, _kind = _export_visual(
        record,
        png_path,
        dpi=dpi,
        line_width=line_width,
        padding=padding,
        crop=crop,
        max_image_size=max_image_size,
    )
    if not ok:
        return False, msg, record.split, record.output_folder, record.sample_id, ""

    _png_to_bmp(png_path, bmp_path)
    if png_path.exists():
        png_path.unlink()

    if write_tex:
        tex_path = output_root / "tex" / record.split / record.rel_stem.with_suffix(".tex")
        tex_path.parent.mkdir(parents=True, exist_ok=True)
        tex_path.write_text(latex + "\n", encoding="utf-8")

    caption_line = f"{record.sample_id}\t{latex}\n"
    return True, msg, record.split, record.output_folder, record.sample_id, caption_line


def run_pipeline(
    data_root: Path,
    output_root: Path,
    splits: Iterable[str] = SPLITS,
    split_map: Optional[Dict[str, str]] = None,
    map_path: Optional[Path] = None,
    workers: Optional[int] = None,
    show_progress: bool = True,
    dpi: int = 150,
    line_width: float = 2.0,
    padding: float = 5.0,
    crop: bool = True,
    max_image_size: float = DEFAULT_MAX_IMAGE_SIZE,
    write_tex: bool = False,
    require_visual: bool = True,
    max_samples: Optional[int] = None,
) -> PipelineReport:
    """Discover LG samples, process in parallel, write caption.txt per output folder."""
    split_map = split_map or DEFAULT_SPLIT_MAP
    records = discover_samples(
        data_root,
        splits=splits,
        split_map=split_map,
        max_samples=max_samples,
    )
    report = PipelineReport()

    for record in records:
        report.folder_stats.setdefault(record.output_folder, SplitStats()).total += 1

    tasks: List[Tuple] = []
    for record in records:
        if require_visual and record.img_path is None and record.inkml_path is None:
            report.folder_stats.setdefault(record.output_folder, SplitStats()).no_visual += 1
            continue
        tasks.append(
            (
                _record_to_dict(record),
                str(output_root.resolve()),
                str((map_path or DEFAULT_MAP).resolve()),
                dpi,
                line_width,
                padding,
                crop,
                max_image_size,
                write_tex,
            )
        )

    results = run_parallel(
        tasks,
        _process_worker,
        workers=workers or default_workers(),
        desc="Pipeline",
        show_progress=show_progress,
    )

    captions: Dict[str, List[str]] = defaultdict(list)
    for success, msg, split, output_folder, sample_id, caption_line in results:
        stats = report.folder_stats.setdefault(output_folder, SplitStats())
        if success:
            stats.ok += 1
            if msg == "img":
                stats.used_img += 1
            elif msg == "inkml":
                stats.used_inkml += 1
            if caption_line:
                captions[output_folder].append(caption_line)
        else:
            stats.fail += 1
            if msg:
                report.errors.append(f"{sample_id}: {msg}")

    for folder, lines in captions.items():
        out_dir = output_root / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        caption_path = out_dir / "caption.txt"
        caption_path.write_text("".join(sorted(lines)), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess TC11: map SymLG to IMG/INKML, export images, write caption.txt."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/habku/anh_project/TC11_CROHME23"),
        help="TC11_CROHME23 root (SymLG/, INKML/, IMG/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "dataset",
        help="Output root; writes data/train, data/val, data/2019, data/2023 under it",
    )
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP, help="symLG_map.csv")
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--line-width", type=float, default=2.0)
    parser.add_argument("--padding", type=float, default=5.0)
    parser.add_argument("--max-image-size", type=float, default=DEFAULT_MAX_IMAGE_SIZE)
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--write-tex", action="store_true", help="Also save .tex under output/tex/")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Process only the first N LG samples (for testing)",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    if not (data_root / "SymLG").is_dir():
        raise SystemExit(f"SymLG not found under {data_root}")

    report = run_pipeline(
        data_root=data_root,
        output_root=args.output.resolve(),
        map_path=args.map.resolve() if args.map else None,
        workers=args.workers,
        show_progress=not args.no_progress,
        dpi=args.dpi,
        line_width=args.line_width,
        padding=args.padding,
        crop=not args.no_crop,
        max_image_size=args.max_image_size,
        write_tex=args.write_tex,
        max_samples=args.max_samples,
    )

    print(f"Output: {args.output.resolve()}")
    for folder in DATASET_OUTPUT_LAYOUT:
        stats = report.folder_stats.get(folder)
        if stats is None:
            continue
        print(
            f"  {folder}: total={stats.total} ok={stats.ok} fail={stats.fail} "
            f"no_visual={stats.no_visual} img={stats.used_img} inkml={stats.used_inkml}"
        )
    for folder, stats in sorted(report.folder_stats.items()):
        if folder in DATASET_OUTPUT_LAYOUT:
            continue
        print(
            f"  {folder}: total={stats.total} ok={stats.ok} fail={stats.fail} "
            f"no_visual={stats.no_visual} img={stats.used_img} inkml={stats.used_inkml}"
        )
    if report.errors:
        print("First errors:")
        for err in report.errors[:10]:
            print(f"  {err}")


if __name__ == "__main__":
    main()
