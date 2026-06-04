"""Analyze exported image width/height distributions per split."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Pillow is required: pip install Pillow") from exc

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_MAX_IMAGE_SIZE = 1000


@dataclass
class NumericStats:
    count: int = 0
    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    stdev: float = 0.0


@dataclass
class SplitImageAnalysis:
    name: str
    num_files: int = 0
    skipped: int = 0
    width: NumericStats = field(default_factory=NumericStats)
    height: NumericStats = field(default_factory=NumericStats)
    aspect_ratio: NumericStats = field(default_factory=NumericStats)
    area: NumericStats = field(default_factory=NumericStats)
    widths: List[int] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)
    aspect_ratios: List[float] = field(default_factory=list)


def iter_image_files(root: Path, split: str) -> Iterable[Path]:
    split_dir = root / split
    if not split_dir.is_dir():
        return
    for path in sorted(split_dir.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def _numeric_stats(values: List[float]) -> NumericStats:
    if not values:
        return NumericStats()
    return NumericStats(
        count=len(values),
        min=min(values),
        max=max(values),
        mean=statistics.mean(values),
        median=statistics.median(values),
        stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
    )


def scan_images(
    root: Path,
    split: str,
    show_progress: bool = True,
) -> Tuple[List[dict], SplitImageAnalysis]:
    rows: List[dict] = []
    analysis = SplitImageAnalysis(name=split)
    paths = list(iter_image_files(root, split))
    iterator = paths
    if show_progress and tqdm is not None:
        iterator = tqdm(paths, desc=f"Scanning {split}", unit="file")

    for path in iterator:
        try:
            with Image.open(path) as img:
                width, height = img.size
            assert width > 0 and height > 0
            assert width <= DEFAULT_MAX_IMAGE_SIZE and height <= DEFAULT_MAX_IMAGE_SIZE
        except OSError as exc:
            analysis.skipped += 1
            print(f"Skip {path}: {exc}")
            continue

        aspect_ratio = width / height if height else 0.0
        area = width * height
        rows.append(
            {
                "split": split,
                "path": str(path.relative_to(root)),
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "area": area,
            }
        )
        analysis.widths.append(width)
        analysis.heights.append(height)
        analysis.aspect_ratios.append(aspect_ratio)
        analysis.num_files += 1

    analysis.width = _numeric_stats([float(v) for v in analysis.widths])
    analysis.height = _numeric_stats([float(v) for v in analysis.heights])
    analysis.aspect_ratio = _numeric_stats(analysis.aspect_ratios)
    analysis.area = _numeric_stats(
        [float(w * h) for w, h in zip(analysis.widths, analysis.heights)]
    )
    return rows, analysis


def analyze_dataset(root: Path, show_progress: bool = True) -> Tuple[List[dict], Dict[str, SplitImageAnalysis]]:
    all_rows: List[dict] = []
    analyses: Dict[str, SplitImageAnalysis] = {}
    for split in SPLITS:
        if not (root / split).is_dir():
            continue
        rows, analysis = scan_images(root, split, show_progress=show_progress)
        all_rows.extend(rows)
        analyses[split] = analysis
    return all_rows, analyses


def _print_stats(label: str, stats: NumericStats) -> None:
    print(
        f"  {label:14} count={stats.count:6d}  "
        f"min={stats.min:6.0f}  max={stats.max:6.0f}  "
        f"mean={stats.mean:8.1f}  median={stats.median:8.1f}"
    )


def print_report(analyses: Dict[str, SplitImageAnalysis]) -> None:
    for split, data in analyses.items():
        print(f"=== {split} ===")
        print(f"  files:   {data.num_files}")
        print(f"  skipped: {data.skipped}")
        _print_stats("width", data.width)
        _print_stats("height", data.height)
        _print_stats("aspect_ratio", data.aspect_ratio)
        _print_stats("area", data.area)
        print()


def save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "path", "width", "height", "aspect_ratio", "area"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_json(analyses: Dict[str, SplitImageAnalysis], path: Path) -> None:
    payload = {"splits": {}}
    for split, data in analyses.items():
        payload["splits"][split] = {
            "num_files": data.num_files,
            "skipped": data.skipped,
            "width": asdict(data.width),
            "height": asdict(data.height),
            "aspect_ratio": asdict(data.aspect_ratio),
            "area": asdict(data.area),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _plot_histogram(
    values: List[float],
    xlabel: str,
    title: str,
    path: Path,
    bins: int = 50,
) -> None:
    if not values:
        return
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, edgecolor="black", alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close()


def save_plots(analyses: Dict[str, SplitImageAnalysis], out_dir: Path, bins: int = 50) -> None:
    combined_widths: List[int] = []
    combined_heights: List[int] = []
    combined_aspects: List[float] = []

    for data in analyses.values():
        combined_widths.extend(data.widths)
        combined_heights.extend(data.heights)
        combined_aspects.extend(data.aspect_ratios)

    _plot_histogram(
        [float(v) for v in combined_widths],
        "Width (px)",
        "Image width distribution",
        out_dir / "image_width_hist.png",
        bins=bins,
    )
    _plot_histogram(
        [float(v) for v in combined_heights],
        "Height (px)",
        "Image height distribution",
        out_dir / "image_height_hist.png",
        bins=bins,
    )
    _plot_histogram(
        combined_aspects,
        "Width / Height",
        "Aspect ratio distribution",
        out_dir / "image_aspect_ratio_hist.png",
        bins=bins,
    )

    if combined_widths and combined_heights:
        plt.figure(figsize=(7, 7))
        plt.scatter(combined_widths, combined_heights, alpha=0.25, s=8)
        plt.xlabel("Width (px)")
        plt.ylabel("Height (px)")
        plt.title("Image size distribution")
        plt.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "image_size_scatter.png", dpi=150)
        plt.close()

    for split, data in analyses.items():
        split_dir = out_dir / split
        _plot_histogram(
            [float(v) for v in data.widths],
            "Width (px)",
            f"{split} width distribution",
            split_dir / "width_hist.png",
            bins=bins,
        )
        _plot_histogram(
            [float(v) for v in data.heights],
            "Height (px)",
            f"{split} height distribution",
            split_dir / "height_hist.png",
            bins=bins,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan exported images and plot width/height distributions."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "img",
        help="Image root with train/, val/, test/ subfolders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "analysis",
        help="Directory for CSV/JSON/plots",
    )
    parser.add_argument("--bins", type=int, default=50, help="Histogram bins")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    args = parser.parse_args()

    root = args.input.resolve()
    if not root.is_dir():
        raise SystemExit(f"Input root not found: {root}")

    rows, analyses = analyze_dataset(root, show_progress=not args.no_progress)
    if not rows:
        raise SystemExit(f"No images found under {root}/{{train,val,test}}")

    print_report(analyses)

    out = args.output.resolve()
    save_csv(rows, out / "image_sizes.csv")
    save_json(analyses, out / "image_stats.json")
    save_plots(analyses, out / "image_plots", bins=args.bins)

    print(f"Saved reports to: {out}")
    print("  image_sizes.csv")
    print("  image_stats.json")
    print("  image_plots/image_width_hist.png")
    print("  image_plots/image_height_hist.png")
    print("  image_plots/image_aspect_ratio_hist.png")
    print("  image_plots/image_size_scatter.png")


if __name__ == "__main__":
    main()
