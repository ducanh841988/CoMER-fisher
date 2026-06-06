#!/usr/bin/env python
"""Randomly render sample images with captions from a CoMER dataset folder."""

from __future__ import annotations

import argparse
import math
import random
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_pairs(data_dir: Path) -> List[Tuple[str, Path, str]]:
    """Load (sample_id, image_path, caption_text) from caption.txt + img/."""
    caption_path = data_dir / "caption.txt"
    img_dir = data_dir / "img"
    if not caption_path.is_file():
        raise FileNotFoundError(f"caption.txt not found: {caption_path}")
    if not img_dir.is_dir():
        raise FileNotFoundError(f"img/ not found: {img_dir}")

    pairs: List[Tuple[str, Path, str]] = []
    for line in caption_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        sample_id = parts[0]
        caption = " ".join(parts[1:])
        img_path = img_dir / f"{sample_id}.bmp"
        if not img_path.is_file():
            for ext in (".png", ".jpg", ".jpeg"):
                alt = img_dir / f"{sample_id}{ext}"
                if alt.is_file():
                    img_path = alt
                    break
            else:
                continue
        pairs.append((sample_id, img_path, caption))
    return pairs


def _grid_shape(n: int, cols: int) -> Tuple[int, int]:
    cols = max(1, min(cols, n))
    rows = math.ceil(n / cols)
    return rows, cols


def render_samples(
    pairs: List[Tuple[str, Path, str]],
    n: int,
    *,
    seed: Optional[int] = None,
    cols: int = 3,
    output: Optional[Path] = None,
    show: bool = False,
    wrap: int = 60,
) -> None:
    if not pairs:
        raise SystemExit("No valid image/caption pairs found.")
    n = min(n, len(pairs))
    rng = random.Random(seed)
    chosen = rng.sample(pairs, n)

    rows, cols = _grid_shape(n, cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if n == 1:
        axes_list = [axes]
    else:
        axes_list = list(axes.ravel())

    for (sample_id, img_path, caption), ax in zip(chosen, axes_list):
        with Image.open(img_path) as img:
            ax.imshow(img.convert("RGB"))
        title = f"{sample_id}\n{textwrap.fill(caption, width=wrap)}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes_list[n:]:
        ax.axis("off")

    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved {n} samples to {output.resolve()}")
    if show:
        plt.show()
    elif output is None:
        default_out = Path("sample_render.png")
        fig.savefig(default_out, dpi=150, bbox_inches="tight")
        print(f"Saved {n} samples to {default_out.resolve()}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomly render images with captions from a dataset folder."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder with caption.txt and img/ (e.g. preprocessing/output/dataset/data/train)",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=20,
        help="Number of random samples to render (default: 9)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=3,
        help="Number of columns in the figure grid (default: 3)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG path (default: sample_render.png in cwd)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open interactive matplotlib window (if display available)",
    )
    args = parser.parse_args()

    pairs = load_pairs(args.data_dir.resolve())
    print(f"Loaded {len(pairs)} pairs from {args.data_dir}")
    render_samples(
        pairs,
        args.num_samples,
        seed=args.seed,
        cols=args.cols,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
