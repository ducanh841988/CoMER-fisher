"""Render INKML stroke files to PNG images."""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

INKML_NS = {"ink": "http://www.w3.org/2003/InkML"}
TRACE_RE = re.compile(r"<trace\b[^>]*>(.*?)</trace>", re.DOTALL | re.IGNORECASE)
Point = Tuple[float, float]
Stroke = List[Point]


def _parse_point_pairs(text: str) -> Stroke:
    points: Stroke = []
    for segment in text.split(","):
        parts = segment.strip().split()
        if len(parts) < 2:
            continue
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError:
            continue
    return points


def _parse_traces_regex(text: str) -> List[Stroke]:
    strokes: List[Stroke] = []
    for match in TRACE_RE.finditer(text):
        stroke = _parse_point_pairs(match.group(1))
        if len(stroke) >= 2:
            strokes.append(stroke)
    return strokes


def parse_inkml_strokes(path: Path) -> List[Stroke]:
    """Extract stroke polylines from an INKML file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    strokes: List[Stroke] = []

    try:
        root = ET.fromstring(text)
        for trace in root.findall(".//ink:trace", INKML_NS):
            if trace.text:
                stroke = _parse_point_pairs(trace.text)
                if len(stroke) >= 2:
                    strokes.append(stroke)
        if not strokes:
            for trace in root.iter("trace"):
                if trace.text:
                    stroke = _parse_point_pairs(trace.text)
                    if len(stroke) >= 2:
                        strokes.append(stroke)
    except ET.ParseError:
        strokes = _parse_traces_regex(text)

    if not strokes:
        strokes = _parse_traces_regex(text)

    return strokes


def _bounds(strokes: Sequence[Stroke]) -> Tuple[float, float, float, float]:
    xs = [x for stroke in strokes for x, _ in stroke]
    ys = [y for stroke in strokes for _, y in stroke]
    return min(xs), min(ys), max(xs), max(ys)


TARGET_AVG_HEIGHT = 30.0
DEFAULT_MAX_IMAGE_SIZE = 1000.0


def _avg_stroke_height(strokes: Sequence[Stroke]) -> float:
    """Mean vertical span of each stroke."""
    heights: List[float] = []
    for stroke in strokes:
        ys = [y for _, y in stroke]
        if len(ys) < 2:
            continue
        heights.append(max(ys) - min(ys))
    if not heights:
        return 1.0
    return sum(heights) / len(heights)


def _scale_strokes(strokes: Sequence[Stroke], scale: float) -> List[Stroke]:
    return [[(x * scale, y * scale) for x, y in stroke] for stroke in strokes]


def _normalize_avg_height(
    strokes: List[Stroke],
    width: float,
    height: float,
    target_avg_height: float = TARGET_AVG_HEIGHT,
) -> Tuple[List[Stroke], float, float]:
    """Scale strokes so the average stroke height equals ``target_avg_height``."""
    avg_h = _avg_stroke_height(strokes)
    if avg_h <= 0:
        return strokes, width, height
    scale = target_avg_height / avg_h
    scaled = _scale_strokes(strokes, scale)
    return scaled, width * scale, height * scale


def _scale_to_max_size(
    strokes: List[Stroke],
    width: float,
    height: float,
    max_image_size: float,
) -> Tuple[List[Stroke], float, float]:
    """Uniformly scale down so ``max(width, height)`` does not exceed ``max_image_size``."""
    if max_image_size <= 0:
        return strokes, width, height
    longest = max(width, height)
    if longest <= max_image_size:
        return strokes, width, height
    scale = max_image_size / longest
    scaled = _scale_strokes(strokes, scale)
    return scaled, width * scale, height * scale


def _translate_strokes_to_origin(
    strokes: Sequence[Stroke],
) -> Tuple[List[Stroke], float, float]:
    """Shift strokes so the bounding box top-left is at (0, 0)."""
    x_min, y_min, x_max, y_max = _bounds(strokes)
    translated = [
        [(x - x_min, y - y_min) for x, y in stroke]
        for stroke in strokes
    ]
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    return translated, width, height


def render_inkml_to_image(
    inkml_path: Path,
    output_path: Path,
    dpi: int = 150,
    line_width: float = 2.0,
    padding: float = 5.0,
    target_avg_height: float = TARGET_AVG_HEIGHT,
    max_image_size: float = DEFAULT_MAX_IMAGE_SIZE,
    background: str = "black",
    stroke_color: str = "white",
) -> Path:
    """Render one INKML file to a PNG image."""
    strokes = parse_inkml_strokes(inkml_path)
    if not strokes:
        raise ValueError(f"no drawable strokes found in {inkml_path}")

    margin = padding
    strokes, width, height = _translate_strokes_to_origin(strokes)
    strokes, width, height = _normalize_avg_height(strokes, width, height, target_avg_height)
    strokes, width, height = _scale_to_max_size(strokes, width, height, max_image_size)

    fig_w = max((width + 2 * margin) / 100.0, 1.0)
    fig_h = max((height + 2 * margin) / 100.0, 1.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    segments = [stroke for stroke in strokes if len(stroke) >= 2]
    lc = LineCollection(segments, colors=stroke_color, linewidths=line_width)
    ax.add_collection(lc)

    ax.set_xlim(-margin, width + margin)
    ax.set_ylim(height + margin, -margin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, facecolor=background)
    plt.close(fig)
    if max_image_size > 0:
        est_max_px = max(fig_w, fig_h) * dpi
        if est_max_px > max_image_size:
            from preprocessing.image_crop import scale_to_max_size

            scale_to_max_size(output_path, max_image_size)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an INKML file to a PNG image.")
    parser.add_argument("--input", required=True, type=Path, help="Input .inkml file")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output image path (default: same name with .png)",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Output DPI")
    parser.add_argument("--line-width", type=float, default=2.0, help="Stroke width")
    parser.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Margin on top, left, bottom, and right (default: 5)",
    )
    parser.add_argument(
        "--target-avg-height",
        type=float,
        default=TARGET_AVG_HEIGHT,
        help="Normalize mean stroke height to this value in ink units (default: 30)",
    )
    parser.add_argument(
        "--max-image-size",
        type=float,
        default=DEFAULT_MAX_IMAGE_SIZE,
        help="Max equation width/height in ink units; scale down if exceeded (default: 1000)",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    output_path = (args.output or input_path.with_suffix(".png")).resolve()

    try:
        render_inkml_to_image(
            input_path,
            output_path,
            dpi=args.dpi,
            line_width=args.line_width,
            padding=args.padding,
            target_avg_height=args.target_avg_height,
            max_image_size=args.max_image_size,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
