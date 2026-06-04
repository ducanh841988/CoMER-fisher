"""Convert SymLG (.lg) label graphs to space-separated LaTeX."""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from preprocessing.batch_utils import default_workers, run_parallel, summarize_results
from preprocessing.lg_srt import SymbolRelationTree, lg_to_srt
from preprocessing.srt_to_latex import DEFAULT_SYMBOL_MAP, srt_to_latex

DEFAULT_MAP = Path(__file__).resolve().parent / "symLG_map.csv"


def load_symbol_map(map_path: Optional[Path] = None) -> Dict[str, str]:
    """Load optional symbol label overrides from ``symLG_map.csv`` SYMBOLS table."""
    symbol_map = dict(DEFAULT_SYMBOL_MAP)
    map_path = map_path or DEFAULT_MAP
    if not map_path.exists():
        return symbol_map

    reading_symbols = True
    with map_path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            key = row[0].strip()
            if key == "SYMBOLS":
                reading_symbols = True
                continue
            if key == "STRUCTURE":
                reading_symbols = False
                continue
            if not reading_symbols:
                continue
            if len(row) >= 3 and row[1].strip() == "->":
                symbol_map[key] = row[2].strip()

    return symbol_map


def latex_from_srt(
    tree: SymbolRelationTree,
    map_path: Optional[Path] = None,
) -> str:
    """Convert a symbol relation tree to LaTeX."""
    return srt_to_latex(tree, symbol_map=load_symbol_map(map_path))


def latex_from_lg(
    lg_path: Path,
    map_path: Optional[Path] = None,
) -> Tuple[Optional[str], str]:
    """Return LaTeX for a SymLG file without writing an output file."""
    try:
        tree = lg_to_srt(lg_path)
        if not tree.nodes:
            return None, f"no symbols found in {lg_path}"
        latex = latex_from_srt(tree, map_path=map_path)
        if not latex.strip():
            return None, f"empty LaTeX from {lg_path}"
        return latex, ""
    except Exception as exc:
        return None, str(exc)


def convert_lg_file(
    lg_path: Path,
    output_path: Path,
    map_path: Optional[Path] = None,
) -> Tuple[bool, Optional[str]]:
    """Write LaTeX for one SymLG file. Returns (success, latex_or_error)."""
    latex, err = latex_from_lg(lg_path, map_path=map_path)
    if latex is None:
        return False, err

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex + "\n", encoding="utf-8")
    return True, latex


def _lg_worker(paths: Tuple[str, str, str]) -> Tuple[bool, str]:
    lg_path, out_path, map_path = paths
    try:
        success, msg = convert_lg_file(
            Path(lg_path), Path(out_path), map_path=Path(map_path) if map_path else None
        )
    except Exception as exc:
        return False, f"{lg_path}: {exc}"
    if success:
        return True, ""
    return False, f"{lg_path}: {msg}"


def convert_lg_dir(
    input_dir: Path,
    output_dir: Path,
    map_path: Optional[Path] = None,
    recursive: bool = True,
    workers: Optional[int] = None,
    show_progress: bool = True,
) -> Tuple[int, int, List[str]]:
    """Convert all ``.lg`` files under ``input_dir``.

    Returns (ok_count, fail_count, error_messages).
    """
    map_path = map_path or DEFAULT_MAP
    pattern = "**/*.lg" if recursive else "*.lg"
    files = sorted(input_dir.glob(pattern))
    tasks = [
        (
            str(lg_path),
            str(output_dir / lg_path.relative_to(input_dir).with_suffix(".tex")),
            str(map_path),
        )
        for lg_path in files
    ]
    results = run_parallel(
        tasks,
        _lg_worker,
        workers=workers or default_workers(),
        desc="SymLG",
        show_progress=show_progress,
    )
    return summarize_results(results)
