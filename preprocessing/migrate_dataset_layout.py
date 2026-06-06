#!/usr/bin/env python
"""Move legacy dataset folders into data/train, data/val, data/2019, data/2023."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.paths import migrate_dataset_layout


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize preprocessing/output/dataset to data/* layout."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "output" / "dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    actions = migrate_dataset_layout(args.dataset.resolve(), dry_run=args.dry_run)
    if not actions:
        print("Nothing to migrate (already using data/* layout).")
        return
    prefix = "Would migrate" if args.dry_run else "Migrated"
    for line in actions:
        print(f"{prefix}: {line}")


if __name__ == "__main__":
    main()
