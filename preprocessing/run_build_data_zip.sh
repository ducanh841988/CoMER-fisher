#!/usr/bin/env bash
# Pack pipeline output into CoMER data.zip (data/train, data/val, data/2019, data/2023)
#
# Usage (from CoMER repo root):
#   bash preprocessing/run_build_data_zip.sh
#   OUT_ZIP=data_2023.zip bash preprocessing/run_build_data_zip.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-$ROOT/preprocessing/output/dataset}"
OUT_ZIP="${OUT_ZIP:-$ROOT/data.zip}"

cd "$ROOT"
# Normalize legacy train/val/test/* -> data/* then zip (same paths on disk and in archive)
python preprocessing/build_data_zip.py --dataset "$DATASET" --output "$OUT_ZIP" "$@"
