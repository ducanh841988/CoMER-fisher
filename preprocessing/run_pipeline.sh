#!/usr/bin/env bash
# Unified preprocessing: SymLG -> LaTeX + IMG/INKML -> BMP + caption.txt
#
# Usage (from CoMER repo root):
#   bash preprocessing/run_pipeline.sh
#   DATA_ROOT=/path/to/TC11_CROHME23 bash preprocessing/run_pipeline.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/habku/anh_project/TC11_CROHME23}"
OUT_ROOT="${OUT_ROOT:-$ROOT/preprocessing/output/dataset}"
WORKERS="${WORKERS:-$(cd "$ROOT" && python -c 'from preprocessing.batch_utils import default_workers; print(default_workers())' 2>/dev/null || echo 4)}"
MAP="${MAP:-$ROOT/preprocessing/symLG_map.csv}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output)
      OUT_ROOT="$2"
      shift 2
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --write-tex)
      EXTRA_ARGS+=(--write-tex)
      shift
      ;;
    --no-crop)
      EXTRA_ARGS+=(--no-crop)
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Run unified preprocessing pipeline.

Reads SymLG/*.lg, maps to IMG or INKML by path/stem, exports CoMER-format:
  data/train, data/val, data/2019, data/2023/
  each with caption.txt and img/{sample_id}.bmp

Environment:
  DATA_ROOT, OUT_ROOT, WORKERS, MAP

Example:
  bash preprocessing/run_pipeline.sh
  bash preprocessing/run_pipeline.sh --write-tex
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d "$DATA_ROOT/SymLG" ]]; then
  echo "SymLG not found under: $DATA_ROOT" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "Data:    $DATA_ROOT"
echo "Output:  $OUT_ROOT"
echo "Workers: $WORKERS"
echo "Map:     $MAP"
echo

python preprocessing/pipeline.py \
  --data-root "$DATA_ROOT" \
  --output "$OUT_ROOT" \
  --map "$MAP" \
  --workers "$WORKERS" \
  "${EXTRA_ARGS[@]}"

echo
echo "Done. Dataset under: $OUT_ROOT"
echo "Pack for CoMER: bash preprocessing/run_build_data_zip.sh"
echo "  (or: DATASET=$OUT_ROOT OUT_ZIP=$ROOT/data.zip bash preprocessing/run_build_data_zip.sh)"
