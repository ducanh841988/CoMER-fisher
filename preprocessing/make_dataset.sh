#!/usr/bin/env bash
# Build CoMER dataset from TC11 CROHME23: caption.txt + img/*.bmp (black bg, white strokes)
#
# Usage (from CoMER repo root):
#   bash preprocessing/make_dataset.sh              # build dataset
#   bash preprocessing/make_dataset.sh --zip        # pack data.zip
#   bash preprocessing/make_dataset.sh --analyze    # LaTeX + image stats
#
# Output:
#   preprocessing/output/dataset/data/{train,val,2019,2023}/{caption.txt, img/*.bmp}

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/habku/anh_project/TC11_CROHME23}"
OUT_DATASET="${OUT_DATASET:-$ROOT/preprocessing/output/dataset}"
OUT_ZIP="${OUT_ZIP:-$ROOT/data.zip}"
WORKERS="${WORKERS:-$(cd "$ROOT" && python -c 'from preprocessing.batch_utils import default_workers; print(default_workers())' 2>/dev/null || echo 4)}"
MAP="${MAP:-$ROOT/preprocessing/symLG_map.csv}"

MODE="build"
EXTRA_PIPELINE=()

usage() {
  cat <<'EOF'
CoMER preprocessing (TC11 CROHME23 -> caption.txt + BMP images).

Commands:
  (default)           SymLG -> LaTeX + images -> preprocessing/output/dataset/
  --zip               Pack dataset into data.zip (CoMER config)
  --analyze           LaTeX vocab + image size reports

Options (build mode):
  --data-root PATH    TC11 root with SymLG/, INKML/, IMG/
  --output PATH       Dataset output root (default: preprocessing/output/dataset)
  --workers N         Parallel workers (default: CPU count)
  --no-crop           Skip content crop when exporting images
  --write-tex         Also save .tex under output/tex/
  --max-samples N     Process only first N samples (testing)
  --no-progress       Disable tqdm

Options (--zip):
  --zip-output PATH   Output zip path (default: data.zip in repo root)
  --dictionary PATH   Drop OOV samples using comer/datamodule/dictionary.txt
  --migrate-only      Move legacy train/val/test/* into data/* (no zip)

Options (--analyze):
  --analysis-output PATH  Report directory (default: preprocessing/output/analysis)

Examples:
  bash preprocessing/make_dataset.sh
  DATA_ROOT=/path/to/TC11 bash preprocessing/make_dataset.sh
  bash preprocessing/make_dataset.sh --max-samples 100
  bash preprocessing/make_dataset.sh --zip
  bash preprocessing/make_dataset.sh --analyze

config.yaml paths:
  train_path: preprocessing/output/dataset/data/train
  val_path:   preprocessing/output/dataset/data/val
  test_path:  preprocessing/output/dataset/data/2023
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --zip|--analyze)
      MODE="${1#--}"
      shift
      ;;
    --data-root)
      DATA_ROOT="$2"
      shift 2
      ;;
    --output)
      OUT_DATASET="$2"
      shift 2
      ;;
    --zip-output)
      OUT_ZIP="$2"
      shift 2
      ;;
    --analysis-output)
      OUT_ANALYSIS="$2"
      shift 2
      ;;
    --dictionary)
      DICTIONARY="$2"
      shift 2
      ;;
    --migrate-only)
      MIGRATE_ONLY=1
      MODE="zip"
      shift
      ;;
    --workers)
      WORKERS="$2"
      shift 2
      ;;
    --no-crop)
      EXTRA_PIPELINE+=(--no-crop)
      shift
      ;;
    --write-tex)
      EXTRA_PIPELINE+=(--write-tex)
      shift
      ;;
    --max-samples)
      EXTRA_PIPELINE+=(--max-samples "$2")
      shift 2
      ;;
    --no-progress)
      EXTRA_PIPELINE+=(--no-progress)
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cd "$ROOT"

case "$MODE" in
  build)
    if [[ ! -d "$DATA_ROOT/SymLG" ]]; then
      echo "SymLG not found under: $DATA_ROOT" >&2
      exit 1
    fi
    echo "=== Build dataset ==="
    echo "Data:    $DATA_ROOT"
    echo "Output:  $OUT_DATASET"
    echo "Workers: $WORKERS"
    echo
    python preprocessing/pipeline.py \
      --data-root "$DATA_ROOT" \
      --output "$OUT_DATASET" \
      --map "$MAP" \
      --workers "$WORKERS" \
      "${EXTRA_PIPELINE[@]}"
    echo
    echo "Done. Point config.yaml at:"
    echo "  train_path: preprocessing/output/dataset/data/train"
    echo "  val_path:   preprocessing/output/dataset/data/val"
    echo "  test_path:  preprocessing/output/dataset/data/2023"
    echo "Pack: bash preprocessing/make_dataset.sh --zip"
    ;;
  zip)
    ZIP_ARGS=(--dataset "$OUT_DATASET" --output "$OUT_ZIP")
    if [[ -n "${DICTIONARY:-}" ]]; then
      ZIP_ARGS+=(--dictionary "$DICTIONARY")
    fi
    if [[ -n "${MIGRATE_ONLY:-}" ]]; then
      ZIP_ARGS+=(--migrate-only)
    fi
    python preprocessing/build_data_zip.py "${ZIP_ARGS[@]}"
    ;;
  analyze)
    DATA_DIR="$OUT_DATASET/data"
    OUT_ANALYSIS="${OUT_ANALYSIS:-$ROOT/preprocessing/output/analysis}"
    echo "=== Analyze LaTeX (caption.txt) ==="
    python preprocessing/analyze_latex.py --input "$DATA_DIR" --output "$OUT_ANALYSIS"
    echo
    echo "=== Analyze images ==="
    python preprocessing/analyze_images.py --input "$DATA_DIR" --output "$OUT_ANALYSIS"
    ;;
esac
