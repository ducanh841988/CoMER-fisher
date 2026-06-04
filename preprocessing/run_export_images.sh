#!/usr/bin/env bash
# Copy PNGs from TC11 IMG or render from INKML into preprocessing/output/img.
#
# Usage (from CoMER repo root):
#   bash preprocessing/run_export_images.sh
#   bash preprocessing/run_export_images.sh --img-only
#   bash preprocessing/run_export_images.sh --inkml-only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/habku/anh_project/TC11_CROHME23}"
OUT_ROOT="${OUT_ROOT:-$ROOT/preprocessing/output/img}"
WORKERS="${WORKERS:-$(cd "$ROOT" && python -c 'from preprocessing.batch_utils import default_workers; print(default_workers())' 2>/dev/null || echo 4)}"
IMG_ONLY=""
INKML_ONLY=""

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
    --img-only)
      IMG_ONLY="--img-only"
      shift
      ;;
    --inkml-only)
      INKML_ONLY="--inkml-only"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Export images: copy from TC11 IMG or render from INKML.

Options:
  --data-root PATH   TC11_CROHME23 root
  --output PATH      Output root (default: preprocessing/output/img)
  --workers N        Parallel workers
  --img-only         Copy all PNGs from IMG (skip INKML rendering)
  --inkml-only       Render all PNGs from INKML (skip IMG lookup/copy)

Example:
  bash preprocessing/run_export_images.sh
  bash preprocessing/run_export_images.sh --img-only
  bash preprocessing/run_export_images.sh --inkml-only
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -n "$IMG_ONLY" && -n "$INKML_ONLY" ]]; then
  echo "Use either --img-only or --inkml-only, not both" >&2
  exit 1
fi

cd "$ROOT"
python preprocessing/export_images.py \
  --data-root "$DATA_ROOT" \
  --output "$OUT_ROOT" \
  --workers "$WORKERS" \
  $IMG_ONLY \
  $INKML_ONLY
