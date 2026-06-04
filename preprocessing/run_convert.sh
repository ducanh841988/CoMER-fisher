#!/usr/bin/env bash
# Convert TC11 SymLG labels to LaTeX (.tex) files.
#
# Usage (from CoMER repo root):
#   bash preprocessing/run_convert.sh
#   DATA_ROOT=/path/to/TC11_CROHME23 bash preprocessing/run_convert.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/home/habku/anh_project/TC11_CROHME23}"
OUT_ROOT="${OUT_ROOT:-$ROOT/preprocessing/output/lg}"
WORKERS="${WORKERS:-$(cd "$ROOT" && python -c 'from preprocessing.batch_utils import default_workers; print(default_workers())' 2>/dev/null || echo 4)}"

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
    -h|--help)
      cat <<'EOF'
Convert TC11 SymLG (.lg) label files to LaTeX (.tex).

Options:
  --data-root PATH    TC11_CROHME23 root
  --output PATH       Output root (default: preprocessing/output/lg)
  --workers N         Parallel worker processes (default: CPU count)

Environment overrides:
  DATA_ROOT, OUT_ROOT, WORKERS

Example:
  bash preprocessing/run_convert.sh
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

INPUT_ROOT="$DATA_ROOT/SymLG"
if [[ ! -d "$INPUT_ROOT" ]]; then
  echo "Input root not found: $INPUT_ROOT" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$OUT_ROOT"

echo "Input:   $INPUT_ROOT"
echo "Output:  $OUT_ROOT"
echo "Workers: $WORKERS"
echo

for split in train val test; do
  split_dir="$INPUT_ROOT/$split"
  [[ -d "$split_dir" ]] || continue

  while IFS= read -r -d '' subdir; do
    rel="${subdir#$INPUT_ROOT/}"
    out_dir="$OUT_ROOT/$rel"
    echo ">>> $rel"
    python preprocessing/convert_labels.py \
      --input "$subdir" \
      --output "$out_dir" \
      --workers "$WORKERS"
    echo
  done < <(find "$split_dir" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
done

echo "All done. LaTeX files are under: $OUT_ROOT"
