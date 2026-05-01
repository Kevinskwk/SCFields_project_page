#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HF_REPO="${SCFIELDS_HF_REPO:-Kevinskwk/scfields-release}"
OUT_DIR="${SCFIELDS_ASSET_CACHE:-$ROOT_DIR/.asset_cache/assets}"

if ! command -v hf >/dev/null 2>&1; then
  cat <<EOF >&2
The Hugging Face CLI is required.

Install it with:

  pip install -U "huggingface_hub[cli]"

Then rerun:

  SCFIELDS_HF_REPO=$HF_REPO bash scripts/download_assets.sh
EOF
  exit 1
fi

mkdir -p "$OUT_DIR"

hf download "$HF_REPO" \
  --repo-type dataset \
  --include "assets/**" \
  --local-dir "$OUT_DIR"

mkdir -p "$ROOT_DIR/assets/tools" "$ROOT_DIR/assets/peeler" "$ROOT_DIR/assets/peelers_combined"

cp -a "$OUT_DIR/assets/tools/." "$ROOT_DIR/assets/tools/"
cp -a "$OUT_DIR/assets/peeler_raw/." "$ROOT_DIR/assets/peeler/"
cp -a "$OUT_DIR/assets/peeler_combined/." "$ROOT_DIR/assets/peelers_combined/"

echo "Installed SCFields assets from $HF_REPO into $ROOT_DIR/assets"
