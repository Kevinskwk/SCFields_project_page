#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(pwd)}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT/contact_field"

python - <<'PY'
import importlib

for name in [
    "dataset",
    "models.tactile_pointnet_concat",
    "models.tactile_pointnet_joint_enhanced",
    "utils.data_utils",
]:
    importlib.import_module(name)
    print(f"ok {name}")
PY
