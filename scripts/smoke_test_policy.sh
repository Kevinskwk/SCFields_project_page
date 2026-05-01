#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(pwd)}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/third_party/d3fields_dev:${PYTHONPATH:-}"
cd "$ROOT/gendp"

python - <<'PY'
import importlib

for name in [
    "gendp.policy.diffusion_unet_hybrid_image_policy",
    "gendp.workspace.train_diffusion_unet_hybrid_workspace",
    "gendp.real_world.real_env_franka_gripper_gelsight",
    "eval_real_franka_terminal_contact_field",
]:
    importlib.import_module(name)
    print(f"ok {name}")
PY
