#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(pwd)}"
export PYTHONDONTWRITEBYTECODE=1
cd "$ROOT"

python - <<'PY'
import importlib.util
import os
import sys

if importlib.util.find_spec("isaacgym") is None:
    msg = "missing optional external package: isaacgym (install NVIDIA Isaac Gym Preview separately)"
    if os.environ.get("SCFIELDS_REQUIRE_ISAACGYM") == "1":
        raise ModuleNotFoundError(msg)
    print(msg, file=sys.stderr)
else:
    import isaacgym
    print("ok isaacgym")

import importlib

for name in [
    "torch",
    "gym",
    "hydra",
    "omegaconf",
    "rl_games",
    "isaacgymenvs",
]:
    importlib.import_module(name)
    print(f"ok {name}")
PY
