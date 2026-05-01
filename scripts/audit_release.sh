#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$(pwd)}"
PRUNE_GENERATED=( \
  -path "$ROOT/.git" -o \
  -path "$ROOT/data" -o \
  -path "$ROOT/artifacts" -o \
  -path "$ROOT/outputs" -o \
  -path "$ROOT/logs" -o \
  -path "$ROOT/runs" -o \
  -path "$ROOT/wandb" \
)

echo "[audit] root: $ROOT"

echo "[audit] nested git directories"
if find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type d -name .git -print | grep -q .; then
  find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type d -name .git -print
  exit 1
fi

echo "[audit] generated Python/install artifacts"
if find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type d \( -name __pycache__ -o -name '*.egg-info' \) -print | grep -q .; then
  find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type d \( -name __pycache__ -o -name '*.egg-info' \) -print
  exit 1
fi
if find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -print | grep -q .; then
  find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type f \( -name '*.pyc' -o -name '*.pyo' \) -print
  exit 1
fi

echo "[audit] private absolute paths or robot IP defaults"
if rg -n '/users/kevinma|/home/kevin|/home/yixuan|192\.168\.1\.143' "$ROOT" \
  -g '*.py' -g '*.yaml' -g '*.yml' -g '*.sh' -g '*.md' -g '*.txt' \
  -g '!scripts/audit_release.sh'; then
  exit 1
fi

echo "[audit] removed release-only clutter"
if find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type f \( -name '*ablation*' -o -name '*demo*' -o -name '*viz*' -o -name '*visualize*' \) \
  -not -path "$ROOT/contact_field/utils/viz_utils.py" \
  -not -path "$ROOT/gendp/demo_real_franka.py" \
  -not -path "$ROOT/gendp/demo_real_franka_terminal.py" \
  -not -path "$ROOT/gendp/gendp/real_world/multi_camera_visualizer.py" \
  -not -path "$ROOT/gendp/gendp/real_world/vis_cam_cali.py" \
  -print | grep -q .; then
  find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type f \( -name '*ablation*' -o -name '*demo*' -o -name '*viz*' -o -name '*visualize*' \) \
    -not -path "$ROOT/contact_field/utils/viz_utils.py" \
    -not -path "$ROOT/gendp/demo_real_franka.py" \
    -not -path "$ROOT/gendp/demo_real_franka_terminal.py" \
    -not -path "$ROOT/gendp/gendp/real_world/multi_camera_visualizer.py" \
    -not -path "$ROOT/gendp/gendp/real_world/vis_cam_cali.py" \
    -print
  exit 1
fi

echo "[audit] large/generated artifact extensions"
if find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type f \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.hdf5' -o -name '*.bag' -o -name '*.mp4' -o -name '*.zip' -o -name '*.so' -o -name '*.gif' \) -print | grep -q .; then
  find "$ROOT" \( "${PRUNE_GENERATED[@]}" \) -prune -o -type f \( -name '*.ckpt' -o -name '*.pt' -o -name '*.pth' -o -name '*.pkl' -o -name '*.hdf5' -o -name '*.bag' -o -name '*.mp4' -o -name '*.zip' -o -name '*.so' -o -name '*.gif' \) -print
  exit 1
fi

echo "[audit] ok"
