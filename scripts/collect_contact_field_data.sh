#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(pwd)}"
export SCFIELDS_ROOT="$ROOT"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"
SUBASSEMBLY="${1:-scraper_8}"
SPLIT="${SPLIT:-train}"
RUN_ID="${RUN_ID:-0}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-200}"
SIM_DEVICE="${SIM_DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-$SIM_DEVICE}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"
ASSET_INFO="${ASSET_INFO:-tacsl_asset_info_generated_tools.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/sim/tools}"
ROLLOUT_PATH="$OUTPUT_DIR/$SPLIT/${SUBASSEMBLY}_${SPLIT}_${RUN_ID}.pkl"
CONTACT_PATH="${ROLLOUT_PATH%.pkl}_contact.pkl"
SEED_OVERRIDE=()
if [[ -n "${SEED:-}" ]]; then
  SEED_OVERRIDE=(seed="$SEED")
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT/isaacgymenvs"

rm -f "$ROLLOUT_PATH" "$CONTACT_PATH"

set +e
python collect_data.py \
  --config-name=data_collection \
  output_dir="$OUTPUT_DIR" \
  task.env.asset_info_filename="$ASSET_INFO" \
  task.env.subassembly="$SUBASSEMBLY" \
  split="$SPLIT" \
  id="$RUN_ID" \
  num_envs="$NUM_ENVS" \
  max_collection_steps="$MAX_STEPS" \
  save_frequency="$SAVE_FREQUENCY" \
  sim_device="$SIM_DEVICE" \
  rl_device="$RL_DEVICE" \
  graphics_device_id="$GRAPHICS_DEVICE_ID" \
  "${SEED_OVERRIDE[@]}"
COLLECT_STATUS=$?
set -e

if [[ "$COLLECT_STATUS" -ne 0 ]]; then
  if [[ "$COLLECT_STATUS" -eq 139 && -s "$ROLLOUT_PATH" ]]; then
    echo "Warning: IsaacGym exited with post-save segfault 139; continuing because rollout exists: $ROLLOUT_PATH" >&2
  else
    echo "IsaacGym collection failed with exit code $COLLECT_STATUS" >&2
    exit "$COLLECT_STATUS"
  fi
fi

python pybullet_contact_collection.py \
  --config-name=pybullet_contact_collection \
  data_path="$ROLLOUT_PATH" \
  task.env.asset_info_filename="$ASSET_INFO" \
  task.env.subassembly="$SUBASSEMBLY"

echo "IsaacGym rollout: $ROLLOUT_PATH"
echo "PyBullet contact labels: $CONTACT_PATH"
