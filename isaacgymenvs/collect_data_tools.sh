#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SCFIELDS_ROOT="$ROOT"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

num_envs="${NUM_ENVS:-8}"
# num_envs=4
# split='train'
# split='val'
split="${SPLIT:-test}"
num_iterations="${NUM_ITERATIONS:-1}" # Number of data files to generate

# Run python collect_data.py command for all tool_shape pairs
# object_names=("peeler_1_shape_1" "peeler_2_shape_2" "peeler_3_shape_3" "peeler_4_shape_4" "peeler_5_shape_5" "peeler_6_shape_6" "peeler_7_shape_7" "peeler_8_shape_8" "peeler_9_shape_9" "peeler_10_shape_10" "peeler_11_shape_11" "peeler_1_1_shape_12" "peeler_1_2_shape_13" "peeler_2_1_shape_14" "peeler_3_1_shape_15" "peeler_3_2_shape_16" "peeler_3_3_shape_17" "peeler_4_1_shape_18" "peeler_4_2_shape_19" "peeler_5_1_shape_20" "peeler_5_2_shape_21" "peeler_6_1_shape_22" "peeler_6_2_shape_23" "peeler_7_1_shape_24" "peeler_7_2_shape_25" "peeler_9_1_shape_26" "peeler_10_1_shape_27" "peeler_10_2_shape_28")
# object_names=("cylinder_1")
geometry_names=("cylinder" "rectangle" "hex_prism" "scraper")
geometry_ids=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
asset_info_filename="tacsl_asset_info_generated_tools.yaml"
# asset_info_filename="tacsl_asset_info_tool_pad.yaml"
sim_device="${SIM_DEVICE:-cuda:0}"

# for object_name in "${object_names[@]}"; do
for geometry_name in "${geometry_names[@]}"; do
    for geometry_id in "${geometry_ids[@]}"; do
        object_name="${geometry_name}_${geometry_id}"
        echo "Processing subassembly: $object_name"

        for n in $(seq 0 $(($num_iterations - 1))); do
            echo "Running iteration $n for object: $object_name..."
            ASSET_INFO="$asset_info_filename" \
            SIM_DEVICE="$sim_device" \
            RL_DEVICE="${RL_DEVICE:-$sim_device}" \
            SPLIT="$split" \
            RUN_ID="$n" \
            NUM_ENVS="$num_envs" \
            bash "$ROOT/scripts/collect_contact_field_data.sh" "$object_name"

            echo "Completed iteration $n for object: $object_name"
            echo "---"
        done

        echo "Completed processing object: $object_name"
        echo "====="
    done
done

echo "All iterations completed successfully!"
