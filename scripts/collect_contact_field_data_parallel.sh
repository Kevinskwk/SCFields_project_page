#!/usr/bin/env bash
set -euo pipefail

ROOT="${SCFIELDS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SCFIELDS_ROOT="$ROOT"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data}"

DEVICES=(${DEVICES:-cuda:0})
SPLIT="${SPLIT:-train}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_STEPS="${MAX_STEPS:-200}"
SAVE_FREQUENCY="${SAVE_FREQUENCY:-$MAX_STEPS}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"
CONDA_ENV="${CONDA_ENV:-}"
ITERATIONS="${ITERATIONS:-1}"
RUN_ID_START="${RUN_ID_START:-0}"
BASE_SEED="${BASE_SEED:-42}"
ASSET_INFO="${ASSET_INFO:-tacsl_asset_info_generated_tools.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATA_ROOT/sim/tools}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/contact_field_data/$(date +%Y%m%d_%H%M%S)}"
KEEP_LOGS=false
DRY_RUN=false

GEOMETRY_NAMES=(${GEOMETRY_NAMES:-scraper})
GEOMETRY_IDS=(${GEOMETRY_IDS:-1-25})
SUBASSEMBLIES=()

show_usage() {
  cat <<EOF
Usage: $0 [options]

Runs IsaacGym rollout collection and PyBullet contact labeling in parallel
across CUDA devices. Each device runs one collection process at a time.

Options:
  --devices "cuda:0 cuda:1"       CUDA devices to use. Default: cuda:0
  --subassemblies "a b c"         Explicit subassemblies. Overrides geometry options.
  --geometry-names "scraper"      Geometry prefixes. Default: scraper
  --geometry-ids "1-25"           IDs or ranges, e.g. "1-10 15 20-25". Default: 1-25
  --split train                   Data split. Default: train
  --iterations N                  Rollouts per subassembly. Default: 1
  --run-id-start N                First run id. Default: 0
  --num-envs N                    IsaacGym envs per process. Default: 8
  --max-steps N                   IsaacGym collection steps. Default: 200
  --save-frequency N              IsaacGym save frequency. Default: max-steps
  --graphics-device-id N          IsaacGym graphics device id for camera sensors. Default: 0
  --conda-env NAME                Run each job with conda run -n NAME.
  --output-dir DIR                Output root. Default: \$DATA_ROOT/sim/tools
  --log-dir DIR                   Log root. Default: logs/contact_field_data/<timestamp>
  --asset-info FILE               Asset info YAML. Default: tacsl_asset_info_generated_tools.yaml
  --base-seed N                   Base seed for deterministic per-job seeds. Default: 42
  --keep-logs                     Keep existing files in --log-dir.
  --dry-run                       Print planned jobs without running them.
  --help                          Show this help.

Examples:
  $0 --devices "cuda:0 cuda:1" --geometry-names "scraper" --geometry-ids "1-25"
  $0 --devices "cuda:0 cuda:1 cuda:2 cuda:3" --subassemblies "scraper_8 cylinder_1" --iterations 3
EOF
}

expand_ids() {
  local token start end id
  for token in "$@"; do
    if [[ "$token" =~ ^[0-9]+-[0-9]+$ ]]; then
      start="${token%-*}"
      end="${token#*-}"
      for ((id=start; id<=end; id++)); do
        printf '%s\n' "$id"
      done
    else
      printf '%s\n' "$token"
    fi
  done
}

split_words() {
  local value="$1"
  value="${value//,/ }"
  # shellcheck disable=SC2086
  printf '%s\n' $value
}

job_seed() {
  local subassembly="$1"
  local run_id="$2"
  local hash
  hash=$(printf '%s' "${subassembly}_${run_id}" | cksum | awk '{print $1}')
  echo $((BASE_SEED + hash % 100000))
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --devices)
        mapfile -t DEVICES < <(split_words "$2")
        shift 2
        ;;
      --subassemblies)
        mapfile -t SUBASSEMBLIES < <(split_words "$2")
        shift 2
        ;;
      --geometry-names)
        mapfile -t GEOMETRY_NAMES < <(split_words "$2")
        shift 2
        ;;
      --geometry-ids)
        mapfile -t GEOMETRY_IDS < <(expand_ids $(split_words "$2"))
        shift 2
        ;;
      --split)
        SPLIT="$2"
        shift 2
        ;;
      --iterations)
        ITERATIONS="$2"
        shift 2
        ;;
      --run-id-start)
        RUN_ID_START="$2"
        shift 2
        ;;
      --num-envs)
        NUM_ENVS="$2"
        shift 2
        ;;
      --max-steps)
        MAX_STEPS="$2"
        shift 2
        ;;
      --save-frequency)
        SAVE_FREQUENCY="$2"
        shift 2
        ;;
      --graphics-device-id)
        GRAPHICS_DEVICE_ID="$2"
        shift 2
        ;;
      --conda-env)
        CONDA_ENV="$2"
        shift 2
        ;;
      --output-dir)
        OUTPUT_DIR="$2"
        shift 2
        ;;
      --log-dir)
        LOG_DIR="$2"
        shift 2
        ;;
      --asset-info)
        ASSET_INFO="$2"
        shift 2
        ;;
      --base-seed)
        BASE_SEED="$2"
        shift 2
        ;;
      --keep-logs)
        KEEP_LOGS=true
        shift
        ;;
      --dry-run)
        DRY_RUN=true
        shift
        ;;
      --help)
        show_usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1" >&2
        show_usage >&2
        exit 1
        ;;
    esac
  done
}

build_subassemblies() {
  local name id
  if [[ "${#SUBASSEMBLIES[@]}" -gt 0 ]]; then
    return
  fi
  mapfile -t GEOMETRY_IDS < <(expand_ids "${GEOMETRY_IDS[@]}")
  for name in "${GEOMETRY_NAMES[@]}"; do
    for id in "${GEOMETRY_IDS[@]}"; do
      SUBASSEMBLIES+=("${name}_${id}")
    done
  done
}

validate_plan() {
  if [[ "${#DEVICES[@]}" -eq 0 ]]; then
    echo "No CUDA devices specified." >&2
    exit 1
  fi
  if [[ "${#SUBASSEMBLIES[@]}" -eq 0 ]]; then
    echo "No subassemblies selected." >&2
    exit 1
  fi
  if [[ "$ITERATIONS" -lt 1 ]]; then
    echo "--iterations must be >= 1." >&2
    exit 1
  fi
}

print_plan() {
  cat <<EOF
Parallel contact-field data collection
  devices: ${DEVICES[*]}
  subassemblies: ${#SUBASSEMBLIES[@]} (${SUBASSEMBLIES[*]})
  split: $SPLIT
  iterations: $ITERATIONS
  run id start: $RUN_ID_START
  num envs/process: $NUM_ENVS
  max steps: $MAX_STEPS
  graphics device id: $GRAPHICS_DEVICE_ID
  conda env: ${CONDA_ENV:-current shell}
  output dir: $OUTPUT_DIR
  log dir: $LOG_DIR
  asset info: $ASSET_INFO
EOF
}

run_job() {
  local subassembly="$1"
  local run_id="$2"
  local device="$3"
  local log_file="$4"
  local seed status
  seed="$(job_seed "$subassembly" "$run_id")"

  {
    echo "[$(date '+%F %T')] start subassembly=$subassembly run_id=$run_id device=$device seed=$seed"
    if [[ -n "$CONDA_ENV" ]]; then
      SIM_DEVICE="$device" \
      RL_DEVICE="$device" \
      GRAPHICS_DEVICE_ID="$GRAPHICS_DEVICE_ID" \
      SPLIT="$SPLIT" \
      RUN_ID="$run_id" \
      NUM_ENVS="$NUM_ENVS" \
      MAX_STEPS="$MAX_STEPS" \
      SAVE_FREQUENCY="$SAVE_FREQUENCY" \
      OUTPUT_DIR="$OUTPUT_DIR" \
      ASSET_INFO="$ASSET_INFO" \
      SEED="$seed" \
      conda run -n "$CONDA_ENV" bash "$ROOT/scripts/collect_contact_field_data.sh" "$subassembly"
    else
      SIM_DEVICE="$device" \
      RL_DEVICE="$device" \
      GRAPHICS_DEVICE_ID="$GRAPHICS_DEVICE_ID" \
      SPLIT="$SPLIT" \
      RUN_ID="$run_id" \
      NUM_ENVS="$NUM_ENVS" \
      MAX_STEPS="$MAX_STEPS" \
      SAVE_FREQUENCY="$SAVE_FREQUENCY" \
      OUTPUT_DIR="$OUTPUT_DIR" \
      ASSET_INFO="$ASSET_INFO" \
      SEED="$seed" \
      bash "$ROOT/scripts/collect_contact_field_data.sh" "$subassembly"
    fi
    status=$?
    if [[ "$status" -eq 0 ]]; then
      echo "[$(date '+%F %T')] done subassembly=$subassembly run_id=$run_id device=$device"
    else
      echo "[$(date '+%F %T')] failed subassembly=$subassembly run_id=$run_id device=$device status=$status"
    fi
    return "$status"
  } >> "$log_file" 2>&1
}

device_worker() {
  local worker_id="$1"
  local device="$2"
  local summary_file="$3"
  local device_log="$LOG_DIR/device_${device//:/}.log"
  local failures=0
  local index=0
  local subassembly run_id log_file status rollout contact

  echo "device=$device worker=$worker_id pid=$$" > "$device_log"
  for subassembly in "${SUBASSEMBLIES[@]}"; do
    if (( index % ${#DEVICES[@]} != worker_id )); then
      ((index += 1))
      continue
    fi
    for ((run_id=RUN_ID_START; run_id<RUN_ID_START+ITERATIONS; run_id++)); do
      log_file="$LOG_DIR/jobs/${subassembly}_${SPLIT}_${run_id}_${device//:/}.log"
      echo "[$(date '+%F %T')] queued $subassembly run_id=$run_id on $device" | tee -a "$device_log"
      set +e
      run_job "$subassembly" "$run_id" "$device" "$log_file"
      status=$?
      set -e

      rollout="$OUTPUT_DIR/$SPLIT/${subassembly}_${SPLIT}_${run_id}.pkl"
      contact="${rollout%.pkl}_contact.pkl"
      if [[ "$status" -eq 0 && -s "$rollout" && -s "$contact" ]]; then
        printf 'ok\t%s\t%s\t%s\t%s\t%s\n' "$device" "$subassembly" "$run_id" "$rollout" "$contact" >> "$summary_file"
        echo "[$(date '+%F %T')] ok $subassembly run_id=$run_id" | tee -a "$device_log"
      else
        printf 'failed\t%s\t%s\t%s\t%s\t%s\n' "$device" "$subassembly" "$run_id" "$rollout" "$log_file" >> "$summary_file"
        echo "[$(date '+%F %T')] failed $subassembly run_id=$run_id status=$status log=$log_file" | tee -a "$device_log"
        ((failures += 1))
      fi
      sleep 2
    done
    ((index += 1))
  done
  return "$failures"
}

main() {
  parse_args "$@"
  build_subassemblies
  validate_plan
  print_plan

  if [[ "$DRY_RUN" == true ]]; then
    return 0
  fi

  if [[ -d "$LOG_DIR" && "$KEEP_LOGS" != true ]]; then
    rm -rf "$LOG_DIR"
  fi
  mkdir -p "$LOG_DIR/jobs" "$OUTPUT_DIR/$SPLIT"

  local summary_file="$LOG_DIR/summary.tsv"
  printf 'status\tdevice\tsubassembly\trun_id\trollout_or_expected\tcontact_or_log\n' > "$summary_file"

  local pids=()
  local worker_id
  for worker_id in "${!DEVICES[@]}"; do
    device_worker "$worker_id" "${DEVICES[$worker_id]}" "$summary_file" &
    pids+=("$!")
    sleep 3
  done

  local failed_workers=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      ((failed_workers += 1))
    fi
  done

  echo "Summary: $summary_file"
  if [[ "$failed_workers" -ne 0 ]] || grep -q '^failed' "$summary_file"; then
    echo "One or more collection jobs failed. Check $LOG_DIR/jobs/*.log" >&2
    return 1
  fi
}

main "$@"
