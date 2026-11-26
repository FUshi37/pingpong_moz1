#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-fplc_wind_early_20261126_r1}"
WANDB_MODE="${2:-online}"
UPDATE_LIMIT="${3:-}"
PLAN_FILE="${4:-$SCRIPT_DIR/TRAIN_PLAN_EARLY.json}"

OUT_DIR="$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID"
GPU_UUID="GPU-3b3e4ba6-dd29-09fe-ba9b-d8da127b3415"

[[ -f "$HOME/.profile" ]] && source "$HOME/.profile"
set +u
source "$HOME/envs/pingpong/bin/activate"
set -u

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export PP_MEMORY_SCOPE_ACTIVE="YES"
export WANDB_DIR="/dev/shm/wandb_${RUN_ID}"
mkdir -p "$WANDB_DIR"

EXTRA_ARGS=()
if [[ -n "$UPDATE_LIMIT" ]]; then
  EXTRA_ARGS+=(--update-limit "$UPDATE_LIMIT")
fi

exec bash "$REPO_ROOT/pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh" \
  "/dev/shm/${RUN_ID}_memguard.log" \
  python "$SCRIPT_DIR/train_early_stability.py" \
    --plan "$PLAN_FILE" \
    --output "$OUT_DIR" \
    --run-id "$RUN_ID" \
    --gpu-uuid "$GPU_UUID" \
    --wandb-mode "$WANDB_MODE" \
    "${EXTRA_ARGS[@]}"
