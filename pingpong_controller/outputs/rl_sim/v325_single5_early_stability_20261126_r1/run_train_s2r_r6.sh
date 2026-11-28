#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-fplc_wind_early_20261128_s2r_r6}"
WANDB_MODE="${2:-offline}"
UPDATE_LIMIT="${3:-}"
PLAN_FILE="${4:-$SCRIPT_DIR/TRAIN_PLAN_S2R_R6.json}"

OUT_DIR="$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID"
GPU_UUID="${GPU_UUID:-$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -n 1)}"

[[ -f "$HOME/.profile" ]] && source "$HOME/.profile"
set +u
source "$HOME/envs/pingpong/bin/activate"
set -u

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export PP_MEMORY_SCOPE_ACTIVE="YES"
RUN_TMP_ROOT="${RUN_TMP_ROOT:-$HOME/.cache/pingpong_run_tmp}"
export MIGRATED_HOST="${MIGRATED_HOST:-192.168.14.52}"
export TMPDIR="$RUN_TMP_ROOT/tmp"
export WANDB_DIR="$RUN_TMP_ROOT/wandb_${RUN_ID}"
export JAX_COMPILATION_CACHE_DIR="$RUN_TMP_ROOT/jax_cache"
mkdir -p "$TMPDIR" "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR"

EXTRA_ARGS=()
if [[ -n "$UPDATE_LIMIT" ]]; then
  EXTRA_ARGS+=(--update-limit "$UPDATE_LIMIT")
fi

exec bash "$REPO_ROOT/pingpong_controller/tools/rl_sim/run_with_host_memory_guard.sh" \
  "$RUN_TMP_ROOT/memguard_${RUN_ID}.log" \
  python3 "$SCRIPT_DIR/train_sim2real_r6.py" \
  --plan "$PLAN_FILE" \
  --output "$OUT_DIR" \
  --run-id "$RUN_ID" \
  --gpu-uuid "$GPU_UUID" \
  --wandb-mode "$WANDB_MODE" \
  "${EXTRA_ARGS[@]}"
