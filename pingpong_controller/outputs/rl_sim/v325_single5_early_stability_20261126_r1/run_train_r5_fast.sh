#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_ID="${1:-fplc_wind_early_20261127_r5_fast}"
WANDB_MODE="${2:-offline}"
UPDATE_LIMIT="${3:-}"
exec bash "$SCRIPT_DIR/run_train_early.sh" "$RUN_ID" "$WANDB_MODE" "$UPDATE_LIMIT" "$SCRIPT_DIR/TRAIN_PLAN_R5_FAST.json"
