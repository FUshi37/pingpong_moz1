#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
PYTHON_BIN=${PYTHON_BIN:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}
XML_PATH=${XML_PATH:-$RL_SIM/moz1_pd.xml}
RMP_OUTPUT_REPLAY_REPORT=${RMP_OUTPUT_REPLAY_REPORT:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json}
BALL_MASS_MEASUREMENT_MANIFEST=${BALL_MASS_MEASUREMENT_MANIFEST:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/fixed_contact_plane_vision_robust_qvel_rmp_v9_gpu1_seed20260917_20260904/mjx_curriculum_best.pkl}
SOURCE_SHA256=781e3edc2869e671d59af0eb3afce1f08e34757e4abf5b6a7f876c218f9215a8

RUN_SEED=${RUN_SEED:-20260979}
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d)}
PROFILE=fixed_contact_plane_vision_robust_qvel_rmp_v39
RUN_ID=${RUN_ID:-${PROFILE}_gpu1_seed${RUN_SEED}_${RUN_DATE}}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID}
WANDB_NAME=${WANDB_NAME:-$RUN_ID}
EXPECTED_GPU_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e

# Resource isolation is mandatory for confirmed launches, including compilation.
# The environment marker prevents recursion; actual kernel limits are verified.
if [[ ${CONFIRM_GPU1_READY:-NO} == YES ]]; then
  if [[ ${PP_MEMORY_SCOPE_ACTIVE:-NO} != YES ]]; then
    exec "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" \
      --log-dir "$RUN_DIR.memory_guard" -- bash "$0" "$@"
  fi
  "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" --verify-only
fi

[[ -x "$PYTHON_BIN" ]] || { echo "missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$XML_PATH" ]] || { echo "missing XML: $XML_PATH" >&2; exit 2; }
[[ -f "$RMP_OUTPUT_REPLAY_REPORT" ]] || { echo "missing RMP replay: $RMP_OUTPUT_REPLAY_REPORT" >&2; exit 2; }
[[ -f "$BALL_MASS_MEASUREMENT_MANIFEST" ]] || { echo "missing ball-mass manifest: $BALL_MASS_MEASUREMENT_MANIFEST" >&2; exit 2; }
[[ -f "$SOURCE_CHECKPOINT" ]] || { echo "missing pinned V9 checkpoint: $SOURCE_CHECKPOINT" >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to overwrite output directory: $RUN_DIR" >&2; exit 2; }

actual_source_sha256=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
if [[ "$actual_source_sha256" != "$SOURCE_SHA256" ]]; then
  echo "V9 source hash mismatch: expected $SOURCE_SHA256, got $actual_source_sha256" >&2
  exit 2
fi
actual_gpu_uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')
if [[ "$actual_gpu_uuid" != "$EXPECTED_GPU_UUID" ]]; then
  echo "GPU1 UUID mismatch: expected $EXPECTED_GPU_UUID, got $actual_gpu_uuid" >&2
  exit 2
fi

echo "GPU1 V39 placed-ball repair audit:"
git -C "$REPO_ROOT" status --short --branch
nvidia-smi -i 1 \
  --query-gpu=index,uuid,name,temperature.gpu,memory.used,memory.total \
  --format=csv,noheader
echo "GPU1 compute processes (must be empty/reviewed):"
nvidia-smi -i 1 \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader || true
sha256sum \
  "$XML_PATH" \
  "$RMP_OUTPUT_REPLAY_REPORT" \
  "$BALL_MASS_MEASUREMENT_MANIFEST" \
  "$SOURCE_CHECKPOINT"
echo "Run directory: $RUN_DIR"
echo "Default trial: 64 stage updates, one stage, W&B offline. Change MAX_STAGE_UPDATES/MAX_STAGES/WANDB_MODE only after reviewing evidence."
echo "Course: ball and racket initialize together over the full fixed workspace; collision-on launch and recurrent juggling begin in Stage 1 without GPU0's acquisition prerequisite."
echo "Geometry repair: keep compiled mesh thickness at 1 mm and use its normal for placement/hold; remove the false 30 mm cylinder-radius floor. Preserve V37 reward, control and PPO to isolate the reset fix."
echo "State transfer: migrate only the pinned V9 actor; reset the falling-ball critic and Adam state for the placed-ball MDP."
echo "Acceptance remains: final full>=0.90, hit12>=0.80, apex 1.38--1.42 m, contact racket-vxy<=0.05 m/s, full omega<=0.50 rad/s, return step<=0.052 m."

if [[ ${CONFIRM_GPU1_READY:-NO} != YES ]]; then
  echo "Preflight only. Inspect the audit, then set CONFIRM_GPU1_READY=YES." >&2
  exit 3
fi
if [[ ${ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE:-NO} != YES ]]; then
  echo "Set ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES for this experimental run." >&2
  exit 2
fi

mkdir -p \
  "$RUN_DIR" \
  "$RUN_DIR/matplotlib" \
  "$RUN_DIR/wandb" \
  "$RUN_DIR/wandb_cache" \
  "$RUN_DIR/wandb_config"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MPLCONFIGDIR="$RUN_DIR/matplotlib"
export WANDB_DISABLE_CODE=true
export WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"

cd "$RL_SIM"
exec > >(tee -a "$RUN_DIR/stdout_stderr.log") 2>&1

exec "$PYTHON_BIN" -u train_juggle_mjx_curriculum.py \
  --xml "$XML_PATH" \
  --save-dir "$RUN_DIR" \
  --seed "$RUN_SEED" \
  --curriculum-profile "$PROFILE" \
  --curriculum-gate-preset v7_strict \
  --delay-ablation-preset baseline_current \
  --measured-ball-mass-range-kg 0.0039 0.0041 \
  --ball-mass-measurement-manifest "$BALL_MASS_MEASUREMENT_MANIFEST" \
  --rmp-output-replay-report "$RMP_OUTPUT_REPLAY_REPORT" \
  --allow-experimental-rmp-dr-bypass \
  --right-arm-pd-profile recovered_rmp_rmpmd_v2 \
  --actuator-compensation-mode none \
  --no-arm-servo-target-tracking-planner \
  --asymmetric-critic \
  --critic-command-history-steps 12 \
  --resume-from "$SOURCE_CHECKPOINT" \
  --resume-start-stage v39_s1_single_outcome \
  --reset-critic-on-resume \
  --reset-optimizer-on-resume \
  --no-allow-obs-dim-migration \
  --n-envs "${N_ENVS:-1024}" \
  --n-steps 128 \
  --minibatch-size 16384 \
  --update-epochs 4 \
  --learning-rate 1e-4 \
  --gamma 0.9995 \
  --gae-lambda 0.99 \
  --time-limit-bootstrap \
  --clip-range 0.10 \
  --target-kl 0.005 \
  --ent-coef 0.0 \
  --min-log-std -4.8 \
  --max-log-std -2.8 \
  --vf-coef 0.5 \
  --max-grad-norm 0.5 \
  --hidden-dim 256 \
  --clear-jax-caches-between-stages \
  --advance-mode converged \
  --convergence-window 24 \
  --convergence-min-episodes 128 \
  --min-stage-updates 40 \
  --stage-metric-warmup-updates 10 \
  --max-stage-updates "${MAX_STAGE_UPDATES:-64}" \
  --max-stages "${MAX_STAGES:-1}" \
  --advance-validation-mode block \
  --advance-eval-n-envs 256 \
  --advance-eval-steps 1200 \
  --advance-eval-min-episodes 128 \
  --advance-eval-retry-updates 20 \
  --advance-eval-hit-ratio 1.0 \
  --advance-eval-len-ratio 1.0 \
  --advance-eval-min-hits 0.0 \
  --advance-eval-min-len-frac 0.0 \
  --advance-eval-min-return -1000000 \
  --advance-eval-hit-rate-margin 0.0 \
  --advance-eval-hit-interval-margin 0.0 \
  --advance-eval-cond-hit-ratio 1.0 \
  --advance-eval-camera-margin 0.0 \
  --advance-eval-ball-view-margin 0.0 \
  --advance-eval-z-ideal-margin 0.0 \
  --advance-eval-reset-bucket-mode cvar \
  --advance-eval-reset-bucket-min-episodes 8 \
  --advance-eval-reset-bucket-cvar-frac 0.25 \
  --advance-eval-reset-bucket-rate-margin 0.0 \
  --advance-eval-reset-bucket-hit-margin 0.0 \
  --save-every-updates 50 \
  --archive-every-updates 500 \
  --gpu-max-temp-c 78 \
  --gpu-check-every-updates 1 \
  --wandb \
  --wandb-mode "${WANDB_MODE:-offline}" \
  --wandb-metrics-only \
  --wandb-project pingpong-mjx \
  --wandb-name "$WANDB_NAME" \
  --wandb-tags \
    gpu1 v39 placed-ball actor-only reset-critic-adam apex-tail-credit \
    80d measured-qvel bounded-qref recovered-rmp v33-finite-mesh evidence-gated
