#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
LAUNCHER_PATH=$(realpath "$0")
PYTHON_BIN=${PYTHON_BIN:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}
XML_PATH=${XML_PATH:-$RL_SIM/moz1_pd.xml}
RMP_OUTPUT_REPLAY_REPORT=${RMP_OUTPUT_REPLAY_REPORT:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json}
BALL_MASS_MEASUREMENT_MANIFEST=${BALL_MASS_MEASUREMENT_MANIFEST:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json}
SOURCE_RUN=${SOURCE_RUN:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/fixed_contact_plane_vision_robust_qvel_rmp_v97_gpu0_formal_1024x128_resume_v95_u1846_seed20261197_20260912_191600}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$SOURCE_RUN/archive_19_v97_s1_single_outcome_update_0500.pkl}
SOURCE_PROGRESS_CSV=${SOURCE_PROGRESS_CSV:-$SOURCE_RUN/curriculum_progress.csv}
EXPECTED_SOURCE_SHA256=e541c26fa891351e89b7e9f3a3cfa9c19bbf4759cbfa811006fc3370d0069a1a
N_ENVS=${N_ENVS:-1024}
N_STEPS=${N_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
SAME_STAGE_REEXEC_EVERY_UPDATES=${SAME_STAGE_REEXEC_EVERY_UPDATES:-500}
V99_EXPERIMENTAL_PROBE=${V99_EXPERIMENTAL_PROBE:-NO}
RUN_SEED=${RUN_SEED:-20261199}
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d_%H%M%S)}
PROFILE=fixed_contact_plane_vision_robust_qvel_rmp_v99
RESUME_STAGE=v99_s1_single_outcome_motion_bridge
RUN_ID=${RUN_ID:-${PROFILE}_gpu0_formal_1024x128_resume_v97_u500_seed${RUN_SEED}_${RUN_DATE}}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID}
EXPECTED_GPU_UUID=GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4
RECOVER_STAGE_REEXEC=${RECOVER_STAGE_REEXEC:-NO}
STAGE_REEXEC_RESUME_FROM=${STAGE_REEXEC_RESUME_FROM:-}
STAGE_REEXEC_EXPECTED_SHA256=${STAGE_REEXEC_EXPECTED_SHA256:-}
STAGE_REEXEC_START_STAGE=${STAGE_REEXEC_START_STAGE:-}
MEMORY_GUARD_DIR="$RUN_DIR.memory_guard"
if [[ "$RECOVER_STAGE_REEXEC" == YES ]]; then
  SOURCE_CHECKPOINT="$STAGE_REEXEC_RESUME_FROM"
  SOURCE_PROGRESS_CSV="$(dirname "$SOURCE_CHECKPOINT")/curriculum_progress.csv"
  EXPECTED_SOURCE_SHA256="$STAGE_REEXEC_EXPECTED_SHA256"
  RESUME_STAGE="$STAGE_REEXEC_START_STAGE"
  MEMORY_GUARD_DIR="$RUN_DIR.memory_guard_stage_reexec_${RUN_DATE}"
fi

if [[ ${CONFIRM_GPU0_READY:-NO} == YES && ${PP_MEMORY_SCOPE_ACTIVE:-NO} != YES ]]; then
  [[ $(tmux display-message -p '#S' 2>/dev/null || true) == pp_gpu0 ]] || {
    echo "formal/probe V99 must launch inside tmux session pp_gpu0" >&2; exit 2;
  }
  exec "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" \
    --memory-high-gib 13 --memory-max-gib 14 --memory-stop-gib 12.5 \
    --host-stop-available-gib 4 --host-stop-streak-samples 3 \
    --log-dir "$MEMORY_GUARD_DIR" -- bash "$0" "$@"
fi
if [[ ${CONFIRM_GPU0_READY:-NO} == YES ]]; then
  "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" --verify-only \
    --memory-high-gib 13 --memory-max-gib 14 --memory-stop-gib 12.5 \
    --host-stop-available-gib 4 --host-stop-streak-samples 3
fi

[[ "$N_ENVS" == 1024 ]] || { echo "V99 fixes N_ENVS=1024" >&2; exit 2; }
[[ "$N_STEPS" == 128 ]] || { echo "V99 fixes N_STEPS=128" >&2; exit 2; }
[[ "$MINIBATCH_SIZE" == 16384 ]] || { echo "V99 fixes MINIBATCH_SIZE=16384" >&2; exit 2; }
[[ "$SAME_STAGE_REEXEC_EVERY_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
  echo "V99 requires a positive same-stage reexec interval" >&2; exit 2;
}
if [[ "$V99_EXPERIMENTAL_PROBE" == YES ]]; then
  [[ "$RECOVER_STAGE_REEXEC" == NO ]] || {
    echo "V99 probe cannot use same-stage recovery" >&2; exit 2;
  }
  [[ "${MAX_STAGE_UPDATES:-0}" =~ ^([1-9]|[1-5][0-9]|60)$ ]] || {
    echo "V99 probe is capped at 1--60 updates" >&2; exit 2;
  }
  [[ "${MAX_STAGES:-50}" == 18 ]] || {
    echo "V99 probe may expose only its motion bridge" >&2; exit 2;
  }
  [[ "${WANDB_MODE:-offline}" == offline && "$RUN_ID" == *probe* ]] || {
    echo "V99 probe requires offline W&B and a probe run identity" >&2; exit 2;
  }
fi
[[ -x "$PYTHON_BIN" && -f "$XML_PATH" ]] || exit 2
[[ -f "$RMP_OUTPUT_REPLAY_REPORT" && -f "$BALL_MASS_MEASUREMENT_MANIFEST" ]] || exit 2
[[ -f "$SOURCE_CHECKPOINT" && -f "$SOURCE_PROGRESS_CSV" ]] || {
  echo "missing pinned V97-u500 checkpoint/progress pair" >&2; exit 2;
}
source_sha256=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
[[ "$source_sha256" == "$EXPECTED_SOURCE_SHA256" ]] || {
  echo "V99 source SHA-256 mismatch" >&2; exit 2;
}
[[ $(dirname "$SOURCE_CHECKPOINT") == $(dirname "$SOURCE_PROGRESS_CSV") ]] || {
  echo "V99 source checkpoint and progress CSV must be siblings" >&2; exit 2;
}
if [[ "$RECOVER_STAGE_REEXEC" == YES ]]; then
  [[ -d "$RUN_DIR" && -n "$STAGE_REEXEC_RESUME_FROM" \
     && -n "$STAGE_REEXEC_EXPECTED_SHA256" \
     && -n "$STAGE_REEXEC_START_STAGE" ]] || {
    echo "V99 same-stage recovery requires its existing run, checkpoint, SHA and stage" >&2; exit 2;
  }
else
  [[ -z "$STAGE_REEXEC_RESUME_FROM" && -z "$STAGE_REEXEC_EXPECTED_SHA256" \
     && -z "$STAGE_REEXEC_START_STAGE" ]] || {
    echo "V99 recovery inputs require RECOVER_STAGE_REEXEC=YES" >&2; exit 2;
  }
  [[ ! -e "$RUN_DIR" ]] || {
    echo "refusing to overwrite: $RUN_DIR" >&2; exit 2;
  }
fi
actual_gpu_uuid=$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]') || {
  echo "GPU0 driver unavailable; refusing CPU fallback" >&2; exit 2;
}
[[ "$actual_gpu_uuid" == "$EXPECTED_GPU_UUID" ]] || { echo "GPU0 UUID mismatch" >&2; exit 2; }

echo "GPU0 V99 audit: V97-u500 pre-collapse state; staged racket-XY objective repair"
git -C "$REPO_ROOT" status --short --branch --untracked-files=no
nvidia-smi -i 0 --query-gpu=index,uuid,name,temperature.gpu,memory.used,memory.total --format=csv,noheader
sha256sum "$SOURCE_CHECKPOINT" "$SOURCE_PROGRESS_CSV" "$XML_PATH" "$RMP_OUTPUT_REPLAY_REPORT" "$BALL_MASS_MEASUREMENT_MANIFEST"
echo "Run directory: $RUN_DIR"
echo "Geometry: 1024x128, minibatch 16384, 2 epochs; block validation covers all 36 XY bins."
echo "Memory: 96-row history, 500-update reexec; MemoryHigh=13 GiB, MemoryMax=14 GiB, stop=12.5 GiB, dual-GPU host floor=4 GiB for 3 samples."
echo "Acceptance: bridge reaches <=85 mm/s and >=0.08 all-ready before unchanged 50 mm/s commit gates."
[[ ${CONFIRM_GPU0_READY:-NO} == YES ]] || { echo "Preflight only" >&2; exit 3; }
[[ ${ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE:-NO} == YES ]] || exit 2

mkdir -p "$RUN_DIR" "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MPLCONFIGDIR="$RUN_DIR/matplotlib" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb" WANDB_CACHE_DIR="$RUN_DIR/wandb_cache" WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
cd "$RL_SIM"
exec > >(tee -a "$RUN_DIR/stdout_stderr.log") 2>&1
SOURCE_FREEZE_RUN_DIR="$RUN_DIR"
RESUME_HISTORY_ARGS=()
if [[ "$RECOVER_STAGE_REEXEC" == YES ]]; then
  SOURCE_FREEZE_RUN_DIR="$RUN_DIR/stage_reexec_recovery_source_${RUN_DATE}"
  mkdir -p "$SOURCE_FREEZE_RUN_DIR"
  RESUME_HISTORY_ARGS=(--resume-curriculum-state)
  echo "Recovering V99 same stage with actor, critic, Adam and bounded history: checkpoint=$SOURCE_CHECKPOINT stage=$RESUME_STAGE"
fi
"$PYTHON_BIN" "$RL_SIM/freeze_training_source.py" \
  --repo-root "$REPO_ROOT" --run-dir "$SOURCE_FREEZE_RUN_DIR" --launcher "$LAUNCHER_PATH"
exec "$PYTHON_BIN" -u train_juggle_mjx_curriculum.py \
  --xml "$XML_PATH" --save-dir "$RUN_DIR" --seed "$RUN_SEED" \
  --curriculum-profile "$PROFILE" --curriculum-gate-preset v7_strict \
  --delay-ablation-preset baseline_current --measured-ball-mass-range-kg 0.0039 0.0041 \
  --ball-mass-measurement-manifest "$BALL_MASS_MEASUREMENT_MANIFEST" \
  --rmp-output-replay-report "$RMP_OUTPUT_REPLAY_REPORT" --allow-experimental-rmp-dr-bypass \
  --right-arm-pd-profile recovered_rmp_rmpmd_v2 --actuator-compensation-mode none \
  --no-arm-servo-target-tracking-planner --asymmetric-critic --critic-command-history-steps 12 \
  --no-allow-obs-dim-migration --resume-from "$SOURCE_CHECKPOINT" \
  --resume-start-stage "$RESUME_STAGE" \
  "${RESUME_HISTORY_ARGS[@]}" \
  --fixed-contact-cartesian-v99-u500-motion-repair \
  --n-envs "$N_ENVS" --n-steps "$N_STEPS" --minibatch-size "$MINIBATCH_SIZE" \
  --update-epochs 2 --learning-rate 1e-4 --gamma 0.9995 --gae-lambda 0.99 \
  --time-limit-bootstrap --clip-range 0.10 --target-kl 0.006 --ent-coef 0.0001 \
  --min-log-std -4.0 --max-log-std -2.8 --vf-coef 0.5 --max-grad-norm 0.5 --hidden-dim 256 \
  --compact-rollout-metrics --reexec-between-stages --reexec-from-frozen-source \
  --same-stage-reexec-every-updates "$SAME_STAGE_REEXEC_EVERY_UPDATES" \
  --max-in-memory-stage-history-rows 96 \
  --clear-jax-caches-between-stages --reuse-advance-validation-runtime \
  --advance-mode converged --convergence-window 24 --convergence-min-episodes 128 \
  --min-stage-updates 40 --stage-metric-warmup-updates 10 \
  --max-stage-updates "${MAX_STAGE_UPDATES:-0}" --max-stages "${MAX_STAGES:-50}" \
  --advance-validation-mode block --advance-eval-n-envs 36 --advance-eval-steps 1200 \
  --advance-eval-min-episodes 36 --advance-eval-retry-updates 20 \
  --advance-eval-hit-ratio 1.0 --advance-eval-len-ratio 1.0 --advance-eval-min-hits 0.0 \
  --advance-eval-min-len-frac 0.0 --advance-eval-min-return -1000000 \
  --advance-eval-hit-rate-margin 0.0 --advance-eval-hit-interval-margin 0.0 \
  --advance-eval-cond-hit-ratio 1.0 --advance-eval-camera-margin 0.0 \
  --advance-eval-ball-view-margin 0.0 --advance-eval-z-ideal-margin 0.0 \
  --advance-eval-reset-bucket-mode cvar --advance-eval-reset-bucket-min-episodes 2 \
  --advance-eval-reset-bucket-cvar-frac 0.25 --advance-eval-reset-bucket-rate-margin 0.0 \
  --advance-eval-reset-bucket-hit-margin 0.0 --save-every-updates 5 --archive-every-updates 500 \
  --gpu-max-temp-c 78 --gpu-check-every-updates 1 --wandb --wandb-mode "${WANDB_MODE:-offline}" \
  --wandb-metrics-only --wandb-project pingpong-mjx --wandb-name "$RUN_ID" \
  --wandb-id "fcv99-1024x128-${RUN_SEED}-${RUN_DATE}" --wandb-resume allow \
  --wandb-tags gpu0 fixed-v99 1024-env 128-step motion-gate-alignment bounded-history periodic-reexec cartesian-hard-qref yaw-free measured-qvel recovered-rmp
