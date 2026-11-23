#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
LAUNCHER_PATH=$(realpath "$0")
PYTHON_BIN=${PYTHON_BIN:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}
XML_PATH=${XML_PATH:-$RL_SIM/moz1_pd.xml}
RMP_OUTPUT_REPLAY_REPORT=${RMP_OUTPUT_REPLAY_REPORT:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json}
BALL_MASS_MEASUREMENT_MANIFEST=${BALL_MASS_MEASUREMENT_MANIFEST:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json}
SOURCE_RUN=${SOURCE_RUN:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/fixed_contact_plane_vision_robust_qvel_rmp_v98_gpu1_formal_1024x128_resume_v96_u1540_seed20261198_20260912_193100}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-$SOURCE_RUN/mjx_curriculum_best.pkl}
SOURCE_PROGRESS_CSV=${SOURCE_PROGRESS_CSV:-$SOURCE_RUN/curriculum_progress.csv}
EXPECTED_SOURCE_SHA256=9f38d884e9f6108f106bf25b358c427fe9c79e9050a6c70f82e6ebd992dba0c5
GPU0_V99_PROGRESS_CSV=${GPU0_V99_PROGRESS_CSV:-}
GPU0_V99_BASE_STEP=1488519168
MIN_GPU0_V99_UPDATES=50
MAX_GPU0_V99_FALLBACK_RATE=0.001
N_ENVS=${N_ENVS:-1024}
N_STEPS=${N_STEPS:-128}
MINIBATCH_SIZE=${MINIBATCH_SIZE:-16384}
SAME_STAGE_REEXEC_EVERY_UPDATES=${SAME_STAGE_REEXEC_EVERY_UPDATES:-500}
V101_EXPERIMENTAL_PROBE=${V101_EXPERIMENTAL_PROBE:-NO}
RUN_SEED=${RUN_SEED:-20261201}
RUN_DATE=${RUN_DATE:-$(date +%Y%m%d_%H%M%S)}
PROFILE=fixed_contact_plane_vision_robust_qvel_rmp_v101
RESUME_STAGE=v101_s2_hard_lane_focus_bridge
RUN_ID=${RUN_ID:-${PROFILE}_gpu1_formal_1024x128_resume_v98_u1978_seed${RUN_SEED}_${RUN_DATE}}
RUN_DIR=${RUN_DIR:-$REPO_ROOT/pingpong_controller/outputs/rl_sim/$RUN_ID}
EXPECTED_GPU_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
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

if [[ ${CONFIRM_GPU1_READY:-NO} == YES && ${PP_MEMORY_SCOPE_ACTIVE:-NO} != YES ]]; then
  [[ $(tmux display-message -p '#S' 2>/dev/null || true) == pp_gpu1 ]] || {
    echo "formal/probe V101 must launch inside tmux session pp_gpu1" >&2; exit 2;
  }
  exec "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" \
    --memory-high-gib 13 --memory-max-gib 14 --memory-stop-gib 12.5 \
    --host-stop-available-gib 4 --host-stop-streak-samples 3 \
    --log-dir "$MEMORY_GUARD_DIR" -- bash "$0" "$@"
fi
if [[ ${CONFIRM_GPU1_READY:-NO} == YES ]]; then
  "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" --verify-only \
    --memory-high-gib 13 --memory-max-gib 14 --memory-stop-gib 12.5 \
    --host-stop-available-gib 4 --host-stop-streak-samples 3
fi

[[ "$N_ENVS" == 1024 ]] || { echo "V101 fixes N_ENVS=1024" >&2; exit 2; }
[[ "$N_STEPS" == 128 ]] || { echo "V101 fixes N_STEPS=128" >&2; exit 2; }
[[ "$MINIBATCH_SIZE" == 16384 ]] || { echo "V101 fixes MINIBATCH_SIZE=16384" >&2; exit 2; }
[[ "$SAME_STAGE_REEXEC_EVERY_UPDATES" =~ ^[1-9][0-9]*$ ]] || {
  echo "V101 requires a positive same-stage reexec interval" >&2; exit 2;
}
if [[ "$V101_EXPERIMENTAL_PROBE" == YES ]]; then
  [[ "$RECOVER_STAGE_REEXEC" == NO ]] || {
    echo "V101 probe cannot use same-stage recovery" >&2; exit 2;
  }
  [[ "${MAX_STAGE_UPDATES:-0}" =~ ^([1-9]|[1-5][0-9]|60)$ ]] || {
    echo "V101 probe is capped at 1--60 updates" >&2; exit 2;
  }
  [[ "${MAX_STAGES:-50}" == 17 ]] || {
    echo "V101 probe may expose only its hard-lane focus bridge" >&2; exit 2;
  }
  [[ "${WANDB_MODE:-offline}" == offline && "$RUN_ID" == *probe* ]] || {
    echo "V101 probe requires offline W&B and a probe run identity" >&2; exit 2;
  }
fi
[[ -x "$PYTHON_BIN" && -f "$XML_PATH" ]] || exit 2
[[ -f "$RMP_OUTPUT_REPLAY_REPORT" && -f "$BALL_MASS_MEASUREMENT_MANIFEST" ]] || exit 2
[[ -f "$SOURCE_CHECKPOINT" && -f "$SOURCE_PROGRESS_CSV" ]] || {
  echo "missing pinned V98-u1978 checkpoint/progress pair" >&2; exit 2;
}
source_sha256=$(sha256sum "$SOURCE_CHECKPOINT" | awk '{print $1}')
[[ "$source_sha256" == "$EXPECTED_SOURCE_SHA256" ]] || {
  echo "V101 source SHA-256 mismatch" >&2; exit 2;
}
[[ $(dirname "$SOURCE_CHECKPOINT") == $(dirname "$SOURCE_PROGRESS_CSV") ]] || {
  echo "V101 source checkpoint and progress CSV must be siblings" >&2; exit 2;
}
if [[ "$RECOVER_STAGE_REEXEC" == YES ]]; then
  [[ -d "$RUN_DIR" && -n "$STAGE_REEXEC_RESUME_FROM" \
     && -n "$STAGE_REEXEC_EXPECTED_SHA256" \
     && -n "$STAGE_REEXEC_START_STAGE" ]] || {
    echo "V101 same-stage recovery requires its existing run, checkpoint, SHA and stage" >&2; exit 2;
  }
else
  [[ -z "$STAGE_REEXEC_RESUME_FROM" && -z "$STAGE_REEXEC_EXPECTED_SHA256" \
     && -z "$STAGE_REEXEC_START_STAGE" ]] || {
    echo "V101 recovery inputs require RECOVER_STAGE_REEXEC=YES" >&2; exit 2;
  }
  [[ ! -e "$RUN_DIR" ]] || {
    echo "refusing to overwrite: $RUN_DIR" >&2; exit 2;
  }
fi
actual_gpu_uuid=$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d '[:space:]') || {
  echo "GPU1 driver unavailable; refusing CPU fallback" >&2; exit 2;
}
[[ "$actual_gpu_uuid" == "$EXPECTED_GPU_UUID" ]] || { echo "GPU1 UUID mismatch" >&2; exit 2; }

if [[ ${CONFIRM_GPU1_READY:-NO} == YES && "$V101_EXPERIMENTAL_PROBE" != YES ]]; then
  [[ -n "$GPU0_V99_PROGRESS_CSV" && -f "$GPU0_V99_PROGRESS_CSV" ]] || {
    echo "V101 requires the selected formal GPU0 V99 progress CSV" >&2; exit 2;
  }
  "$PYTHON_BIN" -c 'import csv,math,sys
path,base_text,count_text,fallback_limit_text=sys.argv[1:]
base=int(base_text); required=int(count_text); steps_per_update=1024*128
fallback_limit=float(fallback_limit_text)
rows=[]
for row in csv.DictReader(open(path,encoding="utf-8")):
    try: step=int(row["global_step"])
    except (KeyError,TypeError,ValueError): continue
    if row.get("stage_name","").startswith("v99_") and step>base:
        rows.append((step,row))
def finite(row,key):
    try: return math.isfinite(float(row[key]))
    except (KeyError,TypeError,ValueError): return False
def clean(row):
    return (finite(row,"mean_return") and finite(row,"approx_kl")
        and float(row.get("ppo_update_rolled_back",0) or 0)<0.5
        and float(row.get("cartesian_pose_projection_current_safe",0) or 0)>=1.0-fallback_limit-1e-12
        and float(row.get("cartesian_pose_qref_position_violation_m",1) or 1)<=2e-5
        and float(row.get("cartesian_pose_qref_orientation_violation_rad",1) or 1)<=2e-5
        and float(row.get("cartesian_pose_projection_fallback_to_hold",1) or 1)<=fallback_limit+1e-12)
last_steps=[step for step,_ in rows[-10:]]
ok=(len(rows)>=required and len(last_steps)==10
    and max((step for step,_ in rows),default=base)>=base+required*steps_per_update
    and all(clean(row) for _,row in rows[-10:])
    and all(b==a+steps_per_update for a,b in zip(last_steps,last_steps[1:])))
raise SystemExit(0 if ok else 2)' \
    "$GPU0_V99_PROGRESS_CSV" "$GPU0_V99_BASE_STEP" "$MIN_GPU0_V99_UPDATES" \
    "$MAX_GPU0_V99_FALLBACK_RATE" || {
      echo "V101 blocked until formal GPU0 V99 has >=50 clean 1024x128 updates" >&2; exit 2;
    }
  echo "GPU0 V99 stagger/order evidence valid; GPU1 V101 gate passed"
elif [[ ${CONFIRM_GPU1_READY:-NO} == YES ]]; then
  echo "Bounded V101 experimental probe; formal GPU0 order gate remains active"
fi

echo "GPU1 V101 audit: V98-u1978 best state; full-domain low-X/high-Y advantage focus"
git -C "$REPO_ROOT" status --short --branch --untracked-files=no
nvidia-smi -i 1 --query-gpu=index,uuid,name,temperature.gpu,memory.used,memory.total --format=csv,noheader
sha256sum "$SOURCE_CHECKPOINT" "$SOURCE_PROGRESS_CSV" "$XML_PATH" "$RMP_OUTPUT_REPLAY_REPORT" "$BALL_MASS_MEASUREMENT_MANIFEST"
if [[ -n "$GPU0_V99_PROGRESS_CSV" ]]; then
  sha256sum "$GPU0_V99_PROGRESS_CSV"
fi
echo "Run directory: $RUN_DIR"
echo "Geometry: 1024x128, minibatch 16384, 2 epochs; full 6x6 XY support remains active."
echo "Focus: low-X<-0.15 and high-Y>0.08 requires both conditions; advantage weight=4.0."
echo "Memory: 96-row history, 500-update reexec; MemoryHigh=13 GiB, MemoryMax=14 GiB, stop=12.5 GiB, dual-GPU host floor=4 GiB for 3 samples."
echo "Acceptance: 36-bin hard-lane recovery without global two-contact or safety regression."
[[ ${CONFIRM_GPU1_READY:-NO} == YES ]] || { echo "Preflight only" >&2; exit 3; }
[[ ${ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE:-NO} == YES ]] || exit 2

mkdir -p "$RUN_DIR" "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
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
  echo "Recovering V101 same stage with actor, critic, Adam and bounded history: checkpoint=$SOURCE_CHECKPOINT stage=$RESUME_STAGE"
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
  --fixed-contact-cartesian-v101-u1978-hard-lane-focus-repair \
  --n-envs "$N_ENVS" --n-steps "$N_STEPS" --minibatch-size "$MINIBATCH_SIZE" \
  --update-epochs 2 --learning-rate 5e-5 --gamma 0.9995 --gae-lambda 0.995 \
  --time-limit-bootstrap --clip-range 0.10 --target-kl 0.004 --ent-coef 0.0001 \
  --min-log-std -4.8 --max-log-std -2.8 --vf-coef 0.5 --max-grad-norm 0.5 --hidden-dim 256 \
  --hard-lane-focus-condition reset_target_x low -0.15 \
  --hard-lane-focus-condition reset_target_y high 0.08 \
  --hard-lane-focus-min-conditions 2 --hard-lane-focus-weight 4.0 \
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
  --wandb-id "fcv101-1024x128-${RUN_SEED}-${RUN_DATE}" --wandb-resume allow \
  --wandb-tags gpu1 fixed-v101 1024-env 128-step full-domain hard-lane-focus complete-36-bin bounded-history periodic-reexec cartesian-hard-qref yaw-free measured-qvel recovered-rmp
