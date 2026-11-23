#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && ( $1 == --formal || $1 == --probe ) ]] || exit 2
KIND=${1#--}
REPO_ROOT=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO_ROOT/pingpong_controller/tools/rl_sim"
OUTPUT_ROOT="$REPO_ROOT/pingpong_controller/outputs/rl_sim"
DESIGN="$OUTPUT_ROOT/v225_near_side_design_20261016_r1"
PYTHON_BIN=/home/yangzhe/miniconda3/envs/pingpong/bin/python
PROFILE=fixed_plane_limit_cycle_qvel_rmp_v225
GPU_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
SOURCE="$OUTPUT_ROOT/fplc_v172_gpu1_quality_gates_formal_20261009_r1/10_fplc_v159_gpu1_s10_clean_commit.pkl"
SOURCE_SHA=96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a
XML="$RL_SIM/moz1_pd_1ms_v98.xml"
RUN_ID=${RUN_ID:-fplc_v225_gpu1_near_side_${KIND}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
[[ $RUN_ID =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing existing output: $RUN_DIR" >&2; exit 2; }
[[ -n ${TMUX:-} && $(tmux display-message -p -t "$TMUX_PANE" '#S') == pp_gpu1 ]] || exit 2
[[ $(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]') == "$GPU_UUID" ]] || exit 2
[[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]] || exit 2
GPU_TEMP=$(nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
DISK_KIB=$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')
(( GPU_TEMP < 78 && AVAILABLE_KIB >= 12*1024*1024 && DISK_KIB >= 5*1024*1024 )) || exit 2
[[ $(sha256sum "$XML" | awk '{print $1}') == ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a ]] || exit 2
[[ $(sha256sum "$SOURCE" | awk '{print $1}') == "$SOURCE_SHA" ]] || exit 2
ADMISSION_ARGS=()
if [[ $KIND == formal ]]; then
    [[ -n ${V225_ADMISSION:-} && -f $V225_ADMISSION ]] || { echo 'V225 requires hashed probe admission' >&2; exit 2; }
    ADMISSION_ARGS=(--admission "$V225_ADMISSION")
fi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    "$PYTHON_BIN" "$RL_SIM/preflight_fixed_plane_v225.py" --checkpoint "$SOURCE" \
    --checkpoint-sha "$SOURCE_SHA" "${ADMISSION_ARGS[@]}" --report "$RUN_DIR/stability_preflight.json"
mkdir -p "$RUN_DIR/matplotlib" "$RUN_DIR/wandb" "$RUN_DIR/wandb_cache" "$RUN_DIR/wandb_config"
if [[ $KIND == formal ]]; then cp "$V225_ADMISSION" "$RUN_DIR/formal_admission.json"; fi
nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.total --format=csv,noheader > "$RUN_DIR/gpu_identity.txt"
"$PYTHON_BIN" "$RL_SIM/freeze_training_source.py" --repo-root "$REPO_ROOT" --run-dir "$RUN_DIR" --launcher "${BASH_SOURCE[0]}"
git -C "$REPO_ROOT" status --porcelain=v1 --untracked-files=no > "$RUN_DIR/tracked_worktree_status.txt"
sha256sum "$SOURCE" "$XML" "$RUN_DIR/source_snapshot/source_snapshot_manifest.json" > "$RUN_DIR/source_manifest.sha256"
RMP_REPORT="$OUTPUT_ROOT/rmp_measured_qvel_design_20260820/replay_datatracer_training_local_v2.json"
MASS_MANIFEST="$OUTPUT_ROOT/gpu1_v143_fixed_base_ball4g_20260830/gpu1_ball_mass_user_attestation_20260830.json"
REPLAY_OBS="$OUTPUT_ROOT/v150_gpu1_preflight_20261006_r3/replay_obs_72d.npy"
REPLAY_ACTIONS="$OUTPUT_ROOT/v150_gpu1_preflight_20261006_r3/replay_actions_7d.npy"
[[ $(sha256sum "$REPLAY_OBS" | awk '{print $1}') == 793575f1da785c295ed110a29e1d18052f4f9fa5edc14a332079e534fb210bc3 ]] || exit 2
[[ $(sha256sum "$REPLAY_ACTIONS" | awk '{print $1}') == 131253156dd2084f6062fc0846cd14a6b411fb502866f2fdf65a7897dfa72439 ]] || exit 2
MAX_UPDATES=-1
MIN_UPDATES=30
if [[ $KIND == probe ]]; then
    MAX_UPDATES=16
    MIN_UPDATES=116
fi
cd "$RL_SIM"
CMD=("$PYTHON_BIN" -u train_juggle_mjx_curriculum.py --xml "$XML" --save-dir "$RUN_DIR" --seed "${V225_SEED:-20266510}"
 --curriculum-profile "$PROFILE" --no-allow-obs-dim-migration --resume-from "$SOURCE" --resume-start-stage fplc_v225_s01_workspace_alignment
 --conditional-reset-done --compact-rollout-metrics --bounded-validation-memory --release-startup-compilation-caches
 --reexec-between-stages --reexec-from-frozen-source --max-in-memory-stage-history-rows 96 --same-stage-reexec-every-updates 500
 --counterfactual-replay-obs "$REPLAY_OBS" --counterfactual-replay-actions "$REPLAY_ACTIONS"
 --counterfactual-supervision-coef .05 --counterfactual-vxy-weight-mode uniform
 --curriculum-gate-preset v7_strict --delay-ablation-preset baseline_current
 --measured-ball-mass-range-kg .0039 .0041 --ball-mass-measurement-manifest "$MASS_MANIFEST"
 --rmp-output-replay-report "$RMP_REPORT" --allow-experimental-rmp-dr-bypass
 --right-arm-pd-profile recovered_rmp_rmpmd_v2 --actuator-compensation-mode none --no-arm-servo-target-tracking-planner
 --asymmetric-critic --critic-command-history-steps 12
 --n-envs 1024 --n-steps 128 --minibatch-size 16384 --update-epochs 4
 --learning-rate 1.5e-5 --gamma .9995 --gae-lambda .99 --time-limit-bootstrap
 --clip-range .12 --target-kl .006 --ent-coef .001 --min-log-std -3.0 --vf-coef .05 --max-grad-norm .5 --hidden-dim 256
 --advance-mode converged --convergence-window 32 --convergence-min-episodes 8 --min-stage-updates "$MIN_UPDATES" --stage-metric-warmup-updates 10
 --max-stage-updates "$MAX_UPDATES" --max-stages 12 --clear-jax-caches-between-stages
 --advance-validation-mode block --advance-eval-stochastic --advance-eval-n-envs 288 --advance-eval-steps 1200 --advance-eval-min-episodes 144
 --advance-eval-retry-updates 20 --advance-eval-hit-ratio 1 --advance-eval-len-ratio 1 --advance-eval-min-hits 1 --advance-eval-min-len-frac .10
 --advance-eval-min-return -1000000 --advance-eval-hit-rate-margin 0 --advance-eval-hit-interval-margin .03 --advance-eval-cond-hit-ratio 1
 --advance-eval-camera-margin 0 --advance-eval-ball-view-margin 0 --advance-eval-z-ideal-margin 0
 --advance-eval-reset-bucket-mode cvar --advance-eval-reset-bucket-min-episodes 8 --advance-eval-reset-bucket-cvar-frac .25
 --advance-eval-reset-bucket-rate-margin .05 --advance-eval-reset-bucket-hit-margin 1
 --advance-retention-stage-name fplc_v225_s01_workspace_alignment
 --save-every-updates 10 --archive-every-updates 100 --gpu-max-temp-c 78 --gpu-check-every-updates 1
 --wandb --wandb-mode offline --wandb-resume never --wandb-id "$RUN_ID" --wandb-metrics-only --wandb-project pingpong-mjx --wandb-name "$RUN_ID"
 --wandb-tags gpu1 fixed-plane-v225 near-side-recovery unchanged72d clean-v172 preserved-physics)
printf '%q ' "${CMD[@]}" > "$RUN_DIR/command.sh"
printf '\n' >> "$RUN_DIR/command.sh"
exec "$PYTHON_BIN" "$RL_SIM/training_memory_scope.py" --log-dir "$RUN_DIR/memory_scope" \
 --memory-high-gib 10 --memory-max-gib 12 --memory-stop-gib 9.5 --host-stop-available-gib 4 --host-stop-streak-samples 3 \
 -- env GUARD_SWAP_PRESSURE_AVAILABLE_KB=4194304 \
 "$RL_SIM/run_with_host_memory_guard.sh" "$RUN_DIR/stdout_stderr.log" \
 env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=$GPU_UUID" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 \
 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 "MPLCONFIGDIR=$RUN_DIR/matplotlib" \
 "JAX_COMPILATION_CACHE_DIR=$DESIGN/compilation_cache" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180 \
 "WANDB_DIR=$RUN_DIR/wandb" "WANDB_CACHE_DIR=$RUN_DIR/wandb_cache" "WANDB_CONFIG_DIR=$RUN_DIR/wandb_config" \
 "${CMD[@]}"
