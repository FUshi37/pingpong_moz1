#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO/pingpong_controller/tools/rl_sim"
ROOT="$REPO/pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812"
SOURCE="$ROOT/candidate_gpu0_v14_heavy_bridge_n1024_u24/mjx_curriculum_last.pkl"
RUN_TAG=formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online
RUN_DIR="$ROOT/$RUN_TAG"

[[ -f "$SOURCE" ]] || { echo "missing source: $SOURCE" >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "refusing to overwrite: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR/snapshot" "$RUN_DIR/final_promotion"
cp "$RL_SIM/mjx_juggle_env.py" "$RUN_DIR/snapshot/"
cp "$RL_SIM/train_juggle_mjx_curriculum.py" "$RUN_DIR/snapshot/"
cp "$RL_SIM/train_juggle_mjx_ppo.py" "$RUN_DIR/snapshot/"
cp "$RL_SIM/validate_juggle_mjx_ppo.py" "$RUN_DIR/snapshot/"
cp "$RL_SIM/moz1_pd.xml" "$RUN_DIR/snapshot/"
git -C "$REPO" status --short >"$RUN_DIR/snapshot/git_status.txt"
git -C "$REPO" diff -- \
  pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py \
  pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py \
  test/test_feedback_break_and_qvel_v11.py \
  >"$RUN_DIR/snapshot/relevant_code.diff"
sha256sum "$SOURCE" "$RL_SIM/moz1_pd.xml" >"$RUN_DIR/snapshot/artifacts.sha256"

{
  echo "tmux_session=pp_gpu0"
  echo "source_checkpoint=$SOURCE"
  echo "profile=goal_d455_sport_taskspace_qvel_vertical_v14"
  echo "resume_start_stage=23"
  echo "resume_curriculum_state=true"
  echo "heavy_ball_bridge_mass_kg=0.00290,0.00370"
  echo "heavy_ball_target_mass_kg=0.00345,0.00395"
  echo "heavy_ball_bridge_solref_damping=0.66,1.06"
  echo "heavy_ball_target_solref_damping=0.72,1.08"
  echo "hit_min_count_interval_s=0.22"
  echo "n_envs=1024"
  echo "n_steps=128"
  echo "max_stage_updates=None"
  echo "final_validate_len_frac=1.0"
  echo "final_validate_view_in_bounds=1.0"
  echo "wandb_mode=online"
  echo "real_robot_experiment=false"
} >"$RUN_DIR/snapshot/run_manifest.txt"

cd "$RL_SIM"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export WANDB_DISABLE_CODE=true
export WANDB_INIT_TIMEOUT=180
export WANDB_DIR="$RUN_DIR/wandb"
export WANDB_CACHE_DIR="$RUN_DIR/wandb_cache"
export WANDB_CONFIG_DIR="$RUN_DIR/wandb_config"
export GUARD_MIN_AVAILABLE_KB=$((3 * 1024 * 1024))
export GUARD_CRITICAL_AVAILABLE_KB=$((1536 * 1024))
export GUARD_LOW_MEMORY_GRACE_CHECKS=4
export GUARD_MAX_SWAP_GROWTH_KB=$((4 * 1024 * 1024))
export GUARD_SWAP_PRESSURE_AVAILABLE_KB=$((5 * 1024 * 1024))

./run_with_host_memory_guard.sh "$RUN_DIR/stdout_stderr.log" \
  /home/yangzhe/miniconda3/envs/pingpong/bin/python -u train_juggle_mjx_curriculum.py \
  --xml "$RL_SIM/moz1_pd.xml" --save-dir "$RUN_DIR" \
  --seed 61010 --n-envs 1024 --n-steps 128 --max-stages 24 \
  --curriculum-gate-preset v7_strict \
  --curriculum-profile goal_d455_sport_taskspace_qvel_vertical_v14 \
  --delay-ablation-preset sport_actuator_replay_dr \
  --resume-from "$SOURCE" --resume-start-stage 23 --resume-curriculum-state \
  --advance-mode converged --convergence-window 32 \
  --convergence-min-episodes 64 --min-stage-updates 0 \
  --minibatch-size 16384 --update-epochs 4 --learning-rate 2e-5 \
  --gamma 0.9995 --gae-lambda 0.99 --clip-range 0.12 --target-kl 0.008 \
  --min-log-std -3.2 --max-log-std -2.5 --ent-coef 0.0006 \
  --actor-anchor-kl-coef 0.015 \
  --failure-focus-hit-threshold 4 --failure-focus-weight 1.5 \
  --failure-focus-tail-steps 160 --asymmetric-critic \
  --critic-command-history-steps 12 --save-every-updates 5 \
  --archive-every-updates 25 --max-recent-rms-hit-vxy-safety 0.220 \
  --max-recent-rms-hit-racket-vxy-safety 0.300 \
  --advance-validation-mode block --advance-eval-n-envs 128 \
  --advance-eval-steps 1200 --advance-eval-min-episodes 64 \
  --advance-eval-final-min-len-frac 1.0 \
  --advance-eval-final-min-ball-view-in-bounds 1.0 \
  --gpu-max-temp-c 80 --gpu-check-every-updates 1 \
  --wandb --wandb-mode online --wandb-metrics-only \
  --wandb-project pingpong-mjx --wandb-name "gpu0-$RUN_TAG" \
  --wandb-tags gpu0 qvel-v14 heavy-ball lower-elasticity apex-rebalance \
  low-vxy low-angular n1024 conservative no-update-cap mirror-excluded

FINAL_CHECKPOINT="$RUN_DIR/mjx_curriculum_last.pkl"
/home/yangzhe/miniconda3/envs/pingpong/bin/python -u validate_juggle_mjx_ppo.py \
  --checkpoint "$FINAL_CHECKPOINT" --episodes 64 --n-envs 64 \
  --one-episode-per-env --seed 20260812 --deterministic \
  --dr-ball-mass-range 0.0037 0.0037 \
  --dr-ball-solref-time-range 0.005 0.005 \
  --dr-ball-solref-damping-range 0.90 0.90 \
  --max-env-steps 1200 --print-every 300 \
  --results-csv "$RUN_DIR/final_promotion/heavy_3p7g_damp0p90.csv" \
  --gpu-max-temp-c 80 \
  2>&1 | tee "$RUN_DIR/final_promotion/heavy_3p7g_damp0p90.log"
