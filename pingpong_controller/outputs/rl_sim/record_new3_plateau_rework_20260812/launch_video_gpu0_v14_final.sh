#!/usr/bin/env bash
set -euo pipefail

REPO=/home/yangzhe/Project/pingpong_controller
RL_SIM="$REPO/pingpong_controller/tools/rl_sim"
ROOT="$REPO/pingpong_controller/outputs/rl_sim/record_new3_plateau_rework_20260812"
RUN="$ROOT/formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online"
CHECKPOINT="$RUN/mjx_curriculum_last.pkl"
OUT="$RUN/final_video"

[[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
mkdir -p "$OUT"

cd "$RL_SIM"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export MUJOCO_GL=egl

/home/yangzhe/miniconda3/envs/pingpong/bin/python -u validate_juggle_mjx_ppo.py \
  --checkpoint "$CHECKPOINT" --episodes 1 --n-envs 1 \
  --one-episode-per-env --seed 20260812 --deterministic \
  --ball-obs-rate-hz 60 --max-env-steps 1200 --print-every 200 \
  --dr-ball-mass-range 0.0037 0.0037 \
  --dr-ball-solref-time-range 0.005 0.005 \
  --dr-ball-solref-damping-range 0.90 0.90 \
  --results-csv "$OUT/validation.csv" \
  --action-trace-csv "$OUT/action_trace.csv" \
  --obs-trace-csv "$OUT/obs_trace.csv" \
  --action-plot-out "$OUT/action_plot.png" \
  --video-out "$OUT/gpu0_v14_final_heavy_3p7g_60hz.mp4" \
  --video-fps 30 --video-width 1280 --video-height 720 \
  --video-slowmo 1.0 --gpu-max-temp-c 80 \
  2>&1 | tee "$OUT/validate.log"

touch "$OUT/COMPLETED"
