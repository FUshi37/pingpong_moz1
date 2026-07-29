# GPU0 measured-observation servo policy video

- Frozen checkpoint: `../mjx_curriculum_best.pkl`
- Checkpoint stage: 21, `launch19_final_measured_obsres2mm_servo_consolidation`
- Checkpoint stage/global update: 295 / 475
- Checkpoint global step: `5585240064`
- Checkpoint SHA-256:
  `ed495f4445c21b4af7fd1acba420ba3e76e68a86dfdf2b35a7201acd641559c4`
- Validation: deterministic policy, seed `20260729`, checkpoint final
  environment, one environment, 1200 control steps, physical GPU1.
- Result: 12 hits, 1200/1200 steps, full rate 1.0, hit-camera rate
  1.0, hit-band rate 1.0, no missing/lost observation, horizon truncation only.
- Video: `launch19_final_best_u295_seed20260729.mp4`, H.264, 1280x720,
  30 fps, 181 frames, 6.03 seconds.
- Video SHA-256:
  `5a513c93f35d3192965d3ecdc213ce5a7f51278eac2093fc252cd60702e6167f`
- Action/trajectory plot: `action_plot.png`. It contains raw/applied policy
  action, nominal/servo/actual normalized acceleration, velocity, joint-angle
  targets/feedback, and ball/racket height over the complete six seconds.
- Action plot SHA-256:
  `8231d109f9186ea1dac4d8e67a02d22d83be95ecc521011fdc4470117f8e621a`

The directory also contains the exact 200 Hz action/joint trace, policy
observation trace, episode metrics, action plot, and a representative frame.

## Reproduction

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID \
CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
MUJOCO_GL=egl \
/home/yangzhe/miniconda3/envs/pingpong/bin/python \
  pingpong_controller/tools/rl_sim/validate_juggle_mjx_ppo.py \
  --checkpoint pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/mjx_curriculum_best.pkl \
  --episodes 1 --n-envs 1 --seed 20260729 --deterministic \
  --max-env-steps 1200 --log-hit-events \
  --video-out pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/video/launch19_final_best_u295_seed20260729.mp4 \
  --video-fps 30 --video-width 1280 --video-height 720 \
  --results-csv pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/video/episode.csv \
  --action-trace-csv pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/video/action_trace.csv \
  --action-plot-out pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/video/action_plot.png \
  --obs-trace-csv pingpong_controller/outputs/rl_sim/goal_d455_gpu0_obsres2mm_servo_submission_20260729/video/obs_trace.csv
```
