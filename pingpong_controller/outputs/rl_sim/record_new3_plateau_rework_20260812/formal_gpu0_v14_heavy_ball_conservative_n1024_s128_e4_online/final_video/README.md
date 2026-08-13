# GPU0 QVEL V14 final deployment replay

This directory contains the compact visual evidence selected for the real-
robot deployment commit. The replay uses the released
`mjx_curriculum_last.pkl`, deterministic policy inference, seed `20260812`, a
fixed 3.7 g ball, contact `solref` time `0.005 s`, damping `0.90`, and a 60 Hz
fractional ball-observation stream.

Result: 1200/1200 steps, 19 hits, full/view/camera/hit-band rate `1.0`, mean
hit-vxy `0.057 m/s`, and normal horizon truncation only.

SHA-256:

- `gpu0_v14_final_heavy_3p7g_60hz.mp4`:
  `61032bed67774e5552f5de73a02f61460c94522a4a3cba8fa4137fde6c653816`
- `action_plot.png`:
  `5e85a4cae6f203f89b8d47176dcd79d014f1e06c5387a7eed66292526cfee208`

The much larger raw `action_trace.csv` and `obs_trace.csv` remain in the local
experiment archive and are intentionally excluded from the deployment commit.
