# GPU0/GPU1 Plateau Rework Report — 2026-08-12

## Outcome

Both plateaued formal runs were stopped, audited, and replaced by
evidence-backed profiles.  A subsequent hit-credit audit found that the
0.32 s count debounce was also suppressing legitimate low-height juggling.
The current replacement jobs use 1024 environments, W&B online, no per-stage
update cap, and count-aligned profiles:

- GPU0: QVEL V13, V12's symmetric 0.18 m apex-band repair plus hit-credit
  alignment, conservative PPO settings.
- GPU1: record_new3 V98, V97's real timing/plant envelope, rebalanced
  view/contact credit and command smoothing plus hit-credit alignment, trained
  at 60 Hz.

The invalid record_new3 sim mirror was not used.  The only record_new3 robot
signals retained in the training design are the measured execution/receipt
timing, one-tick joint-state holds, and the previously identified plant range.

## Stopped runs and videos

| GPU | stopped checkpoint | deterministic video |
|---|---|---|
| GPU0 | `../record_new3_actual_execution_training_20260812/formal_gpu0_qvel_v11_dense_overdue_anchor_n1024_s128_e4_online/mjx_curriculum_interrupted.pkl` | `videos_gpu0_current/gpu0_interrupted_u5578.mp4` |
| GPU1 | `../record_new3_actual_execution_training_20260812/formal_gpu1_v96_actual_proprio_timing_anchor_60hz_n1024_s128_e4_online/mjx_curriculum_interrupted.pkl` | `videos_gpu1_current/gpu1_interrupted_u5509.mp4` |

Each video directory also contains `action_trace.csv`, `obs_trace.csv`, an
action plot, validation CSV/log, and a contact sheet.  Trace summaries are in
`current_video_trace_metrics.json` and generated reproducibly by
`../../../tools/rl_sim/analyze_plateau_rollout_trace.py`.

## Why the old runs plateaued

### GPU0

The run was stable rather than numerically unstable.  Over roughly 5.5k stage
updates, rolling hits changed only from 10.31 to 10.74, full episodes from
0.914 to 0.939, view occupancy stayed near 0.99, and mean hit-vxy improved only
from 0.101 to 0.098 m/s.  Approximate PPO KL stayed near 0.0017; the actor
anchor KL slowly reached about 0.04.  The only persistent gate failure was the
maximum hit interval.

The one-seed video exposed the failure mode directly: 8 hits, 0.779 s mean
interval, 1.49 s maximum interval, and only 0.055 m measured apex height above
the racket between counted hits.  The reward telemetry made low rebounds a
local optimum:

- `center_flat_hit`: +0.0218
- adaptive reflected-velocity loss: -0.0102
- hit-vxy loss: -0.0101
- post-hit overdue loss: -0.0058
- height loss: only about -0.00003

Thus the policy could collect contact/flatness credit with low rebounds and
long waits; the nominal 0.18 m target did not provide a comparable gradient.

### GPU1

The old final stage changed little over 5.5k updates: rolling hits stayed near
11.2, full rate near 0.72, view near 0.79, and mean hit-vxy near 0.166 m/s.
Approximate KL was only about 0.001 and value loss oscillated, but there was no
PPO clipping/instability signature.  This was an objective mismatch.

The deterministic trace had 26.6% of action power above 8 Hz versus 8.6% on
GPU0, action-delta RMS 0.115 versus 0.045, and contact angular-speed P95
1.36 rad/s.  The reward budget showed:

- hit-vxy loss: -0.0268
- center/flat hit credit: +0.0220
- hit bonus: +0.0143
- positive low-angular hit credit: +0.0119
- all view-centering/out-of-bounds losses combined: about -0.0037
- action-delta loss: about -0.000001
- delayed-action jerk loss: about -0.000007

The positive angular term was larger than all view losses, while command
continuity was effectively unpriced.  The policy therefore improved some hit
metrics without learning a sufficiently in-view, band-limited controller.

## Timing ablation: the 8% real joint hold is not the root cause

The GPU1 source/current policies were each evaluated with and without the
third-stage one-tick proprioceptive hold, using the same 64 seeds.

| policy | environment | hits | full rate | weighted view | mean hit-vxy |
|---|---:|---:|---:|---:|---:|
| source | source/no hold | 12.016 | 0.781 | 0.815 | 0.153 |
| source | timing/8% hold | 11.828 | 0.781 | 0.826 | 0.149 |
| old trained | source/no hold | 12.484 | 0.906 | 0.850 | 0.136 |
| old trained | timing/8% hold | 12.328 | 0.922 | 0.855 | 0.130 |

The hold changes mean hits by only 0.16--0.19 and does not reduce full rate.
It must remain in stage 3 because it is measured real timing coverage; deleting
it would not repair the plateau and would make the policy less representative
of record_new3 execution.

## Code changes

### QVEL V12

`goal_d455_sport_taskspace_qvel_vertical_v12` wraps V11 without changing its
plant, qdot action contract, DR, cadence, motion gates, or view gates.  It:

- caps center/flat contact reward at 0.80;
- raises symmetric predicted-apex error weight from 16 to at least 64;
- tightens the low-apex margin to 0.035 m and raises low-apex weight to 64;
- retains explicit first-hit apex credit.

### record_new3 V97

`goal_d455_sport_taskspace_record_new3_plateau_v97` keeps the 67-D actor,
60 Hz fractional ball observation, 8% one-tick joint hold in stage 3, and the
record_new3 actuator/second-order DR.  Across the three stages it:

- reduces positive angular reward to 0.70 / 0.50 / 0.35;
- caps center/flat reward at 1.20 / 1.00 / 0.80;
- raises view-XY cost to 0.30 / 0.45 / 0.60;
- raises out-of-view cost to 1.50 / 2.00 / 3.00;
- raises apex-view cost to 0.35 / 0.50 / 0.70;
- raises action-delta cost to 0.03 / 0.06 / 0.10;
- raises delayed-action jerk cost to 1e-6 / 2e-6 / 3e-6.

The change is reward/control regularization only; it does not use failed real
actions as labels or introduce a deployment-side patch.

## Hit-count and reward semantic audit

The concern about missed valid hits was correct.  `launched_upward_raw`
already requires a separated contact, positive vertical velocity, sufficient
relative height, and a sufficient predicted apex.  Despite passing that
physical/height confirmation, V12/V97 then rejected the event when it occurred
within 0.32 s of the previous confirmed event:

- `new_hit`, `hit_count`, all hit-conditioned bonuses and all hit-conditioned
  height/view/vxy/angular quality terms used only the filtered event;
- `fast_hit_penalty_weight` was 0.0 in both running stages, so the rejected
  event had no dedicated fast-hit penalty either;
- the result was therefore mostly silent loss of both positive credit and
  quality feedback, rather than a deliberate optimization against low-period
  juggling.

The last 32 old updates showed the scale of the mismatch:

| run | confirmed event/step | counted event/step | rejected / confirmed | old count-density period |
|---|---:|---:|---:|---:|
| GPU0 V12 | 0.013366 | 0.009132 | 31.68% | 0.548 s |
| GPU1 V97 | 0.012258 | 0.010469 | 14.60% | 0.478 s |

The GPU0 recorded rollout contained ordinary physical contact cycles of about
0.28--0.40 s (robust mean 0.321 s, median 0.318 s), as well as a genuine
0.06 s double-contact/recontact artifact.  Therefore the debounce was reduced
to 0.22 s rather than removed.  This admits the normal cycles while retaining
separation from contact chatter.  A residual `fast_hit_penalty_weight=0.20`
now explicitly penalizes only sub-0.22 s confirmed recontacts.

The new wrappers are
`goal_d455_sport_taskspace_qvel_vertical_v13` and
`goal_d455_sport_taskspace_record_new3_count_aligned_v98`.  They do not change
the height confirmation, target apex, plant/DR, PPO settings, 60 Hz GPU1
training rate, execution timing, or strict validation gates.

Four-update, 1024-environment same-checkpoint smoke tests produced:

| profile | confirmed event/step | counted event/step | rejected / confirmed | count-density period |
|---|---:|---:|---:|---:|
| GPU0 V13 | 0.012457 | 0.012440 | 0.138% | 0.402 s |
| GPU1 V98 | 0.011774 | 0.011772 | 0.016% | 0.425 s |

The first 10 formal updates confirm the same behavior.  Over updates 6--10,
GPU0 rejected 0.27% and GPU1 rejected 0.037% of height-confirmed events; mean
fast-hit penalty magnitude was only `2.9e-8` and `6.0e-9` per step.  Thus
ordinary low-height juggling now receives its hit/quality reward, while the
anti-chatter path remains active but almost never fires on normal trajectories.

`hit_dt3` still needs to be interpreted carefully: it is currently computed
as completed-episode duration divided by counted hits, not as the mean of
actual consecutive hit timestamps.  It includes time before the first and
after the last hit, so the old displayed 0.58 s was a hit-density proxy rather
than a measured bounce period.  The count-alignment repair removes its largest
bias, but the label/formula itself has not been changed during these running
jobs.

## Parameter experiments

Each candidate used 1024 environments, 128-step rollouts, four PPO epochs, and
24 updates from the same interrupted policy.  All validation rows below use 64
deterministic, same-seed episodes.

### GPU0

| candidate | LR | anchor | entropy | hits | full | view | mean/rms hit-vxy | predicted apex |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| old policy | 2e-5 | 0.03 | 8e-4 | 11.375 | 1.000 | 0.997 | 0.080 / 0.094 | 0.187 m |
| conservative | 2e-5 | 0.015 | 6e-4 | 10.969 | 1.000 | 0.998 | 0.083 / 0.098 | 0.189 m |
| adaptive | 4e-5 | 0.005 | 1e-3 | 10.547 | 1.000 | 0.998 | 0.076 / 0.088 | 0.185 m |

The high-LR candidate bought lower lateral speed by losing more hits and did
not improve apex height.  The conservative candidate was selected because it
preserved full episodes and view while moving the new objective without the
larger hit-count regression.  Formal training must continue longer than 24
updates to repair the 0.57--0.60 s orbit.

### GPU1

| candidate/rate | hits | full | weighted view | mean/rms hit-vxy |
|---|---:|---:|---:|---:|
| conservative 60 Hz | 12.016 | 0.859 | 0.845 | 0.132 / 0.173 |
| conservative 90 Hz | 12.453 | 0.953 | 0.884 | 0.111 / 0.136 |
| adaptive 60 Hz | 11.859 | 0.844 | 0.902 | 0.141 / 0.187 |
| adaptive 90 Hz | 12.563 | 0.938 | 0.931 | 0.111 / 0.136 |

Both candidates prove the required direction: 90 Hz is better than the 60 Hz
training rate.  The adaptive setting improved view but lost 60 Hz survival and
lateral quality.  The conservative setting was selected as the safer formal
optimizer; the unchanged strict gate still requires deterministic length=1 and
view=1 before final promotion.

## Tests

`python -m py_compile` passed for the modified trainer/environment/trace
analyzer.  The two relevant test files, including V13/V98 count-alignment
invariants, completed with 21 passing tests:

```text
21 passed, 13 warnings in 254.96s
```

The warnings are pre-existing MuJoCo contact-access deprecations.

## Formal training

| GPU | tmux | W&B | local run |
|---|---|---|---|
| GPU0 | `pp_gpu0` | `https://wandb.ai/fushi37/pingpong-mjx/runs/06wfos4k` | `formal_gpu0_v13_count_aligned_conservative_n1024_s128_e4_online/` |
| GPU1 | `pp_gpu1` | `https://wandb.ai/fushi37/pingpong-mjx/runs/651srnup` | `formal_gpu1_v98_count_aligned_conservative_60hz_n1024_s128_e4_online/` |

Both launch manifests record `n_envs=1024`, online W&B, selected hyperparameters,
source hashes, code snapshots, no real-robot experiment, and no stage update cap.
GPU1's post-convergence script performs paired 64-episode 60/90 Hz validation
and rejects promotion unless 90 Hz improves hits without view or RMS-vxy
regression.  V13 resumed from the V12 `mjx_curriculum_last.pkl` at update 2070;
V98 resumed from the V97 `mjx_curriculum_last.pkl` at update 2030.  Both current
processes reached full GPU utilization and completed their first 10 updates.

## 2026-08-12 long-run re-audit and measured heavy ball

Both formal runs were stopped safely by the host-memory guard rather than by
PPO/JAX failure.  GPU0 stopped at global update 1068 in V13 stage 22; GPU1
stopped at global update 1336 in V98 stage 3.  The old guard treated a 2 GiB
swap increase under 8 GiB available memory as fatal, even though the jobs still
had recoverable host-memory headroom.  Follow-up candidates use a 3 GiB normal
floor, 1.5 GiB critical floor, four-check grace period, and a 4 GiB swap-growth
limit.

The last 64 updates also confirm that the 0.22 s hit debounce is no longer a
material source of under-counting:

| run | confirmed event/step | ignored event/step | ignored / confirmed | hit-density period | mean fast-hit reward |
|---|---:|---:|---:|---:|---:|
| GPU0 V13 | 0.013236 | 0.0000528 | 0.399% | 0.399 s | -4.6e-8 |
| GPU1 V98 | 0.012209 | 0.0000041 | 0.033% | 0.456 s | -9.2e-9 |

Thus the count gate should not be removed: it now filters only rare
sub-0.22 s recontacts, while practically every height-confirmed normal bounce
receives both hit bonus and hit-conditioned quality feedback.  For comparison,
the superseded 0.32 s V12/V97 gates rejected 32.5% and 14.6%.  Future monitoring
must display `confirmed_hit`, `ignored_fast_hit`, and their ratio alongside
`mean_hits` so a count-density metric cannot silently masquerade as the physical
bounce period again.

The previous ball-mass DR was 2.45--2.95 g, so it excluded the measured real
ball (approximately 3.7 g).  A matched MuJoCo drop probe showed that increasing
positive `solref` damping from 0.82 to 0.90 modestly lowers rebound; mass alone
does not change fixed-plane restitution, but it does affect a moving, compliant
racket.  New continuation stages therefore use:

- bridge: mass 2.90--3.70 g, damping 0.66--1.06;
- target: mass 3.45--3.95 g, damping 0.72--1.08;
- unchanged contact time constant, friction, actuator, timing, observation,
  and all final validation gates.

Same-seed 64-episode fixed-physics validation before fine-tuning produced:

| policy / fixed ball | hits | length | full | weighted view | mean/rms hit-vxy | apex |
|---|---:|---:|---:|---:|---:|---:|
| GPU0, 2.7 g / 0.82 | 16.58 | 0.998 | 0.984 | 0.999 | 0.065 / 0.074 | 0.196 m |
| GPU0, 3.7 g / 0.82 | 16.45 | 1.000 | 1.000 | 1.000 | 0.067 / 0.077 | 0.196 m |
| GPU0, 3.7 g / 0.90 | 16.14 | 1.000 | 1.000 | 0.998 | 0.069 / 0.078 | 0.197 m |
| GPU1, 2.7 g / 0.82 | 14.50 | 0.959 | 0.906 | 0.932 | 0.132 / 0.149 | 0.232 m |
| GPU1, 3.7 g / 0.82 | 14.59 | 0.971 | 0.906 | 0.947 | 0.123 / 0.143 | 0.226 m |
| GPU1, 3.7 g / 0.90 | 13.97 | 0.949 | 0.875 | 0.918 | 0.134 / 0.156 | 0.231 m |

GPU0 is already robust to the measured ball distribution.  GPU1 is sensitive
primarily to lower rebound rather than mass: low elasticity reduces full/view
performance and raises low-ball, lateral-out, and racket-high failures.  This
justifies a physics-only bridge first, followed by a controlled comparison of
causal realized-next-contact guidance and modest PPO exploration.

### Heavy-ball bridge and reward/exploration ablation

All candidates resumed the stopped policies, reset Adam once for the new
bridge, used 1024 environments for 24 updates, and were evaluated with the same
seed at exactly 3.7 g / `solref` damping 0.90.  The short bridge results were:

| candidate / observation rate | hits | length | full | weighted view | mean/rms hit-vxy | next anchor |
|---|---:|---:|---:|---:|---:|---:|
| GPU0 V14 physics bridge, 60 Hz | 16.50 | 1.000 | 1.000 | 1.000 | 0.068 / 0.078 | 0.095 m |
| GPU1 V99 physics bridge, 60 Hz | 14.91 | 0.971 | 0.938 | 0.954 | 0.120 / 0.138 | 0.094 m |
| GPU1 V99 physics bridge, 90 Hz | 15.56 | 1.000 | 1.000 | 0.980 | 0.096 / 0.110 | 0.085 m |
| GPU1 V100 guided/conservative, 60 Hz | 14.97 | 0.989 | 0.969 | 0.913 | 0.129 / 0.145 | 0.092 m |
| GPU1 V100 guided/conservative, 90 Hz | 15.53 | 0.997 | 0.969 | 0.978 | 0.095 / 0.109 | 0.086 m |
| GPU1 V100 guided/explore, 60 Hz | 14.72 | 0.978 | 0.906 | 0.940 | 0.132 / 0.148 | 0.095 m |
| GPU1 V100 guided/explore, 90 Hz | 15.53 | 1.000 | 1.000 | 0.967 | 0.092 / 0.104 | 0.084 m |

V100's realized-contact feedback slightly improved 60 Hz survival, but it
regressed the two primary deployment-quality measures versus V99: view
occupancy and lateral hit speed.  More exploration also reduced 60 Hz survival
and did not improve lateral quality.  V100 is therefore rejected for formal
training; a favorable 90 Hz rollout cannot compensate for a worse 60 Hz
training-domain policy.

V99 is selected for GPU1 because its paired rate comparison satisfies the
requested direction without a trade-off: 90 Hz increases hits by 0.66 and full
rate by 6.25 percentage points while also improving view by 0.026 and RMS
hit-vxy by 0.028 m/s.  GPU0 selects V14 because it preserves perfect fixed-heavy
survival/view and improves next-contact placement without changing its proven
reward objective.

The final training launchers are `launch_formal_gpu0_v14_online.sh` and
`launch_formal_gpu1_v99_online.sh`.  They resume the 24-update bridge histories,
retain the selected optimizer moments, use convergence-controlled bridge and
target stages with no update cap, require exact final length/view validation,
and use online W&B.  GPU1 retains 60 Hz training plus a blocking paired 60/90
Hz promotion check on the measured fixed-heavy physics.

### Formal restart and monitoring contract

The selected runs were started in `pp_gpu0` and `pp_gpu1` with 1024
environments and online W&B:

| GPU | profile/stage | W&B | local run |
|---|---|---|---|
| GPU0 | V14, heavy-ball bridge 23/24 | `https://wandb.ai/fushi37/pingpong-mjx/runs/yjqfl80b` | `formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/` |
| GPU1 | V99, heavy-ball bridge 4/5 | `https://wandb.ai/fushi37/pingpong-mjx/runs/nzn715xa` | `formal_gpu1_v99_heavy_ball_conservative_60hz_n1024_s128_e4_online/` |

Both restored `stage_update=24` and the 24-row bridge history without resetting
the optimizer.  The first 13 new updates completed at about 21.2k/21.8k SPS.
PPO remained conservative (`KL=0.00186/0.00124`, clip fraction about 0.051,
no KL guard), and the current full-episode rollouts returned to approximately
16.2/13.5 hits after the expected fresh-runner startup transient.

Monitoring uses completed 32-update windows after the new runner has supplied
at least one full horizon.  A single noisy update does not trigger tuning.  The
bridge alert conditions are:

- ignored/confirmed hit ratio above 1%, or a non-negligible fast-hit reward;
- GPU0 view below 0.995, RMS hit-vxy above 0.15 m/s, or full rate below 0.95;
- GPU1 view below 0.90, RMS hit-vxy above 0.21 m/s, or full rate below 0.75;
- PPO exact KL at/above the configured target, rollback/guard activation, or
  a sustained anchor-KL jump;
- four-window non-improvement or regression in hits together with survival or
  lateral/view quality, rather than hits alone.

Promotion remains stricter than the monitoring floor: final deterministic
length and view must both equal 1.0, and GPU1's paired 90 Hz result must improve
hits without any view or RMS-vxy regression over 60 Hz.

### Hit-credit semantics and bridge-transfer correction

The concern that the old hit statistic could steer optimization in the wrong
direction is valid.  The environment first creates a physical
`launched_upward_raw` event only after contact separation, upward velocity,
relative-height, and predicted-apex confirmation.  It then applies
`hit_min_count_interval` to obtain `counted_hit`/`new_hit`.  Only `new_hit`
receives the hit bonus and all hit-conditioned height, view, lateral-motion,
angular-motion, and cadence terms.  Therefore a legitimate event discarded by
the debounce is not merely a logging omission; it loses the associated positive
and quality-learning signal.

The superseded V12/V97 configuration was especially problematic because its
0.32 s debounce rejected many confirmed launches while the fast-event penalty
was zero.  V13/V98/V99 now use a 0.22 s anti-chatter debounce and a quadratic
`fast_hit_penalty_weight >= 0.20`.  Thus a residual sub-0.22 s event is both
excluded from hit credit and explicitly penalized, while normal 0.28--0.40 s
cycles are counted.  Removing the debounce entirely remains unsafe because the
recorded telemetry contains approximately 0.06 s repeated-contact artifacts.
`hit_dt3` must not be interpreted as a direct inter-hit timestamp; it is an
episode-duration/count density proxy.

There is no hidden reward penalty over the ordinary low-bounce interval.  In
both active final configurations, `hit_min_interval_penalty_weight=0`; the
declared 0.40 s value is therefore inactive.  GPU1 also has zero cadence-reward
weight.  GPU0 has only a small positive cadence bonus (weight 0.22, centered at
0.50 s), not a penalty.  The 0.32--0.50 s curriculum interval is a promotion
quality gate rather than a per-step loss.  Finally, `hit_reward_cap_mode=off`
on both GPUs, so every counted hit receives base and quality credit; only the
additional count-progress combo stops growing after hit 14.  Hits 15+ are not
ignored by either the metric or the base reward.

At the latest formal samples, GPU0 had `confirmed_hit=0.013756/step` and
`ignored_fast_hit=0.0000305/step` (0.22%), with a mean fast-hit reward of only
-4.7e-9.  GPU1's latest full bridge rollout had zero ignored events.  The
current hit reward direction is therefore aligned with physical juggling; the
remaining hits fluctuation is dominated by reset/episode composition and task
robustness, not silent debounce losses.

GPU1's first bridge-to-target probe passed hits, length, view, lateral-motion,
and worst reset-bucket gates but was blocked solely by mean return
`-5.71 < -2.00`.  The scalar return is not comparable across the ball-physics
change, so a new stage-local admission-only threshold was added: the V99 target
accepts bridge transfer at `-8.0`, but its own rolling convergence and final
self-validation still use the global strict `-2.0` floor.  Three focused tests
verify both the relaxed transfer and unchanged final contract.

GPU1 was restarted from the causally matched update-100 checkpoint and its
76 saved bridge-history rows in `pp_gpu1`; optimizer state was retained.  The
active online W&B run is `https://wandb.ai/fushi37/pingpong-mjx/runs/h0tmsssu`
and the local directory is
`formal_gpu1_v99_heavy_ball_transfergatefix_v2_60hz_n1024_s128_e4_online/`.
The early low-hit rows after process restart are fresh-runner episode warm-up
and will be excluded from conclusions until a complete 32-update window has
replaced them.  GPU0 is already in target stage 24/24 and remains near 15.9
rolling hits, 0.995 length, 0.997 view, and 0.089 m/s RMS hit-vxy.

After the restart transient cleared, GPU1 held the complete bridge gate for 16
updates and advanced to target stage 5/5 at bridge update 149.  The admission
probe recorded `min_mean_return=-8.0`, `passed=1`, 14.48 hits, 0.952 length,
0.911 weighted view, 0.154 m/s RMS hit-vxy, and a passing worst reset-bucket
gate.  This particular draw returned 15.17, so it did not need the relaxed
floor; the important result is that the code and telemetry now distinguish the
one-time transfer floor from the unchanged strict final self-probe.  Both GPUs
are now training on their measured-heavy-ball target distributions.

Future runs also log `hit_credit/confirmed_events`,
`hit_credit/ignored_fast_events`, `hit_credit/ignored_fast_fraction`, and
`hit_credit/counted_fraction` from the unchanged event stream, so a debounce
regression is visible directly rather than reconstructed from step means.  The
current formal processes are intentionally not restarted for a logging-only
change.

### Target-stage drift replay and retention repair

GPU1 advanced from its corrected bridge at bridge update 149, but target-stage
performance then regressed even though single-update PPO KL remained near
0.001.  In successive complete target windows, actor KL to the target-entry
policy increased from about 0.013 to 0.024 and finally 0.033; view, horizon,
full rate, and hits declined together while value loss and gradient norm rose.
This is cumulative small-update drift, not one unstable PPO step.  The old
`actor_anchor_kl_coef=0.02` contributed only approximately 0.0003--0.0007 to
the objective and was too weak to prevent it.

Five checkpoints were replayed with the same deterministic seed, 64 episodes,
60 Hz ball observation, 3.7 g ball, and `solref` damping 0.90:

| GPU1 checkpoint | hits | length | full | weighted view | RMS hit-vxy | return |
|---|---:|---:|---:|---:|---:|---:|
| bridge exit | 14.516 | 0.948 | 0.891 | 0.936 | 0.152 | 17.06 |
| target update 25 | **15.078** | **0.971** | **0.953** | **0.940** | 0.137 | **26.45** |
| stored `best` / update 42 | 14.719 | 0.968 | 0.938 | 0.940 | 0.147 | 20.90 |
| target update 50 | 14.672 | 0.957 | 0.891 | 0.921 | **0.134** | 23.02 |
| target update 75 | 14.797 | 0.967 | 0.922 | 0.936 | 0.147 | 20.53 |

Update 25 is the causal peak.  The stored update-42 `best` was not selected
over update 25: the best-checkpoint writer requires a full 32-valid-update
window, and update 42 was the first eligible target checkpoint, while update
25 was never considered.  Fixed replay is therefore required when selecting a
restart point.

Update 50 also exposes the reward trade-off: it slightly lowers RMS hit-vxy
relative to update 25, but loses 6.2 percentage points of full rate and 1.9
points of view.  V99 has impact-vxy shaping but no dense
`ball_view_vxy_excess` loss, while its positive impact-angular reward explicitly
targets 1.2 rad/s.  A controlled V101 ablation changes only the final heavy-ball
target: view-center weight 0.60 -> 0.75, dense view-vxy soft limit/weight
1.00/0.00 -> 0.30/0.18, and angular positive target/weight 1.20/0.35 ->
0.95/0.20.  Physics, observation, actuator, timing, reset, and gates remain V99.
Three directed tests cover the new profile and unchanged hit-credit/heavy-ball
contracts.

Two 1024-environment, 24-update candidates now restart update 25 with reset
Adam moments, learning rate 7.5e-6, two PPO epochs, clip 0.08, target KL 0.003,
and actor-anchor coefficient 0.08.  The first preserves the exact V99 reward;
the second differs only by the V101 view/angular ablation.  The bounded update
count applies only to this comparison; the selected formal continuation will
again have no per-stage update cap.

GPU0 did not require a restart.  Its first strict final self-probe missed exact
promotion by one of 128 episodes (`length=0.99697`, `view=0.99962`).  Ten
updates later, the second probe passed all 128 episodes with `length=1.0`,
`view=1.0`, `full=1.0`, 16.38 hits, and 0.0745 m/s RMS hit-vxy.  The online W&B
run synced and ended normally.  A separate fixed-heavy 64-episode confirmation
also passed all 64 episodes with full length/view, 16.80 hits, no termination
reason, and 0.061 m/s mean hit-vxy.

### Physical-contact audit, V101 rate gate, and effective-hit A/B

The target-retention candidate disproved `hit_min_count_interval=0.22 s` as
the source of the current hit fluctuations.  Across its 24 updates it observed
approximately 38.7k height-confirmed launches and rejected only 18 through the
count debounce: an ignored fraction of 0.047%.  The active
`hit_min_interval_penalty_weight` and cadence reward are both zero; the small
V98/V99 fast-contact penalty was numerically negligible.  Moreover, a 0.161 m
ballistic launch has a same-height flight time of about 0.36 s, so a 0.22 s
gate is below the physically plausible period of every launch that passes the
current quality threshold.

The missing credit was one layer earlier.  A 64-episode fixed replay now logs
re-armed physical contact edges, upward clearance crossings, low survival
launches, quality-confirmed launches, sub-floor launches, and failed
resolutions separately.  Of 1023 upward clearance crossings:

| launch class | criterion | events | fraction |
|---|---|---:|---:|
| quality hit | predicted apex >= 0.161 m above anchor | 965 | 94.33% |
| low survival launch | 0.115--0.161 m | 56 | 5.47% |
| sub-floor launch | below 0.115 m | 2 | 0.20% |

Thus the user's visual observation was correct, but the responsible boundary
was the hard 70%-of-target apex confirmation, not the debounce.  The active
`failed_hit_penalty_weight` is zero, so these low launches are not explicitly
punished; nevertheless they lose all base, combo, view, pose, and motion hit
credit.  New telemetry exposes both `quality_hits` and `effective_hits`, where
the latter adds only the 50--70% target-height survival bucket.  Quality hit
count and all promotion gates remain unchanged.

The 24-update retention candidates produced the following fixed-seed paired
result:

| candidate / rate | hits | length | full | weighted view | RMS hit-vxy |
|---|---:|---:|---:|---:|---:|
| V99 retention, 60 Hz | 16.031 | 1.0000 | 1.0000 | 0.9343 | 0.1359 |
| V99 retention, 90 Hz | 15.469 | 0.9891 | 0.9688 | 0.9720 | 0.1237 |
| V101 view/angular, 60 Hz | 15.656 | 0.9957 | 0.9688 | 0.9202 | 0.1352 |
| V101 view/angular, 90 Hz | 15.813 | 1.0000 | 1.0000 | 0.9570 | 0.1143 |

V99 retention fails the paired promotion contract because 90 Hz reduces hits,
length, and full rate.  V101 passes every paired check: 90 Hz improves hits,
length, full rate, view, and RMS hit-vxy.  It is therefore the current formal
candidate despite V99's stronger isolated 60 Hz draw.

Before formal launch, V102 performed one bounded reward A/B from the same
update-25 checkpoint.  It changed only the V101 final target's
`low_survival_hit_reward_weight` from 0 to 0.25.  This was deliberately far
below the normal quality-hit base reward of at least 1.0, carried no combo or
quality terms, and did not redefine `hit_count`; its purpose was to remove the
zero-credit discontinuity without making faster low bouncing an attractive
hit-count exploit.

The 24-update V102 screen did not earn promotion.  At 60 Hz it produced 15.469
quality hits, 16.156 effective hits, 0.9803 normalized length, 0.9688 full
rate, 0.9330 weighted view, and 0.1334 m/s RMS hit-vxy.  At 90 Hz, quality hits
rose to 15.875, weighted view to 0.9639, and RMS hit-vxy improved to 0.1152
m/s; however, full rate fell from 0.9688 to 0.9375, so the strict paired gate
failed `full_rate_non_regression`.  This bounded run is only a direction
screen, not evidence that V102 cannot improve with longer training.  V101 is
selected for the formal continuation because it is the only tested branch
that currently passes every 60/90 Hz paired requirement.

The formal V101 continuation is convergence-driven and has no update limit:
the launcher explicitly passes `--max-stage-updates -1`.  Its exact
length/view/full-rate validation gates therefore keep training after a failed
probe instead of stopping because an arbitrary update budget expired.

The selected continuation was launched in tmux session `pp_gpu1` from the
V101 24-update candidate checkpoint with 1024 environments, 128 rollout steps,
two PPO epochs, retained Adam state, and online W&B logging.  Curriculum
continuity restored at target `stage_update=24`; the run is
`fushi37/pingpong-mjx/21cnr90y`.  GPU0 remains stopped after its successful
strict final confirmation, so it is not consuming a second training slot.

GPU0 ended naturally at the hardest stage 24/24, not because of an update cap:
the final heavy 3.7 g/lower-elasticity target converged at stage update 373 and
the run then exited normally.  Its final deterministic 60 Hz video uses the
completed `mjx_curriculum_last.pkl` with fixed 3.7 g mass and 0.90 contact
damping.  The full 1200-step episode completed with 19 hits, full rate 1.0,
camera/view fractions 1.0, and 0.057 m/s mean hit-vxy.  Artifacts are under
`formal_gpu0_v14_heavy_ball_conservative_n1024_s128_e4_online/final_video/`,
including `gpu0_v14_final_heavy_3p7g_60hz.mp4`, action/observation traces, and
the validation CSV.  No GPU0 continuation was launched; the device is reserved
for GPU1 checkpoint replay and diagnostics.

GPU0 was then used for fixed-seed paired replay of the live GPU1 continuation.
At GPU1 update 60, 90 Hz improved hits, mean length, view, and RMS hit-vxy but
lost one additional full episode (58/64 versus 59/64), so the paired gate
failed.  At update 200 the same 64-episode comparison passed every check:

| update 200 metric | 60 Hz | 90 Hz |
|---|---:|---:|
| mean hits | 14.516 | 15.250 |
| mean length | 1156.9 | 1178.9 |
| full rate | 0.9219 | 0.9688 |
| weighted view | 0.9595 | 0.9732 |
| RMS hit-vxy (m/s) | 0.1390 | 0.1187 |

This confirms that the required 90 Hz superiority can emerge during the
uncapped continuation even while randomized on-policy window metrics remain
noisy.  GPU1 training therefore remains active; GPU0 is free again for later
fixed probes.

### V101 long-run plateau and exploration/DR interaction

The uncapped V101 continuation was allowed to run past update 400 before any
stop decision.  From update 97 through update 416, successive 32-update
windows stayed in a narrow plateau: 12.72--12.84 hits, 1041--1049 mean steps
(0.868--0.874 normalized length), 0.890--0.893 weighted view, 0.201--0.204
m/s RMS hit-vxy, and 0.685--0.695 full rate.  PPO remained numerically stable
(`approx_kl` about 0.0006, actor-anchor KL about 0.005--0.007), so the failure
was not an optimizer explosion.  No formal `mjx_curriculum_best.pkl` was
written because none of these windows exceeded the resumed V101 candidate's
stage score.

Fixed update-400 replay also showed no late recovery.  At the measured-center
3.7 g/0.90 contact point, 60 Hz gave 14.609 hits and 0.9219 full rate; 90 Hz
gave 14.672 hits and the same full rate but regressed mean length by 7.4 steps,
so the paired gate failed.  Under the full target DR, deterministic full rate
fell from 0.8906 at update 200 to 0.8281 at update 400, while stochastic full
rate stayed exactly 0.7188.  The run was therefore stopped safely after update
416, with `mjx_curriculum_interrupted.pkl` saved and online W&B run
`21cnr90y` fully synced.

The deterministic/stochastic decomposition identifies the missing factor.
For update 200 at the fixed center physics, stochastic action sampling reduced
full rate only from 0.9219 to 0.8906.  Under the complete mass, elasticity,
contact, actuator, observation-latency, and reset distribution, the same
sampling reduced full rate from 0.8906 to 0.7188 and changed failures toward
`racket_too_high`.  The learned seven-axis `log_std` stayed almost constant
from updates 50--400: six axes near -3.0 and `RightArm-4` near -2.64.  Thus
exploration noise is not independently catastrophic, but its interaction with
hard-tail execution/ball DR reproduces the training plateau.

A first bounded V101 direction screen changed only exploration
(`max_log_std=-3.0`, `min_log_std=-3.4`, entropy coefficient zero).  It briefly
improved on-policy length/view/full rate, then returned to the same plateau by
update 64; fixed 60 Hz full rate was 0.8906, so it was not promoted.  Two
stronger single-family screens now use `max_log_std=-3.4`,
`min_log_std=-4.0`, and zero entropy from the original V101 candidate.  They
differ only in actor-anchor coefficient, 0.08 versus 0.16.  These bounded runs
select a direction; any selected formal continuation will again use
`--max-stage-updates -1`.

### Strong-anneal checkpoint selection and formal restart

The two 64-update strong-anneal screens confirmed the exploration/DR diagnosis.
Their final checkpoints both raised full-target-DR stochastic full rate from
the old update-200/400 value of 0.7188 to 0.8438.  The anchor coefficient was
not the main determinant of that recovery: 0.08 and 0.16 produced the same
full rate.  However, both branches peaked before update 64, so selecting only
`mjx_curriculum_last.pkl` would still preserve avoidable late drift.

The fixed-seed 64-episode validation of each saved `best.pkl` separated the
branches:

| best checkpoint | 60 Hz hits / length / full | 90 Hz hits / length / full | 60/90 gate | full-DR stochastic hits / length / full |
|---|---:|---:|---:|---:|
| anchor 0.08 | 14.875 / 1154.0 / 0.9063 | 15.219 / 1164.3 / 0.9375 | pass | 14.69 / 1145.6 / 0.8906 |
| anchor 0.16 | 14.859 / 1177.5 / 0.9531 | 14.625 / 1148.2 / 0.8906 | fail | 13.89 / 1095.6 / 0.8281 |

For anchor 0.08, 90 Hz also improved weighted view from 0.9355 to 0.9566 and
RMS hit-vxy from 0.1446 to 0.1146 m/s.  It passed all six paired checks.  The
0.16 branch had the stronger isolated 60 Hz draw, but its 90 Hz result
regressed hits, length, and full rate; it was therefore rejected.  This is why
the formal source is the anchor-0.08 best checkpoint rather than either final
checkpoint or the best-looking single-rate rollout.

The formal continuation is
`formal_gpu1_v101_stronganneal_anchor08_lowlr_60hz_n1024_s128_e2_online/`.
It restored target-stage continuity at update 42 and retained Adam state.  The
only additional optimizer change is a conservative learning-rate reduction
from 7.5e-6 to 5e-6, motivated by the repeatable update-45-to-64 drift.  It
uses 1024 environments, 128 rollout steps, two PPO epochs, log-std bounds
-4.0/-3.4, zero entropy bonus, anchor coefficient 0.08, online W&B run
`fushi37/pingpong-mjx/85j8kgqg`, and explicitly sets
`--max-stage-updates -1`.  The final blocking validation still requires exact
normalized length 1.0 and view occupancy 1.0; promotion then runs the external
fixed-heavy paired 60/90 Hz gate and a full-target-DR stochastic replay.

GPU0 remains naturally complete at hardest stage 24/24 and is not restarted.
Its free device was used for the paired checkpoint experiments above and stays
available for future fixed replay of the live GPU1 continuation.

### Low-learning-rate drift and failure-tail diagnosis

The low-learning-rate formal continuation did not solve the plateau.  It was
allowed to run through update 119 and was then stopped with SIGINT; both
`mjx_curriculum_interrupted.pkl` and `mjx_curriculum_last.pkl` were saved and
W&B run `fushi37/pingpong-mjx/85j8kgqg` synced cleanly.  PPO itself remained
stable (last-32 `approx_kl=0.00064`, clip fraction 0.032, actor-anchor KL
0.0045), but the last-32 task window drifted to 13.29 hits, 1072 mean steps,
0.738 full rate, 0.895 view occupancy, and 0.190 m/s RMS hit-vxy.

A frozen update-70 checkpoint proved that the fixed-rate contract had not
failed: deterministic fixed-heavy 60 Hz gave 14.64 hits, 1144.9 steps, 0.906
full rate, 0.933 view, and 0.136 m/s RMS hit-vxy; 90 Hz improved these to
15.11 hits, 1172.9 steps, 0.953 full rate, 0.959 view, and 0.120 m/s.  All six
paired checks passed.  In contrast, full-target-DR stochastic full rate was
only 0.7969, below the source best checkpoint's 0.8906.  The live decline is
therefore erosion of hard-tail DR robustness rather than unstable PPO updates
or a 60/90 Hz observation-rate failure.

Resetting Adam moments did not change this behavior.  Two otherwise identical
64-update screens from the source best checkpoint produced the following
external results:

| reset-Adam branch | fixed 60 Hz hits / full | fixed 90 Hz hits / full | full-DR stochastic hits / full |
|---|---:|---:|---:|
| anchor 0.08 | 14.97 / 0.9219 | 15.78 / 1.0000 | 13.95 / 0.8125 |
| anchor 0.16 | 15.05 / 0.8906 | 15.86 / 0.9844 | 14.03 / 0.8125 |

Both branches passed the paired fixed-rate comparison, but both lost the same
amount of full-DR survival.  Thus stale optimizer moments and insufficient
anchor strength are ruled out as primary causes.  The remaining mismatch is
between the strict full/CVaR promotion gate and PPO's mean transition
objective: rare failure-causing transitions receive too little weight.

The next bounded screen therefore uses the existing tested failure-focus PPO
path.  It only amplifies negative-advantage transitions preceding a true task
termination; time-limit truncations, unfinished rollout suffixes, and positive
advantages are excluded.  The first branch targets failures below 12 hits; a
paired branch targets every true termination below 20 hits, limits credit to
the final 128 control steps (0.64 s), and retains the selected 0.08 anchor.
These are direction-selection screens with a 64-update cap.  A promoted formal
run will again have no update cap.

The completed A/B did not justify replacing the source checkpoint.  Resetting
Adam with anchor 0.08 or 0.16 produced only 0.8125 full-target-DR stochastic
full rate.  Failure focus below 12 hits produced 0.7969.  Expanding focus to
all true terminations improved the last checkpoint to 0.8281 at weight 1.5;
weight 4.0 fell back to 0.7969.  The internally scored best checkpoints were
also checked externally and did not exceed the original strong-anneal source's
0.8906 full-target-DR rate.  Therefore the internal curriculum score is not
used as a proxy for hard-tail robustness.

A final positive-guidance V102 screen assigned only the 50--70% target-height
survival bucket a 0.25 reward while leaving quality hit count, combo, and all
gates unchanged.  Its best checkpoint passed the paired fixed-rate contract:
60 Hz gave 15.25 hits and 0.9531 full rate, while 90 Hz gave 16.06 hits and
0.9844 full rate with view occupancy 1.0.  However, full-target-DR stochastic
full rate was only 0.7969, so V102 was rejected for the formal run.

The final formal launcher is
`launch_formal_gpu1_v101_failurefocus_anchor08_online.sh`.  It restarts from
the original externally selected strong-anneal anchor-0.08 best checkpoint,
resets Adam, trains at 60 Hz with 1024 environments, 128 rollout steps, two
epochs, learning rate 5e-6, log-std bounds -4.0/-3.4, and no entropy bonus.
It keeps the conservative 1.5x failure-tail focus because that was the only
screen to improve the final full-DR checkpoint over the otherwise identical
reset-Adam branch; only negative-advantage steps before true termination are
affected.  Formal training has no update cap, logs to W&B online, and retains
the blocking len=1/view=1 final validation plus the external 90>60 Hz gate.
