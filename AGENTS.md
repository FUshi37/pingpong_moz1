# Repository Guidelines

## Scope and Sources of Truth

This repository contains a ROS 2 controller, MJX/MuJoCo reinforcement-learning tools, deployment adapters, and experiment evidence for robot table-tennis juggling. Read this file before changing any file below the repository root.

Use sources in this order:

1. `AGENTS.md` for mandatory contribution and safety rules.
2. `TRAINING_CURRICULA.md` for released models, course identities, and current/future training contracts.
3. The selected profile in `pingpong_controller/tools/rl_sim/train_juggle_mjx_curriculum.py` and its launch script for executable settings.
4. Dated reports under `pingpong_controller/outputs/` for evidence and historical conclusions.
5. `CLAUDE.md` and `CODEX.md` as dated handoff notes; verify them against code and the active script before use.

Never infer a run's GPU, XML, checkpoint, stage, or action semantics from a filename alone.

For future stable-juggling Sim2Real architecture, implementation sequencing,
interface contracts, and copyable Codex prompts, read
`pingpong_controller/tools/rl_2real/STABLE_JUGGLING_SIM2REAL_ARCHITECTURE_AND_CODEX_PROMPT.md`.
It is a design and phased implementation contract; it does not override a
released profile, authorize training or hardware commands, or make the
separate real-robot repository writable without explicit task scope.

## Project Structure and Module Organization

- `pingpong_controller/`: ROS 2 Python package. Runtime nodes, policy wrappers, calibration helpers, and safety limiting live here.
- `pingpong_controller/tools/rl_sim/`: MJX environments, PPO/curriculum training, validators, replay analysis, XML/meshes, and launch scripts.
- `pingpong_controller/tools/rl_2real/`: checkpoint deployment adapters and model-specific deployment contracts.
- `pingpong_controller/tools/vision_calib/` and `vision_refine/`: camera calibration and vision utilities.
- `test/`: `pytest` unit, regression, curriculum-contract, and ROS ament lint tests. Name tests `test_<behavior>.py` and functions `test_<contract>()`.
- `pingpong_controller/data/`: source datasets and calibration inputs. `pingpong_controller/outputs/`: generated runs, checkpoints, reports, plots, and replay evidence; do not treat every output as versioned source.
- `ReactiveMotionPlanner/rmp-recovery/`: workspace-local recovered RMP implementation, calibration, replay tools, and its own tests. Its canonical configuration is `recovered_rmp/configs/rmp_equiv_refine_baseline.json`. This tree is not tracked by the main repository at the 2026-08-20 reconciliation point; version it deliberately before making a released profile depend on it.
- `build/`, `install/`, `log/`, `wandb/`, caches, and `__pycache__/` are generated artifacts.
- Real-robot behavior may be cross-checked in `/home/yangzhe/Project/pingpong_playreal/pingpong-play`; it is a separate repository and is read-only unless the task explicitly includes it.

## Architecture and Preserved Baselines

The two primary released policies are:

| Name | Release/source commits | Action | Released course |
| --- | --- | --- | --- |
| GPU0-QVEL | `e012d07400a99e95b50c2dcf7200d6ede312dfa6` with companion parent `d269517c4e79cc53b9024dc6bdf9eff496a653d3` | joint velocity | `goal_d455_sport_taskspace_qvel_vertical_v14` |
| GPU1-QACC | model `4f47e6d3c67917acd6a9b05162a488c91b319597`, companion stack `69b9bf3442f5e9473bb291e0fa9439d0f15a53d5` | joint acceleration | `goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1` |

Both actors use 67-D observations and 7-D actions. They were trained with command-state integration and the fitted delayed second-order actuator plant. Do not silently reinterpret an existing checkpoint or fine-tune as measured-state feedback or RMP control. The exact checkpoints, stages, DR, PPO settings, and fine-tune courses are registered in `TRAINING_CURRICULA.md`; update that registry in the same change as every new or modified profile.

## Build, Test, and Development Commands

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-rl-sim.txt  # MJX/JAX training environment
colcon build --packages-select pingpong_controller --symlink-install
source install/setup.bash
ros2 run pingpong_controller pingpong_node
```

Use the established `pingpong` conda environment when reproducing existing MJX results. Run the smallest relevant checks first, then the complete applicable suite:

```bash
python -m pytest test/test_delay_conditioned_control.py -q
python -m pytest test -q
colcon test --packages-select pingpong_controller
colcon test-result --verbose
bash -n pingpong_controller/tools/rl_sim/<launch_script>.sh
python -m py_compile pingpong_controller/tools/rl_sim/<changed_file>.py
python -m unittest discover -s ReactiveMotionPlanner/rmp-recovery/recovered_rmp/tests -v
```

Validators and trainers are launched from `pingpong_controller/tools/rl_sim/`. Use `--help` and a checked-in, reviewed launcher rather than reconstructing a long command from memory. Confirm JAX reports a CUDA device for GPU evidence; a CPU fallback does not count as a GPU experiment.

## Coding Style and Naming Conventions

- Python uses four spaces, PEP 8 layout, concise docstrings, type hints for new interfaces, `snake_case` functions/variables, and `UPPER_SNAKE_CASE` constants. No Black/Ruff configuration is authoritative; the repository uses `ament_flake8` and `ament_pep257` tests.
- Keep physical quantities in SI units internally and include units in names such as `_rad_s`, `_kg`, `_ms`, and `_hz`. State array shapes and the seven-joint ordering at interfaces.
- Prefer immutable configuration updates (`dataclasses.replace`) and pure/JAX-compatible functions in vectorized environment paths. Avoid Python-side mutation or data-dependent control flow inside `jax.jit` code.
- Shell launchers begin with `#!/usr/bin/env bash` and `set -euo pipefail`; quote paths and expansions, use descriptive run-specific variables, and refuse accidental output-directory overwrite.
- Version profiles monotonically (`..._v123`), preserve prior profile builders, and give stages stable descriptive names. Never reuse an existing profile name for changed physics, reward, observation, action, DR, or gate semantics.

## Change, Compatibility, and Experimental Gates

All code changes must have no known defects and must be experimentally shown effective before they replace or enter a normal path. Development may occur behind an explicit opt-in flag/profile, but promotion requires recorded evidence.

- Preserve every previously working path. New modes default off; old checkpoints, profile names, observation layouts, action meanings, launch commands, and deployment behavior must continue to work. Broken historical experiments need not be preserved, but a path may be called broken only with reproducible evidence.
- Add regression tests before changing control, integration, delay, observation, checkpoint migration, reward/gate, or reset semantics. Compare old-mode outputs before and after the change with fixed seeds and tolerances.
- Validate in proportion to risk: unit tests and syntax checks; deterministic CPU smoke; frozen-checkpoint paired replay; isolated GPU probe; then multi-seed training evidence. Hardware-facing changes additionally require recorded replay and staged, supervised real-robot checks with safety limiting.
- Define acceptance metrics before an experiment. At minimum record code/XML status, parent checkpoint and SHA-256, profile/stage, full command, seed, GPU UUID, W&B/run directory, hits/episode length, hit1/hit3/hit12, truncation, view/apex, hit and racket velocity, KL, explained variance, and failure distribution.
- A higher return alone is not proof. A candidate must improve its preregistered target without regressing existing success, safety, view, contact quality, or checkpoint-load contracts. Keep failed and negative results in the experiment report.
- Do not overwrite runs, checkpoints, plots, or W&B histories. Use a new output directory and run identity when source, optimizer, profile, physics, DR, observation, or action semantics change.

## Sim-to-Real and RMP Direction

- Existing QVEL/QACC checkpoints and current fine-tunes retain the `q_cmd/dq_cmd` integration path. For future from-scratch policies, add an explicit measured-feedback path using simulated/real `q/dq`; retain and regression-test the command-state path as a selectable legacy mode.
- Future from-scratch training uses recovered RMP plus low-level PD instead of the fitted actuator model. Do not stack RMP with actuator filters, FOPDT, inverse compensation, a second planner, plant delay, or state projection. Preserve the actuator path for released checkpoint replay and deployment compatibility.
- Before creating an RMP training profile, experimentally identify RMP/PD DR ranges against a pre-registered effective-juggling population. Define validity without consulting the candidate DR result; hash the contract and source evidence; report every data failure, excluded recording, excluded sample reason, and accepted count. Do not require DR to cover corrupt data, safety interventions, failed/post-drop motion, unsupported stale ball states, or jointly extreme replay-divergence outliers. Conversely, do not discard a valid hard case because coverage fails. The accepted envelope must cover joint and racket position, orientation, velocity, and acceleration tails pointwise and produce per-component worst-case coverage plots equivalent to the historical `.../worst_component_curves` artifact family. Among stable envelopes with complete accepted-population coverage, minimize every DR interval; wider is not safer by default. RMP output-layer agreement must not be presented as encoder-feedback agreement.
- Measure and record the new ball mass first, then update mass, inertia, contact/restitution, friction, and spin DR with outcome-matched validation. Do not copy the legacy 3.7 g range without evidence.
- Build the new RMP curriculum as independent runs: QVEL first, then QACC. Do not resume actuator-model checkpoints across this control-stack boundary.

## Pending Simulation Implementation Priorities: Base, Ball, Closed Loop, Contact

Implement the following work strictly in priority order `1 > 2 > 3 > 4`.
This is a pending implementation contract, not authorization to change code,
start a new training run, or modify the real-robot repository. Each behavioral
change requires regression tests first, a new opt-in monotonically versioned
profile/launcher, and a matching `TRAINING_CURRICULA.md` entry. Preserve every
legacy XML, profile, checkpoint, reset distribution and deployment path.

1. **Remove the simulated robot-base settling/sinking gap.** Adopt both layers
   because the physical base will remain stationary for the foreseeable
   deployment: first align the simulator support geometry with the floor so
   reset does not begin with the support approximately 10 mm above contact;
   then provide a fixed-base simulation mode that locks all six base pose DOFs
   to the calibrated initial pose. The observed minimal support candidate moves
   `mjx_base_support` from `z=-0.045 m` to `z=-0.055 m`, which reduced a 700 ms
   nominal base-z settlement from about `-10.036 mm` to `-0.036 mm`; this is
   diagnostic evidence to verify, not permission to edit a shared XML in
   place. Do not hide the problem with an observation-only offset or per-step
   `qpos` overwrite. Validate initial and post-settle base pose, joint state,
   absolute racket FK/pose, ball pose, ball-minus-racket observation and reset
   geometry under static and policy-driven motion. The fixed-base candidate
   must remove base drift without changing the seven arm-joint semantics, and
   the selectable legacy floating-base mode must remain numerically unchanged.

2. **Make 4.0 g the nominal ball mass for the new simulation path.** Set the
   new profile's default mass to exactly `0.004 kg` and define a small DR range
   centered on `0.004 kg`; determine and preregister the numerical half-width
   from repeated scale measurements instead of inventing it or copying the
   legacy 3.7 g range. Update geometrically consistent inertia together with
   mass, log the sampled episode mass, and isolate this change from contact,
   reward, observation, reset and PPO changes. Existing profiles retain their
   current mass and DR contracts. Require fixed-seed legacy regression and a
   paired mass-only simulation check before promotion.

3. **Implement the stable-juggling Sim2Real architecture in ordered phases.**
   The complete source of truth is
   `pingpong_controller/tools/rl_2real/STABLE_JUGGLING_SIM2REAL_ARCHITECTURE_AND_CODEX_PROMPT.md`;
   follow its Phase 0 through Phase 4 prompts one phase at a time. Start with
   common-command replay and timestamp/coordinate/data contracts, then add a
   structured collision-outcome model and DR, deployment-available plant and
   impact history adaptation, exact versioned real-runtime parity, and finally
   a default-off bounded per-impact residual with staged supervised hardware
   gates. Preserve the single real-robot `q_ref -> safety -> robot RMP`
   boundary and never stack recovered RMP, XML PD, actuator filters, inverse
   compensation, a second planner or simulated delay on the robot. Evaluate
   finite-step post-impact error contraction and `P(N>=k)` together with the
   existing hit, survival, view, apex, motion and safety metrics; long open-loop
   trajectory equality and higher mean return are not acceptance evidence.
   Treat raw sensor depth and `surface->center depth` as separately logged
   observation modes and require a fixed-ball, fixed-racket paired A/B audit
   before changing the estimator contract or training observation bias.

4. **Recalibrate ball-racket collision parameters after new physical data are
   available.** Until the user supplies the planned real-robot experiments,
   preserve all current collision parameters and do not widen contact DR.
   Record ball identity/mass/diameter, pre/post 3-D velocity, racket pose and
   twist at the physical contact edge, impact offset, apex/lift/XY outcome and
   validity. Keep static support/launch separate from dynamic impact. Fit on a
   declared population, retain an independent held-out population, compare CPU
   MuJoCo and MJX, and accept changes only from outcome-level normal,
   tangential, 3-D velocity, apex and lateral calibration—not from a single
   restitution/friction number or training-set reward. Collision changes must
   be isolated in a new profile and must not begin until priorities 1--3 have
   their required evidence.

The current opt-in implementation is
`goal_d455_measured_qvel_rmp_vertical_v1`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v1.sh`.
It is an experimental, from-scratch 50-D-actor/7-D-QVEL course using measured
`q/dq`, five recovered-RMP ticks and five MJX ticks at 1 kHz, and the
`recovered_rmp_equiv_v1` PD profile. Keep its no-stack constructor checks and
legacy command-state regression tests. Its 50-D actor layout deliberately
replaces the legacy seven-value `q_cmd-q` block with the encoder-only
`dq(t)-dq(t-1)` block; it must not read `q_cmd`, `dq_cmd`, actuator delay, or
privileged DR values. A higher input dimension is not evidence of greater
policy capacity; any future history augmentation requires a new profile and
an observation ablation. Stages 10--19 retain the fine-tune 0.23 m apex target
with a 0.218--0.248 m graduation band and the GPU0-QVEL 0.22 s physical-hit
debounce. The launcher must continue to reject
resume checkpoints, unmeasured ball-mass inputs, a missing/invalid RMP output
replay, or a missing/invalid DR coverage manifest. The manifest validator must
verify hashed pointwise artifacts and recompute rather than trust declared
pass flags or global min/max summaries. It must cover record_new3/4/5 joint
and racket position/orientation/velocity/acceleration tails, replacement-ball
free-flight/contact/spin/apex outcomes, source hashes, and per-component curve
artifacts. Candidate ranges are not hardware evidence. Any change to the v1
RMP/PD, ball/contact/spin DR, behavior shaping, observation layout, or gates
requires a new monotonically versioned profile and a matching
`TRAINING_CURRICULA.md` update.

The current successor is
`goal_d455_measured_qvel_rmp_vertical_v2`, with manual GPU0 launcher
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v2.sh`.
Preserve v1: its `recovered_rmp_equiv_v1` PD adapter reflects the canonical
JSON's multiplied scale layers, while the RMP.md batch command actually
overwrites the global layer with per-joint scales. V2 alone uses
`recovered_rmp_rmpmd_v2`, whose effective Kp is
`[112320,184320,133650,73200,30420,62100,35100]` and Kv is
`[9048,9604.8,9570,4071.6,3712,5000,2030]`; training-local replay must remain
numerically paired with the RMP.md path. Formal v2 coverage uses a uniform
5 ms `wallTimeS` grid, published-target zero-order hold, and joint-header-time
state interpolation. `sourceTimeS` replay is diagnostic only because it
compresses real missed ticks.

Model 200 Hz missed publications before RMP with
`dr_rmp_target_update_hold_*`, never with the post-RMP
`dr_execution_command_hold_*` path; environment construction must reject
stacking both. V2 stages ramp target-hold probability `0 -> 0.025 -> 0.05 ->
0.08`, with a measured 2.2% tail over 2--9 ticks at the late 0.08 setting.
Its final bounded RMP/PD ranges remain candidates, not hardware evidence.
Every v2 manifest must additionally hash and recompute candidate stability,
use `recovered_rmp_rmpmd_v2`, and pass wall-clock pointwise coverage before
training. The 2026-08-21 worst-four-record, 300-sample GPU0 screen improved
substantially but still failed (joint q/dq/qdd worst coverage
90.1%/92.6%/84.9%; racket families 96.2--99.1%); all 105 candidates were
stable. Do not launch from that negative diagnostic. New-ball mass, diameter,
and at least 30 outcome trials are still absent, so the v2 launcher must remain
evidence-blocked.

The successor search population is frozen by
`rmp_effective_juggling_validity_v1.json`. It requires a >=300-row longest
contiguous policy segment, no safety intervention, physically usable and
fresh ball state backed by either raw observation or an explicit predictor,
and rejects a recording as a nominal-replay outlier only when both position
and angle RMS exceed fixed pre-DR 3-IQR outer fences. The 2026-08-21 audit
accepted 56/64 parseable recordings (15/19/22 from record_new3/4/5); one
additional directory was unparseable. The accepted records contain 27,165
wall-grid rows, of which 24,402 (89.83%) pass the effective-sample rules. This
contract is diagnostic evidence
for the next monotonically versioned course, not permission to reinterpret
V2 or weaken its 64-record validator. A two-effective-record screen showed
that filtering invalid tails improves racket coverage but leaves systematic
valid joint-response gaps. Neither extending RMP output delay to +80 ms nor
directly applying the YAML `[45,35,25,25,10,10,10] Nm` torque limits fixed the
gap; the latter greatly worsened it. Treat the remaining blocker as nominal
RMP/low-level closed-loop calibration/parameterization, not a reason to widen
all existing factors or start PPO.

The explicitly user-accepted experimental successor is
`goal_d455_measured_qvel_rmp_vertical_v3`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v3.sh`.
V3 freezes every V2 control, observation, reward, curriculum, RMP/PD, timing,
and DR value; it does not reinterpret the negative V2 coverage artifacts as a
formal hardware-coverage pass. It exists so a from-scratch GPU0 experiment can
proceed before detailed RMP tuning. The strict V1/V2 launchers and validators
remain unchanged. V3 requires the passing hashed RMP-output replay, an explicit
measured new-ball mass interval, `CONFIRM_GPU0_READY=YES`, and
`ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE=YES`. V3 and its explicitly versioned
experimental successors through V72 may bypass the incomplete RMP/ball-outcome
manifest only through the trainer's profile allowlist for
`--allow-experimental-rmp-dr-bypass`; never broaden that allowlist to an
unreviewed profile. Do not call these profiles formally hardware-covering,
resume an actuator checkpoint, or infer the legacy 3.7 g ball mass. No PPO run
is started by repository changes; the user launches it manually after
reviewing the preflight audit.

The opt-in bounded-reference successors V14 and V15 use a persistent policy
reference, not the stateless measured-q lead used by V13. Their core update is
exactly:

```text
q_ref_next = q_ref + v_policy * dt
error      = clip(q_ref_next - q_measured, -error_max, +error_max)
q_ref_next = q_measured + error
q_target   = q_ref_next
```

Here `dt=0.005 s` and `error_max[j]=velocity_limit[j]*t_i[j]`; the published
target is additionally intersected with the hard joint-position limits. Reset
`q_ref` from measured q and append the seven-value `q_ref-q_measured` error to
the actor observation, giving V14--V37 a 57-D actor while preserving the first
50 values exactly. The scalar and seven-value horizon configuration fields are
mutually exclusive and fail closed. V14 uses uniform `t_i=17.5 ms`; V15 alone
uses `[18,19,19,18,18,18,18] ms`, initialized from the recovered RMP's
per-joint output-delay vector. Preserve V14 and every 50-D predecessor.

Do not equate this stateful error horizon with
`recovered_rmp_qvel_target_lead_s`. Under sustained saturated QVEL they have
the same dimensional displacement bound `velocity_limit*t`, but the bounded
reference accumulates only `v_policy*dt` per tick and retains/unwinds intent;
the stateless lead reapplies `v_policy*t` from measured q every tick. Nor may
RMP output delay and encoder-feedback delay be summed without an experiment.
The 2026-08-21 three-mass GPU0 screen rejected the encoder-feedback-only vector
because it hit only 6/21 onset lanes and rejected the summed vector because
qacc exceeded its limit in 42.86% of samples. The RMP output-delay vector and
uniform 17.5 ms each hit 13/21 with zero screen exceedance. Same-seed 20-update
PPO gave V15/V14 mean hits 0.99489/0.99469 and qacc-exceed fractions
0.00719/0.00608, so V15 is a viable physical-delay-initialized experiment but
not proven superior. Calibrate future `t_i` per joint from the complete
published-target-to-measured-motion closed loop, then choose the smallest
vector that passes preregistered hit, qvel, qacc and workspace gates.

On the physical robot, the recovered RMP is a simulator-side surrogate for the
RMP already inside the robot control stack. Deploy the policy's 200 Hz
measured-feedback-integrated position target as the robot RMP input
(`MechUnitCmd.jnt_pos`); never deploy another recovered RMP, its simulated
output delay, or XML PD on top of the robot RMP. Conversely, do not send the
simulator RMP's final `command_q` as though it were the policy output. The
existing deployment contract covers the fail-closed 50-D V1--V13
measured-feedback formula. V14--V37 are 57-D simulation-only profiles until a
separately tested runtime reproduces their persistent `q_ref` state, reset and
observation suffix; do not silently deploy them through the 50-D adapter. The
exact model boundary and staged hardware gates are registered in
`pingpong_controller/tools/rl_2real/MEASURED_QVEL_RMP_V2_DEPLOYMENT_CONTRACT.md`.

## Pending Implementation and Evidence: V16 In-View Coverage Tail

The current opt-in continuation is
`goal_d455_measured_qvel_rmp_vertical_v16`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v16_resume_v15.sh`.
V16 preserves V15 stages 1--21 exactly. Existing stage 21,
`rmp15_launch19_final_consolidation`, is the required polish/consolidation
stage; do not insert, rename, or reinterpret an earlier stage. V16 is
continuation-only: it must restore a V15/V16 57-D bounded-reference checkpoint,
the matching `curriculum_progress.csv`, the explicit current stage, actor,
critic, Adam moments, and convergence history. It must reject from-scratch
launch, automatic stage guessing, optimizer/critic reset, an actuator/50-D
checkpoint, or a changed bounded-reference/RMP contract. Use a new output
directory. The user explicitly authorized the current launcher to resume W&B
run `9acnp70r`; because its history contains one update newer than the saved
checkpoint, use the trainer's W&B-only 131072-step offset. Do not alter the
true checkpoint/CSV step or generalize this authorization to another run.

After the immutable polish stage, append difficulty in this order only:

1. normal in-view ball-observation noise at the real 90 Hz observation rate;
2. in-view ball-state and episode-target XY support, first a bridge and then
   the widest stable candidate range;
3. measured 200 Hz missed target publications before RMP, first the 5%/1.1%
   bridge and then the 8%/2.2% late setting with a 2--9-tick tail;
4. a final unchanged-domain stability consolidation stage.

The implemented candidate tail uses 8 mm ball-position noise, 0.08 m/s
velocity noise, 7 mm episode-coherent position bias, 0.07/0.07/0.10 m/s
velocity bias, +/-1.2% scale, and episode target support up to +/-0.15 m X by
+/-0.10 m Y. These are candidate training bounds, not evidence that real
in-view observations or ball states are covered. Until a paired real-camera
audit establishes componentwise distributions, do not widen them or call the
profile hardware-covering. The audit must separate sensor noise from true
ball motion, use only fresh physically valid samples inside the calibrated
D455 view, report joint position/velocity and conditional flight/contact/apex
quantiles, hashes and exclusions, and compare pointwise rather than by global
min/max.

Out-of-view prediction, camera/view missing, observation dropout, burst loss,
and reacquisition are explicitly out of scope for this V16 tail. Appended
stages set those mechanisms to zero and terminate when the true ball leaves
the registered D455 bounds. Do not present that termination behavior as a
deployment solution for lost vision. Wide XY is acceptable only if hit/apex,
view, long-survival and safety gates pass while mean/RMS ball and racket
horizontal velocities and full racket angular speed remain bounded; the final
candidate gates are 0.16/0.23 m/s ball vxy, 0.065/0.105 m/s racket vxy and
0.80 rad/s racket angular speed. Remaining blockers include the measured
in-view noise/ball-state coverage audit, new-ball outcome trials, and formal
pointwise RMP/PD encoder-feedback coverage. V16 remains eligible only for the
explicit experimental DR bypass and is not a released or deployable policy.

## Current Experimental Motion-Quality Continuation: V17

The current opt-in continuation is
`goal_d455_measured_qvel_rmp_vertical_v17`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v17_resume_v16.sh`.
V16 remains immutable. V17 must resume the pinned V16 stage-12
`rmp15_launch10_workspace_wide` archive at update 6150 (SHA-256
`5ee9f671a379f09629939f25bf5f2df36051160d2e483af7098044b34450e82f`).
It restores actor, critic and Adam moments, but deliberately starts fresh V17
stage/convergence history because reward and stage identities change. It must
reject another source profile/stage, a non-57-D checkpoint, curriculum-history
resume, optimizer/critic reset, and from-scratch launch. The user explicitly
authorized V17 to resume W&B `9acnp70r`; use the launcher-pinned W&B-only
`54788096` step offset so the first new V17 point follows the V16 history,
without modifying the checkpoint's true optimizer/global step. Do not reuse
that authorization for another run ID.

The source policy is not a hit-acquisition failure: the representative
deterministic replay completed 1200 steps with 16 hits. V16 was stopped after
update 6567 because stage 12 never passed its behavior gate: recent mean hit
racket XY speed stayed about 0.107--0.114 m/s against 0.10 m/s, and full
racket angular speed stayed about 1.29 rad/s against 1.20 rad/s. Mean apex was
about 0.221 m, close to the 0.218 m lower edge. Treat this as a reward/gate
alignment problem, not evidence to change the bounded-reference or RMP path.

V17 is built from a one-to-one copy of the complete 27-stage V16 profile; do
not collapse it into a tail-only course. V17 stages 1--11 are exact, unchanged
copies of V16 stages 1--11; the continuation never replays them. V16 stage 12
remains V17 stage 12 and becomes the first of three
unchanged-domain motion-quality/apex lessons, two additional bridge/commit
stages are inserted after it, and every V16 stage 13--27 remains present in
order as V17 stages 15--29. The three repair stages' mean
racket-XY/full-angular/apex-lower gates are respectively
`0.12/1.35/0.220`, `0.11/1.27/0.223`, and `0.10/1.20/0.225` in SI units;
the upper apex gate remains 0.248 m. Racket-XY and angular event-penalty
weights rise `0.75 -> 1.00 -> 1.25`; bounded low-angular hit reward rises
`0.35 -> 0.55 -> 0.75`; and bounded 0.23 m target-apex reward rises by the
same sequence with 0.030 m sigma. The later V16-derived stages retain the
committed quality rewards and at least the 0.225--0.248 m apex band. Control,
observations, RMP/PD, physics, reset distributions and DR remain unchanged.

V17 stages 12--29 have no update cap, and the launcher supplies no per-stage
update limit; advancement depends only on the complete rolling and
block-validation gates, and unconverged forced advancement remains
prohibited. The launcher uses LR `1.5e-4`, clip
`0.15`, target KL `0.008`, creates a new output directory while continuing the
explicitly authorized W&B history, and retains GPU UUID/ownership,
host-memory, no-preallocation and 78 C checks. V17 remains experimental,
simulation-only, and subject to all V16 real-observation, ball-outcome and
RMP/PD evidence blockers.

## Current Experimental Bounded-Hit/Apex-Survival Continuation: V18

The preserved RMP-only critic diagnostic is
`goal_d455_measured_qvel_rmp_vertical_v18`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v18_resume_v17.sh`.
V17 remains immutable. V18 must resume the V17 checkpoint saved on entry to
stage 14, `13_rmp17_motion_quality_bridge_127_011.pkl`, at true step
`1196032000` and SHA-256
`fe1f2a66becf5fe9d7056112d534961b10e6ae769e34d66231e951f0976a7936`.
This checkpoint passed the preceding bridge before the stalled V17 commit
stage spent 1534 updates trading apex for lower angular speed and more hits.
V18 restores actor, critic and Adam moments, starts a fresh convergence window
at `rmp18_motion_quality_commit_120_010`, and must reject curriculum-history
resume, optimizer/critic reset, another source step/profile, or a non-57-D
bounded-reference checkpoint.

V18 copies V17 stages 1--13 exactly. At V18 stages 14--29 it changes only
reward/cadence fields: confirmed-hit event credit is capped at 14 per
1200-step episode, dense post-hit survival weight rises `1.4 -> 2.4`, the
existing one-sided low-apex loss activates below the 0.225 m gate by using
`low_hit_apex_margin=0.005 m` and `low_hit_penalty_weight=10000 m^-2`, and
bounded cadence credit uses target `0.43 s`, sigma `0.060 s`, weight `0.50`.
The graduation interval becomes `0.40--0.50 s`. The cap bounds event reward;
it does not terminate at 14 hits, change hit accounting, or make extra hits a
failure. Full-episode survival remains desirable and may physically require
later contacts. Do not add a maximum-hit success gate: the intended 13--14
contacts follow from the 0.225--0.248 m apex and cadence contracts, while
`conv_len`, truncation, view and safety remain the primary longevity evidence.

Control, actor/critic observation layouts, measured bounded `q_ref`, RMP/PD,
physics, DR, resets, PPO settings and all non-cadence gates are unchanged.
There is no update cap. The user explicitly authorized reuse of W&B run
`9acnp70r`; V18 uses `resume=must` and a W&B-only offset of `255852544`, so
the first V18 point follows the last V17 point while the true checkpoint step
is unchanged. This authorization does not apply to another W&B ID. V18 remains
experimental, simulation-only, and inherits every V16/V17 evidence blocker.

## Current Experimental True-Cadence/VXY-Bridge Continuation: V24

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v24`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v24_resume_v18.sh`.
V18 and the negative V19--V23 experiments remain immutable. V24 resumes only
the V18 `mjx_curriculum_best.pkl` saved at update 372, true step `1244790784`,
SHA-256
`0c705462508a0c26a3b3f92c1003adcc442240a0e7904f1c42734e98d5472955`.
It restores actor, critic and Adam moments but starts a fresh convergence
window at `rmp24_motion_quality_commit_120_010`; curriculum-history resume,
optimizer/critic reset, another source step/profile, and a non-57-D bounded
reference checkpoint must fail closed.

V18's `episode_duration/hits` cadence proxy was not an adjacent-contact
period. Frozen 128-episode validation of the selected V18 checkpoint measured
a true count-gated interval of `0.34696 +/- 0.02055 s` across the 115 full
episodes, while the proxy remained near `0.39--0.41 s`. V24 therefore uses
the event-weighted interval between adjacent count-gated hits as a health gate
at `0.300--0.400 s`. Cadence reward and min/max interval penalties are zero.
Positive hit-event credit remains capped at 14; apex, episode length,
truncation, view, safety and motion quality remain the optimization and
graduation evidence. Do not reinterpret the 0.23 m anchor-relative apex as a
same-height ballistic flight, and do not infer hit period by dividing episode
duration by hit count.

V19's 10-weight early-period loss, V20's 60-weight loss, V21's +/-0.04 m
racket-z band, and V23's bounded hit-racket-vxy quality multiplier were all
rejected in independent 512-env bounded trials because they did not improve
the 24-update cadence/vxy window; V23 also reduced return. Do not promote or
resume those trial checkpoints. V24 inherits V22/V18 rewards and changes only
the intermediate mean hit-racket-vxy gate ladder to
`[0.105,0.105,0.103,0.101,0.099,0.097,0.095,0.092,0.089,0.086,0.083,0.080,0.077,0.073,0.069,0.065] m/s`.
The final `0.065 m/s` contract is unchanged.

The preregistered 30-update V24 GPU0 trial under
`outputs/rl_sim/v19_cadence_gate_experiments_20260822/` had no failed strict
gate at its final window: hits `14.635`, length fraction `0.9308`, counted-hit
interval `0.34589 s`, mean/RMS hit-racket vxy `0.10307/0.11959 m/s`, racket
angular speed `1.09190 rad/s`, and apex `0.24453 m`. Qvel exceedance was zero,
qacc exceedance fraction averaged `0.00476` over the last 12 updates, and PPO
KL averaged `0.00627`. Its update cap was experimental only; the formal V24
launcher supplies no update cap and retains the original Stage-14 minimum 140
updates, 20-update hold, and block validation.

The user explicitly authorized V24 to continue W&B run `9acnp70r`. The
launcher uses `resume=must` and W&B-only offset `260046848`, placing its first
point one rollout after the unsaved V18 update-404 W&B tail while leaving the
selected checkpoint's true step unchanged. V24 remains experimental,
simulation-only, and subject to every unresolved real-camera, new-ball outcome
and formal RMP/PD coverage blocker.

## Current Experimental Absolute-Apex/Lift Continuation: V25

The current GPU0 successor is
`goal_d455_measured_qvel_rmp_vertical_v25`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v25_resume_v24.sh`.
V24 remains immutable. V25 resumes only the safely stopped V24
`mjx_curriculum_last.pkl` at update 565, true step `1318846464`, SHA-256
`82ff1b38f199043c8fefcda282a539d89caa418e3c70e58cc94f54f11bc8076b`.
It restores actor, critic and Adam moments but starts a fresh convergence
window at `rmp25_apex_abs_bridge_134`; curriculum-history resume,
optimizer/critic reset, another source profile/step, or a non-57-D bounded
reference checkpoint must fail closed.

V24's logged apex was anchor-relative, not absolute world-z and not the
contact-to-apex lift. At the stopped tail, about `0.255 m` anchor-relative
apex corresponded to only about `1.303 m` absolute apex. Its V18-inherited
`10000 m^-2` one-sided loss below the `0.225 m` lower gate contributed much
more optimizer credit than the bounded symmetric apex reward, while its
upper soft/event barriers did not begin until roughly `0.265 m`. This reward
and gate mismatch caused the apex distribution to drift upward without
passing the `0.248 m` upper gate. Treat that as a coordinate/reward bug, not
evidence that `1.45 m` ball apex has been reached.

V25 uses the ball's absolute world-z apex, the physical contact-edge ball and
racket heights, and contact-to-apex lift as separate quantities. Historical
profiles retain delayed-confirmation contact-height metrics; only V25 sets
`hit_contact_height_measurement_mode=contact_edge`. The ball apex target is
bridged `1.34 -> 1.37 -> 1.40 m`; the final mean band is `1.37--1.43 m`, the
final lift band is `0.18--0.26 m`, mean contact ball/racket z must remain at or
below `1.20/1.18 m`, and the true ball view/termination ceiling is tightened
from the registered `1.47 m` bound to the user's preferred `1.45 m`. The
`1.45 m` value is a ball-height ceiling, never a racket-height target.

The three bridge cadence gates are `0.31--0.43`, `0.34--0.45`, and
`0.36--0.46 s`; downstream stages require `0.38--0.46 s`. These use adjacent
counted hit events. Cadence reward and interval penalties remain zero, event
credit remains capped at 14, and survival remains primary. With contact z
around `1.18--1.20 m`, the final `1.37--1.43 m` apex band represents about
`0.17--0.25 m` ballistic rise and roughly `0.37--0.45 s` same-height flight,
leaving at least 20 mm below the preferred ball-height ceiling. V25 disables
the V18 anchor-relative loss. Its replacement is an absolute-world-z lower
barrier aligned to each stage's lower gate, a bounded symmetric absolute-apex
reward, and a separate physical-contact-to-apex lift reward/barrier. The apex
reward is opt-in decoupled from the combined lateral/pose quality multiplier;
otherwise PPO can improve the logged apex term by changing an unrelated
motion-quality score. Historical profiles retain the coupled default. The
first bridge uses source-compatible mean apex/lift floors `1.30/0.159 m` and
strict contact ball/racket ceilings `1.16/1.14 m`; the next lessons tighten
the lift floors to `0.175/0.190 m`, and downstream stages retain the final
`1.37--1.43 m`, `0.18--0.26 m`, `1.20/1.18 m` contract.

The final 512-env, 24-update bounded acceptance from the pinned source had
late-window absolute apex/lift `1.30231/0.15950 m`, versus
`1.30162/0.15914 m` in its first stable four-update window. Late hits were
`14.30`, mean hit-racket vxy `0.10356 m/s`, qvel exceedance zero, qacc
exceedance `0.00554`, and PPO KL `0.00655`. Physical-edge ball/racket contact
z stayed `1.14281/1.12290 m`. Earlier candidates that retained the coupled
reward, used only a stronger absolute reward, or used 2500/5000-weight apex
losses without a lift contract were negative or incomplete and must not be
resumed. The acceptance establishes direction and safety for continuation;
it is not evidence that the final physical apex band is already learned.

V25 adds two bridge stages after the unchanged V24 prefix, for 31 stages in
total. Every V24 stage after the stalled stage remains in order and retains
its lateral-motion/domain contracts; later V25 stages keep the committed
absolute-apex/contact/lift/cadence contract. There is no update cap. The user
explicitly authorized reuse of W&B `9acnp70r`; `resume=must` and W&B-only
offset `260571136` place the first V25 point one rollout after the interrupted
V24 tail without changing the checkpoint's true optimizer step. V25 remains
experimental, simulation-only, and inherits every unresolved real-camera,
new-ball outcome, RMP/PD coverage and deployment-adapter blocker.

## Current Experimental Survival-Balanced Continuation: V26

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v26`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v26_resume_v25.sh`.
V25 remains immutable. V26 resumes only the V25 stage-15 update-100 archive,
`archive_16_rmp25_apex_abs_bridge_137_update_0100.pkl`, at true step
`1345060864`, SHA-256
`c5f9a6997ddef0df08558ac3dbaf1b226758ab35dc6ef2b64ce6315a2f2c6b00`.
This point was selected before V25's long-run longevity regression. It restores
actor, critic and Adam moments, starts fresh V26 stage/convergence history at
`rmp26_survival_recovery_134`, and must reject curriculum-history resume,
optimizer/critic reset, another source profile/stage/update/step, and any
non-57-D bounded-reference checkpoint.

V25 stage 15 optimized height while episode quality declined: its early-to-late
windows moved `full 0.820 -> 0.743`, `conv_len 0.898 -> 0.865`, hits
`13.59 -> 12.49`, absolute apex `1.309 -> 1.337 m`, and lift
`0.163 -> 0.181 m`. About 75% of the added termination rate came from
`racket_too_high`. The existing `post_hit_survival_reward_weight` is not a
whole-flight duration reward: its geometric mask turns off after the falling
ball passes below the racket anchor. V26 therefore adds the opt-in
`post_first_hit_alive_reward_weight`, which pays each nonterminal step after
the first counted hit. Every historical profile retains the zero default.

V26 copies V25 stages 1--14 exactly. It replaces V25 stage 15 onward with
three unchanged-domain lessons, then retains every downstream V25 domain in
order, for 32 stages total. Their direct alive weights are `3.0/3.5/4.0`;
`conv_len` gates are `0.90/0.92/0.94`, full-episode gates are
`0.82/0.84/0.87`, mean hit-racket-vxy gates are `0.105/0.103/0.100 m/s`,
mean hits and conditional mean hits have a lower floor of 13, and full
racket-angular gates are `1.27/1.23/1.20 rad/s`. Height is a bounded
physical target, not a monotone objective: absolute apex target is `1.34 m`,
upper gate `1.37 m`, progressive lower gates `1.30/1.31/1.32 m`, lift target
is `0.18 m` with progressive lower gates `0.160/0.165/0.170 m`, true ball-z
ceiling stays `1.45 m`, contact ball/racket z stays at or below
`1.18/1.16 m`, and adjacent counted-hit cadence remains `0.35--0.42 s`.
Positive hit credit remains capped at 14 and cadence reward/loss remain zero.
Do not add a maximum-hit failure gate: the `13--15` healthy range is enforced
indirectly by capped credit, full survival, bounded height and cadence; later
physical recovery contacts must remain allowed.
Later stages keep target/bands `1.34`, `1.32--1.38`, `0.17--0.21 m` and
`0.36--0.42 s`, with `conv_len >= 0.90` on every stage.

The preregistered 512-env trials are under
`outputs/rl_sim/v26_survival_balance_experiments_20260823/`. Four-epoch
trials at LR `1.5e-4`, `1.0e-4`, and `7.5e-5` were rejected: rollout-wide
exact KL repeatedly landed just above the `0.008` hard limit, causing most
complete PPO updates to roll back and leaving no reliable longevity trend.
The accepted LR `1.0e-4`, three-epoch trial made continuous late updates. Its
updates 19--21 versus 22--24 moved `conv_len 0.89734 -> 0.94368`, full rate
`0.81987 -> 0.87842`, hits `13.443 -> 14.141`, mean hit-racket vxy
`0.09907 -> 0.09794 m/s`, full racket angular speed
`1.27700 -> 1.26536 rad/s`, lift `0.16712 -> 0.16795 m`, and
`racket_too_high 9.2e-5 -> 3.6e-5`. Cadence stayed
`0.36534 -> 0.36529 s`, apex stayed bounded at about `1.311 m`, qvel
exceedance was zero, qacc exceedance improved `0.00627 -> 0.00616`, and the
last-three exact KL averaged `0.00605`. This is bounded directional evidence,
not proof that later stages or hardware distributions are solved.

The formal launcher uses 1024 x 128 rollouts, LR `1.0e-4`, three PPO epochs,
clip `0.15`, target KL `0.008`, no update cap, block validation, GPU ownership,
host-memory/no-preallocation and 78 C guards. The user explicitly authorized
continuing W&B `9acnp70r`; `resume=must` and W&B-only offset `382468096`
place the first V26 rollout one rollout after the stopped V25 W&B tail without
altering true checkpoint/optimizer time. Do not reuse this authorization for
another W&B ID. V26 remains experimental, simulation-only, and inherits all
unresolved real-camera, new-ball-outcome, formal RMP/PD coverage and deployment
adapter blockers.

## Current Experimental Range-Aligned Reward Continuation: V27

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v27`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v27_resume_v25.sh`.
V25 and V26 remain immutable. V27 deliberately resumes the stronger-height
V25 archive `archive_16_rmp25_apex_abs_bridge_137_update_0150.pkl`, at true
step `1351614464`, SHA-256
`422054bf0833d63298ba4d89fbb233bf5d2154ef5e7d854dd433667a53a013ea`.
That source already passed the first V27 gate with hits `13.545`, conv_len
`0.9053`, full rate `0.8318`, counted cadence `0.3648 s`, absolute apex
`1.31447 m`, lift `0.16689 m`, hit-racket vxy `0.09976 m/s`, and full racket
angular speed `1.26871 rad/s`. Do not resume V26 update 50: although it had
slightly higher longevity, its `1.30447/0.16375 m` apex/lift basin failed to
move under several bounded reward probes. V27 restores actor, critic and Adam
moments but starts fresh V27 stage/convergence history at
`rmp27_survival_height_rebalance_134`; another source profile/stage/update,
curriculum-history resume, optimizer/critic reset, or a non-57-D bounded
reference checkpoint must fail closed.

V26's height objective was directionally correct but too weak relative to
survival, hit and motion terms, while its hit-racket-vxy and full-angular soft
limits remained much stricter than the active graduation gates. V27 keeps
survival primary and makes bounded quantities explicitly range-centred. The
absolute-apex and contact-to-apex-lift rewards are symmetric Gaussians centred
at `1.34/0.18 m`; two-sided squared dead-band losses use tolerances
`0.015/0.008 m`. Paired lower-apex and upper-apex soft guards remain active,
and counted cadence receives a small symmetric Gaussian centred at
`0.385/0.385/0.390 s`; one-sided cadence interval losses remain zero. Contact
z, vxy and angular gates are upper bounds and therefore retain upper-only
losses. Their soft limits now sit just inside the corresponding gates instead
of rewarding unnecessary motion suppression after a gate already passes.

The three unchanged-domain lessons use post-first-hit alive weights
`3.0/3.5/4.0`, apex/lift target weights `2.0/2.25/2.5`, apex dead-band weights
`1000/1200/1400 m^-2`, lift dead-band weights `2000/2400/2800 m^-2`, and
cadence weights `0.10/0.12/0.15`. Their conv_len gates are
`0.90/0.92/0.94`, full gates `0.82/0.84/0.87`, apex bands
`1.30--1.37/1.31--1.37/1.32--1.37 m`, lift bands
`0.160--0.205/0.165--0.205/0.170--0.205 m`, and counted-cadence bands
`0.35--0.42/0.35--0.42/0.36--0.42 s`. Mean hits and conditional mean hits
retain a floor of 13, positive hit-event credit stays capped at 14, and there
is no maximum-hit failure gate. The progressive mean hit-racket-vxy gates are
`0.105/0.103/0.100 m/s`; full-angular gates are `1.27/1.23/1.20 rad/s`.
Every later V26 domain remains in order with at least `0.90` conv_len and the
committed `1.32--1.38 m` apex, `0.17--0.21 m` lift and `0.36--0.42 s`
counted-cadence contracts. V27 has 32 stages and no update cap.

The 1024-env, formal-shape 30-update acceptance under
`outputs/rl_sim/v27_range_reward_experiments_20260823/` passed every strict
gate in its final rolling window. Relative to the source, its last-12 mean
conv_len/full/hits were `0.90958/0.83385/13.60364`, counted cadence was
`0.36307 s`, apex/lift were `1.31321/0.16547 m`, hit-racket vxy was
`0.10017 m/s`, and full angular speed was `1.25852 rad/s`. Thus the bounded
test establishes survival preservation, complete gate passage, safety and no
material 30-update height regression; it does not establish a statistically
significant upward height trend. Qvel exceedance was zero, qacc exceedance
was `0.00647`, and exact KL was `0.00611`. The next two stages' progressively
tighter lower gates are the required long-horizon evidence that apex/lift
actually move into the desired interior; remaining inside the first band
alone is not sufficient evidence for promotion.

The formal launcher uses 1024 x 128 rollouts, minibatch 16384, LR `1.0e-4`,
three PPO epochs, clip `0.15`, target KL `0.008`, block validation, all GPU and
host safety guards, and no update cap. The user explicitly authorized reuse
of W&B `9acnp70r`; `resume=must` and W&B-only offset `430833664` place the
first V27 point at `1782579200`, one rollout after the stopped V26 W&B tail,
without altering the checkpoint's true optimizer step. Do not reuse this
authorization for another W&B ID. V27 remains experimental, simulation-only,
and inherits every unresolved real-camera, replacement-ball, formal RMP/PD
coverage and deployment-adapter blocker.

## Current Experimental All-Tail Survival-Primary Continuation: V28

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v28`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v28_resume_v27.sh`.
V27 and every predecessor remain immutable. V28 resumes only the V27
`mjx_curriculum_best.pkl` saved at one-based stage index 16,
`rmp27_range_centered_flight_134`, update 4112, true step `1906966528`,
SHA-256
`a2e0f4efeb531defca3370d983c8075a70ec9176efd06a60aa5ed6319364012c`.
It restores the 57-D actor, 221-D critic and Adam moments, but starts a fresh
V28 convergence window at `rmp28_survival_primary_physics_band` because every
remaining stage changes reward/gate semantics. Curriculum-history resume,
optimizer/critic reset, another source profile/stage/update/step, and a
non-57-D bounded-reference checkpoint must fail closed.

V27 was stopped after more than 4,200 updates in the same stage. Its selected
best already had `conv_len=0.94310`, full rate `0.90665`, hits `14.0372`,
counted cadence `0.36917 s`, absolute apex `1.30492 m`, lift `0.16953 m`,
mean hit-racket vxy `0.08200 m/s`, and full angular speed `1.34606 rad/s`.
This is not a hit-acquisition failure. The reward audit found that its mean
per-step lift and hit-height penalties (`-0.01921/-0.01868`) together exceeded
the complete direct/geometric survival credit (`+0.01552/+0.00688`) even
though the flight was ballistically self-consistent. The former simultaneous
`1.34 m` apex and `0.18 m` lift targets were also inconsistent with the
observed roughly `1.135 m` contact height: they imply lift `0.205 m` and apex
`1.315 m`, respectively.

V28 preserves stages 1--15 exactly and preserves every V27 stage-16--32
domain transition in order, but applies one survival-primary physical contract
to that entire tail. Direct post-first-hit alive weight is 5.0. Absolute apex
and contact-to-apex lift targets are `1.32/0.175 m`, with bounded weights
0.75 and sigmas `0.050/0.040 m`. Their mean gates are
`1.28--1.39/0.145--0.220 m`. The loss dead bands cover those complete mean
bands; outside-band weights are only `100 m^-2`, and the duplicate one-sided
low-apex loss is zero. Adjacent counted-hit cadence is a reward-free health
gate at `0.33--0.44 s`. True ball-z remains below 1.45 m, and mean contact
ball/racket z remains below `1.18/1.16 m`.

All V28 stages 16--32 require mean/conditional hits at least 13 while positive
hit-event credit remains capped at 14. Their `conv_len/full` floors are at
least `0.90/0.82`; the first two unchanged-domain lessons require
`0.93/0.88` and `0.94/0.89`, and final consolidation requires `0.92/0.84`.
Every stage uses the same mean/RMS hit-racket-vxy guards
`0.120/0.170 m/s` and full-angular guard `1.40 rad/s`. Motion is softly
attracted toward `<=1.25 rad/s`; these guards prevent regression and must not
be tightened again into the rejected V24 final `0.065 m/s/0.80 rad/s` ladder
without new evidence. Ball-vxy, view, hit, safety, DR, domain and reset gates
remain stage-specific and unchanged. There is no update cap.

Paired 1024-env, 30-update GPU0 trials are recorded under
`outputs/rl_sim/v28_survival_primary_experiments_20260823/`. LR `1.0e-4`
passed every strict/recoverability/behavior gate in all late-12 windows and
improved the parent to `conv_len/full/hits=0.95201/0.92818/14.1980` while
keeping cadence/apex/lift `0.36897 s/1.30329 m/0.16839 m`, mean/RMS racket
vxy `0.08120/0.09610 m/s`, angular speed `1.34569 rad/s`, zero qvel
exceedance, qacc exceedance `0.00687`, and exact KL `0.00614`. The paired LR
`1.25e-4` trial was rejected: survival fell to
`0.93915/0.89956/13.9190`, angular speed rose to `1.36186 rad/s`, and
qacc/KL/clip were all worse. Do not resume either capped trial checkpoint;
the formal run resumes the pinned V27 parent with LR `1.0e-4`, three epochs,
1024 x 128 rollouts, block validation and no update cap.

The user explicitly authorized V28 to continue W&B `9acnp70r`. Its
`resume=must` W&B-only offset is `450887680`: the stopped V27 update-4264
point was published at `2357723136`. The first formal initialization completed
one update and published it at `2357985280` before the lineage audit stopped
the process; the restart repeats that deterministic update at the same W&B
step and then continues at `2358116352`. Thus there is one intentionally
unfilled 131072-step history slot, but no collision with, overwrite of, or
change to the selected checkpoint's real optimizer step.
Do not reuse this authorization for another W&B ID. V28 remains experimental,
simulation-only, and inherits every unresolved real-camera, replacement-ball,
formal RMP/PD coverage and deployment-adapter blocker.

## Current Experimental Bounded Ball-DR Continuation: V29

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v29`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v29_resume_v28.sh`.
V28 and every predecessor remain immutable. V29 resumes only the checkpoint
saved when V28 successfully completed one-based Stage 17,
`17_rmp28_survival_stability_commit.pkl`, at true step `1994915840`, SHA-256
`901d08e47749e51bd7305b40fd29a594bb0a66fc0009d0caf90a16325c16ec87`.
It restores the 57-D actor, 221-D critic and Adam moments but starts a fresh
convergence window at `rmp29_ball_domain_semantics_bridge`. Curriculum-history
resume, another profile/step/hash, optimizer/critic reset, an actuator/50-D
checkpoint, and a mass declaration other than `0.0030--0.0040 kg` must fail
closed.

V28 was safely stopped in Stage 18 after update 174. Its late-24 means had
collapsed to approximately hits `8.28`, `conv_len=0.569`, and full rate
`0.502`, while the episodes that acquired at least three contacts still
averaged approximately `14.20` hits. The Stage-17-to-18 transition changed
mass `3.0625--3.4375 -> 2.5--4.0 g`, normalized inertia
`0.40--0.54 -> 0.40--2/3`, spin XY/Z `+/-25/20 -> +/-55/40 rad/s`, and
gravity `-9.83..-9.79 -> -9.88..-9.72 m/s^2` together. Stage 19 would then
also widen ball/racket friction, solref time and solref damping. Treat this as
an early-acquisition failure under a simultaneous physical-DR cliff, not as
evidence that the V28 survival reward or bounded measured-q/RMP path failed.

V29 preserves V28 stages 1--17 exactly and expands the complete course to 34
stages. Stage 18 isolates V28's new course/reward semantics while keeping the
learned Stage-17 physical ranges exactly: mass `3.0625--3.4375 g`, normalized
inertia `0.40--0.54`, spin XY/Z `+/-25/20 rad/s`, gravity
`-9.83..-9.79 m/s^2`, ball/racket sliding friction
`0.16--0.28/0.30--0.48`, solref time `3--6 ms`, and damping `0.74--0.94`.
Stage 19 changes only mass/inertia to `3.0--3.7 g/0.45--0.60`. Stage 20 and
every later stage fix mass/inertia at `3.0--4.0 g/0.54--2/3`; spin, gravity
and contact stay at the learned mild Stage-17 ranges. No later stage may
restore V28's wide physical intervals. The hollow-sphere value `2/3`
motivates only the upper inertia endpoint; these intervals remain unmeasured
experimental candidates and are not replacement-ball or hardware-coverage
evidence. The rejected simultaneous Stage-18 candidate and its 30/60-update
negative trials are retained in the V29 experiment report.

The revised Stage-18 semantics bridge passed its preregistered protected
GPU0 screen at production shape (seed `20260822`, 1024 environments by 128
steps). Its final 24-update window after 40 experimental updates measured
hits `14.1671`, length fraction `0.94155`, full rate `0.90933`, conditional
hits `14.7460`, counted-hit interval `0.36442 s`, absolute apex `1.29317 m`,
lift `0.16345 m`, mean/RMS hit-racket vxy `0.08999/0.10480 m/s`, and full
racket angular speed `1.25392 rad/s`. Qvel exceedance was zero; qacc
exceedance averaged `0.00590` and exact KL `0.00639` over the last 12
updates. The trial's update-40 exit is the intentional experimental cap,
which precedes the inherited 160-update minimum and therefore is not a
formal graduation or block-validation result. The uncapped formal course
must still earn every rolling and block-validation gate before Stage 19.

Reward, gates, control, observation, bounded `q_ref`, RMP/PD, resets, other DR
axes and PPO settings are unchanged from V28. The formal launcher retains
1024 x 128 rollouts, LR `1.0e-4`, three epochs, clip `0.15`, target KL
`0.008`, block validation, no update cap, no XLA preallocation, host-memory
and 78 C guards. The user explicitly authorized reuse of W&B run `9acnp70r`.
The last completed V28 row was true step `2017722368`, published at
`2468610048`; V29 uses W&B-only offset `473694208`, so its first completed
rollout is published at `2468741120` without changing true optimizer time.
Do not reuse this authorization for another W&B ID. V29 remains experimental,
simulation-only, and inherits every unresolved real-camera, replacement-ball,
formal RMP/PD coverage and deployment-adapter blocker.

## Provisional Execution-DR Bridge Continuation: V30

The provisional GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v30`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v30_resume_v29.sh`.
Do not launch it formally until the matched five-arm frozen-policy motion-
distribution audit selects the execution range, the builder is updated to
that result, and the resulting course/reward/PPO combination passes its
bounded acceptance experiment.
V29 and every predecessor remain immutable. V30 resumes only the clean V29
Stage-21 pass archive, `21_rmp29_launch12_contact_dynamics_wide.pkl`, at true
step `2087190528`, SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`.
It restores the 57-D actor, 221-D critic and Adam moments but starts fresh
V30 stage/convergence history at `rmp30_rmp_internal_micro_6p25`. Another
profile/step/hash, curriculum-history resume, optimizer/critic reset, an
actuator/50-D checkpoint, or a mass declaration other than `0.0030--0.0040
kg` must fail closed.

The real-data selection contract is
`rmp_effective_juggling_validity_v3.json`, SHA-256
`391bd03a67ad47e923c8b178c4939d74201169556d6445e6ef6d108f429b0f4f`.
It preserves the fixed effective-juggling, safety, freshness, proximity,
speed and nominal-outlier rules and changes only the valid true-ball ceiling
from 1.75 m to the existing in-view/safety contract of 1.45 m. It accepts the
same 56/64 parseable recordings from record_new3/4/5 and retains 20,047
post-warmup wall-grid samples. Failed/post-drop/stale tails and trajectories
above 1.45 m are excluded; valid hard trajectories must not be discarded to
make coverage pass. Audit joint q/dq/qdd and racket-centre position,
orientation, linear/angular velocity and acceleration pointwise.

The old broad envelope is not a formal coverage solution: the prior worst-four
screen achieved only approximately `90.1/92.6/84.9%` pointwise joint
q/dq/qdd coverage despite RMP and PD scales as wide as `0.5--1.5`; all 105
candidates being numerically stable did not close the systematic nominal
closed-loop gap. Do not widen execution ranges merely to improve a global
percentage. The provisional builder tests the smaller
12.5%-toward-broad candidate: RMP Kp/Kd
and PD Kp/Kv `0.85--1.15`, estimator process `0.73125--1.40625`, measurement
`0.6575--1.53875`, velocity feedforward `0.41875--0.58125`, acceleration
weight `0.73125--1.46875`, target-filter length `8--12`, RMP output-delay
offset `-2..2` ticks, damping `0.8375--1.1875`, and armature
`0.8375--1.2125`. These are experimental training candidates, not formal
hardware coverage or a replacement for nominal RMP/low-level calibration.
These values are not selected or final before the matched distribution result.

V29 Stage 22 was causally confounded: it simultaneously widened twelve active
execution axes to its 25% candidate, doubled missed-target probability, added
the long hold tail, increased several auxiliary motion/centering penalties,
tightened soft limits, and tightened graduation gates. It spent 670 updates
below the Stage-21 survival baseline even though PPO remained numerically
healthy. V30 therefore preserves V29 stages 1--21 exactly and expands the
complete course to 40 stages. Stages 22--25 split RMP-internal and PD/plant
uncertainty into 6.25% micro and 12.5% commit lessons. Stages 26--27 then add
3.5% and measured 5%/1.1% missed-publication bridges. Every later V29 task,
observation and XY domain remains in order, but execution DR is capped at the
12.5% candidate and the old broad envelope is never restored. A 6.5%/1.65%
timing-only lesson bridges the late measured 8%/2.2% setting. Auxiliary
motion/centering losses never exceed Stage-21 strength; the direct
post-first-hit survival signal remains primary. No stage has an update cap.

The stopped V29 Stage-22 last-24 PPO window had exact KL `0.00608`, clip
fraction `0.1565`, explained variance `0.8968`, zero raw-action clipping,
zero qvel exceedance and qacc exceedance fraction `0.00550`. V30 therefore
retains 1024 x 128 rollouts, LR `1.0e-4`, three epochs, clip `0.15`, target
KL `0.008` and entropy `2e-4`; do not increase LR without a paired failure of
the isolated bridge. The complete preregistration and candidate evidence are
in `outputs/rl_sim/rmp_execution_dr_reaudit_20260823/V30_EXECUTION_DR_REPORT.md`.

The 1024-env, 30-update Stage-22 entry trial passed its final rolling health
window: hits `13.4798`, `conv_len/full=0.90592/0.85433`, conditional hits
`14.3494`, counted cadence `0.36745 s`, absolute apex/lift
`1.29505/0.16689 m`, mean/RMS hit-racket vxy `0.10045/0.11696 m/s`, and
full racket angular speed `1.23265 rad/s`. Last-12 qvel exceedance was zero,
qacc exceedance averaged `0.00526`, exact protected KL averaged `0.00566`,
and explained variance was `0.89866`. Updates 29--30 proposed KL just above
the `0.008` target and were safely rolled back; this is evidence against an
LR increase, not a policy failure. The 30-update cap is experimental and
below the 80-update lesson minimum. Never resume its checkpoint; formal V30
must restart the pinned clean Stage-21 archive with no update cap.

The user explicitly authorized a validated V30 successor to continue W&B run
`9acnp70r`; this does not waive the preceding DR/course acceptance order. V29 ended
at true step `2175008768`, published at W&B step `2648702976`; V30 resumes
the clean true step `2087190528` archive with W&B-only offset `561512448`, so
its first completed rollout is published at `2648834048` without changing
optimizer time. Do not reuse this authorization for another W&B ID. V30
remains experimental, simulation-only, and inherits every unresolved
real-camera, replacement-ball, formal RMP/PD coverage and deployment-adapter
blocker.

## Current Experimental Execution-DR Polish Continuation: V31

The current opt-in continuation is
`goal_d455_measured_qvel_rmp_vertical_v31`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v31_resume_v29.sh`.
V30 and every predecessor remain immutable. V31 resumes the same clean V29
Stage-21 archive at true step `2087190528`, SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`;
it restores the 57-D actor, 221-D critic and Adam moments and starts fresh
stage/convergence history at `rmp31_rmp_internal_micro_6p25`. It must reject
another source profile/step/hash, curriculum-history resume, optimizer/critic
reset, an actuator/50-D checkpoint, or a ball-mass declaration other than
`0.0030--0.0040 kg`.

V31 changes no V30 physics, DR bound, control, observation, reset, reward,
gate or PPO value. It retains the selected 12.5%-toward-broad execution cap,
the isolated 6.25% RMP and PD/plant half-steps, and the 3.5% then measured
5%/1.1% target-publication progression. After every one of those six
difficulty changes, it inserts a separate unchanged-domain polish stage.
Each polish stage copies the preceding environment configuration, reward and
all graduation gates exactly, resets only the rolling/convergence history,
runs for at least 80 updates with at least a 16-update hold, and still requires
block validation. The late 6.5%/1.65% target-hold bridge receives the same
polish treatment; the existing final unchanged-domain consolidation remains
the polish after the measured 8%/2.2% setting. The resulting course has 47
stages and no update cap.

The polish stages are deliberately not rewarded for merely remaining in the
same stage. `post_first_hit_alive_reward_weight`, the 14-event positive-hit
credit cap, two-sided apex/lift/cadence shaping, and motion/safety penalties
are unchanged from the production-shape V30 entry trial. Primary health is
stable `conv_len`, full rate and approximately 13--15 hits; apex, lift,
counted-hit period, racket vxy, full angular speed, view and qvel/qacc must
remain inside their existing bands without a worsening trend. A polish stage
does not relax a failed gate or force advancement.

The user accepted the effective-trajectory replay screen as sufficient to
proceed with this explicitly experimental curriculum, but that screen is not
formal hardware-coverage evidence. Future coverage calculations must retain
only intervals between adjacent successful contacts and must discard the
entire post-last-contact failure tail; sample-wise z/workspace filtering alone
is insufficient. V31 remains simulation-only and inherits every unresolved
real-camera, replacement-ball, formal RMP/PD and deployment-adapter blocker.

V31 retains 1024 x 128 rollouts, LR `1.0e-4`, three epochs, clip `0.15`,
target KL `0.008`, entropy `2e-4`, no XLA preallocation, the host-memory guard
and the 78 C stop. Do not raise LR: the accepted V30 entry trial already
triggered protected KL rollbacks at updates 29--30. The user explicitly
authorized W&B run `9acnp70r`; because V30 never appended formal points, V31
uses `resume=must` and the same W&B-only offset `561512448`, placing its first
completed rollout at `2648834048` without changing true optimizer time. Do
not reuse this authorization for another W&B ID.

## Structured-RMP/RMP-Privileged-Critic Diagnostic: V35

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v35`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v35_resume_v31_stage23.sh`.
V31 and the bounded V32--V34 diagnostic profiles remain immutable. V35 resumes
only the passed V31 Stage-23 archive,
`23_rmp31_rmp_internal_micro_6p25_polish.pkl`, at true step `2111569920` and
SHA-256
`93a4002fe579dd34b1768b058c7e81f9c9def937afbfea64cac182a5b98498dc`.
It starts fresh V35 convergence history at
`rmp35_rmp_internal_commit_12p5`; curriculum-history resume, another
profile/step/hash, actor migration, optimizer/critic reset, or a ball-mass
declaration other than `0.0030--0.0040 kg` must fail closed.

V32's added 9.375% midpoint/polish and success-focused rollout weighting were
negative diagnostics: after both lessons the direct 12.5% final windows stayed
near `conv_len/full=0.878/0.815` and `0.872/0.805`. Do not add those stages to
the formal course. V33 instead fixes a control-model bug: physical scalar RMP
fields (estimator process/measurement variance, velocity feedforward,
acceleration weight, and target-filter length) draw once per environment and
are broadcast to seven joints. V34 fixes the corresponding gain correlation:
RMP Kp/Kd and XML-PD Kp/Kv draw one global episode multiplier on top of their
fixed calibrated per-joint vectors; output-delay offsets remain per joint.
The historical independent-per-joint sampling remains the default for every
old profile.

The prior 221-D asymmetric critic already contained true ball/racket state,
command errors and history, XML-PD Kp/Kv, damping, armature, ball/contact DR
and legacy timing values. It did not contain `rmp_jnt_kp/kd`, either RMP
estimator variance, RMP velocity feedforward, RMP acceleration weight, RMP
target-filter length, or per-joint RMP output delay; the optional 48-D
sim2real critic block omitted them too. V35 appends these eight normalized
seven-joint vectors only to the critic, producing 277-D critic input while
the deployable actor remains exactly 57-D. The historical 221 values are an
exact prefix. Resume migration must preserve the complete old critic, actor,
Adam first/second moments and Adam step and zero-initialize only the 56 new
critic input rows. It requires the explicit `--allow-obs-dim-migration` flag.

V35 changes no V31 stage count, stage ordering, range endpoint, reward, gate,
reset, control, actor observation or PPO field. It has the same 47 stages and
begins at Stage 24; no new bridge/polish stage is added. The matched 64-update
GPU0 screen at LR `1e-4` improved the V31 first-64 window from
hits/length/full `12.879/0.871/0.798` to `13.236/0.892/0.830`. Apex/lift were
`1.295/0.167 m`, cadence `0.367 s`, mean/RMS hit-racket vxy
`0.101/0.119 m/s`, full angular speed `1.209 rad/s`, qvel exceedance zero,
qacc exceedance `0.00537`, and exact KL `0.00571`. The final-window survival
slopes were positive, but the capped screen was not a formal `0.90` length
graduation. A matched LR `1.5e-4` trial regressed to length/full
`0.886/0.813` with negative survival slopes and is rejected.

The V35 launcher retains LR `1.0e-4`, 1024 x 128 rollouts,
three epochs, clip `0.15`, target KL `0.008`, block validation, no update cap,
the host-memory guard and 78 C stop. The user explicitly authorized W&B run
`9acnp70r`. V31's last point was true step `2228748288`, published at
`2790260736`; V35 uses W&B-only offset `678690816`, mapping its first rollout
to `2790391808` without changing source or optimizer time. The bounded
diagnostic checkpoints are evidence only; formal V35 restarts the clean V31
Stage-23 pass. Complete evidence is in
`outputs/rl_sim/v35_rmp_privileged_critic_experiments_20260824/V35_RMP_PRIVILEGED_CRITIC_REPORT.md`.

V35 is not the complete asymmetric-critic DR contract. Its attempted formal
launch on 2026-08-24 was interrupted after CUDA/environment initialization and
before the first `update=` record, so it changed neither the policy nor the
last W&B metric step. Preserve that output directory as aborted evidence and
do not reuse it.

## Preserved Complete-DR Privileged-Critic Continuation: V36

The current GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v36`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v36_resume_v31_stage23.sh`.
V35 remains immutable. V36 resumes the same passed V31 Stage-23 archive, true
step and SHA-256 as V35, starts fresh at
`rmp36_rmp_internal_commit_12p5`, and requires the explicit append-only critic
migration. It must reject curriculum-history resume, actor migration,
optimizer/critic reset, another source profile/step/hash, or a mass declaration
other than `0.0030--0.0040 kg`.

V36 preserves V35's 56-D RMP critic block and adds the complete 91-D
complement. The resulting 368-D critic is exactly the historical 221-D prefix,
then the eight seven-joint RMP groups, then every remaining realized DR value:
episode delay base/bin; second-order residuals; ball inertia, reset spin and
hard-tail selection; observation-frame position/rotation/velocity bias and
scale; previous-action scale; target/execution hold state; observation
missing/dropout/correlated-noise state and randomized high-z visibility; and
hidden reset/target/time-limit context. Actor input stays 57-D. Range endpoints
are stage configuration, not environment samples; the critic receives the
realized values. Per-refresh independent noise is already recoverable from the
actor-observation prefix and privileged true state, while its correlated
latent state is explicit.

`DOMAIN_RANDOMIZED_STATE_FIELDS`, `RESET_RANDOMIZED_STATE_FIELDS`, and the
three critic-coverage registries are a mandatory omission guard. Tests must
show that their critic union exactly covers the randomized-state registries
and that every `EnvState` field named `dr_*` is registered. A new DR field may
not enter any profile until this audit remains passing.

V36 changes no stage count/order, DR support or probability, reward, gate,
reset distribution, control, actor input or PPO setting. In the matched
64-update GPU0 screen, final-window V36 versus V35 hits/length/full were
`13.2235/0.8916/0.8292` versus `13.2359/0.8922/0.8297`; this is materially
neutral. V36 improved full racket angular speed `1.1957` versus `1.2089 rad/s`,
explained variance `0.9028` versus `0.9020`, and value loss `64.09` versus
`65.71`. Exact KL was `0.00630`, qvel exceedance zero and qacc exceedance
`0.00536`. Cadence `0.3669 s`, apex/lift `1.2947/0.1664 m`, and recurrent
racket vxy `0.1030/0.1181 m/s` remained protected.

The formal V36 launcher uses LR `1.0e-4`, 1024 x 128 rollouts, three epochs,
clip `0.15`, target KL `0.008`, the unchanged 47-stage course from Stage 24,
block validation and no update cap. It resumes W&B `9acnp70r` with `must` and
the unchanged W&B-only offset `678690816`; V35 wrote no completed point, so
V36's first rollout maps to `2790391808`. Formal V36 restarts the clean V31
Stage-23 checkpoint, never either capped diagnostic. Full evidence is in
`outputs/rl_sim/v36_complete_dr_critic_experiments_20260824/V36_COMPLETE_DR_CRITIC_REPORT.md`.

V36 was stopped safely on GPU0 at Stage 26
`rmp36_pd_plant_micro_6p25`, update 725, true step `2276458496`. The selected
`mjx_curriculum_last.pkl` SHA-256 is
`79e8b64cc033d7a2c4fcc4fc8abba51f593fb7e06dfa8c6a37a2b655a386a3c3`.
Preserve this checkpoint and output directory as V36 stop evidence.

## Current Non-Execution-First Full-Episode Continuation: V37

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v37`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v37_resume_v36.sh`.
It resumes only the screened V36 Stage-26 `mjx_curriculum_best.pkl` at update
303, true step `2221146112`, SHA-256
`eb6cf8c1c71c5438aa3ef0eb60a1965f7636bc1990b08a6020f27b2f10a8520d`, and preserves the 57-D
actor, 368-D complete-DR critic, Adam moments, true global step, bounded-qref
control, reward and every V36 DR endpoint. It starts fresh V37 curriculum
history and must reject curriculum-history resume, optimizer/critic reset,
observation migration, another source profile/stage/step/hash, or a mass
declaration other than `0.0030--0.0040 kg`.

V37 changes ordering, not support. Stages 1--19 first learn V36's remaining
target-publication, racket-geometry, observation-calibration, camera-missing,
in-view ball-state/target-XY and final consolidation domains. Throughout this
prefix, only the twelve recovered-RMP, XML-PD, damping and armature interval
fields remain at the V36 Stage-21 small envelope; their physical scalar/global
sampling contract is already active. Stage 20
`rmp37_complete_nonexecution_full_episode_polish` holds the complete V36
non-execution domain and requires both rolling and block validation to reach
exact `conv_len=1.0` and full-episode rate `1.0` for 24 consecutive eligible
updates. Its normal stage archive is the protected runnable-policy checkpoint.

Only after Stage 20 passes do Stages 21--28 widen RMP internal ranges to V36's
6.25% and complete 12.5% envelopes, then PD/plant to their 6.25% and complete
12.5% envelopes, with an unchanged-domain polish after each increment. All
non-execution DR stays fixed at the Stage-20 maximum domain and never rolls
back. The final configuration must equal V36's final configuration exactly.
No stage has an update cap and unconverged advancement remains prohibited.

The user explicitly authorized V37 to continue W&B run `9acnp70r` with
`resume=must`. V36's last saved/history point is true/W&B step
`2276458496/2955149312`; V37 uses W&B-only offset `734003200`, mapping the
selected older checkpoint's first rollout to `2955280384` without changing
checkpoint or optimizer time. The launcher retains LR `1.0e-4`, 1024 x 128 rollouts,
three epochs, clip `0.15`, target KL `0.008`, block validation, host-memory and
78 C guards, and manual `CONFIRM_GPU0_READY=YES` plus incomplete-evidence
acknowledgement. V37 remains experimental, simulation-only, non-deployable and
inherits every unresolved camera, replacement-ball and formal pointwise
RMP/PD coverage blocker. Repository changes do not start PPO automatically.

## Current Reward-Component-Attributed Continuation: V43

V38--V42 are preserved bounded diagnostics, not formal continuations. V38
made every policy-controlled gate reward-visible; V39 increased apex/survival
weights; V40 continued V39; V41 shifted more credit to survival and racket-z;
V42 applied V41 rewards from the V39 update-95 source. None passed the raw
primary contract. In particular, do not repeat the earlier analysis error of
averaging already rolling `convergence/recent_*` values: V39 raw final-24
length/full/apex were `0.93228/0.90855/1.28208 m`, and interrupted V42 raw
final-24 values were `0.92361/0.89048/1.28774 m`.

The preserved bounded candidate is
`goal_d455_measured_qvel_rmp_vertical_v43`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v43_resume_v39.sh`.
It resumes only V39 Stage-1 best update 95, true step `2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`,
with actor, critic and Adam state preserved and fresh V43 convergence history.
It must reject another source, curriculum-history resume, optimizer/critic
reset, observation migration and from-scratch launch.

V43 follows a 256-lane paired frozen-policy attribution, not a parameter
sweep. Under one V42 environment/seed, V42-minus-V39 return was `+1.233` even
though full was `-0.00391`; `+2.144` came from reducing the low-apex loss while
all other components summed to `-0.912`. V43 therefore changes exactly three
reward semantics throughout the otherwise unchanged 28-stage V42 course:
low-apex weight `7000 -> 2500 m^-2`, ball-miss base cost `11.46 -> 40`
including pre-first-hit misses, and direct full-horizon completion reward
`0 -> 100`. Frozen counterfactual scoring reverses V42-minus-V39 to `-0.648`
and raises the full/failed reward gap from about `190` to `316`. All gates,
RMP/PD, bounded reference, observation/critic layouts, physics, resets and DR
remain unchanged. Full attribution and hashes are in
`outputs/rl_sim/v38_gate_reward_experiments_20260824/REWARD_COMPONENT_ROOT_CAUSE_ANALYSIS.md`.

The preregistered 512-env, 128-update bounded trial rejected V43. Raw final-24
length/full/hits were `0.92910/0.90034/14.1851` against `0.95/0.92/13`; the
best raw 24-update length/full windows were only `0.93272/0.90306`. The direct
completion term contributed `0.08170` per simulator step against total reward
`0.22407`, while 63/64 late PPO updates were accepted, so neither missing
reward magnitude nor KL rejection explains the plateau. Frozen post-trial
paired replay moved full only `0.930 -> 0.934`, below its sampling resolution.
Do not launch V43 formally or use its checkpoints as a continuation source.

## Preserved Negative Historical-Survival Bridge: V44

The preserved bounded candidate is
`goal_d455_measured_qvel_rmp_vertical_v44`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v44_resume_v39.sh`.
It resumes the same immutable V39 Stage-1 best update-95 checkpoint and starts
fresh curriculum history. V44 rejects V43's terminal-completion intervention
and restores the actual V28 training contract: rollout 128, gamma `0.9995`,
GAE lambda `0.99`, LR `1e-4`, clip `0.15`, target KL `0.008`, three epochs.

V37 Stage 1 already matches V28's later contact-wide reward values, but it
skipped the earlier V28 survival-primary reward bridge that produced the
accepted `conv_len/full=0.95201/0.92818` bounded result. V44 inserts exactly
one bridge before the complete 28-stage V37 course. It changes only fourteen
secondary view, racket-z/pose, contact-centering and cycle-motion penalty
weights to their V28 Stage-16 values, about 15--20% below V37 Stage 1. Primary
survival, apex/lift, hit rewards, every observation/control/physics/DR value,
small pre-polish execution DR and the 57-D actor/368-D critic remain unchanged.
Stage 2 restores V37 Stage-1 weights before any new domain lesson.

The 1024-environment, 30-update bounded trial rejected V44. Raw final-12
length/full/hits were `0.92396/0.88843/14.2112`; full had a negative
`-0.000173/update` slope despite accepted PPO updates. Frozen 256-lane paired
replay moved V39 source to V44 last from `0.91797 -> 0.88281` full and
`1127.3 -> 1105.2` steps. Seven previously failed lanes recovered, but sixteen
previously full lanes failed, for a net loss of nine. The small secondary
motion-quality component improvements were outweighed by lower dense/alive,
hit-floor, hit and apex/lift credit. Do not launch V44 formally or resume its
checkpoint. V44 remains experimental, simulation-only negative evidence.

## Preserved Soft-Anchor Diagnostic: V45

The preserved bounded diagnostic is
`goal_d455_measured_qvel_rmp_vertical_v45`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v45_resume_v39.sh`.
It resumes the immutable V39 Stage-1 best update-95 source and preserves its
actor, critic, Adam moments, reward, gates, observations, control, physics,
resets and every DR value. V45 starts fresh convergence history at
`rmp45_successful_replay_survival_polish`; curriculum-history resume,
optimizer/critic reset, observation migration and another source fail closed.

V43/V44 establish the root cause as successful-lane erosion under a mean PPO
objective, not missing reward magnitude or rollout length. V45 therefore
uses the repository's earlier successful GPU0 replay-anchor polish contract:
1024 x 128 rollouts, two epochs, LR `5e-6`, clip `0.08`, target KL `0.003`,
entropy zero, current-source KL coefficient `0.02`, successful-replay KL
coefficient `0.05`, and failure focus disabled. The replay artifact contains
only 227 full 1200-step V39 lanes from a 256-lane screen, deterministically
subsampled to `65536 x 57`, SHA-256
`850b581f23d63d7eea012414b70d73bda412d913c35024a202f5facbbdfe2042`;
all 29 failed lanes are excluded.

The 64-update, 1024-environment bounded trial rejected V45. Its best 12-update
window reached length/full `0.94051/0.91078`, but final-24 settled at
`0.92322/0.88997`, below `0.95/0.92`. Hits, apex/lift, cadence, view and safety
remained healthy; critic EV was about `0.963` and no PPO update rolled back.
Successful-replay KL nevertheless accumulated to `0.00848`. The early
improving window occupied about `0.004--0.006` replay KL, after which survival
stalled. Thus a soft anchor reduces update size but neither assigns lane-level
success credit nor bounds cumulative erosion. Do not launch V45 formally or
resume its checkpoints.

## Preserved Full-Success Risk Diagnostic: V46

The preserved bounded diagnostic is
`goal_d455_measured_qvel_rmp_vertical_v46`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v46_resume_v39.sh`.
It again resumes only V39 Stage-1 best update 95 and keeps every V39 reward,
gate, observation, control, reset, physics and DR value. It preserves actor,
critic and Adam moments while starting fresh history at
`rmp46_full_success_risk_polish`.

V46 keeps V45's low-LR replay-anchor optimizer but adds the missing actor-loss
semantics: positive normalized advantages attributable to completed
time-limit lanes with at least 13 hits receive weight `3.0` within the final
128 rollout transitions. Negative advantages in successful lanes and all
ordinary failed/unfinished-lane gradients remain active, so this is not
behavior cloning or a failure-data exclusion. The measured positive-focus
fraction must be logged. Cumulative source-policy KL on the hashed successful
replay set is capped at `0.006`; crossing candidates are interpolated to the
largest safe step, retaining learning tangent to or back toward the boundary.

Two 1024-environment bounded trials reject V46 for formal use. The first
64-update run ended with a promising final-12 length/full window of
`0.93907/0.91348`, but the independent 128-update replication ended at only
`0.93036/0.90015` over final 24; raw length sloped `-0.55 step/update` in that
window and the best rolling length/full was only `0.93420/0.90518`. Secondary
metrics remained stable: apex/lift `1.28572/0.15771 m`, view `0.96835`, qvel
exceedance zero and qacc exceedance `0.00477`. Total reward per step improved
by `+0.00542` from first to final 24 while full did not, so reward magnitude is
not the remaining cause. The focused success fraction was only `0.0161`; a
`3x` sample multiplier therefore adds about 3.2% aggregate actor mass and no
explicit failure-tail mass. Do not launch V46 formally or resume its trial
checkpoints.

## Preserved Negative Balanced-Outcome Diagnostic: V47

The current candidate is `goal_d455_measured_qvel_rmp_vertical_v47`, launched
only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v47_resume_v39.sh`.
It resumes the same immutable V39 Stage-1 update-95 source and preserves every
V39 reward, gate, observation, control, reset, physics and DR value, plus the
critic and Adam moments. It starts fresh history at
`rmp47_balanced_outcome_risk_polish`.

V47 fixes the population-frequency defect rather than changing another reward
component. Within the final 128 causally attributable rollout transitions,
negative normalized advantages from completed true failures below 13 hits and
positive normalized advantages from completed time-limit successes with at
least 13 hits each receive `0.15` additional *batch-average* actor mass. Thus
each outcome group retains a fixed optimization share when it occupies about
1% of samples; wrong-sign advantages and unfinished episodes remain ordinary.
The combined sample-weight mean is `1.30` when both groups are present, after
which advantages are RMS-normalized. Legacy multiplier behavior is unchanged
when both balanced-mass options are zero.

V47 retains V46's 1024 x 128, two-epoch, LR `5e-6`, clip `0.08`, target-KL
`0.003`, entropy-zero, current/replay KL `0.02/0.05` and absolute successful
replay KL `0.006` contract. The complete 128-update bounded run is preserved at
`outputs/rl_sim/v47_balanced_outcome_risk_trial_seed20260908_u128`. It did
oscillate and recover from its middle low, but did not show an upward platform:
episode-weighted length/full over updates 9--32 were `0.92524/0.89901`, versus
`0.92793/0.89594` over updates 97--128. The best mature rolling pair was only
`0.93059/0.90081`. Secondary metrics and safety remained stable.

The decisive first-episode paired screen used the same 256 environment lanes
and seed for the V39 source, V47 best and V47 last. Source/best/last achieved
full rates `241/235/237` of 256 and mean lengths
`1147.73/1130.85/1135.58` steps. V47 best rescued four source failures but
broke ten source successes; last rescued five but broke nine. Thus this is not
reasonable short-term oscillation hiding a net improvement. Balanced
critic-signed advantage mass changes which successful and failed lanes trade
places, but does not impose a no-regression constraint on the primary episode
outcomes. Do not launch V47 formally or resume its checkpoints.

This evidence closes reward/weight/optimizer scanning as the remedy for the
current polish failure. PPO optimizes a scalar transition objective; mean KL
limits distribution displacement but not discontinuous episode survival;
curriculum gates detect damage only after an update; and the historical
best-checkpoint weighted sum does not give `conv_len/full` lexicographic
priority. The next implementation must therefore make policy promotion
transactional and default-off for legacy profiles: compare candidate and
incumbent on identical frozen lanes, permit oscillation inside a candidate
block, commit actor/critic/Adam state only when the primary length/full
platform improves without a secondary/safety bound violation, and otherwise
roll back the complete optimizer transaction. Do not create another reward or
hyperparameter variant before this update-acceptance contract is implemented
and regression-tested.

## Preserved Transactional Main-Metric Experiment: V48

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v48`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v48_resume_v39.sh`.
It resumes only the immutable V39 Stage-1 best at update 95, true step
`2253586432`, SHA-256
`355f518458e8913e1898c4c17e751984425139ac03f7e484fe64da0be04e8a7c`.
V48 is a value-for-value copy of all 28 V39 stages: reward, gates, actor and
critic observations, bounded-reference/RMP/PD control, physics, resets, DR
and PPO stay unchanged. It starts fresh history at
`rmp48_transactional_main_metric_polish`; curriculum-history resume,
optimizer/critic reset, observation migration, outcome reweighting and
another source must fail closed.

V48 fixes the update-acceptance defect established by V43--V47. Every eight
PPO updates form a candidate transaction. The candidate and accepted
incumbent are evaluated deterministically on the same 256 first-episode lanes
at seed `20260909`. While either primary gate is missed, both individual
`conv_len/full` gaps must be non-increasing and their combined gap must
strictly decrease. Once both are in range, ordinary movement inside the
accepted range is allowed. Every active stage behavior gate must remain in
range, or move no farther from its range if the incumbent has not reached it.
Transaction-only safety bands additionally require zero qvel-limit
exceedance and at most `0.01` qacc-limit exceedance. Actor, critic and all Adam
moments commit atomically; a rejected transaction restores the last accepted
training state. Candidate oscillation within the eight-update block remains
allowed. Historical profiles retain the old behavior because the guard is
default-off.

The single preregistered bounded GPU0 run is
`outputs/rl_sim/v48_transactional_rootfix_trial_seed20260829_u32`. It used the
formal 1024 x 128 rollout, minibatch 16384, three epochs, LR `1e-4`, clip
`0.15`, target KL `0.010`, 256 transaction lanes and four eight-update
transactions. The frozen accepted length/full sequence was
`0.92379/0.89062 -> 0.94777/0.92578 -> 0.94945/0.92188`. The first two blocks
were accepted with zero secondary/safety regressions. The next two candidates
fell to `0.93973/0.90234` and `0.93717/0.89844` and were rejected. Their lane
transitions were respectively `+8/-13` and `+7/-13`; those regressions never
entered a saved accepted policy. The saved Adam counter advanced by exactly
384 minibatches, equal to 16 accepted updates x 3 epochs x 8 minibatches,
despite 32 attempted updates, proving complete optimizer rollback. Raw
boundary rollouts had qvel exceedance zero and qacc exceedance
`0.00449--0.00480`. The final accepted checkpoint and progress hashes are
`70ced04838db78f6a4083736ae2fbf6f6727f570d9e6c9484312f04d76da0530`
and `ce5a9cd2cafb950ca77d41d44c65c03b201bce526d8fbc06cabb3a47ae0e77ab`.

The bounded cap intentionally ended before stage convergence: accepted
`conv_len=0.94945` is still `0.00055` below the `0.95` gate, so do not call
this a converged model. It is evidence that the root failure is now contained:
harmful oscillations are observed and rolled back while safe progress is
retained. The formal launcher has no update cap, retains W&B run `9acnp70r`
with `resume=must` and W&B-only offset `728039424`, and must run in tmux
`pp_gpu0`. Monitor the accepted transaction series, not every raw rollout.
Do not weaken the transaction population, interval, primary rule or safety
bands in response to rejected candidates, and do not return to reward or PPO
parameter scanning. V48 remains experimental, simulation-only and inherits
all unresolved real-camera, replacement-ball and formal pointwise RMP/PD
coverage blockers.

Formal V48 started in tmux `pp_gpu0` on 2026-08-24/25 at
`outputs/rl_sim/measured_qvel_rmp_vertical_v48_gpu0_seed20260829_20260824_transactional_resume_v39_best95`.
W&B `9acnp70r` resumed successfully. Its process-local frozen incumbent began
at length/full `0.92795/0.89453`. The first update-8 transaction improved to
`0.94257/0.92188`, changed full lanes by `+11/-4`, reported zero
secondary/safety regressions, qvel exceedance `0`, and qacc exceedance
`0.004428`; it was committed. The user authorized a safe stop on 2026-08-25
after attempted update 107. Of 13 complete transaction boundaries through
update 104, only updates 8 and 80 committed; the other 11 were rolled back.
The accepted incumbent at update 80 was length/full `0.94390/0.92188`; the
saved complete-state checkpoint is at true step `2264072192`, SHA-256
`f7ee8ca8dd48347b8c8eb1c96a7b8a02223d329db5b1038f092f3e5a8ad0be07`.
V48 therefore remains useful evidence that rollback works, but its hard
fixed-lane acceptance gate is rejected for long training: it repeatedly
discarded numerically safe update blocks and can suppress the temporary
regression needed to adapt to difficult DR cases. Preserve its code and
artifacts; do not resume it as the current optimizer path.

## Current Normal Long-Trend Continuation: V49

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v49`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v49_resume_v48.sh`.
V49 resumes only V48's accepted update-80 checkpoint and preserves its 57-D
actor, 368-D complete-DR critic and Adam moments. It starts fresh convergence
history at `rmp49_normal_long_trend_polish`; curriculum-history resume,
optimizer/critic reset, observation migration, another checkpoint, outcome
reweighting and the transaction guard must fail closed.

All 28 V49 stages are value-for-value copies of V39/V48 apart from names and
notes. Reward, gates, bounded q_ref, recovered RMP/PD, observations, physics,
resets, non-execution and execution DR ranges, and PPO stay fixed: 1024
environments x 128 steps, minibatch 16384, three epochs, LR `1e-4`, clip
`0.15`, target KL `0.010`, entropy coefficient `0.0002`, gamma `0.9995` and
GAE lambda `0.99`. Every numerically valid PPO update now becomes the next
rollout policy, restoring the ordinary continuous path. Short-term
`conv_len/full` oscillation is expected and must be judged over the 24-update
rolling window and block validation rather than by monotonic per-update or
fixed-lane acceptance. Existing behavior, view, apex, contact and arm-safety
gates remain active, so removing rollback does not remove curriculum or
safety constraints.

V49 has no update cap and continues W&B `9acnp70r` with `resume=must`.
V48 logged attempted update 107 at true step `2267611136` with offset
`728039424`, while the resumed accepted checkpoint is at `2264072192`.
V49 therefore uses W&B-only offset `731578368`, placing its first point one
rollout after the V48 tail without altering true checkpoint/optimizer time.
Run it only in tmux `pp_gpu0`; evaluate the long trend before proposing
another reward, DR or PPO change. V49 remains experimental, simulation-only
and inherits every unresolved real-camera, replacement-ball and formal
pointwise RMP/PD coverage blocker.

## Preserved Strict Long-Block Retention Diagnostic: V50

The preserved opt-in GPU0 diagnostic is
`goal_d455_measured_qvel_rmp_vertical_v50`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v50_resume_v49.sh`.
V50 resumes only V49's periodic Stage-1 update-850 checkpoint at true step
`2375483392`, SHA-256
`1e611b69373c2d1fe636578c7f88030c78f22e2b7775e603fe865581f96ea2ef`.
It preserves the 57-D actor, 368-D complete-DR critic, Adam moments and every
V49 environment, reward, gate, control, observation, reset, physics, DR and
PPO value. It starts fresh history at `rmp50_long_block_retention_polish` and
must reject curriculum-history resume, optimizer/critic reset, observation
migration, outcome reweighting and another source.

V49 was stopped after update 854 because its stochastic 24-update window
remained near `conv_len/full=0.931/0.894` after 854 updates in Stage 1. Its
training-internal best was update 432. A separate deterministic 256-lane
paired diagnostic at seed `20260910`, with observation missing and coherent
missing both forced to zero, showed that the update-850 periodic state was
stronger than the internal best in that easier domain: mean length
`1143.26/1200 = 0.95271`, full rate `0.92578` and hits `14.21094`, versus
`1127.24/1200`, `0.91406` and `14.15625` for update 432. This is evidence of
latent no-missing capability and misleading training-window selection, not a
pass of the complete Stage-1 domain. The matching V50 transaction evaluator,
with the full Stage-1 missing/coherent-observation contract active, measured
the update-850 incumbent at only `0.92587/0.89844`; V50 must improve those
unchanged primary gaps. PPO remained numerically healthy, so the defect is
policy retention/selection under the full domain rather than contact-learning
or optimizer divergence. The paired no-missing evidence is under
`outputs/rl_sim/v50_plateau_rewind_experiments_20260825/`.

V50 uses the existing atomic candidate/incumbent guard at a fixed 128-update
interval with 256 deterministic first-episode lanes and seed `20260910`. The
candidate trains continuously inside the whole block; only the block boundary
can commit or restore actor, critic and every Adam moment. This retains 16
times V48's adaptation horizon while preventing another unlimited V49-style
erosion: while the full-domain incumbent misses the unchanged `0.95/0.92`
primary gates, neither individual primary gap may increase and their combined
gap must strictly improve. Every active secondary gate must stay in range,
qvel exceedance must be zero and qacc exceedance must be at most `0.01`. Do
not shorten the block,
weaken the population or safety bands, or reinterpret transaction metrics as
graduation. The ordinary 24-update rolling gates and separate stochastic block
validation still control every curriculum advance. No stage has an update cap.

V50 used 1024 x 128 rollouts, minibatch 16384, three epochs, LR `1e-4`, clip
`0.15`, target KL `0.010`, entropy `0.0002`, gamma `0.9995` and GAE lambda
`0.99`. Its unused formal launcher pins W&B `9acnp70r`, `resume=must` and
W&B-only offset `732102656`, but V50 must not be launched formally. V50
remains experimental, simulation-only and inherits every unresolved
real-camera, replacement-ball and formal pointwise RMP/PD coverage blocker.

The single 128-update V50 bounded run is
`outputs/rl_sim/v50_long_block_retention_trial_seed20260911_u128`. Its
full-domain candidate improved length `0.92587 -> 0.93225` and combined
primary gap `0.04569 -> 0.04322`, but full rate moved by exactly one of the
256 evaluation lanes, `0.89844 -> 0.89453` (`+13/-14` full-lane changes).
The strict individual non-regression rule therefore rejected it. It had zero
secondary regressions, zero qvel exceedance and qacc exceedance `0.004199`.
The rejected checkpoint's parameter-tree and optimizer-tree SHA-256 digests
exactly match the V49 source (`25f2e143...18ebaba` and
`9d913c33...bc30ea8`); Adam `t=481901`. V50 proves complete rollback but is
rejected for formal continuation because a one-lane full-rate quantization
change can discard a candidate whose combined unchanged-gate gap improves.

## Current One-Lane-Resolution Retention Continuation: V51

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v51`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v51_resume_v49.sh`.
V51 resumes the same V49 Stage-1 update-850 checkpoint and preserves every
V50 stage configuration, reward, curriculum gate, observation, control,
physics, DR and PPO value. It starts fresh history at
`rmp51_one_lane_resolution_retention_polish`; another source, curriculum
history, optimizer/critic reset, observation migration or outcome reweighting
must fail closed.

V51 keeps the 128-update, 256-lane atomic transaction and changes only the
below-gate individual full-rate comparison: the full gap may increase by at
most `1/256 = 0.00390625`, the exact resolution of one deterministic
evaluation lane, while the length gap may not increase and the combined
length/full gap must still strictly decrease. Once the unchanged `0.95/0.92`
gates are both reached, both must remain in range; the tolerance cannot admit
a below-gate candidate. Every secondary and safety rule is unchanged. The
ordinary committed 24-update window and independent stochastic block
validation remain the only curriculum-advance contract. No stage has an
update cap.

The V50 boundary is the preregistered V51 decision case: its one-lane full
movement is within resolution, its length and combined gap improve, and its
secondary/safety audit is clean. A regression test fixes that exact decision
while legacy profiles retain zero tolerance. Formal V51 continues W&B
`9acnp70r` with `resume=must`, W&B-only offset `732102656`, only after the
user explicitly authorizes external metrics for V51. The active 2026-08-25
launch was instead kept W&B-offline after external upload approval was not
granted. It runs in tmux `pp_gpu0` at
`outputs/rl_sim/measured_qvel_rmp_vertical_v51_gpu0_seed20260911_20260825_one_lane_retention_resume_v49_last850_offline`.
Its first 128-update transaction was accepted: deterministic length/full
improved `0.92402/0.88281 -> 0.92828/0.88672`, combined gap fell
`0.06317 -> 0.05500`, full lanes changed `+15/-14`, secondary regressions
were zero, qvel exceedance was zero and qacc exceedance was `0.004187`. The
accepted update-128 checkpoint SHA-256 is
`c4b7eda6dce751954f9ea55cd2771309bde92df2b39f0114513bb07171f60e5d`;
Adam `t=484973`. Training continued beyond update 130. This is promotion
evidence, not Stage-1 graduation; the unchanged rolling and block gates still
apply. V51 remains experimental, simulation-only and inherits every
unresolved real-camera, replacement-ball and formal pointwise RMP/PD coverage
blocker.

## Rejected Hard-Lane Gradient-Coverage Trial: V52

The opt-in bounded diagnostic is
`goal_d455_measured_qvel_rmp_vertical_v52`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v52_resume_v51.sh`.
It resumes the V51 Stage-1 `mjx_curriculum_best.pkl` at update 256, true step
`2409037824`, SHA-256
`c6b150781a37648277e7994c2a074f75c40005f8aaaeffcc27ae4eff972853fb`.
Actor, 368-D critic and Adam moments are preserved; curriculum history is
fresh. Another source, reset/migration, or curriculum-history resume must fail
closed.

V52 preserves every V51 reward, gate, environment, observation, control,
physics, DR and PPO value. It changes only actor-gradient allocation. A
transition is a hard-lane sample when at least two of these preregistered
conditions are active: ball solref time `>=0.00525 s`, reset spin-x
`<=-12.5 rad/s`, aggregate reset disturbance `>=3.20`, normalized ball
inertia `>=0.635`, target-y `>=0.035 m`, reset ball-vxy `>=0.0029 m/s`, or
ball mass `>=0.00375 kg`. Hard samples receive advantage weight `1.5`;
ordinary samples retain weight `1.0`. The critic still uses the unchanged full
batch. Outcome/failure/success weighting is prohibited.

The thresholds come from the frozen 1024-lane V51 update-256 audit, not a
changed simulator distribution. On that audit the `>=2` group occupied about
53.4% of lanes and achieved deterministic length/full `0.90093/0.84461`,
while ordinary lanes achieved `0.97115/0.95388`. The stochastic split was
`0.89900/0.85192` versus `0.96978/0.94340`. Thus V52 targets the deficient
group without removing ordinary samples or changing any curriculum threshold.

The offline 64-update trial was rejected. On the same 1024-lane deterministic
screen, hard length/full improved only `+0.00242/+0.01250`, below the
preregistered `+0.01/+0.02`; ordinary length regressed `-0.00530`, just beyond
its `-0.005` protection limit. The full stochastic screen regressed length/full
`-0.00981/-0.01660`. Optimization and safety remained healthy, so this is an
objective result rather than a PPO failure. Do not resume V52 or start an
uncapped V52 run. The complete decision is recorded under
`outputs/rl_sim/v52_multiseed_generalization_screen_20260825/`.

## Rejected Reward-Only Apex/Survival Balance Trial: V53

The opt-in bounded successor is
`goal_d455_measured_qvel_rmp_vertical_v53`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v53_resume_v51.sh`.
It restarts from the same frozen V51 update-256 best checkpoint and must not
inherit the rejected V52 actor, critic or optimizer. V53 changes exactly two
reward scalars in all copied stages: `low_hit_penalty_weight 7000 -> 5000`
and `post_hit_survival_reward_weight 3.5 -> 4.5`. The symmetric 1.30 m apex
target and its weight remain unchanged because the 3361-update analysis found
that term positively aligned with future full episodes; only the one-sided
1.28 m lower barrier was anti-aligned. Every curriculum gate, environment,
DR, observation, control, PPO and entropy value remains V51-exact. Hard-lane
or outcome advantage weighting is prohibited.

The offline 64-update V53 trial was rejected. Deterministic length/full
regressed `-0.00319/-0.00293`; stochastic length/full regressed
`-0.00248/-0.00098`. Mean absolute apex remained in-band at `1.29718 m`, and
PPO/safety remained healthy, but deterministic `ball_too_low` terminations
increased from 45 to 52. Thus weakening the exact lower-apex barrier was the
wrong causal intervention; the earlier history correlation was confounded by
early termination. Do not resume V53 or launch it uncapped. A future reward
ablation must restore `low_hit_penalty_weight=7000` and isolate survival
credit. The full decision is under
`outputs/rl_sim/v52_multiseed_generalization_screen_20260825/`.

## Preserved Old-Course Reentry Continuation: V54

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v54`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v54_resume_v36_stage25.sh`.
It resumes only the passed V36 Stage-25 archive
`25_rmp36_rmp_internal_commit_12p5_polish.pkl`, true step `2181431296`,
SHA-256
`310fc93f8c69e0f4fcefe59a161948e33cb1b23e57f880d75050dd56713f6a1f`.
The 57-D actor, 368-D complete-DR critic, Adam moments and true optimizer time
are preserved. V37--V53 checkpoints, curriculum-history resume, optimizer or
critic reset, observation migration, transactional promotion and advantage
reweighting must fail closed.

The source is not a filename-based choice. A fixed 256-lane deterministic
screen in the exact V37 Stage-1 environment, seed `20260914`, measured the
selected V36 Stage-25 pass at hits/full/length/view
`14.7227/0.94531/1147.23/0.96478`. The later V36 Stage-26 best originally
used by V37 achieved only `14.3125/0.91797/1127.57/0.96591`; V37 best, V39
best, V49 update 850 and V51 best also ranked below the Stage-25 source on the
primary full/length platform. Preserve the structured and per-lane evidence
under `outputs/rl_sim/v54_source_selection_20260825/`.

V54 has 29 stages. Stage 1,
`rmp54_v36_stage25_reentry_polish`, copies the complete V36 Stage-25
environment, reward, observation, control, physics and DR values exactly. It
requires at least 160 updates plus a 24-update `conv_len/full >= 0.95/0.92`
hold and stochastic block validation. It has no update cap and may not force
advancement. Only after that pass do Stages 2--29 traverse all V37 Stages
1--28 value-for-value. Do not collapse, skip or reinterpret either the old-
course bridge or any V37 stage.

V54 restores V37's ordinary PPO contract: 1024 x 128, minibatch 16384, three
epochs, LR `1e-4`, clip `0.15`, target KL `0.008`, entropy `0.0002`, gamma
`0.9995` and GAE lambda `0.99`. It inherits none of the rejected V38--V53
reward, KL, transaction or actor-weighting interventions. The active launcher
uses a new local W&B offline identity and must not modify `9acnp70r` without a
separate explicit authorization. V54 remains experimental, simulation-only
and non-deployable, and inherits every unresolved real-camera,
replacement-ball and formal pointwise RMP/PD coverage blocker.

## Current Corrected Execution-DR Continuation: V55

The current opt-in GPU0 continuation is
`goal_d455_measured_qvel_rmp_vertical_v55`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v55_resume_v54_u100.sh`.
V55 resumes only V54 Stage-1 update 100 at true step `2194538496`, SHA-256
`5f02f9ed65afb87adfcd4b9ba6b54ded4dac1a7cf3ae9498165ac51314e4acde`.
It restores the 57-D actor, 368-D critic and Adam state while starting fresh
curriculum history. Another source, history resume, optimizer/critic reset,
observation migration, transaction promotion and advantage reweighting must
fail closed.

The V54 entry was a course-order error: V36 Stage 25 already applied the full
12.5% RMP-internal interval. Fixed seed-20260916, 256-lane screening of the
selected V54 update-100 actor measured full/length/hits
`0.879/1100.6/13.984` in the Stage-25 domain, `0.875/1098.0/13.984` at the
6.25% RMP domain, `0.926/1134.5/14.543` at the V36 Stage-21 small execution
domain and `0.914/1122.9/14.496` in exact V55 Stage 1 with execution DR
disabled. Removing
observation missing alone did not recover survival. Treat execution-DR order
as the first causal fix; do not change rewards unless nominal/small-DR
training still fails. Preserve the evidence under
`outputs/rl_sim/v55_primary_reward_repair_20260825/`.

V55 has exactly 28 stages. Stage 1 keeps the V37 Stage-1 task, ball, contact,
observation, reward and gates but disables RMP/PD/plant randomization and pins
all execution parameters nominal. Stages 2--19 are the original V37 non-
execution lessons with only the V36 Stage-21 small execution envelope. Stage
20 remains the exact `conv_len=1/full=1` protected polish. Only Stages 21--28
may introduce the existing 6.25%, 12.5%, PD and plant widening ladder. Stages
2--28 otherwise remain V37-exact. The 14-event positive-hit cap and 13-hit
gate are unchanged; the goal is 13--15 contacts over all 1200 steps.

V55 preserves V37 PPO exactly: 1024 x 128, minibatch 16384, three epochs, LR
`1e-4`, clip `0.15`, target KL `0.008`, entropy `0.0002`, gamma `0.9995` and
GAE lambda `0.99`. The formal launcher must create a new W&B online run with
`resume=never`; never append V55 to `9acnp70r` or the V54 offline identity.
V55 remains experimental, simulation-only and non-deployable and inherits all
unresolved real-camera, replacement-ball and formal RMP/PD evidence blockers.

## Rejected Joint Absolute-Apex/Flight-Lift Reward Trial: V56

The preserved opt-in reward experiment is
`goal_d455_measured_qvel_rmp_vertical_v56`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v56_resume_v55_u241.sh`.
It resumes only the safely stopped V55 Stage-1 update 241 checkpoint, true
step `2226126848`, SHA-256
`ed02beca220da604e9e13da6fbd0905d3de8f8305d0ae4003c200fd76fe13ca7`.
Actor, 368-D critic and Adam moments are restored, but reward semantics and
convergence history start fresh. Another source/profile/stage/step,
curriculum-history resume, optimizer/critic reset, observation migration,
transaction mode and advantage reweighting must fail closed.

V56 retains V55's complete 28-stage nominal-to-small-to-wide execution-DR
order, every control/observation/reset/physics/DR value, and the 14-event
positive hit cap. It jointly targets a 1.32 m absolute world-z apex and a
0.195 m physical contact-to-apex lift. These imply a 1.125 m ball contact
height, close to the stopped V55 source, so absolute apex may not be reached
by raising the whole juggling position. Contact-edge ball/racket soft limits
are 1.18/1.16 m with active penalties. The counted cadence target is 0.390 s;
no maximum-hit termination or hard maximum-hit gate is introduced.

The 32-update bounded trial was rejected. A paired seed-`20260920`, 256-lane
screen measured V56 update 32 versus V55 at absolute apex/lift/contact-ball-z
deltas `+1.90/+2.99/-1.09 mm`, while full improved `0.91797 -> 0.92969` and
length changed `1128.3 -> 1133.4`. This proves the joint reward moved safely
and did not raise the whole contact location, but final absolute apex/lift
were only `1.2771/0.1561 m`, below the `1.29/0.175 m` acceptance thresholds.
Do not launch V56 formally or resume its trial actor. V56 remains experimental,
simulation-only and non-deployable and inherits all unresolved real-camera,
replacement-ball and formal RMP/PD coverage blockers.

## Preserved Height-Escape PPO Trial: V57

The preserved bounded GPU0 optimization experiment is
`goal_d455_measured_qvel_rmp_vertical_v57`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v57_resume_v55_u241.sh`.
V57 restarts from the exact safely stopped V55 Stage-1 update 241 source and
must not inherit the rejected V56 actor. It copies all 28 V56 stages and every
environment, reward, gate, curriculum and DR value exactly. Stage 1 remains
nominal execution, Stage 20 remains complete non-execution polish, and
execution widening still starts only at Stage 21.

V57 changes only the PPO optimization contract: LR `1.5e-4`, clip `0.20`,
target KL `0.012`, entropy `0.001`, and minimum log standard deviation `-4.0`;
rollout length 128, three epochs, gamma `0.9995` and GAE lambda `0.99` remain
unchanged. This tests the observed exploration floor: three V55/V56 policy
dimensions were pinned near `log_std=-4.8`, while V56's correctly directed
height improvement stalled after update 20. The default bounded trial is 64
updates, seed `20260921`, W&B offline and a unique directory. It must reach
both absolute apex/lift `>=1.29/0.175 m` without raising contact heights,
while protecting 13--15 hits, full/length, view, motion, qvel/qacc, KL and
failure distribution.

The 64-update V57 trial did not reach the preregistered `1.29/0.175 m`
height thresholds, so do not launch it formally. Its last policy is still a
useful platform: paired seed-`20260922`, 128-lane frozen evaluation measured
hits/full/length `14.969/0.97656/1177.4`, absolute apex/lift/contact ball z
`1.28375/0.16212/1.12163 m`. Versus V55, absolute apex and lift improved
`+8.30/+7.60 mm` while contact changed only `+0.69 mm`. V57 therefore proved
that exploration unlocked a safer, longer and moderately higher policy, but
event-only lift credit remained too slow. Preserve its complete update-60
archive for V58; do not resume the metadata-poor update-64 last file.

## Rejected Short-Horizon Joint-Height Credit Trial: V58

The current bounded GPU0 reward-timing experiment is
`goal_d455_measured_qvel_rmp_vertical_v58`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v58_resume_v57_u60.sh`.
It resumes only the complete V57 update-60 archive at true step `2233991168`,
SHA-256
`0880235d472e8c599f2711707f148409a99251600f8bd15fc9a3ed97d932c8a2`.
Another source/profile/stage/step/hash, curriculum-history resume,
optimizer/critic reset or observation migration must fail closed.

V58 adds exactly one reward component. For the first `0.080 s` after each of
the first 14 counted hits, while the ball is ascending, it supplies dense
weight-40 credit equal to the product of the 1.32 m absolute-apex score and
the 0.195 m contact-to-apex-lift score. The product prevents either coordinate
from substituting for the other; hits 15+ receive zero such credit. Every V57
event/dense reward, contact-height barrier, gate, control, observation,
physics, DR, curriculum order and PPO value is unchanged. The default trial
is 48 updates, seed `20260923`, W&B offline and a unique directory. It must
reach absolute apex/lift `>=1.29/0.175 m` while retaining safe contact height,
13--15 hits, full/length, view, motion, qvel/qacc, KL and failure distribution.
The 48-update trial was rejected. In paired 128-lane seed-`20260924` frozen
evaluation, the V57-u60 source achieved hits/full/length
`15.289/0.97656/1177.7`, absolute apex/lift/contact ball z
`1.28233/0.15818/1.12415 m`. V58-u48 changed those to
`14.508/0.92188/1131.0` and `1.29462/0.16559/1.12903 m`: absolute apex rose,
but lift remained below 0.175 m, contact rose by 4.88 mm, and full survival
regressed by 5.47 percentage points. Do not launch V58 formally or resume its
actor. V58 remains experimental, simulation-only and non-deployable.

## Rejected Vertical-Strike-Antagonist Ablation: V59

The current bounded GPU0 experiment is
`goal_d455_measured_qvel_rmp_vertical_v59`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v59_resume_v57_u60.sh`.
It restarts from the same complete V57-u60 archive and must not inherit the
rejected V58 actor. V59 copies all 28 V58 stages and changes one field only:
`racket_up_drift_penalty_weight` becomes zero instead of 6.0. The V58 joint
absolute-apex/lift dense credit, contact-height barriers, survival rewards,
PPO, control, observations, physics, curriculum and nominal-small-then-wide
DR order are unchanged.

The 48-update ablation was rejected. Paired 128-lane seed-`20260926` frozen
evaluation changed V57-u60 to V59-u48 by absolute apex/lift/contact ball z
`+8.84/+6.77/+2.07 mm`, full `+0.78` percentage points and length `+10.8`
steps. The direction was safe and absolute apex reached 1.2920 m, but lift
remained only 0.1658 m. Thus the up-drift term is locally antagonistic but is
not the root cause. Do not launch V59 formally or resume its actor.

## Rejected Completion-Conditioned Physical Objective: V60

The current bounded GPU0 experiment is
`goal_d455_measured_qvel_rmp_vertical_v60`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v60_resume_v57_u60.sh`.
It restarts from the same exact V57-u60 actor, critic and Adam state and must
not inherit V58/V59. V60 copies all 28 V57 stages and adds one terminal term:
only at the true 1200-step truncation, and only with 13--15 counted hits, pay
weight 120 times the product of the first 14 rewardable contacts' mean
absolute-apex score and mean physical contact-to-apex-lift score. The targets
and sigmas remain `1.32/0.040 m` and `0.195/0.025 m`.

This directly represents the requested behavior. Incomplete episodes, 16+
contacts, low-height full episodes, and policies that raise apex and contact
together cannot maximize it. Every V57 dense/event reward, the original
weight-6 up-drift term, contact-height barriers, gates, PPO, control,
observations, physics and nominal-small-then-wide DR order are unchanged.
The 48-update trial was rejected. Paired seed-`20260928` frozen evaluation
held full at `0.92188` and improved length by 9.5 steps, but V60-u48 versus
V57-u60 changed abs/lift/contact by `+10.40/+4.34/+6.06 mm`, ending at
`1.29399/0.16639/1.12760 m`. The new event raised return and protected
longevity, but most absolute-height gain came from contact height; it did not
provide sufficient action-specific lift credit across the long GAE trace.
Do not launch V60 formally or resume its actor.

## Rejected Direct Reflected-Velocity Continuation: V61

The preserved bounded GPU0 experiment is
`goal_d455_measured_qvel_rmp_vertical_v61`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v61_resume_v57_u60.sh`.
It restarts from the exact V57-u60 source, not V58--V60. V61 retains V60's
completion-conditioned objective and enables the existing paper-derived
pseudo-Huber `hit_adaptive_reflected_velocity_penalty` at weight 5. Its
vertical sigma is 0.20 m/s. The XY sigma is deliberately 100 m/s, making the
trial a vertical-only causal lesson rather than a lateral retuning.

V61 was stopped at update 17 before a frozen acceptance screen. Formula audit
found that the target uses delayed confirmation `bpos`, while the physical
lift contract uses cached contact-edge height. Its early trajectory did not
show the required height/longevity direction. Do not launch V61 formally,
resume its actor or infer a causal result from the incomplete run.

## Current Existing-Reward Rebalance Experiment: V62

The current bounded GPU0 candidate is
`goal_d455_measured_qvel_rmp_vertical_v62`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v62_resume_v57_u60.sh`.
It resumes only the complete V57 update-60 archive and returns to V57's
factorized reward; V58's dense joint term, V60's terminal joint term and V61's
reflected-velocity term must all remain zero.

The complete V55 formula inventory covers all 122 components (35 active, 87
inactive). V55's best-height window later regressed by about 10.0 mm absolute
apex and 4.4 mm physical lift without increasing hits or shortening cadence;
return also fell. V57 frozen full episodes split into a desired 14-hit mode
at cadence/apex/lift `0.3833 s/1.2928 m/0.1754 m` and a dominant low/fast 15+
mode at `0.3564 s/1.2789 m/0.1577 m`. Existing V56 rewards already rank the
desired mode about 27.3 return higher, so no new reward function is justified.

V62 changes only seven existing fields in every one of V57's 28 stages:
up-drift weight `6 -> 0`, center-flat hit `2.4 -> 1.2`, hit-count floor
`1.5 -> 0.5`, base hit `1.0 -> 0.75`, and true-counted cadence
target/sigma/weight `0.390/0.045/0.5 -> 0.390/0.035/2.0`. Absolute-apex,
physical-lift, contact-height, survival, view, safety, control, observation,
physics and DR ordering remain unchanged. Stage 1 has no execution DR, Stage
20 remains complete polish, and Stage 21 begins wide execution-DR lessons.

V62 must first pass a bounded offline run and paired deterministic/stochastic
screens for full/length, 13--15 hits, absolute apex, physical lift, contact z,
view and qvel/qacc. Only a passing checkpoint may start a no-cap `pp_gpu0`
run with a new W&B online ID and `resume=never`.

## Rejected V31 Exact Reentry V63 and Corrected V29 Reentry V64

V63 is a completed negative diagnostic. It resumed V31 Stage 23 with zero
configuration or PPO gap for 32 bounded updates. The best periodic checkpoint
was update 20, but paired first-episode screens at seeds `20260932/20260933`
measured full/length/hits `0.875/1109.6/13.883`; no checkpoint passed the
preregistered `0.93/1135` mean acceptance. Do not resume V63 or select its
last checkpoint. The run and all paired evidence are under
`outputs/rl_sim/measured_qvel_rmp_vertical_v63_gpu0_seed20260934_20260825_v31_stage23_zero_gap_u32/`
and `outputs/rl_sim/v63_robust_source_selection_20260825/`.

V63 also exposed a source-selection error: the first historical ranking used
V62's nominal-execution environment, while V31 Stage 23 actually applies
6.25% internal-RMP DR. Repeating the paired 128-lane screens in the exact V31
Stage-23 environment selected V29 Stage 21, not V31. V29's two-seed
worst/mean full were `0.875/0.891`, worst/mean length `1103.6/1121.1`, mean
hits `14.039`, full-episode 13--15 fraction `0.583`, view `0.965`, absolute
apex `1.2932 m` and lift `0.1668 m`. V31 measured only `0.8438/0.8633` full
and `1078.9/1094.2` length in that domain. V16/V17 retained low/fast modes,
with 16+ contacts in `72.2%/74.8%` of their full episodes and lift near
`0.137 m`; they are not acceptable fallbacks.

The preserved corrected diagnostic is
`goal_d455_measured_qvel_rmp_vertical_v64`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v64_resume_v29_stage21.sh`.
It resumes only V29 Stage 21 at true step `2087190528`, SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`,
with the 57-D actor, 221-D critic and complete Adam state. Another source,
curriculum history, reset or observation migration must fail closed.

V64 has one stage, `rmp64_v29_stage21_exact_reentry`, copied value-for-value
from V29 Stage 21 except for name and notes. PPO is also source-exact: rollout
128, three epochs, learning rate `1e-4`, clip `0.15`, target KL `0.008`,
entropy `0.0002` and minimum log standard deviation `-4.8`. The next V31
Stage-23 environment differs by only seven internal-RMP range fields, but V64
must not introduce that widening: its purpose is to prove source retention
before a separately versioned bridge. Reward changes, critic growth,
transactional guards and advantage reweighting are prohibited.

The completed V64 32-update offline run is rejected. Paired exact-source
screens selected the original V29 checkpoint: its mean/worst full and length
were `0.8945/0.875` and `1117.5/1109.8`; best V64 update 30 regressed them to
`0.8789/0.875` and `1112.3/1105.1`. No candidate passed `0.93/0.90` and
`1135/1110`. Do not resume V64 or use its last checkpoint.

V29 full/failed episodes score about `110.66/-1.24`, while PPO remains
numerically healthy. The failure is therefore a mean-objective long-tail
credit problem, not missing scalar reward magnitude or a PPO numerical fault.
Before any PPO parameter change, the next implementation must be an opt-in
constrained-MDP objective with separate long-horizon failure/length cost
critics and adaptive dual variables. It must begin in the exact V29 Stage-21
domain, preserve actor/control/reward/PPO values, and isolate the seven-field
RMP bridge until the survival constraint passes. The preregistered analysis is
`outputs/rl_sim/v63_robust_source_selection_20260825/V63_V64_SOURCE_SELECTION_AND_CONSTRAINED_OBJECTIVE_DECISION.md`.

The opt-in constrained successor is
`goal_d455_measured_qvel_rmp_vertical_v65`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v65_cmdp_resume_v29_stage21.sh`.
Its sole stage, `rmp65_v29_stage21_cmdp_survival`, is a value-for-value copy of
V29 Stage 21; actor observations remain 57-D, the reward critic remains 221-D,
and reward, gates, control, physics, DR and PPO values are unchanged. V65
appends a six-output cost critic without resetting the V29 actor, reward
critic, Adam moments or optimizer time. Its persistent projected-dual costs
are any true failure `<=0.07`, normalized expected length shortfall
`<=65/1200`, and each of ball-low, ball-high, racket-low and racket-high
terminal probability `<=0.02`. Cost GAE uses gamma `1`, lambda `0.99`; cost
value coefficient is `0.5`; dual learning rate/max are `0.1/2.0`. The actor
uses normalized reward advantage minus the stop-gradient dual-weighted,
independently normalized cost advantages. This is an optimizer constraint,
not another environment reward.

V65 may resume only the pinned V29 Stage-21 source or a complete V65
checkpoint containing the cost head, duals and their Adam leaves. The CMDP
path is default-off and allowlisted only for V65. Transactional guards,
legacy failure/success/hard-lane advantage reweighting, observation migration,
reward changes and the later seven-field RMP widening are prohibited. First
run a bounded 32-update GPU0 trial and paired deterministic first-episode
screens at two frozen seeds. Formal training on `pp_gpu0` is permitted only if
one checkpoint passes mean/worst full `0.93/0.90`, mean/worst length
`1135/1110`, mean hits 13--15, full-episode 13--15 fraction `>=0.55`, absolute
apex `>=1.29 m`, lift `>=0.165 m`, view `>=0.96`, zero qvel exceedance and
qacc exceedance `<=0.01`. A failed bounded trial must be preserved and must
not start formal training.

The V65 bounded trial is complete and rejected. In exact V29 Stage-21
conditions (15% camera/view missing, coherent probability zero), paired seeds
`20260932/20260933` selected update 10 as the least-regressed CMDP candidate,
but its mean/worst full were only `0.8945/0.8906` and mean/worst length
`1115.0/1113.6`; mean full did not improve over the source. It changed the
256-episode ball-low/racket-high counts from `8/12` to `11/9`, demonstrating
failure-mode exchange rather than constraint satisfaction. Updates 20--32
did not pass and update 30 regressed full to `0.8750`. Cost-critic explained
variance remained near zero after warmup even while failure/shortfall duals
rose to `0.448/0.370`; the 128-step rollout still did not propagate useful
hundreds-of-steps risk. Do not resume V65 or launch `pp_gpu0`. The report is
`outputs/rl_sim/v65_cmdp_20260825/V65_CMDP_BOUNDED_VALIDATION_REPORT.md`.
A successor must pretrain and held-out-calibrate cost values from complete
frozen episodes before any actor update; do not respond by increasing scalar
termination rewards, dual gain, PPO updates, or widening RMP DR.

The subsequent frozen V29 Stage-21 bimodal-failure audit supersedes cost-value
pretraining as the immediate next experiment. CMDP remains default-off and
V65 must not resume. In 256 paired lanes, full DR produced `0.88672` full and
`1108.91` mean length; all episode-constant DR fixed at nominal produced
`1.0/1200`, while a centered reset with full DR produced only
`0.90625/1136.20`. Ball/contact, static observation/racket-mount and RMP/PD
nominal ablations had paired net full-lane changes `+15/+13/+11`; removing
temporal sensing/missing/target-hold randomness was `+16`. Treat this as a
joint valid-DR robustness-margin problem, not proof of invalid or impossible
samples, and do not narrow reset or DR support from this evidence.

Failure lane identity is itself unstable: across four exact same-graph runs,
205/256 lanes were always full, 14 always failed and 37 flipped. Future
screens must repeat each episode-constant DR/reset tuple across independent
temporal/action draws rather than using a single deterministic label. A new
bounded successor must stratify DR families, allocate fixed episode-level
mass to empirically hard replicated tuples, and isolate a proximal recovery
signal before any formal training. Code audit shows that `pre_hit_intercept_*`
is active only while `hit_count <= 0`, but 27/29 audited failures occur after
the first hit. The measured first candidate must therefore use the existing
post-first-hit `descending_intercept_excess_penalty` with a `0.025 m`
zero-loss radius; do not combine it with DR, PPO, completion-reward or
failed-contact changes in one trial. The full report is
`outputs/rl_sim/v66_bimodal_failure_analysis_20260825/V29_BIMODAL_FAILURE_CAUSAL_REPORT.md`.

The opt-in replicated-tail successor is
`goal_d455_measured_qvel_rmp_vertical_v66`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v66_resume_v29_stage21.sh`.
It resumes the pinned V29 Stage-21 checkpoint at true step `2087190528`,
SHA-256 `8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`,
with the 57-D actor, 221-D critic and Adam state intact. CMDP, completion
bonuses, outcome-advantage weighting and failed-contact rewards are disabled.
Stage 1 changes only the post-first-hit descending-intercept excess component:
weight `2.4`, zero-loss radius `0.025 m`, sigma `0.050 m`, and the existing
`0.55 s` horizon. At the failed median error of about `0.049 m`, this costs
about 11% of one post-first-hit alive tick; at `0.10 m` it is comparable to
that tick, while the measured successful contact population remains mostly
inside the zero-loss radius.

V66 Stage 1 has no hard-key replay and must run at least 64 updates. Stage 2
keeps its environment exactly and assigns 10% of eligible episode resets to
the six reset keys derived from tuples with <=4/18 replicated successes. Each
actual eligible episode reset independently receives the fixed 10% mixture
probability, so early termination cannot change the per-reset probability. It
does not change a DR interval or assert physical
unreachability. Stage conv_len gates are `0.90` and `0.92`; never lower either
below `0.90`. Existing V29 ranges and historical independent-per-joint
sampling are preserved for causal isolation. The boundary audit may identify
an unmeasured, overly broad candidate family, but any range/correlation change
requires a later stage/profile and separate test.

Before formal launch, preserve a bounded 32-update run and repeat frozen
two-seed screens across independent process/action draws. A candidate must
reach aggregate full `>=0.90`, mean length `>=1125`, worst-seed full `>=0.87`,
mean hits 13--15, full-episode 13--15 fraction `>=0.55`, absolute apex
`>=1.28 m`, lift `>=0.145 m`, view `>=0.96`, zero qvel exceedance and qacc
exceedance `<=0.01`, while materially improving the six-tuple hard group.
Formal uses a new output/W&B identity, no update cap, and tmux `pp_gpu0`.

V66 is now a bounded negative result and must not be launched formally. Its
least-regressed update 30 passed deterministic two-seed full
(`0.90234`, worst `0.89844`) but missed mean length by `0.18` step. Across the
registered ten execution conditions it fell to full `0.89414`, length
`1113.55`, with only `8/60` hard-six successes versus source `12/108`.
Healthy KL, critic and safety telemetry do not override those frozen-policy
failures. Preserve the checkpoint and report; do not resume V66 or increase
its reward weight.

The opt-in successor is `goal_d455_measured_qvel_rmp_vertical_v67`, launched
only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v67_resume_v29_stage21.sh`.
It must restart the pinned V29 Stage-21 checkpoint at true step `2087190528`
and SHA-256
`8fb721d1a620f7c229676d6381df2a3e86495c9f589f3dd4e64e9f926cd8ed41`;
V66 checkpoints are invalid sources. V67 restores the original V29 reward and
Stage 1 changes only ball-spin support from `+/-25,+/-25,+/-20` to
`+/-12.5,+/-12.5,+/-10 rad/s`. Solref, every other DR field, resets,
observations, control and PPO remain V29-exact. This half-spin support was the
first preregistered fallback because the frozen four-repeat screen reached
full `0.92480`, length `1146.58`, hard-six `16/24`; half solref is not combined
with it.

V67 Stage 1 uses `conv_len/full=0.92/0.88`, minimum 64 updates and a 12-update
hold. Stage 2 preserves its environment and applies the six-key 10% eligible
episode-reset mixture, with `conv_len/full=0.92/0.90`, minimum 96 updates and
a 16-update hold. CMDP and V66's intercept reward remain off. The smaller spin
interval is an unmeasured candidate, never hardware coverage; new-ball
spin/contact evidence remains required. Formal training restarts V29, uses a
new run/W&B identity, no update cap and tmux `pp_gpu0`.

The 2026-08-26 V67 bounded 32-update trial remained PPO/safety stable and
preserved the half-spin contract. The formal offline-evidence run completed
normally at
`outputs/rl_sim/measured_qvel_rmp_vertical_v67_gpu0_seed20260939_20260826_V29_Stage21_half_spin_replicated_tail_formal1`.
It completed 75 Stage-1 updates and graduated Stage 2 at update 314 after its
minimum-96/16-update strict hold. The final true step is `2138177536`; the
final checkpoint SHA-256 is
`1da7940585d1b22d8b7bbe4974b1e700bce645cc1a7983b51b68192e3431c2e4`.
Its final 24-update window was hits `13.8536`, length fraction `0.93270`, full
`0.90104`, absolute apex/lift `1.29067/0.16719 m`. This is an accepted
continuation source, not a released or hardware-covering model.

The current opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v68`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v68_resume_v67_final.sh`.
Its initial entry must resume the exact completed V67 final checkpoint above
with its 57-D actor, 221-D critic and Adam time `441976`; another entry source,
optimizer/critic reset, curriculum-history resume or from-scratch launch must
fail closed.
Append-only critic migration preserves the actor and the first 221 critic and
Adam rows exactly, initializes only the new privileged rows to zero, and grows
the critic input to 368-D without resetting Adam time.

V68 is the complete 28-stage V37 non-execution-first course, not V67's
two-stage isolated diagnostic. It starts at Stage 1 with fresh convergence
history because V67 never trained the intervening observation, target-support
and physical-correlation lessons. Relative to V37, V68 changes ball-spin
support to the accepted V67 candidate `+/-12.5,+/-12.5,+/-10 rad/s`; the V67
10% six-key reset oversampling is removed and CMDP remains disabled. It also
moderately lowers only two statistically excessive survival gates: Stage 1
uses `conv_len/full=0.93/0.92`, and Stage 20 uses `0.95/0.90` instead of
requiring zero failures across a 24-update stochastic-DR window. Every V68
`conv_len` gate remains at least `0.90`; hit, apex, view, safety and all other
gates are unchanged. Stage 20 remains the complete non-execution polish
`rmp68_complete_nonexecution_full_episode_polish`. Only after it graduates may
Stages 21--28 widen RMP
internal and PD/plant execution DR from the small envelope through the final
12.5% ranges. Do not skip directly to Stage 21 or describe Stage 2 of V67 as
the new course's Stage 20.

The formal V68 launcher uses GPU0, 1024 environments x 128 steps, minibatch
16384, three epochs, the V67/V37 PPO scalars, block validation, no stage update
cap and a new offline W&B/output identity. The first authorized run was at
`outputs/rl_sim/measured_qvel_rmp_vertical_v68_gpu0_seed20260940_20260826_V67_final_complete_28stage_resume1`.
It reported CUDA device 0 and the required 57/221 -> 57/368 append-only
migration. At Stage-1 update 13 its eligible rolling window was hits `14.578`,
length fraction `0.9751` and full `0.9525`; update KL was `0.00567`, qvel
exceedance was zero and qacc exceedance was `0.00447`. It later graduated
Stages 1--3. At the Stage-4 entry, true step `2188640256`, global update 385,
the host-memory guard safely stopped it for `sustained_low_available` after
GPU1 had moved to a concurrent 1024-environment V130 run. This was not a PPO,
NaN, GPU-temperature or checkpoint failure. The interrupted/last checkpoint
SHA-256 is
`c4a591d6ad09b583f5c8e40530a09672f1afef57b88bf180ca495766718573e9`;
the causal progress CSV SHA-256 is
`d9d6c80cbae35388e860fd3320ff0deadd7901d83db5aced31558e07784a7a76`.

Resume this safe stop only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v68_resume_interrupted_online.sh`.
It preserves the Stage-4 entry, global update 385, 57-D actor, 368-D critic,
Adam time `449228`, curriculum history and every training setting; observation
migration and all resets are prohibited. A stage-entry checkpoint with
`stage_update=0` legitimately has no same-stage CSV rows and restores an empty
Stage-4 window while retaining its causal global update. The recovery creates
a new output directory, keeps 1024 environments and resumes the already
created online W&B run `fushi37/pingpong-mjx/v68g0a1` with `resume=must` and
zero true/W&B step offset. Do not overwrite the offline source directory.
The recovery is active in `pp_gpu0` at
`outputs/rl_sim/measured_qvel_rmp_vertical_v68_gpu0_seed20260940_20260826_stage4_interrupted_resume2_online`.
At Stage-4 update 17 its rolling hits/conv_len/full were
`14.4666/0.96280/0.94307`, KL was `0.00599`, qvel exceedance was zero and qacc
exceedance was `0.00418`. W&B reported the run `running` and had uploaded
through step `2190606336`, within two updates of the local process. Host
available memory was about 7.0 GiB under simultaneous GPU0/GPU1 load; retain
the memory guard and do not stop either run merely to improve headroom.
V68 remains experimental, simulation-only and subject to the unresolved
camera, new-ball outcome and formal RMP/PD coverage blockers.

## Current Strict Stage-7 Restart: V69

The current opt-in GPU0 successor is
`goal_d455_measured_qvel_rmp_vertical_v69`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v69_restart_stage7_strict.sh`.
V68 remains immutable. V69 must resume the V68 Stage-6 completion archive
`06_rmp68_nonexecution_launch14_racket_geometry_wide.pkl` at true step
`2250244096`, Adam time `460120`, SHA-256
`b82411bb6e19546fa2396c21112b90c2b2d3e3241b8c68a81719620f6bbf4513`.
This is the boundary before any Stage-7 training. It preserves the exact
57-D actor, 368-D critic and Adam moments but deliberately starts fresh V69
Stage-7/convergence history. Another source, curriculum-history resume,
optimizer/critic reset, observation migration, from-scratch launch or CMDP
state must fail closed.

V68 Stage 7 used the compensatory `balanced_probe` readiness gate and advanced
after a 12-update hold even though its final rolling hits/conv_len/full were
only `12.758/0.87214/0.77906`, below the registered
`13.0/0.9025/0.82` primary targets. V68 Stage 8 then degraded to
`10.476/0.74008/0.55959` at update 305 before the host-memory guard safely
stopped the run. Treat this as a gate-semantics defect: neither checkpoint is
an eligible continuation source.

V69 copies all 28 V68 environments, rewards, resets, DR intervals, numerical
targets, minimum updates, hold lengths and next-domain block probes exactly.
It changes only profile/stage identity and the current-domain `gate_mode` of
Stages 7--10 from `balanced_probe` to `strict`. Thus every registered hit,
`conv_len`, full/truncation, hit1/hit3/hit12, view, contact-quality, cadence,
recoverability and safety condition must pass independently for the complete
rolling hold; a strong secondary metric cannot compensate for a failed
primary condition. Keep `advance_gate_mode=collapse`: it is a separate
stochastic anti-collapse probe on the next, intentionally harder curriculum
domain, not the current-domain graduation contract. Every `conv_len` target
remains at least `0.90`.

The formal V69 launcher uses GPU0, 1024 environments, the exact V68 PPO
settings, block validation, no update cap, W&B online and a new W&B/output
identity. Do not append it to the V68 W&B run because gate/profile semantics
changed. V69 remains experimental, simulation-only and inherits all unresolved
camera, replacement-ball and formal RMP/PD coverage blockers.

V69 was safely stopped by user request at Stage-7 update 1257, true step
`2415001600`. Its final rolling hits/conv_len/full were
`12.8388/0.89022/0.81036`; absolute apex/lift/contact ball z were
`1.27654/0.16032/1.11623 m`. The exact interrupted/last checkpoint SHA-256 is
`00f914db575ef2c0bd9c5c0daa438b2576fe991fe8b4d82c67f8d551e1d814c5`.
This is negative low-orbit convergence evidence: do not resume the V69 actor,
critic or optimizer into a successor.

## Current Height/Survival-Aligned Stage-7 Restart: V70

The opt-in successor is `goal_d455_measured_qvel_rmp_vertical_v70`, launched
only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v70_restart_stage7_reward_aligned.sh`.
V70 restarts from the same clean V68 Stage-6 completion archive used by V69,
at true step `2250244096`, Adam time `460120`, SHA-256
`b82411bb6e19546fa2396c21112b90c2b2d3e3241b8c68a81719620f6bbf4513`.
It must reject V69 or another source, curriculum-history resume, optimizer or
critic reset, observation migration, CMDP and advantage reweighting.

V70 preserves all 28 V69 environments, control, observations, resets, physics,
DR intervals, gates and next-domain collapse probes. Stages 1--6 retain their
exact V69/V68 reward configs and are never replayed. From Stage 7 onward it
changes only a coherent height/survival reward group: absolute-apex target /
sigma / event weight become `1.30 m / 0.035 m / 1.5`; the one-sided 1.28 m
lower edge is activated with weight `7000 m^-2`; lift target / sigma / event
weight become `0.175 m / 0.030 m / 1.0`, with a `0.145--0.205 m` weight-500
barrier; 60 ms joint apex/lift credit uses weight 8; full 1200-step episodes
with 13--15 hits receive weight-20 joint height credit; and the confirmed
antagonistic `racket_up_drift_penalty_weight` becomes zero. Dense first-hit
alive weight 5, post-hit survival weight 2.4, the 14-event positive-hit cap,
contact-height barriers and every safety/view term remain unchanged. The
completion-conditioned term is essential: height credit at that level cannot
be earned by sacrificing the 1200-step objective.

To reduce update-driven oscillation without copying GPU1's incompatible QACC
residual actor, V70 uses 1024 environments, rollout 128, LR `5e-5`, two PPO
epochs, clip `0.10`, target KL `0.004`, a 48-update convergence window and a
16-update warmup. Stages 7--10 require a 24-update strict hold. The first run
must be the launcher's exact-shape 64-update offline bounded trial. Pre-register
acceptance as final-window hits 13--15, conv_len >=0.90, full >=0.82, absolute
apex >=1.285 m without an early-to-late decline, lift >=0.160 m, zero qvel
exceedance, qacc exceedance <=0.01 and mean exact KL <=0.004. Only a passing
trial permits a new no-cap W&B-online run; never reinterpret the trial cap as
forced curriculum advancement. V70 remains experimental, simulation-only and
inherits every unresolved camera, replacement-ball and formal RMP/PD blocker.

## Rejected Coupled Height/Survival Stage-7 Restart: V71

The rejected reward-only successor is
`goal_d455_measured_qvel_rmp_vertical_v71`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v71_restart_stage7_coupled_survival.sh`.
V70 remains immutable negative bounded-trial evidence. Its exact 64-update,
1024-environment Stage-7 trial held absolute apex/lift at
`1.29160/0.16572 m`, with mean exact KL `0.00283`, zero qvel exceedance and
qacc exceedance `0.00425`, but failed hits/conv_len/full at
`12.9638/0.88073/0.79077`. Do not launch V70 formally or resume its trial
checkpoint.

V71 restarts the same clean V68 Stage-6 boundary and preserves all V70
environments, observations, reset distributions, physics, DR, gates, PPO and
reward fields except two Stage-7-onward weights. The complete 1200-step,
13--15-hit joint apex/lift reward rises `20 -> 40`, and the upward-racket
drift guard is restored only at weight `2` rather than V69's antagonistic
weight `6`. Raw post-first-hit alive and post-hit survival weights remain
`5/2.4`; the `1.28 m` absolute-apex lower barrier and every contact/view/
safety guard remain exact. This is a coupled objective: extra completion
credit cannot be earned by a low orbit or an early-terminated episode.

The exact 64-update trial failed at hits/conv_len/full
`12.8766/0.87763/0.77721`, while apex/lift remained
`1.29112/0.16589 m`. Racket-too-high terminations stayed at about `12.79` per
131072-sample update, indistinguishable from V70's `12.75`. Doubling the
coupled reward and restoring a weak drift guard did not repair survival. Do
not launch V71 formally or resume its checkpoint.

## Preserved Stable Observation-DR-Cap Restart: V72

The current opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v72`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v72_restart_stage7_obsdr_cap.sh`.
V72 restarts the exact clean V68 Stage-6 boundary, preserves V70's reward,
all 28 stages, Stage-20 complete polish, every gate, PPO, control, reset,
physics and non-observation DR field, and changes only six in-view
observation-calibration fields from Stage 7 onward. Their cap is the exact
12.5% midpoint between V70 Stage 6 and its reproducibly failing 25% Stage-7
boundary: position/velocity noise `0.004375 m / 0.04375 m/s`, coherent
position bias `0.0025 m`, rotation bias
`[0.43125,0.43125,0.625] deg`, velocity bias
`[0.025,0.025,0.03625] m/s`, and scale `[0.996125,1.003875]`.

This cap is a stable candidate, not hardware-coverage evidence. Four
independent histories at the 25% boundary plateaued near conv_len `0.88`,
while the Stage-6 domain completed at `0.961`; no measured real-camera audit
currently justifies keeping the failing wider bound. The V72 trial retains
conv_len >=0.90 and every V70 acceptance criterion.

While GPU1 is running, V72 additionally pins a GPU0-only host-memory guard:
GPU0 stops first at 6.25 GiB sustained available memory, 5 GiB critical
memory, or more than 256 MiB new swap under 8.5 GiB available memory. Never signal,
restart, stop, reconfigure or otherwise mutate GPU1; inspection is read-only.
Only a passing 64-update V72 trial permits a no-cap W&B-online launch in
`pp_gpu0`.

The exact 1024-environment, 64-update offline trial passed. Its final
48-update window had hits/conv_len/full
`13.69995/0.92047/0.85965`, absolute apex/lift
`1.28948/0.16540 m`, zero qvel exceedance, qacc exceedance `0.00403`, and
mean exact KL `0.00300`. Mean absolute apex increased from `1.28742 m` in
updates 1--16 to `1.28980 m` in updates 49--64, so the survival improvement
was not obtained by lowering the orbit. Relative to the matched V70 Stage-7
trial, true-view fraction improved `0.96898 -> 0.97903`, ball/racket contact
z decreased `1.12588/1.10607 -> 1.12409/1.10434 m`, qacc exceedance
decreased, and racket-high terminations fell from `12.75` to `8.02` per
131072-sample update. The bounded actor is evidence only and must not be
resumed; the formal no-cap W&B-online run must restart the pinned clean V68
Stage-6 boundary through the V72 launcher.

That formal restart was launched in `pp_gpu0` on 2026-08-26 as
`measured_qvel_rmp_vertical_v72_gpu0_seed20260943_20260826_stage7_obsdr_cap_online1`,
W&B run `v72g0s7a1`. Startup verified online sync, physical GPU0 UUID
`GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`, JAX `CudaDevice(id=0)`, the
pinned V68 source, Stage 7/28 and `max_stage_updates=-1`. It completed initial
updates while the independent GPU1 process remained alive; do not touch the
GPU1 process or session while monitoring V72.

That first formal process later stopped safely at Stage 7 update 248, true
step `2282749952`, because its GPU0-only guard observed sustained low host
available memory. This was not PPO, CUDA, temperature or checkpoint failure.
The interrupted checkpoint SHA-256 is
`f1990c963a4c98f109a654bb5cb5f7e9b675c28d92023547a91813d601f07891`;
its causal progress CSV SHA-256 is
`4e454abb0c67a6c53f49ebddb46f1b374731d5f4b5f28bd24bc42947b6ceb6c9`.
Resume only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v72_resume_interrupted_online.sh`.
It restores the exact actor, critic, Adam `t=464071`, stage anchor, update
counts and all 248 causal history rows in a new output directory while using
W&B `v72g0s7a1` with `resume=must` and zero step offset. The validator permits
same-profile V72 curriculum resume only for a signal-stopped, dimension-,
domain-, optimizer- and causal-step-consistent payload; the launcher further
pins the exact checkpoint and CSV hashes. Keep 1024 environments and the
GPU0-only host-memory guard. Never signal or reconfigure GPU1.

## Current GPU0 Spinless Continuation: V73

The current opt-in GPU0 successor is
`goal_d455_measured_qvel_rmp_vertical_v73`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v73_zero_spin_resume_v72.sh`.
It resumes the safely stopped V72 Stage-9-entry checkpoint at true step
`2326003712`, global update `578`, Adam time `468247`, SHA-256
`365ab6ef01e7fc79b7f3102f3263ba684b44a6b4ea10fef166191da723392832`.
The source progress CSV SHA-256 is
`a19fe50b0969a8fe600c8bbf9619a5cd1369f28b55540fd48aac80053329e498`.
The V72 process had completed Stage 8 and entered Stage 9 at local update zero
before its GPU0-only host-memory guard safely requested SIGINT. This was not a
PPO, CUDA, temperature or checkpoint failure.

V73 preserves the exact V72 actor, critic, Adam moments, 28-stage ordering,
Stage-20 complete polish, PPO, control, reward, gates, observation support,
reset sampling and every non-spin physics/DR field. On every stage it changes
only the complete spin-related DR contract used by GPU1 V131: reset-spin X/Y/Z
intervals become exactly zero and normalized inertia is fixed at the solid-
sphere value `0.40`. MuJoCo contact physics is unchanged, so contacts may
still generate natural ball angular velocity; V73 does not force the ball to
remain non-rotating.

Because the DR domain changes, V73 starts a fresh Stage-9 convergence window
and stage-local actor anchor; it must not restore V72 curriculum history. It
must restore actor, critic and Adam, and prohibits optimizer/critic reset,
observation migration, another source checkpoint or from-scratch launch. The
launcher keeps 1024 environments, the conservative V72 PPO/window values, no
stage update cap and the GPU0-only memory guard. By explicit user direction it
continues W&B-online run `v72g0s7a1` with `resume=must` and zero step offset,
using a new V73 output directory. GPU1 remains strictly read-only and must not
be signalled, restarted, stopped or reconfigured by this workflow.

The formal V73 run was launched in `pp_gpu0` on 2026-08-26 as
`measured_qvel_rmp_vertical_v73_gpu0_seed20260943_20260826_stage9_zero_spin_online1`.
Startup confirmed W&B `v72g0s7a1` resumed online, JAX
`CudaDevice(id=0)`, the pinned V72 source step and V73 Stage 9/28. It completed
at least eight 1024-by-128 updates; the CSV recorded normalized inertia near
`0.400000` and zero reset-spin draws. The independent GPU1 process remained
alive and was not controlled.

That run later stopped safely at Stage-9/global update 263, true step
`2360475648`, when the GPU0-only host-memory guard reported
`swap_growth_under_pressure`; it was not a PPO, CUDA, temperature or gate
failure. The interrupted checkpoint SHA-256 is
`020b40a1bdf9cbd00d163084349cfaa52146b9744d174f4eb8a0f04499bdef53`
and its progress CSV SHA-256 is
`914726f4c1bd73631224d49f84bbeae866814559f0eb56cc962d14225c094e57`.
The final Stage-9 row had the complete 24/24 strict hold and a passing Stage-10
block validation, but SIGINT was handled before the normal transition commit.
Resume this exact boundary only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v73_resume_interrupted_online.sh`.
The validator requires the causal convergence and next-domain evidence,
restores actor, critic and Adam, enters exact Stage 10 with a fresh convergence
window and actor anchor, and continues the same W&B run `v72g0s7a1` with
`resume=must` and zero offset. It writes
`measured_qvel_rmp_vertical_v73_gpu0_seed20260943_20260826_stage10_zero_spin_online1_resume1`.
Startup on 2026-08-26 confirmed JAX `CudaDevice(id=0)`, source true step
`2360475648`, and Stage 10/28
`rmp73_nonexecution_launch18_camera_missing_wide`. GPU1 remains strictly
read-only. The recovery completed at least 22 1024-by-128 Stage-10 updates,
reaching true step `2363359232` at about `19.6k steps/s`; after the intended
16-update warmup, rolling metrics became available normally.

## Current GPU0 Monotone-Survival Transaction Continuation: V86

The current opt-in GPU0 successor is
`goal_d455_measured_qvel_rmp_vertical_v86`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v86_cgroup_resume_v85_stage23_pass.sh`.
V85 remains immutable. V86 resumes only V85's Stage-23 pass-boundary
checkpoint at true step `2900099072`, Adam time `529643`, SHA-256
`e48c5b50ebfea05df3cd44748eaa13094ce3f0409315a32b31d4914185c0383d`,
and enters Stage 24 with fresh convergence and transaction history. It must
reject V85's Stage-24 best/last checkpoints, another source step/profile,
curriculum-history resume, optimizer/critic reset, or a non-57-D actor and
368-D critic.

V86 changes no stage, reward, gate, domain, control, observation, physics,
DR or PPO field. It requires the trainer's atomic main-metric transaction
guard with 16-update candidate blocks and 256 deterministic first-episode
lanes at seed `20261004`. While the incumbent is below either Stage-24
`conv_len/full=0.95/0.90` gate, promotion requires a strict reduction of the
combined gate gap, no increase in either individual gap, and no increase in
the distance to any quality/safety gate. Actor, critic and Adam moments are
committed or rolled back together. Qvel exceedance must remain zero and qacc
exceedance at most `0.01`. Pending/rejected blocks never enter convergence
history or best/periodic checkpoints.

V86 additionally requires
`--transaction-primary-value-nonregression`: paired deterministic
`conv_len` and `full` must each be componentwise non-decreasing even when the
incumbent is already above both Stage-24 gates. Do not rely only on zero gate
gap, because the generic historical transaction behavior permits movement
inside the accepted interval.

The V85 audit includes all 539 progress rows: Stage 23 peaked at rolling
`conv_len/full=0.92678/0.87533` but ended at `0.92211/0.85969`; Stage 24
ended at `0.90233/0.83283`, with post-window slopes
`-6.38e-5/-3.64e-5` per update. PPO KL, explained variance and safety stayed
healthy, identifying unprotected optimizer drift rather than numerical
instability. The accepted source is the Stage-23 pass boundary, not the V85
Stage-24 best ranking, because paired replay measured the boundary at
`1131.5` mean steps, `0.898` full and `14.172` hits.

Before formal training, run a separate 512-environment, 32-update offline
trial from the same source. Accept only if every transaction decision obeys
the pinned non-regression rule and at least one block is promotable without
quality/safety regression. Formal training uses 1024 environments,
minibatch 16384, V85's LR/clip/KL `5e-5/0.10/0.004`, no stage cap, a new W&B
identity (never append V85's `v72g0s7a1` history), a new output directory and
the 13/14 GiB zero-swap cgroup. Run only in tmux `pp_gpu0`; GPU1 remains
read-only. V86 is experimental, simulation-only and inherits every unresolved
real-camera, replacement-ball and formal RMP/PD coverage blocker.

The 2026-08-28 strict bounded evidence passed. Trial 2 rejected two harmful
candidates (`0.92969/0.87891 -> 0.92380/0.85938` and
`0.92969/0.87891 -> 0.93615/0.87500`). Independent trial 3 rejected
`0.94306/0.89844 -> 0.92963/0.87891`, then accepted the componentwise
non-regressing update-32 candidate `0.94306/0.89844 ->
0.94323/0.89844`; qvel exceedance was zero, qacc exceedance was `0.004870`,
and no secondary gate regressed. Pending work after update 32 is not promoted,
and formal training must restart the pinned V85 source.

Formal V86 started on 2026-08-28 in `pp_gpu0:v86_train`, output directory
`measured_qvel_rmp_vertical_v86_gpu0_seed20261005_20260828_stage24_monotone_transaction_online1`,
with new W&B run `v86g0txn1`. Startup confirmed `CudaDevice(id=0)`, the exact
source true step `2900099072`, Stage 24, 57/368 observation dimensions and
1024 environments. The initial formal paired incumbent is
`conv_len/full=0.93621/0.88281`; monitor transaction decisions against this
accepted baseline, not pending raw rollout rows.

GPU1 V129 was safely interrupted at update 537 solely by the host-memory
guard, not a NaN, safety, GPU-temperature or checkpoint failure. Resume it
only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v129_resume_interrupted_nenv512.sh`,
which pins checkpoint SHA-256
`ccffab6fc6cbac204a7362628a4b58dd13a48939c1359688158c9420d9abc22e`
and progress SHA-256
`65e87b96e7ba1be62d45085a1d36c7e5ad3e737ad4c5774d13a83833fbc3f6a3`.
The recovery preserves Stage 23, its 48-update history, actor, critic, Adam,
reward, DR, rollout length and all gates. It changes only resource
parallelism from 1024 to 512 environments, so `512*256=131072` samples/update
matches the concurrently stable GPU0 resource envelope. GPU1 ball-spin DR is
already exactly zero; do not copy V67's half-spin interval. The GPU0 half-
solref screen was weaker than half spin, so V129 solref also remains exact.
The recovery is active in `pp_gpu1` with a new offline W&B/output identity;
do not reuse the interrupted output directory or reset optimizer/critic.

## Current GPU1 Spinless Observation-Rate Continuation: V131

The current opt-in GPU1 successor is
`goal_d455_sport_taskspace_record_new3_sim2real_spinless_polish_v131`,
launched only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v131_spinless_stage35_from_v130_stage34.sh`.
V130 Stage 35 regressed after introducing latent reset X-spin; its stopped
Stage-35 checkpoint is negative evidence and must not be resumed. V131 instead
resumes the completed V130 one-based Stage-34 boundary checkpoint at true step
`10120822784`, Adam time `960897`, SHA-256
`d3c87a12fc14765eb26706ddb4c8af2703a8b2206e5277404904d40042929988`.
It restores the exact 67-D actor, 279-D critic and Adam moments, starts V131
Stage 35 with fresh convergence history, and prohibits curriculum-history
resume, optimizer/critic reset, observation migration and another source.

V131 preserves V130 Stages 1--34 and every non-spin environment, reward,
gate, reset, control and PPO value. From Stage 35 onward, reset spin is zero
on all axes and normalized inertia is fixed at `0.40`; natural contact-created
rotation remains enabled. Exact fractional observation scheduling is 60 Hz
for Stages 35--40, direct 90 Hz for Stages 41--42, and 60 Hz again for the
Stage 43--47 polish/proof tail. There is no 75 Hz bridge. The formal run keeps
1024 environments by 128 steps and W&B online. By explicit user authorization,
it resumes W&B ID `v130g1r1` with W&B-only offset `128974848`; true checkpoint
steps remain unchanged and output uses a new V131 directory.

## Current GPU1 Post-Hit Ball-Latency Continuation: V132

The current opt-in GPU1 successor is
`goal_d455_sport_taskspace_record_new3_sim2real_post_hit_latency_v132`, launched
only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v132_post_hit_latency_from_v131_stage38.sh`.
V131 Stage 38 passed, but V131 Stage 39 plateaued after reset-sampled
`dr_randomize_latency` delayed the complete base observation. V132 must resume
only the completed Stage-38 archive at true step `10828742656`, Adam time
`1047298`, SHA-256
`e436e690b28d0b4bdb8d907e1cdcf459bb769a9814f18dd982ead7577d7afedb`.
It restores actor, critic and Adam moments, starts fresh V132 Stage-39
convergence history, and rejects curriculum-history resume, optimizer/critic
reset, observation migration, another source or the negative V131 Stage-39
checkpoint.

V132 preserves V131 Stages 1--38 exactly. It replaces V131 Stages 39--40 with
five exact-60-Hz post-hit ball-latency lessons whose fresh-frame activation
probabilities are `0.25,0.50,0.75,1.00,1.00`; the sampled support remains
0/1/2 control ticks. Before the first confirmed hit injected latency is zero.
Delay only ball position/velocity, ball-racket relative position and ball age;
never delay q/dq, base, racket or action-feedback features through this mode,
and never stack it with whole-state `dr_randomize_latency`. The remaining
90-Hz then final-60-Hz tail retains the committed post-hit ball-only contract.

The existing dense geometric post-hit-survival reward ramps
`2.10 -> 2.20 -> 2.30 -> 2.40` across the first four lessons and stays `2.40`;
the next-contact anchor penalty stays at the passed Stage-38 value `0.60`
rather than taking V131 Stage 39's reduction to `0.50`. Do not add an
unconditional alive reward or relax the existing complete gate set. V132 has
50 stages, keeps 1024 environments by 128 steps and all V131
PPO/control/no-spin/contact/actuator settings. By explicit user direction,
the launcher appends to W&B-online ID `v130g1r1` with `resume=must` despite the
changed observation-DR/reward semantics. Its W&B-only offset `355074048`
places the first V132 rollout one batch after V131's final history point while
leaving true checkpoint/optimizer steps unchanged. Output remains a new V132
directory and must not overwrite V131 artifacts.

The explicitly authorized Stage-42 recovery sets
`V132_RESUME_FROM_STAGE41=YES` on the same launcher. Stage 41 passed and its
boundary archive is true step `10876059648`, Adam time `1053074`, SHA-256
`7778275b59e5e46dc177792015f75aec258237f00fa80ceafd87208b7ab937cd`.
Resume it directly at `record_new3_sim2real_v132_post_hit_ball_latency_p100_60hz_commit`
with fresh stage/convergence history; discard the single Stage-42 update saved
after the host-memory guard requested a safe stop. Continue W&B `v130g1r1`
with W&B-only offset `355205120` and use a new output directory. Clear JAX
caches between subsequent stages to prevent compiled stage variants from
accumulating host memory; this does not change checkpoint optimizer steps.

## Current GPU1 Timestamp-Consistent Ball-Latency Continuation: V135

V133 is preserved negative reward-repair evidence and V134 was rejected
before any optimizer update; neither may be formal-launched. The current
opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_timestamp_latency_v135`,
launched only by
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v135_timestamp_latency_from_v132_stage41.sh`.
It must restart the clean, naturally passed V132 Stage-41 p75 boundary at true
step `10876059648`, Adam time `1053074`, SHA-256
`7778275b59e5e46dc177792015f75aec258237f00fa80ceafd87208b7ab937cd`.
Restore the 67-D actor, 279-D critic and Adam moments, but start fresh Stage-42
history with the 16-update warm-up. Never formal-resume a bounded-trial actor
or append cold environment-reset episodes to an old convergence window.

V135 has the same 50 stages as V133 and preserves Stages 1--41 exactly. It
enters p100 directly with no bridge/micro-bridge. Stage 42 and later make the
selected delayed ball position/velocity, ball-racket relative position and
actor `ball_obs_age` share one camera acquisition timestamp; held 60 Hz frames
age at the 200 Hz control rate. The first clean 64-update trial recovered
after warm-up and stayed PPO/safety healthy, but its terminal
hits/`conv_len`/full were `11.6895/0.89307/0.78503`. A matched-optimizer-scale
diagnostic showed that the inherited V123 actor anchor at coefficient `0.01`
was pulling this already adapted Stage-41 actor strongly back toward the old
residual-100 policy. The final one-variable confirmation therefore reduces
only that launcher coefficient to `0.002`. Its terminal window was
`11.78915/0.90033/0.79765`, view `0.93363`, apex `0.22322 m`, and its last-12
exact KL/qacc-exceedance were `0.00200/0.00262`; the executable strict stage
gate passed. The auxiliary full-rate target missed `0.80` by `0.00235`, so do
not overstate it as an exact auxiliary-gate pass. By explicit user direction,
stop bounded parameter scanning and use the uncapped formal run to continue
from this evidence. Never use a bounded-trial actor as the formal source.

Formal V135 uses 1024 environments by 128 steps, minibatch 16384, two epochs,
LR `3e-5`, clip `0.08`, target KL `0.003`, actor-anchor coefficient `0.002`,
no stage-update cap and block validation. It starts in tmux `pp_gpu1`, writes
a new output directory, and continues W&B-online ID `v130g1r1` with
`resume=must` and W&B-only offset `610009088`; checkpoint/optimizer time is
unchanged. V135 is simulation-only and is blocked from hardware by the
age/prediction deployment contract below.

## Current GPU1 Stage-43 Gate-Calibrated Continuation: V136

V135 is preserved. Its Stage 43 ran for 6554 updates and was safely stopped at
true step `11839438848` after rolling performance deteriorated to hits
`11.51411`, length fraction `0.90019`, full-episode rate `0.79597` and view
rate `0.93297`. The 0.92 length gate was never reached. This is not a new DR
or model-capacity failure: V135 Stages 42 and 43 have identical environment,
reward, control, observation, physics and DR values; only their graduation
gates differ.

The current opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_gate_calibrated_v136`,
launched only by
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v136_gate_calibrated_from_v135_stage43.sh`.
It restores only the V135 Stage-43 update-5400 archive at true step
`11688181760`, Adam time `1152135`, SHA-256
`67c731bb518b051910508bfd3d3a8a37743eaadc309edd46cb3fd6105b23bed7`.
Frozen 128-episode validation measured hits `12.15625`, length fraction
`0.91665`, full rate `0.83594`, view rate `0.94789` and qacc exceedance
`0.00389`, making it a stronger joint parent than the stop checkpoint. Restore
actor, critic and Adam moments, but start a fresh Stage-43 convergence window
with the 16-update warm-up. Never resume the stopped window or either bounded
trial actor.

V136 preserves all 50 V135 stages and keeps Stages 1--42 exactly unchanged.
At Stage 43 only, rolling hits/length gates become `11.8/0.91`; the full rate
`0.70`, view rate `0.94`, apex `0.218--0.248 m`, hit-distribution, cadence,
motion-quality and safety gates are unchanged. Stage 42 used `11.5/0.90`, so
V136 Stage 43 remains strictly harder. Re-evaluating every recorded V135 gate
leaf showed a 17-update fully passing interval at updates 2515--2531 under
`11.8/0.91`, sufficient for the unchanged 16-update hold; `12.0/0.91` reached
at most two consecutive updates and `12.0/0.92` never passed. Stages 44--50
retain every V135 gate, including later `12.0/0.92` consolidations.

Do not promote the two 512-environment diagnostic changes. Lowering the actor
anchor coefficient to `0.0005` ended at hits `11.7688` and pushed apex below
the gate while anchor KL grew to `1.61`. A bounded 30-point, 12-hit completion
reward briefly created a joint pass but ended at hits/length `11.7465/0.90`
with anchor KL `1.85`; it did not establish sustained improvement. V136
therefore changes no reward, DR, PPO or anchor parameter. Formal training keeps
1024 environments, 128 rollout steps, minibatch 16384, two epochs, LR `3e-5`,
clip `0.08`, target KL `0.003`, anchor coefficient `0.002`, no stage cap and
block validation. Launch inside tmux `pp_gpu1`, continue W&B-online ID
`v130g1r1` with `resume=must` and W&B-only offset `761266176`, and use a new
output directory. V136 remains simulation-only and inherits the deployment
blockers below.

## Current GPU1 Success-Stability Continuation: V137

V136 was stopped after 697 fresh Stage-43 updates at true step
`11779538944`. Its first fresh window could meet the original task contract
(`12.0586` hits and `0.92380` length fraction at update 19), but the final
48-update means degraded to about `11.51` hits, `0.891` length fraction and
`0.923` view. The unchanged DR and healthy PPO/safety diagnostics rule out a
new DR or model-capacity boundary. The old V123 anchor KL simultaneously fell
from roughly `1.65` toward `1.18`, identifying continued attraction toward an
obsolete pre-latency actor as one source of the drift.

The current opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_success_stability_v137`,
launched only by
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v137_success_stability_from_v135_stage43.sh`.
It restores the same clean V135 Stage-43 update-5400 checkpoint used by V136,
SHA-256
`67c731bb518b051910508bfd3d3a8a37743eaadc309edd46cb3fd6105b23bed7`,
and uses that selected checkpoint itself as the resumed Stage-43 actor anchor.
Genuine stage advances retain the trainer's fresh stage-entry anchor. Never
resume a bounded V137 trial actor or the degraded V136 tail.

V137 returns to every original V135 graduation threshold; Stage 43 is again
`12.0` hits and `0.92` length fraction. Stages 1--42 are exact V135 copies.
From Stage 43 onward only the full-episode completion event changes from
20 points after 10 hits to 30 points after 12 hits. Stage 43 is already the
same domain in which the selected parent trained for 5400 updates, so its
profile minimum becomes 16 updates and its unchanged 16-update full strict
hold is the effective admission test. All later stage minimums, holds, gates,
DR, observations, control, physics and other rewards remain unchanged.

The selected PPO settings are LR `1.5e-5`, clip `0.06`, target KL `0.002`,
log-std bounds `[-4.2,-3.8]` and stage-local anchor coefficient `0.01`.
Matched 512-environment trials rejected stage-local anchoring alone and the
completion reward at the old PPO scale. The selected low-step/low-exploration
trial produced a 16-update complete strict pass at updates 22--37, peaking at
hits/length/full/view `12.4615/0.95892/0.90769/0.95366`; qacc exceedance was
`0.00259` and exact KL `0.00104` at that peak. A repeat was weaker and is kept
as variance evidence, so formal training uses 1024 environments and no trial
checkpoint. The formal launcher keeps 128 rollout steps, minibatch 16384, two
epochs, no stage cap and block validation. Run only inside tmux `pp_gpu1` and
continue W&B-online ID `v130g1r1` with `resume=must` and W&B-only offset
`852623360`. V137 remains simulation-only and inherits the deployment
blockers below.

## Current GPU1 Failure-Time Survival Continuation: V138

The opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_failure_time_v138`, launched
formally only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v138_failure_time_from_v137_best.sh`.
V137 Stage 43 ran for 1293 updates without reaching its unchanged 0.92 rolling
length gate: rolling hits passed 12 on 130 updates, while rolling length peaked
at `0.91642` and passed zero times. The closest joint window was update 366 at
hits/length/full/view `12.0966/0.91642/0.82829/0.94220`; the final window was
about `11.893/0.906/0.806/0.935`. PPO remained numerically conservative, so
this is reward-to-survival credit assignment rather than DR, model capacity or
optimizer instability.

V138 copies V137 exactly through Stage 42. From Stage 43 onward it adds only
`early_termination_remaining_horizon_penalty=18`: a true termination at step
`t` receives `-18*(1200-t)/1200`, while a normal step-1200 truncation receives
zero. Eighteen is the exact six-second scale of the existing direct
post-first-hit alive credit (`3 reward/s * 0.005 s * 1200`), not an unbounded
failure multiplier. V137's 30-point completion event gated by 12 hits, all
other rewards, every gate, PPO, model, control, observation, physics and DR
remain unchanged. This follows the accepted GPU0 V85 failure-time repair: the
event is emitted at the actual failure so finite-lambda GAE can assign it to
the preceding causal contact cycle.

Both the bounded trial and formal run restart from the V137 best Stage-43
checkpoint at update 483, true step `11751489536`, Adam `t=1159863`, SHA-256
`c1c4217447445f21393f981a497125050c6f3b61970bd3375efbcbcef2843499`.
Never promote the 512-environment trial actor. Formal training keeps 1024
environments by 128 steps, the V137 PPO values, no update cap and block
validation, except for the accepted LR increase `1.5e-5 -> 2.0e-5`, and
continues W&B `v130g1r1` with `resume=must` and W&B-only
offset `958922752`. V138 remains simulation-only and inherits every V135
deployment blocker.

The 64-update 512-environment V138 LR comparison selected `2.0e-5`: its final
48-update hits/length/full/view were
`12.1314/0.91732/0.82742/0.94311`, versus
`12.0372/0.91210/0.82525/0.94080` at `1.5e-5`. Its final rolling hits/length
were `12.06497/0.91555`; exact KL and qacc exceedance remained
`0.00127/0.00307`. The trial actor is not a formal source. A separate V139
30-point penalty trial was rejected at `11.8454/0.90728/0.81309` because the
larger bound was worse than V138; preserve V139 only as negative evidence.

### V138 Stage-43 plateau optimizer continuation

The first formal V138 continuation was stopped safely at Stage 43 update
2702, true step `12105646080`. Across 2702 updates it reached the unchanged
`0.92` rolling-length gate only three times, despite 1030 hits-gate passes;
the peak rolling length was `0.92157`. The final 256-update rolling means were
hits/length/full about `12.0913/0.91340/0.82446`, with essentially zero hits
slope and a negative length slope. Exact KL `0.00100`, explained variance
`0.96773` and qacc exceedance `0.00301` reject numerical instability and a
safety trade as the cause. Do not continue the original PPO settings and do
not lower the Stage-43 gate solely from this history.

The reproducible bounded comparison is launched by
`run_gpu1_record_new3_v138_plateau_trial_from_v138_best.sh`. Every arm starts
from the formal V138 best checkpoint at true step `11890425856`, SHA-256
`ab5b2fa160e828b16a5156ad9cac31247ae723ae640910bd4965fed9b9454f80`,
uses the frozen V138 trainer/environment/XML, 512 environments, 128 steps and
at most 64 updates. Trial actors are never formal sources. Reducing the actor
anchor to `0.005` and using GAE `0.999` with LR `1.5e-5` were rejected. The
selected GAE `0.999`, LR `2e-5`, anchor `0.01` arm produced 21 complete
performance-gate passes, including 17 consecutive updates 23--39, at the
unchanged hits/length gates `12.0/0.92`. Its last-48
hits/length/full/view were `11.7759/0.91546/0.82792/0.94396`; exact KL and
qacc exceedance were `0.00131/0.00295`. This is evidence that Stage 43 still
has optimizer headroom, so lowering the gate is not authorized.

The formal optimizer continuation uses
`run_gpu1_record_new3_v138_gae999_from_v138_best.sh`. It restarts the pinned
formal V138 best checkpoint, never a trial actor, and changes only GAE lambda
`0.995 -> 0.999`; V138 reward, gates, model, DR, control, observation and all
other PPO values remain unchanged. It uses 1024 environments, 128 steps, a
new output directory and W&B-online `v130g1r1` with `resume=must` and W&B-only
offset `1174142976`. Preserve the original V138 launcher and history.

## Current GPU1 Evidence-Calibrated Stable-Ceiling Continuation: V140

The 1024-environment GAE-0.999 continuation above is rejected. Across 1341
updates it never passed the old `0.92` rolling length gate and peaked at only
`0.918375`; final-256 rolling hits/length/full were
`11.86187/0.912210/0.819313`. Exact KL, explained variance and qacc remained
healthy, so this is a failed replication of the bounded optimizer arm, not a
reason to continue GAE `0.999`.

The current opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_stable_ceiling_v140`, launched
only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v140_ceiling_from_v138_best.sh`.
V140 preserves V138 stages 1--42. It changes only late length gates: stages
43, 45 and 47 use `0.915` instead of `0.92`, and the Stage-50 final proof uses
`0.92` instead of `0.95`; all intermediate `0.90` gates and every hits, full,
view, apex, contact, motion and safety gate remain unchanged. Reward, actor
and critic, control, observations, physics, DR and PPO are V138-exact.

The original V138 GAE-0.995 history passed the complete Stage-43 gate at
length `>=0.915` on 188 updates, with a longest 44-update run, after spending
2702 updates in that unchanged domain. V140 accepts that history and resumes
at Stage 44 rather than damaging the policy by retraining Stage 43. It restores
only the protected V138 best at update 1060, true step `11890425856`, SHA-256
`ab5b2fa160e828b16a5156ad9cac31247ae723ae640910bd4965fed9b9454f80`,
including critic and full Adam state. It must retain the original V137 best
as immutable actor anchor, true step `11751489536`, SHA-256
`c1c4217447445f21393f981a497125050c6f3b61970bd3375efbcbcef2843499`.

Moving the anchor to the V138 best was rejected: its 64-update trial ended at
rolling hits/length/full `11.4582/0.89297/0.78660`. The preserved-anchor
Stage-44 trial instead ended at rolling hits/length/full/view
`12.3683/0.93180/0.86781/0.95528`, with every performance, behavior and safety
gate true. It hit the 96-update trial cap only because 512 environments yielded
22/48 updates with at least 64 completed episodes; this eligibility shortfall
is not a strategy failure. Never promote either trial actor.

Formal V140 uses 1024 environments, 128 rollout steps, minibatch 16384, two
epochs, LR `2e-5`, gamma/GAE `0.9995/0.995`, clip `0.06`, target KL `0.002`,
actor-anchor coefficient `0.01`, no stage cap and block validation. Run only
in existing `pp_gpu1` and continue W&B-online `v130g1r1` with `resume=must`
and W&B-only offset `1349910528`. Do not alter the true checkpoint/Adam step,
resume Stage 43, use GAE `0.999`, move the actor anchor, or use a trial actor.
V140 remains simulation-only and inherits every V135 deployment blocker.

## Current GPU1 Late View-Budget Experiment: V141

The opt-in successor is
`goal_d455_sport_taskspace_record_new3_sim2real_view_budget_v141`.
V141 preserves V140 stages 1--46 exactly. From Stage 47 onward, it changes
only `hit_count_floor_reward_weight 8.0 -> 7.5` and the existing continuous
`ball_view_xy_center_penalty_weight 0.375 -> 0.5`. Every gate and every other
reward, model, domain, control, observation, physics, DR and PPO value remains
unchanged. Do not reinterpret V141 as a lower view gate or a stronger generic
failure penalty.

V141 may resume only the protected formal V140 Stage-47 archive at update
2400, true step `12308414464`, SHA-256
`d382ee87be1bb6787828a4f67c7cce0b34bbed7a7f0df357304510f4382fe6df`.
The immutable actor anchor remains the V140 Stage-47 entry at true step
`11933024256`, SHA-256
`9b5d1e7741eddb9d83cb9ff455cc2c456666b6af552fe6d4097892eb82f4c718`.
The 64-update launcher
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v141_view_budget_trial.sh`
is bounded evidence only; never promote its actor. Formal training, if the
paired frozen and online checks pass, restarts from the protected parent via
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v141_view_budget_formal.sh`
in existing `pp_gpu1`, reuses W&B ID `v130g1r1` with `resume=must`, and uses
the W&B-only offset `1586364416`. Do not change the true checkpoint/Adam step.

The 64-update bounded V141 trial is rejected for its primary view objective.
Final convergence hits/length/full/view were
`12.0980/0.91080/0.82353/0.94312`. Frozen paired view improved only
`0.927 -> 0.928` at seed 82731, below the preregistered `+0.002`, although it
improved `0.921 -> 0.926` at seed 97731 and retained hits/survival. Do not run
the formal V141 launcher or promote its trial actor.

## Current GPU1 Isolated Optimizer Experiment: V142

The next serialized arm is
`goal_d455_sport_taskspace_record_new3_sim2real_optimizer_lr_v142`. It returns
to the exact V140 reward, gate, model, domain, control, observation, physics
and DR contract; only the matching launcher changes LR `2e-5 -> 3e-5`.
Gamma/GAE, clip, target KL, entropy, actor anchor, rollout, minibatch and
epochs remain unchanged. Stages 1--46 are exact V140 copies, while only the
late resumed stage identities use `v142`.

Bounded testing uses only
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v142_lr3e5_trial.sh`,
the protected V140 update-2400 parent and the immutable V140 Stage-47 entry
anchor. Never promote the bounded actor. If the preregistered online and
two-seed frozen checks pass, formal continuation restarts from the protected
parent through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v142_lr3e5_formal.sh`
in existing `pp_gpu1`, reuses W&B `v130g1r1` with `resume=must`, and uses
W&B-only offset `1586364416`. GPU0 tuning must not begin until the selected
GPU1 formal continuation has completed at least 50 new updates.

The 64-update V142 result is negative. Final convergence
hits/length/full/view were `11.8680/0.90207/0.82197/0.93383`; paired view
regressed from `0.927 -> 0.916` and `0.929 -> 0.924` on seeds 82731/97731,
with hits and length also lower. PPO/critic/safety diagnostics stayed healthy.
Do not run formal V142 or promote its bounded actor. The selected formal GPU1
recovery is the protected V140 update-2400 parent with the original LR
`2e-5`, original reward, and unchanged Stage-47 entry actor anchor.
Launch it only through
`pingpong_controller/tools/rl_sim/run_gpu1_record_new3_v140_stage47_resume_u2400.sh`
in existing `pp_gpu1`. It uses W&B `v130g1r1`, `resume=must`, and W&B-only
offset `1586364416`; the true source step and Adam time remain unchanged.
GPU0 tuning must not begin before this formal recovery completes at least 50
new updates.

## Preserved Negative GPU0 Long-Credit Continuation: V87

V86 was stopped safely at Stage 24 update 543. Its complete 543-row,
1345-column audit found 33 transaction boundaries but only four promotions,
at updates 16, 80, 176 and 224. No candidate after update 224 was accepted:
319 later PPO updates could not alter the incumbent actor, critic or Adam
state. The fixed 256-lane selector reported length/full improvement from
`0.936214/0.882812` to `0.959372/0.929688`, while final random rolling values
were only `0.924169/0.865002`. Treat this as fixed-set selection overfit and
exploration blockage, not monotone generalization.

The opt-in successor is
`goal_d455_measured_qvel_rmp_vertical_v87`, launched only by
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v87_resume_v86_best_u224.sh`
through its matching zero-swap cgroup wrapper. V85 and V86 remain immutable.
V87 copies all 32 V85 stages one-to-one; it changes only stage identity/notes.
Every environment, reward, gate, control, observation, physics and DR field is
exactly preserved. It must resume only V86's accepted update-224 best, true
step `2929459200`, Adam `t=530443`, SHA-256
`ff7332438b2704db3028dadb422d8bd83e7644878fc4249bbf298634726afc06`,
with the 57-D actor and 368-D critic. Curriculum-history resume, actor/critic
reset, optimizer reset, observation migration, another V86 state and the
unaccepted V86 tail must fail closed.

V87 removes the V86 transaction guard and changes temporal credit only. It
uses 512 environments by 256 rollout steps, minibatch 16384, two epochs, LR
`5e-5`, gamma `0.9995`, GAE lambda `0.995`, clip `0.10`, target KL `0.004`,
entropy `0.0002`, min log std `-4.8`, a 48-update convergence window and
16-update warm-up. The 131072-sample batch per update is unchanged. Do not add
a new scalar survival reward or enlarge the actor unless this isolated
long-credit experiment first fails its preregistered online and fresh-lane
tests; existing reward already separates full and failed independent episodes
by about 154 return points, and action clipping, PPO, critic and safety
diagnostics provide no evidence of a representation-capacity ceiling.

The complete diagnosis and acceptance contract are in
`pingpong_controller/outputs/rl_sim/v86_generalization_reward_diagnosis_20260828/V86_ROOT_CAUSE_AND_V87_PREFLIGHT.md`.
Formal V87 has no stage-update cap, must reuse the existing `pp_gpu0:0` window,
and continues W&B ID `v72g0s7a1` with `resume=must` and a W&B-only offset of
`1950744576`. Never create a replacement tmux session/window, change the true
checkpoint step/Adam time, or promote a bounded trial actor. V87 remains
experimental, simulation-only and inherits every unresolved V16/V17 real-
camera, new-ball-outcome and RMP/PD coverage blocker.

The bounded V87 result is negative: updates 17--32 versus 33--48 changed mean
length/full from `1101.82/0.86627` to `1094.17/0.85989`. Do not launch V87
formally or use its bounded actor as a source. A frozen factor-domain audit of
the immutable V86 update-224 actor measured full rates `0.910`, `0.926`,
`0.895` and `0.867` in respectively in-view-noise, bridge-XY, wide-XY and
wide-XY-plus-final-hold domains. This rejects a 57-D actor-capacity ceiling and
isolates inadequate margin before composing wide XY and target-publication
hold.

## Preserved GPU0 Factorized Conservative Continuation: V89

V88 is a preserved negative experiment. It copies V85 stages 1--23, inserts
wide-XY/no-hold, bridge-XY/final-hold and unchanged joint-domain margin stages
at strict length/full `0.95/0.90`, then retains every later V85 stage in order.
Its 48-update bounded run raised mean length only `1118.04 -> 1119.11` while
full rate fell `0.882585 -> 0.879746`; do not launch V88 formally or promote
its actor.

The preserved opt-in continuation is
`goal_d455_measured_qvel_rmp_vertical_v89`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v89_cgroup_resume_v86_best_u224.sh`.
V89 copies all 34 V88 stage configurations exactly, including the three
factorized risk-margin stages and every reward, gate, control, observation,
physics and DR field. Its only change is conservative source-anchored PPO: LR
`2e-5`, clip `0.06`, target KL `0.002` and actor-anchor KL coefficient `0.01`.
It keeps 128 rollout steps, 1024 formal environments, minibatch 16384, two
epochs, gamma `0.9995`, GAE lambda `0.99`, entropy `0.0002`, min log std
`-4.8`, transaction guard off, 48-update convergence window and 16-update
warm-up.

V89 may resume only the immutable V86 update-224 best at true step
`2929459200`, Adam `t=530443`, SHA-256
`ff7332438b2704db3028dadb422d8bd83e7644878fc4249bbf298634726afc06`.
The same checkpoint is its actor-anchor reference. It must restore the 57-D
actor, 368-D critic and full Adam moments, start fresh at
`rmp89_wide_xy_no_hold_margin`, and reject curriculum-history resume,
optimizer/critic reset, observation migration, another source or a trial
actor.

The preregistered 512-environment, 48-update offline trial improved early to
late mean length `1088.896 -> 1122.583`, full rate
`0.855201 -> 0.887403`, with positive fitted slopes
`+1.308 steps/update` and `+0.001253/update`. Late exact KL was `0.000629`,
qvel exceedance zero and qacc exceedance `0.004863`. A fresh seed-20261015
paired 512-lane screen against the immutable source changed length
`1088.04 -> 1101.68`, full `0.861328 -> 0.867188`, hits
`13.5977 -> 13.8203`, and gained/lost full lanes `32/29`; paired intervals
cross zero, so this is directional preflight evidence, not statistical proof.
Never promote the bounded actor; formal training replicates from the immutable
source with 1024 environments and no stage-update cap.

Formal V89 must run only in the existing `pp_gpu0:0` window and continue
W&B-online ID `v72g0s7a1` with `resume=must` and W&B-only offset
`1950744576`. Do not create another tmux session/window or change the true
checkpoint/Adam step. V89 remains experimental, simulation-only and inherits
every V16/V17 real-camera, new-ball-outcome and formal RMP/PD coverage blocker.

## Current GPU0 Evidence-Calibrated Stable-Ceiling Continuation: V91

V90 is a preserved negative local-anchor experiment. It kept all V89
stages/PPO values and moved only the actor-anchor origin to the protected V89
update-1287 best. Although its 64-update online window improved, a fresh
seed-20261017 512-lane frozen screen reduced mean length
`1115.273 -> 1111.646` and increased hit-racket mean/RMS vxy. Do not launch
V90 formally or use its bounded actor as a source.

The current opt-in continuation is
`goal_d455_measured_qvel_rmp_vertical_v91`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v91_cgroup_resume_v89_best_u1287.sh`.
It resumes only the protected V89 update-1287 best, true step `3098148864`,
Adam `t=550987`, SHA-256
`b714491493fa9f4403872d7a22def60402902677a9a3b843059a52381277d8ac`.
It restores the exact 57-D actor, 368-D critic and Adam moments, uses the same
checkpoint as the immutable local anchor and starts a fresh convergence
window at `rmp91_wide_xy_no_hold_margin`. Curriculum-history resume,
optimizer/critic reset, another source and the V90 trial actor must fail
closed.

The complete V89 run stopped after 1435 updates. Its rolling length/full
peaked at `0.935100/0.894780`; neither metric ever reached the old joint
`0.95/0.90` gate. After the selected best, both fitted slopes were negative
while exact KL, explained variance and safety stayed healthy. Treat this as a
local stable-juggling ceiling for the current actor/reward/domain/PPO
combination. The evidence-calibrated `0.93/0.88` joint gate passed on 293
updates, with a longest 61-update consecutive run.

V91 changes only every prior `0.95/0.90` polish gate at 1-based stages 24,
25, 26, 28, 30, 32 and 34 to `0.93/0.88`. Intermediate `0.92/0.84` gates are
unchanged. Stage 24 requires a fresh 48 updates and retains the original
20-update hold; rewards, domains, control, observations, physics, DR, PPO and
all other validation gates remain V89-exact. Formal training uses 1024
environments, no stage-update cap, the existing `pp_gpu0:0` window, W&B ID
`v72g0s7a1` with `resume=must`, and W&B-only offset `1970143232`. Never
create a replacement tmux session/window or alter the true checkpoint/Adam
step. V91 remains experimental, simulation-only and inherits every V16/V17
real-camera, new-ball-outcome and formal RMP/PD coverage blocker.

## Current GPU0 Paired-Promoted Kp/Kd Continuation: V96

V91 was safely stopped at Stage-28 update 7242, true step `4354998272`. The
serialized V92--V95 factor campaign selected protected V91 update 7050 at
true step `4329832448`, SHA-256
`c51199753bb5fb81029bb7c3653bf0ebf2037632899ae5005d6eae6a79f968dd`,
as its shared parent. V92 output-delay rollback failed online; V95
feedforward/acceleration-weight rollback failed frozen seed 20261019; V94
estimator rollback was not promoted because its seed-20261018 frozen
hit-racket vxy regressed to `0.122836 m/s`. Do not resume or stack V92, V94
or V95.

The current formal successor is
`goal_d455_measured_qvel_rmp_vertical_v96`, launched only through
`pingpong_controller/tools/rl_sim/run_gpu0_measured_qvel_rmp_vertical_v96_cgroup_resume_v93_trial64.sh`.
It resumes only the paired-promoted V93 update-64 last checkpoint at true
step `4334026752`, Adam `t=701217`, SHA-256
`b9364d90b0c129b1d31a8bb22626c696f199654426db6713ef834afbe5a68850`.
Across the two 512-lane frozen seeds, V93 improved aggregate length
`1108.590 -> 1115.913`, full rate `0.86035 -> 0.87305`, hits
`13.8945 -> 13.9297`, and mean hit-racket vxy
`0.121417 -> 0.119822 m/s` relative to the protected V91 parent.

V96 changes only Stage-28 RMP joint Kp/Kd multiplier DR from
`[0.875,1.125]` to `[0.9,1.1]`. Reward, gates, observations, bounded
reference, RMP/PD, physics, every other DR field and PPO remain V91-exact.
It restores actor, critic and Adam moments, retains the immutable protected
V89 update-1287 actor anchor, and starts a fresh convergence window at
`rmp96_rmp_internal_micro_6p25`. Another source, curriculum-history resume,
optimizer/critic reset, observation migration or a changed anchor must fail
closed. Formal V96 uses 1024 environments, no stage-update cap, existing
`pp_gpu0:0`, W&B ID `v72g0s7a1` with `resume=must`, and W&B-only offset
`1991114752`. V96 remains experimental, simulation-only and inherits all
unresolved real-camera, new-ball-outcome and formal RMP/PD coverage blockers.

## Ball-State Age and Prediction Deployment Contract

Treat `ball_obs_age` as the age of the ball state represented in the actor
observation: policy evaluation time minus that state's timestamp. It is not
the time since the last raw camera detection, the predictor horizon, a
confidence score, an in-view flag, or a generic missing-vision timer. Keep
those quantities separate in both simulation and deployment. Never attach an
old raw-measurement timestamp to a state that has already been propagated to
the current query time; that combination double-counts delay and creates a
sim-to-real observation mismatch.

When the real predictor propagates a valid measurement to the current query
time, the predicted position/velocity is a current-time state and its actor
age starts near zero. Only actual sampling, scheduling, publication and
post-prediction hold time may then increase actor age. The source-measurement
age, prediction horizon, predictor validity/confidence, raw-detection status,
in-view status and reacquisition state must remain available as separate
runtime diagnostics or safety inputs; do not silently substitute any of them
for actor `ball_obs_age` without a newly versioned observation contract and
matching training profile.

High-ball top-of-view occlusion and short detector failures may continue to
use a bounded current-time prediction while it remains physically valid. Do
not freeze policy observations merely because raw detections stop, and do not
make actor age grow solely with raw-detection absence when the supplied state
continues to be predicted to the present. Conversely, lateral out-of-view
flight, predictor timeout/divergence, physically invalid state and
unrecoverable loss are validity and safety-state events, not ordinary latency
samples. They must transition to the registered fail-safe/stop behavior once
the predictor validity contract fails; never feed an indefinitely stale or
unbounded extrapolated state as if its age were normal. Reacquisition must
preserve timestamp semantics and use an explicitly tested continuity/smoothing
path.

The current real launcher
`/home/yangzhe/Project/pingpong_playreal/pingpong-play/RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh`
and its simcode5/simcode6 runtime paths deliberately force valid-ball-age use
off. Therefore an age-conditioned checkpoint is not deployment-compatible
with those paths merely because its actor dimension is still 67. Before such
a checkpoint is used on hardware, create a separately versioned, fail-closed
runtime adapter that reproduces the training profile's exact state timestamp,
age normalization/clipping, observation-rate scheduling, prediction and
reacquisition semantics. Preserve the existing age-disabled profiles.

Do not claim that activating age has no material deployment gap from
simulation-only training. Pre-register and pass paired recorded-data replay
for at least fresh in-view tracking, top-of-view exit and return, transient
detection failure, lateral exit, bounded prediction timeout and
reacquisition. Record represented-state timestamps, raw-source age,
prediction horizon/validity, actor age, position/velocity continuity, policy
actions and safety transitions. Define acceptable observation/action
divergence and safety bounds before consulting the candidate result, then use
a staged supervised hardware probe. Short post-hit sampling-latency DR alone
does not cover long occlusion, arbitrary dropout, predictor error or
reacquisition.

## Training and Hardware Safety

Before any training, validation, recording, or robot run, inspect `git status --short --branch`, GPU UUID/process ownership, the complete launcher, XML path, checkpoint hash, stage numbering, output directory, and W&B identity. The training XML is normally `pingpong_controller/tools/rl_sim/moz1_pd.xml`, not `pingpong_controller/models/moz1_pd.xml`.

The 2026-09-06 V38/V39 incident was host global OOM, followed by user-session
termination, not a recorded GPU Xid or reboot. Their confirmed launchers must
retain the verified `training_memory_scope.py` cgroup boundary (11 GiB high,
12 GiB max per trainer, zero swap), pressure logging/stop escalation and
cross-stage cache cleanup. Do not bypass these guards or increase their limits
without a measured combined host budget. Fifty initial updates or low VRAM
usage do not establish sustained host-memory safety. See
`pingpong_controller/outputs/rl_sim/host_oom_audit_20260906/REPORT_CN.md`.

Formal sessions use tmux names `pp_gpu0` and `pp_gpu1`. Existing formal jobs are read-only unless the user explicitly authorizes intervention; do not send tmux input, stop a process, reuse its W&B run, or write into its directory. Use physical UUIDs, not logical labels, when ownership matters:

- GPU0: `GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4`
- GPU1: `GPU-e74458ec-002a-1bac-e0be-d0a5713b661e`

The current `run_gpu1_record_new3_v122_delay_prior.sh` selects device 1 but has default run/W&B labels containing `gpu0`. Treat this as an unresolved naming hazard: verify or override every identity before a future launch, without interrupting an existing job solely to rename it.

## Commit and Pull Request Guidelines

Recent history uses short imperative subjects, commonly scoped as `sim: ...` and `deploy: ...`. Continue with focused prefixes such as `sim:`, `deploy:`, `fix:`, `test:`, or `docs:`; keep one behavioral purpose per commit.

Never use `git add .` in this experiment-heavy worktree. Stage explicit files and review `git diff --cached --check`, `--stat`, and `--name-only`. Commit generated checkpoints or media only when they are named release artifacts with hashes and compact validation evidence; exclude caches, W&B internals, temporary XML, and bulk traces.

Pull requests must state the preserved baseline, changed contract, opt-in/default behavior, files and profiles affected, commands/tests run, experiment design/results, hardware/GPU use, and artifact paths/hashes. Link the issue when one exists and include plots or trace summaries for control, DR, observation, or sim-to-real changes.
