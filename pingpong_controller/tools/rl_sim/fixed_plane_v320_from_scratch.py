"""V320 pure-PPO course with user-authorized recurrent acquisition repair."""
from dataclasses import replace
from argparse import Namespace
from typing import Any, Mapping, Sequence
import math

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v320_from_scratch'
REVISION = 'recurrent_recovery_r5_20261029'
PARENT_STAGE_COUNT = 10
STAGE_COUNT = 21
FINAL_STAGE = 'fplc_v274_s11_camera_ar1_1x_consolidation'
# Source versions are those that passed native rolling and block validation.
SOURCE_SLICES = ((150, 0, 3), (155, 3, 7), (159, 7, 8),
                 (172, 8, 10), (258, 0, 6), (274, 6, 11))


def recurrent_guidance_reward(*, hit_count, pending_hit, target_valid,
        time_to_contact_s, racket_z_m, target_z_m, racket_vz_m_s,
        target_vz_m_s, weight_per_s: float, dt_s: float):
    """Bounded reward for recovery and a timed stroke after three real hits.

    Inputs are per-lane arrays shaped (N,). Targets come from the existing
    causal workspace-bounded stroke calculation, never a teacher or dataset.
    This function does not change commands, observations or contact counting.
    """
    import jax.numpy as jnp

    if not math.isfinite(weight_per_s) or weight_per_s < 0:
        raise ValueError('V320 recurrent guidance weight must be finite and nonnegative')
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError('V320 recurrent guidance dt must be finite and positive')
    active = ((weight_per_s > 0) & (hit_count >= 3) & ~pending_hit
              & target_valid & (time_to_contact_s > 0) & (time_to_contact_s <= .3))
    z_error = (racket_z_m - target_z_m) / .015
    vz_error = (racket_vz_m_s - target_vz_m_s) / .3
    score = jnp.exp(-.5 * (z_error * z_error + vz_error * vz_error))
    return jnp.where(active, weight_per_s * dt_s * score, 0.), active


def third_apex_guard_reward(*, hit_count, confirmed_event,
                           predicted_apex_z_m, weight: float):
    """Charge third-hit overshoot before it can destroy the fourth-hit flight."""
    import jax.numpy as jnp

    if not math.isfinite(weight) or weight < 0:
        raise ValueError('V320 third apex weight must be finite and nonnegative')
    excess_m = jnp.maximum(predicted_apex_z_m - 1.45, 0.)
    return jnp.where(confirmed_event & (hit_count == 3),
                     -weight * jnp.log1p(excess_m / .025), 0.)


def build_historical_stages(sources: Mapping[int, Sequence[Any]]) -> list[Any]:
    """Reuse complete historical stage objects, except authorized min/hold waits."""
    result = []
    for version, start, stop in SOURCE_SLICES:
        source = sources[version]
        if len(source) < stop:
            raise ValueError(f'V320 is missing historical V{version} stages')
        result.extend(replace(stage, min_updates=0, convergence_hold_updates=16)
                      for stage in source[start:stop])
    if len(result) != STAGE_COUNT or result[-1].name != FINAL_STAGE:
        raise ValueError('V320 requires all 21 historical graduated stages')
    return result


def repair_stages(stages: Sequence[Any]) -> list[Any]:
    """Repair only lessons 3--6; retain all later task and DR contracts."""
    result = list(stages)
    names = {3: 'fplc_v320_s3_recurrent_acquisition',
             4: 'fplc_v320_s4_recurrent_commit',
             5: 'fplc_v320_s5_guidance_bridge',
             6: 'fplc_v320_s6_guidance_fade'}
    for index, weight in ((3, 40.), (4, 40.), (5, 20.), (6, 10.)):
        source = result[index-1]
        # The environment phase authenticates the historical contact contract.
        # Only the curriculum stage identity changes for this reward repair.
        cfg = replace(source.cfg,
            fixed_plane_v320_recurrent_guidance_weight_per_s=weight,
            fixed_plane_v320_third_apex_weight=weight)
        gates = dict(target_mean_hits=5., target_hit6_rate=.35) if index == 3 else {}
        result[index-1] = replace(source, name=names[index], cfg=cfg, **gates,
            notes=('V320 user-authorized pure-PPO repair of ' + source.name + '. '
                'Reward-only post-third recovery/stroke guidance and third-hit '
                'overshoot cost. S3 must acquire >=5 mean hits and P6>=.35 '
                'before precision entry. No teacher, state/control modification, '
                'forced advance or relaxed final gates. Guidance fades by S7.'))
    return result


def build_stages(sources: Mapping[int, Sequence[Any]]) -> list[Any]:
    return repair_stages(build_historical_stages(sources))


def optimizer_settings(stage_index: int) -> tuple[str, float, float, float, float]:
    """Keep learned early phases; use V151's acquisition LR at the S4 boundary."""
    if not 1 <= stage_index <= STAGE_COUNT:
        raise ValueError('V320 optimizer stage is outside the complete course')
    if stage_index <= 3:
        return 'continuity', 3e-5, .12, .006, .05
    if stage_index == 4:
        return 'recurrent_acquisition', 3e-5, .12, .006, .05
    return 'quality_dr', 1.5e-5, .12, .006, .05


def exploration_settings(stage_index: int) -> tuple[float, float | None]:
    """Preserve proven S1--S3; bound stochastic action noise on recurrent lessons."""
    if not 1 <= stage_index <= STAGE_COUNT:
        raise ValueError('V320 exploration stage is outside the complete course')
    return (.001, None) if stage_index <= 3 else (0., -1.5)


def validation_settings(stage_index: int) -> dict[str, float]:
    """Restore actual V150/V155 versus V159+ independent block thresholds."""
    if not 1 <= stage_index <= STAGE_COUNT:
        raise ValueError('V320 validation stage is outside the complete course')
    early = stage_index <= 7
    return dict(advance_eval_hit_ratio=.5 if early else 1.,
        advance_eval_len_ratio=.5 if early else 1.,
        advance_eval_cond_hit_ratio=.9 if early else 1.,
        advance_eval_hit_rate_margin=.05 if early else 0.,
        advance_eval_camera_margin=.05 if early else 0.,
        advance_eval_ball_view_margin=.05 if early else 0.,
        advance_eval_z_ideal_margin=.05 if early else 0.)


def validate_pure_rl(args: Namespace) -> None:
    """Fail before loading any offline actor target, teacher or initial weights."""
    paths = ('resume_from', 'resume_curriculum_state',
        'actor_anchor_reference_checkpoint', 'actor_anchor_replay_obs', 'phase_teacher_reference',
        'residual_teacher_checkpoint', 'teacher_distill_checkpoint',
        'teacher_distill_replay_obs', 'counterfactual_replay_obs',
        'counterfactual_replay_actions', 'noise_invariance_clean_obs',
        'noise_invariance_noisy_obs')
    coefficients = ('counterfactual_supervision_coef', 'teacher_distill_coef',
        'actor_anchor_kl_coef', 'actor_anchor_replay_kl_coef',
        'noise_invariance_coef', 'actor_action_feedback_sensitivity_coef')
    if any(getattr(args, key, None) for key in (*paths, *coefficients)):
        raise ValueError('V320 pure PPO rejects checkpoints and offline/teacher supervision')


def validate_startup(args: Namespace) -> None:
    """Reject hidden checkpoint initialization and shortened/unvalidated courses."""
    validate_pure_rl(args)
    if args.max_stages not in (0, STAGE_COUNT) or args.resume_start_stage not in (None, 'auto'):
        raise ValueError('V320 must start at S1 and retain the entire parent chain')
    if (args.advance_mode != 'converged' or args.advance_validation_mode != 'block'
            or args.allow_unconverged_advance or args.min_stage_updates != 0
            or args.max_stage_updates != -1 or args.convergence_window != 32
            or args.stage_metric_warmup_updates != 10):
        raise ValueError('V320 requires uncapped native convergence and block validation')
    if not args.final_model_only or not args.runtime_timing:
        raise ValueError('V320 requires final-only storage and wall-clock accounting')
    if not args.allow_obs_dim_migration:
        raise ValueError('V320 requires the append-only 393-to-398 critic transition')
