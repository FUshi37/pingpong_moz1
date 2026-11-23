"""V234 learns every PPO rollout and repairs stability through rewards only."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from fixed_plane_v231_recovery_only import RecoveryOnlyRewardConfig
from fixed_plane_v232_full_survival import (
    PARENT_PROFILE, PARENT_SHA256, REPLAY_SHA256, full_episode_gate,
    retention_stage, relaxed_hit_admission,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v234'
START_STAGE = 'fplc_v234_s01_workspace_alignment'
RETENTION_STAGE = START_STAGE
REWARD_FIELDS = frozenset({
    'fixed_plane_v158_late_squared_vxy_weight',
    'fixed_plane_v158_late_squared_omega_weight',
    'fixed_plane_v166_physical_ball_xy_weight',
    'v231_nominal_reward_parity',
    'hit_racket_vxy_penalty_weight',
    'fixed_plane_v156_adjacent_height_penalty_multiplier',
    'low_hit_penalty_weight',
})
PINNED_TRAINING = {
    'n_envs': 1024, 'n_steps': 128, 'minibatch_size': 16384,
    'update_epochs': 4, 'learning_rate': 1.5e-5, 'gamma': .9995,
    'gae_lambda': .99, 'clip_range': .12, 'target_kl': .006,
    'ent_coef': .001, 'min_log_std': -3., 'vf_coef': .05,
    'max_grad_norm': .5, 'hidden_dim': 256,
    'convergence_window': 32, 'convergence_min_episodes': 8,
    'min_stage_updates': 30, 'stage_metric_warmup_updates': 10,
    'failure_focus_hit_threshold': 14, 'failure_focus_tail_steps': 128,
    'failure_focus_weight': 1., 'failure_focus_balanced_mass': .02,
    'counterfactual_supervision_coef': .05,
    'actor_anchor_kl_coef': 0., 'actor_anchor_replay_kl_coef': 10.,
    'same_stage_reexec_every_updates': 0,
    'advance_eval_n_envs': 288, 'advance_eval_steps': 1200,
    'advance_eval_retry_updates': 20,
    'max_stage_updates': -1, 'max_stages': 17,
}


@dataclass(frozen=True)
class RewardOnlyConfig(RecoveryOnlyRewardConfig):
    """Identify the saved reward schema without extending observations/state."""

    v234_reward_only: bool = False


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all seventeen domains/gates; change existing reward weights only."""
    if len(source) != 17 or source[0].name != 'fplc_v233_s01_workspace_alignment':
        raise ValueError('V234 requires the complete unchanged V233 course')
    result = []
    for stage in source:
        cfg = RewardOnlyConfig(**vars(stage.cfg))
        cfg = replace(cfg, v234_reward_only=True,
            fixed_plane_v158_late_squared_vxy_weight=5.,
            fixed_plane_v158_late_squared_omega_weight=4.,
            fixed_plane_v166_physical_ball_xy_weight=2.,
            v231_nominal_reward_parity=False,
            hit_racket_vxy_penalty_weight=1.5,
            fixed_plane_v156_adjacent_height_penalty_multiplier=4.5,
            low_hit_penalty_weight=7500.)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v233_', 'fplc_v234_', 1),
            notes='V234 reward-only continuation: all17 domains and complete '
            'graduation gates are unchanged, including exact conv_len1. '
            'Late racket XY/omega squared weights2.5/2->5/4; all-hit racket '
            'XY excess weight1->1.5; physical outgoing-ball XY weight2 now '
            'also applies to nominal episodes. Adjacent-apex cost3->4.5 and '
            'low-apex weight5000->7500. No PPO/physics/control/observation/DR '
            'change. No transaction acceptance, source-KL hard projection, '
            'update cap or discarded convergence history. Stage graduation '
            'still requires original task gates and frozenS5 paired stability.'))
    return result


def validate_training_contract(settings: Any) -> None:
    """Freeze numeric training settings and prohibit extra proposal filters."""
    changed = [name for name, value in PINNED_TRAINING.items()
               if getattr(settings, name, None) != value]
    if changed:
        raise ValueError('V234 preserves training settings: ' + ', '.join(changed))
    if (settings.transactional_main_metric_guard
            or settings.transaction_backtrack_rejected_candidate
            or settings.transaction_revalidate_incumbent_on_rejected_boundary
            or settings.actor_anchor_replay_max_kl != 0.
            or settings.allow_unconverged_advance
            or settings.advance_mode != 'converged'
            or settings.advance_validation_mode != 'block'
            or settings.advance_retention_stage_name != RETENTION_STAGE
            or settings.advance_eval_deterministic
            or settings.reset_optimizer_on_resume
            or settings.reset_critic_on_resume
            or not settings.time_limit_bootstrap):
        raise ValueError('V234 requires uncapped learning without extra acceptance filters')


def validate_anchor_contract(settings: Any, *, source_digest: str,
                             replay_digest: str) -> None:
    """Keep the existing soft KL objective; remove only its hard projection."""
    if (source_digest != PARENT_SHA256 or replay_digest != REPLAY_SHA256
            or settings.actor_anchor_reference_checkpoint is None
            or settings.actor_anchor_replay_obs is None
            or settings.actor_anchor_kl_coef != 0.
            or settings.actor_anchor_replay_kl_coef != 10.
            or settings.actor_anchor_replay_max_kl != 0.):
        raise ValueError('V234 pins the original soft successful-S5 replay objective')


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool) -> None:
    """Restore full passed S5 or exact own history without old graduations."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        from fixed_plane_v232_full_survival import validate_resume_payload as validate
        if requested_start_stage != START_STAGE:
            raise ValueError('V234 starts all17 reward-only lessons at S1')
        validate(payload, requested_start_stage='fplc_v232_s01_workspace_alignment',
            resume_curriculum_state=resume_curriculum_state,
            reset_optimizer=reset_optimizer, reset_critic=reset_critic,
            allow_obs_dim_migration=allow_obs_dim_migration, probe=False)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V234 requires its unreset full-state uncapped continuation')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V234 requires exact causal stage/history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
