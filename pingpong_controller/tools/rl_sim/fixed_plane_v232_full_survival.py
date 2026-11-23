"""V232 retests all recovery lessons with exact full-episode survival."""
from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from fixed_plane_v229_noninferiority import PARENT_PROFILE, PARENT_SHA256, REPLAY_SHA256
from fixed_plane_v226_stability_recovery import STABILITY_LIMITS

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v232'
START_STAGE = 'fplc_v232_s01_workspace_alignment'
PROBE_STAGE = 'fplc_v232_s08_near_y_020mm_bridge'
RETENTION_STAGE = START_STAGE
ANCHOR_COEFFICIENT = 10.
ANCHOR_BOUND = .0005
STRICT_HITS = 12.9
RECOVERY_HITS = 12.85


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve all17 V231 domains/rewards, and revalidate every new gate."""
    if len(source) != 17 or source[7].name != 'fplc_v231_s08_near_y_020mm_bridge':
        raise ValueError('V232 requires the complete preserved V231 course')
    result = []
    for index, stage in enumerate(source, 1):
        if stage.fixed_contact_condition_recurrent_episode_gates:
            raise ValueError('V232 cannot hide early failures behind recurrent conditioning')
        label = stage.name.split('_', 3)[3]
        result.append(replace(stage, name=f'fplc_v232_s{index:02d}_{label}',
            target_mean_len_frac=1., target_episode_truncation_rate=1.,
            target_mean_hits=STRICT_HITS if index <= 7 else RECOVERY_HITS,
            notes='V232 preserves V231 physics/control/DR/reward and all17 domains. '
            'Retest all lessons: no old graduation transfers to exact conv_len1. '
            'Every evaluated first episode must end at1200 ticks by truncation. '
            'S8--S17 may use12.85 rather than12.9 hits only jointly with complete '
            'survival, unchanged quality/safety gates and strict frozen-S5 paired '
            'non-degradation. Strengthen nominal replay KL10/bound0.0005; '
            'do not promote a negative probe or force advancement.'))
    return result


def full_episode_gate(outputs: Mapping[str, Any], *, max_steps: int) -> dict[str, float]:
    """Audit all initial lanes [T,N], including missing ends and early failures."""
    done = np.asarray(outputs['first_episode_done'], dtype=bool)
    lengths = np.asarray(outputs['episode_length'])
    truncated = np.asarray(outputs['metrics']['truncated'])
    terminated = np.asarray(outputs['metrics']['terminated'])
    if (done.ndim != 2 or max_steps <= 0 or done.shape[1] == 0
            or any(x.shape != done.shape for x in (lengths, truncated, terminated))):
        raise ValueError('V232 survival expects nonempty matching time x lanes arrays')
    ended = done.any(axis=0)
    first = np.argmax(done, axis=0)
    lanes = np.arange(done.shape[1])
    first_len = lengths[first, lanes]
    first_truncated = truncated[first, lanes]
    first_terminated = terminated[first, lanes]
    good = (ended & (done.sum(axis=0) == 1) & (first == max_steps - 1)
            & np.isfinite(first_len) & (first_len == max_steps)
            & np.isfinite(first_truncated) & (first_truncated == 1.)
            & np.isfinite(first_terminated) & (first_terminated == 0.))
    return dict(passed=float(np.all(good)), lanes=float(len(lanes)),
        complete_lanes=float(good.sum()), missing_end_lanes=float((~ended).sum()),
        failed_lanes=float((~good).sum()),
        first_episode_min_len=float(first_len[ended].min()) if ended.any() else 0.)


def episode_audit_arrays(outputs: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Retain lossless first-episode evidence alongside the evaluation metrics."""
    return {f'audit/{name}': np.asarray(outputs[name]) for name in
            ('first_episode_done', 'episode_length', 'done', 'evaluation_active')}


def relaxed_hit_admission(task: Mapping[str, Any], retention: Mapping[str, Any]) -> bool:
    """Admit the small count relaxation only after both full strict validations."""
    required = ((task, 'advance_eval/passed'),
                (task, 'advance_eval/v232_full_episode_passed'),
                (retention, 'retention_eval/passed'),
                (retention, 'retention_eval/v226_noninferior'),
                (retention, 'retention_eval/v232_full_episode_passed'))
    hits = float(task.get('advance_eval/mean_hits', float('nan')))
    return (math.isfinite(hits) and hits >= RECOVERY_HITS
            and all(float(row.get(key, 0.)) == 1. for row, key in required))


def retention_stage(stage):
    return replace(stage, target_mean_len_frac=1., target_episode_truncation_rate=1.,
        target_mean_hits=STRICT_HITS,
        max_recent_mean_hit_vxy=STABILITY_LIMITS['ball_mean_m_s'],
        max_recent_rms_hit_vxy=STABILITY_LIMITS['ball_rms_m_s'])


def validate_anchor_contract(settings: Any, *, source_digest: str, replay_digest: str) -> None:
    if (source_digest != PARENT_SHA256 or replay_digest != REPLAY_SHA256
            or settings.actor_anchor_reference_checkpoint is None
            or settings.actor_anchor_replay_obs is None
            or settings.actor_anchor_kl_coef != 0.
            or settings.actor_anchor_replay_kl_coef != ANCHOR_COEFFICIENT
            or settings.actor_anchor_replay_max_kl != ANCHOR_BOUND):
        raise ValueError('V232 requires successful S5 replay and KL10/bound0.0005')


def validate_paired_stability_contract(settings: Any) -> None:
    if (settings.advance_mode != 'converged' or settings.advance_validation_mode != 'block'
            or settings.advance_retention_stage_name != RETENTION_STAGE
            or settings.advance_retention_final_stage_name not in (None, '')
            or settings.advance_eval_deterministic or settings.advance_eval_n_envs != 288
            or settings.advance_eval_steps != 1200):
        raise ValueError('V232 requires blocking288x1200 unconditional S5 retention')


def validate_training_contract(settings: Any) -> None:
    """Keep the reviewed batch and small true-failure actor focus explicit."""
    if (settings.n_envs != 1024 or settings.n_steps != 128
            or settings.failure_focus_hit_threshold != 14
            or settings.failure_focus_tail_steps != 128
            or settings.failure_focus_weight != 1.
            or settings.failure_focus_balanced_mass != .02):
        raise ValueError('V232 pins1024x128 and2% negative true-failure focus')


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool,
                            probe: bool = False) -> None:
    """Restore passed S5, but transfer none of its old survival graduations."""
    source = payload.get('args', {}).get('curriculum_profile')
    full = (not reset_optimizer and not reset_critic and not allow_obs_dim_migration
            and payload.get('opt') is not None
            and {'pi', 'v'}.issubset(payload.get('params', {}))
            and tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) == (72, 393, 7))
    if source == PARENT_PROFILE:
        row = payload.get('last_row', {})
        if (not full or resume_curriculum_state
                or requested_start_stage != (PROBE_STAGE if probe else START_STAGE)
                or payload.get('stage_name') != 'fplc_v225_s05_placed_xy_08mm'
                or payload.get('stage_index') != 5 or payload.get('stage_update') != 129
                or payload.get('step') != 1443758080 or int(payload['opt'].t) != 362007
                or any(float(row.get(k, 0.)) != 1. for k in
                       ('convergence/stage_converged', 'advance_eval/passed', 'retention_eval/passed'))):
            raise ValueError('V232 requires pinned S5 and fresh explicit full-course history')
        return
    if (not full or source != PROFILE or probe
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V232 rejects negative/capped/predecessor actors as parents')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V232 requires exact causal curriculum history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
