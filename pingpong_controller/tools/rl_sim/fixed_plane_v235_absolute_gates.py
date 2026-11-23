"""V235 graduates exclusively by explicit absolute task and safety gates."""

from dataclasses import replace
import math
from types import SimpleNamespace

from fixed_plane_v234_reward_only import (
    validate_anchor_contract, PINNED_TRAINING,
    validate_training_contract as validate_previous_training,
)
from fixed_plane_v232_full_survival import full_episode_gate, retention_stage

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v235'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v234'
RETENTION_STAGE = 'fplc_v235_s01_workspace_alignment'


def build_stages(source):
    """Retain every reward, domain and absolute threshold from V234."""
    if len(source) != 17 or source[0].name != 'fplc_v234_s01_workspace_alignment':
        raise ValueError('V235 requires the complete V234 course')
    return [replace(stage,
        name=stage.name.replace('fplc_v234_', 'fplc_v235_', 1),
        notes='V235 preserves V234 rewards, physics/control/observation/DR, '
        'PPO and every explicit absolute task/nominal/safety threshold. '
        'Graduation evaluates only the current actor against those thresholds; '
        'no historical actor, reference delta or noninferiority condition. '
        'Keep conv_len1, unconditional full1200 proof, all17 lessons and '
        'uncapped learning. Resume the current lesson with full Adam/history; '
        'do not restart S1 or invent prior graduations.') for stage in source]


def validate_training_contract(settings):
    """Keep the prior numeric settings and validation domains exactly."""
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V235 retains the explicit nominal absolute-gate domain')
    proxy = SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v234_s01_workspace_alignment'})
    validate_previous_training(proxy)


def relaxed_hit_admission(task, retention):
    """Keep the existing hit floor conditioned on complete absolute proofs."""
    hits = float(task.get('advance_eval/mean_hits', float('nan')))
    required = ((task, 'advance_eval/passed'),
                (task, 'advance_eval/v232_full_episode_passed'),
                (retention, 'retention_eval/passed'),
                (retention, 'retention_eval/v232_full_episode_passed'))
    return (math.isfinite(hits) and hits >= 12.85
            and all(float(row.get(key, 0.)) == 1. for row, key in required))


def relabel_resume_history(payload):
    """Relabel metadata only, preserving every learned array and raw sample."""
    if payload.get('args', {}).get('curriculum_profile') != PARENT_PROFILE:
        return payload
    result = dict(payload)
    def relabel(row):
        changed = dict(row)
        name = str(changed.get('stage_name', ''))
        if not name.startswith('fplc_v234_'):
            raise ValueError('V235 history requires an explicit V234 stage identity')
        changed['stage_name'] = name.replace('fplc_v234_', 'fplc_v235_', 1)
        return changed
    result['stage_name'] = relabel(payload)['stage_name']
    for key in ('stage_history_tail', 'committed_stage_history_tail'):
        if key in payload:
            result[key] = [relabel(row) for row in payload[key]]
    if isinstance(payload.get('last_row'), dict):
        result['last_row'] = relabel(payload['last_row'])
    # Old validation decisions included a removed relative gate. Re-evaluate
    # the unchanged actor under the new rule instead of inheriting a pass.
    result['last_advance_eval_update'] = -10**9
    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic, allow_obs_dim_migration):
    source = payload.get('args', {}).get('curriculum_profile')
    if (source not in (PARENT_PROFILE, PROFILE) or reset_optimizer or reset_critic
            or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V235 requires the current unreset full-state continuation')
    if source == PARENT_PROFILE:
        if not resume_curriculum_state or not payload.get('stage_history_tail'):
            raise ValueError('V235 must carry the current V234 stage/counters/history')
        expected = str(payload.get('stage_name', '')).replace('fplc_v234_', 'fplc_v235_', 1)
        if requested_start_stage != expected:
            raise ValueError('V235 cannot restart an earlier lesson or skip the current one')
    elif resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V235 requires exact causal stage history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
