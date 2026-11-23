"""V271: preserve V270 behavior and modestly relax two late DR tail gates."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

import fixed_plane_v270_angular_relaxation as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v271_dr_tail_tolerance'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v270_', 'v271_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=128, global_update=128, step=1668939776, adam_t=416244,
    sha256='385e3e45ed4bcf7db96501dabfa3c5e45b26246d9aa53da85160de7cfcf1be82')
GATE_CHANGES = {
    'max_recent_late_physical_below_floor_rate': .005,
    'min_recent_late_physical_in_band_rate': .9925,
}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep every reward/domain; change only S7--S11 height-tail tolerance."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V271 requires the complete preserved V270 course')
    return [replace(stage, name=name, **(GATE_CHANGES if index >= 6 else {}),
        notes=('V271 user-authorized gate-only continuation from full V270S7u128. '
            'S7--S11 low physical tail0.35->0.50%, in-band99.40->99.25%. '
            'S1--S6 gates and all rewards/control/physics/DR/observations/PPO '
            'remain V270 exact; retain high0.35%, hits/hit12/full, all motion/center/'
            'workspace gates, rolling32/hold32/block/S1 retention. '
            'Fresh same-S7 gate history, no forced advancement.'))
        for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True))]


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin the selected full state; never carry old gate history or skip S7."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V271 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V271 pins V270S7u128 and requires fresh same-S7 gate history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V271 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V271 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V271 rejects another source profile')


validate_training_contract = parent.validate_training_contract
