"""V273: calibrated late-height tolerance with V272 behavior preserved."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

import fixed_plane_v272_height_margin as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v273_gate_calibration'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v272_', 'v273_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=87, global_update=87, step=1687027712, adam_t=420660,
    sha256='9a71f2c539c3de047f28c7a1cdd3931f677f630b70a3e51a331315d87a30076b')
GATE_CHANGES = {
    'max_recent_late_physical_below_floor_rate': .0075,
    'min_recent_late_physical_in_band_rate': .99,
}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Retain eleven lessons and adjust only S7--S11 late-height gates."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V273 requires the complete preserved V272 course')
    return [replace(stage, name=name, **(GATE_CHANGES if index >= 6 else {}),
        notes=('V273 user-authorized height-tail gate calibration. '
            'S7--S11 low tail0.50->0.75%, in-band99.25->99.00%. '
            'Preserve V272 rewards, all11 physical lessons, S1--S6 gates, '
            'high0.35%, success/survival/motion/contact/workspace limits, '
            'rolling32/hold32 and independent block/S1 retention. '
            'Full actor/critic/Adam; fresh same-S7 gate history. '
            'Uncapped formal training; no retrospective graduation.'))
        for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True))]


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Restore the tested full parent; reject stale gates and forced advancement."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V273 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V273 pins the tested V272 source and fresh same-S7 gate history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V273 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V273 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V273 rejects another source profile')


validate_training_contract = parent.validate_training_contract
