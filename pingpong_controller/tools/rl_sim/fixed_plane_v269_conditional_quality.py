"""V269: reward conditional impact quality using frozen source references."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v267_motion_alignment as parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v269_conditional_quality'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v267_', 'v269_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=128, global_update=128, step=1642856448, adam_t=410020,
    sha256='12c5d9a29cdb151e18bed5accb3c22a4c9f54637fa46e9284a622cccead16a8f')
# Center [m], racket XY squared [(m/s)^2], angular squared [(rad/s)^2],
# adjacent apex jump [m], ball XY squared [(m/s)^2]. All frozen source samples.
REFERENCES = (0.0049677617108413894, 0.0034876512600135533,
              0.04047390770287627, 0.008364882527126206, 0.001138999451256442)
REWARD_CHANGES = {
    'fixed_plane_v158_late_squared_vxy_weight': 0.,
    'fixed_plane_v168_quality_weights': (6., 24., 24., 0., 0.),
    'fixed_plane_v168_quality_references': REFERENCES,
}
CHANGES = parent.CHANGES | REWARD_CHANGES


@dataclass(frozen=True)
class ConditionalQualityConfig(parent.MotionAlignmentConfig):
    """Opt-in coefficients for existing V168 rewards; no new physical state."""
    v269_enabled: bool = True
    fixed_plane_v168_quality_weights: tuple[float, ...] = (0.,) * 5
    fixed_plane_v168_quality_references: tuple[float, ...] = REFERENCES


def validate_config(cfg: ConditionalQualityConfig) -> None:
    """Reject duplicate XY cost, partial coefficient bundles and new references."""
    if cfg.v269_enabled is not True:
        raise ValueError('V269 requires its explicit reward marker')
    weights = tuple(cfg.fixed_plane_v168_quality_weights)
    pair = (cfg.fixed_plane_v158_late_squared_vxy_weight, weights)
    if pair not in ((3., (0.,) * 5), (0., REWARD_CHANGES['fixed_plane_v168_quality_weights'])):
        raise ValueError('V269 rejects mixed or duplicated conditional-quality prices')
    if tuple(cfg.fixed_plane_v168_quality_references) != REFERENCES:
        raise ValueError('V269 requires the frozen measured source references')
    fields = vars(cfg).copy()
    for key in ('v269_enabled', 'fixed_plane_v168_quality_weights', 'fixed_plane_v168_quality_references'):
        fields.pop(key)
    inherited = parent.MotionAlignmentConfig(**fields)
    if pair[0] == 0.:
        inherited = replace(inherited, fixed_plane_v158_late_squared_vxy_weight=6.)
    parent.validate_config(inherited)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve eleven physical lessons and gates, changing only S7+ rewards."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V269 requires the complete preserved V267 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = ConditionalQualityConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V269 experimental reward-only continuation from full V267S7u128. '
            'Replace late per-event XY-square6 with existing fixed-reference '
            'mean quality (center6, XY24, angular24). References are frozen '
            'two-seed source means, never online targets. Keep height, center4, '
            'adjacent, band/miss, variation.005 and global.6; retain all11 '
            'domains, control/observations/PPO and original rolling32/hold32, '
            'block288x1200/S1-retention gates. Restore full actor/critic/Adam '
            'with fresh same-S7 reward history; no graduation is implied.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin a complete ungraduated source; allow only causal own continuation."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V269 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V269 pins terminal V267S7u128 and fresh same-S7 reward history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V269 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V269 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V269 rejects another source profile')


validate_training_contract = parent.validate_training_contract
