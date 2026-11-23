"""V272: protect late height tails with a stronger existing bilateral margin."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import fixed_plane_v271_dr_tail_tolerance as parent
import fixed_plane_v270_angular_relaxation as reward_parent

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v272_height_margin'
PARENT_PROFILE = parent.PROFILE
STAGE_NAMES = tuple(s.replace('v271_', 'v272_', 1) for s in parent.STAGE_NAMES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=51, global_update=51, step=1675624448, adam_t=417876,
    sha256='7c71571a2465879900593bc759e6d6bf56fc95e386554169c395d90905ca9022')
REWARD_CHANGES = {'fixed_plane_v191_margin_weight': 36.}
CHANGES = reward_parent.CHANGES | REWARD_CHANGES


@dataclass(frozen=True)
class HeightMarginConfig(reward_parent.AngularRelaxationConfig):
    """One existing reward coefficient, no new physical state or observation."""
    v272_enabled: bool = True


def validate_config(cfg: HeightMarginConfig) -> None:
    """Preserve the parent contract and require the phase-specific margin."""
    if cfg.v272_enabled is not True:
        raise ValueError('V272 requires its explicit reward marker')
    late = tuple(cfg.fixed_plane_v168_quality_weights) != (0.,) * 5
    if cfg.fixed_plane_v191_margin_weight != (36. if late else 6.):
        raise ValueError('V272 requires exact early6 or late36 height margin')
    fields = vars(cfg).copy()
    fields.pop('v272_enabled')
    inherited = reward_parent.AngularRelaxationConfig(**fields)
    if late:
        inherited = replace(inherited, fixed_plane_v191_margin_weight=24.)
    reward_parent.validate_config(inherited)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all eleven lessons; increase only the S7+ height-margin cost."""
    if tuple(s.name for s in source) != parent.STAGE_NAMES:
        raise ValueError('V272 requires the complete preserved V271 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = HeightMarginConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V272 reward-only continuation from full V271S7u51. Increase the '
            'existing bilateral height-margin coefficient24->36 at S7--S11. '
            'Zero extra cost within1.415--1.435m; both tails are penalized. '
            'Preserve all other rewards,11 physical lessons, gates, PPO and '
            'control/observation/reset semantics. Fresh reward history; '
            'rolling32/hold32 and independent block/S1-retention remain required. '
            'Formal training has no update cap; no graduation is implied.')))

    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin the complete partial source and prohibit forced stage advancement."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('V272 requires full72/398/7 actor/critic/Adam without reset')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7 or payload.get('stage_name') != parent.STAGE_NAMES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('global_update') != SOURCE['global_update']
                or payload.get('step') != SOURCE['step'] or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V272 pins selected V271S7u51 and fresh same-S7 reward history')
    elif source == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index - 1]:
            raise ValueError('V272 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V272 continuation requires exact causal stage history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V272 rejects another source profile')


validate_training_contract = parent.validate_training_contract
