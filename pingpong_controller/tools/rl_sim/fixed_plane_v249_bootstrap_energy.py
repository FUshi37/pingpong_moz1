"""V249 tests first-launch energy reserve with reward-only targets."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from fixed_plane_v247_physical_risk import PhysicalRiskConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v249'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v248'
START_STAGE = 'fplc_v249_s09_near_y_035mm_bridge'
RETENTION_STAGE = 'fplc_v249_s01_workspace_alignment'


@dataclass(frozen=True)
class BootstrapEnergyConfig(PhysicalRiskConfig):
    """Default-zero reward target offset; never a physical/observed goal."""

    v249_first_apex_reward_margin_m: float = 0.


def first_apex_reward_target(base_goal_m: Any, *, margin_m: float) -> Any:
    """Offset only the input of the existing confirmed-first-hit reward."""
    if not math.isfinite(margin_m) or not 0. <= margin_m <= .025:
        raise ValueError('V249 first reward margin must be finite in[0,.025]m')
    if margin_m == 0.:
        return base_goal_m
    return base_goal_m + margin_m


def validate_config(cfg: BootstrapEnergyConfig) -> None:
    from fixed_plane_v247_physical_risk import validate_config as previous
    previous(cfg)
    first_apex_reward_target(1.14, margin_m=float(cfg.v249_first_apex_reward_margin_m))
    if cfg.v249_first_apex_reward_margin_m and (
            not cfg.v225_enabled or not cfg.v242_startup_vz_outcome_only
            or cfg.fixed_plane_v94_first_hit_apex_reward_weight <= 0.
            or cfg.fixed_plane_v94_first_hit_apex_sigma_m <= 0.):
        raise ValueError('V249 requires the preserved confirmed-first-hit task')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Add first-launch reward reserve and price late horizontal speed."""
    if len(source) != 17 or source[7].name != 'fplc_v248_s08_near_y_020mm_bridge':
        raise ValueError('V249 requires the complete V248 curriculum')
    result = []
    for stage in source:
        cfg = replace(BootstrapEnergyConfig(**vars(stage.cfg)),
                      v249_first_apex_reward_margin_m=.020,
                      fixed_plane_v94_first_hit_apex_sigma_m=.015,
                      v241_late_ball_xy_weight=28.)
        validate_config(cfg)
        result.append(replace(
            stage, cfg=cfg,
            name=stage.name.replace('fplc_v248_', 'fplc_v249_', 1),
            notes='V249 tests first-confirmed-apex reward target1.16m '
                  '(physical goal remains1.14m), sigma0.2->0.015m and '
                  'late ball-XY extra20->28. Keep all other V248 rewards, '
                  '17 domains/gates,conv_len1,72/393/7,control/physics/DR/reset '
                  'and every original uncapped1024x128 PPO setting. Restore '
                  'latest full V248 actor/critic/Adam and genuine graduations. '
                  'First static separation and later hits get no new credit.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v248_startup_band_weights import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V249 retains nominal absolute validation')
    previous(SimpleNamespace(**{
        **vars(settings),
        'advance_retention_stage_name': 'fplc_v248_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))
            != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V249 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v248_', 'fplc_v249_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V249 starts new reward statistics in the current lesson')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V249 own continuation must restore exact history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V249 rejects another source')
