"""V252 preserves startup energy reserve without penalizing safe higher bounces."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from fixed_plane_v251_startup_energy_band import EnergyBandConfig

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v252'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v251'
START_STAGE='fplc_v252_s09_near_y_035mm_bridge'
RETENTION_STAGE='fplc_v252_s01_workspace_alignment'


@dataclass(frozen=True)
class EnergyReserveConfig(EnergyBandConfig):
    """Default-zero reward floor margins; no new observations or physical state."""
    v252_second_energy_band_margin_m: float=0.
    v252_third_energy_band_margin_m: float=0.


def validate_config(cfg: EnergyReserveConfig) -> None:
    from fixed_plane_v251_startup_energy_band import validate_config as previous
    previous(cfg)
    margins=(cfg.v252_second_energy_band_margin_m,cfg.v252_third_energy_band_margin_m)
    if any(not math.isfinite(x) or not 0.<=x<=.03 for x in margins):
        raise ValueError('V252 reward energy-band margins must be finite0..30mm')
    if any(margins) and not cfg.v251_startup_energy_band_enabled:
        raise ValueError('V252 reserve margins require the startup energy-band reward')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Restore15/25mm lower reserve while retaining the1.45m safe ceiling."""
    if len(source)!=17 or source[8].name!='fplc_v251_s09_near_y_035mm_bridge':
        raise ValueError('V252 requires all17 V251 lessons')
    result=[]
    for stage in source:
        cfg=replace(EnergyReserveConfig(**vars(stage.cfg)),
            v252_second_energy_band_margin_m=.015,
            v252_third_energy_band_margin_m=.025)
        validate_config(cfg)
        result.append(replace(stage,cfg=cfg,
            name=stage.name.replace('fplc_v251_','fplc_v252_',1),
            notes='V252 restores original15/25mm lower startup energy reserve: '
                  'phase1/2 actual-release Huber distance to1.264..1.45/1.425..1.45m. '
                  'Underpowered release loss matches originalV250 exactly; safe '
                  'higher startup releases are unpenalized inside those bands. '
                  'Firstconfirmed1.16..1.45 credit64, fourthplus1.425 Huber40000, '
                  'weak600,lateband240,lateXY28 and all17 original domains/gates '
                  'remain. Full latestV251actor/critic/Adam,72/393/7,uncapped '
                  'original1024x128 PPO; no sample rejection/rollback/reset.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v251_startup_energy_band import validate_training_contract as previous
    if settings.advance_retention_stage_name!=RETENTION_STAGE:
        raise ValueError('V252 retains original absolute nominal validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v251_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V252 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v251_', 'fplc_v252_', 1)
        if resume_curriculum_state or requested_start_stage != expected or int(payload.get('stage_index', 0)) < 9:
            raise ValueError('V252 starts new reward statistics in the current lesson')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V252 own continuation restores exact history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V252 rejects another source')
