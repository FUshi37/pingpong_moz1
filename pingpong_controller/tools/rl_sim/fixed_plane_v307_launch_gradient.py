"""Opt-in broadening of only the under-target first-launch reward tail."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence
import jax.numpy as jnp
from fixed_plane_v306_first_launch_reserve import FirstLaunchReserveConfig
PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v307_launch_gradient'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v306_first_launch_reserve'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v307_first_launch_low_sigma_m'})

@dataclass(frozen=True)
class LaunchGradientConfig(FirstLaunchReserveConfig):
    """Identity default;no state,observation,controller or target changes."""
    v307_first_launch_low_sigma_m: float = 0.025

def validate_sigma(value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or (not 0.025 <= value <= 0.05):
        raise ValueError('V307 low-energy reward sigma must be finite25--50mm')

def low_tail_correction(*, old_effective_credit, first_event, first_apex_m, support, terminated, low_sigma_m: float, weight: float):
    """Add only the nonnegative low-tail credit difference on real(N,) events.

    Keep the parent's1.20--1.35m band,512cap and25mm upper tail exactly.
    Parent first-event/support/terminal tensors remain auditable and unchanged.
    No extra hit,progress counter or reference command is created.
    """
    validate_sigma(low_sigma_m)
    if isinstance(weight, bool) or not math.isfinite(weight) or weight != 512.0:
        raise ValueError('V307 preserves the existing512 first-credit cap')
    old = jnp.asarray(old_effective_credit)
    values = [jnp.asarray(x) for x in (first_event, first_apex_m, support, terminated)]
    if old.ndim != 1 or any((x.shape != old.shape for x in values)):
        raise ValueError('V307 requires identical(N,) reward tensors')
    event, apex, scale, terminal = values
    deficit = jnp.maximum(1.2 - apex, 0.0)
    narrow = weight * jnp.exp(-0.5 * jnp.square(deficit / 0.025))
    broad = weight * jnp.exp(-0.5 * jnp.square(deficit / low_sigma_m))
    active = (event > 0.5) & (scale > 0.0) & ~terminal.astype(bool) & (apex < 1.2)
    correction = jnp.where(active, scale * (broad - narrow), 0.0)
    return (correction, dict(active=active.astype(jnp.float32), effective_credit=old + correction, first_apex_m=jnp.where(active, apex, 0.0)))

def validate_config(cfg: LaunchGradientConfig) -> None:
    from fixed_plane_v306_first_launch_reserve import validate_config as previous
    previous(cfg)
    validate_sigma(cfg.v307_first_launch_low_sigma_m)
    if cfg.v307_first_launch_low_sigma_m != 0.025 and cfg.v306_near_first_launch_credit != 512.0:
        raise ValueError('V307 requires the unchanged V306 first-event ledger')

def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V307 requires all17 unchanged V306 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = LaunchGradientConfig(**vars(stage.cfg), v307_first_launch_low_sigma_m=0.04 if index >= 11 else 0.025)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if index < 11 else 'V307 reward-only under-target first-launch sigma25->40mm. Use exact V306 actual-first-event/support/terminal tensors. Only below1.20m add the nonnegative Gaussian credit difference; preserve target band1.20--1.35m,512cap,25mm upper tail and all other rewards. First11 lessons,all17 IDs/domains/gates,72/393/7, q_ref,RMP/PD,physics,DR,reset and numeric1024x128 PPO/BC/KL remain exact. Latest full actor/critic/Adam/raw history;all32, no rejection,rollback,rewind,reset,update cap or forced advancement.'))
    return result

def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v306_first_launch_reserve import validate_training_contract as previous
    previous(settings)

def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str, resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool, allow_obs_dim_migration: bool) -> None:
    if reset_optimizer or reset_critic or allow_obs_dim_migration or (payload.get('opt') is None) or (not {'pi', 'v'}.issubset(payload.get('params', {}))) or (tuple((payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))) != (72, 393, 7)) or (payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V307 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        u = int(payload.get('global_update', 0))
        total = u - 4737
        tail = payload.get('stage_history_tail')
        if not resume_curriculum_state or requested_start_stage != START_STAGE or payload.get('stage_name') != START_STAGE or (payload.get('stage_index') != 12) or (int(payload.get('stage_update', 0)) != u) or (u < 4865) or (payload.get('step') != 3793747968 + total * 131072) or (int(payload['opt'].t) != 935678 + total * 32) or (not isinstance(tail, list)) or (not tail) or (int(tail[-1]['stage_update']) != u) or (tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V307 requires latest V306 S12 full history after>=128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V307 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V307 rejects another source or from-scratch launch')
