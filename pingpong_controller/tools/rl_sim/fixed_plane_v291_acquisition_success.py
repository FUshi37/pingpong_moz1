"""Default-off real acquisition credit with true-failure clawback."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v290_startup_potential import StartupPotentialConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v291_acquisition_success'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v290_startup_potential'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v291_near_acquisition_credit'})


@dataclass(frozen=True)
class AcquisitionSuccessConfig(StartupPotentialConfig):
    """One reward weight; no contact counter, observation or state extension."""
    v291_near_acquisition_credit: float = 0.


def acquisition_reward(*, previous_hits, next_hits, terminated,
        group, initial_y_m, weight):
    """Credit real phase2->3 once; reclaim on true failure, retain at timeout.

    All arrays have shape(N,). The original smooth50--65mm hard-near support
    is retained. Counters are read only. A failed life that reached hit3 has
    zero undiscounted extra credit. Discounted residual grows with actual
    post-acquisition survival; no failed-life discount-zero claim is made.
    """
    previous, following = jnp.asarray(previous_hits), jnp.asarray(next_hits)
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    support = fraction * fraction * (3. - 2. * fraction)
    support = jnp.where(jnp.asarray(group) == 1, support, 0.)
    credit = (previous < 3) & (following >= 3) & (support > 0.) & (weight > 0.)
    clawback = (jnp.asarray(terminated, dtype=bool) & (following >= 3)
                & (support > 0.) & (weight > 0.))
    reward = weight * support * (credit.astype(jnp.float32) - clawback.astype(jnp.float32))
    return reward, dict(credit_event=credit.astype(jnp.float32),
        clawback_event=clawback.astype(jnp.float32),
        event_weight=weight * support)


def validate_config(cfg: AcquisitionSuccessConfig) -> None:
    from fixed_plane_v290_startup_potential import validate_config as previous
    previous(cfg)
    weight = cfg.v291_near_acquisition_credit
    if not math.isfinite(weight) or not 0. <= weight <= 4096.:
        raise ValueError('V291 acquisition credit must be finite in[0,4096]')
    if weight > 0. and (not cfg.v225_enabled or cfg.v290_startup_potential_weight <= 0.):
        raise ValueError('V291 requires the original physical near-side/potential course')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V291 requires all17 physical V290 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = AcquisitionSuccessConfig(**vars(stage.cfg),
            v291_near_acquisition_credit=1024. if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V291 reward-only true near-side acquisition priority. Original '
            'group1/Y50--65mm support: actual confirmed count transition to3 '
            'receives1024 extra credit once; any later true failure reclaims '
            'the same credit, including simultaneous third/failure. Natural '
            'timeout retains credit. Counts and all gates are unchanged. '
            'Keep V290 potential and every other reward,17 V281 domains, '
            '72/393/7,control/physics/DR/reset and all numeric PPO/BC/KL. '
            'Latest full state/history, all32 minibatches, no cap/reset/rewind/'
            'rejection/rollback or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v290_startup_potential import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V291 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 2000
                or payload.get('step') != 3272605696 + (update - 761) * 131072
                or int(payload['opt'].t) != 808446 + (update - 761) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V291 requires latest complete stalled V290 S12 after update2000')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V291 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V291 rejects another source or from-scratch launch')
