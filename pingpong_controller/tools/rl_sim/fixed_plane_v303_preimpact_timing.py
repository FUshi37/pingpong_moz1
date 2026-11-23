"""Opt-in early reference preparation and late physical preparation rewards."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v301_launch_center import LaunchCenterConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v303_preimpact_timing'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v302_preimpact_price'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v303_late_action_label_fade'})


@dataclass(frozen=True)
class PreimpactTimingConfig(LaunchCenterConfig):
    """Default-off reward repricing; no new observation or controller state."""
    v303_late_action_label_fade: bool = False


def validate_config(cfg: PreimpactTimingConfig) -> None:
    from fixed_plane_v301_launch_center import validate_config as previous
    previous(cfg)
    if type(cfg.v303_late_action_label_fade) is not bool:
        raise ValueError('V303 reward switch must be an explicit boolean')
    if cfg.v303_late_action_label_fade and (
            cfg.v294_critical_preparation_scale != 1.
            or cfg.v295_mid_near_preparation_scale != 1.
            or cfg.v296_action_prepare_weight_per_s != 512.
            or cfg.v297_feasible_reward_multiplier != 1.
            or cfg.v301_center_return_full_hits != 10):
        raise ValueError('V303 requires the exact V302 physical preparation context')


def late_label_correction(*, old_cost, ttc_s):
    """Reprice an existing (N,) cost, preserving its exact eligibility.

    Use the existing action-label diagnostic's pre-action ballistic TTC.
    Retain the original cost at >=90ms, smoothstep it down over90--60ms,
    and remove it below60ms. The separate actual-racket preparation costs
    remain intact. Correction never exceeds the old negative cost. Outside
    the physical positive-TTC window, or for inactive zero costs, add zero.
    """
    old, ttc = jnp.asarray(old_cost), jnp.asarray(ttc_s)
    if old.ndim != 1 or old.shape != ttc.shape:
        raise ValueError('V303 requires identically shaped(N,) cost and TTC')
    active = (old < 0.) & jnp.isfinite(ttc) & (ttc > 0.) & (ttc < .090)
    fraction = jnp.clip((ttc-.060)/.030, 0., 1.)
    retained = fraction*fraction*(3.-2.*fraction)
    correction = jnp.where(active, -old*(1.-retained), 0.)
    return correction, dict(effective_cost=old+correction,
        retained_fraction=jnp.where(active, retained, 1.),
        active=active.astype(jnp.float32))


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V303 requires the complete V302 course')
    result = []
    for index, stage in enumerate(source):
        cfg = PreimpactTimingConfig(**vars(stage.cfg),
            v303_late_action_label_fade=index >= 11)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V303 reward-only timing: keep reference-action preparation at '
            '>=90ms TTC, smoothly fade its existing cost over90--60ms,remove '
            'it below60ms so physical upward motion may coexist with reference '
            'braking. Use the existing exact pre-action label TTC. Both actual '
            'racket preparation weights remain1.0;all other rewards unchanged. '
            'Nominal/mirror/nonstartup,S1--S11,17 IDs/domains,gates,72/393/7, '
            'control,physics,DR,reset and numerical1024x128 PPO/BC/KL unchanged. '
            'Latest full actor/critic/Adam/history,all32,no rejection,rollback, '
            'rewind,reset,update cap or forced advancement. Native rolling, '
            'independent and strict nominal graduation required.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v302_preimpact_price import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *,
        requested_start_stage: str, resume_curriculum_state: bool,
        reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(key) for key in
                     ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V303 requires full uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        total = update-3833
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update
                or update < 3993
                or payload.get('step') != 3675258880+total*131072
                or int(payload['opt'].t) != 906750+total*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V303 requires latest V302 S12 full causal state '
                             'after at least160 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V303 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V303 rejects another source or from-scratch launch')
