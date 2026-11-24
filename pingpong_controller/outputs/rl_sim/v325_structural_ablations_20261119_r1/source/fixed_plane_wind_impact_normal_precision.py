"""Default-off feasible normal alignment at real physical contact edges."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)


MODE = 'clean10_dynamics_reference_center_negy_v19'
WARMUP_STEPS = 8192


def impact_normal_contract() -> dict:
    """Supplement the original task objective with bounded impact alignment."""
    return dict(base=BASE_MODE, revision='feasible_physical_edge_normal_precision_v1',
        rate_per_s=8., sigma_deg=1., loss_cap=8., interval_cap_s=1.2,
        target='existing normal from actual incoming and physical-height outgoing reference',
        active='physical edge,recent/live,>=5confirmed hits,valid/feasible reference',
        populations='clean and every allowed wind mode through4m/s',
        gain='original five-hit shelter and1second ramp',
        warmup_policy_steps=WARMUP_STEPS, original_reward_clock='unchanged at birth',
        own_resume='preserve original precision and new-cost reward phases only',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
        preserved='all original rewards,physics,observations,control,PPO,sampling,gates;no BC')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindImpactNormalPrecisionConfig', [
        ('wind_impact_normal_precision', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_impact_normal_config(cfg: Any, enabled: bool) -> Any:
    """Retain exact configuration identity when the new recipe is disabled."""
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def impact_normal_active(*, edge: Any, confirmed_hits: Any, recent: Any,
                         done: Any, reference_valid: Any, target_tilt_deg: Any,
                         available_tilt_deg: Any, xp: Any) -> Any:
    """Exclude shelter, incomplete references and unavailable pose targets."""
    feasible = (xp.isfinite(target_tilt_deg) & xp.isfinite(available_tilt_deg)
                & (target_tilt_deg >= 0.) & (target_tilt_deg <= available_tilt_deg))
    return edge & (confirmed_hits >= 5) & recent & ~done & reference_valid & feasible


def impact_normal_cost(error_deg: Any, *, active: Any, gain: Any, interval_s: Any,
                       reward_steps: Any, xp: Any) -> Any:
    """Return bounded nonpositive [N] event costs, normalized by physical time."""
    finite = xp.isfinite(error_deg)
    error = xp.where(finite, xp.abs(error_deg), 0.)
    huber = .5 * xp.minimum(error, 1.)**2 + xp.maximum(error - 1., 0.)
    progress = xp.clip(reward_steps / WARMUP_STEPS, 0., 1.)
    warmup = progress**2 * (3. - 2. * progress)
    cost = -8. * xp.clip(interval_s, 0., 1.2) * xp.minimum(huber, 8.)
    return xp.where(active & finite, xp.clip(gain, 0., 1.) * warmup * cost, 0.)


def normal_reward_phase(payload: dict, sha256: str) -> int:
    """Authenticate the complete original learner or its exact own continuation."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_impact_normal_precision')
    cfg = payload.get('env_cfg', {})
    incompatible = (
        'wind_clean_group_priority', 'wind_normalized_group_priority',
        'wind_clean_apex_priority', 'wind_precontact_center', 'wind_precontact_balance',
        'wind_release_precision', 'wind_cycle_release_precision',
        'wind_measured_precision', 'wind_steady_braking',
        'wind_confirmed_launch_braking', 'wind_center_tail', 'wind_calm_release',
        'wind_early_recovery', 'wind_mature_precision_reward', 'wind_calm_return_height',
    )
    if any(cfg.get(flag, False) for flag in incompatible):
        raise ValueError('Impact normal precision cannot stack a different reward recipe')
    return min(WARMUP_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)
