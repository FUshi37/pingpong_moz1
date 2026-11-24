"""Default-off midpoint restoration during detached calm recovery."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)
from fixed_plane_wind_tail_precision import height_recovery_cost


MODE = 'clean10_dynamics_reference_center_negy_v18'
WARMUP_STEPS = 8192


def calm_return_contract() -> dict:
    """Reuse the existing wind midpoint cost without its collision-time mask."""
    return dict(base=BASE_MODE, revision='calm_detached_midpoint_return_v1',
        rate_per_s=2., height_sigma_m=.025, squared_cost_cap=9.,
        target='existing native racket-Z bounds midpoint',
        active='calm,live,recent,>=5confirmed hits,detached recovery,no pending or imminent contact',
        later_hits='all later ordinals; no hit16 cutoff',
        gain='original five-hit shelter and1second ramp',
        warmup_policy_steps=WARMUP_STEPS, original_reward_clock='unchanged at birth',
        own_resume='preserve both original precision and added-cost reward phase only',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
        preserved='all original wind and clean rewards; physics,observations,control,PPO,sampling,gates')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCalmReturnConfig', [
        ('wind_calm_return_height', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_calm_return_config(cfg: Any, enabled: bool) -> Any:
    """Add a reward-only switch; leave every historical config object exact."""
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def return_active(*, confirmed_hits: Any, recent: Any, recovery_phase: Any,
                  pending_before: Any, pending_after: Any, done: Any) -> Any:
    """Protect both real collisions and the unresolved confirmation interval."""
    return ((confirmed_hits >= 5) & recent & recovery_phase
            & ~pending_before & ~pending_after & ~done)


def return_height_cost(z_m: Any, target_z_m: Any, *, calm: Any, active: Any,
                       gain: Any, reward_steps: Any, dt_s: float, xp: Any) -> Any:
    """Return a bounded nonpositive cost [N] during free recovery only."""
    progress = xp.clip(reward_steps / WARMUP_STEPS, 0., 1.)
    warmup = progress**2 * (3. - 2. * progress)
    cost = height_recovery_cost(z_m, target_z_m, dt_s=dt_s, xp=xp)
    return xp.where(calm & active, -gain * warmup * cost, 0.)


def return_reward_phase(payload: dict, sha256: str) -> int:
    """Restore an authenticated reward phase alongside the complete learner."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_calm_return_height')
    cfg = payload.get('env_cfg', {})
    incompatible = ('wind_normalized_group_priority', 'wind_mature_precision_reward',
                    'wind_clean_apex_priority', 'wind_early_recovery', 'wind_calm_release',
                    'wind_steady_braking', 'wind_precontact_center', 'wind_center_tail',
                    'wind_measured_precision', 'wind_confirmed_launch_braking')
    if any(cfg.get(flag, False) for flag in incompatible):
        raise ValueError('Calm midpoint return cannot stack a different reward recipe')
    return min(WARMUP_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)
