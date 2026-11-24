"""Optional signed racket-height braking during the first recovered strokes."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_lower_guard import lower_stop_cost
from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)

MODE = 'clean10_dynamics_reference_center_negy_v15'
WARMUP_STEPS = 8192
EARLY_HIT_RANGE = (5, 16)
RATE_PER_S = 2.
DEADBAND = .25


def early_recovery_contract() -> dict:
    """Record the narrow change to the original v2 reward."""
    return dict(base=BASE_MODE, confirmed_hits=list(EARLY_HIT_RANGE),
        targeted_edge={'clean': 'lower', 'wind': 'upper'},
        preview='existing 200 ms lower-stop cost; mirror height and speed for upper',
        rate_per_s=RATE_PER_S, squared_cost_deadband=DEADBAND,
        active='live native full-flight recovery after confirmation and outside imminent contact',
        warmup_policy_steps=WARMUP_STEPS, original_reward_clock='unchanged',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15))


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindEarlyRecoveryConfig', [
        ('wind_early_recovery', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_early_recovery_config(cfg: Any, enabled: bool) -> Any:
    """Keep prior recipe configs exact when the new cost is disabled."""
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def recovery_active(*, confirmed_hits, recent, recovery_phase, pending_before,
                    pending_after, done):
    """Limit the added reward to the measured early post-release phase."""
    first, last = EARLY_HIT_RANGE
    return ((confirmed_hits >= first) & (confirmed_hits <= last)
            & recent & recovery_phase & ~pending_before & ~pending_after & ~done)


def recovery_cost(z_m, vz_m_s, bounds_m, *, windy, active, gain, phase_steps,
                  dt_s: float, xp: Any):
    """Apply a bounded per-time cost to motion toward that group's risk edge."""
    lower_m, upper_m = bounds_m
    clearance_m = xp.where(windy, upper_m - z_m, z_m - lower_m)
    clearance_rate_m_s = xp.where(windy, -vz_m_s, vz_m_s)
    raw = lower_stop_cost(clearance_m, clearance_rate_m_s, 0., xp=xp)
    excess = xp.maximum(raw - DEADBAND, 0.)
    progress = xp.clip(phase_steps / WARMUP_STEPS, 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    return xp.where(active, -dt_s * RATE_PER_S * gain * warmup * excess, 0.)


def reward_phase(payload: dict, sha256: str) -> int:
    """Authenticate the original full learner or a compatible reward resume."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_early_recovery')
    return min(WARMUP_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)
