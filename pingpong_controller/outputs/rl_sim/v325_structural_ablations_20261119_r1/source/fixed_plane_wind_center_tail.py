"""Bounded extra feedback on large offsets at confirmed paddle contacts."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)


MODE = 'clean10_dynamics_reference_center_negy_v13'
OFFSET_DEADBAND_M = .011
OFFSET_SPAN_M = .011
CLEAN_RATE_PER_S = 12.
WIND_RATE_PER_S = 6.
WARMUP_STEPS = 8192


def center_tail_contract() -> dict:
    """Record the single change from the original centering recipe."""
    return dict(
        base=BASE_MODE, offset_deadband_m=OFFSET_DEADBAND_M,
        offset_span_m=OFFSET_SPAN_M, capped_linear=True,
        rate_per_s=dict(clean=CLEAN_RATE_PER_S, wind=WIND_RATE_PER_S),
        timing='existing live recent confirmed-hit center and hit interval, capped at 1.2 s',
        shelter='reuse the original five-hit shelter and one-second ramp',
        warmup_policy_steps=WARMUP_STEPS,
        original_reward_clock='unchanged',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
    )


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCenterTailConfig', [
        ('wind_center_tail', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_center_tail_config(cfg: Any, enabled: bool) -> Any:
    """Keep the original configuration object on every other recipe."""
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def center_tail_cost(metrics: dict, *, clean: Any, gain: Any, phase_steps: Any, xp: Any):
    """Price only the portion of native contact offset beyond 11 mm."""
    hit = metrics['hit_event_count']
    offset_m = metrics['hit_contact_center_dist_sum'] / xp.maximum(hit, 1.)
    severity = xp.clip((offset_m - OFFSET_DEADBAND_M) / OFFSET_SPAN_M, 0., 1.)
    progress = xp.clip(phase_steps / WARMUP_STEPS, 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    live = (metrics['terminated'] < .5) & (metrics['dynamics/recent'] > .5)
    interval_s = xp.clip(metrics['dynamics/hit_interval_s'], 0., 1.2)
    rate = xp.where(clean, CLEAN_RATE_PER_S, WIND_RATE_PER_S)
    return xp.where(live & (hit > 0.),
                    -severity * interval_s * hit * rate * gain * warmup, 0.)


def center_tail_phase(payload: dict, sha256: str) -> int:
    """Restore the reviewed full learner and the added cost's independent phase."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_center_tail')
    return min(WARMUP_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)
