"""V321: M04 continuation with uncapped hit credit and 1200-second episodes.

The four lessons change only the duration of the independent acceptance test.
They never shorten, reset or mix the training episodes at a lesson boundary.
"""
from collections import deque
from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

import numpy as np

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v321'
PARENT_SHA256 = '96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a'
XML_SHA256 = 'ace35229f4d643410a95481719c962e3b8ad172c2eba5901ee3d3e76e966ad8a'
GPU_UUID = 'GPU-e74458ec-002a-1bac-e0be-d0a5713b661e'
SURVIVAL_RATE = 40.0
MAX_HIT_GAP_S = 1.2
# Combo is disabled in M04; remove its stale 12 cap without introducing an
# unbounded per-hit combo multiplier. Both episode completion rewards are zero.
CONFIG_OVERRIDES = dict(horizon_sec=1200., hit_reward_cap_mode='off',
    hit_reward_count_cap=0, hit_combo_count_cap=0, hit_reward_combo=0.)


@dataclass(frozen=True)
class LongLesson:
    name: str
    seconds: float
    min_survival: float
    min_length_fraction: float


COURSE = tuple(LongLesson(f'fplc_v321_s{i}_{seconds}s', float(seconds),
    .95, .99) for i, seconds in enumerate((30, 120, 300, 1200), 1))


def make_config(parent):
    """Fail closed on dormant cap-dependent terms before changing the horizon."""
    if (parent.horizon_sec != 6. or parent.hit_reward_count_cap != 12
            or parent.hit_reward_cap_mode != 'fixed'
            or parent.fixed_contact_survival_reward_weight != SURVIVAL_RATE
            or parent.post_hit_joint_apex_lift_reward_weight != 0.
            or parent.full_episode_completion_reward != 0.
            or parent.full_episode_joint_height_reward_weight != 0.
            or parent.hit_reward_combo != 0.
            or parent.episode_phase_stagger_min_frac != 1.
            or parent.terminate_after_confirmed_hits != 0):
        raise ValueError('V321 requires the exact M04 reward/horizon contract')
    return replace(parent, **CONFIG_OVERRIDES)


def survival_reward(hit_count, age_s, done, dt_s, *, xp):
    """Pay 40 per second after launch while a confirmed bounce is recent.

The 1.2 s grace exceeds twice the nominal 0.5 s cycle; it prevents indefinite
standing/sticking credit without imposing a new termination or cadence target.
"""
    active = ((hit_count > 0) & (age_s >= 0.)
              & (age_s <= MAX_HIT_GAP_S) & ~done)
    return xp.where(active, SURVIVAL_RATE * dt_s, 0.)


class EndpointWindow:
    """One outcome per episode at a deadline or earlier failure, bounded RAM.

This is mixed-policy training telemetry, not frozen-policy evaluation evidence.
Ongoing episodes younger than the deadline are explicitly not resolved yet.
"""
    def __init__(self, seconds: float, dt_s: float, capacity: int = 4096):
        self.steps = round(seconds / dt_s)
        self.outcomes = deque(maxlen=capacity)

    def add(self, length, hits, done, recent, *, terminated=None):
        terminal = done if terminated is None else terminated
        selected = (length == self.steps) | (done & (length < self.steps))
        success = (length >= self.steps) & ~terminal & recent
        # Native time-limit truncation at 1200 s is success; physical failure
        # on that same step is failure even if length reached the deadline.
        for t, h, s in zip(length[selected], hits[selected], success[selected]):
            self.outcomes.append((min(float(t) / self.steps, 1.), float(h), bool(s)))

    def summary(self) -> dict:
        if not self.outcomes:
            return {'n': 0}
        values = np.asarray(self.outcomes, dtype=float)
        return dict(n=len(values), len_fraction=float(values[:, 0].mean()),
                    hits_mean=float(values[:, 1].mean()),
                    survival=float(values[:, 2].mean()))


QUALITY_FIELDS = (
    'rapid_steady_steady_events', 'rapid_steady_steady_apex_sum_z_m',
    'rapid_steady_steady_racket_vxy_sq_sum',
    'rapid_steady_steady_racket_omega_sq_sum',
    'v92_late_contact_event', 'v92_in_1p4_1p45_event',
    'v80_late_pair_event', 'v80_late_pair_jump_sum_m',
    'hit_event_count', 'hit_contact_center_dist_sum',
    'recurrent_hit_event_count', 'recurrent_hit_racket_vxy_sum',
    'recurrent_hit_racket_vxy_sq_sum', 'hit_vxy_sum', 'hit_vxy_sq_sum',
    'contact_edge_to_hit_numerator', 'contact_edge_to_hit_denominator',
)


def quality_summary(sums: dict) -> dict:
    """Use event-weighted physical outcome statistics, with absent data omitted."""
    result = {}
    fields = (
        ('apex_m', 'rapid_steady_steady_apex_sum_z_m', 'rapid_steady_steady_events', False),
        ('racket_vxy_rms_m_s', 'rapid_steady_steady_racket_vxy_sq_sum', 'rapid_steady_steady_events', True),
        ('racket_omega_rms_rad_s', 'rapid_steady_steady_racket_omega_sq_sum', 'rapid_steady_steady_events', True),
        ('in_band', 'v92_in_1p4_1p45_event', 'v92_late_contact_event', False),
        ('adjacent_jump_m', 'v80_late_pair_jump_sum_m', 'v80_late_pair_event', False),
        ('center_m', 'hit_contact_center_dist_sum', 'hit_event_count', False),
        ('recurrent_vxy_m_s', 'recurrent_hit_racket_vxy_sum', 'recurrent_hit_event_count', False),
        ('recurrent_vxy_rms_m_s', 'recurrent_hit_racket_vxy_sq_sum', 'recurrent_hit_event_count', True),
        ('ball_vxy_m_s', 'hit_vxy_sum', 'hit_event_count', False),
        ('ball_vxy_rms_m_s', 'hit_vxy_sq_sum', 'hit_event_count', True),
    )
    for name, numerator, denominator, root in fields:
        if sums.get(denominator, 0.) > 0 and numerator in sums:
            value = sums[numerator] / sums[denominator]
            result[name] = math.sqrt(max(0., value)) if root else value
    result['steady_events'] = sums.get('rapid_steady_steady_events', 0.)
    return result


def quality_passed(q: dict) -> bool:
    """Declared long-run tolerances, close to the M04 S10 measured motion."""
    limits = dict(racket_vxy_rms_m_s=.055, racket_omega_rms_rad_s=.28,
                  center_m=.0055, adjacent_jump_m=.012,
                  recurrent_vxy_m_s=.05, recurrent_vxy_rms_m_s=.055,
                  ball_vxy_m_s=.035, ball_vxy_rms_m_s=.045)
    return (q.get('steady_events', 0.) >= 100
            and 1.40 <= q.get('apex_m', 0.) <= 1.45
            and q.get('in_band', 0.) >= .99
            and all(math.isfinite(q.get(k, math.inf))
                    and q[k] <= limit for k, limit in limits.items()))


class CheckpointRetention:
    """Delete only owned periodic candidates; converged models are immutable."""
    def __init__(self, directory: Path, cleanup_every: int = 5):
        self.directory = directory.resolve()
        self.cleanup_every = cleanup_every
        self.saved = {}
        self.since_cleanup = 0
        self.best = None

    def register(self, path: Path, score: tuple) -> None:
        if (path.is_symlink() or path.parent.resolve() != self.directory
                or not re.fullmatch(r'periodic_u\d{8,}\.pkl', path.name)
                or not all(math.isfinite(float(s)) for s in score)):
            raise ValueError('Refusing checkpoint cleanup outside owned candidates')
        if not path.is_file():
            raise ValueError('Checkpoint must be atomically saved before pruning')
        self.saved[path] = tuple(score)
        self.since_cleanup += 1
        self.best = max(self.saved, key=lambda p: (self.saved[p], p.name))
        if self.since_cleanup >= self.cleanup_every:
            self.prune()

    def prune(self) -> None:
        if self.best is None:
            return
        for path in list(self.saved):
            if path != self.best:
                if path.is_symlink():
                    raise ValueError('Owned candidate changed into a symlink')
                path.unlink()
                del self.saved[path]
        self.since_cleanup = 0
