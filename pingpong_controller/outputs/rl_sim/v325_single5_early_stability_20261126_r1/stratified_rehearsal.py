"""Stratified 20s launch & early-juggling rehearsal across all 27 X-wind cells.

Fixes the lane-overlap issue between StartupRehearsalEnv (which took lanes 0..N-1,
i.e. clean only) and XWindEnv.layout (which places clean in lanes 0..101).
Here, half of every exact (mode, sector, bin, axis, hard) cell runs 20s early-juggling
rehearsal with unbiased time_limit_bootstrap truncation, while the other half runs
full 1200s episodes. First-cycle rehearsal cutoffs are staggered across [2.56s, 20s]
so every PPO update immediately observes launch, wind-onset, and early-juggling states.
"""
from collections import deque, namedtuple
import math
import jax.numpy as jnp
import numpy as np

from xwind_sampling import layout as xwind_layout, labels_for_wind

FIELDS = (
    "sampling/rehearsal_lane",
    "sampling/rehearsal_cutoff",
    "sampling/rehearsal_full_horizon",
)

RehearsalState = namedtuple("StratifiedRehearsalState", ("core", "rehearsal_cutoff_steps"))


class _RehearsalStateWrapper(RehearsalState):
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(self.core, name)

    def _replace(self, **values):
        local = {k: v for k, v in values.items() if k in self._fields}
        fwd = {k: v for k, v in values.items() if k not in self._fields}
        if fwd:
            local["core"] = self.core._replace(**fwd)
        return super()._replace(**local)


def stratified_rehearsal_mask(n_envs: int, fraction: float = 0.5) -> np.ndarray:
    """Select `fraction` of lanes within every exact XWindEnv cell."""
    if not (0.0 < fraction < 1.0):
        raise ValueError("fraction must be in (0, 1)")
    mode, sector, bins, axis, hard = xwind_layout(n_envs)
    groups = {}
    for lane in range(n_envs):
        key = (int(mode[lane]), int(sector[lane]), int(bins[lane]), int(axis[lane]), bool(hard[lane]))
        groups.setdefault(key, []).append(lane)
    mask = np.zeros(n_envs, dtype=bool)
    for key in sorted(groups):
        lanes = groups[key]
        k = max(1, min(len(lanes) - 1, int(round(len(lanes) * fraction)))) if len(lanes) > 1 else 0
        chosen_idx = np.linspace(0, len(lanes) - 1, k, dtype=int) if k > 0 else []
        for idx in chosen_idx:
            mask[lanes[idx]] = True
    return mask


def initial_staggered_cutoffs(rehearsal_mask: np.ndarray, full_cutoff_steps: int,
                              min_steps: int = 512, seed: int = 2026112602) -> np.ndarray:
    """Assign evenly spaced first-cycle truncation horizons across [min_steps, full_cutoff_steps]."""
    cutoffs = np.full(len(rehearsal_mask), full_cutoff_steps, dtype=np.int32)
    idx = np.flatnonzero(rehearsal_mask)
    if len(idx) > 0:
        grid = np.linspace(min_steps, full_cutoff_steps, len(idx)).round().astype(np.int32)
        np.random.default_rng(seed).shuffle(grid)
        cutoffs[idx] = grid
    return cutoffs


class StratifiedStartupRehearsalEnv:
    """Censor stratified rehearsal lanes at `seconds` with unbiased timeout bootstrapping."""

    def __init__(self, inner, *, fraction: float = 0.5, seconds: float = 20.0,
                 stagger_min_steps: int = 512, stagger_seed: int = 2026112602):
        self.inner = inner
        self.seconds = float(seconds)
        self.cutoff_steps = int(round(seconds / inner.dt))
        if not (0 < self.cutoff_steps < inner.max_steps and math.isclose(self.cutoff_steps * inner.dt, seconds, abs_tol=1e-9)):
            raise ValueError("Invalid rehearsal duration")
        self.role_np = stratified_rehearsal_mask(inner.n_envs, fraction=fraction)
        self.role = jnp.asarray(self.role_np, dtype=jnp.bool_)
        self.n_rehearsal = int(self.role_np.sum())
        self.n_long = int((~self.role_np).sum())
        self.init_cutoffs_np = initial_staggered_cutoffs(
            self.role_np, self.cutoff_steps, min_steps=stagger_min_steps, seed=stagger_seed
        )
        self.init_cutoffs = jnp.asarray(self.init_cutoffs_np, dtype=jnp.int32)
        self.full_cutoffs = jnp.full((inner.n_envs,), self.cutoff_steps, dtype=jnp.int32)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def wrap_state(self, core, *, staggered: bool = True):
        cutoffs = self.init_cutoffs if staggered else self.full_cutoffs
        return _RehearsalStateWrapper(core, cutoffs)

    def reset(self, keys):
        core, obs = self.inner.reset(keys)
        return self.wrap_state(core, staggered=True), obs

    def reset_done(self, state, obs, done, keys):
        core, obs = self.inner.reset_done(state.core, obs, done, keys)
        next_cutoffs = jnp.where(done, self.full_cutoffs, state.rehearsal_cutoff_steps)
        return _RehearsalStateWrapper(core, next_cutoffs), obs

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state.core, obs)

    def _obs(self, state, native_obs):
        return self.inner._obs(state.core, native_obs)

    def step(self, state, action):
        after_core, obs, reward, done, metrics = self.inner.step(state.core, action)
        cutoff = self.role & (after_core.step_count >= state.rehearsal_cutoff_steps) & ~done
        full_horizon = (state.rehearsal_cutoff_steps >= self.cutoff_steps).astype(jnp.float32)
        metrics = {
            **metrics,
            "truncated": jnp.where(cutoff, jnp.ones_like(metrics["truncated"]), metrics["truncated"]),
            "sampling/rehearsal_lane": self.role.astype(jnp.float32),
            "sampling/rehearsal_cutoff": cutoff.astype(jnp.float32),
            "sampling/rehearsal_full_horizon": full_horizon,
        }
        return _RehearsalStateWrapper(after_core, state.rehearsal_cutoff_steps), obs, reward, done | cutoff, metrics


def masked_rollout(metrics: dict, done: np.ndarray, lane_mask: np.ndarray):
    """Slice 2D [T, N] rollout arrays along lane axis by boolean `lane_mask`."""
    done_np = np.asarray(done)
    n_envs = done_np.shape[1]
    idx = np.flatnonzero(lane_mask)
    selected = {
        k: (v[:, idx] if np.ndim(v) >= 2 and np.shape(v)[1] == n_envs else v)
        for k, v in metrics.items()
    }
    return selected, done_np[:, idx]


class StratifiedStartupProgress:
    """Track full 20s launch & early-juggling survival overall and by wind subgroup."""

    def __init__(self, *, seconds: float = 20.0, capacity: int = 2048, half_angle_deg: float = 45.0):
        self.seconds = float(seconds)
        self.half_angle_deg = float(half_angle_deg)
        self.outcomes = deque(maxlen=capacity)
        self.completed_episodes = 0
        self.fail_tags = deque(maxlen=capacity)

    def add(self, metrics: dict, done: np.ndarray, termination_fields: tuple[str, ...]):
        done_b = np.asarray(done, bool)
        if not np.any(done_b):
            return
        labels = labels_for_wind(
            metrics["wind/mode"],
            metrics["polish/background_speed_m_s"],
            metrics["single5/wind_angle_rad"],
            self.half_angle_deg,
        )
        for t, lane in zip(*np.nonzero(done_b)):
            full_h = float(metrics["sampling/rehearsal_full_horizon"][t, lane]) > 0.5
            terminated = float(metrics["terminated"][t, lane]) > 0.5
            cutoff = float(metrics["sampling/rehearsal_cutoff"][t, lane]) > 0.5
            if not full_h and not terminated:
                continue
            duration = float(metrics["diagnostic/episode_seconds"][t, lane])
            hits = float(metrics["hit_count"][t, lane])
            conf_ratio = float(metrics["timing/confirmed_max_ratio"][t, lane])
            phys_ratio = float(metrics["timing/physical_max_ratio"][t, lane])
            success = (
                full_h and cutoff and not terminated
                and duration >= self.seconds - 1e-5
                and hits >= math.ceil(1.2 * self.seconds)
                and conf_ratio <= 1.0 + 1e-6
                and phys_ratio <= 1.0 + 1e-6
            )
            lbl = int(labels[t, lane])
            racket_z_fail = bool(terminated and float(metrics.get("done/racket_z_out_of_bounds", np.zeros_like(done_b))[t, lane]) > 0.5)
            self.outcomes.append((duration, hits, success, terminated, lbl, racket_z_fail))
            if terminated:
                tags = [tag for tag in termination_fields if tag in metrics and float(metrics[tag][t, lane]) > 0.5]
                self.fail_tags.append((lbl, duration, hits, tags))
            self.completed_episodes += 1

    def summary(self) -> dict:
        n = len(self.outcomes)
        if n == 0:
            return dict(n=0, survival=0.0, mean_seconds=0.0, mean_hits=0.0,
                        completed_episodes=self.completed_episodes, seconds=self.seconds)

        def sub(pred):
            rows = [r for r in self.outcomes if pred(r[4])]
            return (len(rows), float(np.mean([r[2] for r in rows])) if rows else 0.0)

        clean_n, clean_s = sub(lambda l: l == 0)
        clean_rows = [r for r in self.outcomes if r[4] == 0]
        clean_phys_s = float(np.mean([not r[3] for r in clean_rows])) if clean_rows else 1.0
        peak_n, peak_s = sub(lambda l: l >= 17)
        negx_peak_n, negx_peak_s = sub(lambda l: l in (17, 18, 19, 20, 25))
        posx_peak_n, posx_peak_s = sub(lambda l: l in (21, 22, 23, 24, 26))
        dir08_n, dir08_s = sub(lambda l: l == 25)
        hard_n, hard_s = sub(lambda l: l in (4, 16))
        fails = [r for r in self.outcomes if r[3]]
        rz_frac = float(np.mean([r[5] for r in fails])) if fails else 0.0
        pre5_fails = sum(1 for r in fails if r[1] < 5.0)
        early_fails = sum(1 for r in fails if 5.0 <= r[1] < 20.0)
        return dict(
            n=n,
            survival=float(np.mean([r[2] for r in self.outcomes])),
            mean_seconds=float(np.mean([r[0] for r in self.outcomes])),
            mean_hits=float(np.mean([r[1] for r in self.outcomes])),
            completed_episodes=self.completed_episodes,
            seconds=self.seconds,
            clean_n=clean_n, clean_survival=clean_s, clean_phys_survival=clean_phys_s,
            peak_n=peak_n, peak_survival=peak_s,
            negx_peak_n=negx_peak_n, negx_peak_survival=negx_peak_s,
            posx_peak_n=posx_peak_n, posx_peak_survival=posx_peak_s,
            dir08_n=dir08_n, dir08_survival=dir08_s,
            hard_bin3_n=hard_n, hard_bin3_survival=hard_s,
            fail_count=len(fails),
            fail_racket_z_fraction=rz_frac,
            fail_pre_hit5=pre5_fails,
            fail_early_hit5_20=early_fails,
        )


class EarlyHitQualityTracker:
    """Fast vectorized sliding-window tracker for hits 1-5, 6-15, and 16-30 stability.

    Tracks exact contact center RMS (mm) and contact racket horizontal speed RMS (m/s)
    both overall and on clean (no-wind) lanes so live training can be compared directly
    against the V321 clean benchmark (1-5: 9.6mm / 0.062m/s; 6-15: 5.95mm / 0.040m/s).
    """

    BUCKETS = (("h1_5", 1, 5), ("h6_15", 6, 15), ("h16_30", 16, 30))

    def __init__(self, window_updates: int = 16):
        self.window = deque(maxlen=max(1, int(window_updates)))

    def add(self, metrics: dict):
        if "hit_event_count" not in metrics or "hit_count" not in metrics:
            return
        hit_evt = np.asarray(metrics["hit_event_count"]) > 0.5
        if not np.any(hit_evt):
            return
        hits = np.rint(np.asarray(metrics["hit_count"])).astype(np.int32)
        center = np.asarray(metrics["hit_contact_center_dist_sum"], dtype=np.float64)
        vxy_sq = np.asarray(metrics.get("hit_racket_vxy_sq_sum", np.zeros_like(center)), dtype=np.float64)
        is_clean = np.asarray(metrics["wind/mode"]) < 0.5

        rec = {}
        for name, lo, hi in self.BUCKETS:
            m_all = hit_evt & (hits >= lo) & (hits <= hi)
            m_cln = m_all & is_clean
            for tag, m in ((name, m_all), (f"{name}_clean", m_cln)):
                cnt = int(m.sum())
                if cnt > 0:
                    c_vals = center[m]
                    rec[tag] = (cnt, float(np.sum(c_vals * c_vals)), float(np.sum(vxy_sq[m])))
                else:
                    rec[tag] = (0, 0.0, 0.0)
        self.window.append(rec)

    def summary(self) -> dict:
        out = {}
        for name, _, _ in self.BUCKETS:
            for tag in (name, f"{name}_clean"):
                cnt = sum(r.get(tag, (0, 0.0, 0.0))[0] for r in self.window)
                csq = sum(r.get(tag, (0, 0.0, 0.0))[1] for r in self.window)
                vsq = sum(r.get(tag, (0, 0.0, 0.0))[2] for r in self.window)
                out[f"{tag}_n"] = cnt
                out[f"{tag}_center_rms_mm"] = (math.sqrt(max(csq / cnt, 0.0)) * 1000.0) if cnt > 0 else 0.0
                out[f"{tag}_vxy_rms_m_s"] = math.sqrt(max(vsq / cnt, 0.0)) if cnt > 0 else 0.0
        return out
