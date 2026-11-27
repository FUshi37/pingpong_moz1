"""Early-juggling (hits 1-25) stability reward targeting V321 clean-level smoothness.

Adds bounded Huber penalties ONLY during launch & early juggling (hits 1-15 full weight,
linear taper 16-25, strictly zero for hits > 25 so steady-state 31-115 equilibrium is
untouched):
  1. reward/early_center   : contact center eccentricity > 4.0 mm (hits 1-25)
  2. reward/early_hit_vxy  : contact racket horizontal speed > 0.040 (+0.025*wind) m/s (hits 1-25)
  3. reward/early_step_vxy : per-step racket horizontal speed > 0.060 (+0.050*wind) m/s (hits 0-20)
  4. reward/early_apex     : post-launch apex deviation |z - 1.425| > 8.0 mm (hits 5-25)
"""
import jax.numpy as jnp

EARLY_REWARD_FIELDS = (
    "reward/early_center",
    "reward/early_hit_vxy",
    "reward/early_step_vxy",
    "reward/early_apex",
    "reward/early_total",
)


def _huber(x):
    return jnp.sqrt(1.0 + x * x) - 1.0


def compute_early_stability_terms(
    metrics: dict,
    *,
    confirmed_hits,
    racket_vel_xy,
    dt_s: float,
    cfg: dict,
):
    """Compute bounded early-hit stability terms using only native physics/diagnostics."""
    zero = jnp.zeros_like(metrics["terminated"])
    if not cfg.get("enabled", False):
        return {k: zero for k in EARLY_REWARD_FIELDS}

    live = metrics["terminated"] < 0.5
    h = confirmed_hits.astype(jnp.float32)

    full_end = float(cfg.get("full_weight_until_hit", 15.0))
    taper_end = float(cfg.get("taper_zero_at_hit", 25.0))
    dense_end = float(cfg.get("dense_zero_at_hit", 20.0))
    w_early = jnp.clip((taper_end - h) / max(taper_end - full_end, 1.0), 0.0, 1.0)
    w_post5 = jnp.where(h >= 5.0, w_early, 0.0)
    w_dense = jnp.clip((dense_end - h) / max(dense_end - full_end, 1.0), 0.0, 1.0)

    wind_frac = jnp.clip(metrics["wind/speed_m_s"] / 1.6, 0.0, 1.0)

    # 1. Early hit center eccentricity (hits 1..25)
    hit_evt = metrics["hit_event_count"]
    center_m = metrics["hit_contact_center_dist_sum"] / jnp.maximum(hit_evt, 1.0)
    c_dead = float(cfg.get("center_deadzone_m", 0.004))
    c_scale = float(cfg.get("center_scale_m", 0.004))
    c_wt = float(cfg.get("center_weight", 8.0))
    c_exc = jnp.maximum(center_m - c_dead, 0.0) / c_scale
    r_center = jnp.where(live, -c_wt * w_early * hit_evt * jnp.minimum(_huber(c_exc), 6.0), 0.0)

    # 2. Early contact racket horizontal speed (hits 1..25)
    hit_vxy = jnp.sqrt(jnp.maximum(metrics["hit_racket_vxy_sq_sum"], 0.0) / jnp.maximum(hit_evt, 1.0))
    v_dead = float(cfg.get("hit_vxy_deadzone_m_s", 0.040)) + float(cfg.get("hit_vxy_wind_allowance_m_s", 0.025)) * wind_frac
    v_scale = float(cfg.get("hit_vxy_scale_m_s", 0.025))
    v_wt = float(cfg.get("hit_vxy_weight", 6.0))
    v_exc = jnp.maximum(hit_vxy - v_dead, 0.0) / v_scale
    r_hit_vxy = jnp.where(live, -v_wt * w_early * hit_evt * jnp.minimum(_huber(v_exc), 6.0), 0.0)

    # 3. Dense early approach/transition horizontal quietness (hits 0..20)
    step_vxy = jnp.linalg.norm(racket_vel_xy, axis=-1)
    sv_dead = float(cfg.get("step_vxy_deadzone_m_s", 0.060)) + float(cfg.get("step_vxy_wind_allowance_m_s", 0.050)) * wind_frac
    sv_scale = float(cfg.get("step_vxy_scale_m_s", 0.040))
    sv_rate = float(cfg.get("step_vxy_rate_per_s", 12.0))
    sv_exc = jnp.maximum(step_vxy - sv_dead, 0.0) / sv_scale
    r_step_vxy = jnp.where(live, -sv_rate * dt_s * w_dense * jnp.minimum(_huber(sv_exc), 6.0), 0.0)

    # 4. Early post-launch apex precision (hits 5..25, excluding 1-4 launch transition)
    apex_evt = metrics["wind_height/apex_events"]
    apex_m = metrics["wind_height/apex_sum_m"] / jnp.maximum(apex_evt, 1.0)
    a_dead = float(cfg.get("apex_deadzone_m", 0.008))
    a_scale = float(cfg.get("apex_scale_m", 0.010))
    a_wt = float(cfg.get("apex_weight", 6.0))
    a_exc = jnp.maximum(jnp.abs(apex_m - 1.425) - a_dead, 0.0) / a_scale
    r_apex = jnp.where(live, -a_wt * w_post5 * apex_evt * jnp.minimum(_huber(a_exc), 6.0), 0.0)

    r_total = r_center + r_hit_vxy + r_step_vxy + r_apex
    return {
        "reward/early_center": r_center,
        "reward/early_hit_vxy": r_hit_vxy,
        "reward/early_step_vxy": r_step_vxy,
        "reward/early_apex": r_apex,
        "reward/early_total": r_total,
    }


class EarlyStabilityRewardEnv:
    """Opt-in wrapper adding early-hit stability penalties without modifying frozen sources."""

    def __init__(self, inner, cfg: dict | None = None):
        self.inner = inner
        self.early_cfg = dict(cfg or {})
        self.enabled = bool(self.early_cfg.get("enabled", False))

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self, keys):
        return self.inner.reset(keys)

    def reset_done(self, state, obs, done, keys):
        return self.inner.reset_done(state, obs, done, keys)

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state, obs)

    def _obs(self, state, native_obs):
        return self.inner._obs(state, native_obs)

    def step(self, state, action):
        after, obs, reward, done, metrics = self.inner.step(state, action)
        terms = compute_early_stability_terms(
            metrics,
            confirmed_hits=after.wind_startup_hit_count,
            racket_vel_xy=after.prev_racket_vel[:, :2],
            dt_s=self.dt,
            cfg=self.early_cfg,
        )
        if self.enabled:
            reward = reward + terms["reward/early_total"]
            metrics = {**metrics, **terms, "reward/total": reward}
        else:
            metrics = {**metrics, **terms}
        return after, obs, reward, done, metrics
