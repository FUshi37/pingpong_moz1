"""Opt-in wind-only precontact miss cost; native dynamics and inputs unchanged."""
from typing import Any


CAPTURE_FIELDS = ('reward/capture_margin_cost', 'capture/active',
                  'capture/risk_frames', 'capture/predicted_radius_sum_m',
                  'capture/maturity')


def capture_cost(radius_m: Any, *, active: Any, gain: Any, reward_steps: Any,
                 dt_s: float, xp: Any) -> dict:
    """Penalize predicted local-XY misses beyond25mm without a dead far tail.

    The native racket radius is75mm and the ball radius20mm. A25mm preview
    tolerance leaves50mm radial recovery margin. This is a training target,
    not a contact model, gate, physical limit, or controllability guarantee.
    """
    excess = xp.maximum((radius_m - .025) / .020, 0.)
    loss = xp.sqrt(1. + excess * excess) - 1.
    progress = xp.clip(reward_steps / 8192., 0., 1.)
    maturity = progress * progress * (3. - 2. * progress)
    enabled = active & (gain > 0.)
    cost = -dt_s * 80. * gain * maturity * xp.where(enabled, loss, 0.)
    return {
        'reward/capture_margin_cost': cost,
        'capture/active': enabled.astype(radius_m.dtype),
        'capture/risk_frames': (enabled & (radius_m > .025)).astype(radius_m.dtype),
        'capture/predicted_radius_sum_m': xp.where(enabled, radius_m, 0.),
        'capture/maturity': maturity,
    }


class CaptureEnv:
    """An explicit reward-only wrapper; disabled mode preserves exact outputs."""

    def __init__(self, inner: Any, enabled: bool = False):
        self.inner = inner
        self.enabled = enabled

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def step(self, state: Any, action: Any) -> tuple:
        import jax.numpy as jnp
        after, obs, reward, done, metrics = self.inner.step(state, action)
        if self.enabled:
            from fixed_plane_wind_pose import shelter_gain
            from fixed_plane_wind_precontact_center import environment_credit
            gain = shelter_gain(state, self.dt, xp=jnp) * (state.wind_mode > 0)
            preview = environment_credit(self.inner, after, metrics,
                clean=state.wind_mode == 0, gain=gain, reward_steps=after.total_env_steps)
            terms = capture_cost(preview['precontact/center_m'],
                active=preview['precontact/active'] > .5, gain=gain,
                reward_steps=after.total_env_steps, dt_s=self.dt, xp=jnp)
            # Preserve the old reward bit-for-bit on clean/sheltered/precise lanes.
            cost = terms['reward/capture_margin_cost']
            reward = jnp.where(cost < 0., reward + cost, reward)
            metrics.update(terms)
            from fixed_plane_wind_gate_precision import group_reward_metrics
            metrics.update(group_reward_metrics(reward, state.wind_mode > 0, metrics, xp=jnp))
            metrics['reward/total'] = reward
        else:
            metrics.update({name: jnp.zeros_like(reward) for name in CAPTURE_FIELDS})
        return after, obs, reward, done, metrics
