"""Explicit ABCD experiment adapter; legacy V325/V326 paths remain unchanged.

All four actors have a 114-value native prefix, 46 past camera packets of
(relative xyz, actual age, validity), and two current horizontal wind values.
Only C/D expose wind. The asymmetric critic retains its original 453 values.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from fixed_plane_wind_history import LAGS

DENSE_COUNT = 46
SPARSE_SLOTS = (0, 5, 9, 14, 18, 27, 36, 45)
ACTOR_DIM = 114 + DENSE_COUNT * 5 + 2
REVISION = 'v325_abcd_observed_camera_temporal_v1'


def temporal_packet(state, *, dense: bool):
    """Return [lane,46,5] causal observed samples, newest first, with masks."""
    pos, steps = state.ball_positions, state.ball_sample_steps
    age = (state.step_count[:, None] - steps) * .005
    valid = (state.ball_valid & jnp.all(jnp.isfinite(pos), axis=-1)
             & (age >= 0.) & (age <= .6))
    if dense:
        # Camera samples repeat at the 200 Hz control rate. Retain each
        # timestamp once, including the newest sample of the buffer.
        sentinel = jnp.iinfo(steps.dtype).max
        # An invalid held copy must not hide an older valid copy of the
        # same camera sample. Sample timestamps are newest-to-oldest.
        seen = jax.lax.associative_scan(jnp.minimum,
            jnp.where(valid, steps, sentinel), axis=1)
        preceding = jnp.concatenate((jnp.full_like(seen[:, :1], sentinel),
                                     seen[:, :-1]), axis=1)
        unique = valid & (steps != preceding)
        scores = jnp.where(unique, 101 - jnp.arange(101), -1)
        scores, indices = jax.lax.top_k(scores, DENSE_COUNT)
        present = scores >= 0
    else:
        indices = jnp.broadcast_to(jnp.asarray(LAGS), (pos.shape[0], len(LAGS)))
        present = jnp.take_along_axis(valid, indices, axis=1)
    selected = jnp.take_along_axis(pos, indices[..., None], axis=1)
    ages = jnp.take_along_axis(age, indices, axis=1)
    reference_ok = valid[:, :1]
    delta = jnp.where((present & reference_ok)[..., None], selected - pos[:, :1], 0.)
    delta = delta / jnp.asarray([.1, .1, .5], dtype=pos.dtype)
    packet = jnp.concatenate((delta, jnp.where(present, ages / .6, 0.)[..., None],
                              present.astype(pos.dtype)[..., None]), axis=-1)
    if not dense:
        packet = jnp.zeros((pos.shape[0], DENSE_COUNT, 5), pos.dtype).at[
            :, jnp.asarray(SPARSE_SLOTS)].set(packet)
    return packet


def augment_observation(state, native_obs, effective_wind, case: str):
    if case not in ('A', 'B', 'C', 'D') or native_obs.shape[-1] != 114:
        raise ValueError('ABCD requires an explicit case and unchanged114-D prefix')
    packet = temporal_packet(state, dense=case in ('B', 'D'))
    wind = (effective_wind[:, :2] / 4. if case in ('C', 'D')
            else jnp.zeros((native_obs.shape[0], 2), native_obs.dtype))
    return jnp.concatenate((native_obs, packet.reshape((-1, DENSE_COUNT * 5)), wind), axis=-1)


def initialize_adapter(key):
    keys = iter(jax.random.split(key, 6))

    def layer(shape):
        return dict(w=jax.random.normal(next(keys), shape) / jnp.sqrt(float(shape[-2])),
                    b=jnp.zeros(shape[-1], jnp.float32))

    return dict(frame=layer((5, 16)), conv1=layer((3, 16, 16)),
                conv2=layer((3, 16, 16)), latent=layer((DENSE_COUNT * 16, 32)),
                fusion=layer((114 + 32 + 2, 64)),
                out=dict(w=jnp.zeros((64, 7), jnp.float32), b=jnp.zeros(7, jnp.float32)))


def initialize_learner(state, key):
    """Append only new weights/zero moments; retain all parent leaves and Adam t."""
    if 'wind_ablation_v1' in state.params or state.params['pi']['l1']['w'].shape[0] != 114:
        raise ValueError('ABCD initialization requires the unmodified114-D parent')
    adapter = initialize_adapter(key)
    zeros = jax.tree.map(jnp.zeros_like, adapter)
    return state._replace(params={**state.params, 'wind_ablation_v1': adapter},
        opt=state.opt._replace(m={**state.opt.m, 'wind_ablation_v1': zeros},
                               v={**state.opt.v, 'wind_ablation_v1': zeros}))


def adapter_output(params, obs):
    packet = obs[:, 114:-2].reshape((-1, DENSE_COUNT, 5))
    mask = packet[..., 4:5]
    x = jnp.tanh(packet @ params['frame']['w'] + params['frame']['b']) * mask
    for name in ('conv1', 'conv2'):
        p = params[name]
        x = jnp.tanh(jax.lax.conv_general_dilated(x, p['w'], (1,), 'SAME',
            dimension_numbers=('NWC', 'WIO', 'NWC')) + p['b']) * mask
    p = params['latent']
    latent = jnp.tanh(x.reshape((-1, DENSE_COUNT * 16)) @ p['w'] + p['b'])
    fused = jnp.concatenate((obs[:, :114], latent, obs[:, -2:]), axis=-1)
    p = params['fusion']
    x = jnp.tanh(fused @ p['w'] + p['b'])
    return x @ params['out']['w'] + params['out']['b']


def policy_components(params, obs):
    """Use the original authority gate on its native prefix, never on wind."""
    if obs.ndim != 2 or obs.shape[-1] != ACTOR_DIM:
        raise ValueError('ABCD actor requires its explicit346-D observation')
    from train_juggle_mjx_ppo import apply_mlp
    from fixed_plane_wind_actor import apply_bound_gate
    prefix = obs[:, :114]
    teacher = jax.lax.stop_gradient(apply_mlp(params['teacher_pi'], prefix))
    logits = apply_mlp(params['pi'], prefix) + adapter_output(params['wind_ablation_v1'], obs)
    correction = jax.lax.stop_gradient(params['residual_action_scale']) * jnp.tanh(logits)
    baseline, correction = apply_bound_gate(params, prefix, teacher, correction)
    return baseline + correction, teacher, correction


class AblationEnv:
    """Observation-only adapter: native state, rewards, resets and critic unchanged."""

    def __init__(self, inner, case: str):
        if (inner.obs_dim, inner.critic_obs_dim, inner.act_dim) != (114, 453, 7):
            raise ValueError('ABCD requires the original history-enabled V325 environment')
        if case not in 'ABCD' or len(case) != 1:
            raise ValueError('Unknown ablation case')
        self.inner, self.case = inner, case
        self.obs_dim, self.critic_obs_dim, self.act_dim = ACTOR_DIM, 453, 7

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def _obs(self, state, obs):
        from fixed_plane_v325_wind import wind_velocity
        velocity = wind_velocity(state, state.step_count * self.dt, self.cfg.wind_config)
        return augment_observation(state, obs, velocity, self.case)

    def reset(self, keys):
        state, obs = self.inner.reset(keys)
        return state, self._obs(state, obs)

    def step(self, state, action):
        state, obs, reward, done, metrics = self.inner.step(state, action)
        return state, self._obs(state, obs), reward, done, metrics

    def reset_done(self, state, obs, done, keys):
        state, native = self.inner.reset_done(state, obs[:, :114], done, keys)
        return state, self._obs(state, native)

    def get_critic_obs(self, state, obs):
        return self.inner.get_critic_obs(state, obs[:, :114])
