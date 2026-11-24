"""Opt-in actor imitation of executed seven-joint commands, never a controller."""
from __future__ import annotations

import numpy as np

REVISION = 'observed_bound_conditioned_feasible_actor_20261114'


def initialize_gated_residual(state):
    """Preserve the old actor as a frozen teacher; create a new residual family."""
    import jax
    import jax.numpy as jnp
    pi = state.params['pi']
    residual = {**pi, 'out': jax.tree.map(jnp.zeros_like, pi['out'])}
    extras = dict(teacher_pi=pi, residual_action_scale=jnp.full(7, 2.),
                  wind_pose_bound_gate_v1=jnp.asarray(.25))
    zeros = jax.tree.map(jnp.zeros_like, {**extras, 'pi': residual})
    return state._replace(params={**state.params, **extras, 'pi': residual},
        opt=state.opt._replace(m={**state.opt.m, **zeros}, v={**state.opt.v, **zeros}))


def validate_demonstrations(obs, actions) -> None:
    """Require the registered114-D camera history and normalized7-D QVEL."""
    if (obs.ndim != 2 or obs.shape[1] != 114 or actions.shape != (len(obs), 7)
            or len(obs) < 2 or not np.isfinite(obs).all()
            or not np.isfinite(actions).all() or (np.abs(actions) > 1.000001).any()):
        raise ValueError('Invalid executed-command demonstrations')


def conditioned_demonstrations(obs, projected_actions, raw_actions, late):
    """Condition imitation on existing authority observations, never hit truth."""
    xs, ys = [obs.copy()], [raw_actions.copy()]
    xs[0][:, -2:] = .1
    weights = [np.where(late, 4., 40.).astype(np.float32)]
    for degrees in (3., 5., 10., 20.):
        x = obs[late].copy()
        x[:, -2:] = degrees / 20.
        xs.append(x)
        ys.append(projected_actions[late])
        weights.append(np.ones(len(x), np.float32))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def replace_actor_update(state, new_pi, actor_opt, *, legacy_width: int = 72,
                         conditioning_columns: int = 0):
    """Retain critic/log-std and moments; new history rows remain zero initially."""
    def zero_history(pi):
        return {**pi, 'l1': {**pi['l1'],
            'w': pi['l1']['w'].at[legacy_width:
                -conditioning_columns if conditioning_columns else None].set(0.)}}
    return state._replace(params={**state.params, 'pi': zero_history(new_pi)},
        opt=state.opt._replace(m={**state.opt.m, 'pi': zero_history(actor_opt.m)},
            v={**state.opt.v, 'pi': zero_history(actor_opt.v)}, t=actor_opt.t))


def make_actor_bootstrap_step(ppo, *, learning_rate: float = 1e-4,
                              conditioning_columns: int = 0):
    """Use inherited actor Adam moments; account every extra optimizer step."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def step(state, obs, actions, sample_weights):
        def objective(pi):
            prediction = ppo.policy_mean({**state.params, 'pi': pi}, obs)
            error = jnp.mean(jnp.square(prediction - actions), axis=-1)
            return jnp.sum(error * sample_weights) / jnp.maximum(sample_weights.sum(), 1e-12)
        loss, gradient = jax.value_and_grad(objective)(state.params['pi'])
        gradient = {**gradient, 'l1': {**gradient['l1'],
            'w': gradient['l1']['w'].at[72:
                -conditioning_columns if conditioning_columns else None].set(0.)}}
        opt = state.opt._replace(m=state.opt.m['pi'], v=state.opt.v['pi'])
        pi, opt, norm = ppo.adam_step(state.params['pi'], gradient, opt,
                                    learning_rate, .5)
        return replace_actor_update(state, pi, opt,
            conditioning_columns=conditioning_columns), loss, norm
    return step
