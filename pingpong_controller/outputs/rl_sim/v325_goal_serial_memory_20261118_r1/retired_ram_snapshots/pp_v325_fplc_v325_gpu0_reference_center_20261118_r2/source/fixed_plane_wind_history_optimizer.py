"""Explicit Adam conditioning for V325's existing observed XY history."""
from __future__ import annotations

import math

HISTORY_XY_ROWS = tuple(72 + 3 * lag + axis for lag in range(1, 8) for axis in (0, 1))
HISTORY_XY_OPTIMIZER_CONTRACT = dict(
    identity='residual_history_xy_adam_epsilon_2em6_v1',
    legacy_epsilon=1e-5, history_xy_epsilon=2e-6,
    actor_input_dim=114, rows=HISTORY_XY_ROWS,
    scope='residual pi.l1.w nonzero-lag XY rows only; every other update is legacy',
    moments='unchanged Adam m/v/t equations and global clipping',
    observation='unchanged 114-D actor; no new or privileged input')
HISTORY_XY_FAST_OPTIMIZER_CONTRACT = {
    **HISTORY_XY_OPTIMIZER_CONTRACT,
    'identity': 'residual_history_xy_adam_lr10_eps2em6_v2',
    'history_xy_lr_multiplier': 10.,
}


def make_history_xy_adam_step(legacy_adam_step, *, history_xy_epsilon: float = 1e-5,
                             history_xy_lr_multiplier: float = 1.):
    """Change selected denominators after the exact legacy moment update.

    The baseline function owns clipping, moment estimates, bias correction and
    every other parameter update. The selected coordinates still use those
    same moments. Default configuration returns the original function itself.
    """
    epsilon = float(history_xy_epsilon)
    multiplier = float(history_xy_lr_multiplier)
    if not math.isfinite(epsilon) or not 0. < epsilon <= 1e-5:
        raise ValueError('History XY epsilon must be finite and in (0, 1e-5]')
    if not math.isfinite(multiplier) or not 0. < multiplier <= 10.:
        raise ValueError('History XY LR multiplier must be finite and in (0,10]')
    base = getattr(legacy_adam_step, '_wind_history_optimizer_base', legacy_adam_step)
    if epsilon == 1e-5 and multiplier == 1.:
        return base

    def update(params, gradients, opt, learning_rate, max_grad_norm,
               beta1=.9, beta2=.999, eps=1e-5):
        import jax.numpy as jnp
        if eps != 1e-5:
            raise ValueError('History optimizer requires the registered legacy epsilon')
        try:
            weights = params['pi']['l1']['w']
            layouts = (gradients['pi']['l1']['w'], opt.m['pi']['l1']['w'],
                       opt.v['pi']['l1']['w'])
            compatible = ('teacher_pi' in params and weights.ndim == 2
                          and weights.shape[0] == 114
                          and all(x.shape == weights.shape for x in layouts))
        except (KeyError, AttributeError) as error:
            raise ValueError('History optimizer requires the complete 114-D residual actor') from error
        if not compatible:
            raise ValueError('History optimizer requires the complete 114-D residual actor')
        result, next_opt, norm = base(params, gradients, opt, learning_rate,
                                      max_grad_norm, beta1, beta2, eps)
        rows = jnp.asarray(HISTORY_XY_ROWS)
        m_hat = next_opt.m['pi']['l1']['w'][rows] / (1. - beta1 ** next_opt.t)
        v_hat = next_opt.v['pi']['l1']['w'][rows] / (1. - beta2 ** next_opt.t)
        selected = weights[rows] - (float(learning_rate) * multiplier) * m_hat / (jnp.sqrt(v_hat) + epsilon)
        result = {**result, 'pi': {**result['pi'], 'l1': {**result['pi']['l1'],
                  'w': result['pi']['l1']['w'].at[rows].set(selected)}}}
        return result, next_opt, norm

    update._wind_history_optimizer_base = base
    update._wind_history_xy_epsilon = epsilon
    update._wind_history_xy_lr_multiplier = multiplier
    return update


def configure_history_optimizer(ppo, recipe: dict | None) -> None:
    """Select only the exact saved recipe, without accumulating wrappers."""
    contract = (recipe or {}).get('history_xy_optimizer')
    if contract is not None and contract not in (
            HISTORY_XY_OPTIMIZER_CONTRACT, HISTORY_XY_FAST_OPTIMIZER_CONTRACT):
        raise ValueError('Unregistered history XY optimizer contract')
    current = ppo.adam_step
    base = getattr(current, '_wind_history_optimizer_base', current)
    if contract is None:
        ppo.adam_step = base
    elif (getattr(current, '_wind_history_xy_epsilon', None) != contract['history_xy_epsilon']
          or getattr(current, '_wind_history_xy_lr_multiplier', 1.) != contract.get('history_xy_lr_multiplier', 1.)):
        ppo.adam_step = make_history_xy_adam_step(
            base, history_xy_epsilon=contract['history_xy_epsilon'],
            history_xy_lr_multiplier=contract.get('history_xy_lr_multiplier', 1.))
