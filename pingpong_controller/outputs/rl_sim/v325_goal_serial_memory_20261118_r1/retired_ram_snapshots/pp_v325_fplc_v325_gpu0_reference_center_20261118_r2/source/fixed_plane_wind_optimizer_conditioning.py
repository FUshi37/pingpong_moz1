"""Explicit wind-only optimizer conditioning; historical modes stay exact."""
from fixed_plane_wind_history_optimizer import (
    HISTORY_XY_OPTIMIZER_CONTRACT, HISTORY_XY_FAST_OPTIMIZER_CONTRACT,
    configure_history_optimizer,
)


def optimizer_contract(*, fast: bool = False) -> dict:
    return dict(revision='wind_history_optimizer_conditioning_v1',
        history_xy_optimizer=dict(HISTORY_XY_FAST_OPTIMIZER_CONTRACT
                                  if fast else HISTORY_XY_OPTIMIZER_CONTRACT),
        freeze_fixed_std_gradient=True,
        scope='residual pi.l1 history XY rows only; fixed exploration scale has zero gradient',
        preserved='all existing parameter/moment values and Adam t; every other LR and forward output')


def fixed_std_gradient_function(current, *, enabled: bool):
    base = getattr(current, '_wind_fixed_std_base', current)
    if not enabled:
        return base
    if getattr(current, '_wind_fixed_std_base', None) is base:
        return current

    def effective(log_std, min_log_std, max_log_std=None):
        value = base(log_std, min_log_std, max_log_std)
        if (min_log_std is not None and max_log_std is not None
                and float(min_log_std) == float(max_log_std)):
            import jax
            return jax.lax.stop_gradient(value)
        return value

    effective._wind_fixed_std_base = base
    return effective


def configure_optimizer(ppo, saved_recipe: dict | None, mode: str) -> None:
    from fixed_plane_wind_clean_rehearsal import OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX, GATE_PRECISION_MIX, POSITIVE_PRECISION_MIX, REFERENCE_CONSTRAINT_MIX
    selected = mode in (OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX, GATE_PRECISION_MIX, POSITIVE_PRECISION_MIX, REFERENCE_CONSTRAINT_MIX)
    recipe = dict(saved_recipe or {})
    if selected:
        recipe['history_xy_optimizer'] = optimizer_contract(
            fast=mode == FAST_OPTIMIZER_MIX)['history_xy_optimizer']
    configure_history_optimizer(ppo, recipe)
    if hasattr(ppo, 'effective_log_std'):
        ppo.effective_log_std = fixed_std_gradient_function(
            ppo.effective_log_std, enabled=selected)
    elif selected:
        raise ValueError('Optimizer conditioning requires the PPO exploration-scale function')
