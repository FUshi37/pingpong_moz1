"""Opt-in V322 MJX scheduling; keep physical values and iteration caps."""

from collections.abc import Callable
import hashlib
from pathlib import Path

import jax
import jax.numpy as jnp

EXECUTION_VERSION = 'mjx_line_search_chunk5_v1'
REQUIRED_JAX_VERSION = '0.10.1'
REQUIRED_MUJOCO_VERSION = '3.8.1'
REQUIRED_SOLVER_SHA256 = (
    'ad9c611bb8fe4116b0bba10f66b3473a958fe770fa82f9080a0f66a206d299ab')


def validate_v322_execution_contract(
    reward_preset: str, loss_preset: str, gate_preset: str,
) -> None:
    """Limit this execution option to the paired training contract."""
    if (reward_preset, loss_preset, gate_preset) != (
            'late_center_precision', 'anchor1', 'deployment_groups'):
        raise ValueError('MJX schedule requires the validated V322 preset tuple')


def make_chunked_loop(chunk_size: int) -> Callable:
    """Retain the scan's cached condition and per-lane masked iteration.

    Skip only blocks after convergence. Keep the original maximum number
    of body calls, including a final partial block. This primal experiment
    does not support reverse-mode differentiation through the simulator.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')

    def loop(cond_fun, body_fun, init_val, max_iter):
        def iteration(value):
            next_value = body_fun(value)
            return next_value, cond_fun(next_value)

        def scan_step(carry, unused):
            count, value, condition = carry
            value, condition = jax.lax.cond(
                condition & (count < max_iter), iteration,
                lambda current: (current, False), value)
            return (count + 1, value, condition), unused

        def condition(carry):
            count, _, active = carry
            return (count < max_iter) & active

        def block(carry):
            return jax.lax.scan(
                scan_step, carry, None, length=chunk_size)[0]

        initial = (jnp.int32(0), init_val, cond_fun(init_val))
        return jax.lax.while_loop(condition, block, initial)[1]

    return loop


def install_chunk5_schedule() -> dict:
    """Install the validated primal schedule only after an explicit opt-in.

    This changes one Python callable in this trainer process, never library
    files. Call after the unchanged initial reset and before rollout tracing.
    The original path remains the default. Reject unvalidated libraries and
    unrelated patches rather than stacking solver changes.
    """
    import mujoco
    from mujoco.mjx._src import solver

    if (jax.__version__ != REQUIRED_JAX_VERSION
            or mujoco.__version__ != REQUIRED_MUJOCO_VERSION):
        raise RuntimeError(
            'Solver schedule requires validated JAX/MuJoCo versions')
    solver_sha256 = hashlib.sha256(
        Path(solver.__file__).read_bytes()).hexdigest()
    if solver_sha256 != REQUIRED_SOLVER_SHA256:
        raise RuntimeError(
            'Solver schedule requires the validated solver source')
    current = solver._while_loop_scan
    marker = getattr(current, '_pingpong_execution_version', None)
    if marker != EXECUTION_VERSION:
        if (current.__module__ != solver.__name__
                or current.__name__ != '_while_loop_scan'):
            raise RuntimeError(
                'Solver was already patched by another execution mode')
        current = make_chunked_loop(5)
        current._pingpong_execution_version = EXECUTION_VERSION
        solver._while_loop_scan = current
    return dict(version=EXECUTION_VERSION, block_size=5,
                scope='MJX line-search loop scheduling only',
                default_enabled=False, solver_sha256=solver_sha256,
                jax_version=jax.__version__, mujoco_version=mujoco.__version__,
                differentiation='primal simulator execution only')


def make_scheduled_collect(collect: Callable) -> tuple[Callable, dict]:
    """Scope scheduling to tracing the single-threaded training collector.

    Restore the original callable before returning, including on errors.
    A compiled collector retains its staged operations. Other simulator
    calls, including independent evaluation, continue using the original
    solver. No device synchronization or training input changes are added.
    """
    from mujoco.mjx._src import solver

    original = solver._while_loop_scan
    if (original.__module__ != solver.__name__
            or original.__name__ != '_while_loop_scan'):
        raise RuntimeError('Collect scope requires the original solver')
    try:
        metadata = install_chunk5_schedule()
        scheduled = solver._while_loop_scan
    finally:
        solver._while_loop_scan = original

    def scoped(*args, **kwargs):
        if solver._while_loop_scan is not original:
            raise RuntimeError('Solver changed outside the collect scope')
        solver._while_loop_scan = scheduled
        try:
            return collect(*args, **kwargs)
        finally:
            solver._while_loop_scan = original

    return scoped, {**metadata, 'scope': 'training collect tracing only'}
