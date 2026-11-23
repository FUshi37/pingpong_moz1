"""Diagnostic-only precision copies and directional checks; no production edits."""

from __future__ import annotations

import ast
from dataclasses import fields, is_dataclass, replace
import inspect
import textwrap
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


def private_clone(function: Callable, *, float64: bool = False,
                  overrides: dict | None = None) -> Callable:
    """Copy one function namespace; optional dtype substitutions stay in the copy."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    if float64:
        class Promote(ast.NodeTransformer):
            def visit_Attribute(self, node):
                if (isinstance(node.value, ast.Name) and node.value.id == 'jnp'
                        and node.attr == 'float32'):
                    node.attr = 'float64'
                return self.generic_visit(node)
        tree = Promote().visit(tree)
    namespace = dict(function.__globals__)
    namespace.update(overrides or {})
    exec(compile(ast.fix_missing_locations(tree), '<WMJ-private-precision-probe>', 'exec'), namespace)
    return namespace[function.__name__]


def cast_floats(value: Any, dtype: Any) -> Any:
    """Promote numeric context, preserving discrete queue indices and masks."""
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: cast_floats(getattr(value, f.name), dtype)
                                 for f in fields(value)})
    if isinstance(value, dict):
        return {k: cast_floats(v, dtype) for k, v in value.items()}
    if isinstance(value, tuple):
        parts = [cast_floats(v, dtype) for v in value]
        return type(value)(*parts) if hasattr(value, '_fields') else tuple(parts)
    if isinstance(value, list):
        return [cast_floats(v, dtype) for v in value]
    if hasattr(value, 'dtype') and np.dtype(value.dtype).kind in 'fc':
        return jnp.asarray(value, dtype)
    return value


def distribution(value: Any) -> dict:
    data = np.asarray(value, float).ravel()
    finite = data[np.isfinite(data)]
    result = dict(count=int(data.size), nonfinite=int(data.size-finite.size))
    if finite.size:
        result.update(mean=float(finite.mean()), p50=float(np.quantile(finite, .5)),
                      p95=float(np.quantile(finite, .95)), maximum=float(finite.max()))
    return result


def probe_direction(function: Callable, epsilons: list[float], *, dtype: Any,
                    check: Callable = lambda: None) -> dict:
    """Probe Y(N,K) along a scalar; report branch changes and per-lane errors."""
    forward = jax.jit(function)
    derivative = jax.jit(lambda a: jax.jvp(function, (a,), (jnp.ones_like(a),), has_aux=True))
    zero = jnp.asarray(0., dtype)
    value, modes = jax.device_get(forward(zero))
    primal, tangent, ad_modes = jax.device_get(derivative(zero))
    if value.ndim != 2 or modes.shape[0] != value.shape[0]:
        raise ValueError('Module probes require numeric(N,K) and leading-N branch fingerprints')
    n = len(value)
    cotangent = np.random.default_rng(1327).normal(size=value.shape)
    cotangent /= max(float(np.linalg.norm(cotangent)), 1e-12)
    cotangent = jnp.asarray(cotangent, dtype)
    reverse = jax.jit(jax.grad(lambda a: jnp.sum(function(a)[0]*cotangent)))
    vjp = float(reverse(zero))
    jvp_projection = float(np.sum(tangent*np.asarray(cotangent)))
    repeated, _ = jax.device_get(forward(zero))
    report = dict(output_shape=list(value.shape), output_dtype=str(value.dtype),
        automatic_norm=distribution(np.linalg.norm(tangent, axis=-1)),
        forward_vs_ad_primal_abs=distribution(np.abs(value-primal)),
        forward_repeat_abs=distribution(np.abs(value-repeated)),
        ad_primal_branch_changed_lanes=int(np.any(modes.reshape(n,-1)!=ad_modes.reshape(n,-1), axis=-1).sum()),
        jvp_projection=jvp_projection if np.isfinite(jvp_projection) else None,
        vjp_projection=vjp if np.isfinite(vjp) else None,
        jvp_vjp_abs_difference=abs(vjp-jvp_projection) if np.isfinite(vjp+jvp_projection) else None,
        rows=[])
    for epsilon in epsilons:
        check()
        plus, plus_modes = jax.device_get(forward(jnp.asarray(epsilon, dtype)))
        minus, minus_modes = jax.device_get(forward(jnp.asarray(-epsilon, dtype)))
        central = (plus-minus)/(2*epsilon)
        changed = np.any(plus_modes.reshape(n,-1)!=minus_modes.reshape(n,-1), axis=-1)
        error = np.linalg.norm(central-tangent, axis=-1)
        magnitude = np.maximum(np.linalg.norm(central, axis=-1), np.linalg.norm(tangent, axis=-1))
        relative = error/np.maximum(magnitude, 1e-8)
        signal = magnitude > 1e-5
        stable = ~changed & signal
        finite = np.isfinite(relative)
        bad = np.flatnonzero(stable & ((relative>.1) | ~finite))
        report['rows'].append(dict(epsilon=float(epsilon), branch_changed_lanes=int(changed.sum()),
            branch_changed_indices=np.flatnonzero(changed).tolist(), informative_lanes=int(signal.sum()),
            relative_norm_error=distribution(relative), absolute_norm_error=distribution(error),
            unchanged_branch_relative_error=distribution(relative[stable]),
            unchanged_branch_bad_count=int(len(bad)), bad_indices=bad.tolist(),
            directional_cosine=distribution(np.sum(central*tangent,axis=-1)/np.maximum(
                np.linalg.norm(central,axis=-1)*np.linalg.norm(tangent,axis=-1),1e-12)),
            worst_relative_indices=np.argsort(np.nan_to_num(relative,nan=np.inf))[-min(4,n):].tolist()))
    forward.clear_cache()
    derivative.clear_cache()
    reverse.clear_cache()
    return report
