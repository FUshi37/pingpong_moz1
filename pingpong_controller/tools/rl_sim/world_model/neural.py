"""Small FP32 JAX networks and serializable Adam without optional dependencies."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp


def init_mlp(key, widths, *, zero_output=False):
    """Initialize tanh hidden layers and a linear output in FP32."""
    if len(widths) < 2 or any(width < 1 for width in widths):
        raise ValueError('Positive input, hidden and output widths are required')
    keys = jax.random.split(key, len(widths)-1)
    layers = []
    for index, (left, right) in enumerate(zip(widths[:-1], widths[1:])):
        weight = jax.random.normal(keys[index], (left, right), dtype=jnp.float32)/jnp.sqrt(float(left))
        if zero_output and index == len(widths)-2:
            weight = jnp.zeros_like(weight)
        layers.append(dict(w=weight, b=jnp.zeros(right, jnp.float32)))
    return tuple(layers)


def mlp(parameters, inputs):
    value = jnp.asarray(inputs, jnp.float32)
    for index, layer in enumerate(parameters):
        value = value @ layer['w'] + layer['b']
        if index < len(parameters)-1:
            value = jnp.tanh(value)
    return value


class AdamState(NamedTuple):
    step: Any
    first_moment: Any
    second_moment: Any


def init_adam(parameters) -> AdamState:
    zeros = jax.tree.map(jnp.zeros_like, parameters)
    return AdamState(jnp.asarray(0, jnp.int32), zeros, zeros)


def adam_update(parameters, gradients, state: AdamState, *, learning_rate,
                 max_gradient_norm=None, beta1=.9, beta2=.999, epsilon=1e-8):
    """Return updated parameters/state plus pre-clipping gradient diagnostics."""
    squared = sum(jnp.sum(jnp.square(value)) for value in jax.tree.leaves(gradients))
    norm = jnp.sqrt(squared)
    multiplier = (jnp.asarray(1., jnp.float32) if max_gradient_norm is None
                  else jnp.minimum(1., max_gradient_norm/jnp.maximum(norm, 1e-12)))
    gradients = jax.tree.map(lambda g: g*multiplier, gradients)
    first = jax.tree.map(lambda old, g: beta1*old+(1-beta1)*g, state.first_moment, gradients)
    second = jax.tree.map(lambda old, g: beta2*old+(1-beta2)*g*g, state.second_moment, gradients)
    step = state.step+1
    first_correction, second_correction = 1-beta1**step, 1-beta2**step
    parameters = jax.tree.map(lambda p, m, v: p-learning_rate*(m/first_correction)/
        (jnp.sqrt(v/second_correction)+epsilon), parameters, first, second)
    return parameters, AdamState(step, first, second), dict(gradient_norm=norm,
                                                           gradient_scale=multiplier)
