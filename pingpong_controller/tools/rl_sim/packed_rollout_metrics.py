"""Read every compact rollout mean with one compiled, packed host transfer."""

from collections.abc import Callable, Mapping

import jax
import jax.numpy as jnp

COMPACT_PREFIX = '__rollout_mean__/'
EXECUTION_VERSION = 'packed_host_metrics_v1'


def make_packed_metric_reader(
    *, prefix: str = COMPACT_PREFIX,
) -> Callable[[Mapping[str, jax.Array]], dict[str, float]]:
    """Build a host-only reader without changing rollout arrays or consumers.

    Apply the original ``jnp.mean`` separately to each compact channel, pack
    the scalars, and copy one vector. Per-lane channels remain in the rollout
    for their existing consumers. No input is donated or modified. JAX caches
    each channel-layout/shape variant independently.
    """
    if not isinstance(prefix, str) or not prefix:
        raise ValueError('Compact metric prefix must be a nonempty string')

    @jax.jit
    def reduce_means(values: tuple[jax.Array, ...]) -> jax.Array:
        return jnp.stack([jnp.mean(value) for value in values])

    def read(metrics: Mapping[str, jax.Array]) -> dict[str, float]:
        keys = tuple(sorted(key for key in metrics if key.startswith(prefix)))
        if not keys:
            return {}
        packed = jax.device_get(
            reduce_means(tuple(metrics[key] for key in keys)))
        return {key.removeprefix(prefix): float(value)
                for key, value in zip(keys, packed, strict=True)}

    return read
