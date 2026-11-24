"""Default-off parameter constraint; deployment retains the same single MLP."""
from pathlib import Path
import hashlib

import jax.numpy as jnp
import numpy as np


def load_projection(folder: Path, config: dict):
    """Load hashed parent-specific wind-row constraint and origin."""
    path = folder / config['artifact']
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != config['sha256']:
            raise ValueError('Calm projection artifact hash mismatch')
    with np.load(path, allow_pickle=False) as packet:
        basis = packet['basis']
        origin = packet['origin']
    if (basis.ndim != 2 or basis.shape[0] != 564 or basis.shape[1] >= 564
            or origin.shape != (2, 282) or not np.isfinite(basis).all()
            or not np.isfinite(origin).all()):
        raise ValueError('Invalid or fully constrained wind projection')
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=2.e-5)
    return jnp.asarray(basis), jnp.asarray(origin)


def project_wind_optimizer(native, basis, origin):
    """Project the proposed wind weights; preserve all other Adam outputs.

    An affine projection retains all accepted motion in its nullspace. It
    never repeatedly rescales accepted progress toward the initial actor.
    """
    if basis is None:
        return native

    def step(params, grads, opt, *args, **kwargs):
        updated, after, norm = native(params, grads, opt, *args, **kwargs)
        delta = (updated['pi']['l1']['w'][114:116] - origin).reshape(-1)
        projected = delta - basis @ (basis.T @ delta)
        wind = origin + projected.reshape(origin.shape)
        pi = {**updated['pi'], 'l1': {**updated['pi']['l1'],
              'w': updated['pi']['l1']['w'].at[114:116].set(wind)}}
        return {**updated, 'pi': pi}, after, norm

    return step
