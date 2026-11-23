"""Authenticated V321 actor migration without loading its legacy simulator."""

from __future__ import annotations

from collections import namedtuple
import copy
from pathlib import Path
import pickle

import jax.numpy as jnp
import numpy as np

from .contracts import authenticate, V321_SHA256

_LegacyOptimState = namedtuple('_LegacyOptimState', 'm v t')


class _ActorArchiveReader(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        # Importing train_juggle_mjx_ppo would also import the mutable current
        # simulator, contaminating later frozen-stack loading in this process.
        if (module, name) == ('train_juggle_mjx_ppo', 'OptimState'):
            return _LegacyOptimState
        return super().find_class(module, name)


def load_v321_actor(root: Path, contract: dict) -> dict:
    """Read the pinned release; return only copied actor weights, not critic/Adam."""
    if contract['sha256'] != V321_SHA256:
        raise ValueError('Source SHA-256 must be the registered V321 S4/u2097 release')
    path = authenticate(root, contract)
    with path.open('rb') as stream:
        payload = _ActorArchiveReader(stream).load()
    if ((payload['step'], payload['global_update'], payload['stage_update'])
            != (1609695232, 2097, 400)
            or payload['stage_name'] != 'fplc_v321_s4_1200s'
            or (payload['obs_dim'], payload['critic_obs_dim'], payload['act_dim'])
            != (72, 393, 7)
            or set(payload['params']) != {'pi', 'v', 'log_std'}):
        raise ValueError('Pinned archive does not implement the registered actor contract')
    actor = copy.deepcopy(payload['params']['pi'])
    _validate_actor(actor, 72)
    return actor


def _validate_actor(actor: dict, input_dim: int) -> None:
    expected = {'l1': (input_dim, 256), 'l2': (256, 256), 'out': (256, 7)}
    if set(actor) != set(expected):
        raise ValueError('Actor layer names changed')
    for name, shape in expected.items():
        for key, expected_shape in (('w', shape), ('b', (shape[1],))):
            value = np.asarray(actor[name][key])
            if (value.shape != expected_shape or value.dtype != np.float32
                    or not np.isfinite(value).all()):
                raise ValueError(f'Expected finite float32 {input_dim}-D actor: {name}/{key}')


def expand_actor_73(actor: dict) -> dict:
    """Preserve all original weights and append exactly one zero input row."""
    _validate_actor(actor, 72)
    expanded = {name: {key: jnp.asarray(value) for key, value in layer.items()}
                for name, layer in actor.items()}
    expanded['l1']['w'] = jnp.concatenate(
        (expanded['l1']['w'], jnp.zeros((1, 256), dtype=jnp.float32)), axis=0)
    return expanded


def expand_actor_history(actor: dict) -> dict:
    """Copy the real 72-D actor and zero all goal-Z/paired-history input weights."""
    from .paired_history import ACTOR_INPUT_DIM
    _validate_actor(actor, 72)
    expanded = {name: {key: jnp.asarray(value) for key, value in layer.items()}
                for name, layer in actor.items()}
    expanded['l1']['w'] = jnp.concatenate((expanded['l1']['w'],
        jnp.zeros((ACTOR_INPUT_DIM-72, 256), jnp.float32)), axis=0)
    return expanded


def actor_mean(actor: dict, observation):
    """Raw tanh/tanh/linear mean; clipping belongs to the control boundary."""
    x = jnp.tanh(jnp.asarray(observation) @ actor['l1']['w'] + actor['l1']['b'])
    x = jnp.tanh(x @ actor['l2']['w'] + actor['l2']['b'])
    return x @ actor['out']['w'] + actor['out']['b']
