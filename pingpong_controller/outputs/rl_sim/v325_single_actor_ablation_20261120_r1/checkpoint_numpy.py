"""Inspect experimental checkpoints without importing JAX or any simulator."""
from collections import namedtuple
import pickle


OptimState = namedtuple('OptimState', 'm v t')
ActorOptim = namedtuple('ActorOptim', 'm v t actor_t')


def reconstruct_numpy(fun, args, state, aval_state):
    value = fun(*args)
    value.__setstate__(state)
    return value


class NumpyCheckpoint(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'train_juggle_mjx_ppo' and name == 'OptimState':
            return OptimState
        if module == 'single_actor' and name == 'ActorOptim':
            return ActorOptim
        if module == 'jax._src.array' and name == '_reconstruct_array':
            return reconstruct_numpy
        return super().find_class(module, name)
