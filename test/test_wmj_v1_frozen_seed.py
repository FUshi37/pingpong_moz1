"""Fixed evaluation restores the actual warmed state, not a rerun of warmup."""

from pathlib import Path
from typing import NamedTuple
import sys
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.frozen_seed import save_warmed_seed, load_warmed_seed
from world_model.resources import StorageBudget
from world_model.state_adapter import numeric_tree_digest


class Seed(NamedTuple):
    measured_q: object
    controller_queue: object
    phase: object
    rng: object


def test_warmed_cache_preserves_physics_queue_phase_rng_and_admission(tmp_path):
    seed=Seed(jnp.array([[.3,.7]]),jnp.arange(14.).reshape(1,2,7),jnp.array(3),jnp.array([[7,8]],jnp.uint32))
    context=dict(cases='fixed',physics='immutable',control='immutable',ids=[0])
    admission=[dict(case_id='nominal_00',status='ADMITTED')]
    ref=save_warmed_seed(tmp_path/'seed.json',seed,admission,context,StorageBudget(tmp_path,2**20,0))
    restored,rows=load_warmed_seed(ref,seed,context)
    assert numeric_tree_digest(restored)==numeric_tree_digest(seed)
    assert rows==admission
    np.testing.assert_array_equal(restored.phase,3)
    with pytest.raises(ValueError,match='context'):
        load_warmed_seed(ref,seed,dict(context,ids=[1]))
