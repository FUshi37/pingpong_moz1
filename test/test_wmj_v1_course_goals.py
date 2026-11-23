"""Goal sampling respectsXYZ distance and retains unknown far transitions."""

from pathlib import Path
import sys
import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.course_goals import build_goal_table,draw_next_goals,validate_start_goals
from world_model.sampling import SamplingShortage


def test_goal_distance_is3d_and_far_edges_require_explicit_evidence():
    xyz=np.array([[0.,-.5,1.3],[0.,-.5,1.375],[.06,-.5,1.375],[.3,-.5,1.375]],np.float32)
    table=build_goal_table(xyz,static_admitted=np.ones(4,bool))
    current=jnp.repeat(jnp.array(xyz[:1]),100,axis=0)
    keys=jax.random.split(jax.random.PRNGKey(4),100)
    for stage,expected in [('S01',0),('S02',1),('S03',2),('S04',None)]:
        goals,_,available,info=draw_next_goals(table,current,keys,stage_id=stage)
        assert np.all(available)
        if expected is not None:np.testing.assert_allclose(goals,np.broadcast_to(xyz[expected],goals.shape),atol=1e-7)
        else:assert not np.any(np.all(np.isclose(np.asarray(goals),xyz[3]),axis=-1))
    with pytest.raises(SamplingShortage):
        validate_start_goals(table,xyz[3:4],stage_id='S02')


def test_goal_frontier_fallback_is_logged_and_unknown_static_goals_are_rejected():
    xyz=np.array([[0.,-.5,1.3],[0.,-.5,1.35],[.03,-.5,1.3]],np.float32)
    with pytest.raises(ValueError,match='static'):
        build_goal_table(xyz,static_admitted=[True,False,True])
    table=build_goal_table(xyz,static_admitted=[True]*3)
    _,_,_,info=draw_next_goals(table,jnp.asarray(xyz[:1]),jax.random.split(jax.random.PRNGKey(3),1),stage_id='S02')
    np.testing.assert_allclose(info['mixture_weights'],[[1.,0.,0.]])
    table=table._replace(success_rate=jnp.array([0.,.5,.95]),
        improvement=jnp.array([0.,.1,.1]),observations=jnp.ones(3))
    _,_,_,info=draw_next_goals(table,jnp.asarray(xyz[:1]),jax.random.split(jax.random.PRNGKey(3),1),stage_id='S02')
    np.testing.assert_allclose(info['probability'],[[0.,.55,.45]],atol=1e-7)
