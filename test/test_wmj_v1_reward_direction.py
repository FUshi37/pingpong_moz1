"""Paired reward direction uses realized events, failure masks and all sources."""
from pathlib import Path
import sys
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_reward_comparison_keeps_unresolved_reversed_and_failed_sources():
    from world_model.reward_direction import compare_returns
    def run(values):return {'candidate':values,'old':[0.]*4,'hits':[1,1,0,0],'failed':[False,False,True,True]}
    n={'base':run([0,0,0,0]),'repeat':run([0,0,0,.2]),'plus':run([1,-1,0,.1]),'minus':run([-1,1,0,-.1])}
    w={'base':run([0,0,0,0]),'repeat':run([0,0,0,0]),'plus':run([1,1,0,1]),'minus':run([-1,-1,0,-1])}
    criteria=dict(return_difference_floor=1e-4,repeat_multiplier=3.,minimum_resolved=2,minimum_direction_fraction=.9)
    d=compare_returns(n,w,criteria)
    assert d['candidate']['resolved_indices']==[0,1]
    assert d['candidate']['unresolved_indices']==[2,3]
    assert d['candidate']['direction_agreement_fraction']==.5
    assert not d['candidate']['diagnostic_pass']


def test_metric_observer_keeps_exact_physical_apex_and_no_fake_launch_success():
    from world_model.reward_direction import metric_task
    from world_model.imagined_env import task_substep
    from world_model.task import init_task
    from world_model.imagined_rewards import RewardContract
    task=init_task(jnp.array([[0.,-.5,1.4]]))
    labels=dict(impact_id=jnp.array([1]),contact_release=jnp.array([True]),controlled_release=jnp.array([True]),
      release_position_m=jnp.array([[0.,-.5,1.1]]),release_velocity_m_s=jnp.array([[.1,0.,2.]]),
      actual_apex=jnp.array([False]),apex_position_m=jnp.array([[.02,-.5,1.4]]),apex_velocity_m_s=jnp.array([[.1,0.,0.]]))
    observer=metric_task(task_substep)
    after,terms,metrics=observer(task,labels,task.goal_xyz,contract=RewardContract())
    original=task_substep(task,labels,task.goal_xyz,contract=RewardContract())
    for a,b in zip(jax.tree.leaves(after),jax.tree.leaves(original[0])):np.testing.assert_array_equal(a,b)
    assert metrics['valid_release'][0] and not metrics['valid_apex'][0]
    np.testing.assert_array_equal(metrics['metric_actual_apex_position_m'],[[0.,0.,0.]])
    labels=dict(labels,contact_release=jnp.array([False]),actual_apex=jnp.array([True]))
    final,_,m=observer(after,labels,task.goal_xyz,contract=RewardContract())
    np.testing.assert_array_equal(m['metric_actual_apex_position_m'],labels['apex_position_m'])
    assert int(final.apex_count[0])==1
    _,_,empty=observer(task,labels,task.goal_xyz,contract=RewardContract())
    assert not empty['valid_apex'][0]


def test_private_course_observation_preserves_control_and_numeric_metrics():
    from world_model.reward_direction import observed_course
    from world_model.projection_precision import make_runtime
    from world_model.course_env import course_step
    runtime=make_runtime(None,None,{},None);step=observed_course(runtime)
    assert step.__globals__['control_step'] is runtime.controller
    assert step.__globals__['physical_bounds'] is runtime.bounds
    assert step.__globals__['task_substep'] is not course_step.__globals__['task_substep']
    assert 'metric_keys' in step.__globals__


def test_preparation_stops_before_any_rollout_and_keeps_authentication():
    from world_model.reward_direction import prepare_source
    from world_model.audit_free_imagination import audit
    tree=ast.parse(prepare_source(inspect.getsource(audit)))
    code=ast.unparse(tree)
    assert 'restore_native_sources' in code and 'load_training_bank' in code
    assert 'assess_native_pair' not in code and 'range(128)' not in code
    assert 'initial_course._replace(world=initial, context=context)' in code
