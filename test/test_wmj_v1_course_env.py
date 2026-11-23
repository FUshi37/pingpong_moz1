"""Course integration preserves the legacy path and causal pulse accounting."""

import jax
import jax.numpy as jnp
import numpy as np
from test_wmj_v1_control_projection import fixture
from test_wmj_v1_critic_features import setup_value
from test_wmj_v1_full_prior_world import components
from test_wmj_v1_world_control import setup_control
from world_model.imagined_env import init_imagined,imagined_step,EpisodeContract
from world_model.imagined_rewards import RewardContract
from world_model.course_env import init_course,course_step,course_critic_features,course_critic_layout
from world_model.critic_features import critic_features


def setup_course(fixture):
    world,task,events,context,_=setup_value(fixture)
    world=world._replace(physics=world.physics._replace(ball=world.physics.ball._replace(
        position_m=jnp.array([[0.,-.5,1.4]]),velocity_m_s=jnp.array([[.01,0.,.005]]))))
    base=init_imagined(world,context,task.goal_xyz,events=events)
    chain,_,_,_,control=setup_control(fixture)
    robot,auxiliary,_,options=components()
    kwargs=dict(options,chain=chain,control_contract=control,
        episode_contract=EpisodeContract(),reward_contract=RewardContract())
    return base,init_course(base,jax.random.PRNGKey(7)),robot,auxiliary,kwargs


def test_s01_preserves_legacy_outputs_and_appends_exactly_six_zero_features(fixture):
    base,state,robot,auxiliary,kwargs=setup_course(fixture)
    action=jnp.ones((1,7))*.01
    old,old_transition=imagined_step(robot,auxiliary,base,action,base.task.goal_xyz,
        ar1_enabled=False,**kwargs)
    new,transition=course_step(robot,auxiliary,state,action,base.task.goal_xyz,stage_id='S01',**kwargs)
    for name in base._fields:
        for a,b in zip(jax.tree.leaves(getattr(old,name)),jax.tree.leaves(getattr(new,name))):
            np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(transition.reward,old_transition.reward)
    expected=critic_features(old.world,old.task,old.events,old.context,old.episode_steps)
    actual=course_critic_features(new)
    np.testing.assert_array_equal(actual[:,:-6],expected)
    np.testing.assert_array_equal(actual[:,-6:],0.)
    assert actual.shape[-1]==course_critic_layout(new)['dimension']


def test_goal_apex_arms_pulse_at_substep_and_full_state_continues(fixture):
    _,state,robot,auxiliary,kwargs=setup_course(fixture)
    task=state.task._replace(hit_count=jnp.array([16]),apex_count=jnp.array([15]),
        flight_hit_index=jnp.array([16]),awaiting_apex=jnp.array([True]),
        stability_qualified=jnp.array([True]))
    state=state._replace(task=task,schedule=state.schedule._replace(goals_completed=jnp.array([2])))
    goal=jnp.array([[.03,-.5,1.4]])
    step=jax.jit(lambda s:course_step(robot,auxiliary,s,jnp.zeros((1,7)),goal,stage_id='S04',**kwargs))
    after,transition=step(state)
    assert int(transition.task_events['goal_switched'][0])==1
    assert int(transition.task_events['disturbance_started'][0])==1
    assert int(transition.task_events['disturbance_applied'][0])==4
    assert int(after.schedule.remaining_steps[0])==16
    assert 0.<=float(transition.task_events['pulse_onset_delay_s'][0])<=.001
    assert int(after.episode_steps[0])==1 and int(after.world.observation.camera.step_count[0])==1
    np.testing.assert_array_equal(after.task.goal_xyz,goal)
    total=4
    for _ in range(4):
        after,transition=step(after)
        total+=int(transition.task_events['disturbance_applied'][0])
    assert total==20 and int(after.schedule.remaining_steps[0])==0
    assert int(after.completed_goals[0])==1
    np.testing.assert_array_equal(after.schedule.force_world_n,0.)


def test_invalid_later_substep_cannot_commit_partial_task_or_pulse(fixture, monkeypatch):
    import world_model.course_env as module
    _, state, robot, auxiliary, kwargs = setup_course(fixture)
    state = state._replace(task=state.task._replace(hit_count=jnp.array([16]),
        apex_count=jnp.array([15]), flight_hit_index=jnp.array([16]),
        awaiting_apex=jnp.array([True]), stability_qualified=jnp.array([True])),
        schedule=state.schedule._replace(goals_completed=jnp.array([2])))
    original = module.forced_physics_step

    def invalid_after_apex(robot, auxiliary, physics, command, force, **options):
        predicted, trace = original(robot, auxiliary, physics, command, force, **options)
        position = jnp.where((physics.ball.velocity_m_s[:, 2] < 0.)[:, None],
                             jnp.nan, predicted.ball.position_m)
        return predicted._replace(ball=predicted.ball._replace(position_m=position)), trace

    monkeypatch.setattr(module, 'forced_physics_step', invalid_after_apex)
    after, transition = module.course_step(robot, auxiliary, state, jnp.zeros((1, 7)),
        jnp.array([[.03, -.5, 1.4]]), stage_id='S04', **kwargs)
    assert bool(after.model_truncated[0])
    for name in ('world', 'task', 'events', 'schedule', 'completed_goals', 'successful_goals'):
        for before, actual in zip(jax.tree.leaves(getattr(state, name)),
                                  jax.tree.leaves(getattr(after, name))):
            np.testing.assert_array_equal(before, actual)
    assert float(transition.reward[0]) == float(transition.discount[0]) == 0.
    assert int(transition.task_events['goal_switched'][0]) == 0
    assert int(transition.task_events['disturbance_started'][0]) == 0
