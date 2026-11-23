"""Residual-free integration preserves causal physical and observation timing."""

import jax
import jax.numpy as jnp
import numpy as np

from test_wmj_v1_control_projection import fixture
from test_wmj_v1_world_control import setup_control
from test_wmj_v1_world_physics import setup_physics
from test_wmj_v1_soft_contact import contract as soft_contract
from world_model.neural import init_mlp
from world_model.normal_coupling import acceleration_terms, own_robot_motion
from world_model.observation_engine import init_world_observation
from world_model.robot_model import RobotScale
from world_model.tangent_contact import tangent_contact_rollout
from world_model.full_prior_world import (
    FullPriorWorldState, full_prior_physics_step, full_prior_world_step,
    full_prior_world_rollout, measured_racket_pose,
)


def components():
    robot, _, state, options = setup_physics()
    auxiliary = dict(parameters=init_mlp(jax.random.PRNGKey(3), (47, 16, 6)),
        scale=RobotScale(jnp.zeros(47), jnp.ones(47), jnp.zeros(6), jnp.ones(6)))
    return robot, auxiliary, state, dict(options, contact_contract=soft_contract(),
                                       matmul_precision='highest')


def test_full_prior_recursion_matches_frozen_component_without_residual():
    robot, auxiliary, state, options = components()
    commands = {k:jnp.zeros((10, 1, 7)) for k in ('q', 'dq', 'ddq')}
    raw = dict(q_before=state.q_rad[None], dq_before=state.dq_rad_s[None],
        **{'delayed_'+k+'_command':v for k,v in commands.items()})
    with jax.default_matmul_precision('highest'):
        motion, expected_robot, _ = own_robot_motion(robot, raw,
            scale=options['robot_scale'], chain=options['chain'])
        terms = acceleration_terms(auxiliary, expected_robot['q_before_rad'],
                                    expected_robot['dq_before_rad_s'], commands)
        expected_ball, expected_trace = tangent_contact_rollout(None, options['surface'],
            state.ball, motion, terms, ball_radius_m=.02,
            position_scale=options['position_scale'], contract=options['contact_contract'], learned=False)
    step = lambda s,c:full_prior_physics_step(robot, auxiliary, s, c, **options)
    after, trace = jax.jit(lambda s:jax.lax.scan(step, s, commands))(state)
    for a,b in zip(after.ball, expected_ball):
        np.testing.assert_allclose(a, b, atol=3e-7, rtol=2e-6)
    np.testing.assert_allclose(after.q_rad, expected_robot['q_after_rad'][-1], atol=1e-8)
    np.testing.assert_array_equal(trace['contact']['entry'], expected_trace['entry'])
    np.testing.assert_array_equal(trace['contact']['release'], expected_trace['release'])
    assert np.isfinite(trace['solver_terms']).all()


def test_own_world_split_resume_camera_clock_and_pose_timing(fixture):
    chain, q, control, context, control_contract = setup_control(fixture)
    robot, auxiliary, physics, options = components()
    # The native final forward refreshes the site at the 5 ms encoder endpoint.
    physics = physics._replace(q_rad=q, dq_rad_s=jnp.ones_like(q)*.01,
        ball=physics.ball._replace(position_m=jnp.array([[0., -.5, 1.4]])))
    options = dict(options, chain=chain)
    observation, _ = init_world_observation(physics.ball.position_m, q,
        jax.random.split(jax.random.PRNGKey(10), 1),
        pose_from_q=lambda x:measured_racket_pose(chain, x), ar1_enabled=True)
    initial = FullPriorWorldState(physics, control, observation,
                                  measured_racket_pose(chain, q))
    goal = jnp.array([[0., -.5, 1.4]])
    actions = jnp.zeros((4, 1, 7))
    kwargs = dict(context=context, control_contract=control_contract,
                  task_goal_xyz=goal, ar1_enabled=True, **options)
    run = jax.jit(lambda s,a:full_prior_world_rollout(robot, auxiliary, s, a, **kwargs))
    whole, trace = run(initial, actions)
    middle, _ = run(initial, actions[:2])
    split, _ = run(middle, actions[2:])
    for a,b in zip(jax.tree.leaves(whole), jax.tree.leaves(split)):
        np.testing.assert_array_equal(a, b)
    assert trace['actor_observation'].shape == (4, 1, 969)
    np.testing.assert_array_equal(trace['observation_diagnostics']['camera_fresh'][:,0],
                                  [False, False, True, False])
    np.testing.assert_allclose(trace['cached_racket_pose'][0],
        measured_racket_pose(chain, q+.005*physics.dq_rad_s), atol=3e-7)
    assert np.max(np.abs(trace['cached_racket_pose'][0]
        -measured_racket_pose(chain, q+.004*physics.dq_rad_s))) > 1e-6
    from test_wmj_v1_control_projection import CPUData
    env, _, fixed, addresses, _, _ = fixture
    endpoint_q = np.asarray(trace['physics']['q_after_rad'][0,-1])
    native_q = np.broadcast_to(fixed, (1,len(fixed))).copy()
    native_q[:,addresses] = endpoint_q
    native = env.batched_forward(None, CPUData(native_q,np.zeros((1,env.nv))))
    rotation = np.asarray(native.site_xmat[:,env.racket_site_id]).reshape(-1,3,3)
    expected = np.concatenate((native.site_xpos[:,env.racket_site_id],
                               rotation[:,:,0],rotation[:,:,1]),axis=-1)
    np.testing.assert_allclose(trace['cached_racket_pose'][0],expected,atol=5e-7)
    assert int(whole.observation.history.paired.present.sum()) == 2


def test_goal_change_leaves_all_world_memory_untouched(fixture):
    from world_model.full_prior_world import world_actor_observation
    chain, q, control, _, _ = setup_control(fixture)
    _, _, physics, _ = components()
    physics = physics._replace(q_rad=q)
    obs, _ = init_world_observation(physics.ball.position_m, q,
        jax.random.split(jax.random.PRNGKey(8), 1),
        pose_from_q=lambda x:measured_racket_pose(chain, x), ar1_enabled=False)
    state = FullPriorWorldState(physics, control, obs, measured_racket_pose(chain, q))
    before = world_actor_observation(state, jnp.array([[0., -.5, 1.4]]))
    after = world_actor_observation(state, jnp.array([[.1, -.4, 1.35]]))
    changed = np.flatnonzero(np.asarray(before != after)[0])
    np.testing.assert_array_equal(changed, [69, 70, 72])


def test_cached_teacher_pose_accepts_time_and_lane_axes():
    from world_model.audit_full_prior_world import cached_pose_sequence
    position = np.arange(18,dtype=np.float32).reshape(3,2,3)
    rotation = np.broadcast_to(np.eye(3,dtype=np.float32),(3,2,3,3))
    actual = cached_pose_sequence(position,rotation)
    assert actual.shape == (3,2,9)
    np.testing.assert_array_equal(actual[...,:3],position)
    np.testing.assert_array_equal(actual[...,3:],np.broadcast_to([1,0,0,0,1,0],(3,2,6)))


def test_frozen_auxiliary_retains_command_gradient_without_contact_residual():
    robot, auxiliary, state, options = components()
    commands = {key:jnp.zeros((1,7)) for key in ('q','dq','ddq')}
    def value(x):
        after,_ = full_prior_physics_step(robot,auxiliary,state,
            dict(commands,q=commands['q'].at[0,0].set(x)),**options)
        return after.ball.velocity_m_s[0,0]
    gradient = jax.jit(jax.grad(value))(jnp.float32(.01))
    finite_difference = (value(jnp.float32(.02))-value(jnp.float32(0.)))/.02
    assert np.isfinite(gradient) and abs(float(gradient))>1e-8
    np.testing.assert_allclose(gradient,finite_difference,rtol=.05,atol=5e-7)


def test_teacher_seed_bank_is_causal_and_preserves_full_history(fixture):
    from world_model.full_prior_seeds import reconstruct_teacher_bank, select_teacher_seeds
    chain,q,control,context,control_contract = setup_control(fixture)
    _,_,physics,_ = components()
    physics = physics._replace(q_rad=q)
    obs,_ = init_world_observation(physics.ball.position_m,q,
        jax.random.split(jax.random.PRNGKey(8),1),
        pose_from_q=lambda x:measured_racket_pose(chain,x),ar1_enabled=False)
    initial = FullPriorWorldState(physics,control,obs,measured_racket_pose(chain,q))
    endpoints = jax.tree.map(lambda x:jnp.broadcast_to(x,(4,)+x.shape),physics)
    endpoints = endpoints._replace(q_rad=endpoints.q_rad+jnp.arange(1,5)[:,None,None]*.001,
        ball=endpoints.ball._replace(position_m=endpoints.ball.position_m+jnp.arange(1,5)[:,None,None]*.01))
    cached = jax.vmap(lambda x:measured_racket_pose(chain,x))(endpoints.q_rad)
    refs = jnp.broadcast_to(control.reference_q_rad,(4,1,7))
    accepted = jnp.zeros_like(refs)
    run = jax.jit(lambda ep:reconstruct_teacher_bank(initial,ep,cached,refs,accepted,
        context=context,control_contract=control_contract,chain=chain,ar1_enabled=False)[0])
    bank = run(endpoints)
    changed = run(endpoints._replace(q_rad=endpoints.q_rad.at[3].add(1.)))
    seed = select_teacher_seeds(bank,jnp.array([3]))
    alternate = select_teacher_seeds(changed,jnp.array([3]))
    for a,b in zip(jax.tree.leaves(seed),jax.tree.leaves(alternate)):
        np.testing.assert_array_equal(a,b)
    assert seed.observation.history.feedback.q_rad.shape == (1,129,13)
    assert int(seed.observation.history.paired.present.sum()) == 2
    np.testing.assert_allclose(seed.observation.history.paired.racket_pose[:,0],cached[2],atol=1e-7)
