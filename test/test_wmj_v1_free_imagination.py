"""Private attitude removal must reach control, world and episode boundaries."""

import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from test_wmj_v1_control_projection import fixture


def test_free_projection_matches_private_native_method_and_keeps_old_path(fixture):
    from world_model.control_projection import ProjectionContract, project_reference
    from world_model.free_orientation import remove_reference_attitude_backstops, position_only_projection
    from world_model.free_imagination import make_projector
    from test_wmj_v1_control_projection import CPUData
    env, chain, fixed, addresses, refs, original = fixture
    # A formerly illegal tilt has no influence on candidate admission. Position
    # bounds remain independently evaluated by both implementations.
    q = np.broadcast_to(fixed, (8, len(fixed))).copy()
    q[:, addresses[:7]] = refs[:8]
    q[:, addresses[5]] += np.linspace(-.3, .3, 8)
    measured = jnp.asarray(q[:, addresses], jnp.float32)
    reference = measured[:, :7]
    velocity = jnp.full((8, 7), .05)
    contract = ProjectionContract.from_native(env)
    before = project_reference(chain, contract, measured, reference, velocity)
    candidate = make_projector()
    got = candidate(chain, contract, measured, reference, velocity)
    # Reconstruct the authenticated native function from its original file.
    import ast
    from world_model.contracts import authenticate, load_config, repository_root
    path = authenticate(repository_root(), load_config()['authenticated_inputs']['xml']).parent / 'mjx_juggle_env.py'
    node = next(n for n in ast.walk(ast.parse(path.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == '_project_cartesian_pose_reference')
    scope = dict(original.__globals__, project_cartesian_pose_qvel_jax=position_only_projection,
                 relative_xyz_euler_rate_jacobian_jax=lambda a, r, w: jnp.zeros_like(w))
    exec('from __future__ import annotations\n' + remove_reference_attitude_backstops(ast.unparse(node)), scope)
    state = SimpleNamespace(arm_cmd_q=reference, model=None,
        data=CPUData(jnp.asarray(q, jnp.float32), jnp.zeros((8, env.nv))))
    expected = scope[original.__name__](env, state, reference, velocity)
    for actual, wanted in zip(got, expected):
        np.testing.assert_allclose(actual, wanted, atol=1e-3, rtol=1e-4)
    after = project_reference(chain, contract, measured, reference, velocity)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    assert np.all(np.asarray(got.orientation_violation) == 0.)
    assert np.any(np.asarray(before.orientation_violation) > .01)


def test_runtime_dependencies_reach_both_steps_without_patching_defaults():
    from world_model import course_env, course_control, full_prior_world
    from world_model.free_imagination import make_runtime
    old = (course_env.course_step, course_control.control_step, full_prior_world.full_prior_world_step)
    runtime = make_runtime(None, None, {}, None)
    assert runtime.course_step.__globals__['control_step'] is runtime.controller
    assert runtime.world_step.__globals__['control_step'] is runtime.controller
    assert runtime.course_step.__globals__['physical_bounds'] is runtime.bounds
    assert (course_env.course_step, course_control.control_step, full_prior_world.full_prior_world_step) == old
    assert runtime.controller.__globals__['project_reference'] is runtime.projector


def test_selected_seed_authenticates_original_arrays_and_shared_clock():
    from world_model.free_imagination import select_verified_state
    from world_model.state_adapter import pack_numeric_state
    state = dict(native=dict(q=jnp.arange(12, dtype=jnp.float32).reshape(4, 3), clock=jnp.array(2)))
    arrays, manifest = pack_numeric_state(state, context_sha256='test')
    template = dict(q=jnp.zeros((2, 3), jnp.float32), clock=jnp.array(0))
    got = select_verified_state(template, arrays, manifest, prefix="['native']", lanes=[3, 1],
                                original_count=4, context_sha256='test')
    np.testing.assert_array_equal(got['q'], [[9, 10, 11], [3, 4, 5]])
    assert int(got['clock']) == 2
    arrays['leaf_000001'][0, 0] += 1
    with pytest.raises(ValueError, match='integrity'):
        select_verified_state(template, arrays, manifest, prefix="['native']", lanes=[3, 1],
                              original_count=4, context_sha256='test')


def test_native_endpoint_observer_is_traceable_without_host_seed_restore():
    from world_model.audit_free_imagination import endpoint_bounds
    from world_model.imagined_env import EpisodeContract
    env = SimpleNamespace(arm_qadr=np.arange(7), ball_qadr=7)
    adapter = SimpleNamespace(_values=lambda native: (None, native.pose))
    control = SimpleNamespace(projection=SimpleNamespace(reference_rotation=jnp.eye(3),
        q_lower=-jnp.ones(7), q_upper=jnp.ones(7)))
    run = jax.jit(lambda q, p: endpoint_bounds(env,
        SimpleNamespace(data=SimpleNamespace(qpos=q),pose=p),adapter,control,EpisodeContract()))
    q = jnp.array([[0.]*7+[0.,-.5,1.3]])
    pose = jnp.array([[0.,-.5,1.1,1.,0.,0.,0.,0.,1.]])
    okay, _ = run(q, pose)
    assert bool(okay[0])
    okay, _ = run(q, pose.at[0,0].set(1.))
    assert not bool(okay[0])
