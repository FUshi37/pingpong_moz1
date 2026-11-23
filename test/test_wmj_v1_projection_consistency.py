"""Frozen projection probes preserve the native carry and constraint contract."""
import inspect
import sys
from pathlib import Path
import ast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_capture_retains_native_actions_and_only_appends_diagnostics():
    from world_model.audit_free_imagination import audit
    from world_model.projection_consistency import capture_source
    tree = ast.parse(capture_source(inspect.getsource(audit)))
    collect = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    original = next(n for n in ast.walk(ast.parse(inspect.getsource(audit)))
                    if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    assert ast.dump(ast.Module(body=collect.body[:-1], type_ignores=[])) == ast.dump(
        ast.Module(body=original.body[:-1], type_ignores=[]))
    assert ast.dump(collect.body[-1].value.elts[:-1][0]) == ast.dump(original.body[-1].value.elts[0])
    assert 'metrics' in ast.unparse(collect.body[-1])
    assert 'native_tail=native' in ast.unparse(tree)


def test_quaternion_candidate_preserves_projection_bounds_and_shared_functions():
    from test_wmj_v1_control_projection import fixture
    from world_model.control_kinematics import extract_control_chain
    from world_model.audit_support_replay import load_support_chain
    from world_model.contracts import load_config, repository_root
    from world_model.control_projection import ProjectionContract
    from world_model.free_imagination import make_projector
    from world_model.projection_consistency import make_quaternion_projector
    import json
    import jax
    import jax.numpy as jnp
    import numpy as np
    env, chain, fixed, addresses, _, _ = fixture.__wrapped__()
    geometry = json.loads((repository_root()/'pingpong_controller/outputs/rl_sim/wmj_v1/domain_atlas_20261106_a01.json').read_text())['geometry']
    model, _ = load_support_chain(load_config(), geometry)
    quat = extract_control_chain(model, fixed, joint_ids=chain.joint_ids,
        site_id=geometry['racket']['site_id'], geom_id=geometry['racket']['geom_id'])
    contract = ProjectionContract.from_native(env)
    q = jnp.asarray(fixed[addresses][None], jnp.float32)
    reference = q[:, :7]
    baseline = make_projector()
    candidate = make_quaternion_projector(quat)
    for action in (jnp.zeros((1, 7)), jnp.ones((1, 7))):
        a = baseline(chain, contract, q, reference, action)
        b = candidate(chain, contract, q, reference, action)
        for x, y in zip(a, b):
            np.testing.assert_allclose(x, y, atol=1e-4, rtol=1e-4)
    bad = q.at[:, 0].add(1.)
    held = candidate(chain, contract, bad, bad[:, :7], jnp.ones((1, 7)))
    assert held.fallback_to_hold[0]
    np.testing.assert_array_equal(held.reference_q, bad[:, :7])
    grad = jax.jacfwd(lambda v: candidate(chain, contract, q, reference, v).reference_q)(jnp.zeros((1, 7)))
    assert np.isfinite(grad).all()


def test_coordinate_precision_is_local_and_retains_native_projector_code():
    import jax
    import jax.numpy as jnp
    from world_model.free_imagination import make_projector
    from world_model.projection_precision import coordinate_precision, make_projector as fixed_projector
    original = make_projector()
    fixed = fixed_projector()
    assert fixed.__code__.co_code == original.__code__.co_code
    assert fixed.__globals__['projection_v324'].__code__.co_code == original.__globals__['projection_v324'].__code__.co_code
    assert original.__globals__['jnp'] is jnp
    expression = jax.make_jaxpr(lambda a,b: coordinate_precision.einsum('ij,nj->ni',a,b))(
        jnp.eye(3),jnp.ones((32,3)))
    dots = [e for e in expression.jaxpr.eqns if e.primitive.name=='dot_general']
    assert dots and all(e.params['precision']==(jax.lax.Precision.HIGHEST,)*2 for e in dots)


def test_coordinate_candidate_reaches_both_recursions_without_global_patch():
    from world_model import course_env, course_control, full_prior_world
    from world_model.projection_precision import make_runtime, coordinate_precision
    before = (course_env.course_step, course_control.control_step, full_prior_world.full_prior_world_step)
    runtime = make_runtime(None, None, {}, None)
    assert runtime.world_step.__globals__['control_step'] is runtime.controller
    assert runtime.course_step.__globals__['control_step'] is runtime.controller
    assert runtime.controller.__globals__['project_reference'] is runtime.projector
    assert runtime.projector.__globals__['jnp'] is coordinate_precision
    assert (course_env.course_step, course_control.control_step, full_prior_world.full_prior_world_step) == before
