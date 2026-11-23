"""Full-chain projection is paired with the frozen method and CPU MuJoCo FK."""

import __future__
import ast
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

import cartesian_pose_action_projection_jax as projection
from world_model.audit_support_replay import load_support_chain
from world_model.contracts import authenticate, file_hash, load_config, repository_root
from world_model.control_projection import ProjectionContract, project_reference


@dataclass
class CPUData:
    qpos: object
    qvel: object
    site_xpos: object = None
    site_xmat: object = None
    xaxis: object = None
    xanchor: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@pytest.fixture(scope='module')
def fixture():
    cfg, root = load_config(), repository_root()
    atlas = json.loads((root/'pingpong_controller/outputs/rl_sim/wmj_v1/domain_atlas_20261106_a01.json').read_text())
    model, chain = load_support_chain(cfg, atlas['geometry'])
    fixed = np.array(atlas['geometry']['reset_fk']['qpos'])
    ids = np.array(chain.joint_ids)
    addresses = model.jnt_qposadr[ids]
    native_cfg = json.loads(authenticate(root, cfg['authenticated_inputs']['v324_complete_env']).read_text())['binding']['env_cfg']
    data = mujoco.MjData(model)
    data.qpos[:] = fixed
    mujoco.mj_forward(model, data)
    lower, upper = (jnp.array(cfg['domain'][key], jnp.float32) for key in ('racket_xyz_lower_m', 'racket_xyz_upper_m'))
    position_margin = native_cfg['cartesian_pose_position_margin_m']
    inner_rp = jnp.deg2rad(jnp.array([1.9, 1.9], jnp.float32))
    env = SimpleNamespace(dt=.005, nv=model.nv, arm_qadr=addresses[:7], arm_vadr=model.jnt_dofadr[ids[:7]],
        arm_jids=ids[:7], racket_site_id=atlas['geometry']['racket']['site_id'],
        arm_lo=jnp.array(model.jnt_range[ids[:7], 0], jnp.float32),
        arm_hi=jnp.array(model.jnt_range[ids[:7], 1], jnp.float32),
        arm_vel_limit_rad_s=jnp.deg2rad(jnp.array([210., 210., 240., 240., 300., 300., 300.])),
        recovered_rmp_qvel_reference_error_horizon_s_per_joint=jnp.ones(7)*10.,
        cartesian_world_to_policy_rotation=jnp.eye(3), cartesian_policy_origin_world=jnp.zeros(3),
        fixed_contact_racket_reference_rotation=jnp.array(data.site_xmat[0].reshape(3, 3), jnp.float32),
        cartesian_pose_position_lower=lower+position_margin, cartesian_pose_position_upper=upper-position_margin,
        cartesian_pose_outer_position_lower=lower, cartesian_pose_outer_position_upper=upper,
        cartesian_pose_roll_pitch_limit=inner_rp, cartesian_pose_outer_roll_pitch_limit=jnp.deg2rad(jnp.array([2., 2.])),
        cfg=SimpleNamespace(**native_cfg))

    def forward(unused_model, state):
        fields = {name: [] for name in ('site_xpos', 'site_xmat', 'xaxis', 'xanchor')}
        for q, dq in zip(np.asarray(state.qpos), np.asarray(state.qvel)):
            data.qpos[:], data.qvel[:] = q, dq
            mujoco.mj_forward(model, data)
            for name in fields:
                fields[name].append(getattr(data, name).copy())
        return state.replace(**{name: jnp.array(np.stack(values), jnp.float32) for name, values in fields.items()})

    env.batched_forward = forward
    source = authenticate(root, cfg['authenticated_inputs']['xml']).parent/'mjx_juggle_env.py'
    assert file_hash(source) == '39ba2da83f498cb84b041e016d6f1031130007dad9efafc21ead388f214e3b09'
    method = next(node for node in ast.walk(ast.parse(source.read_text()))
                  if isinstance(node, ast.FunctionDef) and node.name == '_project_cartesian_pose_reference')
    namespace = dict(jnp=jnp, matrix_to_euler_xyz_jax=projection.matrix_to_euler_xyz_jax,
        relative_xyz_euler_rate_jacobian_jax=projection.relative_xyz_euler_rate_jacobian_jax,
        project_cartesian_pose_qvel_jax=projection.project_cartesian_pose_qvel_jax)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec',
                 flags=__future__.annotations.compiler_flag), namespace)
    references = [fixed[addresses[:7]]]
    for row in atlas['rows']:
        if row['status'] == 'STATIC_POSE_FEASIBLE':
            references.append(next(a['q_rad'] for a in row['attempts']
                if a.get('self_collision', {}).get('status') == 'PASS'))
    return env, chain, fixed, addresses, np.asarray(references), namespace['_project_cartesian_pose_reference']


def test_full_chain_projection_matches_frozen_method_with_independent_fk(fixture):
    env, chain, fixed, addresses, references, original = fixture
    rng = np.random.default_rng(2026110702)
    n = len(references)
    qpos = np.broadcast_to(fixed, (n, len(fixed))).copy()
    qpos[:, addresses[:7]] = references
    qpos[:, addresses[7:]] += rng.normal(0., .002, (n, 6))
    measured = jnp.array(qpos[:, addresses], jnp.float32)
    reference = jnp.array(references, jnp.float32)
    velocity = jnp.array(rng.uniform(-1., 1., (n, 7)), jnp.float32)*env.arm_vel_limit_rad_s
    state = SimpleNamespace(arm_cmd_q=reference, model=None,
        data=CPUData(jnp.array(qpos, jnp.float32), jnp.zeros((n, env.nv))))
    expected = original(env, state, measured[:, :7], velocity)
    contract = ProjectionContract.from_native(env)
    actual = jax.jit(lambda q, ref, v: project_reference(chain, contract, q, ref, v))(measured, reference, velocity)
    for i, (left, right) in enumerate(zip(actual, expected)):
        if np.asarray(right).dtype == bool:
            np.testing.assert_array_equal(left, right)
        else:
            np.testing.assert_allclose(left, right, atol=1e-3 if i in (1, 2, 3) else 1e-5, rtol=1e-4)
    assert actual.fallback_to_hold.any()
    assert (~actual.fallback_to_hold).any()


def test_projection_action_gradient_and_hold_behavior(fixture):
    env, chain, fixed, addresses, _, _ = fixture
    contract = ProjectionContract.from_native(env)
    q = jnp.array(fixed[addresses][None], jnp.float32)
    ref = q[:, :7]
    zero = jnp.zeros((1, 7))
    fun = jax.jit(lambda v: project_reference(chain, contract, q, ref, v).reference_q)
    derivative = jax.jacfwd(fun)(zero)
    epsilon = .01
    finite = np.stack([np.asarray((fun(zero+jnp.eye(7)[i][None]*epsilon)
        -fun(zero-jnp.eye(7)[i][None]*epsilon))/(2*epsilon)) for i in range(7)], axis=-1)
    assert np.isfinite(derivative).all()
    np.testing.assert_allclose(np.asarray(derivative)[:, :, 0, :], finite, atol=2e-5, rtol=.02)
    unsafe = q.at[:, 0].add(.5)
    result = project_reference(chain, contract, unsafe, unsafe[:, :7], jnp.ones((1, 7)))
    assert bool(result.fallback_to_hold[0])
    np.testing.assert_array_equal(result.reference_q, unsafe[:, :7])
