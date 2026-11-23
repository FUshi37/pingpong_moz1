"""Native quaternion ancestry FK agrees with independent compiled-model geometry."""

import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.audit_support_replay import load_support_chain
from world_model.contracts import load_config, repository_root
from world_model.control_kinematics import extract_control_chain, forward_control_kinematics


def test_native_quaternion_chain_matches_mujoco_and_derivatives():
    cfg = load_config()
    atlas = json.loads((repository_root()/'pingpong_controller/outputs/rl_sim/wmj_v1/domain_atlas_20261106_a01.json').read_text())
    model, old = load_support_chain(cfg, atlas['geometry'])
    fixed = np.asarray(atlas['geometry']['reset_fk']['qpos'])
    chain = extract_control_chain(model, fixed, joint_ids=old.joint_ids, site_id=0, geom_id=64)
    address = model.jnt_qposadr[np.asarray(chain.joint_ids)]
    velocity_address = model.jnt_dofadr[np.asarray(chain.joint_ids)]
    rng = np.random.default_rng(2026110906)
    q = fixed[address]+rng.uniform(-.05, .05, (16, 13))
    # This is a CPU numerical oracle, not a compilation/performance qualification.
    predicted = jax.vmap(lambda x: forward_control_kinematics(chain, x))(jnp.asarray(q, jnp.float32))
    data = mujoco.MjData(model)
    for i, values in enumerate(q):
        data.qpos[:], data.qpos[address] = fixed, values
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, 0)
        for actual, expected in ((predicted.site_position[i], data.site_xpos[0]),
            (predicted.site_rotation[i], data.site_xmat[0].reshape(3,3)),
            (predicted.geom_position[i], data.geom_xpos[64]),
            (predicted.geom_rotation[i], data.geom_xmat[64].reshape(3,3)),
            (predicted.site_jacobian[i], jp[:, velocity_address]),
            (predicted.angular_jacobian[i], jr[:, velocity_address])):
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=1e-5)
    point = jnp.asarray(q[0], jnp.float32)
    derivative = jax.jacfwd(lambda x: forward_control_kinematics(chain,x).site_position)(point)
    np.testing.assert_allclose(derivative, predicted.site_jacobian[0], atol=2e-6, rtol=1e-5)
