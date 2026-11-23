"""Independent compiled-MuJoCo pose and Jacobian oracle for learned dynamics."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.kinematics import extract_chain, forward_kinematics
from world_model.support_losses import racket_kinematics


def fixture():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body pos=".2 -.3 .8" euler=".2 -.1 .3">
        <body pos=".1 .2 .3" euler=".1 .3 -.2">
          <joint name="a" pos=".03 -.02 .01" axis="0 0 1" ref="12"/>
          <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02"/>
          <body pos=".2 0 0" euler="-.1 .2 .1">
            <joint name="b" pos="-.01 .04 0" axis="0 1 0"/>
            <geom type="capsule" fromto="0 0 0 .3 0 .1" size=".02"/>
            <body pos=".3 0 .1" euler=".2 .1 -.3">
              <geom name="racket" type="box" size=".1 .12 .01" pos=".03 .04 .05"
                    euler=".1 .2 .3"/>
              <site name="racket_site" pos=".01 -.02 .04" euler="-.1 .3 .2"/>
            </body>
          </body>
        </body>
      </body></worldbody></mujoco>''')
    return model, mujoco.MjData(model)


def test_compiled_chain_matches_cpu_pose_and_independent_jacobians():
    model, data = fixture()
    chain = extract_chain(model, model.qpos0, joint_ids=[0, 1], site_id=0, geom_id=2)
    random = np.random.default_rng(20261105)
    for q in random.uniform(-1., 1., (12, 2)):
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        result = forward_kinematics(chain, jnp.asarray(q, jnp.float32))
        np.testing.assert_allclose(result.site_position, data.site_xpos[0], atol=3e-7)
        np.testing.assert_allclose(result.site_rotation, data.site_xmat[0].reshape(3, 3), atol=3e-7)
        np.testing.assert_allclose(result.geom_position, data.geom_xpos[2], atol=3e-7)
        np.testing.assert_allclose(result.geom_rotation, data.geom_xmat[2].reshape(3, 3), atol=3e-7)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, 0)
        np.testing.assert_allclose(result.site_jacobian, jp, atol=3e-7)
        np.testing.assert_allclose(result.angular_jacobian, jr, atol=3e-7)


def test_fk_derivatives_match_analytic_jacobian_and_jit_batch():
    model, _ = fixture()
    chain = extract_chain(model, model.qpos0, joint_ids=[0, 1], site_id=0, geom_id=2)
    q = jnp.array([.4, -.3])
    fk = forward_kinematics(chain, q)
    derivative = jax.jacfwd(lambda value: forward_kinematics(chain, value).site_position)(q)
    np.testing.assert_allclose(derivative, fk.site_jacobian, atol=2e-7)
    batch = jax.jit(jax.vmap(lambda value: forward_kinematics(chain, value)))(jnp.stack((q, q)))
    np.testing.assert_allclose(batch.geom_position, jnp.stack((fk.geom_position, fk.geom_position)), atol=2e-7)


def test_full_chain_pose_twist_handles_joint_dimension_without_arm_only_reshape():
    model, data = fixture()
    chain = extract_chain(model, model.qpos0, joint_ids=[0, 1], site_id=0, geom_id=2)
    q, dq = jnp.array([.4, -.3]), jnp.array([.2, .5])
    result = racket_kinematics(chain, q[None, None], dq[None, None])
    data.qpos[:], data.qvel[:] = np.asarray(q), np.asarray(dq)
    mujoco.mj_forward(model, data)
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacGeom(model, data, jp, jr, 2)
    np.testing.assert_allclose(result['position'][0, 0], data.site_xpos[0], atol=3e-7)
    np.testing.assert_allclose(result['velocity'][0, 0], jp@data.qvel, atol=3e-7)
    np.testing.assert_allclose(result['omega'][0, 0], jr@data.qvel, atol=3e-7)


def test_fixed_context_slide_uses_its_calibrated_pose():
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 1">
      <joint type="slide" axis="0 0 1" ref=".1"/>
      <geom type="sphere" size=".1"/>
      <body pos=".1 0 0"><joint type="hinge" axis="0 1 0"/>
        <geom type="box" size=".1 .1 .01"/>
        <site pos=".2 0 0"/>
      </body></body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qpos[:] = [.14, .3]
    mujoco.mj_forward(model, data)
    chain = extract_chain(model, data.qpos, joint_ids=[1], site_id=0, geom_id=1)
    prediction = forward_kinematics(chain, jnp.array([.3]))
    np.testing.assert_allclose(prediction.site_position, data.site_xpos[0], atol=2e-7)
