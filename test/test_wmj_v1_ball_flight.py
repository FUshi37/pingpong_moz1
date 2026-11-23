"""Known free-flight dynamics and proposal geometry have independent oracles."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.ball_flight import free_flight_substep, ballistic_apex, launch_to_contact, velocity_for_apex


def test_five_free_flight_substeps_match_mujoco_implicitfast():
    model = mujoco.MjModel.from_xml_string('''<mujoco>
      <option timestep=".001" integrator="implicitfast" gravity="0 0 -9.81"/>
      <worldbody><body pos="0 0 1.2"><freejoint/>
      <geom type="sphere" size=".02" mass=".004" contype="0" conaffinity="0"/>
      </body></worldbody></mujoco>''')
    data = mujoco.MjData(model)
    data.qvel[:3] = [.23, -.12, -1.4]
    position, velocity = jnp.asarray(data.qpos[:3]), jnp.asarray(data.qvel[:3])
    step = jax.jit(free_flight_substep)
    for _ in range(5):
        mujoco.mj_step(model, data)
        position, velocity = step(position, velocity)
    np.testing.assert_allclose(position, data.qpos[:3], rtol=0, atol=3e-7)
    np.testing.assert_allclose(velocity, data.qvel[:3], rtol=0, atol=3e-7)


def test_proposed_contact_launch_solves_the_continuous_ballistic_equations():
    pc, vc, tau = jnp.array([.1, -.4, 1.08]), jnp.array([.2, -.1, -2.]), .08
    p0, v0 = launch_to_contact(pc, vc, tau)
    gravity = jnp.array([0., 0., -9.81])
    np.testing.assert_allclose(p0+tau*v0+.5*gravity*tau**2, pc, atol=1e-7)
    np.testing.assert_allclose(v0+gravity*tau, vc, atol=1e-7)


def test_required_outgoing_velocity_predicts_requested_xyz_apex():
    hit = jnp.array([[0., -.5, 1.08], [.1, -.4, 1.1]])
    target = jnp.array([[.15, -.45, 1.4], [-.1, -.6, 1.35]])
    velocity, valid = velocity_for_apex(hit, target)
    apex, time, apex_valid = ballistic_apex(hit, velocity)
    assert bool(jnp.all(valid & apex_valid & (time > 0)))
    np.testing.assert_allclose(apex, target, atol=2e-7)
    derivative = jax.jacfwd(lambda x: ballistic_apex(hit[0], x)[0])(velocity[0])
    assert np.isfinite(derivative).all() and np.linalg.norm(derivative) > 0


def test_invalid_apex_candidates_are_masked_without_nan_or_fabricated_launch():
    p = jnp.array([0., -.5, 1.1])
    velocity, valid = velocity_for_apex(p, p)
    assert not valid and np.array_equal(velocity, np.zeros(3))
    apex, time, valid = ballistic_apex(p, jnp.array([0., 0., -1.]))
    assert not valid and time == 0
    np.testing.assert_array_equal(apex, p)
