"""Causal multirate history: one ball sample, one matching racket pose."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.paired_history import (
    HISTORY_INPUT_DIM, init_feedback, init_history, append_feedback,
    append_ball, encode_history, reset_selected, realign_pending,
)
from world_model.observation_engine import init_camera, advance_camera


def pose_from_q(q):
    angle = q[..., 0]
    zero = jnp.zeros_like(angle)
    return jnp.stack((angle**2, zero, zero,
        jnp.cos(angle), jnp.sin(angle), zero,
        -jnp.sin(angle), jnp.cos(angle), zero), axis=-1)


def feedback_at(times_us):
    state = init_feedback(1)
    for time_us in times_us:
        q = jnp.full((1, 7), time_us / 10000., jnp.float32)
        state, accepted = append_feedback(state, q, jnp.array([time_us]), jnp.array([time_us]))
        assert bool(accepted[0])
    return state


def sample(history, feedback, source_us, now_us, *, fresh=True):
    return append_ball(history, feedback, jnp.array([[.01, -.5, 1.4]]),
        jnp.array([source_us]), jnp.array([now_us]), jnp.array([fresh]),
        jnp.array([True]), pose_from_q)


def test_sample_time_alignment_interpolates_joints_before_fk():
    feedback = feedback_at([0, 5000, 10000])
    history, report = sample(init_history(1), feedback, 7500, 10000)
    assert bool(report['inserted'][0])
    np.testing.assert_allclose(history.racket_pose[0, 0],
                               pose_from_q(jnp.full((7,), .75)), atol=1e-7)
    # Squared position distinguishes FK(q_interp) from averaging FK endpoints.
    np.testing.assert_allclose(history.racket_pose[0, 0, 0], .5625, atol=1e-7)
    encoded = encode_history(history, jnp.array([10000])).reshape(1, 64, 14)
    np.testing.assert_allclose(encoded[0, 0, 12], .0025/.640, atol=1e-7)
    assert encoded[0, 0, 13] == 1


def test_no_future_feedback_or_extrapolated_racket_pose_enters_actor():
    feedback = feedback_at([0, 5000])
    before = feedback
    feedback, accepted = append_feedback(feedback, jnp.ones((1, 7)),
        jnp.array([10000]), jnp.array([7500]))
    assert not bool(accepted[0])
    for a, b in zip(jax.tree.leaves(feedback), jax.tree.leaves(before)):
        np.testing.assert_array_equal(a, b)
    history, _ = sample(init_history(1), feedback, 7500, 7500)
    assert not bool(history.aligned[0, 0])
    assert np.all(np.asarray(encode_history(history, jnp.array([7500]))) == 0.)
    feedback, _ = append_feedback(feedback, jnp.ones((1, 7)), jnp.array([10000]), jnp.array([10000]))
    history = realign_pending(history, feedback, jnp.array([10000]), pose_from_q)
    assert bool(history.aligned[0, 0])
    assert int(history.sample_time_us[0, 0]) == 7500


def test_hold_duplicate_and_out_of_order_packets_do_not_add_history():
    feedback = feedback_at([0, 5000, 10000])
    history, _ = sample(init_history(1), feedback, 10000, 10000)
    for time_us, fresh in ((10000, False), (10000, True), (5000, True)):
        after, report = sample(history, feedback, time_us, 10000, fresh=fresh)
        assert not bool(report['inserted'][0])
        for a, b in zip(jax.tree.leaves(after), jax.tree.leaves(history)):
            np.testing.assert_array_equal(a, b)


def test_640ms_history_tracks_only_fresh_90hz_samples_not_200hz_holds():
    camera = init_camera(jnp.array([[0., -.5, 1.4]]),
                         jax.random.split(jax.random.PRNGKey(4), 1), ar1_enabled=False)
    feedback = feedback_at([0])
    history, _ = sample(init_history(1), feedback, 0, 0)

    def tick(carry, index):
        camera, feedback, history = carry
        camera, fresh = advance_camera(camera, jnp.array([[0., -.5, 1.4]]), ar1_enabled=False)
        now = jnp.full((1,), index * 5000, jnp.int32)
        q = jnp.ones((1, 7)) * index * .005
        feedback, _ = append_feedback(feedback, q, now, now)
        history, _ = append_ball(history, feedback, camera.held_position_m,
            camera.last_sample_step*5000, now, fresh, camera.observed, pose_from_q)
        return (camera, feedback, history), None

    (_, _, history), _ = jax.jit(lambda carry: jax.lax.scan(tick, carry, jnp.arange(1, 201)))(
        (camera, feedback, history))
    rows = np.asarray(encode_history(history, jnp.array([1000000]))).reshape(64, 14)
    times = np.asarray(history.sample_time_us[0])[rows[:, 13] > .5]
    assert 57 <= len(times) <= 59
    assert np.all(np.diff(times) < 0)
    assert times.max() == 1000000 and times.min() >= 360000
    assert HISTORY_INPUT_DIM == 64 * 14


def test_goal_switch_does_not_require_history_reset_and_physical_reset_is_masked():
    history = init_history(2)
    feedback = init_feedback(2)
    feedback, _ = append_feedback(feedback, jnp.ones((2, 7)), jnp.array([0, 0]), jnp.array([0, 0]))
    history, _ = append_ball(history, feedback, jnp.ones((2, 3)), jnp.array([0, 0]),
        jnp.array([0, 0]), jnp.array([True, True]), jnp.array([True, True]), pose_from_q)
    changed = reset_selected(history, jnp.array([True, False]))
    assert not np.any(np.asarray(changed.present[0]))
    np.testing.assert_array_equal(changed.racket_pose[1], history.racket_pose[1])
    assert np.all(np.asarray(encode_history(history, jnp.array([640001, 640001]))) == 0.)


def test_exact_sample_uses_timestamped_measured_fk_pose():
    feedback = init_feedback(1)
    measured_pose = jnp.array([[.1, -.4, 1.1, 1., 0., 0., 0., 1., 0.]])
    feedback, _ = append_feedback(feedback, jnp.ones((1, 7)),
        jnp.array([0]), jnp.array([0]), racket_pose=measured_pose)
    history, _ = sample(init_history(1), feedback, 0, 0)
    np.testing.assert_array_equal(history.racket_pose[:, 0], measured_pose)
    reset = reset_selected(feedback, jnp.array([True]))
    assert not np.any(np.asarray(reset.valid))


def test_invalid_packets_and_dense_fake_ball_frames_fail_closed():
    feedback = feedback_at([0, 5000])
    history, report = append_ball(init_history(1), feedback, jnp.full((1, 3), jnp.nan),
        jnp.array([0]), jnp.array([0]), jnp.array([True]), jnp.array([True]), pose_from_q)
    assert bool(report['invalid_measurement'][0])
    assert np.all(np.isfinite(encode_history(history, jnp.array([0]))))
    assert np.all(np.asarray(encode_history(history, jnp.array([0]))) == 0.)
    # All slots are deliberately filled within 63ms, violating the 90Hz
    # contract. Do not silently drop a still-in-window source sample.
    history = history._replace(sample_time_us=jnp.arange(64000, 0, -1000)[None],
        present=jnp.ones((1, 64), bool), last_source_time_us=jnp.array([64000]))
    after, report = sample(history, feedback, 65000, 65000)
    assert bool(report['capacity_exceeded'][0])
    assert not bool(report['inserted'][0])
    np.testing.assert_array_equal(after.sample_time_us, history.sample_time_us)


def test_native_single_configuration_pose_preserves_column_order():
    import mujoco
    from world_model.kinematics import extract_chain
    from world_model.native_observation import racket_pose_from_chain
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody><body>
      <joint type="hinge" axis="0 0 1"/><geom type="sphere" size=".1"/>
      <site pos=".2 0 0"/>
      </body></worldbody></mujoco>''')
    chain = extract_chain(model, model.qpos0, joint_ids=[0], site_id=0, geom_id=0)
    pose = jax.jit(lambda q: racket_pose_from_chain(chain, q))(jnp.array([jnp.pi/2]))
    np.testing.assert_allclose(pose, [0., .2, 0., 0., 1., 0., -1., 0., 0.], atol=1e-7)
def test_full_chain_internal_cache_aligns_support_motion_without_actor_joint_slots():
    from world_model.paired_history import reset_selected
    feedback = init_feedback(1, n_joints=13)
    q0, q1 = jnp.zeros((1, 13)), jnp.zeros((1, 13)).at[:, 7].set(.2)
    pose = lambda q: jnp.concatenate((q[:, 7:8], jnp.zeros((len(q), 8))), axis=-1)
    for q, t in ((q0, 0), (q1, 5000)):
        feedback, _ = append_feedback(feedback, q, jnp.array([t]), jnp.array([t]))
    history, info = append_ball(init_history(1), feedback, jnp.array([[0., -.5, 1.3]]),
        jnp.array([2500]), jnp.array([5000]), jnp.array([True]), jnp.array([True]), pose)
    assert bool(info['paired'][0])
    rows = encode_history(history, jnp.array([5000])).reshape(1, 64, 14)
    np.testing.assert_allclose(rows[0, 0, 3], .1, atol=1e-8)
    assert rows.shape == (1, 64, 14)
    reset = reset_selected(feedback, jnp.array([True]))
    assert reset.q_rad.shape == (1, 129, 13)
    assert not bool(reset.valid.any())
