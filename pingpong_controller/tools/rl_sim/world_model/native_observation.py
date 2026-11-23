"""969-D observational adapter around the untouched V324 teacher transition."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v92 import rotation_matrix_to_6d

from .kinematics import extract_chain, forward_kinematics
from .observation_engine import (
    CameraState, ObservationHistory, WorldObservationState,
    actor_observation_with_history, advance_observation_history,
)
from .paired_history import init_feedback, init_history


def racket_pose_from_chain(chain, q_rad):
    """Convert one configuration to site XYZ and the original column-wise 6D rotation."""
    k = forward_kinematics(chain, q_rad)
    return jnp.concatenate((k.site_position, rotation_matrix_to_6d(k.site_rotation[None])[0]))


def camera_from_native(state) -> CameraState:
    return CameraState(state.rng, state.step_count, state.last_ball_obs_step,
        state.cached_ball_obs_pos, state.actor_ball_position_history,
        state.actor_ball_age_history, state.ball_position_noise_epsilon_m,
        state.fixed_contact_ball_observed)


class NativeObservationAdapter:
    """Keep sparse actor history plus internal encoder alignment cache in seeds.

The callable accepts only measured joint states, observed camera positions and
measured-state racket FK. It never uses true ball velocity/spin or contacts.
Native rewards/done flags remain outside this adapter.
    """

    def __init__(self, environment, calibrated_state, *, full_chain=False):
        self.environment = environment
        self.full_chain = bool(full_chain)
        joint_ids = np.asarray(environment.arm_jids, dtype=np.int32)
        if full_chain:
            support_ids = np.array([environment.mj_model.joint(f'LegWaist-{i}').id for i in range(6)])
            joint_ids = np.concatenate((joint_ids, support_ids))
        self.feedback_qadr = environment.mj_model.jnt_qposadr[joint_ids].copy()
        chain = extract_chain(environment.mj_model, np.asarray(calibrated_state.data.qpos[0]),
            joint_ids=joint_ids, site_id=environment.racket_site_id,
            geom_id=environment.racket_geom_id)

        def pose(q):
            return racket_pose_from_chain(chain, q)

        self.pose_from_q = jax.vmap(pose)
        self._update = jax.jit(self._update_samples)

    def _update_samples(self, history, q, observed_ball, sample_us,
                        now_us, observed, reset_mask):
        return advance_observation_history(history, q, observed_ball, sample_us,
            now_us, observed, pose_from_q=self.pose_from_q, physical_reset_mask=reset_mask)

    def _values(self, state):
        env = self.environment
        q = state.data.qpos[:, env.arm_qadr]
        # The inherited73-D prefix keeps the native cached-site timing in both
        # modes. Full-chain same-source-time FK belongs to the appended sparse
        # history; it must not silently replace the V321 current-pose slots.
        pose = jnp.concatenate((state.data.site_xpos[:, env.racket_site_id],
            rotation_matrix_to_6d(state.data.site_xmat[:, env.racket_site_id].reshape(-1, 3, 3))), axis=-1)
        return q, pose

    def initialize(self, state):
        """Start at a true physical reset; resumed/forked states need full history."""
        if np.any(np.asarray(state.step_count) != 0):
            raise ValueError('Cannot invent 640ms history from a mid-episode native state; restore the full seed')
        n = state.step_count.shape[0]
        empty = ObservationHistory(init_feedback(n, n_joints=len(self.feedback_qadr)), init_history(n))
        return self.advance(state, empty)

    def advance(self, native_after, history, *, physical_reset_mask=None):
        """Run once after each native control tick, including explicit reset lanes."""
        q = native_after.data.qpos[:, self.feedback_qadr]
        reset = (jnp.zeros_like(native_after.step_count, bool)
                 if physical_reset_mask is None else physical_reset_mask)
        return self._update(history, q, native_after.cached_ball_obs_pos,
            native_after.last_ball_obs_step * 5000, native_after.step_count * 5000,
            native_after.fixed_contact_ball_observed, reset)

    def world_observation_state(self, state, history) -> WorldObservationState:
        """Seed imagination from the existing camera phase and complete history."""
        return WorldObservationState(camera_from_native(state), history)

    def observation(self, state, history, task_goal_xyz):
        """Use independent task goals; never write the native workspace anchor."""
        q, pose = self._values(state)
        return actor_observation_with_history(q, state.data.qvel[:, self.environment.arm_vadr],
            state.arm_cmd_q, state.prev_action, pose, camera_from_native(state),
            task_goal_xyz, history.paired, state.step_count * 5000)
