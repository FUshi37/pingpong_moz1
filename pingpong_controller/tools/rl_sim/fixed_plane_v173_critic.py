"""Critic-only restoration of the task information removed from the actor."""

from dataclasses import replace

import jax
import jax.numpy as jnp

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v173"
PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v172"
ACTOR_DIM = 72
OLD_CRITIC_DIM = 393
CRITIC_CONTEXT_DIM = 84
CRITIC_DIM = OLD_CRITIC_DIM + CRITIC_CONTEXT_DIM
CRITIC_SCHEMA = "fixed_plane_v173_legacy393_history63_phase2_contact19"


def append_task_context(
    prefix: jax.Array, *, racket_pose_history: jax.Array,
    desired_apex_z_m: jax.Array, hit_count: jax.Array,
    predicted_contact_context: jax.Array,
) -> jax.Array:
    """Append (N,84) while preserving the complete (N,393) legacy prefix.

    Seven older racket XYZ/6-D-rotation poses contribute 63 values. The
    current pose is already in the actor prefix. Desired apex and hit ordinal
    restore the hidden reward phase. The 19-value true-contact geometry uses
    simulated position/velocity; true ball velocity already exists in the
    legacy privileged block and is not duplicated as an extra velocity slot.
    """
    n = prefix.shape[0]
    if (prefix.shape != (n, OLD_CRITIC_DIM)
            or racket_pose_history.shape != (n, 8, 9)
            or jnp.shape(desired_apex_z_m) != (n,)
            or jnp.shape(hit_count) != (n,)
            or predicted_contact_context.shape != (n, 19)):
        raise ValueError("V173 requires legacy393, eight poses and contact19")
    return jnp.concatenate((
        prefix,
        jnp.asarray(racket_pose_history[:, 1:], dtype=jnp.float32).reshape(n, 63),
        jnp.asarray(desired_apex_z_m, dtype=jnp.float32)[:, None],
        jnp.asarray(hit_count, dtype=jnp.float32)[:, None] / 14.,
        jnp.asarray(predicted_contact_context, dtype=jnp.float32),
    ), axis=-1)


def build_stages(source):
    """Keep every actor, reward, domain, gate and original stage identity."""
    return [replace(stage,
        cfg=replace(stage.cfg, critic_fixed_plane_task_context=True),
        notes="V173 retains V172 exactly except append-only privileged task context. "
              "Actor72 is unchanged; critic393 grows to477 without a value/Adam reset.")
        for stage in source]


def validate_resume_payload(
    payload, *, requested_start_stage, resume_curriculum_state,
    reset_optimizer, reset_critic, allow_obs_dim_migration,
):
    """Require an explicit critic-only migration or exact own continuation."""
    source = payload.get("args", {}).get("curriculum_profile")
    dims = (payload.get("obs_dim"), payload.get("critic_obs_dim"), payload.get("act_dim"))
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))):
        raise ValueError("V173 preserves actor, critic and complete Adam state")
    stage = payload.get("stage_name", "")
    if not stage.startswith("fplc_v159_gpu1_s"):
        raise ValueError("V173 requires a registered V172 stage")
    if source == PARENT_PROFILE:
        if (dims != (ACTOR_DIM, OLD_CRITIC_DIM, 7)
                or not allow_obs_dim_migration or resume_curriculum_state
                or stage != requested_start_stage):
            raise ValueError("V173 requires explicit72/393->72/477 migration and fresh history")
    elif source == PROFILE:
        if dims != (ACTOR_DIM, CRITIC_DIM, 7):
            raise ValueError("V173 continuation requires exact72/477/7 dimensions")
        if resume_curriculum_state:
            if stage != requested_start_stage:
                raise ValueError("V173 continuation requires exact stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V173 rejects an unrelated checkpoint profile")
