#!/usr/bin/env python3
"""Paired deterministic checkpoint replay under V21/V22 control only.

This is a preflight diagnostic, not PPO convergence or deployment evidence.
The same V20 Stage-5 actor and reset keys are used for both profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import MjxJuggleEnv
from train_juggle_mjx_curriculum import (
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V21_PROFILE,
    FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V22_PROFILE,
    build_curriculum,
)
from train_juggle_mjx_ppo import policy_mean
from validate_juggle_mjx_ppo import load_checkpoint


def _hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _profile_cfg(profile: str):
    return build_curriculum(
        curriculum_profile=profile,
        measured_ball_mass_range_kg=(0.0039, 0.0041),
        asymmetric_critic=True,
        critic_command_history_steps=12,
        right_arm_pd_profile="recovered_rmp_rmpmd_v2",
    )[5].cfg


def _replay(
    *,
    profile: str,
    xml_path: Path,
    params: dict,
    seed: int,
    n_envs: int,
    steps: int,
) -> dict:
    cfg = _profile_cfg(profile)
    env = MjxJuggleEnv(xml_path, n_envs=n_envs, cfg=cfg)
    key, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    del key
    state, obs = jax.jit(env.reset)(jax.random.split(reset_key, n_envs))
    reset_hash = hashlib.sha256(
        np.asarray(jax.device_get(state.data.qpos)).tobytes()
    ).hexdigest()
    arm_velocity_limit = np.asarray(env.arm_vel_limit_rad_s)

    @jax.jit
    def step(current_state, current_obs):
        action = jnp.clip(policy_mean(params, current_obs), -1.0, 1.0)
        next_state, next_obs, reward, done, metrics = env.step(
            current_state, action
        )
        measured_q = next_state.data.qpos[:, env.arm_qadr]
        error = next_state.arm_cmd_q - measured_q
        site_world = next_state.data.site_xpos[:, env.racket_site_id]
        site_policy = jnp.einsum(
            "ij,nj->ni", env.cartesian_world_to_policy_rotation,
            site_world - env.cartesian_policy_origin_world[None, :],
        )
        workspace_violation = jnp.max(jnp.maximum(jnp.maximum(
            env.cartesian_pose_outer_position_lower[None, :] - site_policy,
            site_policy - env.cartesian_pose_outer_position_upper[None, :],
        ), 0.0), axis=-1)
        return next_state, next_obs, {
            "done": done,
            "reward": reward,
            "hits": metrics["hit_count"],
            "new_hit": metrics["new_hit"],
            "racket_z_m": metrics["racket_z"],
            "ball_z_m": metrics["ball_z"],
            "qref_error_norm_rad": jnp.linalg.norm(error, axis=-1),
            "qref_error_max_ratio_original": jnp.max(
                jnp.abs(error) / (
                    env.arm_vel_limit_rad_s[None, :]
                    * jnp.asarray((0.018, 0.019, 0.019, 0.018, 0.018, 0.018, 0.018))[None, :]
                ), axis=-1,
            ),
            "actual_workspace_violation_m": workspace_violation,
            "projection_intervened": metrics[
                "cartesian_pose_projection_intervened"
            ],
            "projection_fallback": metrics[
                "cartesian_pose_projection_fallback_to_hold"
            ],
            "qvel_exceed": metrics["arm_qvel_limit_exceed_fraction"],
            "qacc_exceed": metrics["arm_qacc_limit_exceed_fraction"],
            "action_abs_max": jnp.max(jnp.abs(action), axis=-1),
            "joint_velocity_limit_ratio": jnp.max(
                jnp.abs(next_state.data.qvel[:, env.arm_vadr])
                / env.arm_vel_limit_rad_s[None, :], axis=-1,
            ),
        }

    first_done = np.full(n_envs, -1, dtype=np.int32)
    first_hits = np.zeros(n_envs, dtype=np.float64)
    values: dict[str, list[np.ndarray]] = {}
    for tick in range(steps):
        state, obs, metrics = step(state, obs)
        host = jax.device_get(metrics)
        for name, value in host.items():
            values.setdefault(name, []).append(np.asarray(value))
        done_now = np.asarray(host["done"], dtype=bool)
        newly_done = (first_done < 0) & done_now
        first_done[newly_done] = tick + 1
        first_hits[newly_done] = np.asarray(host["hits"])[newly_done]
        if tick % 100 == 99:
            print(f"[{profile}] step={tick + 1}/{steps}", flush=True)
    trace = {name: np.stack(series) for name, series in values.items()}
    alive_mask = (first_done[None, :] < 0) | (
        np.arange(1, steps + 1)[:, None] <= first_done[None, :]
    )

    def active(name: str) -> np.ndarray:
        return trace[name][alive_mask]

    return {
        "profile": profile,
        "seed": seed,
        "n_envs": n_envs,
        "steps": steps,
        "reset_qpos_sha256": reset_hash,
        "first_episode_completed_count": int(np.sum(first_done > 0)),
        "first_episode_length_mean": float(np.mean(np.where(
            first_done > 0, first_done, steps
        ))),
        "first_episode_hits_mean": float(np.mean(np.where(
            first_done > 0, first_hits, trace["hits"][-1]
        ))),
        "hit3_fraction": float(np.mean(np.where(
            first_done > 0, first_hits, trace["hits"][-1]
        ) >= 3)),
        "hit6_fraction": float(np.mean(np.where(
            first_done > 0, first_hits, trace["hits"][-1]
        ) >= 6)),
        "max_qref_error_norm_rad": float(np.max(active("qref_error_norm_rad"))),
        "max_qref_error_ratio_original": float(np.max(active(
            "qref_error_max_ratio_original"
        ))),
        "max_actual_workspace_violation_m": float(np.max(active(
            "actual_workspace_violation_m"
        ))),
        "projection_intervention_fraction": float(np.mean(active(
            "projection_intervened"
        ))),
        "projection_fallback_fraction": float(np.mean(active(
            "projection_fallback"
        ))),
        "qvel_exceed_fraction": float(np.mean(active("qvel_exceed"))),
        "qacc_exceed_fraction": float(np.mean(active("qacc_exceed"))),
        "max_joint_velocity_limit_ratio": float(np.max(active(
            "joint_velocity_limit_ratio"
        ))),
        "max_action_abs": float(np.max(active("action_abs_max"))),
        "first_episode_lengths": first_done.tolist(),
        "first_episode_hits": first_hits.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20263509)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--physical-gpu-index", type=int, default=0)
    parser.add_argument("--expected-gpu-uuid", required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    if args.n_envs <= 0 or args.steps <= 0:
        parser.error("n-envs and steps must be positive")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu_index):
        parser.error("CUDA_VISIBLE_DEVICES mismatch")
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(args.physical_gpu_index),
         "--query-gpu=uuid", "--format=csv,noheader"], text=True,
    ).strip()
    if gpu_uuid != args.expected_gpu_uuid or jax.default_backend() != "gpu":
        parser.error("physical GPU UUID or JAX CUDA mismatch")
    payload = load_checkpoint(args.checkpoint)
    if payload.get("stage_name") != "fplc_v20_gpu0_s5_steady_entry_prefix":
        parser.error("expected pinned, passed V20 Stage-5 source")
    xml_path = Path(payload["xml"])
    profiles = (
        FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V21_PROFILE,
        FIXED_PLANE_LIMIT_CYCLE_QVEL_RMP_GPU0_V22_PROFILE,
    )
    results = [
        _replay(
            profile=profile, xml_path=xml_path, params=payload["params"],
            seed=args.seed, n_envs=args.n_envs, steps=args.steps,
        )
        for profile in profiles
    ]
    if results[0]["reset_qpos_sha256"] != results[1]["reset_qpos_sha256"]:
        raise RuntimeError("paired reset qpos mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "fixed_plane_v22_checkpoint_pair_v1",
        "command": " ".join(sys.argv),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _hash(args.checkpoint),
        "xml": str(xml_path.resolve()),
        "xml_sha256": _hash(xml_path),
        "script_sha256": _hash(Path(__file__)),
        "trainer_sha256": _hash(RL_SIM_DIR / "train_juggle_mjx_curriculum.py"),
        "environment_sha256": _hash(RL_SIM_DIR / "mjx_juggle_env.py"),
        "gpu_uuid": gpu_uuid,
        "jax_backend": jax.default_backend(),
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[checkpoint-pair] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
