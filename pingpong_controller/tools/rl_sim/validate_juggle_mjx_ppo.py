"""Validate a Stage 1a MJX/JAX PPO juggling checkpoint."""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco as mj
import numpy as np

RL_SIM_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = RL_SIM_DIR.parents[1]
OUTPUT_DIR = PACKAGE_DIR / "outputs" / "rl_sim"
DEFAULT_CHECKPOINT = OUTPUT_DIR / "logs_mjx_stage1a" / "mjx_ppo_last.pkl"
DEFAULT_RESULTS = OUTPUT_DIR / "logs_mjx_stage1a" / "validation.csv"

if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))

from mjx_juggle_env import MjxJuggleConfig, MjxJuggleEnv
from train_juggle_mjx_ppo import policy_value


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a MJX/JAX PPO juggling checkpoint.")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--xml", type=Path, default=None, help="Override XML path. Defaults to the checkpoint XML.")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true", help="Use policy mean instead of sampling.")
    p.add_argument("--action-gain", type=float, default=1.0, help="Multiply policy action before clipping.")
    p.add_argument("--max-env-steps", type=int, default=0, help="0 means auto from episodes, envs, and horizon.")
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--log-hit-events", action="store_true")
    p.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--no-save-csv", action="store_true")
    p.add_argument("--racket-z-hard-limit-down", type=float, default=None)
    p.add_argument("--racket-z-hard-limit-up", type=float, default=None)
    p.add_argument("--no-terminate-on-racket-z-limit", action="store_true")
    p.add_argument("--render", action="store_true", help="Render env 0 with MuJoCo viewer.")
    p.add_argument("--realtime", action="store_true", help="Sleep according to env.dt while rendering.")
    p.add_argument("--slowmo", type=float, default=1.0)
    p.add_argument("--video-out", type=Path, default=None, help="Save env 0 validation render to an mp4/gif file.")
    p.add_argument("--video-fps", type=int, default=30, help="Playback FPS for --video-out.")
    p.add_argument("--video-width", type=int, default=1280)
    p.add_argument("--video-height", type=int, default=720)
    p.add_argument(
        "--video-slowmo",
        type=float,
        default=1.0,
        help="Playback slow-motion factor. 1.0 is realtime, 2.0 is half speed.",
    )
    p.add_argument(
        "--video-camera",
        type=str,
        default=None,
        help="Optional MuJoCo camera name/id for video rendering. Defaults to the free camera.",
    )
    return p.parse_args()


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    with path.open("rb") as f:
        payload = pickle.load(f)
    if "params" not in payload:
        raise SystemExit(f"Checkpoint does not contain policy params: {path}")
    return payload


def save_episode_rows(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def env_config_from_checkpoint(payload: dict, args: argparse.Namespace) -> MjxJuggleConfig:
    cfg_payload = payload.get("env_cfg") or {}
    valid_fields = {f.name for f in fields(MjxJuggleConfig)}
    cfg_kwargs = {k: v for k, v in cfg_payload.items() if k in valid_fields}
    cfg = MjxJuggleConfig(**cfg_kwargs)
    if args.racket_z_hard_limit_down is not None:
        cfg = replace(cfg, racket_z_hard_limit_down=float(args.racket_z_hard_limit_down))
    if args.racket_z_hard_limit_up is not None:
        cfg = replace(cfg, racket_z_hard_limit_up=float(args.racket_z_hard_limit_up))
    if args.no_terminate_on_racket_z_limit:
        cfg = replace(cfg, terminate_on_racket_z_limit=False)
    return cfg


def copy_env0_to_mujoco_data(env: MjxJuggleEnv, env_state, render_data: mj.MjData) -> None:
    qpos = np.asarray(jax.device_get(env_state.data.qpos[0]))
    qvel = np.asarray(jax.device_get(env_state.data.qvel[0]))
    step_count = int(np.asarray(jax.device_get(env_state.step_count[0])))
    render_data.qpos[:] = qpos
    render_data.qvel[:] = qvel
    render_data.ctrl[:] = np.asarray(jax.device_get(env_state.data.ctrl[0]))
    render_data.time = step_count * env.dt
    mj.mj_forward(env.mj_model, render_data)


def append_video_frame(writer, renderer: mj.Renderer, env: MjxJuggleEnv, env_state, render_data: mj.MjData, camera) -> None:
    copy_env0_to_mujoco_data(env, env_state, render_data)
    if camera is None:
        renderer.update_scene(render_data)
    else:
        renderer.update_scene(render_data, camera=camera)
    writer.append_data(np.asarray(renderer.render(), dtype=np.uint8))


def ensure_offscreen_framebuffer(model: mj.MjModel, width: int, height: int) -> tuple[int, int, int, int]:
    old_width = int(model.vis.global_.offwidth)
    old_height = int(model.vis.global_.offheight)
    model.vis.global_.offwidth = max(old_width, int(width))
    model.vis.global_.offheight = max(old_height, int(height))
    return old_width, old_height, int(model.vis.global_.offwidth), int(model.vis.global_.offheight)


def make_eval_step(env: MjxJuggleEnv, deterministic: bool, action_gain: float):
    def eval_step(params, env_state, obs, rng, running_return, running_length):
        rng, action_key, reset_key = jax.random.split(rng, 3)
        mean, _ = policy_value(params, obs)
        if deterministic:
            raw_action = mean
        else:
            log_std = params["log_std"]
            raw_action = mean + jnp.exp(log_std) * jax.random.normal(action_key, mean.shape)
        action = jnp.clip(raw_action * float(action_gain), -1.0, 1.0)

        next_env_state, next_obs, reward, done, metrics = env.step(env_state, action)
        completed_return = running_return + reward
        completed_length = running_length + 1

        reset_keys = jax.random.split(reset_key, env.n_envs)
        next_env_state, next_obs = env.reset_done(next_env_state, next_obs, done, reset_keys)
        next_running_return = jnp.where(done, 0.0, completed_return)
        next_running_length = jnp.where(done, 0, completed_length)

        step_metrics = {
            "done": done,
            "reward": reward,
            "episode_return": completed_return,
            "episode_length": completed_length,
            "hit_count": metrics["hit_count"],
            "new_hit": metrics["new_hit"],
            "in_contact": metrics["in_contact"],
            "ball_z": metrics["ball_z"],
            "racket_z": metrics["racket_z"],
            "racket_z_rel": metrics["racket_z_rel"],
            "action_norm": jnp.linalg.norm(action, axis=-1),
            "terminated": metrics["terminated"],
            "truncated": metrics["truncated"],
            "episode_step": metrics["episode_step"],
        }
        for key, value in metrics.items():
            if key.startswith("done/") or key.startswith("reward/"):
                step_metrics[key] = value
        return next_env_state, next_obs, rng, next_running_return, next_running_length, step_metrics

    return jax.jit(eval_step)


def summarize(rows: list[dict[str, float]]) -> str:
    if not rows:
        return "no completed episodes"
    returns = np.asarray([r["return"] for r in rows], dtype=np.float32)
    lengths = np.asarray([r["length"] for r in rows], dtype=np.float32)
    hits = np.asarray([r["hits"] for r in rows], dtype=np.float32)
    return (
        f"episodes={len(rows)} "
        f"return_mean={returns.mean():.3f} return_std={returns.std():.3f} "
        f"len_mean={lengths.mean():.1f} hits_mean={hits.mean():.2f} "
        f"hits_max={hits.max():.0f}"
    )


def done_reason_summary(rows: list[dict[str, float]]) -> str:
    if not rows:
        return "none"
    keys = sorted({k for row in rows for k in row if k.startswith("done/")})
    parts = []
    for key in keys:
        count = sum(1 for row in rows if float(row.get(key, 0.0)) > 0.5)
        if count > 0:
            parts.append(f"{key.removeprefix('done/')}={count}")
    truncated = sum(1 for row in rows if float(row.get("truncated", 0.0)) > 0.5)
    if truncated > 0:
        parts.append(f"truncated={truncated}")
    return ", ".join(parts) if parts else "none"


def row_done_reasons(row: dict[str, float]) -> str:
    reasons = [k.removeprefix("done/") for k, v in row.items() if k.startswith("done/") and float(v) > 0.5]
    if float(row.get("truncated", 0.0)) > 0.5:
        reasons.append("truncated")
    return ",".join(reasons) if reasons else "none"


def main() -> None:
    args = parse_args()
    payload = load_checkpoint(args.checkpoint)
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    xml_path = args.xml or Path(payload.get("xml", RL_SIM_DIR / "moz1_pd.xml"))
    cfg = env_config_from_checkpoint(payload, args)
    results_csv = args.results_csv
    if args.results_csv == DEFAULT_RESULTS:
        results_csv = args.checkpoint.parent / "validation.csv"

    if (args.render or args.video_out is not None) and args.n_envs != 1:
        print("[validate_mjx] render/video shows env 0 only; use --n-envs 1 for smoother visual validation.")

    env = MjxJuggleEnv(xml_path, n_envs=args.n_envs, cfg=cfg)
    print(f"[validate_mjx] JAX devices: {jax.devices()}")
    print(f"[validate_mjx] checkpoint: {args.checkpoint}")
    print(f"[validate_mjx] XML: {xml_path}")
    print(f"[validate_mjx] MJX XML: {env.mjx_xml}")
    print(
        f"[validate_mjx] episodes={args.episodes}, n_envs={args.n_envs}, "
        f"deterministic={args.deterministic}, action_gain={args.action_gain}"
    )
    print(
        "[validate_mjx] env_cfg: "
        f"horizon_sec={cfg.horizon_sec}, max_steps={env.max_steps}, "
        f"mujoco_timestep={env.timestep:.4f}s, frame_skip={cfg.frame_skip}, "
        f"control_dt={env.dt:.4f}s, control_hz={1.0 / env.dt:.1f}Hz, "
        f"racket_z_limit_down={cfg.racket_z_hard_limit_down}, "
        f"racket_z_limit_up={cfg.racket_z_hard_limit_up}, "
        f"terminate_on_racket_z_limit={cfg.terminate_on_racket_z_limit}"
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, args.n_envs)
    env_state, obs = jax.jit(env.reset)(reset_keys)
    running_return = jnp.zeros((args.n_envs,), dtype=jnp.float32)
    running_length = jnp.zeros((args.n_envs,), dtype=jnp.int32)
    eval_step = make_eval_step(env, args.deterministic, args.action_gain)

    viewer_ctx = None
    viewer = None
    render_data = None
    video_writer = None
    video_renderer = None
    next_video_time = 0.0
    video_frame_interval = 0.0
    if args.render:
        import mujoco.viewer

        render_data = mj.MjData(env.mj_model)
        copy_env0_to_mujoco_data(env, env_state, render_data)
        viewer_ctx = mujoco.viewer.launch_passive(env.mj_model, render_data)
        viewer = viewer_ctx.__enter__()
    if args.video_out is not None:
        try:
            import imageio.v2 as imageio
        except ModuleNotFoundError as exc:
            raise SystemExit("imageio is required for --video-out. Install with: python -m pip install imageio imageio-ffmpeg") from exc

        args.video_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            video_writer = imageio.get_writer(str(args.video_out), fps=max(1, int(args.video_fps)))
        except ValueError as exc:
            raise SystemExit(
                f"Could not open video writer for {args.video_out}. "
                "For mp4 install a backend with: python -m pip install imageio-ffmpeg "
                "or conda install -c conda-forge imageio-ffmpeg. "
                "Alternatively save a .gif file."
            ) from exc
        if render_data is None:
            render_data = mj.MjData(env.mj_model)
        video_width = max(64, int(args.video_width))
        video_height = max(64, int(args.video_height))
        old_w, old_h, off_w, off_h = ensure_offscreen_framebuffer(env.mj_model, video_width, video_height)
        if off_w != old_w or off_h != old_h:
            print(f"[validate_mjx] resized offscreen framebuffer: {old_w}x{old_h} -> {off_w}x{off_h}")
        video_renderer = mj.Renderer(
            env.mj_model,
            height=video_height,
            width=video_width,
        )
        video_frame_interval = 1.0 / max(1e-6, float(args.video_fps) * max(1e-6, float(args.video_slowmo)))
        append_video_frame(video_writer, video_renderer, env, env_state, render_data, args.video_camera)
        next_video_time = video_frame_interval
        print(
            f"[validate_mjx] recording video: {args.video_out} "
            f"({args.video_width}x{args.video_height}, fps={args.video_fps}, slowmo={args.video_slowmo})"
        )

    episode_rows: list[dict[str, float]] = []
    env_episode_counts = np.zeros((args.n_envs,), dtype=np.int32)
    max_steps = args.max_env_steps
    if max_steps <= 0:
        max_steps = int(np.ceil(args.episodes / max(1, args.n_envs)) * env.max_steps * 2)
        max_steps = max(max_steps, env.max_steps)

    try:
        t0 = time.perf_counter()
        for step_idx in range(1, max_steps + 1):
            env_state, obs, rng, running_return, running_length, metrics = eval_step(
                params,
                env_state,
                obs,
                rng,
                running_return,
                running_length,
            )
            host = jax.device_get(metrics)
            done = np.asarray(host["done"], dtype=bool)

            if args.log_hit_events:
                hit_envs = np.flatnonzero(np.asarray(host["new_hit"]) > 0.5)
                for env_i in hit_envs:
                    print(
                        f"[hit] step={step_idx} env={env_i} "
                        f"hits={float(host['hit_count'][env_i]):.0f} "
                        f"ball_z={float(host['ball_z'][env_i]):.3f}"
                    )

            for env_i in np.flatnonzero(done):
                env_episode_counts[env_i] += 1
                row = {
                    "episode": len(episode_rows) + 1,
                    "env": int(env_i),
                    "env_episode": int(env_episode_counts[env_i]),
                    "return": float(host["episode_return"][env_i]),
                    "length": int(host["episode_length"][env_i]),
                    "hits": float(host["hit_count"][env_i]),
                    "last_reward": float(host["reward"][env_i]),
                    "action_norm": float(host["action_norm"][env_i]),
                    "ball_z": float(host["ball_z"][env_i]),
                    "racket_z": float(host["racket_z"][env_i]),
                    "racket_z_rel": float(host["racket_z_rel"][env_i]),
                    "terminated": float(host["terminated"][env_i]),
                    "truncated": float(host["truncated"][env_i]),
                    "episode_step": float(host["episode_step"][env_i]),
                }
                for key, value in host.items():
                    if key.startswith("done/") or key.startswith("reward/"):
                        row[key] = float(value[env_i])
                episode_rows.append(row)
                print(
                    f"[episode] ep={row['episode']} env={row['env']} "
                    f"len={row['length']} return={row['return']:.3f} "
                    f"hits={row['hits']:.0f} done={row_done_reasons(row)} "
                    f"last_reward={row['last_reward']:.3f}"
                )
                if len(episode_rows) >= args.episodes:
                    break

            if args.render and viewer is not None and render_data is not None:
                copy_env0_to_mujoco_data(env, env_state, render_data)
                viewer.sync()
                if args.realtime:
                    time.sleep(max(0.0, env.dt * float(args.slowmo)))

            if video_writer is not None and video_renderer is not None and render_data is not None:
                sim_time = step_idx * env.dt
                while sim_time + 1e-9 >= next_video_time:
                    append_video_frame(video_writer, video_renderer, env, env_state, render_data, args.video_camera)
                    next_video_time += video_frame_interval

            if args.print_every > 0 and step_idx % args.print_every == 0:
                done_count = len(episode_rows)
                mean_reward = float(np.mean(np.asarray(host["reward"])))
                mean_hits = float(np.mean(np.asarray(host["hit_count"])))
                elapsed = time.perf_counter() - t0
                sps = step_idx * args.n_envs / max(elapsed, 1e-9)
                print(
                    f"[validate_mjx] step={step_idx} completed={done_count}/{args.episodes} "
                    f"sps={sps:,.0f} reward={mean_reward:.4f} hits={mean_hits:.2f}"
                )

            if len(episode_rows) >= args.episodes:
                break
        else:
            print(f"[validate_mjx] stopped at max_env_steps={max_steps} before completing all episodes")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.__exit__(None, None, None)
        if video_writer is not None:
            video_writer.close()
        if video_renderer is not None:
            video_renderer.close()

    episode_rows = episode_rows[: args.episodes]
    print(f"[validate_mjx] {summarize(episode_rows)}")
    print(f"[validate_mjx] done reasons: {done_reason_summary(episode_rows)}")
    if not args.no_save_csv:
        save_episode_rows(results_csv, episode_rows)
        print(f"[validate_mjx] wrote: {results_csv}")
    if args.video_out is not None:
        print(f"[validate_mjx] wrote video: {args.video_out}")


if __name__ == "__main__":
    main()
