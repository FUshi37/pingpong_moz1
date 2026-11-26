"""Single-compile paired GPU evaluation of multiple V325 checkpoints."""
from collections import Counter
from pathlib import Path
import pickle
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import jax
import jax.numpy as jnp
import numpy as np

from migrated_stack import load_migrated_v325_stack, setup_sys_path
setup_sys_path()

from single5_runtime import install_pose_mode, Single5Env
from train_s15_fresh_velocity_r31 import ActualContactEnv
from capture_margin import CaptureEnv
from xwind_sampling import XWindEnv, labels_for_wind
from wind_features import WindFeatureEnv
from termination_diagnostics import TERMINATION_FIELDS


def build_env(n_envs: int = 256):
    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
    install_pose_mode()
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=stack.cfg)
    env = XWindEnv(CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
                   half_angle_deg=45.0, enabled=True, calm_only=False)
    env = Single5Env(env, reward_enabled=False, goal_reward_enabled=False,
                     random_onset_enabled=True, targeted_reward_enabled=False,
                     weak_release_floor_enabled=True, weak_release_floor_weight=4.0)
    env = WindFeatureEnv(env)
    return stack, env


def make_chunk_fn(stack, env, stochastic: bool, chunk: int = 128):
    policy_mean = stack.ppo.policy_mean

    @jax.jit
    def step_chunk(params, state, obs, rng):
        def body(carry, _):
            st, ob, key = carry
            key, akey = jax.random.split(key)
            mean = policy_mean(params, ob)
            if stochastic:
                std = jnp.exp(params["log_std"])
                action = mean + std * jax.random.normal(akey, mean.shape)
            else:
                action = mean
            st2, ob2, rew, done, m = env.step(st, action)
            out = {
                "terminated": m["terminated"],
                "hits": st2.wind_startup_hit_count,
                "wind_mode": m["wind/mode"],
                "wind_speed": m["polish/background_speed_m_s"],
                "wind_angle": m["single5/wind_angle_rad"],
                "rz_fail": m["done/racket_z_out_of_bounds"],
                "recontact_fail": m["done/pending_hit_recontact"],
                "conf_ratio": m["timing/confirmed_max_ratio"],
                "phys_ratio": m["timing/physical_max_ratio"],
            }
            return (st2, ob2, key), out

        (st_end, ob_end, rng_end), traj = jax.lax.scan(body, (state, obs, rng), None, length=chunk)
        return st_end, ob_end, rng_end, traj

    return step_chunk


def eval_one(env, reset_fn, step_chunk, ckpt_path: Path, seconds: float = 20.0, seed: int = 2026112601):
    with ckpt_path.open("rb") as f:
        saved = pickle.load(f)
    params = jax.device_put(saved["params"])
    chunk = 128
    n_chunks = int(round(seconds / env.dt)) // chunk
    keys = jax.random.split(jax.random.PRNGKey(seed), env.n_envs)
    state, obs = reset_fn(keys)
    rng = jax.random.PRNGKey(seed + 1)

    alive = np.ones(env.n_envs, dtype=bool)
    rz_fails = 0
    rec_fails = 0
    init_labels = None
    last_conf_ratio = np.zeros(env.n_envs)
    last_phys_ratio = np.zeros(env.n_envs)
    last_hits = np.zeros(env.n_envs)
    t0 = time.monotonic()

    for c in range(n_chunks):
        state, obs, rng, traj = step_chunk(params, state, obs, rng)
        t_np = {k: np.asarray(jax.device_get(v)) for k, v in traj.items()}
        if init_labels is None:
            init_labels = labels_for_wind(t_np["wind_mode"][0], t_np["wind_speed"][0], t_np["wind_angle"][0], 45.0)
        term = t_np["terminated"] > 0.5
        for t in range(chunk):
            newly_dead = alive & term[t]
            if np.any(newly_dead):
                rz_fails += int(np.sum(t_np["rz_fail"][t, newly_dead] > 0.5))
                rec_fails += int(np.sum(t_np["recontact_fail"][t, newly_dead] > 0.5))
                alive &= ~newly_dead
        last_conf_ratio = t_np["conf_ratio"][-1]
        last_phys_ratio = t_np["phys_ratio"][-1]
        last_hits = t_np["hits"][-1]

    dt = time.monotonic() - t0
    strict_ok = alive & (last_hits >= int(round(1.2 * seconds))) & (last_conf_ratio <= 1.0 + 1e-6) & (last_phys_ratio <= 1.0 + 1e-6)

    def grp(pred):
        m = np.array([pred(int(l)) for l in init_labels], dtype=bool)
        if not np.any(m): return (0, 0.0, 0.0)
        return (int(m.sum()), float(alive[m].mean())*100, float(strict_ok[m].mean())*100)

    clean = grp(lambda l: l == 0)
    peak = grp(lambda l: l >= 17)
    negx = grp(lambda l: l in (17, 18, 19, 20, 25))
    dir08 = grp(lambda l: l == 25)
    dir00 = grp(lambda l: l == 17)
    hard = grp(lambda l: l in (4, 16))
    print(
        f"  {ckpt_path.name:28s} | wall={dt:5.1f}s | phys={alive.sum():3d}/{env.n_envs} ({alive.mean()*100:5.2f}%) "
        f"strict={strict_ok.mean()*100:5.2f}% | rz={rz_fails:2d} rec={rec_fails:2d} | "
        f"clean={clean[1]:5.1f}%/{clean[2]:5.1f}% | peak={peak[1]:5.1f}% | negX={negx[1]:5.1f}% | "
        f"dir08={dir08[1]:5.1f}% | dir00={dir00[1]:5.1f}% | hard3={hard[1]:5.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    n_envs = int(sys.argv[1])
    seconds = float(sys.argv[2])
    ckpts = [Path(p) for p in sys.argv[3:]]
    stack, env = build_env(n_envs=n_envs)
    reset_fn = jax.jit(env.reset)
    print(f"=== PAIRED STOCHASTIC (sigma=0.005) EVAL: n_envs={n_envs}, seconds={seconds}s ===", flush=True)
    stoch_fn = make_chunk_fn(stack, env, stochastic=True)
    for c in ckpts:
        eval_one(env, reset_fn, stoch_fn, c, seconds=seconds)
    print(f"\n=== PAIRED DETERMINISTIC EVAL: n_envs={n_envs}, seconds={seconds}s ===", flush=True)
    det_fn = make_chunk_fn(stack, env, stochastic=False)
    for c in ckpts:
        eval_one(env, reset_fn, det_fn, c, seconds=seconds)
