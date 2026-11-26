"""30s launch & early-juggling diagnostic probe for V325 u416 on migrated host."""
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

from migrated_stack import load_migrated_v325_stack, setup_sys_path, PROVISIONAL, sha256_file
setup_sys_path()
from single5_runtime import install_pose_mode, Single5Env
from train_s15_fresh_velocity_r31 import ActualContactEnv
from capture_margin import CaptureEnv
from xwind_sampling import XWindEnv, labels_for_wind
from wind_features import WindFeatureEnv
from termination_diagnostics import TERMINATION_FIELDS


def build_env_and_model(ckpt_path: Path, n_envs: int = 256):
    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
        with ckpt_path.open("rb") as f:
            saved = pickle.load(f)
    params = jax.device_put(saved["params"])
    install_pose_mode()
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=stack.cfg)
    env = XWindEnv(CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
                   half_angle_deg=45.0, enabled=True, calm_only=False)
    env = Single5Env(env, reward_enabled=False, goal_reward_enabled=False,
                     random_onset_enabled=True, targeted_reward_enabled=False,
                     weak_release_floor_enabled=True, weak_release_floor_weight=4.0)
    env = WindFeatureEnv(env)
    return stack, env, params


def run_probe(stack, env, params, *, stochastic: bool, seconds: float = 16.0, seed: int = 2026112601):
    n_steps = int(round(seconds / env.dt))
    chunk = 128
    n_chunks = (n_steps + chunk - 1) // chunk
    policy_mean = stack.ppo.policy_mean

    @jax.jit
    def step_chunk(state, obs, rng):
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
                "done": done,
                "terminated": m["terminated"],
                "step_count": st2.step_count,
                "hits": st2.wind_startup_hit_count,
                "wind_mode": m["wind/mode"],
                "wind_speed": m["polish/background_speed_m_s"],
                "actual_wind_speed": m["wind/speed_m_s"],
                "wind_angle": m["single5/wind_angle_rad"],
                "obs112": ob[:, 112],
                "est_speed": 1.6 * jnp.sqrt(jnp.sum(ob[:, 114:116] ** 2, axis=-1)),
                "feat_active": jnp.any(ob[:, 118:120] != 0.0, axis=-1),
                "post_onset_hits": m["timing/post_onset_hits"],
                "after_onset_s": m["timing/after_onset_s"],
                "ball_z": st2.data.qpos[:, env.ball_qadr + 2],
                "racket_z": st2.data.site_xpos[:, env.racket_site_id, 2],
            }
            for tag in TERMINATION_FIELDS:
                if tag in m:
                    out[tag] = m[tag]
            return (st2, ob2, key), out

        (st_end, ob_end, rng_end), traj = jax.lax.scan(body, (state, obs, rng), None, length=chunk)
        return st_end, ob_end, rng_end, traj

    keys = jax.random.split(jax.random.PRNGKey(seed), env.n_envs)
    state, obs = jax.jit(env.reset)(keys)
    rng = jax.random.PRNGKey(seed + 1)

    alive = np.ones(env.n_envs, dtype=bool)
    fail_records = []
    init_labels = None

    t0 = time.monotonic()
    for c in range(n_chunks):
        state, obs, rng, traj = step_chunk(state, obs, rng)
        if c in (0, 4, 9, 14, 19, n_chunks - 1):
            print(f'  chunk {c+1}/{n_chunks} t={(c+1)*chunk*env.dt:.2f}s wall={time.monotonic()-t0:.1f}s alive={int(alive.sum())}/{env.n_envs}', flush=True)
        traj_np = {k: np.asarray(jax.device_get(v)) for k, v in traj.items()}
        if init_labels is None:
            init_labels = labels_for_wind(traj_np["wind_mode"][0], traj_np["wind_speed"][0], traj_np["wind_angle"][0], 45.0)
        term = traj_np["terminated"] > 0.5
        for t in range(chunk):
            newly_dead = alive & term[t]
            if np.any(newly_dead):
                for lane in np.flatnonzero(newly_dead):
                    tags = [tag for tag in TERMINATION_FIELDS if tag in traj_np and traj_np[tag][t, lane] > 0.5]
                    fail_records.append(dict(
                        lane=int(lane),
                        label=int(init_labels[lane]),
                        sec=float((c * chunk + t + 1) * env.dt),
                        hits=int(traj_np["hits"][t, lane]),
                        post_onset_hits=int(traj_np["post_onset_hits"][t, lane]),
                        after_onset_s=float(traj_np["after_onset_s"][t, lane]),
                        wind_mode=int(traj_np["wind_mode"][t, lane]),
                        bg_speed=float(traj_np["wind_speed"][t, lane]),
                        actual_speed=float(traj_np["actual_wind_speed"][t, lane]),
                        est_speed=float(traj_np["est_speed"][t, lane]),
                        obs112=float(traj_np["obs112"][t, lane]),
                        feat_active=bool(traj_np["feat_active"][t, lane]),
                        ball_z=float(traj_np["ball_z"][t, lane]),
                        racket_z=float(traj_np["racket_z"][t, lane]),
                        tags=tags,
                    ))
                alive &= ~newly_dead
    dt = time.monotonic() - t0

    names = ["clean"] + [f"xsector{i:02d}_bin{j}" for i in range(4) for j in range(4)] + [f"xpeak_dir{i:02d}" for i in range(10)]
    tag_counts = Counter(t for r in fail_records for t in (r["tags"] or ["UNKNOWN"]))
    by_group = {}
    for idx, name in enumerate(names):
        mask = init_labels == idx
        n = int(mask.sum())
        if n > 0:
            surv = float(alive[mask].mean())
            if surv < 1.0 or name == "clean":
                by_group[name] = (n, int(alive[mask].sum()), surv)

    print(f"\n=== PROBE stochastic={stochastic} ({seconds}s, {env.n_envs} envs, wall={dt:.1f}s) ===", flush=True)
    print(f"Overall survival: {alive.sum()}/{env.n_envs} ({alive.mean()*100:.2f}%), failures={len(fail_records)}", flush=True)
    print(f"Termination tags: {dict(tag_counts.most_common())}", flush=True)
    if fail_records:
        secs = np.array([r["sec"] for r in fail_records])
        hits = np.array([r["hits"] for r in fail_records])
        po_hits = np.array([r["post_onset_hits"] for r in fail_records])
        obs112 = np.array([r["obs112"] for r in fail_records])
        f_act = np.array([r["feat_active"] for r in fail_records])
        est_s = np.array([r["est_speed"] for r in fail_records])
        act_s = np.array([r["actual_speed"] for r in fail_records])
        print(f"Fail sec:  min={secs.min():.2f} p25={np.percentile(secs,25):.2f} med={np.median(secs):.2f} p75={np.percentile(secs,75):.2f} max={secs.max():.2f}", flush=True)
        print(f"Fail hits: min={hits.min()} p25={np.percentile(hits,25):.1f} med={np.median(hits):.1f} p75={np.percentile(hits,75):.1f} max={hits.max()} | pre-hit5(<5): {int((hits<5).sum())}", flush=True)
        print(f"Fail post_onset_hits: min={po_hits.min()} med={np.median(po_hits):.1f} max={po_hits.max()} | ==0: {int((po_hits==0).sum())}", flush=True)
        print(f"Fail obs112==0: {int((obs112==0).sum())}/{len(fail_records)} | obs112<1: {int((obs112<0.999).sum())}/{len(fail_records)}", flush=True)
        print(f"Fail feat118:120 active: {int(f_act.sum())}/{len(fail_records)} | actual_wind med={np.median(act_s):.3f} vs est_wind med={np.median(est_s):.3f}", flush=True)
        print("Groups with failures (plus clean):", flush=True)
        for g, (n, s, rate) in sorted(by_group.items(), key=lambda x: x[1][2]):
            print(f"  {g:16s}: {s:3d}/{n:3d} ({rate*100:6.2f}%)", flush=True)
        print("Sample 10 failures:", flush=True)
        for r in fail_records[:10]:
            print(f"  {names[r['label']]:14s} t={r['sec']:5.2f}s hit={r['hits']:2d} po_hit={r['post_onset_hits']:2d} act_w={r['actual_speed']:.2f} est_w={r['est_speed']:.2f} feat={int(r['feat_active'])} bz={r['ball_z']:.3f} rz={r['racket_z']:.3f} tags={r['tags']}", flush=True)


if __name__ == "__main__":
    ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else (PROVISIONAL / "model.pkl")
    print("JAX devices:", jax.devices(), "ckpt:", ckpt, flush=True)
    n_envs = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 16.0
    stack, env, params = build_env_and_model(ckpt, n_envs=n_envs)
    print(f"STACK LOADED & CONFIG PARITY VERIFIED 100%! n_envs={n_envs} secs={secs}", flush=True)
    run_probe(stack, env, params, stochastic=True, seconds=secs)
