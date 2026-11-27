"""Paired 1280-env x 60s frozen deterministic benchmark: r5_fast u6006 vs u6247.

Packs 5 regimes into ONE single 1280-env GPU pass per model (shared XLA graph):
  - Lanes     0..255: Clean (mode=0, 0 m/s)                       [exact eval_vs_v321 seed]
  - Lanes   256..511: Constant Wind, Early Onset (5..100 steps)   [exact eval_vs_v321 seed]
  - Lanes   512..767: Dynamic Wind,  Early Onset (5..100 steps)   [exact eval_vs_v321 seed]
  - Lanes  768..1023: Constant Wind, Late Onset (2000..4000 steps, 10..20s after hit 5)
  - Lanes 1024..1279: Dynamic Wind,  Late Onset (2000..4000 steps, 10..20s after hit 5)

Lanes 0..767 are bit-for-bit aligned with eval_vs_v321.py (seed 2026112701) for direct
comparison against V321_u2097_det and V325_u4768_det.
"""
from dataclasses import replace
import json
import math
from pathlib import Path
import pickle
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


HERE = Path(__file__).resolve().parent


def main():
    script_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE
    out_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else HERE.parent / "fplc_wind_early_20261127_r5_fast"
    )
    sys.path.insert(0, str(script_dir))

    from migrated_stack import load_migrated_v325_stack, setup_sys_path
    setup_sys_path()
    import eval_vs_v321 as ev
    import fixed_plane_wind_pose
    from single5_runtime import install_pose_mode, Single5Env, constant_degrees
    from train_s15_fresh_velocity_r31 import ActualContactEnv
    from capture_margin import CaptureEnv
    from xwind_sampling import XWindEnv, labels_for_wind, vectors_from_uniforms
    from wind_features import WindFeatureEnv

    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
    install_pose_mode()
    fixed_plane_wind_pose.effective_degrees = constant_degrees

    n_per = 256
    n_envs = 5 * n_per
    seconds = 60.0
    seed = 2026112701

    wcfg = replace(
        stack.cfg.wind_config,
        mode_weights=(0.0, 0.0, 0.5, 0.5),
        max_speed_m_s=1.6,
        wind_y_nonpositive=True,
        startup_no_wind_hits=5,
        startup_ramp_s=1.0,
    )
    cfg = replace(stack.cfg, wind_config=wcfg)
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=cfg)
    xenv = XWindEnv(
        CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
        half_angle_deg=45.0,
        enabled=False,
        calm_only=False,
    )
    env = Single5Env(
        xenv,
        reward_enabled=False,
        goal_reward_enabled=False,
        random_onset_enabled=True,
        targeted_reward_enabled=False,
        weak_release_floor_enabled=True,
        weak_release_floor_weight=4.0,
    )
    env = WindFeatureEnv(env)
    ball_qadr = env.ball_qadr

    @jax.jit
    def reset_1280():
        keys_768 = jax.random.split(jax.random.PRNGKey(seed), 3 * n_per)
        keys_1280 = jnp.concatenate([keys_768, keys_768[n_per:]], axis=0)
        st, ob = env.reset(keys_1280)

        c_keys = keys_768[n_per : 2 * n_per]
        uniforms = jax.vmap(
            lambda key: jax.random.uniform(jax.random.fold_in(key, 2026112312), (2,))
        )(c_keys)
        c_vecs, c_modes = vectors_from_uniforms(
            uniforms, ev.constant_wind_layout(n_per), 45.0, xp=jnp
        )

        idx = jnp.arange(n_envs)
        is_clean = idx < n_per
        is_const_e = (idx >= n_per) & (idx < 2 * n_per)
        is_const_l = (idx >= 3 * n_per) & (idx < 4 * n_per)
        is_zero_dyn = is_clean | is_const_e | is_const_l

        new_mode = (
            st.wind_mode.at[:n_per]
            .set(0)
            .at[n_per : 2 * n_per]
            .set(c_modes)
            .at[3 * n_per : 4 * n_per]
            .set(c_modes)
        )
        new_bg = (
            st.wind_background_m_s.at[:n_per]
            .set(0.0)
            .at[n_per : 2 * n_per]
            .set(c_vecs)
            .at[3 * n_per : 4 * n_per]
            .set(c_vecs)
        )
        new_ou = jnp.where(is_zero_dyn[:, None], 0.0, st.wind_ou_m_s)
        new_gust = jnp.where(is_zero_dyn[:, None], 0.0, st.wind_gust_vector_m_s)

        late_delays = jax.vmap(
            lambda key: jax.random.randint(
                jax.random.fold_in(key, 2026112799), (), 2000, 4001, dtype=jnp.int32
            )
        )(keys_768[n_per:])
        new_delay = st.delay_steps.at[3 * n_per :].set(late_delays)

        st = st._replace(
            wind_mode=new_mode,
            wind_background_m_s=new_bg,
            wind_ou_m_s=new_ou,
            wind_gust_vector_m_s=new_gust,
            delay_steps=new_delay,
        )
        return st, ob

    chunk = 128

    @jax.jit
    def step_chunk(params, state, obs):
        def body(carry, _):
            st, ob = carry
            action = stack.ppo.policy_mean(params, ob)
            st2, ob2, rew, done, m = env.step(st, action)
            out = {
                "terminated": m["terminated"],
                "hits": st2.wind_startup_hit_count,
                "hit_evt": m["hit_event_count"],
                "center_sum": m["hit_contact_center_dist_sum"],
                "racket_vxy_sq": m["recurrent_hit_racket_vxy_sq_sum"],
                "racket_vxy_n": m["recurrent_hit_event_count"],
                "ball_z": st2.data.qpos[:, ball_qadr + 2],
                "wind_mode": m["wind/mode"],
                "bg_speed": m["polish/background_speed_m_s"],
                "wind_angle": m["single5/wind_angle_rad"],
                "pose_limit": m["pose/outer_limit_deg"],
                "delay_steps": st2.delay_steps,
                "rz_fail": m["done/racket_z_out_of_bounds"],
                "rec_fail": m["done/pending_hit_recontact"],
                "low_fail": m["done/ball_too_low"],
                "high_fail": m["done/ball_too_high"],
                "conf_ratio": m["timing/confirmed_max_ratio"],
                "phys_ratio": m["timing/physical_max_ratio"],
            }
            return (st2, ob2), out

        (st_end, ob_end), traj = jax.lax.scan(body, (state, obs), None, length=chunk)
        return st_end, ob_end, traj

    n_chunks = int(math.ceil((seconds / env.dt) / chunk))
    chunk_20s = int(math.ceil((20.0 / env.dt) / chunk))
    BUCKETS = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 115)]
    REGIMES = [
        ("clean", 0, n_per),
        ("constant_early", n_per, 2 * n_per),
        ("dynamic_early", 2 * n_per, 3 * n_per),
        ("constant_late", 3 * n_per, 4 * n_per),
        ("dynamic_late", 4 * n_per, 5 * n_per),
    ]

    def bucket_of(h):
        for lo, hi in BUCKETS:
            if lo <= h <= hi:
                return (lo, hi)
        return None

    def regime_of(lane):
        return REGIMES[lane // n_per][0]

    models = [
        ("V325_r5fast_u6006_det", out_dir / "best_pareto_u00006006.pkl"),
        ("V325_r5fast_u6247_det", out_dir / "interrupted_u00006247.pkl"),
    ]
    out_json = out_dir / "benchmark_r5fast_u6006_vs_u6247.json"
    all_results = []

    for label, pkl_path in models:
        with pkl_path.open("rb") as f:
            params = jax.device_put(pickle.load(f)["params"])
        state, obs = reset_1280()

        alive = np.ones(n_envs, dtype=bool)
        alive_20s = np.ones(n_envs, dtype=bool)
        strict_20s = np.zeros(n_envs, dtype=bool)
        fail_sec = np.full(n_envs, seconds, dtype=np.float32)
        max_hits = np.zeros(n_envs, dtype=np.int32)
        fail_cause = np.zeros(n_envs, dtype=np.int32)
        cur_hit_idx = np.zeros(n_envs, dtype=np.int32)
        flight_peak_z = np.zeros(n_envs, dtype=np.float32)
        prev_apex = np.full(n_envs, np.nan, dtype=np.float32)
        b_stats = {
            rname: {
                b: {
                    "n_hits": 0,
                    "center_sq_sum": 0.0,
                    "center_abs_sum": 0.0,
                    "vxy_sq_sum": 0.0,
                    "vxy_n": 0,
                    "n_apex": 0,
                    "apex_sum": 0.0,
                    "apex_err_sq_sum": 0.0,
                    "in_band": 0,
                    "jump_sum": 0.0,
                    "n_jump": 0,
                    "fails": 0,
                }
                for b in BUCKETS
            }
            for rname, _, _ in REGIMES
        }

        init_mode = init_bg = init_ang = None
        last_conf = np.zeros(n_envs)
        last_phys = np.zeros(n_envs)
        t0 = time.monotonic()

        for c in range(n_chunks):
            state, obs, traj = step_chunk(params, state, obs)
            t_np = {k: np.asarray(jax.device_get(v)) for k, v in traj.items()}
            if c == 0:
                np.testing.assert_allclose(t_np["pose_limit"], 5.0, atol=1e-4)
                init_mode = t_np["wind_mode"][0].astype(int)
                init_bg = t_np["bg_speed"][0]
                init_ang = t_np["wind_angle"][0]
                d0 = t_np["delay_steps"][0]
                assert np.all(init_mode[:n_per] == 0)
                assert np.all(init_mode[n_per : 2 * n_per] == 1)
                assert np.all(np.isin(init_mode[2 * n_per : 3 * n_per], [2, 3]))
                assert np.all(init_mode[3 * n_per : 4 * n_per] == 1)
                assert np.all(np.isin(init_mode[4 * n_per : 5 * n_per], [2, 3]))
                assert np.all((d0[n_per : 3 * n_per] >= 5) & (d0[n_per : 3 * n_per] <= 100))
                assert np.all((d0[3 * n_per :] >= 2000) & (d0[3 * n_per :] <= 4000))
                print(f"[{label}] chunk 0 verified in {time.monotonic() - t0:.1f}s", flush=True)

            term = t_np["terminated"] > 0.5
            hit_evt = t_np["hit_evt"] > 0.5
            ball_z = t_np["ball_z"]
            for t in range(chunk):
                sec_now = (c * chunk + t + 1) * env.dt
                flight_peak_z = np.where(alive, np.maximum(flight_peak_z, ball_z[t]), flight_peak_z)
                h_mask = alive & hit_evt[t]
                if np.any(h_mask):
                    for lane in np.flatnonzero(h_mask):
                        rname = regime_of(int(lane))
                        prev_h = int(cur_hit_idx[lane])
                        new_h = int(t_np["hits"][t, lane])
                        if new_h <= prev_h:
                            new_h = prev_h + 1
                        b = bucket_of(new_h)
                        if b is not None:
                            bs = b_stats[rname][b]
                            c_dist = float(t_np["center_sum"][t, lane])
                            bs["n_hits"] += 1
                            bs["center_sq_sum"] += c_dist * c_dist
                            bs["center_abs_sum"] += c_dist
                            if t_np["racket_vxy_n"][t, lane] > 0.5:
                                bs["vxy_sq_sum"] += float(t_np["racket_vxy_sq"][t, lane])
                                bs["vxy_n"] += 1
                            if prev_h >= 1 and flight_peak_z[lane] > 1.05:
                                apex = float(flight_peak_z[lane])
                                bs["n_apex"] += 1
                                bs["apex_sum"] += apex
                                bs["apex_err_sq_sum"] += (apex - 1.425) ** 2
                                if 1.40 <= apex <= 1.45:
                                    bs["in_band"] += 1
                                if not math.isnan(prev_apex[lane]):
                                    bs["jump_sum"] += abs(apex - float(prev_apex[lane]))
                                    bs["n_jump"] += 1
                                prev_apex[lane] = apex
                        cur_hit_idx[lane] = new_h
                        if new_h > max_hits[lane]:
                            max_hits[lane] = new_h
                        flight_peak_z[lane] = float(ball_z[t, lane])

                newly_dead = alive & term[t]
                if np.any(newly_dead):
                    for lane in np.flatnonzero(newly_dead):
                        rname = regime_of(int(lane))
                        h_fail = int(cur_hit_idx[lane]) + 1
                        fail_sec[lane] = sec_now
                        fb = bucket_of(h_fail)
                        if fb is not None:
                            b_stats[rname][fb]["fails"] += 1
                        if t_np["rz_fail"][t, lane] > 0.5:
                            fail_cause[lane] = 1
                        elif t_np["rec_fail"][t, lane] > 0.5:
                            fail_cause[lane] = 2
                        elif t_np["low_fail"][t, lane] > 0.5:
                            fail_cause[lane] = 3
                        elif t_np["high_fail"][t, lane] > 0.5:
                            fail_cause[lane] = 4
                        else:
                            fail_cause[lane] = 5
                    alive &= ~newly_dead

            last_conf = np.where(alive, t_np["conf_ratio"][-1], last_conf)
            last_phys = np.where(alive, t_np["phys_ratio"][-1], last_phys)
            if c + 1 == chunk_20s:
                alive_20s = alive.copy()
                strict_20s = (
                    alive_20s
                    & (max_hits >= 24)
                    & (t_np["conf_ratio"][-1] <= 1.0 + 1e-6)
                    & (t_np["phys_ratio"][-1] <= 1.0 + 1e-6)
                )

        wall = time.monotonic() - t0
        strict_60s = (
            alive
            & (max_hits >= int(math.ceil(1.2 * seconds)))
            & (last_conf <= 1.0 + 1e-6)
            & (last_phys <= 1.0 + 1e-6)
        )
        print(f"\n================ [{label} | 1280 envs x 60s] WALL={wall:.1f}s ================", flush=True)

        for rname, lo, hi in REGIMES:
            sl = slice(lo, hi)
            fc = {
                k: int(np.sum(fail_cause[sl] == v))
                for k, v in [("rz", 1), ("rec", 2), ("low", 3), ("high", 4), ("other", 5)]
            }
            subgroups = {}
            if rname.startswith("constant"):
                lbl = labels_for_wind(init_mode[sl], init_bg[sl], init_ang[sl], 45.0)
                bg = init_bg[sl]
                for sname, m in [
                    ("low_0_0.8", bg < 0.8),
                    ("mid_0.8_1.2", (bg >= 0.8) & (bg < 1.2)),
                    ("high_1.2_1.6", (bg >= 1.2) & (bg < 1.599)),
                    ("peak_1.6", lbl >= 17),
                    ("negX_peak", np.isin(lbl, [17, 18, 19, 20, 25])),
                    ("posX_peak", np.isin(lbl, [21, 22, 23, 24, 26])),
                    ("dir08", lbl == 25),
                ]:
                    if np.any(m):
                        subgroups[sname] = {
                            "n": int(m.sum()),
                            "phys_20s": round(float(alive_20s[sl][m].mean()) * 100, 2),
                            "phys_60s": round(float(alive[sl][m].mean()) * 100, 2),
                            "strict_60s": round(float(strict_60s[sl][m].mean()) * 100, 2),
                            "mean_hits": round(float(max_hits[sl][m].mean()), 1),
                        }
            elif rname.startswith("dynamic"):
                md = init_mode[sl]
                for sname, m in [("slow_OU_mode2", md == 2), ("gust_mode3", md == 3)]:
                    if np.any(m):
                        subgroups[sname] = {
                            "n": int(m.sum()),
                            "phys_20s": round(float(alive_20s[sl][m].mean()) * 100, 2),
                            "phys_60s": round(float(alive[sl][m].mean()) * 100, 2),
                            "strict_60s": round(float(strict_60s[sl][m].mean()) * 100, 2),
                            "mean_hits": round(float(max_hits[sl][m].mean()), 1),
                        }

            bucket_report = {}
            for b in BUCKETS:
                bs = b_stats[rname][b]
                nh, na = bs["n_hits"], bs["n_apex"]
                bucket_report[f"{b[0]}-{b[1]}"] = {
                    "n_hits": nh,
                    "fails": bs["fails"],
                    "center_rms_mm": round(math.sqrt(bs["center_sq_sum"] / nh) * 1000, 2) if nh else None,
                    "apex_mean_m": round(bs["apex_sum"] / na, 4) if na else None,
                    "apex_err_rms_mm": round(math.sqrt(bs["apex_err_sq_sum"] / na) * 1000, 2) if na else None,
                    "in_band_pct": round(bs["in_band"] / na * 100, 2) if na else None,
                    "adj_jump_mm": round((bs["jump_sum"] / bs["n_jump"]) * 1000, 2) if bs["n_jump"] else None,
                    "racket_vxy_rms": round(math.sqrt(bs["vxy_sq_sum"] / bs["vxy_n"]), 4) if bs["vxy_n"] else None,
                }
            st_nh = sum(b_stats[rname][b]["n_hits"] for b in BUCKETS[1:])
            st_na = sum(b_stats[rname][b]["n_apex"] for b in BUCKETS[1:])
            st_nj = sum(b_stats[rname][b]["n_jump"] for b in BUCKETS[1:])
            st_nv = sum(b_stats[rname][b]["vxy_n"] for b in BUCKETS[1:])
            steady = {
                "n_hits": st_nh,
                "center_rms_mm": round(
                    math.sqrt(sum(b_stats[rname][b]["center_sq_sum"] for b in BUCKETS[1:]) / st_nh) * 1000, 2
                )
                if st_nh
                else None,
                "apex_mean_m": round(sum(b_stats[rname][b]["apex_sum"] for b in BUCKETS[1:]) / st_na, 4)
                if st_na
                else None,
                "apex_err_rms_mm": round(
                    math.sqrt(sum(b_stats[rname][b]["apex_err_sq_sum"] for b in BUCKETS[1:]) / st_na) * 1000, 2
                )
                if st_na
                else None,
                "in_band_pct": round(sum(b_stats[rname][b]["in_band"] for b in BUCKETS[1:]) / st_na * 100, 2)
                if st_na
                else None,
                "adj_jump_mm": round(sum(b_stats[rname][b]["jump_sum"] for b in BUCKETS[1:]) / st_nj * 1000, 2)
                if st_nj
                else None,
                "racket_vxy_rms": round(
                    math.sqrt(sum(b_stats[rname][b]["vxy_sq_sum"] for b in BUCKETS[1:]) / st_nv), 4
                )
                if st_nv
                else None,
            }
            res = {
                "model": label,
                "checkpoint": pkl_path.name,
                "regime": rname,
                "pose_deg": 5.0,
                "n_envs": n_per,
                "seconds": seconds,
                "seed": seed,
                "phys_20s_pct": round(float(alive_20s[sl].mean()) * 100, 2),
                "strict_20s_pct": round(float(strict_20s[sl].mean()) * 100, 2),
                "phys_60s_pct": round(float(alive[sl].mean()) * 100, 2),
                "strict_60s_pct": round(float(strict_60s[sl].mean()) * 100, 2),
                "mean_duration_s": round(float(fail_sec[sl].mean()), 2),
                "mean_hits": round(float(max_hits[sl].mean()), 1),
                "fail_causes": fc,
                "subgroups": subgroups,
                "steady_6_115": steady,
                "hit_buckets": bucket_report,
            }
            all_results.append(res)
            b15 = bucket_report["1-5"]
            b615 = bucket_report["6-15"]
            p20 = res["phys_20s_pct"]
            p60 = res["phys_60s_pct"]
            s60 = res["strict_60s_pct"]
            print(
                f">>> [{label} | {rname:14s}] 20s={p20:6.2f}% 60s={p60:6.2f}% (strict={s60:6.2f}%) "
                f"| h1_5={b15['center_rms_mm']}mm/{b15['racket_vxy_rms']} "
                f"| h6_15={b615['center_rms_mm']}mm/{b615['racket_vxy_rms']} "
                f"| steady={steady['center_rms_mm']}mm/inB={steady['in_band_pct']}%/vxy={steady['racket_vxy_rms']} "
                f"| fails={fc}",
                flush=True,
            )
            if subgroups:
                print(f"    subgroups: {subgroups}", flush=True)

        out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        print(f"Checkpoint saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
