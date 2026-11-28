"""Final 60s Large-Sample Benchmark + Showcase Rollout Recorder for S2R u7447.

Runs three verified suites on GPU for `milestone_S09_D2_dr65_u00007447.pkl`:
  1. Frozen 1280-env x 60s paired benchmark (`seed=2026112701`) under BOTH:
     - `P_warm` (real-robot steady-state RMP Kalman covariance)
     - `P_cold` (legacy cold-start parity vs u6247)
  2. 640-env x 60s Real-Robot Sim2Real Twin benchmark across 5 regimes x 128 envs:
     - `RealExact_0w`, `RealHard_0w`, `RealSoft_0w`, `RealExact_W16_Const`, `RealExact_W16_Dyn`
  3. 6-lane x 20s RealExact showcase trajectory recording (`u7447_s2r_showcase_20s.npz`)
     for clean background-only wind-visualized video rendering.
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


def run_frozen_1280x60s(stack, ckpt_path: Path, out_json: Path, script_dir: Path):
    import eval_vs_v321 as ev
    import fixed_plane_wind_pose
    from single5_runtime import install_pose_mode, Single5Env, constant_degrees
    from train_s15_fresh_velocity_r31 import ActualContactEnv
    from capture_margin import CaptureEnv
    from xwind_sampling import XWindEnv, labels_for_wind, vectors_from_uniforms
    from wind_features import WindFeatureEnv
    from sim2real_r5 import P_COLD_3X3, P_WARM_3X3

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
    p_warm_j = jnp.asarray(P_WARM_3X3, dtype=jnp.float32)
    p_cold_j = jnp.asarray(P_COLD_3X3, dtype=jnp.float32)

    @jax.jit
    def reset_1280(use_warm_p):
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
        is_zero_dyn = (idx < 2 * n_per) | ((idx >= 3 * n_per) & (idx < 4 * n_per))

        new_mode = (
            st.wind_mode.at[:n_per].set(0)
            .at[n_per : 2 * n_per].set(c_modes)
            .at[3 * n_per : 4 * n_per].set(c_modes)
        )
        new_bg = (
            st.wind_background_m_s.at[:n_per].set(0.0)
            .at[n_per : 2 * n_per].set(c_vecs)
            .at[3 * n_per : 4 * n_per].set(c_vecs)
        )
        new_ou = jnp.where(is_zero_dyn[:, None], 0.0, st.wind_ou_m_s)
        new_gust = jnp.where(is_zero_dyn[:, None], 0.0, st.wind_gust_vector_m_s)

        late_delays = jax.vmap(
            lambda key: jax.random.randint(
                jax.random.fold_in(key, 2026112799), (), 2000, 4001, dtype=jnp.int32
            )
        )(keys_768[n_per:])
        new_delay = st.delay_steps.at[3 * n_per :].set(late_delays)

        cov_mat = jnp.where(use_warm_p, p_warm_j, p_cold_j)
        cov_b = jnp.broadcast_to(cov_mat[None, None, :, :], st.rmp_state.target_covariance.shape)
        new_rmp = st.rmp_state._replace(target_covariance=cov_b)

        st = st._replace(
            wind_mode=new_mode,
            wind_background_m_s=new_bg,
            wind_ou_m_s=new_ou,
            wind_gust_vector_m_s=new_gust,
            delay_steps=new_delay,
            rmp_state=new_rmp,
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

    with ckpt_path.open("rb") as f:
        params = jax.device_put(pickle.load(f)["params"])

    all_results = []
    for label, use_warm in [
        ("V325_s2r_u7447_Pwarm_det", True),
        ("V325_s2r_u7447_Pcold_det", False),
    ]:
        state, obs = reset_1280(jnp.asarray(use_warm, dtype=jnp.bool_))
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
                b: dict(
                    n_hits=0, center_sq_sum=0.0, center_abs_sum=0.0,
                    vxy_sq_sum=0.0, vxy_n=0, n_apex=0, apex_sum=0.0,
                    apex_err_sq_sum=0.0, in_band=0, jump_sum=0.0, n_jump=0, fails=0,
                )
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
                print(f"[{label}] chunk 0 verified in {time.monotonic()-t0:.1f}s", flush=True)

            term = t_np["terminated"] > 0.5
            hit_evt = t_np["hit_evt"] > 0.5
            ball_z = t_np["ball_z"]
            for t in range(chunk):
                sec_now = (c * chunk + t + 1) * env.dt
                flight_peak_z = np.where(alive, np.maximum(flight_peak_z, ball_z[t]), flight_peak_z)
                h_mask = alive & hit_evt[t]
                if np.any(h_mask):
                    for lane in np.flatnonzero(h_mask):
                        rname = REGIMES[int(lane) // n_per][0]
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
                        rname = REGIMES[int(lane) // n_per][0]
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
                    alive_20s & (max_hits >= 24)
                    & (t_np["conf_ratio"][-1] <= 1.0 + 1e-6)
                    & (t_np["phys_ratio"][-1] <= 1.0 + 1e-6)
                )

        wall = time.monotonic() - t0
        strict_60s = (
            alive & (max_hits >= int(math.ceil(1.2 * seconds)))
            & (last_conf <= 1.0 + 1e-6)
            & (last_phys <= 1.0 + 1e-6)
        )
        print(f"\n================ [{label} | 1280 envs x 60s] WALL={wall:.1f}s ================", flush=True)
        for rname, lo, hi in REGIMES:
            sl = slice(lo, hi)
            fc = {k: int(np.sum(fail_cause[sl] == v)) for k, v in [("rz", 1), ("rec", 2), ("low", 3), ("high", 4), ("other", 5)]}
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
                    "n_hits": nh, "fails": bs["fails"],
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
                "center_rms_mm": round(math.sqrt(sum(b_stats[rname][b]["center_sq_sum"] for b in BUCKETS[1:]) / st_nh) * 1000, 2) if st_nh else None,
                "apex_mean_m": round(sum(b_stats[rname][b]["apex_sum"] for b in BUCKETS[1:]) / st_na, 4) if st_na else None,
                "apex_err_rms_mm": round(math.sqrt(sum(b_stats[rname][b]["apex_err_sq_sum"] for b in BUCKETS[1:]) / st_na) * 1000, 2) if st_na else None,
                "in_band_pct": round(sum(b_stats[rname][b]["in_band"] for b in BUCKETS[1:]) / st_na * 100, 2) if st_na else None,
                "adj_jump_mm": round(sum(b_stats[rname][b]["jump_sum"] for b in BUCKETS[1:]) / st_nj * 1000, 2) if st_nj else None,
                "racket_vxy_rms": round(math.sqrt(sum(b_stats[rname][b]["vxy_sq_sum"] for b in BUCKETS[1:]) / st_nv), 4) if st_nv else None,
            }
            res = dict(
                model=label, checkpoint=ckpt_path.name, kalman_init="P_warm" if use_warm else "P_cold",
                regime=rname, pose_deg=5.0, n_envs=n_per, seconds=seconds, seed=seed,
                phys_20s_pct=round(float(alive_20s[sl].mean()) * 100, 2),
                strict_20s_pct=round(float(strict_20s[sl].mean()) * 100, 2),
                phys_60s_pct=round(float(alive[sl].mean()) * 100, 2),
                strict_60s_pct=round(float(strict_60s[sl].mean()) * 100, 2),
                mean_duration_s=round(float(fail_sec[sl].mean()), 2),
                mean_hits=round(float(max_hits[sl].mean()), 1),
                fail_causes=fc, subgroups=subgroups, steady_6_115=steady, hit_buckets=bucket_report,
            )
            all_results.append(res)
            b15, b615 = bucket_report["1-5"], bucket_report["6-15"]
            print(
                f">>> [{label} | {rname:14s}] 60s={res['phys_60s_pct']:6.2f}% (strict={res['strict_60s_pct']:6.2f}%) "
                f"| h1_5={b15['center_rms_mm']}mm h6_15={b615['center_rms_mm']}mm "
                f"| steady={steady['center_rms_mm']}mm/inB={steady['in_band_pct']}%/vxy={steady['racket_vxy_rms']} | fails={fc}",
                flush=True,
            )
        out_json.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"Saved frozen 1280x60s benchmark to {out_json}", flush=True)


def run_s2r_640x60s_and_showcase(stack, ckpt_path: Path, out_json: Path, out_npz: Path):
    from sim2real_r5 import P_WARM_3X3
    from observed_force import observed_force_features
    from wind_features import WindFeatureEnv, encode as wind_features_encode
    from train_s15_fresh_velocity_r31 import ActualContactEnv
    from capture_margin import CaptureEnv
    from single5_runtime import Single5Env, install_pose_mode

    install_pose_mode()
    n_scen = 5
    per_scen = 128
    n_envs = n_scen * per_scen  # 640 envs
    seconds = 60.0

    cfg = replace(
        stack.cfg,
        ball_spawn_xy_jitter=0.0,
        dr_randomize_recovered_rmp=False,
        wind_config=replace(
            stack.cfg.wind_config,
            max_speed_m_s=1.6,
            mode_weights=(0.0, 0.0, 0.5, 0.5),
            wind_y_nonpositive=True,
            startup_no_wind_hits=5,
            startup_ramp_s=1.0,
        ),
    )
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=cfg)
    env = Single5Env(
        CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
        reward_enabled=False, goal_reward_enabled=False,
        random_onset_enabled=False, targeted_reward_enabled=False,
    )
    env = WindFeatureEnv(env)

    pair_id = int(native.ball_racket_pair_ids[0]) if hasattr(native, "ball_racket_pair_ids") and len(native.ball_racket_pair_ids) else 0
    ball_qadr = int(native.mj_model.jnt_qposadr[native.ball_joint_id])
    racket_site_id = int(native.racket_site_id)
    nom_kp = np.asarray(jax.device_get(native.recovered_rmp_jnt_kp), dtype=np.float32)
    nom_kd = np.asarray(jax.device_get(native.recovered_rmp_jnt_kd), dtype=np.float32)

    scen_names = [
        "RealExact_0w",
        "RealHard_0w",
        "RealSoft_0w",
        "RealExact_W16_Const",
        "RealExact_W16_Dyn",
    ]
    rng = np.random.default_rng(2026112860)
    cov_arr = np.broadcast_to(P_WARM_3X3[None, None, :, :], (n_envs, 7, 3, 3)).copy()
    grav_arr = np.tile(np.array([0.0, 0.135, -9.67], dtype=np.float32), (n_envs, 1))
    damp_arr = np.full((n_envs,), 0.285, dtype=np.float32)
    lag_arr = np.where(np.arange(n_envs) % 2 == 0, 2, 3).astype(np.int32)
    bias_arr = np.tile(np.array([0.0, 0.0, -0.012], dtype=np.float32), (n_envs, 1))
    nstd_arr = np.tile(np.array([0.0006, 0.0006, 0.0008], dtype=np.float32), (n_envs, 1))
    ang_all = np.linspace(0, 2 * np.pi, per_scen, endpoint=False)
    sdx_arr = np.zeros((n_envs, 2), dtype=np.float32)
    for si in range(n_scen):
        s = slice(si * per_scen, (si + 1) * per_scen)
        sdx_arr[s, 0] = 0.006 * np.cos(ang_all)
        sdx_arr[s, 1] = 0.006 * np.sin(ang_all)
    kp_arr = np.tile(nom_kp[None, :], (n_envs, 1))
    kd_arr = np.tile(nom_kd[None, :], (n_envs, 1))
    wmode_arr = np.zeros((n_envs,), dtype=np.int32)
    wbg_arr = np.zeros((n_envs, 3), dtype=np.float32)
    wou_arr = np.zeros((n_envs, 3), dtype=np.float32)
    wg_arr = np.zeros((n_envs, 3), dtype=np.float32)

    # Scen 1: RealHard_0w
    s1 = slice(1 * per_scen, 2 * per_scen)
    grav_arr[s1, 0] = rng.uniform(-0.04, 0.04, per_scen)
    grav_arr[s1, 1] = rng.uniform(0.10, 0.18, per_scen)
    grav_arr[s1, 2] = rng.uniform(-9.74, -9.62, per_scen)
    damp_arr[s1] = rng.uniform(0.26, 0.315, per_scen)
    lag_arr[s1] = rng.choice([2, 3, 4], size=per_scen, p=[0.35, 0.45, 0.20]).astype(np.int32)
    bias_arr[s1, 0] = rng.uniform(-0.004, 0.004, per_scen)
    bias_arr[s1, 1] = rng.uniform(-0.004, 0.004, per_scen)
    bias_arr[s1, 2] = rng.uniform(-0.015, -0.008, per_scen)
    sdx_arr[s1] = rng.uniform(-0.006, 0.006, (per_scen, 2))
    kp_arr[s1] *= rng.uniform(0.92, 1.08, (per_scen, 1))
    kd_arr[s1] *= rng.uniform(0.92, 1.08, (per_scen, 1))

    # Scen 2: RealSoft_0w (damp=0.33)
    s2 = slice(2 * per_scen, 3 * per_scen)
    damp_arr[s2] = 0.33

    # Scen 3: RealExact + 1.6 m/s Const X-cone
    s3 = slice(3 * per_scen, 4 * per_scen)
    wmode_arr[s3] = 1
    angles = np.concatenate([
        np.linspace(-np.pi, -0.75 * np.pi, per_scen // 2),
        np.linspace(-0.25 * np.pi, 0.0, per_scen // 2),
    ])
    wbg_arr[s3, 0] = 1.6 * np.cos(angles)
    wbg_arr[s3, 1] = np.minimum(1.6 * np.sin(angles), 0.0)
    # Pin exact clean showcase lanes inside s3: lane 0 -> -X (-180deg), lane 64 -> -45deg, lane 127 -> +X (0deg)

    # Scen 4: RealExact + 1.6 m/s Dyn (64 OU + 64 Gust)
    s4 = slice(4 * per_scen, 5 * per_scen)
    wmode_arr[s4] = np.array([2] * (per_scen // 2) + [3] * (per_scen // 2), dtype=np.int32)
    angs4 = np.linspace(-np.pi, 0.0, per_scen // 2)
    for k in range(per_scen // 2):
        dx, dy = np.cos(angs4[k]), min(np.sin(angs4[k]), 0.0)
        wbg_arr[4 * per_scen + k] = [1.0 * dx, 1.0 * dy, 0.0]
        wou_arr[4 * per_scen + k] = [0.6 * dx, 0.6 * dy, 0.0]
        wbg_arr[4 * per_scen + (per_scen // 2) + k] = [0.25 * dx, 0.25 * dy, 0.0]
        wg_arr[4 * per_scen + (per_scen // 2) + k] = [1.35 * dx, 1.35 * dy, 0.0]

    # Pick 6 showcase lanes from this exact 640-env batch for 20s video recording!
    # 0: RealExact 0w | 3*128+127: +X 1.6 | 3*128+64: -45deg 1.6 | 3*128+0: -X 1.6 | 4*128+64: Gust -X | 4*128+0: Slow OU -X
    showcase_lanes = np.array([0, 3 * per_scen + 127, 3 * per_scen + 64, 3 * per_scen + 0, 4 * per_scen + 64, 4 * per_scen + 0], dtype=np.int32)
    showcase_j = jnp.asarray(showcase_lanes)

    cov_j, grav_j, damp_j = jnp.asarray(cov_arr), jnp.asarray(grav_arr), jnp.asarray(damp_arr)
    lag_j, bias_j, nstd_j = jnp.asarray(lag_arr), jnp.asarray(bias_arr), jnp.asarray(nstd_arr)
    sdx_j, kp_j, kd_j = jnp.asarray(sdx_arr), jnp.asarray(kp_arr), jnp.asarray(kd_arr)
    wmode_j, wbg_j, wou_j, wg_j = jnp.asarray(wmode_arr), jnp.asarray(wbg_arr), jnp.asarray(wou_arr), jnp.asarray(wg_arr)

    def assemble(co, r28, r72):
        idx = jnp.arange(n_envs)
        slot = 4 - lag_j
        obs112 = co[:, :112].at[:, 28:52].set(r28[idx, slot]).at[:, 72:112].set(r72[idx, slot])
        obs118 = jnp.concatenate([obs112, co[:, 112:114], observed_force_features(obs112, xp=jnp)], axis=-1)
        return wind_features_encode(obs118, xp=jnp)

    @jax.jit
    def reset_640():
        keys = jax.random.split(jax.random.PRNGKey(2026112861), n_envs)
        st, clean_obs = env.reset(keys)
        new_qpos = st.data.qpos.at[:, ball_qadr : ball_qadr + 2].add(sdx_j)
        st = st._replace(
            wind_mode=wmode_j, wind_background_m_s=wbg_j, wind_ou_m_s=wou_j, wind_gust_vector_m_s=wg_j,
            rmp_state=st.rmp_state._replace(target_covariance=cov_j),
            rmp_jnt_kp=kp_j, rmp_jnt_kd=kd_j,
            model=st.model.replace(opt=st.model.opt.replace(gravity=grav_j), pair_solref=st.model.pair_solref.at[:, pair_id, 1].set(damp_j)),
            data=st.data.replace(qpos=new_qpos),
        )
        p28_0 = clean_obs[:, 28:52].reshape((-1, 8, 3)).at[:, :, :2].add(sdx_j[:, None, :])
        p28_0 = (p28_0 + bias_j[:, None, :]).reshape((-1, 24))
        r28 = jnp.repeat(p28_0[:, None, :], 5, axis=1)
        r72 = jnp.repeat(clean_obs[:, 72:112][:, None, :], 5, axis=1)
        nrng = jax.random.split(jax.random.PRNGKey(77771), n_envs)
        r0 = st.data.site_xmat[:, racket_site_id].reshape((n_envs, 3, 3))
        return st, assemble(clean_obs, r28, r72), r28, r72, nrng, r0

    chunk = 128

    @jax.jit
    def step_chunk_640(params, st, ob, r28, r72, kr, r0):
        def body(carry, _):
            s, o, b28, b72, rng_k = carry
            a = stack.ppo.policy_mean(params, o)
            s2, co2, rew, done, m = env.step(s, a)
            ks = jax.vmap(lambda k: jax.random.split(k, 2))(rng_k)
            rng_k2 = ks[:, 0]
            eps = jax.vmap(lambda k: jax.random.normal(k, (16, 3), dtype=jnp.float32))(ks[:, 1])
            p28 = (co2[:, 28:52].reshape((-1, 8, 3)) + bias_j[:, None, :] + eps[:, :8, :] * nstd_j[:, None, :]).reshape((-1, 24))
            v8 = co2[:, 104:112] > 0.5
            eh = (eps[:, 8:, :] * (nstd_j[:, None, :] * 1.4142135)).at[:, 0, :].set(0.0)
            h72 = jnp.where(v8[:, :, None], co2[:, 72:96].reshape((-1, 8, 3)) + eh, co2[:, 72:96].reshape((-1, 8, 3))).reshape((-1, 24))
            p72 = jnp.concatenate([h72, co2[:, 96:112]], axis=-1)
            b28_2 = jnp.concatenate([b28[:, 1:, :], p28[:, None, :]], axis=1)
            b72_2 = jnp.concatenate([b72[:, 1:, :], p72[:, None, :]], axis=1)
            o2 = assemble(co2, b28_2, b72_2)
            r_cur = s2.data.site_xmat[:, racket_site_id].reshape((n_envs, 3, 3))
            r_rel = jnp.einsum("nij,nkj->nik", r_cur, r0)
            tilt = jnp.degrees(jnp.arccos(jnp.clip(r_rel[:, 2, 2], -1.0, 1.0)))
            wvec = s2.wind_active_m_s if hasattr(s2, "wind_active_m_s") else (m.get("wind/speed_m_s", jnp.zeros(n_envs))[:, None] * jnp.stack([jnp.cos(m["single5/wind_angle_rad"]), jnp.sin(m["single5/wind_angle_rad"]), jnp.zeros(n_envs)], axis=-1))
            out = {
                "terminated": m["terminated"],
                "hits": s2.wind_startup_hit_count,
                "hit_evt": m["hit_event_count"],
                "center_sum": m["hit_contact_center_dist_sum"],
                "tilt": tilt,
                "sc_qpos": s2.data.qpos[showcase_j],
                "sc_wx": wvec[showcase_j, 0],
                "sc_wy": wvec[showcase_j, 1],
                "sc_hits": s2.wind_startup_hit_count[showcase_j],
                "sc_hevt": m["hit_event_count"][showcase_j],
                "sc_csum": m["hit_contact_center_dist_sum"][showcase_j],
                "sc_tilt": tilt[showcase_j],
            }
            return (s2, o2, b28_2, b72_2, rng_k2), out

        (s_e, o_e, b28_e, b72_e, kr_e), traj = jax.lax.scan(body, (st, ob, r28, r72, kr), None, length=chunk)
        return s_e, o_e, b28_e, b72_e, kr_e, traj

    with ckpt_path.open("rb") as f:
        params = jax.device_put(pickle.load(f)["params"])

    t0 = time.monotonic()
    st, ob, r28, r72, kr, r0 = reset_640()
    n_chunks = int(math.ceil((seconds / env.dt) / chunk))  # 94 chunks -> 60.16s
    alive = np.ones(n_envs, dtype=bool)
    dur = np.zeros(n_envs, dtype=np.float32)
    max_hits = np.zeros(n_envs, dtype=np.int32)
    tilt_sum = np.zeros(n_envs, dtype=np.float64)
    tilt_max = np.zeros(n_envs, dtype=np.float32)
    live_steps = np.zeros(n_envs, dtype=np.float64)
    csq = np.zeros(n_envs, dtype=np.float64)
    ccnt = np.zeros(n_envs, dtype=np.float64)

    # Record first 4000 steps (20.0s) for showcase video (stride=4 -> 1000 frames @ 50 fps)
    sc_qpos_list, sc_wx_list, sc_wy_list, sc_hits_list, sc_hevt_list, sc_csum_list, sc_tilt_list = [], [], [], [], [], [], []
    steps_recorded = 0

    for c in range(n_chunks):
        st, ob, r28, r72, kr, traj = step_chunk_640(params, st, ob, r28, r72, kr, r0)
        t_np = {k: np.asarray(jax.device_get(v)) for k, v in traj.items()}
        if c == 0:
            print(f"[S2R 640x60s] chunk 0 compiled+ran in {time.monotonic()-t0:.1f}s", flush=True)
        if steps_recorded < 4000:
            take = min(chunk, 4000 - steps_recorded)
            sc_qpos_list.append(t_np["sc_qpos"][:take])
            sc_wx_list.append(t_np["sc_wx"][:take])
            sc_wy_list.append(t_np["sc_wy"][:take])
            sc_hits_list.append(t_np["sc_hits"][:take])
            sc_hevt_list.append(t_np["sc_hevt"][:take])
            sc_csum_list.append(t_np["sc_csum"][:take])
            sc_tilt_list.append(t_np["sc_tilt"][:take])
            steps_recorded += take

        term = t_np["terminated"] > 0.5
        he = t_np["hit_evt"]
        cs = t_np["center_sum"]
        hits_t = t_np["hits"]
        tilt_t = t_np["tilt"]
        for t in range(chunk):
            if c * chunk + t >= int(round(seconds / env.dt)):
                break
            dur += alive.astype(np.float32) * float(env.dt)
            max_hits = np.where(alive, np.maximum(max_hits, hits_t[t]), max_hits)
            tilt_sum += np.where(alive, tilt_t[t], 0.0)
            tilt_max = np.where(alive, np.maximum(tilt_max, tilt_t[t]), tilt_max)
            live_steps += alive.astype(np.float64)
            hm = alive & (he[t] > 0.5)
            if np.any(hm):
                cd = cs[t] / np.maximum(he[t], 1.0)
                csq += np.where(hm, cd * cd, 0.0)
                ccnt += hm.astype(np.float64)
            alive &= ~term[t]

    wall = time.monotonic() - t0
    print(f"\n================ [S2R u7447 | 640 envs x 60s Real-Robot Twin] WALL={wall:.1f}s ================", flush=True)
    s2r_summary = {"model": "V325_s2r_u7447_det", "checkpoint": ckpt_path.name, "seconds": seconds, "per_scenario": per_scen, "scenarios": {}}
    for si, sname in enumerate(scen_names):
        s = slice(si * per_scen, (si + 1) * per_scen)
        surv = float(alive[s].mean()) * 100.0
        strict = float((alive[s] & (max_hits[s] >= 72)).mean()) * 100.0
        t_avg = float((tilt_sum[s] / np.maximum(live_steps[s], 1.0)).mean())
        t_mx = float(tilt_max[s].mean())
        c_rms = float(np.sqrt(csq[s].sum() / max(ccnt[s].sum(), 1.0)) * 1000.0)
        s2r_summary["scenarios"][sname] = dict(
            n_envs=per_scen, phys_60s_pct=round(surv, 2), strict_60s_pct=round(strict, 2),
            mean_duration_s=round(float(dur[s].mean()), 2), mean_hits=round(float(max_hits[s].mean()), 1),
            tilt_avg_deg=round(t_avg, 2), tilt_max_deg=round(t_mx, 2), center_rms_mm=round(c_rms, 2),
        )
        print(
            f">>> [{sname:20s}] 60s={surv:6.2f}% (strict>=72h={strict:6.2f}%) | "
            f"s={dur[s].mean():5.2f}s h={max_hits[s].mean():5.1f} | "
            f"tilt(avg/max)={t_avg:4.2f}/{t_mx:4.2f}d | cRMS={c_rms:4.1f}mm",
            flush=True,
        )
    out_json.write_text(json.dumps(s2r_summary, indent=2, ensure_ascii=False))
    print(f"Saved S2R 640x60s benchmark to {out_json}", flush=True)

    # Save 20.0s showcase trajectories (stride=4 -> 1000 frames @ 50 fps)
    stride = 4
    sc_qpos = np.concatenate(sc_qpos_list, axis=0)[::stride].astype(np.float32)
    sc_wx = np.concatenate(sc_wx_list, axis=0)[::stride].astype(np.float32)
    sc_wy = np.concatenate(sc_wy_list, axis=0)[::stride].astype(np.float32)
    sc_hits = np.concatenate(sc_hits_list, axis=0)[::stride].astype(np.int32)
    sc_tilt = np.concatenate(sc_tilt_list, axis=0)[::stride].astype(np.float32)
    sc_hevt_full = np.concatenate(sc_hevt_list, axis=0).astype(np.float32)
    sc_csum_full = np.concatenate(sc_csum_list, axis=0).astype(np.float32)
    # Compute exact onset time per lane
    wspd_sc = np.hypot(sc_wx, sc_wy)
    onset_s = np.full((len(showcase_lanes),), -1.0, dtype=np.float32)
    for k in range(len(showcase_lanes)):
        nz = np.flatnonzero(wspd_sc[:, k] > 0.05)
        if len(nz) > 0:
            onset_s[k] = float(nz[0] * env.dt * stride)
    np.savez_compressed(
        out_npz,
        qpos=sc_qpos, wind_x=sc_wx, wind_y=sc_wy, hits=sc_hits, tilt=sc_tilt,
        onset_s=onset_s, hit_evt_full=sc_hevt_full, center_sum_full=sc_csum_full,
        ball_qadr=np.int32(ball_qadr), dt=np.float32(env.dt), stride=np.int32(stride),
        lane_labels=np.array([
            "RealExact 0w (P_warm+ay+0.135+lag+dz)",
            "RealExact +X Peak (0 deg, 1.6 m/s)",
            "RealExact Diag Peak (-45 deg, 1.6 m/s)",
            "RealExact -X Peak (-180 deg, 1.6 m/s)",
            "RealExact Dynamic Gust (<=1.6 m/s)",
            "RealExact Dynamic Slow OU (<=1.6 m/s)",
        ]),
    )
    print(f"Saved 20s showcase npz to {out_npz} (onset_s={np.round(onset_s, 2).tolist()}, end_hits={sc_hits[-1].tolist()})", flush=True)


def main():
    script_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    sys.path.insert(0, str(script_dir))
    from migrated_stack import REPO, load_migrated_v325_stack, setup_sys_path
    setup_sys_path()

    ckpt_path = REPO / "pingpong_controller/outputs/rl_sim/fplc_wind_early_20261128_s2r_v2/milestone_S09_D2_dr65_u00007447.pkl"
    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)

    run_frozen_1280x60s(stack, ckpt_path, out_dir / "evaluation_1280x60s.json", script_dir)
    run_s2r_640x60s_and_showcase(
        stack, ckpt_path,
        out_dir / "evaluation_s2r_60s.json",
        out_dir / "u7447_s2r_showcase_20s.npz",
    )
    print("ALL 60s FINAL EVALUATIONS & SHOWCASE RECORDINGS COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
