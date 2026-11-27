"""Fast unified 768-env x 60s paired benchmark: Best V325 (u4768) vs V321 (S4/u2097).

Packs all 3 regimes into ONE single 768-env GPU pass per model (3x faster!):
  - Lanes   0..255: Clean (mode=0, 0 m/s)
  - Lanes 256..511: Constant Wind (mode=1, 0-1.6 m/s X-side, 25% 1.6m/s peak)
  - Lanes 512..767: Dynamic Wind (50% mode=2 slow OU + 50% mode=3 gust, 0-1.6 m/s, Y<=0)
Both models run in deterministic deployment mode with identical XML, DR, ball mass,
and initial seeds; authentic pose bounds enforced (5.0 deg for V325, 2.0 deg for V321).
"""
from dataclasses import replace
import json
import math
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

import fixed_plane_wind_pose
from single5_runtime import install_pose_mode, Single5Env, constant_degrees
from train_s15_fresh_velocity_r31 import ActualContactEnv
from capture_margin import CaptureEnv
from xwind_sampling import XWindEnv, labels_for_wind, vectors_from_uniforms
from wind_features import WindFeatureEnv


def constant_degrees_2(state, dt_s, cfg, *, xp):
    return xp.ones_like(state.step_count, dtype=xp.float32) * 2.0


def apply_v321_pi(pi, obs120):
    x = obs120[..., :72]
    x = jnp.tanh(jnp.matmul(x, pi["l1"]["w"], precision=jax.lax.Precision.HIGHEST) + pi["l1"]["b"])
    x = jnp.tanh(jnp.matmul(x, pi["l2"]["w"], precision=jax.lax.Precision.HIGHEST) + pi["l2"]["b"])
    return jnp.matmul(x, pi["out"]["w"], precision=jax.lax.Precision.HIGHEST) + pi["out"]["b"]


def constant_wind_layout(n: int):
    assert n % 16 == 0
    mode = np.ones(n, np.int32)
    sector = np.zeros(n, np.int32)
    bins = np.zeros(n, np.int32)
    axis = np.full(n, -1, np.int32)
    hard = np.zeros(n, bool)
    peak_n = n // 4
    uniform = peak_n * 5 // 8
    sector[:peak_n] = np.arange(peak_n) % 4
    bins[:peak_n] = 4
    axis[uniform:peak_n] = np.arange(peak_n - uniform) % 2
    rem = n - peak_n
    slots = np.arange(rem)
    sector[peak_n:] = slots % 4
    bins[peak_n:] = (slots // 4) % 4
    return mode, sector, bins, axis, hard


def build_unified_env(stack, pose_deg: float, n_per_regime: int = 256):
    n_envs = 3 * n_per_regime
    wcfg = replace(
        stack.cfg.wind_config,
        mode_weights=(0.0, 0.0, 0.5, 0.5),
        max_speed_m_s=1.6,
        wind_y_nonpositive=True,
        startup_no_wind_hits=5,
        startup_ramp_s=1.0,
    )
    cfg = replace(stack.cfg, wind_config=wcfg)
    if pose_deg == 2.0:
        cfg = replace(
            cfg,
            cartesian_pose_roll_limit_deg=2.0,
            cartesian_pose_pitch_limit_deg=2.0,
            wind_pose=replace(cfg.wind_pose, start_deg=2.0, target_deg=2.0),
        )
    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=cfg)
    xenv = XWindEnv(CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
                    half_angle_deg=45.0, enabled=False, calm_only=False)
    env = Single5Env(xenv, reward_enabled=False, goal_reward_enabled=False,
                     random_onset_enabled=True, targeted_reward_enabled=False,
                     weak_release_floor_enabled=True, weak_release_floor_weight=4.0)
    env = WindFeatureEnv(env)
    return env


def assign_unified_winds(state, keys, n_per: int = 256):
    """Lanes 0..n-1: clean; n..2n-1: constant X-side; 2n..3n-1: native dynamic (50% OU + 50% gust)."""
    n_envs = 3 * n_per
    c_keys = keys[n_per:2 * n_per]
    uniforms = jax.vmap(lambda key: jax.random.uniform(jax.random.fold_in(key, 2026112312), (2,)))(c_keys)
    c_vecs, c_modes = vectors_from_uniforms(uniforms, constant_wind_layout(n_per), 45.0, xp=jnp)

    idx = jnp.arange(n_envs)
    is_clean = idx < n_per
    is_const = (idx >= n_per) & (idx < 2 * n_per)

    new_mode = state.wind_mode.at[:n_per].set(0).at[n_per:2 * n_per].set(c_modes)
    new_bg = state.wind_background_m_s.at[:n_per].set(0.0).at[n_per:2 * n_per].set(c_vecs)
    new_ou = jnp.where((is_clean | is_const)[:, None], 0.0, state.wind_ou_m_s)
    new_gust = jnp.where((is_clean | is_const)[:, None], 0.0, state.wind_gust_vector_m_s)
    return state._replace(
        wind_mode=new_mode,
        wind_background_m_s=new_bg,
        wind_ou_m_s=new_ou,
        wind_gust_vector_m_s=new_gust,
    )


def run_model_unified(label: str, params, policy_fn, stack, pose_deg: float,
                      n_per: int = 256, seconds: float = 60.0, seed: int = 2026112701):
    n_envs = 3 * n_per
    env = build_unified_env(stack, pose_deg, n_per)
    fixed_plane_wind_pose.effective_degrees = constant_degrees_2 if pose_deg == 2.0 else constant_degrees
    ball_qadr = env.ball_qadr

    @jax.jit
    def reset_unified(keys):
        st, ob = env.reset(keys)
        st = assign_unified_winds(st, keys, n_per)
        return st, ob

    chunk = 128

    @jax.jit
    def step_chunk(params, state, obs):
        def body(carry, _):
            st, ob = carry
            action = policy_fn(params, ob)
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

    keys = jax.random.split(jax.random.PRNGKey(seed), n_envs)
    state, obs = reset_unified(keys)

    alive = np.ones(n_envs, dtype=bool)
    alive_20s = np.ones(n_envs, dtype=bool)
    strict_20s = np.zeros(n_envs, dtype=bool)
    fail_sec = np.full(n_envs, seconds, dtype=np.float32)
    max_hits = np.zeros(n_envs, dtype=np.int32)
    fail_cause = np.zeros(n_envs, dtype=np.int32)  # 0=alive, 1=rz, 2=rec, 3=low, 4=high, 5=other

    cur_hit_idx = np.zeros(n_envs, dtype=np.int32)
    flight_peak_z = np.zeros(n_envs, dtype=np.float32)
    prev_apex = np.full(n_envs, np.nan, dtype=np.float32)

    BUCKETS = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 115)]
    REGIMES = [("clean", 0, n_per), ("constant", n_per, 2 * n_per), ("dynamic", 2 * n_per, 3 * n_per)]
    b_stats = {
        rname: {
            b: {
                "n_hits": 0, "center_sq_sum": 0.0, "center_abs_sum": 0.0,
                "vxy_sq_sum": 0.0, "vxy_n": 0,
                "n_apex": 0, "apex_sum": 0.0, "apex_err_sq_sum": 0.0, "in_band": 0,
                "jump_sum": 0.0, "n_jump": 0, "fails": 0,
            }
            for b in BUCKETS
        }
        for rname, _, _ in REGIMES
    }

    def bucket_of(h):
        for lo, hi in BUCKETS:
            if lo <= h <= hi:
                return (lo, hi)
        return None

    def regime_of(lane):
        if lane < n_per: return "clean"
        if lane < 2 * n_per: return "constant"
        return "dynamic"

    init_mode = None
    init_bg = None
    init_ang = None
    last_conf = np.zeros(n_envs)
    last_phys = np.zeros(n_envs)
    t0 = time.monotonic()

    for c in range(n_chunks):
        state, obs, traj = step_chunk(params, state, obs)
        t_np = {k: np.asarray(jax.device_get(v)) for k, v in traj.items()}
        if c == 0:
            np.testing.assert_allclose(t_np["pose_limit"], pose_deg, atol=1e-4)
            init_mode = t_np["wind_mode"][0].astype(int)
            init_bg = t_np["bg_speed"][0]
            init_ang = t_np["wind_angle"][0]
            assert np.all(init_mode[:n_per] == 0), "Clean lanes not mode 0"
            assert np.all(init_mode[n_per:2*n_per] == 1), "Constant lanes not mode 1"
            assert np.all(np.isin(init_mode[2*n_per:], [2, 3])), "Dynamic lanes not mode 2/3"

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
                    if prev_h >= 1 and flight_peak_z[lane] > 0.5:
                        ap = float(flight_peak_z[lane])
                        pb = bucket_of(prev_h)
                        if pb is not None:
                            bs = b_stats[rname][pb]
                            bs["n_apex"] += 1
                            bs["apex_sum"] += ap
                            bs["apex_err_sq_sum"] += (ap - 1.425) ** 2
                            if 1.40 <= ap <= 1.45:
                                bs["in_band"] += 1
                            if not np.isnan(prev_apex[lane]):
                                bs["jump_sum"] += abs(ap - float(prev_apex[lane]))
                                bs["n_jump"] += 1
                        prev_apex[lane] = ap
                    nb = bucket_of(new_h)
                    if nb is not None:
                        bs = b_stats[rname][nb]
                        cd = float(t_np["center_sum"][t, lane])
                        bs["n_hits"] += 1
                        bs["center_abs_sum"] += cd
                        bs["center_sq_sum"] += cd * cd
                        if t_np["racket_vxy_n"][t, lane] > 0.5:
                            bs["vxy_sq_sum"] += float(t_np["racket_vxy_sq"][t, lane])
                            bs["vxy_n"] += 1
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
                    if t_np["rz_fail"][t, lane] > 0.5: fail_cause[lane] = 1
                    elif t_np["rec_fail"][t, lane] > 0.5: fail_cause[lane] = 2
                    elif t_np["low_fail"][t, lane] > 0.5: fail_cause[lane] = 3
                    elif t_np["high_fail"][t, lane] > 0.5: fail_cause[lane] = 4
                    else: fail_cause[lane] = 5
                alive &= ~newly_dead

        last_conf = np.where(alive, t_np["conf_ratio"][-1], last_conf)
        last_phys = np.where(alive, t_np["phys_ratio"][-1], last_phys)
        if c + 1 == chunk_20s:
            alive_20s = alive.copy()
            strict_20s = alive_20s & (max_hits >= 24) & (t_np["conf_ratio"][-1] <= 1.0 + 1e-6) & (t_np["phys_ratio"][-1] <= 1.0 + 1e-6)

    wall = time.monotonic() - t0
    strict_60s = alive & (max_hits >= int(math.ceil(1.2 * seconds))) & (last_conf <= 1.0 + 1e-6) & (last_phys <= 1.0 + 1e-6)
    print(f"\n================ [{label} | pose={pose_deg}deg] TOTAL WALL={wall:.1f}s ({n_envs} envs x {seconds}s) ================", flush=True)

    out_list = []
    for rname, lo, hi in REGIMES:
        sl = slice(lo, hi)
        fc = {
            "rz": int(np.sum(fail_cause[sl] == 1)),
            "rec": int(np.sum(fail_cause[sl] == 2)),
            "low": int(np.sum(fail_cause[sl] == 3)),
            "high": int(np.sum(fail_cause[sl] == 4)),
            "other": int(np.sum(fail_cause[sl] == 5)),
        }
        subgroups = {}
        if rname == "constant":
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
        elif rname == "dynamic":
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

        milestones = [5, 10, 15, 20, 30, 50, 75, 100, 110]
        hit_curve = {k: round(float(np.mean(max_hits[sl] >= k)) * 100, 2) for k in milestones}
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
            "center_rms_mm": round(math.sqrt(sum(b_stats[rname][b]["center_sq_sum"] for b in BUCKETS[1:]) / st_nh) * 1000, 2) if st_nh else None,
            "apex_mean_m": round(sum(b_stats[rname][b]["apex_sum"] for b in BUCKETS[1:]) / st_na, 4) if st_na else None,
            "apex_err_rms_mm": round(math.sqrt(sum(b_stats[rname][b]["apex_err_sq_sum"] for b in BUCKETS[1:]) / st_na) * 1000, 2) if st_na else None,
            "in_band_pct": round(sum(b_stats[rname][b]["in_band"] for b in BUCKETS[1:]) / st_na * 100, 2) if st_na else None,
            "adj_jump_mm": round(sum(b_stats[rname][b]["jump_sum"] for b in BUCKETS[1:]) / st_nj * 1000, 2) if st_nj else None,
            "racket_vxy_rms": round(math.sqrt(sum(b_stats[rname][b]["vxy_sq_sum"] for b in BUCKETS[1:]) / st_nv), 4) if st_nv else None,
        }
        res = {
            "model": label, "regime": rname, "pose_deg": pose_deg, "n_envs": n_per, "seconds": seconds,
            "phys_20s_pct": round(float(alive_20s[sl].mean()) * 100, 2),
            "strict_20s_pct": round(float(strict_20s[sl].mean()) * 100, 2),
            "phys_60s_pct": round(float(alive[sl].mean()) * 100, 2),
            "strict_60s_pct": round(float(strict_60s[sl].mean()) * 100, 2),
            "mean_duration_s": round(float(fail_sec[sl].mean()), 2),
            "mean_hits": round(float(max_hits[sl].mean()), 1),
            "fail_causes": fc, "subgroups": subgroups,
            "hit_survival_curve": hit_curve, "steady_6_115": steady, "hit_buckets": bucket_report,
        }
        out_list.append(res)
        print(f"\n>>> [{label} | {rname}] 20s phys={res['phys_20s_pct']:.2f}% strict={res['strict_20s_pct']:.2f}% | "
              f"60s phys={res['phys_60s_pct']:.2f}% strict={res['strict_60s_pct']:.2f}% | "
              f"mean_dur={res['mean_duration_s']}s mean_hits={res['mean_hits']} | fails={fc}", flush=True)
        print(f"    hit_surv: {hit_curve}", flush=True)
        print(f"    steady(6-115): {steady}", flush=True)
        if subgroups:
            print(f"    subgroups: {subgroups}", flush=True)
        for bk, bv in bucket_report.items():
            print(f"    bucket[{bk:6s}]: hits={bv['n_hits']:5d} fails={bv['fails']:3d} "
                  f"center_rms={bv['center_rms_mm']}mm apex_err={bv['apex_err_rms_mm']}mm "
                  f"in_band={bv['in_band_pct']}% jump={bv['adj_jump_mm']}mm vxy={bv['racket_vxy_rms']}", flush=True)
    return out_list


def main():
    rl_sim = Path.home() / "pingpong/pingpong_moz1/pingpong_controller/outputs/rl_sim"
    u4768_path = rl_sim / "fplc_wind_early_20261127_r3/best_pareto_u00004768.pkl"
    v321_path = rl_sim / "releases/fplc_v321_s4_u2097_20261029_r1/policy.pkl"
    out_json = rl_sim / "fplc_wind_early_20261127_r3/benchmark_u4768_vs_v321.json"

    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
    install_pose_mode()

    with u4768_path.open("rb") as f:
        u4768_params = jax.device_put(pickle.load(f)["params"])
    with v321_path.open("rb") as f:
        v321_params = jax.device_put(pickle.load(f)["params"])

    results = []
    results.extend(run_model_unified("V325_u4768_det", u4768_params, stack.ppo.policy_mean, stack, pose_deg=5.0))
    results.extend(run_model_unified("V321_u2097_det", v321_params, lambda p, o: apply_v321_pi(p["pi"], o), stack, pose_deg=2.0))

    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved full JSON results to {out_json}", flush=True)


if __name__ == "__main__":
    main()
