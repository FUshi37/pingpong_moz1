"""Comprehensive 576-env x 60s Closed-Loop Diagnostic & +X Failure Autopsy for `s2r_r6_fast` u7832.

Evaluates `best_pareto_w28_u00007832.pkl` (pose=+/-15 deg, obs_wind_max=4.5 m/s, 4.0 g ball) across
18 regimes x 32 deterministic 60s episodes (= 576 episodes) in ONE single JIT compile:
  - Group A (Real-Robot Zero-Wind Parity): RealExact_0w, RealHard_0w, RealSoft_0w, P_cold_0w
  - Group B (Real-Fan +X & Dyn Mirror on RealExact S2R): PosX 1.6/2.4/2.8, NegX 2.8, Dyn 2.4/2.8
  - Group C (Clean P_warm Directional & Speed Sweep): PosX/NegX 2.4/2.8/3.2 + +/-X Cone 2.8
Includes full termination-cause breakdown, racket workspace/tilt excursions, wRMSE, and per-episode
failure autopsy to pinpoint why +X lags -X at 2.8 m/s.
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

SCRIPT_DIR = Path(sys.argv[1])
CKPT_PATH = Path(sys.argv[2])
OUT_JSON = Path(sys.argv[3])
sys.path.insert(0, str(SCRIPT_DIR))

from migrated_stack import load_migrated_v325_stack, setup_sys_path
setup_sys_path()

from capture_margin import CaptureEnv
from high_wind_r6 import (
    HighWindFeatureEnv,
    HighWindSingle5Env,
    encode_r6,
    install_high_wind_pose_mode,
    observed_force_features_r6,
)
from sim2real_r5 import P_COLD_3X3, P_WARM_3X3
from sim2real_r6 import configure_s2r_r6_cfg
from train_s15_fresh_velocity_r31 import ActualContactEnv


def main():
    obs_max = 4.5
    pose_deg = 15.0
    seconds = 60.0
    per_scen = 32

    scen_names = [
        "RealExact_0w",
        "RealHard_0w",
        "RealSoft_0w",
        "P_cold_0w",
        "RealExact_PosX_1p6",
        "RealExact_PosX_2p4",
        "RealExact_PosX_2p8",
        "RealExact_NegX_2p8",
        "RealExact_W24_Dyn",
        "RealExact_W28_Dyn",
        "CleanPW_PosX_2p4",
        "CleanPW_NegX_2p4",
        "CleanPW_PosX_2p8",
        "CleanPW_NegX_2p8",
        "CleanPW_ConePosX_2p8",
        "CleanPW_ConeNegX_2p8",
        "CleanPW_PosX_3p2",
        "CleanPW_NegX_3p2",
    ]
    n_scen = len(scen_names)
    n_envs = n_scen * per_scen  # 576 envs

    with jax.default_device(jax.devices("cpu")[0]):
        stack = load_migrated_v325_stack(verify_source_manifest=True)
    s2r_cfg = dict(ball_spawn_xy_jitter_m=0.0, dr_rmp_target_update_hold_probability=0.0)
    cfg = configure_s2r_r6_cfg(
        stack.cfg, pose_deg=pose_deg, max_speed_m_s=4.0, force_limit_n=0.025, s2r_cfg=s2r_cfg
    )
    cfg = replace(
        cfg,
        ball_spawn_xy_jitter=0.0,
        dr_randomize_recovered_rmp=False,
        dr_rmp_target_update_hold_probability=0.0,
        wind_config=replace(
            cfg.wind_config,
            max_speed_m_s=4.0,
            mode_weights=(0.0, 0.0, 0.5, 0.5),
            wind_y_nonpositive=True,
            startup_no_wind_hits=5,
            startup_ramp_s=1.0,
        ),
    )
    install_high_wind_pose_mode(pose_deg)

    native = stack.env_class(stack.xml, n_envs=n_envs, cfg=cfg)
    env = HighWindSingle5Env(
        CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),
        obs_wind_max_m_s=obs_max,
        reward_enabled=False,
        goal_reward_enabled=False,
        random_onset_enabled=False,
        targeted_reward_enabled=False,
    )
    env = HighWindFeatureEnv(env, obs_wind_max_m_s=obs_max)

    pair_id = int(native.ball_racket_pair_ids[0]) if hasattr(native, "ball_racket_pair_ids") and len(native.ball_racket_pair_ids) else 0
    ball_qadr = int(native.mj_model.jnt_qposadr[native.ball_joint_id])
    racket_site_id = int(native.racket_site_id)
    nom_kp = np.asarray(jax.device_get(native.recovered_rmp_jnt_kp), dtype=np.float32)
    nom_kd = np.asarray(jax.device_get(native.recovered_rmp_jnt_kd), dtype=np.float32)
    NOM_DAMP = 0.228086

    # Load checkpoint and verify contract
    with CKPT_PATH.open("rb") as f:
        saved = pickle.load(f)
    assert abs(float(saved.get("obs_wind_max_m_s", 4.5)) - obs_max) < 1e-6
    assert abs(float(saved.get("pose_limit_deg", 15.0)) - pose_deg) < 1e-6
    params = jax.tree.map(jnp.asarray, saved["params"])
    print(
        f"[EVAL] Loaded {CKPT_PATH.name} (u{saved['actual_updates']}, pose=+/-{saved.get('pose_limit_deg')}deg, "
        f"obs_max={saved.get('obs_wind_max_m_s')}m/s, active_w={saved.get('active_max_speed_m_s')}m/s) | "
        f"{n_scen} regimes x {per_scen} = {n_envs} envs x {seconds:.0f}s",
        flush=True,
    )

    rng = np.random.default_rng(2026112888)
    # Default: Clean P_warm physics
    cov_arr = np.broadcast_to(P_WARM_3X3[None, None, :, :], (n_envs, 7, 3, 3)).copy()
    grav_arr = np.tile(np.array([0.0, 0.0, -9.81], dtype=np.float32), (n_envs, 1))
    damp_arr = np.full((n_envs,), NOM_DAMP, dtype=np.float32)
    lag_arr = np.zeros((n_envs,), dtype=np.int32)
    bias_arr = np.zeros((n_envs, 3), dtype=np.float32)
    nstd_arr = np.zeros((n_envs, 3), dtype=np.float32)
    sdx_arr = np.zeros((n_envs, 2), dtype=np.float32)
    kp_arr = np.tile(nom_kp[None, :], (n_envs, 1))
    kd_arr = np.tile(nom_kd[None, :], (n_envs, 1))
    wmode_arr = np.zeros((n_envs,), dtype=np.int32)
    wbg_arr = np.zeros((n_envs, 3), dtype=np.float32)
    wou_arr = np.zeros((n_envs, 3), dtype=np.float32)
    wg_arr = np.zeros((n_envs, 3), dtype=np.float32)

    ang_spawn = np.linspace(0, 2 * np.pi, per_scen, endpoint=False)

    def set_real_exact(si: int):
        s = slice(si * per_scen, (si + 1) * per_scen)
        grav_arr[s] = [0.0, 0.135, -9.67]
        damp_arr[s] = 0.285
        lag_arr[s] = np.where(np.arange(per_scen) % 2 == 0, 2, 3).astype(np.int32)
        bias_arr[s] = [0.0, 0.0, -0.012]
        nstd_arr[s] = [0.0006, 0.0006, 0.0008]
        sdx_arr[s, 0] = 0.006 * np.cos(ang_spawn)
        sdx_arr[s, 1] = 0.006 * np.sin(ang_spawn)

    def set_const_dir(si: int, spd: float, ang_deg: float):
        s = slice(si * per_scen, (si + 1) * per_scen)
        rad = math.radians(ang_deg)
        wmode_arr[s] = 1
        wbg_arr[s, 0] = spd * math.cos(rad)
        wbg_arr[s, 1] = min(spd * math.sin(rad), 0.0)

    def set_const_cone(si: int, spd: float, lo_deg: float, hi_deg: float):
        s = slice(si * per_scen, (si + 1) * per_scen)
        angs = np.radians(np.linspace(lo_deg, hi_deg, per_scen))
        wmode_arr[s] = 1
        wbg_arr[s, 0] = spd * np.cos(angs)
        wbg_arr[s, 1] = np.minimum(spd * np.sin(angs), 0.0)

    def set_dyn(si: int, spd: float):
        s = slice(si * per_scen, (si + 1) * per_scen)
        half = per_scen // 2
        wmode_arr[s] = np.array([2] * half + [3] * half, dtype=np.int32)
        angs = np.linspace(-np.pi, 0.0, half)
        for k in range(half):
            dx, dy = math.cos(angs[k]), min(math.sin(angs[k]), 0.0)
            wbg_arr[si * per_scen + k] = [0.625 * spd * dx, 0.625 * spd * dy, 0.0]
            wou_arr[si * per_scen + k] = [0.375 * spd * dx, 0.375 * spd * dy, 0.0]
            wbg_arr[si * per_scen + half + k] = [0.15 * spd * dx, 0.15 * spd * dy, 0.0]
            wg_arr[si * per_scen + half + k] = [0.85 * spd * dx, 0.85 * spd * dy, 0.0]

    # 0: RealExact_0w
    set_real_exact(0)
    # 1: RealHard_0w
    set_real_exact(1)
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
    # 2: RealSoft_0w
    set_real_exact(2)
    damp_arr[slice(2 * per_scen, 3 * per_scen)] = 0.33
    # 3: P_cold_0w
    cov_arr[slice(3 * per_scen, 4 * per_scen)] = P_COLD_3X3[None, None, :, :]
    # 4..9: RealExact + wind
    for si in (4, 5, 6, 7, 8, 9):
        set_real_exact(si)
    set_const_dir(4, 1.6, 0.0)
    set_const_dir(5, 2.4, 0.0)
    set_const_dir(6, 2.8, 0.0)
    set_const_dir(7, 2.8, -180.0)
    set_dyn(8, 2.4)
    set_dyn(9, 2.8)
    # 10..17: Clean P_warm directional & speed sweep (still apply +/-6mm spawn circle for fair diversity)
    for si in range(10, 18):
        s = slice(si * per_scen, (si + 1) * per_scen)
        sdx_arr[s, 0] = 0.004 * np.cos(ang_spawn)
        sdx_arr[s, 1] = 0.004 * np.sin(ang_spawn)
    set_const_dir(10, 2.4, 0.0)
    set_const_dir(11, 2.4, -180.0)
    set_const_dir(12, 2.8, 0.0)
    set_const_dir(13, 2.8, -180.0)
    set_const_cone(14, 2.8, -45.0, 0.0)
    set_const_cone(15, 2.8, -180.0, -135.0)
    set_const_dir(16, 3.2, 0.0)
    set_const_dir(17, 3.2, -180.0)

    cov_j, grav_j, damp_j = jnp.asarray(cov_arr), jnp.asarray(grav_arr), jnp.asarray(damp_arr)
    lag_j, bias_j, nstd_j = jnp.asarray(lag_arr), jnp.asarray(bias_arr), jnp.asarray(nstd_arr)
    sdx_j, kp_j, kd_j = jnp.asarray(sdx_arr), jnp.asarray(kp_arr), jnp.asarray(kd_arr)
    wmode_j, wbg_j, wou_j, wg_j = jnp.asarray(wmode_arr), jnp.asarray(wbg_arr), jnp.asarray(wou_arr), jnp.asarray(wg_arr)

    def corrupt(co, rng_k):
        sp = jax.vmap(lambda k: jax.random.split(k, 2))(rng_k)
        nxt, dk = sp[:, 0], sp[:, 1]
        eps = jax.vmap(lambda k: jax.random.normal(k, (16, 3), dtype=jnp.float32))(dk)
        p28 = co[:, 28:52].reshape((-1, 8, 3)) + bias_j[:, None, :] + eps[:, :8, :] * nstd_j[:, None, :]
        v8 = co[:, 104:112] > 0.5
        eh = (eps[:, 8:, :] * (nstd_j[:, None, :] * math.sqrt(2.0))).at[:, 0, :].set(0.0)
        h72 = jnp.where(v8[:, :, None], co[:, 72:96].reshape((-1, 8, 3)) + eh, co[:, 72:96].reshape((-1, 8, 3)))
        return p28.reshape((-1, 24)), jnp.concatenate([h72.reshape((-1, 24)), co[:, 96:112]], axis=-1), nxt

    def assemble(co, r28, r72):
        idx = jnp.arange(n_envs)
        slot = 4 - lag_j
        obs112 = co[:, :112].at[:, 28:52].set(r28[idx, slot]).at[:, 72:112].set(r72[idx, slot])
        obs118 = jnp.concatenate([obs112, co[:, 112:114], observed_force_features_r6(obs112, max_wind_m_s=obs_max, xp=jnp)], axis=-1)
        return encode_r6(obs118, max_wind_m_s=obs_max, xp=jnp)

    @jax.jit
    def reset_all():
        keys = jax.random.split(jax.random.PRNGKey(2026112889), n_envs)
        st, clean_obs = env.reset(keys)
        new_qpos = st.data.qpos.at[:, ball_qadr : ball_qadr + 2].add(sdx_j)
        new_model = st.model.replace(opt=st.model.opt.replace(gravity=grav_j))
        if hasattr(new_model, "pair_solref") and new_model.pair_solref.ndim == 3:
            new_model = new_model.replace(pair_solref=new_model.pair_solref.at[:, pair_id, 1].set(damp_j))
        st = st._replace(
            data=st.data.replace(qpos=new_qpos),
            model=new_model,
            rmp_state=st.rmp_state._replace(target_covariance=cov_j),
            rmp_jnt_kp=kp_j,
            rmp_jnt_kd=kd_j,
            wind_mode=wmode_j,
            wind_background_m_s=wbg_j,
            wind_ou_m_s=wou_j,
            wind_gust_vector_m_s=wg_j,
        )
        nrng = jax.vmap(lambda k: jax.random.fold_in(k, 9999))(keys)
        n28, n72, nrng = corrupt(clean_obs, nrng)
        r28 = jnp.repeat(n28[:, None, :], 5, axis=1)
        r72 = jnp.repeat(n72[:, None, :], 5, axis=1)
        ob = assemble(clean_obs, r28, r72)
        return st, ob, r28, r72, nrng

    chunk = 128

    @jax.jit
    def step_chunk(p, st, ob, r28, r72, nrng):
        def body(carry, _):
            s, o, b28, b72, rk = carry
            a = stack.ppo.policy_mean(p, o)
            s2, co2, rew, done, m = env.step(s, a)
            n28, n72, rk2 = corrupt(co2, rk)
            b28_2 = jnp.concatenate([b28[:, 1:, :], n28[:, None, :]], axis=1)
            b72_2 = jnp.concatenate([b72[:, 1:, :], n72[:, None, :]], axis=1)
            o2 = assemble(co2, b28_2, b72_2)
            nrm = jnp.cross(co2[:, 55:58], co2[:, 58:61])
            roll_deg = jnp.degrees(jnp.arctan2(nrm[:, 1], jnp.maximum(nrm[:, 2], 1e-6)))
            pitch_deg = jnp.degrees(jnp.arctan2(-nrm[:, 0], jnp.maximum(nrm[:, 2], 1e-6)))
            z0 = jnp.zeros((n_envs,), dtype=jnp.float32)
            out = {
                "term": m["terminated"],
                "hits": s2.wind_startup_hit_count,
                "hit_evt": m["hit_event_count"],
                "center_sum": m["hit_contact_center_dist_sum"],
                "bpos": s2.data.qpos[:, ball_qadr : ball_qadr + 3],
                "rpos": s2.data.site_xpos[:, racket_site_id],
                "roll": roll_deg,
                "pitch": pitch_deg,
                "w_err_sq": m.get("observed_force/error_sq_m2_s2", z0),
                "est_w": o2[:, 114:116] * obs_max,
                "act_abs_max": jnp.max(jnp.abs(a), axis=-1),
                "rz_fail": m.get("done/racket_z_out_of_bounds", z0),
                "rec_fail": m.get("done/pending_hit_recontact", z0),
                "low_fail": m.get("done/ball_too_low", z0),
                "high_fail": m.get("done/ball_too_high", z0),
                "conf_ratio": m.get("timing/confirmed_max_ratio", z0),
                "phys_ratio": m.get("timing/physical_max_ratio", z0),
            }
            return (s2, o2, b28_2, b72_2, rk2), out
        (st_e, ob_e, r28_e, r72_e, nrng_e), traj = jax.lax.scan(body, (st, ob, r28, r72, nrng), None, length=chunk)
        return st_e, ob_e, r28_e, r72_e, nrng_e, traj

    n_chunks = int(math.ceil((seconds / env.dt) / chunk))
    chunk_20s = int(math.ceil((20.0 / env.dt) / chunk))

    t0 = time.monotonic()
    st, ob, r28, r72, nrng = reset_all()
    alive = np.ones(n_envs, dtype=bool)
    alive_20s = np.ones(n_envs, dtype=bool)
    fail_sec = np.full(n_envs, seconds, dtype=np.float64)
    fail_cause = np.zeros(n_envs, dtype=np.int32)
    fail_detail = [None] * n_envs
    max_hits = np.zeros(n_envs, dtype=np.int32)
    last_conf = np.zeros(n_envs, dtype=np.float64)
    last_phys = np.zeros(n_envs, dtype=np.float64)

    # Per-lane running stats (updated only while alive)
    hit_c_sq = np.zeros(n_envs, dtype=np.float64)
    hit_n = np.zeros(n_envs, dtype=np.int32)
    w_err_sum = np.zeros(n_envs, dtype=np.float64)
    step_n = np.zeros(n_envs, dtype=np.int32)
    max_tilt = np.zeros(n_envs, dtype=np.float64)
    max_abs_roll = np.zeros(n_envs, dtype=np.float64)
    max_abs_pitch = np.zeros(n_envs, dtype=np.float64)
    min_rx = np.full(n_envs, 9.0)
    max_rx = np.full(n_envs, -9.0)
    min_ry = np.full(n_envs, 9.0)
    max_ry = np.full(n_envs, -9.0)
    min_rz = np.full(n_envs, 9.0)
    max_rz = np.full(n_envs, -9.0)
    max_act = np.zeros(n_envs, dtype=np.float64)
    flight_peak_z = np.zeros(n_envs, dtype=np.float64)
    apex_sum = np.zeros(n_envs, dtype=np.float64)
    apex_n = np.zeros(n_envs, dtype=np.int32)
    apex_inband = np.zeros(n_envs, dtype=np.int32)
    cur_h = np.zeros(n_envs, dtype=np.int32)
    # Sampled tilt/pitch distributions after t=4s (post-onset)
    tilt_samples = [[] for _ in range(n_scen)]
    pitch_samples = [[] for _ in range(n_scen)]

    for c in range(n_chunks):
        st, ob, r28, r72, nrng, traj = step_chunk(params, st, ob, r28, r72, nrng)
        t_np = jax.device_get(traj)
        if c == 0:
            print(f"[EVAL] Compiled + chunk 0 finished in {time.monotonic() - t0:.1f}s", flush=True)

        term = t_np["term"] > 0.5
        hit_evt = t_np["hit_evt"] > 0.5
        bpos = t_np["bpos"]
        rpos = t_np["rpos"]
        roll = t_np["roll"]
        pitch = t_np["pitch"]
        tilt = np.hypot(roll, pitch)

        for k in range(chunk):
            sec_now = (c * chunk + k + 1) * env.dt
            al = alive
            if not np.any(al):
                break
            flight_peak_z = np.where(al, np.maximum(flight_peak_z, bpos[k, :, 2]), flight_peak_z)
            w_err_sum += np.where(al, t_np["w_err_sq"][k], 0.0)
            step_n += al.astype(np.int32)
            max_tilt = np.where(al, np.maximum(max_tilt, tilt[k]), max_tilt)
            max_abs_roll = np.where(al, np.maximum(max_abs_roll, np.abs(roll[k])), max_abs_roll)
            max_abs_pitch = np.where(al, np.maximum(max_abs_pitch, np.abs(pitch[k])), max_abs_pitch)
            min_rx = np.where(al, np.minimum(min_rx, rpos[k, :, 0]), min_rx)
            max_rx = np.where(al, np.maximum(max_rx, rpos[k, :, 0]), max_rx)
            min_ry = np.where(al, np.minimum(min_ry, rpos[k, :, 1]), min_ry)
            max_ry = np.where(al, np.maximum(max_ry, rpos[k, :, 1]), max_ry)
            min_rz = np.where(al, np.minimum(min_rz, rpos[k, :, 2]), min_rz)
            max_rz = np.where(al, np.maximum(max_rz, rpos[k, :, 2]), max_rz)
            max_act = np.where(al, np.maximum(max_act, t_np["act_abs_max"][k]), max_act)

            if sec_now >= 4.0 and (k % 16 == 0):
                for si in range(n_scen):
                    sl = slice(si * per_scen, (si + 1) * per_scen)
                    m_al = al[sl]
                    if np.any(m_al):
                        tilt_samples[si].append(tilt[k, sl][m_al])
                        pitch_samples[si].append(pitch[k, sl][m_al])

            h_m = al & hit_evt[k]
            if np.any(h_m):
                for lane in np.flatnonzero(h_m):
                    cd = float(t_np["center_sum"][k, lane])
                    hit_c_sq[lane] += cd * cd
                    hit_n[lane] += 1
                    nh = int(t_np["hits"][k, lane])
                    if nh <= cur_h[lane]:
                        nh = cur_h[lane] + 1
                    if cur_h[lane] >= 1 and flight_peak_z[lane] > 1.05:
                        ap = float(flight_peak_z[lane])
                        apex_sum[lane] += ap
                        apex_n[lane] += 1
                        if 1.40 <= ap <= 1.45:
                            apex_inband[lane] += 1
                    cur_h[lane] = nh
                    if nh > max_hits[lane]:
                        max_hits[lane] = nh
                    flight_peak_z[lane] = float(bpos[k, lane, 2])

            nd = al & term[k]
            if np.any(nd):
                for lane in np.flatnonzero(nd):
                    fail_sec[lane] = sec_now
                    if t_np["rz_fail"][k, lane] > 0.5:
                        c_id, c_str = 1, "racket_z"
                    elif t_np["rec_fail"][k, lane] > 0.5:
                        c_id, c_str = 2, "recontact"
                    elif t_np["low_fail"][k, lane] > 0.5:
                        c_id, c_str = 3, "ball_low"
                    elif t_np["high_fail"][k, lane] > 0.5:
                        c_id, c_str = 4, "ball_high"
                    else:
                        c_id, c_str = 5, "other(xy_oob)"
                    fail_cause[lane] = c_id
                    fail_detail[lane] = dict(
                        lane=int(lane),
                        scen=scen_names[int(lane) // per_scen],
                        sec=round(sec_now, 2),
                        hits=int(max_hits[lane]),
                        cause=c_str,
                        bpos=[round(float(x), 3) for x in bpos[k, lane]],
                        rpos=[round(float(x), 3) for x in rpos[k, lane]],
                        roll=round(float(roll[k, lane]), 2),
                        pitch=round(float(pitch[k, lane]), 2),
                        est_w=[round(float(x), 2) for x in t_np["est_w"][k, lane]],
                        true_w=[round(float(x), 2) for x in wbg_arr[lane, :2]],
                        rx_rng=[round(float(min_rx[lane]), 3), round(float(max_rx[lane]), 3)],
                        ry_rng=[round(float(min_ry[lane]), 3), round(float(max_ry[lane]), 3)],
                        rz_rng=[round(float(min_rz[lane]), 3), round(float(max_rz[lane]), 3)],
                        max_tilt=round(float(max_tilt[lane]), 2),
                    )
                alive &= ~nd

        last_conf = np.where(alive, t_np["conf_ratio"][-1], last_conf)
        last_phys = np.where(alive, t_np["phys_ratio"][-1], last_phys)
        if c + 1 == chunk_20s:
            alive_20s = alive.copy()

    wall = time.monotonic() - t0
    strict_60s = alive & (max_hits >= int(math.ceil(1.2 * seconds))) & (last_conf <= 1.0 + 1e-6) & (last_phys <= 1.0 + 1e-6)

    print(f"\n==================== u7832 60s CLOSED-LOOP DIAGNOSTIC (WALL={wall:.1f}s) ====================", flush=True)
    print(
        f"{'Regime':22s} | {'20s%':>5} {'60s%':>6} {'str60%':>6} {'dur':>5} {'hits':>5} | "
        f"{'cRMS':>5} {'apx':>5} {'inB%':>4} {'wRMS':>4} | {'tlt95':>5} {'pit50':>5} {'rX[cm]':>11} {'rY[cm]':>11} | fails(rz/rec/low/hi/oob)"
    )
    results = []
    for si, sname in enumerate(scen_names):
        sl = slice(si * per_scen, (si + 1) * per_scen)
        fc = {k: int(np.sum(fail_cause[sl] == v)) for k, v in [("rz", 1), ("rec", 2), ("low", 3), ("high", 4), ("oob", 5)]}
        c_rms = math.sqrt(hit_c_sq[sl].sum() / max(hit_n[sl].sum(), 1)) * 1000.0
        apx_m = apex_sum[sl].sum() / max(apex_n[sl].sum(), 1)
        inb = apex_inband[sl].sum() / max(apex_n[sl].sum(), 1) * 100.0
        w_rms = math.sqrt(w_err_sum[sl].sum() / max(step_n[sl].sum(), 1))
        t95 = float(np.percentile(np.concatenate(tilt_samples[si]), 95)) if tilt_samples[si] else 0.0
        p50 = float(np.median(np.concatenate(pitch_samples[si]))) if pitch_samples[si] else 0.0
        p95 = float(np.percentile(np.abs(np.concatenate(pitch_samples[si])), 95)) if pitch_samples[si] else 0.0
        rx_lo, rx_hi = float(min_rx[sl].min() * 100), float(max_rx[sl].max() * 100)
        ry_lo, ry_hi = float(min_ry[sl].min() * 100), float(max_ry[sl].max() * 100)
        row = dict(
            regime=sname,
            n_envs=per_scen,
            phys_20s=round(float(alive_20s[sl].mean() * 100), 2),
            phys_60s=round(float(alive[sl].mean() * 100), 2),
            strict_60s=round(float(strict_60s[sl].mean() * 100), 2),
            mean_dur_s=round(float(fail_sec[sl].mean()), 2),
            mean_hits=round(float(max_hits[sl].mean()), 1),
            center_rms_mm=round(c_rms, 1),
            apex_mean_m=round(apx_m, 3),
            in_band_pct=round(inb, 1),
            wrmse_m_s=round(w_rms, 3),
            tilt_p95_deg=round(t95, 2),
            tilt_max_deg=round(float(max_tilt[sl].max()), 2),
            pitch_p50_deg=round(p50, 2),
            abs_pitch_p95_deg=round(p95, 2),
            rx_cm=[round(rx_lo, 1), round(rx_hi, 1)],
            ry_cm=[round(ry_lo, 1), round(ry_hi, 1)],
            rz_cm=[round(float(min_rz[sl].min() * 100), 1), round(float(max_rz[sl].max() * 100), 1)],
            max_act=round(float(max_act[sl].max()), 3),
            fails=fc,
        )
        results.append(row)
        fstr = f"{fc['rz']}/{fc['rec']}/{fc['low']}/{fc['high']}/{fc['oob']}"
        print(
            f"{sname:22s} | {row['phys_20s']:5.1f} {row['phys_60s']:6.1f} {row['strict_60s']:6.1f} {row['mean_dur_s']:5.1f} {row['mean_hits']:5.1f} | "
            f"{c_rms:5.1f} {apx_m:5.3f} {inb:4.1f} {w_rms:4.2f} | {t95:5.1f} {p50:+5.1f} [{rx_lo:+4.0f},{rx_hi:+4.0f}] [{ry_lo:+4.0f},{ry_hi:+4.0f}] | {fstr}",
            flush=True,
        )

    # Print failure autopsies for key +X and RealExact regimes
    print("\n==================== FAILURE AUTOPSY (KEY REGIMES) ====================", flush=True)
    for target in ("RealExact_PosX_2p4", "RealExact_PosX_2p8", "CleanPW_PosX_2p8", "CleanPW_ConePosX_2p8", "CleanPW_NegX_2p8", "RealExact_W28_Dyn", "CleanPW_PosX_3p2"):
        fails = [d for d in fail_detail if d is not None and d["scen"] == target]
        if fails:
            print(f"\n--- {target} ({len(fails)}/{per_scen} failed) ---", flush=True)
            for fd in fails[:8]:
                print(
                    f"  lane={fd['lane']:3d} t={fd['sec']:5.2f}s h={fd['hits']:3d} cause={fd['cause']:12s} | "
                    f"ball={fd['bpos']} rack={fd['rpos']} | r/p=({fd['roll']:+.1f},{fd['pitch']:+.1f}) maxT={fd['max_tilt']:.1f}d | "
                    f"w_true={fd['true_w']} w_est={fd['est_w']} | rX={fd['rx_rng']} rY={fd['ry_rng']} rZ={fd['rz_rng']}",
                    flush=True,
                )

    OUT_JSON.write_text(json.dumps(dict(checkpoint=str(CKPT_PATH), wall_s=round(wall, 1), results=results, failures=[d for d in fail_detail if d is not None]), indent=2))
    print(f"\nSaved full diagnostic JSON to {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
