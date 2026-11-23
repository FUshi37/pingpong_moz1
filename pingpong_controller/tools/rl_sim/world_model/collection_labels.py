"""CPU labels separating task bounds from usable teacher dynamics samples."""

from __future__ import annotations

import numpy as np


def label_task_bounds(cfg: dict, states: dict, *, reference_rotation,
                      joint_lower_rad, joint_upper_rad) -> dict:
    """Label saved endpoints without rejecting physically valid task failures.

    This does not replace the 1 ms other-body collision/finite masks. There
    are no mesh or IK queries, and no claimed dynamic self-collision audit.
    The first-state eligibility flag is only one part of full-seed admission.
    """
    domain = cfg['domain']
    q = np.asarray(states['q_rad'])
    ball = np.asarray(states['ball_qpos'])[..., :3]
    racket = np.asarray(states['racket_position_m'])
    rotation = np.asarray(states['racket_rotation_world'])
    relative = np.einsum('ji,...jk->...ik', reference_rotation, rotation)
    roll = np.arctan2(-relative[..., 1, 2], relative[..., 2, 2])
    pitch = np.arcsin(np.clip(relative[..., 0, 2], -1., 1.))
    rp = np.stack((roll, pitch), -1)
    finite = (np.isfinite(q).all(-1) & np.isfinite(ball).all(-1)
              & np.isfinite(racket).all(-1) & np.isfinite(rotation).all((-1, -2)))
    joint_ok = ((q >= np.asarray(joint_lower_rad)) & (q <= np.asarray(joint_upper_rad))).all(-1)
    racket_xyz_ok = ((racket >= domain['racket_xyz_lower_m']) & (racket <= domain['racket_xyz_upper_m'])).all(-1)
    racket_rp_ok = (np.abs(rp) <= np.deg2rad(domain['roll_pitch_limit_deg'])).all(-1)
    ball_xy_ok = ((ball[..., :2] >= domain['view_xy_lower_m']) & (ball[..., :2] <= domain['view_xy_upper_m'])).all(-1)
    ball_z_ok = ((ball[..., 2] >= domain['ball_z_physical_bounds_m'][0])
                 & (ball[..., 2] <= domain['ball_z_physical_bounds_m'][1]))
    within = finite & joint_ok & racket_xyz_ok & racket_rp_ok & ball_xy_ok & ball_z_ok
    first = np.where(np.any(~within, axis=0), np.argmax(~within, axis=0), -1)
    return dict(physics_invalid=~finite, joint_position_valid=joint_ok,
        racket_xyz_valid=racket_xyz_ok, racket_roll_pitch_valid=racket_rp_ok,
        ball_xy_valid=ball_xy_ok, ball_z_valid=ball_z_ok, relative_roll_pitch_rad=rp,
        within_task_bounds=within, initial_task_eligible=within[0],
        first_task_bound_failure_control=first)
