"""Opt-in pose-authority timing, independent of wind and reward startup."""
import math


def pose_gain(state, dt_s: float, seconds: float, *, xp):
    if seconds <= 0.:
        raise ValueError('Pose startup ramp must be positive')
    elapsed = state.step_count*dt_s-state.wind_startup_started_s
    phase = xp.clip(elapsed/seconds, 0., 1.)
    active = (state.wind_startup_hit_count >= 5) & (state.wind_startup_started_s >= 0.)
    return xp.where(active, xp.sin(.5*math.pi*phase)**2, 0.)


def degrees(state, dt_s: float, cfg, seconds: float, *, xp):
    from fixed_plane_wind_pose import scheduled_degrees, pose_clock_steps
    target = scheduled_degrees(pose_clock_steps(state,cfg),cfg,xp=xp)
    return 2.+pose_gain(state,dt_s,seconds,xp=xp)*(target-2.)
