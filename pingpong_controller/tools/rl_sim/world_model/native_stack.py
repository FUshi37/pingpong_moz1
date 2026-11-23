"""Private WMJ adapters over the authenticated V324 simulator, never shared edits."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import SimpleNamespace

from .contracts import repository_root, validate_config


def load_native_stack(cfg: dict):
    """Authenticate and load V324's frozen source in a fresh worker process."""
    validate_config(cfg)
    from train_fixed_plane_ball_obs_noise import load_stack
    root = repository_root()
    return load_stack(root, root / 'pingpong_controller/tools/rl_sim', 324)


def build_native_environment(cfg: dict, *, n_envs: int, stage_index: int = 0, stack=None):
    """Construct an opt-in teacher with the explicitly confirmed domain changes.

    Native rewards and done flags remain diagnostics. The WMJ task adapter
    owns task reward, task targets and 30-second truncation; it must not feed
    the old fixed-target reward to any WMJ learner.
    """
    if not 0 <= stage_index < len(cfg['curriculum']['stages']):
        raise ValueError('Unknown WMJ-V1 stage')
    stack = load_native_stack(cfg) if stack is None else stack
    if stack.spec.version != 324:
        raise ValueError('Only the authenticated V324 native stack may be reused')
    domain = cfg['domain']
    stage = cfg['curriculum']['stages'][stage_index]
    overrides = dict(
        horizon_sec=stage['training_episode_s'],
        cartesian_pose_position_lower_m=tuple(domain['racket_xyz_lower_m']),
        cartesian_pose_position_upper_m=tuple(domain['racket_xyz_upper_m']),
        ball_view_x_bounds_m=(domain['view_xy_lower_m'][0], domain['view_xy_upper_m'][0]),
        ball_view_y_bounds_m=(domain['view_xy_lower_m'][1], domain['view_xy_upper_m'][1]),
        ball_position_noise_ar1_enabled=stage['ar1_enabled'],
    )
    teacher_cfg = replace(stack.cfg, **overrides)
    changed = {k for k, value in asdict(teacher_cfg).items() if value != asdict(stack.cfg)[k]}
    if not changed.issubset(overrides):
        raise ValueError('Unregistered native environment override')
    environment = stack.env_class(stack.xml, n_envs=n_envs, cfg=teacher_cfg)
    return SimpleNamespace(environment=environment, stack=stack,
                           overrides=overrides, native_config=teacher_cfg)
