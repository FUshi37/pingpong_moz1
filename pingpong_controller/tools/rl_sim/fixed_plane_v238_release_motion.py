"""V238 couples bounded weak-third-launch cost with stronger ball-XY cost."""

from dataclasses import replace
from types import SimpleNamespace

from fixed_plane_v237_release_margin import (
    validate_training_contract as previous_training,
    validate_resume_payload as previous_resume,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v238'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v236'
START_STAGE = 'fplc_v238_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v238_s01_workspace_alignment'


def build_stages(source):
    """Retain the V237 candidate; double the existing normalized XY loss."""
    if len(source) != 17 or not source[7].name.startswith('fplc_v237_s08_'):
        raise ValueError('V238 requires the complete release-margin candidate')
    return [replace(stage,
        name=stage.name.replace('fplc_v237_','fplc_v238_',1),
        cfg=replace(stage.cfg,fixed_plane_v166_physical_ball_xy_weight=4.),
        notes='V238 retains V236 phased centering and adds the V237 bounded '
              'non-nominal third-release shortfall cost30. Existing physical '
              'outgoing ball-XY squared cost2 becomes4 in both cohorts. '
              'All17 domains and absolute gates,72/393/7, control/physics/DR '
              'and1024x128 PPO stay unchanged. No update cap or sample filter. '
              'Continue the latest full V236 state, not a selected old actor.')
            for stage in source]


def validate_training_contract(settings):
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V238 requires the unchanged nominal absolute domain')
    previous_training(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v237_s01_workspace_alignment'}))


def validate_resume_payload(payload, *, requested_start_stage, **kwargs):
    """Validate through a metadata-only proxy; never alter arrays/history."""
    proxy=dict(payload)
    proxy['args']=dict(payload.get('args',{}))
    if proxy['args'].get('curriculum_profile')==PROFILE:
        proxy['args']['curriculum_profile']='fixed_plane_limit_cycle_qvel_rmp_v237'
        proxy['stage_name']=str(proxy.get('stage_name','')).replace('fplc_v238_','fplc_v237_',1)
    previous_resume(proxy,requested_start_stage=requested_start_stage.replace(
        'fplc_v238_','fplc_v237_',1),**kwargs)
