"""V278: learn every minibatch and prioritize genuine startup survival."""
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v278_all_samples'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v277_second_reserve'
START_STAGE = 'fplc_v278_s11_near_y_080mm'
RETENTION_STAGE = 'fplc_v278_s01_workspace_alignment'
GATE_CHANGES = {'target_mean_hits':12.5, 'target_mean_len_frac':.99,
                'target_episode_truncation_rate':.98,'target_hit12_rate':.96}
REWARD_CHANGES = {'v276_second_prepare_weight_per_s':200.,
                  'v255_near_preparation_gain':7.,
                  'fixed_plane_v115_third_hit_bonus_weight':128.}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep every physical lesson and motion gate; permit small DR degradation."""
    if len(source)!=17 or source[10].name!='fplc_v277_s11_near_y_080mm':
        raise ValueError('V278 requires the complete17-stage V277 course')
    return [replace(stage, name=stage.name.replace('fplc_v277_','fplc_v278_',1),
        cfg=replace(stage.cfg,**REWARD_CHANGES) if i>=10 else stage.cfg,
        **GATE_CHANGES,
        notes=('V278 keeps1024x128 and all numerical PPO/BC/KL settings, but '
               'fixes the prohibited data filter: apply every minibatch without '
               'KL backtracking/rejection/rollback; retain KL telemetry and PPO clipping. '
               'S11+ second preparation100->200/s, third near gain3->7, '
               'genuine confirmed third-hit bonus7.5->128. Retain bounded '
               'second reserve credit. Near-full gates length.99/full.98/hit12.96/hits12.5; '
               'apex1.37--1.41 and ballXY.026/.030 and all safety/motion gates unchanged. '
               'All17 domains/control/physics/DR/reset and72/393/7 unchanged. '
               'Latest full state; no optimizer reset, update cap or forced advancement.'))
        for i,stage in enumerate(source)]


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v277_second_reserve import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V278 requires unchanged nominal retention')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v277_s01_workspace_alignment'}))


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V278 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 11
                or payload.get('stage_name') != 'fplc_v277_s11_near_y_080mm'
                or payload.get('stage_update') != 104 or payload.get('step') != 3100246016
                or int(payload['opt'].t) != 766366):
            raise ValueError('V278 pins the full V277 S11u104 and fresh reward history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V278 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V278 rejects another source or from-scratch launch')
