"""Modest survival tolerance for DR, with unchanged motion-quality gates."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v280_stable_tolerance'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v279_energy_progress'
START_STAGE = 'fplc_v280_s11_near_y_080mm'
RETENTION_STAGE = 'fplc_v280_s01_workspace_alignment'
GATE_CHANGES = dict(target_mean_len_frac=.97,
                    target_episode_truncation_rate=.95,
                    target_mean_hits=12.1, target_hit12_rate=.92)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve all 17 lessons, rewards, domains and quality/safety limits."""
    if len(source) != 17 or source[10].name != 'fplc_v279_s11_near_y_080mm':
        raise ValueError('V280 requires the complete 17-stage V279 course')
    return [replace(
        stage, name=stage.name.replace('fplc_v279_', 'fplc_v280_', 1),
        **(GATE_CHANGES if i >= 10 else {}),
        notes=('V280 permits modest DR survival degradation only at S11--S17: '
               'length .97/full .95/hits 12.1/P12 .92. Every reward, physical '
               'lesson, observation/control/physics/reset/DR and remaining '
               'motion/apex/view/safety gate is V279-exact. S1--S10 including '
               'the nominal retention gate stay exact. Native rolling32, '
               'hold16, independent288x1200 and nominal validation remain; '
               'no forced advancement. Latest full actor/critic/Adam; '
               '1024x128, every numerical PPO setting and learning of every '
               'minibatch remain, without an update cap.'))
        for i, stage in enumerate(source)]


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v279_energy_progress import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V280 requires unchanged strict nominal retention')
    previous(SimpleNamespace(**{
        **vars(settings),
        'advance_retention_stage_name': 'fplc_v279_s01_workspace_alignment',
    }))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    """Retain genuine graduations and full learning state; no earlier best."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in
                     ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V280 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 11
                or payload.get('stage_name') != 'fplc_v279_s11_near_y_080mm'
                or update < 300
                or payload.get('step') != 3114008576 + update * 131072
                or int(payload['opt'].t) != 769726 + update * 32):
            raise ValueError('V280 requires the latest full V279 after >=300 updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V280 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V280 rejects another source or from-scratch launch')
