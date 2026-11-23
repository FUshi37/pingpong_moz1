"""V323/M12 and V324/M13: noise-only long-episode continuation contracts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, make_dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

from fixed_plane_v321_long_episode import LongLesson

MEAN_M = (.000048, .000072, -.000227)
STD_M = (.001451, .001418, .002970)
RHO_AT_90HZ = (.935, .907, .790)
NOISE_FIELDS = ('ball_position_noise_ar1_enabled', 'ball_position_noise_mean_m',
                'ball_position_noise_std_m', 'ball_position_noise_rho_at_90hz')
EXECUTION_VERSION = 'mjx_topology_execution_v2'
OUTPUT_ROOT = Path('pingpong_controller/outputs/rl_sim')
DOMAIN_RANDOMIZED_STATE_FIELDS = frozenset({'ball_position_noise_epsilon_m'})
CRITIC_STATE_FIELDS = DOMAIN_RANDOMIZED_STATE_FIELDS


@dataclass(frozen=True)
class NoiseContinuation:
    """One authenticated parent, unchanged learner, and two new sensing lessons."""

    version: int
    parent_model: str
    parent_checkpoint: Path
    parent_sha256: str
    parent_step: int
    parent_update: int
    parent_adam_t: int
    native_env_sha256: str
    parent_stage_update: int
    min_survival: float
    min_length_fraction: float
    critic_obs_dim: int = 396
    execution_version: str = EXECUTION_VERSION

    @property
    def gpu_index(self) -> int:
        return 0 if self.version == 323 else 1

    @property
    def gpu_uuid(self) -> str:
        return ('GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4' if self.gpu_index == 0
                else 'GPU-e74458ec-002a-1bac-e0be-d0a5713b661e')

    @property
    def profile(self) -> str:
        return f'fixed_plane_limit_cycle_qvel_rmp_gpu{self.gpu_index}_v{self.version}'

    @property
    def parent_profile(self) -> str:
        return f'fixed_plane_limit_cycle_qvel_rmp_gpu1_v{self.version - 2}'

    @property
    def course(self) -> tuple[LongLesson, ...]:
        return tuple(LongLesson(f'fplc_v{self.version}_s{i}_{label}', seconds,
            self.min_survival, self.min_length_fraction)
            for i, (label, seconds) in enumerate((('camera_ar1_1x', 30.),
                ('camera_ar1_1x_consolidation', 1200.)), 1))


SPECS = {
    323: NoiseContinuation(323, 'M12', OUTPUT_ROOT /
        'releases/fplc_v321_s4_u2097_20261029_r1/policy.pkl',
        '889eb976b18567d607ab9f86f35457c89f901e2ecf221b9391128cb471f0d001',
        1609695232, 2097, 402875,
        '67eed27b5d9cddd392682022c3efd220081bb079ce20fe7884651883cfdc2db3',
        400, .95, .99),
    324: NoiseContinuation(324, 'M13', OUTPUT_ROOT /
        'fplc_v322_gpu1_late_center_precision_20261103_r1/training_qualified_inherited1200s.pkl',
        'd35ea68508cbdfbaddfb3766f21c3f4e82f6f127e3caab8262d9e69a9ad307bc',
        8238268416, 31361, 2020766,
        '39ba2da83f498cb84b041e016d6f1031130007dad9efafc21ead388f214e3b09',
        48, .93, .95),
}


def spec_for(version: int) -> NoiseContinuation:
    """Reject unreviewed versions instead of widening a historical allowlist."""
    if version not in SPECS:
        raise ValueError('Only V323 and V324 are registered noise continuations')
    return SPECS[version]


SPEC = SPECS[323]
PROFILE = SPEC.profile
PARENT_SHA256 = SPEC.parent_sha256
COURSE = SPEC.course


def canonical(value: Any) -> str:
    """Serialize configuration contracts including nested immutable dataclasses."""
    return json.dumps(value, sort_keys=True, default=lambda x:
        asdict(x) if hasattr(x, '__dataclass_fields__') else str(x))


def digest_file(path: Path) -> str:
    """Hash a source or input artifact without retaining its bytes."""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=4)
def _config_type(parent_type: type) -> type:
    # The two parents have different frozen schemas. Extend their actual
    # dataclass without importing or substituting the current shared env.
    values = (True, MEAN_M, STD_M, RHO_AT_90HZ)
    return make_dataclass('BallObservationNoiseConfig',
        [(name, type(value), field(default=value))
         for name, value in zip(NOISE_FIELDS, values)],
        bases=(parent_type,), frozen=True)


def make_noise_config(parent: Any, *, enabled: bool = True) -> Any:
    """Add M10's exact 1x position noise; preserve all native config values."""
    zero_fields = ('ball_obs_pos_noise_std', 'ball_obs_vel_noise_std',
        'ball_obs_vel_xy_noise_std', 'ball_obs_posthit_vel_xy_noise_std',
        'ball_obs_dropout_prob', 'ball_obs_dropout_burst_prob')
    off_fields = ('dr_randomize_ball_obs_frame', 'dr_randomize_latency',
                 'dr_post_hit_ball_obs_latency')
    if (parent.horizon_sec != 1200. or parent.ball_obs_rate_hz != 90.
            or not parent.ball_obs_fractional_rate
            or any(getattr(parent, name) != 0. for name in zero_fields)
            or any(getattr(parent, name) for name in off_fields)
            or tuple(parent.ball_obs_nominal_pos_bias_base) != (0., 0., 0.)
            or tuple(parent.ball_obs_nominal_vel_bias_base) != (0., 0., 0.)
            or any(hasattr(parent, name) for name in NOISE_FIELDS)):
        raise ValueError('Noise continuation requires the clean 90Hz parent contract')
    return _config_type(type(parent))(**vars(parent),
                                    ball_position_noise_ar1_enabled=enabled)


def validate_parent_payload(payload: dict, sha256: str,
                            spec: NoiseContinuation) -> None:
    """Pin full source identity, dimensions and optimizer time before migration."""
    if (sha256 != spec.parent_sha256 or payload.get('step') != spec.parent_step
            or payload.get('global_update') != spec.parent_update
            or payload.get('stage_update') != spec.parent_stage_update
            or payload.get('stage_index') != 3
            or payload.get('stage_name') != f'fplc_v{spec.version - 2}_s4_1200s'
            or payload.get('args', {}).get('curriculum_profile') != spec.parent_profile
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))
                != (72, 393, 7)
            or not {'pi', 'v', 'log_std'}.issubset(payload.get('params', {}))
            or payload.get('opt') is None
            or int(payload['opt'].t) != spec.parent_adam_t):
        raise ValueError(f'V{spec.version} requires the pinned full {spec.parent_model} state')
    if spec.version == 324:
        saved = payload.get('v322_contract', {})
        if (saved.get('reward_preset') != 'late_center_precision'
                or saved.get('loss_preset') != 'anchor1'
                or saved.get('gate_preset') != 'deployment_groups'):
            raise ValueError('V324 preserves the exact M13 reward/loss/gate presets')


def append_noise_critic_state(train_state: Any) -> Any:
    """Append three zero critic rows to parameters and Adam m/v; retain time."""
    import jax.numpy as jnp

    def extend(tree):
        if tree['v']['l1']['w'].shape[0] != 393:
            raise ValueError('Noise critic migration requires the exact 393-D prefix')
        layer = dict(tree['v']['l1'])
        layer['w'] = jnp.pad(jnp.asarray(layer['w']), ((0, 3), (0, 0)))
        return {**tree, 'v': {**tree['v'], 'l1': layer}}

    return train_state._replace(params=extend(train_state.params),
        opt=train_state.opt._replace(m=extend(train_state.opt.m),
                                    v=extend(train_state.opt.v)))
