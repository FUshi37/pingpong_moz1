"""Additional precision feedback on confirmed windy impacts and measured apexes."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

MODE = 'clean10_dynamics_reference_center_negy_v10'
BASE_MODE = 'clean10_dynamics_reference_center_negy_v2'
SOURCE_SHA256 = '64b6aed4c5177a0fb1d97d8bf946a88c258399aca96708d01b902e77337012d9'
SOURCE_UPDATE = 14500
SOURCE_STEP = 3510239232
SOURCE_ADAM_T = 966575


def measured_precision_contract() -> dict:
    """Record the two reward coefficients and their physical event sources."""
    return dict(
        base=BASE_MODE,
        center_rate_per_s=6.,
        apex_rate_per_s=8.,
        center_mean_m={325: .0055, 326: .0065},
        center_rms_m={325: .007, 326: .008},
        apex_target_m=1.425,
        apex_width_m=.010,
        per_term_loss_cap=4.,
        affected_population='wind only',
        events='existing confirmed contact center and measured flight apex',
        warmup_policy_steps=8192,
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
    )


@lru_cache(maxsize=16)
def _reward_config_type(parent: type) -> type:
    return make_dataclass(
        'WindMeasuredPrecisionConfig',
        [('wind_measured_precision', bool, field(default=True))],
        bases=(parent,),
        frozen=True,
    )


def with_measured_precision_config(cfg: Any, enabled: bool) -> Any:
    """Add the reward switch only for the selected recipe."""
    return _reward_config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def measured_precision_terms(
    metrics: dict, *, windy: Any, center_gain: Any, height_gain: Any,
    center_interval_s: Any, height_interval_s: Any, reward_steps: Any,
    version: int, xp: Any,
) -> dict:
    """Return two nonpositive per-lane costs using existing physical metrics."""
    if version not in (325, 326):
        raise ValueError('Measured precision requires V325 or V326')
    hit_count = xp.asarray(metrics['hit_event_count'])
    apex_count = xp.asarray(metrics['wind_height/apex_events'])
    live_wind = windy & (metrics['terminated'] < .5)
    center = metrics['hit_contact_center_dist_sum'] / xp.maximum(hit_count, 1.)
    apex = metrics['wind_height/apex_sum_m'] / xp.maximum(apex_count, 1.)
    mean_width = .0055 if version == 325 else .0065
    rms_width = .007 if version == 325 else .008
    center_loss = .5 * (center / mean_width + (center / rms_width)**2)
    apex_loss = ((apex - 1.425) / .010)**2

    progress = xp.clip(xp.asarray(reward_steps, dtype=hit_count.dtype) / 8192., 0., 1.)
    warmup = progress**2 * (3. - 2. * progress)

    def cost(count, error, gain, interval, rate):
        weight = count * gain * warmup * xp.clip(interval, 0., 1.2)
        return xp.where(live_wind & (count > 0.), -rate * weight * xp.clip(error, 0., 4.), 0.)

    return {
        'reward/measured_wind_center_cost': cost(hit_count, center_loss, center_gain, center_interval_s, 6.),
        'reward/measured_wind_apex_cost': cost(apex_count, apex_loss, height_gain, height_interval_s, 8.),
    }


def verify_source(payload: dict, sha256: str, *, mode: str = MODE,
                  config_flag: str = 'wind_measured_precision',
                  source: tuple[int, int, int, str] = (
                      SOURCE_UPDATE, SOURCE_STEP, SOURCE_ADAM_T, SOURCE_SHA256,
                  )) -> None:
    """Admit only the retained S16 parent or an exact continuation."""
    args = payload.get('args', {})
    update = payload.get('global_update')
    stage = payload.get('stage_index')
    opt = payload.get('opt')
    adam = opt.get('t') if isinstance(opt, dict) else getattr(opt, 't', None)
    source_update, source_step, source_adam_t, source_sha256 = source
    if (args.get('curriculum_profile') != 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v325'
            or type(update) is not int or update < source_update
            or type(stage) is not int or not 15 <= stage < 32
            or adam is None):
        raise ValueError('Expected the reviewed S16 learner or its continuation')
    elapsed = update - source_update
    if payload.get('step') != source_step + elapsed * 131072 or int(adam) != source_adam_t + elapsed * 32:
        raise ValueError('Inconsistent S16 optimizer accounting')
    saved_mode = args.get('wind_training_mix')
    if saved_mode == BASE_MODE:
        if elapsed or sha256 != source_sha256 or stage != 15 or payload.get('stage_converged', False):
            raise ValueError('Unreviewed S16 reward source')
    elif saved_mode != mode or payload.get('env_cfg', {}).get(config_flag) is not True:
        raise ValueError('Incompatible S16 reward continuation')
