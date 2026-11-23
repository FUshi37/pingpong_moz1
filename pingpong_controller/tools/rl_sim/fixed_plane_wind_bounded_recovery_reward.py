"""Recovery-only bounded outcome shaping; native dynamics and gates are untouched."""
from dataclasses import field, make_dataclass
from functools import lru_cache
import math
from typing import Any, Sequence


REVISION = 'bounded_quality_real_survival_v1'
GATE_MARGIN_SCORE_CONTRACT = dict(identity='gate_margin_v1', plateau_fraction=.8,
    excess_fraction=.2, apex_target_m=1.425, apex_margin_m=.010,
    apex_excess_sigma_m=.005,
    ball_xy_allowance='subtract causal wind budget before base-limit normalization')
GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT = {
    key: value for key, value in GATE_MARGIN_SCORE_CONTRACT.items()
    if key not in ('apex_margin_m', 'apex_excess_sigma_m')}
GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT.update(
    identity='gate_margin_gaussian_apex_v1', apex_mode='original_gaussian', apex_sigma_m=.0125)
HEIGHT_PRIORITY_SCORE_CONTRACT = {**GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT,
    'identity': 'gate_margin_height_priority_v1',
    'height_focus': 'geometric quality times sqrt(apex score times max(band score,0.01))',
    'extra_apex_exponent': .5, 'extra_band_exponent': .5, 'band_miss_floor': .01}
Z_MARGIN_SCORE_CONTRACT = {**HEIGHT_PRIORITY_SCORE_CONTRACT,
    'identity': 'gate_margin_height_z_margin_v1', 'racket_z_margin_m': .015,
    'z_margin_point': 'native racket_z termination site in world metres',
    'z_margin_credit': 'multiply all positive recovery credit by clipped distance to either hard Z edge / margin',
    'hard_bounds': 'unchanged fixed_contact_racket_z_bounds_m'}
MOTION_PRIORITY_SCORE_CONTRACT = {**Z_MARGIN_SCORE_CONTRACT,
    'identity': 'gate_margin_height_motion_z_margin_v1',
    'motion_focus': 'joint hit quality times fourth root of racket XY and recurrent mean scores',
    'extra_racket_xy_exponent': .25, 'extra_recurrent_mean_exponent': .25,
    'motion_scope': 'hit credit only; height focus, survival, dense credit and failure cost unchanged'}
Z_ANTICIPATION_SCORE_CONTRACT = {**MOTION_PRIORITY_SCORE_CONTRACT,
    'identity': 'gate_margin_height_motion_z_anticipation_v1',
    'racket_z_prediction_horizon_s': .020,
    'z_margin_credit': 'minimum current and constant-velocity predicted distance to each hard Z edge / 15mm',
    'velocity_source': 'native prev_racket_vel at the completed physical step; no state or control change'}
APEX_PRECISION_SCORE_CONTRACT = {**Z_ANTICIPATION_SCORE_CONTRACT,
    'identity': 'gate_margin_height_motion_z_anticipation_apex008_v1',
    'apex_mode': 'precision_gaussian', 'apex_sigma_m': .008,
    'precision_scope': 'apex Gaussian width only; all other scores and physical gates unchanged'}
NATIVE_Z_FAILURE_SCORE_CONTRACT = {**APEX_PRECISION_SCORE_CONTRACT,
    'identity': 'gate_margin_height_motion_z_anticipation_apex008_native_z_floor_v1',
    'failure_scope': 'failed steps only; minimum of -100 and existing native Z penalty',
    'native_penalty_metric': 'reward/racket_z_limit_termination_penalty',
    'positive_credit': 'exactly the preceding apex008 recipe'}
AUXILIARY_APEX_SCORE_CONTRACT = {**NATIVE_Z_FAILURE_SCORE_CONTRACT,
    'identity': 'gate_margin_height_motion_z_anticipation_apex008_native_z_floor_apex_support_v1',
    'auxiliary_target_m': 1.425, 'auxiliary_sigma_m': .012,
    'auxiliary_max_credit_per_s': 4.,
    'auxiliary_score': 'exp(-Huber(abs(error)/sigma)); unit Huber transition',
    'auxiliary_scope': 'independent confirmed steady-hit credit, actual interval, same shelter/Z/failure gates'}


def auxiliary_apex_score(apex_m: Any, *, xp: Any) -> Any:
    """Retain a Gaussian core and graded feedback on larger apex errors."""
    cfg = AUXILIARY_APEX_SCORE_CONTRACT
    normalized = xp.abs(apex_m-cfg['auxiliary_target_m'])/cfg['auxiliary_sigma_m']
    loss = .5*xp.minimum(normalized, 1.)**2+xp.maximum(normalized-1., 0.)
    return xp.exp(-loss)


def racket_z_margin_score(z_m: Any, bounds_m: Sequence[float], margin_m: float,
                          *, xp: Any) -> Any:
    """Return a scalar or [environment] score from the native world-Z site."""
    lower, upper = map(float, bounds_m)
    if (not all(math.isfinite(v) for v in (lower, upper, margin_m))
            or upper <= lower or not 0. < margin_m < .5*(upper-lower)):
        raise ValueError('Invalid racket Z bounds or soft margin')
    distance_m = xp.minimum(z_m-lower, upper-z_m)
    return xp.clip(distance_m/margin_m, 0., 1.)


def anticipatory_racket_z_margin_score(
        z_m: Any, vz_m_s: Any, bounds_m: Sequence[float], margin_m: float,
        horizon_s: float, *, xp: Any) -> Any:
    """Discount outward Z momentum before it reaches the native soft margin.

    This is a reward score, not a dynamics prediction or a safety filter.
    Moving away from an edge never removes its existing position discount.
    """
    current = racket_z_margin_score(z_m, bounds_m, margin_m, xp=xp)
    if not math.isfinite(horizon_s) or horizon_s < 0.:
        raise ValueError('Invalid racket Z prediction horizon')
    if horizon_s == 0.:
        return current
    lower, upper = map(float, bounds_m)
    lower_distance_m = z_m-lower + xp.minimum(vz_m_s, 0.)*horizon_s
    upper_distance_m = upper-z_m - xp.maximum(vz_m_s, 0.)*horizon_s
    return xp.clip(xp.minimum(lower_distance_m, upper_distance_m)/margin_m, 0., 1.)


@lru_cache(maxsize=4)
def _config_type(parent):
    return make_dataclass('BoundedWindRecoveryRewardConfig',
        [('wind_bounded_recovery_reward', bool, field(default=True))],
        bases=(parent,), frozen=True)


@lru_cache(maxsize=4)
def _joint_config_type(parent):
    return make_dataclass('JointQualityWindRecoveryConfig',
        [('wind_recovery_quality_aggregation', str, field(default='geometric'))],
        bases=(parent,), frozen=True)


@lru_cache(maxsize=4)
def _margin_config_type(parent):
    return make_dataclass('GateMarginWindRecoveryConfig',
        [('wind_recovery_quality_score_mode', str, field(default='gate_margin_v1'))],
        bases=(parent,), frozen=True)


def with_reward_config(cfg, enabled, *, aggregation='arithmetic', score_mode='gaussian'):
    if not enabled:
        return cfg
    result = _config_type(type(cfg))(**vars(cfg), wind_bounded_recovery_reward=True)
    if aggregation == 'arithmetic':
        if score_mode != 'gaussian':
            raise ValueError('Gate-margin scores require the geometric recovery recipe')
        return result
    if aggregation != 'geometric':
        raise ValueError('Unknown recovery quality aggregation')
    result = _joint_config_type(type(result))(**vars(result), wind_recovery_quality_aggregation='geometric')
    if score_mode == 'gaussian':
        return result
    if score_mode not in (GATE_MARGIN_SCORE_CONTRACT['identity'],
                           GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT['identity'],
                           HEIGHT_PRIORITY_SCORE_CONTRACT['identity'],
                           Z_MARGIN_SCORE_CONTRACT['identity'],
                           MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                           Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                           APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        raise ValueError('Unknown recovery quality score mode')
    return _margin_config_type(type(result))(**vars(result), wind_recovery_quality_score_mode=score_mode)


def bounded_reward(native_reward, metrics, *, gain, interval_s, dt_s, xp,
                   aggregation='arithmetic', score_mode='gaussian', z_margin_score=None):
    """Pay positive real survival and balanced quality, with time-normalized hits.

    Every native outcome remains in telemetry. A physical termination costs100;
    the explicit native-Z-floor mode retains any larger native Z cost.
    no hit recency gives no income. Quality credit is proportional to the actual
    counted-hit interval, avoiding a reward for merely increasing contact rate.
    """
    from fixed_plane_wind_objectives import quality_limits
    if score_mode not in ('gaussian', GATE_MARGIN_SCORE_CONTRACT['identity'],
                          GATE_MARGIN_GAUSSIAN_APEX_SCORE_CONTRACT['identity'],
                          HEIGHT_PRIORITY_SCORE_CONTRACT['identity'],
                          Z_MARGIN_SCORE_CONTRACT['identity'],
                          MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                          Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                          APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        raise ValueError('Unknown recovery quality score mode')
    if score_mode != 'gaussian' and aggregation != 'geometric':
        raise ValueError('Gate-margin scores require geometric aggregation')
    if score_mode in (Z_MARGIN_SCORE_CONTRACT['identity'],
                      MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                      Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                      APEX_PRECISION_SCORE_CONTRACT['identity'],
                      NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                      AUXILIARY_APEX_SCORE_CONTRACT['identity']) and z_margin_score is None:
        raise ValueError('Z-margin reward requires the measured native racket Z score')
    q = quality_limits(325)
    hit = metrics['hit_event_count']
    steady = metrics['rapid_steady_steady_events']
    recurrent = metrics['recurrent_hit_event_count']
    pair = metrics['v80_late_pair_event']
    allowance = metrics['wind/required_outgoing_vxy_m_s']

    def score(value, event, limit, *, squared=False, wind_allowance=0.):
        if score_mode != 'gaussian':
            physical = value / xp.maximum(event, 1.)
            physical_limit = limit
            if squared:
                physical = xp.sqrt(xp.maximum(physical, 0.))
                physical_limit = xp.sqrt(xp.maximum(limit, 0.))
            base_limit = xp.maximum(physical_limit - wind_allowance, 1e-12)
            normalized = xp.maximum(physical - wind_allowance, 0.) / base_limit
            excess = xp.maximum(normalized - GATE_MARGIN_SCORE_CONTRACT['plateau_fraction'], 0.)
            exponent = (excess / GATE_MARGIN_SCORE_CONTRACT['excess_fraction'])**2
            return xp.where(event > 0., xp.exp(-.5*exponent), 0.)
        normalized = value / xp.maximum(event, 1.) / xp.maximum(limit, 1e-12)
        exponent = normalized if squared else normalized**2
        return xp.where(event > 0., xp.exp(-.5*xp.maximum(exponent, 0.)), 0.)

    apex = metrics['rapid_steady_steady_apex_sum_z_m'] / xp.maximum(steady, 1.)
    apex_sigma_m = (APEX_PRECISION_SCORE_CONTRACT['apex_sigma_m']
        if score_mode in (APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']) else .0125)
    scores = dict(
        center=score(metrics['hit_contact_center_dist_sum'], hit, q['center_m']),
        apex=xp.where(steady > 0., xp.exp(-.5*((apex-1.425)/apex_sigma_m)**2), 0.),
        band=metrics['v92_in_1p4_1p45_event'] / xp.maximum(metrics['v92_late_contact_event'], 1.),
        racket_xy=score(metrics['rapid_steady_steady_racket_vxy_sq_sum'], steady,
                        q['racket_vxy_rms_m_s']**2, squared=True),
        angular=score(metrics['rapid_steady_steady_racket_omega_sq_sum'], steady,
                      q['racket_omega_rms_rad_s']**2, squared=True),
        ball_mean=score(metrics['hit_vxy_sum'], hit, q['ball_vxy_m_s']+allowance,
                        wind_allowance=allowance),
        ball_rms=score(metrics['hit_vxy_sq_sum'], hit,
                       (q['ball_vxy_rms_m_s']+allowance)**2, squared=True,
                       wind_allowance=allowance),
        recurrent_mean=score(metrics['recurrent_hit_racket_vxy_sum'], recurrent,
                            q['recurrent_vxy_m_s']),
        recurrent_rms=score(metrics['recurrent_hit_racket_vxy_sq_sum'], recurrent,
                           q['recurrent_vxy_rms_m_s']**2, squared=True),
        jump=score(metrics['v80_late_pair_jump_sum_m'], pair, q['adjacent_jump_m']))
    if score_mode == 'gate_margin_v1':
        error = xp.maximum(xp.abs(apex - GATE_MARGIN_SCORE_CONTRACT['apex_target_m'])
                           - GATE_MARGIN_SCORE_CONTRACT['apex_margin_m'], 0.)
        scores['apex'] = xp.where(steady > 0., xp.exp(-.5*(
            error / GATE_MARGIN_SCORE_CONTRACT['apex_excess_sigma_m'])**2), 0.)
    weights = {name: 2. if name in ('center', 'apex', 'band') else 1. for name in scores}
    if aggregation == 'arithmetic':
        quality = sum(weights[name]*value for name,value in scores.items()) / sum(weights.values())
    elif aggregation == 'geometric':
        # Use a small band-miss score, retaining a continuous signal from the
        # other components. Log-space avoids underflow from rare large errors.
        joint_scores = {**scores, 'band': xp.maximum(scores['band'], .01)}
        quality = xp.exp(sum(weights[name]*xp.log(xp.maximum(value, xp.exp(-20.)))
            for name,value in joint_scores.items()) / sum(weights.values()))
    else:
        raise ValueError('Unknown recovery quality aggregation')
    if score_mode in (HEIGHT_PRIORITY_SCORE_CONTRACT['identity'], Z_MARGIN_SCORE_CONTRACT['identity'],
                      MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                      Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                      APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        # The geometric root dilutes the original apex exponent to2/13.
        # Preserve its useful center/motion balance, while making height
        # precision and physical band outcomes costly to trade away.
        quality = quality * xp.sqrt(scores['apex'] * xp.maximum(
            scores['band'], HEIGHT_PRIORITY_SCORE_CONTRACT['band_miss_floor']))
    unfocused_quality = quality
    if score_mode in (MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                      Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                      APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        motion_focus = (scores['racket_xy'] * scores['recurrent_mean'])**.25
        quality = quality * motion_focus
    recent = metrics['v321/recent_hit'] > .5
    failed = metrics['terminated'] > .5
    active = recent & ~failed
    survival = xp.where(active, 2.*dt_s, 0.)
    dense = xp.where(active & (hit < .5), .5*dt_s*(1.+xp.tanh(native_reward)), 0.)
    event = xp.where(active, 12.*xp.clip(interval_s, 0., 1.2)*hit*quality, 0.)
    failure_cost = -100.
    if score_mode in (NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                      AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        penalty_key = NATIVE_Z_FAILURE_SCORE_CONTRACT['native_penalty_metric']
        if penalty_key not in metrics:
            raise ValueError('Native-Z-floor reward requires the native Z penalty metric')
        failure_cost = xp.minimum(-100., metrics[penalty_key])
    failure = xp.where(failed, failure_cost, 0.)
    replacement = survival+dense+event+failure
    if score_mode == AUXILIARY_APEX_SCORE_CONTRACT['identity']:
        auxiliary_score = xp.where(steady > 0., auxiliary_apex_score(apex, xp=xp), 0.)
        auxiliary_credit = xp.where(active & (steady > 0.),
            AUXILIARY_APEX_SCORE_CONTRACT['auxiliary_max_credit_per_s']
            *xp.clip(interval_s, 0., 1.2)*hit*auxiliary_score, 0.)
        replacement = replacement+auxiliary_credit
    if score_mode in (Z_MARGIN_SCORE_CONTRACT['identity'],
                      MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                      Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                      APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        z_margin_score = xp.clip(z_margin_score, 0., 1.)
        positive_credit = survival+dense+event
        if score_mode == AUXILIARY_APEX_SCORE_CONTRACT['identity']:
            positive_credit = positive_credit+auxiliary_credit
        replacement = z_margin_score*positive_credit+failure
    result = xp.where(gain <= 0., native_reward, (1.-gain)*native_reward+gain*replacement)
    terms = {'reward/recovery_native_replaced': -gain*native_reward,
             'reward/recovery_survival': gain*survival,
             'reward/recovery_dense_quality': gain*dense,
             'reward/recovery_quality': gain*event,
             'reward/recovery_failure': gain*failure}
    if score_mode in (Z_MARGIN_SCORE_CONTRACT['identity'],
                      MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                      Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                      APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        terms['reward/recovery_z_margin'] = gain*(z_margin_score-1.)*positive_credit
        terms['recovery/z_margin_score'] = z_margin_score
    if score_mode in (MOTION_PRIORITY_SCORE_CONTRACT['identity'],
                      Z_ANTICIPATION_SCORE_CONTRACT['identity'],
                      APEX_PRECISION_SCORE_CONTRACT['identity'],
                          NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        unfocused_event = xp.where(active,
            12.*xp.clip(interval_s, 0., 1.2)*hit*unfocused_quality, 0.)
        terms['recovery/motion_focus_score'] = motion_focus
        terms['recovery/motion_focus_reward_delta'] = gain*z_margin_score*(event-unfocused_event)
    if score_mode in (APEX_PRECISION_SCORE_CONTRACT['identity'],
                      NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                          AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        terms['recovery/apex_sigma_m'] = xp.full_like(native_reward, apex_sigma_m)
    if score_mode in (NATIVE_Z_FAILURE_SCORE_CONTRACT['identity'],
                      AUXILIARY_APEX_SCORE_CONTRACT['identity']):
        terms['recovery/native_z_failure_extra_cost'] = gain*xp.where(
            failed, -failure_cost-100., 0.)
    if score_mode == AUXILIARY_APEX_SCORE_CONTRACT['identity']:
        terms['reward/recovery_auxiliary_apex'] = gain*auxiliary_credit
        terms['recovery/auxiliary_apex_score'] = auxiliary_score
    terms.update({'recovery/score_'+name: value for name,value in scores.items()})
    return result, terms
