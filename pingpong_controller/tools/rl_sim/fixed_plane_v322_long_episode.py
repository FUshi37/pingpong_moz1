"""V322: M11 long-episode continuation with a cheap final acceptance test."""
import math
from dataclasses import asdict, replace

from fixed_plane_v321_long_episode import (
    CONFIG_OVERRIDES, GPU_UUID, MAX_HIT_GAP_S, XML_SHA256, QUALITY_FIELDS,
    EndpointWindow, CheckpointRetention, LongLesson, quality_summary,
    survival_reward, make_config as make_long_config,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v322'
PARENT_SHA256 = 'fc1ae6e78a3b2647a4398708fdeb98cfdc710dd5a6aa1bc04980db09e265185f'
PARENT_RUN = 'fplc_v319_gpu0_mean0265_completion_20261026_r1'
PARENT_FILE = '17_fplc_v281_s17_unchanged_consolidation.pkl'
COURSE = tuple(LongLesson(f'fplc_v322_s{i}_{seconds}s', float(seconds), .90, .97)
               for i, seconds in enumerate((30, 120, 300, 1200), 1))
EVAL_ENVS = 16
EVAL_SECONDS = 60
EVAL_WALL_BUDGET_S = 600
EVAL_RETRY_UPDATES = 1000
EVAL_LESSON = LongLesson('fplc_v322_final_short_acceptance_60s', 60., .875, .97)
GATE_PRESETS = {'baseline': (.90, .97), 'm11_reference': (.93, .95),
                'deployment_groups': (.93, .95)}


def course_for(preset: str = 'baseline') -> tuple:
    """Calibrate endpoint gates explicitly; keep all four 1200s lessons."""
    if preset not in GATE_PRESETS:
        raise ValueError(f'Unknown V322 gate preset: {preset}')
    survival, length = GATE_PRESETS[preset]
    return tuple(replace(s, min_survival=survival, min_length_fraction=length)
                 for s in COURSE)


def evaluation_lesson_for(preset: str = 'baseline') -> LongLesson:
    """Retain the cheap final sample and its original survival requirement."""
    course_for(preset)
    return replace(EVAL_LESSON, min_length_fraction=GATE_PRESETS[preset][1])


def resolve_gate_resume(contract: dict, requested: str | None,
                        *, allow_change: bool) -> str:
    """Authenticate previous thresholds before an explicitly approved change."""
    previous = contract.get('gate_preset', 'baseline')
    expected = [asdict(s) for s in course_for(previous)]
    if contract.get('course', expected) != expected:
        raise ValueError('V322 source course does not match its gate preset')
    if previous == 'deployment_groups':
        from fixed_plane_v322_condition_gates import condition_gate_settings, required_conditions
        saved_scope = contract.get('condition_gate_required_by_stage')
        expected_scope = [required_conditions(i) for i in range(4)]
        if (contract.get('condition_gate_settings') != condition_gate_settings()
                or saved_scope is None
                or [tuple(v) for v in saved_scope] != expected_scope):
            raise ValueError('V322 source condition gate settings changed')
    selected = previous if requested is None else requested
    course_for(selected)
    if selected != previous and not allow_change:
        raise ValueError('V322 resume requires an explicit gate change acknowledgment')
    return selected


PPO_EXPECTED = dict(n_envs=1024, n_steps=128, minibatch_size=16384,
    update_epochs=4, learning_rate=1.5e-5, gamma=.9995, gae_lambda=.99,
    clip_range=.12, target_kl=.006, ent_coef=.001, vf_coef=.05,
    min_log_std=-3., max_grad_norm=.5, counterfactual_supervision_coef=.05,
    actor_anchor_kl_coef=0., actor_anchor_replay_kl_coef=10., actor_anchor_replay_max_kl=0.,
    residual_l2_coef=.01, failure_focus_hit_threshold=14,
    failure_focus_tail_steps=128, failure_focus_balanced_mass=.02)
LOSS_PRESETS = {'baseline': {}, 'anchor1': {'actor_anchor_replay_kl_coef': 1.}}


def loss_settings(preset: str = 'baseline') -> dict:
    """Keep the historical learner exact unless the approved scalar is selected."""
    if preset not in LOSS_PRESETS:
        raise ValueError(f'Unknown V322 loss preset: {preset}')
    return {**PPO_EXPECTED, **LOSS_PRESETS[preset]}


def resolve_loss_resume(contract: dict, source_args: dict, requested: str | None,
                        *, allow_change: bool) -> str:
    """Authenticate the source before acknowledging a one-scalar continuation."""
    previous = contract.get('loss_preset', 'baseline')
    for name, value in loss_settings(previous).items():
        if source_args.get(name) != value:
            raise ValueError(f'V322 resume PPO mismatch: {name}')
    selected = previous if requested is None else requested
    loss_settings(selected)
    if selected != previous and not allow_change:
        raise ValueError('V322 resume requires explicit loss change acknowledgment')
    return selected


REPLAY_HASHES = dict(
    actor_anchor_reference_checkpoint='b102831e018585b7127668fcc40a8de258325e2a7a58551f02ddc3b6ddce2608',
    actor_anchor_replay_obs='2741d1898620b62d227b0ee45a4f73aae6691503e2bb1e41054ed0761f079cf7',
    counterfactual_replay_obs='793575f1da785c295ed110a29e1d18052f4f9fa5edc14a332079e534fb210bc3',
    counterfactual_replay_actions='131253156dd2084f6062fc0846cd14a6b411fb502866f2fdf65a7897dfa72439')

# The user explicitly requires retaining V322. Distinguish reward experiments
# by a recorded opt-in preset and a new run identity, preserving the baseline.
REWARD_PRESETS = {
    'baseline': {},
    'stability': {
        'fixed_plane_v158_late_squared_vxy_penalty': 3.,
        'fixed_plane_v241_physical_apex_huber_penalty': 1.5,
        'fixed_plane_v247_late_physical_band_penalty': 2.,
    },
}
REWARD_PRESETS['stability_launch'] = dict(REWARD_PRESETS['stability'])
REWARD_PRESETS['contact_quality'] = dict(REWARD_PRESETS['stability'])
REWARD_PRESETS['balanced_contact'] = {
    **REWARD_PRESETS['contact_quality'],
    'fixed_plane_v253_flight_xy_reward': .25,
}
REWARD_PRESETS['startup_release'] = dict(REWARD_PRESETS['balanced_contact'])
REWARD_PRESETS['startup_transfer'] = dict(REWARD_PRESETS['startup_release'])
# User-authorized reward-only trial: restore M11's native preparation timing
# and severity, keeping every other startup_transfer reward exactly unchanged.
REWARD_PRESETS['native_prepare'] = dict(REWARD_PRESETS['startup_transfer'])
REWARD_PRESETS['startup_lead'] = dict(REWARD_PRESETS['native_prepare'])
REWARD_PRESETS['shared_prepare'] = dict(REWARD_PRESETS['startup_lead'])
REWARD_PRESETS['shared_early_prepare'] = dict(REWARD_PRESETS['shared_prepare'])
REWARD_PRESETS['native_third_focus'] = dict(REWARD_PRESETS['shared_early_prepare'])
REWARD_PRESETS['native_reference_prepare'] = dict(REWARD_PRESETS['native_third_focus'])
REWARD_PRESETS['native_reference_gain6'] = dict(REWARD_PRESETS['native_reference_prepare'])
REWARD_PRESETS['native_return_prepare'] = dict(REWARD_PRESETS['native_reference_gain6'])
REWARD_PRESETS['projected_return_prepare'] = dict(REWARD_PRESETS['native_reference_gain6'])
REWARD_PRESETS['projected_return_acquisition'] = dict(REWARD_PRESETS['projected_return_prepare'])
REWARD_PRESETS['weak_release_outcome'] = dict(REWARD_PRESETS['projected_return_acquisition'])
REWARD_PRESETS['reference_timing'] = dict(REWARD_PRESETS['weak_release_outcome'])
REWARD_PRESETS['late_cycle_repair'] = dict(REWARD_PRESETS['reference_timing'])
REWARD_PRESETS['late_center_precision'] = dict(REWARD_PRESETS['reference_timing'])
ACQUISITION_XY_SETTINGS = dict(apex_target_m=1.39, apex_sigma_m=.040,
    xy_sigma_m_s=.070, preserved_clean_floor_budget=512.)
WEAK_RELEASE_OUTCOME_SETTINGS = dict(second_floor_m=1.224,
    third_floor_m=1.375, second_scale_m=.025, third_scale_m=.040,
    second_max_cost=256., third_max_cost=512.)
REFERENCE_TIMING_SETTINGS = dict(onset_low_s=.110, full_low_s=.120,
    full_high_s=.160, onset_high_s=.170, reference_vz_floor_m_s=0.,
    velocity_scale_m_s=.25, weight_per_s=6400.)
NATIVE_THIRD_FOCUS_SETTINGS = dict(previous_hits=2, additional_native_cost_gain=1.)
NATIVE_REFERENCE_PREPARATION_SETTINGS = dict(second_restore_gain=1., third_restore_gain=3.)
NATIVE_REFERENCE_GAIN6_SETTINGS = dict(second_restore_gain=1., third_restore_gain=6.)
THIRD_RETURN_PREPARATION_SETTINGS = dict(return_ceiling_m=1.040,
    height_scale_m=.020, onset_s=.180, onset_ramp_s=.030,
    fade_low_s=.100, fade_high_s=.140, weight_per_s=4500.)
PROJECTED_RETURN_PREPARATION_SETTINGS = dict(gravity_m_s2=9.81,
    ball_radius_m=.020, return_ceiling_m=1.040, height_scale_m=.020,
    onset_s=.260, onset_ramp_s=.040, deadline_s=.140,
    fade_low_s=.140, fade_high_s=.180, weight_per_s=800.)
STARTUP_LEAD_SETTINGS = dict(onset_s=.180, ramp_start_s=.140,
    full_ready_lead_s=.060, onset_ramp_s=.030, return_speed_m_s=.4,
    velocity_scale_m_s=.25, weight_per_s=600.)
SHARED_PREPARATION_SETTINGS = {**STARTUP_LEAD_SETTINGS,
    'weight_per_s': 100., 'velocity_margin_m_s': .05}
SHARED_EARLY_SETTINGS = {**SHARED_PREPARATION_SETTINGS,
    'onset_s': .220, 'ramp_start_s': .180, 'full_ready_lead_s': .100,
    'weight_per_s': 600.}
STARTUP_TRANSFER_SETTINGS = dict(late_apex_lower_m=1.40)
STARTUP_RELEASE_SETTINGS = dict(window_s=.120, clearance_m=.060,
    lookahead_s=.060, relative_speed_scale_m_s=.25,
    apex_scale_m=.030, apex_reserve_m=.005,
    separation_weight_per_s=300., energy_weight_per_s=300.)
STARTUP_PREPARATION_SETTINGS = dict(onset_s=.180, ramp_start_s=.140,
    full_ready_lead_s=.060, onset_ramp_s=.030, return_speed_m_s=.4,
    velocity_scale_m_s=.25, weight_per_s=200., shape_cap=4.)
CONTACT_QUALITY_SETTINGS = dict(preimpact_window_s=.120,
    horizontal_soft_m_s=.040, horizontal_scale_m_s=.040,
    horizontal_weight_per_s=600., second_confirmation_potential=256.,
    third_confirmation_potential=768.,
    acquisition_discount=PPO_EXPECTED['gamma'], late_physical_failure_cost=400.)


def startup_release_terms(metrics: dict, previous_hits, dt_s: float, *, xp,
                          transfer_weak_late: bool = False):
    """Price actual release energy and early ascending-flight clearance.

    Negative dense costs only, with no action labels or joint targets. The
    first launch, contact frames and terminal steps retain their reward.
    The default preserves all steady motion; explicit weak-late transfer
    adds costs only below the existing1.40m quality floor. A clearance-speed
    margin discourages following an
    ascending near-racket ball, including after pending clears. The native
    scheduled apex remains the energy goal; no early recentering is required.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('V322 release reward requires an SI timestep')
    s = STARTUP_RELEASE_SETTINGS
    early_free = ((previous_hits >= 1) & (previous_hits <= 2)
        & (metrics['in_contact'] == 0.) & (metrics['terminated'] == 0.))
    pending = metrics['v322_pending_after'] > 0.
    timely = metrics['v322_pending_age_s'] <= s['window_s']
    gap = metrics['ball_z'] - metrics['racket_z']
    required = xp.maximum(s['clearance_m'] - gap, 0.) / s['lookahead_s']
    relative_vz = metrics['ball_vz'] - metrics['racket_contact_point_vz']
    speed_error = xp.maximum(required - relative_vz, 0.) / s['relative_speed_scale_m_s']
    separation = -s['separation_weight_per_s'] * dt_s * (xp.sqrt(1. + speed_error**2) - 1.)
    upward = xp.maximum(metrics['ball_vz'], 0.)
    predicted_apex = metrics['ball_z'] + upward**2 / (
        2. * xp.maximum(metrics['v322_gravity_m_s2'], 1e-6))
    eligible = early_free
    if transfer_weak_late:
        eligible = eligible | ((previous_hits >= 3)
            & (metrics['in_contact'] == 0.) & (metrics['terminated'] == 0.)
            & (predicted_apex < STARTUP_TRANSFER_SETTINGS['late_apex_lower_m']))
    active = eligible & pending & timely
    target = metrics['fixed_contact_limit_cycle_target_apex_z_m'] + s['apex_reserve_m']
    if transfer_weak_late:
        target = xp.where(previous_hits >= 3,
            STARTUP_TRANSFER_SETTINGS['late_apex_lower_m'], target)
    energy_error = xp.maximum(target - predicted_apex, 0.) / s['apex_scale_m']
    energy = -s['energy_weight_per_s'] * dt_s * (xp.sqrt(1. + energy_error**2) - 1.)
    return {
        'reward/v322_startup_separation_penalty': xp.where(
            eligible & (~pending | timely) & (gap < s['clearance_m'])
            & (metrics['ball_vz'] > 0.), separation, 0.),
        'reward/v322_startup_release_energy_penalty': xp.where(active, energy, 0.),
    }


def contact_quality_terms(metrics: dict, previous_hits, dt_s: float, *, hit_count, xp):
    """Dense preimpact braking and honest acquisition/failure credit, (N,).

    The native approach population excludes ascending/support motion. Its
    center-point XY speed matches the existing hit-speed statistic. Add a
    soft-margin cost over the last 120 ms after acquisition, with a quadratic
    onset and a continuous linear tail. Leave early lateral recovery free.
    Actual second/third confirmations advance a potential regardless of
    centering quality. Pay gamma*Phi(next)-Phi(previous), with terminal Phi=0,
    and Phi=0 again at hit four. This telescopes under the unchanged discount;
    repeated startup/reset cannot farm net extra acquisition reward. Existing
    physical outcome costs and gates still apply.
    Every late physical failure pays 400 (40/s times a 10s reward horizon).
    No time-limit truncation pays that cost. Nothing filters data or changes
    any event, reset, action, termination, transition or learner setting.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('V322 contact quality requires an SI control timestep')
    settings = CONTACT_QUALITY_SETTINGS
    ttc = metrics['intercept_time_true']
    terminal = metrics['terminated'] > 0.
    late = previous_hits >= 3
    active = (late & ~terminal & (metrics['approach_racket_vxy_active'] > 0.)
              & (ttc > 0.) & (ttc <= settings['preimpact_window_s']))
    urgency = xp.clip(1. - ttc / settings['preimpact_window_s'], 0., 1.) ** 2
    excess = xp.maximum(metrics['approach_racket_vxy_m_s']
        - settings['horizontal_soft_m_s'], 0.) / settings['horizontal_scale_m_s']
    shape = xp.where(excess <= 2., excess ** 2, 4. * excess - 4.)
    braking = xp.where(active,
        -settings['horizontal_weight_per_s'] * dt_s * urgency * shape, 0.)
    def potential(count):
        return xp.where(count == 2, settings['second_confirmation_potential'],
            xp.where(count == 3, settings['third_confirmation_potential'], 0.))

    previous = potential(previous_hits)
    following = xp.where(terminal, 0., potential(hit_count))
    credit = settings['acquisition_discount'] * following - previous
    failure = xp.where(late & terminal, -settings['late_physical_failure_cost'], 0.)
    return {
        'reward/v322_preimpact_xy_margin_penalty': braking,
        'reward/v322_acquisition_progress_shaping': credit,
        'reward/v322_late_physical_failure_penalty': failure,
    }


def startup_preparation_correction(metrics: dict, previous_hits, dt_s: float, *, xp):
    """Reprice only native near-side second/third physical preparation, (N,).

    Reward readiness 60 ms before impact instead of only at impact. Keep the
    native ballistic energy target, TTC, support, physical state and event
    definitions. Replace one already-present physical cost; never add it twice.
    The bounded auxiliary loss is at most 800*dt per active lane, not a sample
    filter, policy target, actuator change or new terminal rule.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('V322 preparation requires an SI control timestep')
    settings = STARTUP_PREPARATION_SETTINGS
    ttc = metrics['v304_ttc']
    support = metrics['v309_support']

    def smooth(value):
        bounded = xp.clip(value, 0., 1.)
        return bounded * bounded * (3. - 2. * bounded)

    returning = -settings['return_speed_m_s'] * smooth(
        (ttc - settings['ramp_start_s'])
        / (settings['onset_s'] - settings['ramp_start_s']))
    ready = metrics['v304_required_vz'] * smooth(
        (settings['ramp_start_s'] - ttc)
        / (settings['ramp_start_s'] - settings['full_ready_lead_s']))
    floor = xp.where(ttc >= settings['ramp_start_s'], returning, ready)
    shortfall = xp.maximum(floor - metrics['racket_contact_point_vz'], 0.)
    shape = xp.sqrt(1. + (shortfall / settings['velocity_scale_m_s']) ** 2) - 1.
    urgency = smooth((settings['onset_s'] - ttc) / settings['onset_ramp_s'])
    active = ((previous_hits >= 1) & (previous_hits <= 2)
              & (metrics['v304_active'] > 0.) & (support > 0.)
              & (metrics['terminated'] == 0.))
    desired = -settings['weight_per_s'] * dt_s * support * urgency * xp.minimum(
        shape, settings['shape_cap'])
    target = xp.where(active, desired, 0.)
    correction = xp.where(active, desired - metrics['v304_physical_cost'], 0.)
    return correction, target


def startup_lead_penalty(metrics: dict, previous_hits, dt_s: float, *, xp):
    """Price only earlier near-start readiness above native M11's loss, (N,).

    Native V304/V311 physical cost is never replaced or capped. Add the
    nonnegative pseudo-Huber shortfall difference between an earlier floor
    and the exact native floor. Thus the extra timing price tends to zero
    at physical contact, where the original physical cost still applies.
    Only native descending preparation at previous-count1/2 is eligible.
    No action, reference, reset, transition or observation is prescribed.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('V322 startup lead requires an SI timestep')
    s = STARTUP_LEAD_SETTINGS
    ttc = metrics['v304_ttc']
    required = metrics['v304_required_vz']

    def smooth(value):
        x = xp.clip(value, 0., 1.)
        return x*x*(3.-2.*x)

    native_floor = xp.where(ttc >= .100,
        -.4*smooth((ttc-.100)/.080), required*smooth((.100-ttc)/.100))
    early_floor = xp.where(ttc >= s['ramp_start_s'],
        -s['return_speed_m_s']*smooth((ttc-s['ramp_start_s'])
            / (s['onset_s']-s['ramp_start_s'])),
        required*smooth((s['ramp_start_s']-ttc)
            / (s['ramp_start_s']-s['full_ready_lead_s'])))
    velocity = metrics['racket_contact_point_vz']

    def shape(floor):
        shortfall = xp.maximum(floor-velocity, 0.)/s['velocity_scale_m_s']
        return xp.sqrt(1.+shortfall**2)-1.

    additional = xp.maximum(shape(early_floor)-shape(native_floor), 0.)
    active = ((previous_hits >= 1) & (previous_hits <= 2)
        & (metrics['v304_active'] > 0.) & (metrics['v309_support'] > 0.)
        & (metrics['terminated'] == 0.))
    urgency = smooth((s['onset_s']-ttc)/s['onset_ramp_s'])
    return xp.where(active, -s['weight_per_s']*dt_s
        * metrics['v309_support']*urgency*additional, 0.)


def shared_preparation_penalty(metrics: dict, previous_hits, dt_s: float, *, xp,
                               earlier: bool = False):
    """Train the same physical readiness during ongoing descending flights.

    Use the authenticated native crossing TTC and executable vertical-speed
    goal, without recomputing physics or adding an actor input/action label.
    Apply only after startup, outside contact/pending/terminal states. A
    0.05m/s dead band leaves adequate preparation unpriced; the 100/s cost
    teaches vertical readiness from the plentiful long-episode population.
    It is a reward proxy, not an applied target or a promise of transfer.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('V322 shared preparation requires an SI timestep')
    s = SHARED_EARLY_SETTINGS if earlier else SHARED_PREPARATION_SETTINGS
    ttc = metrics['fixed_contact_time_to_contact_s']
    required = metrics['fixed_plane_limit_cycle_executable_racket_vz_target_m_s']

    def smooth(value):
        x = xp.clip(value, 0., 1.)
        return x*x*(3.-2.*x)

    floor = xp.where(ttc >= s['ramp_start_s'],
        -s['return_speed_m_s']*smooth((ttc-s['ramp_start_s'])
            / (s['onset_s']-s['ramp_start_s'])),
        required*smooth((s['ramp_start_s']-ttc)
            / (s['ramp_start_s']-s['full_ready_lead_s'])))
    shortfall = xp.maximum(floor-metrics['racket_contact_point_vz']
        - s['velocity_margin_m_s'], 0.)/s['velocity_scale_m_s']
    shape = xp.sqrt(1.+shortfall**2)-1.
    urgency = smooth((s['onset_s']-ttc)/s['onset_ramp_s'])
    active = ((previous_hits >= 3) & (metrics['fixed_contact_crossing_valid'] > 0.)
        & (metrics['v322_pending_after'] == 0.) & (metrics['in_contact'] == 0.)
        & (metrics['terminated'] == 0.) & (metrics['ball_vz'] < 0.)
        & (ttc > 0.) & (ttc < s['onset_s']))
    return xp.where(active, -s['weight_per_s']*dt_s*urgency*shape, 0.)


def native_third_focus_penalty(metrics: dict, previous_hits, *, xp):
    """Add one native third-impact physical cost, already time/Y weighted.

    Successful historical third preparations nearly satisfy the native floor;
    demanding full impact velocity60ms early also penalizes those successes.
    Reuse the original V304/V311 cost and eligibility instead. No new timing,
    target, action label, contact event or state is introduced. The first
    launch, second preparation, terminal frames and ongoing flight are exact.
    """
    settings = NATIVE_THIRD_FOCUS_SETTINGS
    active = ((previous_hits == settings['previous_hits'])
        & (metrics['v304_active'] > 0.) & (metrics['v309_support'] > 0.)
        & (metrics['terminated'] == 0.))
    return xp.where(active, settings['additional_native_cost_gain']
        * metrics['v304_physical_cost'], 0.)


def native_reference_preparation_penalty(metrics: dict, previous_hits, *, xp,
                                         third_restore_gain: float | None = None):
    """Reprice the removed native command-height readiness cost, (N,).

    V309 removes clean_support of the native V304 reference cost; V311
    restores only its physical velocity cost. Reuse the existing command-pose
    height-lead metric at native previous-count1/2, without applying a joint
    target or changing observations. Native TTC/lead/fade/dt/Y weights are
    already inside reference_cost. Multiply by clean_support once to restore
    that removed share, with phase gains1/3. Adequate readiness costs zero.
    """
    settings = NATIVE_REFERENCE_PREPARATION_SETTINGS
    third_gain = (settings['third_restore_gain'] if third_restore_gain is None
                  else third_restore_gain)
    if not math.isfinite(third_gain) or third_gain <= 0.:
        raise ValueError('Native preparation reward gain must be finite and positive')
    gain = xp.where(previous_hits == 1, settings['second_restore_gain'],
                    third_gain)
    active = (((previous_hits == 1) | (previous_hits == 2))
        & (metrics['v304_active'] > 0.) & (metrics['v309_support'] > 0.)
        & (metrics['terminated'] == 0.))
    return xp.where(active, gain * metrics['v309_support']
        * metrics['v304_reference_cost'], 0.)


def third_return_preparation_penalty(metrics: dict, previous_hits, dt_s: float,
                                     *, old_reference_penalty, xp):
    """Reprice high third-preparation states without rewarding a missed return.

    Inputs are native (N,) metrics and the already weighted gain6 reference
    penalty. Return readiness uses actual racket height, not an applied
    position/action target. The total preparation loss is the maximum of the
    old reference loss and a return-aware blend. Thus the correction cannot
    add credit, while physical return replaces premature upward preparation
    where its loss dominates. Earlier launches, terminals and ongoing flight
    retain their exact rewards; no data or transition is rejected.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('Third-return reward requires an SI timestep')
    s = THIRD_RETURN_PREPARATION_SETTINGS

    def smooth(value):
        bounded = xp.clip(value, 0., 1.)
        return bounded*bounded*(3.-2.*bounded)

    ttc = metrics['v304_ttc']
    height_error = xp.maximum(metrics['racket_z']-s['return_ceiling_m'], 0.) / s['height_scale_m']
    urgency = smooth((s['onset_s']-ttc)/s['onset_ramp_s']) * smooth(
        (ttc-s['fade_low_s'])/(s['fade_high_s']-s['fade_low_s']))
    mode = smooth(height_error)*urgency
    old_loss = xp.maximum(-old_reference_penalty, 0.)
    return_loss = (s['weight_per_s']*dt_s*metrics['v309_support']*urgency
                   * (xp.sqrt(1.+height_error**2)-1.))
    active = ((previous_hits == 2) & (metrics['v304_active'] > 0.)
        & (metrics['v309_support'] > 0.) & (metrics['terminated'] == 0.))
    return xp.where(active, -xp.maximum(return_loss-mode*old_loss, 0.), 0.)


def projected_return_preparation_penalty(metrics: dict, previous_hits, dt_s: float,
                                        *, xp):
    """Score early third-flight height/rate readiness before native180ms cues.

    Native V322 gravity and radius are fixed at9.81m/s^2 and20mm. Reconstruct
    its ball-only held-racket TTC without the native180ms eligibility mask.
    A constant-rate projection to140ms is only a reward proxy; it never
    predicts/changes the actuator, applies a target, or changes observations.
    Returning early costs zero once adequate, with no positive extra credit.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('Projected-return reward requires an SI timestep')
    s = PROJECTED_RETURN_PREPARATION_SETTINGS

    def smooth(value):
        bounded = xp.clip(value, 0., 1.)
        return bounded*bounded*(3.-2.*bounded)

    gap = metrics['ball_z']-metrics['racket_z']-s['ball_radius_m']
    vz = metrics['ball_vz']
    ttc = (vz+xp.sqrt(xp.maximum(vz*vz
        + 2.*s['gravity_m_s2']*xp.maximum(gap, 0.), 0.))) / s['gravity_m_s2']
    urgency = smooth((s['onset_s']-ttc)/s['onset_ramp_s']) * smooth(
        (ttc-s['fade_low_s'])/(s['fade_high_s']-s['fade_low_s']))
    projected_z = (metrics['racket_z'] + metrics['racket_contact_point_vz']
                   * xp.maximum(ttc-s['deadline_s'], 0.))
    error = xp.maximum(projected_z-s['return_ceiling_m'], 0.)/s['height_scale_m']
    active = ((previous_hits == 2) & (metrics['v309_support'] > 0.)
        & (metrics['terminated'] == 0.) & (metrics['in_contact'] == 0.)
        & (metrics['v322_pending_after'] == 0.) & (gap > 0.) & (ttc > 0.))
    return xp.where(active, -s['weight_per_s']*dt_s*metrics['v309_support']
        * urgency*(xp.sqrt(1.+error*error)-1.), 0.)


def acquisition_xy_credit_correction(metrics: dict, previous_hits, *, xp):
    """Broaden only native V310 third-acquisition XY credit, within its budget.

    Metrics and counts are (N,) arrays. Preserve the original V309 clean
    floor, native event/support eligibility and later failure clawbacks. No
    credit is added on first/second hits, ongoing flight or terminal frames.
    Native ball velocity is the position difference over one physical step.
    """
    s = ACQUISITION_XY_SETTINGS
    gravity = xp.maximum(metrics['v322_gravity_m_s2'], 1e-6)
    apex = metrics['ball_z'] + xp.maximum(metrics['ball_vz'], 0.)**2/(2.*gravity)
    vxy_sq = metrics['ball_vx']**2 + metrics['ball_vy']**2
    quality = xp.exp(-.5*((apex-s['apex_target_m'])/s['apex_sigma_m'])**2
                     -.5*vxy_sq/s['xy_sigma_m_s']**2)
    extra_budget = xp.maximum(metrics['v310_legacy_budget']
        - s['preserved_clean_floor_budget']*metrics['v309_support'], 0.)
    correction = xp.maximum(quality*extra_budget-metrics['v310_extra_credit'], 0.)
    active = ((previous_hits == 2) & (metrics['v310_active'] > 0.)
        & (metrics['v309_support'] > 0.) & (metrics['terminated'] == 0.))
    return xp.where(active, correction, 0.)


def weak_release_outcome_penalty(metrics: dict, previous_hits, *, xp):
    """Price native weak-clearance events before the sparse confirmed credit.

    Reuse the unchanged native low-survival/subfloor event flags. Only clean
    near-side second/third attempts incur a bounded nonpositive energy loss.
    Floors are the serialized M11 quality-confirmation floors, not new gates.
    The event is already resolved by the simulator; no new state, event,
    count, timing window, action target or positive credit is introduced.
    """
    s = WEAK_RELEASE_OUTCOME_SETTINGS
    second = previous_hits == 1
    floor = xp.where(second, s['second_floor_m'], s['third_floor_m'])
    scale = xp.where(second, s['second_scale_m'], s['third_scale_m'])
    weight = xp.where(second, s['second_max_cost'], s['third_max_cost'])
    apex = metrics['ball_z'] + xp.maximum(metrics['ball_vz'], 0.)**2 / (
        2.*xp.maximum(metrics['v322_gravity_m_s2'], 1e-6))
    deficit = xp.maximum(floor-apex, 0.)/scale
    shape = deficit**2/(1.+deficit**2)
    weak_event = ((metrics['low_survival_launch'] > 0.)
                  | (metrics['subfloor_launch'] > 0.))
    active = (((previous_hits == 1) | (previous_hits == 2)) & weak_event
        & (metrics['v309_support'] > 0.) & (metrics['terminated'] == 0.))
    return xp.where(active, -weight*metrics['v309_support']*shape, 0.)


def reference_timing_active(metrics: dict, previous_hits, *, xp):
    """Retain native third-preparation support inside the calibrated clock."""
    s = REFERENCE_TIMING_SETTINGS
    ttc = metrics['v304_ttc']
    return ((previous_hits == 2) & (metrics['v304_active'] > 0.)
        & (metrics['v309_support'] > 0.) & (metrics['terminated'] == 0.)
        & (metrics['in_contact'] == 0.) & (metrics['v322_pending_after'] == 0.)
        & (metrics['ball_vz'] < 0.) & (ttc > s['onset_low_s'])
        & (ttc < s['onset_high_s']))


def reference_timing_penalty(metrics: dict, previous_hits, dt_s: float, *, xp):
    """Price late reversal of the published reference, not physical motion.

    Successful baseline strokes reverse the reference while the actual
    racket is still returning. Reward only the reference's upward direction
    in that native preparation window; retain the original physical floor,
    every event reward and all motion-quality settings unchanged.
    """
    if not math.isfinite(dt_s) or not 0. < dt_s < .1:
        raise ValueError('Reference timing reward requires an SI timestep')
    s = REFERENCE_TIMING_SETTINGS
    ttc = metrics['v304_ttc']

    def smooth(value):
        value = xp.clip(value, 0., 1.)
        return value*value*(3.-2.*value)

    urgency = smooth((ttc-s['onset_low_s'])/(s['full_low_s']-s['onset_low_s'])) * smooth(
        (s['onset_high_s']-ttc)/(s['onset_high_s']-s['full_high_s']))
    shortfall = xp.maximum(s['reference_vz_floor_m_s']-metrics['v322_reference_vz_m_s'], 0.)
    error = shortfall/s['velocity_scale_m_s']
    cost = -s['weight_per_s']*dt_s*metrics['v309_support']*urgency*(xp.sqrt(1.+error**2)-1.)
    return xp.where(reference_timing_active(metrics,previous_hits,xp=xp),cost,0.)


def reward_adjustments(metrics: dict, preset: str, zero, *, hit_count=None,
                       previous_hits=None, dt_s: float = .005, xp=None):
    """Return native-cost differences only, without editing native telemetry.

    Inputs and outputs are (N,) arrays, NumPy or JAX. Native event populations
    and timing are retained, including weak physical releases and failures.
    No reward change touches the physical transition, gates or observations.
    """
    if preset not in REWARD_PRESETS:
        raise ValueError(f'Unknown V322 reward preset: {preset}')
    if preset in ('late_cycle_repair', 'late_center_precision'):
        from fixed_plane_v322_late_cycle_reward import (
            late_cycle_reward_terms, LATE_CENTER_PRECISION_SETTINGS,
        )
        correction, terms = reward_adjustments(metrics, 'reference_timing', zero,
            hit_count=hit_count, previous_hits=previous_hits, dt_s=dt_s, xp=xp)
        extra = late_cycle_reward_terms(metrics, previous_hits, xp=xp,
            settings=LATE_CENTER_PRECISION_SETTINGS if preset == 'late_center_precision' else None)
        return correction + sum(extra.values(), zero), {**terms, **extra}
    if REWARD_PRESETS[preset] and hit_count is None:
        raise ValueError('V322 stability rewards require the native hit count')
    late = True if hit_count is None else hit_count >= 4
    terms = {f'reward/v322_{name}_correction': (scale - 1.) * metrics[f'reward/{name}'] * late
             for name, scale in REWARD_PRESETS[preset].items()}
    if preset in ('stability_launch', 'contact_quality', 'balanced_contact', 'startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        if previous_hits is None or xp is None:
            raise ValueError('V322 preparation requires native previous counts and array operations')
        if preset not in ('native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
            correction, _ = startup_preparation_correction(metrics, previous_hits, dt_s, xp=xp)
            terms['reward/v322_startup_preparation_correction'] = correction
    if preset in ('contact_quality', 'balanced_contact', 'startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms.update(contact_quality_terms(metrics, previous_hits, dt_s,
                                          hit_count=hit_count, xp=xp))
    if preset in ('startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms.update(startup_release_terms(metrics, previous_hits, dt_s, xp=xp,
            transfer_weak_late=preset in ('startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing')))
    if preset in ('startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        near_early = ((previous_hits >= 1) & (previous_hits <= 2)
                      & (metrics['v309_support'] > 0.))
        terms['reward/v322_near_release_energy_correction'] = xp.where(
            near_early, -terms['reward/v322_startup_release_energy_penalty'], 0.)
        terms['reward/v322_startup_lead_penalty'] = (zero if preset in ('native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing')
            else startup_lead_penalty(metrics, previous_hits, dt_s, xp=xp))
    if preset in ('native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms['reward/v322_native_third_focus_penalty'] = (zero if preset in ('native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing')
            else native_third_focus_penalty(metrics, previous_hits, xp=xp))
    if preset in ('native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms['reward/v322_native_reference_preparation_penalty'] = native_reference_preparation_penalty(
            metrics, previous_hits, xp=xp,
            third_restore_gain=(NATIVE_REFERENCE_GAIN6_SETTINGS['third_restore_gain']
                                if preset in ('native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else None))
    if preset == 'native_return_prepare':
        terms['reward/v322_third_return_preparation_penalty'] = third_return_preparation_penalty(
            metrics, previous_hits, dt_s,
            old_reference_penalty=terms['reward/v322_native_reference_preparation_penalty'], xp=xp)
    if preset in ('projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms['reward/v322_projected_return_preparation_penalty'] = projected_return_preparation_penalty(
            metrics, previous_hits, dt_s, xp=xp)
    if preset in ('shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms['reward/v322_shared_preparation_penalty'] = shared_preparation_penalty(
            metrics, previous_hits, dt_s, xp=xp,
            earlier=preset in ('shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'))
    if preset in ('projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        terms['reward/v322_acquisition_xy_credit_correction'] = acquisition_xy_credit_correction(
            metrics, previous_hits, xp=xp)
    if preset in ('weak_release_outcome', 'reference_timing'):
        terms['reward/v322_weak_release_outcome_penalty'] = weak_release_outcome_penalty(
            metrics, previous_hits, xp=xp)
    if preset == 'reference_timing':
        terms['reward/v322_reference_timing_penalty'] = reference_timing_penalty(
            metrics, previous_hits, dt_s, xp=xp)
    return sum(terms.values(), zero), terms


def validate_reward_resume(contract: dict, preset: str, *, allow_change: bool) -> None:
    """Require an explicit opt-in before continuing across reward definitions."""
    previous = contract.get('reward_preset', 'baseline')
    if preset not in REWARD_PRESETS or previous not in REWARD_PRESETS:
        raise ValueError('Unknown V322 reward preset in resume contract')
    if preset != previous and not allow_change:
        raise ValueError('V322 resume requires an explicit reward change acknowledgement')


def restore_nested_config(cfg, predictor_cls, target_cls):
    """Undo the historical subclass loader's asdict flattening, value exactly."""
    changes = {}
    for name, cls in (('fixed_contact_high_ball_predictor', predictor_cls),
                      ('fixed_contact_target', target_cls)):
        value = getattr(cfg, name)
        if isinstance(value, dict):
            changes[name] = cls(**value)
        elif not isinstance(value, cls):
            raise ValueError(f'Invalid M11 nested configuration: {name}')
    return replace(cfg, **changes)


def make_config(parent):
    """Retain every M11 field except the declared horizon/reward-cap fields."""
    if not getattr(parent, 'v317_late_ball_xy_margin_gain_enabled', False):
        raise ValueError('V322 requires the M11 late horizontal-velocity reward')
    return make_long_config(parent)


def should_validate(stage_index: int, update_idx: int, last_eval_update: int) -> bool:
    """Only the last lesson pauses for a short frozen-policy acceptance test."""
    return stage_index == len(COURSE)-1 and update_idx-last_eval_update >= EVAL_RETRY_UPDATES


def quality_passed(q: dict) -> bool:
    """M11-compatible long-run tolerances, retaining its low horizontal motion."""
    limits = dict(racket_vxy_rms_m_s=.055, racket_omega_rms_rad_s=.28,
                  center_m=.0065, adjacent_jump_m=.015,
                  recurrent_vxy_m_s=.05, recurrent_vxy_rms_m_s=.055,
                  ball_vxy_m_s=.030, ball_vxy_rms_m_s=.035)
    return (q.get('steady_events', 0.) >= 100
            and 1.40 <= q.get('apex_m', 0.) <= 1.45
            and q.get('in_band', 0.) >= .98
            and all(math.isfinite(q.get(k, math.inf)) and q[k] <= limit
                    for k, limit in limits.items()))
