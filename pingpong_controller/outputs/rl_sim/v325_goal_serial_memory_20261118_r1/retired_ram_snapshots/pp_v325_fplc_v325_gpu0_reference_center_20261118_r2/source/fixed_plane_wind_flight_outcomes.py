"""Opt-in measured wind-flight outcomes; no physics, observations or hit redefinition."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

import numpy as np


HEIGHT_FIELDS = tuple('wind_height/' + name for name in (
    'active_samples', 'band_events', 'in_band_events', 'apex_events', 'apex_sum_m',
    'pair_events', 'pair_jump_sum_m', 'incomplete_events', 'censored_events'))
Tracker = namedtuple('WindFlightTracker',
    'active eligible released peak_z last_peak_z last_peak_valid gain domain_s')


def flight_quality_summary(sums: dict) -> dict:
    """Extend quality only in the explicit mode; never edit authenticated M12/M13 code."""
    from fixed_plane_v321_long_episode import quality_summary
    result = quality_summary(sums)
    if sums.get('wind_height/active_samples', 0.) > 0.:
        for name, numerator, denominator in (
                ('in_band', 'wind_height/in_band_events', 'wind_height/band_events'),
                ('apex_m', 'wind_height/apex_sum_m', 'wind_height/apex_events'),
                ('adjacent_jump_m', 'wind_height/pair_jump_sum_m', 'wind_height/pair_events')):
            if sums.get(denominator, 0.) > 0.:
                result[name] = sums[numerator] / sums[denominator]
            else:
                result.pop(name, None)
    return result


def initial_tracker(n: int, *, xp: Any) -> Tracker:
    zero = xp.zeros(n, dtype=xp.float32)
    flag = xp.zeros(n, dtype=xp.bool_)
    return Tracker(flag, flag, flag, zero, zero, flag, zero, zero)


def observe_flight(state: Tracker, *, edge, release, contact, z_m, vz_m_s,
                   terminated, truncated, eligible, gain, domain_s, xp: Any):
    """Count a physical apex once; charge unresolved failures, censor time limits.

    All arrays are [N]. A contact trial starts at a native physical edge and
    becomes a flight on detachment. The peak uses actual200Hz ball positions.
    Five1ms native ticks remain unchanged; smooth ballistic peak undersampling
    is at most approximately0.031mm. No future wind or projected apex is read.
    """
    same = state.domain_s == domain_s
    active = state.active & same
    released = state.released & same
    peak = xp.maximum(state.peak_z, z_m)
    apex = active & released & ~contact & (vz_m_s <= 0.)
    unresolved = active & (edge | terminated | (~contact & ~released & (vz_m_s <= 0.)))
    valid = apex & ~edge & ~terminated
    event = active & state.eligible & (valid | unresolved)
    in_band = event & valid & (peak >= 1.40) & (peak <= 1.45)
    pairs = event & valid & state.last_peak_valid & same
    censored = active & truncated & ~event
    closed = apex | unresolved | truncated
    begin = edge | (release & ~active)
    next_state = Tracker(
        xp.where(begin, True, active & ~closed) & ~terminated & ~truncated,
        xp.where(begin, eligible, state.eligible),
        xp.where(begin, release, released | release),
        xp.where(begin, z_m, peak),
        xp.where(event & valid, peak, state.last_peak_z),
        xp.where(event, valid, state.last_peak_valid & same) & ~terminated & ~truncated,
        xp.where(release, gain, xp.where(begin, 0., state.gain)), domain_s)
    error = xp.abs(peak - 1.425) / .010
    huber = .5 * xp.minimum(error, 1.) ** 2 + xp.maximum(error - 1., 0.)
    # Output-velocity remains the primary feedback. This prices actual band
    # misses and model residuals rather than rewarding another hit counter.
    cost = xp.where(valid, 2. * xp.minimum(huber, 4.), 0.) + 8. * (~in_band)
    result = dict(band_event=event.astype(xp.float32),
        in_band=in_band.astype(xp.float32), apex_event=(event & valid).astype(xp.float32),
        apex_m=xp.where(event & valid, peak, 0.), pair_event=pairs.astype(xp.float32),
        pair_jump_m=xp.where(pairs, xp.abs(peak - state.last_peak_z), 0.),
        incomplete=(event & ~valid).astype(xp.float32), censored=censored.astype(xp.float32),
        reward=-state.gain * event * cost)
    return next_state, result


def alignment_contract() -> dict:
    return dict(revision='observed_wind_flight_apex_v1',
        height_band_m=[1.40, 1.45], target_m=1.425, min_band_fraction='unchanged99%/98%',
        height='observed free-flight maximum, not vacuum ballistic projection',
        weak_contacts='physical edge trials; failed/unreleased trials count as band misses',
        time_limit='censored if the flight apex has not been observed',
        clean='native projected height and native adjacent-height gates unchanged',
        onset='first5 confirmed-hit shelter; released-flight wind gain gates new costs',
        velocity_sigma_m_s=[.035, .035, .030], measured_apex_sigma_m=.010,
        measured_apex_huber_weight=2., band_miss_cost=8.,
        impact='native200Hz physical contact edge,5ms interval; incoming from preceding step',
        impact_logs='event-weighted physical-edge and confirmed-hit normal error; clean/wind/onset',
        cycle='fixed return point and existing outgoing/cadence reference retained; incoming/time diagnostics',
        cycle_cost=.25, actor_critic_action=[114, 453, 7], physics='unchanged')


@lru_cache(maxsize=4)
def _config_type(parent: type) -> type:
    return make_dataclass('WindFlightOutcomeConfig',
        [('wind_flight_alignment', bool, field(default=True))], bases=(parent,), frozen=True)


@lru_cache(maxsize=4)
def _precision_config_type(parent: type) -> type:
    return make_dataclass('WindTailPrecisionConfig',
        [('wind_tail_precision', bool, field(default=True))], bases=(parent,), frozen=True)


@lru_cache(maxsize=4)
def _recovery_config_type(parent: type) -> type:
    return make_dataclass('WindRecoverySafetyConfig',
        [('wind_recovery_safety', bool, field(default=True))], bases=(parent,), frozen=True)


@lru_cache(maxsize=4)
def _full_flight_config_type(parent: type) -> type:
    return make_dataclass('WindFullFlightRecoveryConfig',
        [('wind_full_flight_recovery', bool, field(default=True))], bases=(parent,), frozen=True)


@lru_cache(maxsize=4)
def _lower_guard_config_type(parent: type) -> type:
    return make_dataclass('WindLowerStopGuardConfig',
        [('wind_lower_stop_guard', bool, field(default=True))], bases=(parent,), frozen=True)


def with_flight_config(cfg, enabled: bool, *, tail_precision: bool = False,
                       recovery_safety: bool = False, full_flight: bool = False,
                       lower_guard: bool = False):
    if lower_guard and not full_flight:
        raise ValueError('Lower stop guard requires full-flight recovery')
    if full_flight and not recovery_safety:
        raise ValueError('Full-flight recovery requires the explicit recovery mode')
    if recovery_safety and not tail_precision:
        raise ValueError('Recovery safety requires independent vertical precision')
    if tail_precision and not enabled:
        raise ValueError('Tail precision requires observed flight-height alignment')
    result = _config_type(type(cfg))(**vars(cfg), wind_flight_alignment=True) if enabled else cfg
    if tail_precision:
        result = _precision_config_type(type(result))(**vars(result), wind_tail_precision=True)
    if recovery_safety:
        result = _recovery_config_type(type(result))(**vars(result), wind_recovery_safety=True)
    if full_flight:
        result = _full_flight_config_type(type(result))(**vars(result), wind_full_flight_recovery=True)
    return (_lower_guard_config_type(type(result))(**vars(result), wind_lower_stop_guard=True)
            if lower_guard else result)


class FlightState(namedtuple('WindFlightStateBase',
        'core tracker impact_error impact_target_tilt impact_actual_tilt '
        'expected_incoming reference_valid release_time expected_period release_gain')):
    __slots__ = ()

    def __getattr__(self, name):
        return getattr(self.core, name)

    def _replace(self, **kwargs):
        own = {k: v for k, v in kwargs.items() if k in self._fields}
        native = {k: v for k, v in kwargs.items() if k not in self._fields}
        if native:
            own['core'] = own.get('core', self.core)._replace(**native)
        return super()._replace(**own)


def initialize_state(core):
    import jax.numpy as jnp
    n = core.step_count.shape[0]
    zero = jnp.zeros(n, jnp.float32)
    tracker = initial_tracker(n, xp=jnp)._replace(domain_s=core.wind_sampled_s)
    return FlightState(core, tracker, zero, zero, zero, jnp.zeros((n, 3), jnp.float32),
                       jnp.zeros(n, jnp.bool_), zero, zero, zero)


def flight_environment(inner_class):
    """Wrap the native wind state inside camera history; all sensor prefixes retain parity."""
    class FlightEnv:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def attach_training_state(self, core):
            return core if isinstance(core, FlightState) else initialize_state(core)

        def reset(self, keys):
            core, obs = self.inner.reset(keys)
            return initialize_state(core), obs

        def reset_done(self, state, obs, done, keys):
            import jax
            import jax.numpy as jnp
            core, obs = self.inner.reset_done(state.core, obs, done, keys)
            fresh = initialize_state(core)
            select = lambda new, old: jnp.where(done.reshape(
                (done.shape[0],) + (1,) * (new.ndim - 1)), new, old)
            fields = jax.tree.map(select, tuple(fresh)[1:], tuple(state)[1:])
            return FlightState(core, *fields), obs

        def get_critic_obs(self, state, obs):
            return self.inner.get_critic_obs(state.core, obs)

        def step(self, state, action):
            import jax.numpy as jnp
            from fixed_plane_wind_dynamics_guidance import ideal_contact_reference
            from fixed_plane_wind_pose import shelter_gain
            core, obs, reward, done, metrics = self.inner.step(state.core, action)
            if not getattr(getattr(self.cfg, 'wind_pose', None), 'strong_wind_reward', False):
                metrics.update({k: jnp.zeros_like(reward) for k in HEIGHT_FIELDS})
                return state._replace(core=core), obs, reward, done, metrics
            before = state.core
            edge = metrics['raw_physical_contact_edge'] > .5
            release = metrics['dynamics/release_event'] > .5
            terminated = metrics['terminated'] > .5
            gain = shelter_gain(before, self.dt, xp=jnp) * (before.wind_mode > 0)
            if getattr(self.cfg, 'wind_full_flight_recovery', False):
                from fixed_plane_wind_recovery_safety import full_flight_recovery_phase, safety_terms
                now = core.step_count * self.dt
                released_at = jnp.where(release, now, state.release_time)
                phase = full_flight_recovery_phase(now - released_at,
                    core.prev_contact, metrics['dynamics/precontact_geometry'] > .5,
                    before.wind_startup_hit_count, xp=jnp) & (released_at > 0.)
                metrics['dynamics/recovery_phase'] = phase.astype(jnp.float32)
                terms = safety_terms(metrics, gain=gain, dt_s=self.dt, xp=jnp)
                reward = reward + sum(terms.values())
                metrics.update(terms)
            if getattr(self.cfg, 'wind_lower_stop_guard', False):
                from fixed_plane_wind_lower_guard import lower_stop_reward
                guard = lower_stop_reward(metrics['racket_z'], core.prev_racket_vel[:, 2],
                    self.cfg.fixed_contact_racket_z_bounds_m[0], gain=gain,
                    confirmed_hits=before.wind_startup_hit_count, terminated=metrics['terminated'],
                    dt_s=self.dt, xp=jnp)
                reward = reward + guard
                metrics['reward/dynamics_lower_stop_guard'] = guard
            tracker, measured = observe_flight(state.tracker, edge=edge, release=release,
                contact=core.prev_contact, z_m=core.data.qpos[:, self.ball_qadr + 2],
                vz_m_s=core.data.qvel[:, self.ball_vadr + 2], terminated=terminated,
                truncated=done & ~terminated, eligible=before.hit_count >= 3,
                gain=gain, domain_s=before.wind_sampled_s, xp=jnp)
            windy = before.wind_mode > 0
            native = {
                'active_samples': jnp.ones_like(reward),
                'band_events': metrics['v92_late_contact_event'],
                'in_band_events': metrics['v92_in_1p4_1p45_event'],
                'apex_events': metrics['rapid_steady_steady_events'],
                'apex_sum_m': metrics['rapid_steady_steady_apex_sum_z_m'],
                'pair_events': metrics['v80_late_pair_event'],
                'pair_jump_sum_m': metrics['v80_late_pair_jump_sum_m'],
                'incomplete_events': jnp.zeros_like(reward),
                'censored_events': jnp.zeros_like(reward)}
            observed = dict(active_samples=jnp.ones_like(reward),
                band_events=measured['band_event'], in_band_events=measured['in_band'],
                apex_events=measured['apex_event'], apex_sum_m=measured['apex_m'],
                pair_events=measured['pair_event'], pair_jump_sum_m=measured['pair_jump_m'],
                incomplete_events=measured['incomplete'], censored_events=measured['censored'])
            metrics.update({'wind_height/' + k: jnp.where(windy, observed[k], native[k])
                            for k in native})
            height_reward = jnp.where(windy, measured['reward'], 0.)
            outgoing = jnp.stack([metrics[f'dynamics/target_v{k}_m_s'] for k in 'xyz'], axis=-1)
            incoming = before.data.qvel[:, self.ball_vadr:self.ball_vadr + 3]
            reference = ideal_contact_reference(incoming, outgoing, xp=jnp)
            normal = core.data.site_xmat[:, self.racket_site_id].reshape((-1, 3, 3))[:, :, 2]
            normal = jnp.where((normal[:, 2] < 0.)[:, None], -normal, normal)
            error = jnp.rad2deg(jnp.arccos(jnp.clip(jnp.sum(
                normal * reference['normal'], axis=-1), -1., 1.)))
            actual_tilt = jnp.rad2deg(jnp.arccos(jnp.clip(normal[:, 2], -1., 1.)))
            edge_error = jnp.where(edge, error, state.impact_error)
            edge_target = jnp.where(edge, reference['tilt_deg'], state.impact_target_tilt)
            edge_actual = jnp.where(edge, actual_tilt, state.impact_actual_tilt)
            confirmed = jnp.maximum(core.hit_count - before.hit_count, 0).astype(jnp.float32)
            metrics['impact/confirmed_count'] = confirmed
            metrics['impact/confirmed_normal_error_deg_sum'] = confirmed * edge_error
            metrics['impact/confirmed_normal_error_deg_sq_sum'] = confirmed * edge_error ** 2
            for prefix, selected in (
                    ('impact', jnp.ones_like(windy)), ('clean_impact', ~windy),
                    ('wind_impact', windy & (gain > 0.)),
                    ('onset_impact', windy & (gain > 0.) & (before.wind_startup_hit_count < 12))):
                event = (edge & selected).astype(jnp.float32)
                metrics.update({prefix + '/count': event,
                    prefix + '/normal_error_deg_sum': event * error,
                    prefix + '/normal_error_deg_sq_sum': event * error ** 2,
                    prefix + '/target_tilt_deg_sum': event * reference['tilt_deg'],
                    prefix + '/actual_tilt_deg_sum': event * actual_tilt,
                    prefix + '/racket_z_m_sum': event * metrics['racket_z'],
                    prefix + '/z_margin_m_sum': event * metrics['dynamics/z_margin_m'],
                    prefix + '/racket_vz_m_s_sum': event * core.prev_racket_vel[:, 2]})
            same = state.tracker.domain_s == before.wind_sampled_s
            cycle = edge & state.reference_valid & same & windy & (state.release_gain > 0.)
            dv = incoming - state.expected_incoming
            time_error = core.step_count * self.dt - state.release_time - state.expected_period
            normalized = jnp.abs(dv) / jnp.asarray([.060, .060, .100])
            huber = lambda x: .5 * jnp.minimum(jnp.abs(x), 1.) ** 2 + jnp.maximum(jnp.abs(x) - 1., 0.)
            cycle_cost = -.25 * cycle * state.release_gain * (
                jnp.minimum(jnp.sum(huber(normalized), axis=-1), 4.)
                + jnp.minimum(huber(time_error / .060), 4.))
            metrics['cycle/count'] = cycle.astype(jnp.float32)
            metrics['cycle/time_error_s_sum'] = cycle * time_error
            metrics['cycle/time_error_s_sq_sum'] = cycle * time_error ** 2
            for index, axis in enumerate('xyz'):
                metrics[f'cycle/incoming_error_v{axis}_sum'] = cycle * dv[:, index]
                metrics[f'cycle/incoming_error_v{axis}_sq_sum'] = cycle * dv[:, index] ** 2
            expected = jnp.stack([metrics[f'dynamics/target_incoming_v{k}_m_s'] for k in 'xyz'], axis=-1)
            next_state = FlightState(core, tracker, edge_error, edge_target, edge_actual,
                jnp.where(release[:, None], expected, state.expected_incoming),
                jnp.where(release, metrics['dynamics/reference_valid'] > .5,
                          state.reference_valid & same & ~edge) & ~done,
                jnp.where(release, core.step_count * self.dt, state.release_time),
                jnp.where(release, metrics['dynamics/period_s'], state.expected_period),
                jnp.where(release, gain, state.release_gain))
            # Real physical failure/truncation costs remain native; no positive
            # edge/apex credit can be farmed with contact chatter.
            reward = reward + height_reward + cycle_cost
            metrics['reward/observed_wind_apex'] = height_reward
            metrics['reward/periodic_incoming'] = cycle_cost
            if getattr(self.cfg, 'wind_survival_precision', False):
                from fixed_plane_wind_survival_precision import reward_terms
                elapsed = jnp.where((state.release_time > 0.) & same,
                    core.step_count * self.dt - state.release_time, 0.)
                clean_gain = ((before.wind_mode == 0)
                              & (before.wind_startup_hit_count >= 5)).astype(jnp.float32)
                terms = reward_terms(metrics, wind_gain=gain, clean_gain=clean_gain,
                    dt_s=self.dt, release_elapsed_s=elapsed, xp=jnp)
                reward = reward + sum(terms.values())
                metrics.update(terms)
            if getattr(self.cfg, 'wind_gate_precision', False):
                from fixed_plane_wind_gate_precision import reward_terms as gate_terms
                elapsed = jnp.where((state.release_time > 0.) & same,
                    core.step_count*self.dt-state.release_time,0.)
                terms = gate_terms(metrics, measured, wind_gain=gain,
                    height_gain=state.tracker.gain * windy,
                    version=self.cfg.wind_gate_precision_version, xp=jnp,
                    positive=getattr(self.cfg,'wind_gate_precision_positive',False),
                    release_elapsed_s=elapsed,apex_elapsed_s=elapsed,
                    clean_gain=shelter_gain(before,self.dt,xp=jnp)*(~windy),dt_s=self.dt)
                reward = reward + sum(terms.values())
                metrics.update(terms)
                from fixed_plane_wind_gate_precision import group_reward_metrics
                metrics.update(group_reward_metrics(reward,windy,metrics,xp=jnp))
            metrics['reward/total'] = reward
            return next_state, obs, reward, done, metrics

    def factory(*args, **kwargs):
        inner = inner_class(*args, **kwargs)
        return FlightEnv(inner) if getattr(inner.cfg, 'wind_flight_alignment', False) else inner
    return factory
