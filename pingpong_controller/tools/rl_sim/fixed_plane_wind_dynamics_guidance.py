"""Default-off flight/outgoing-velocity guidance; no control or physics changes."""
from __future__ import annotations

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any, Callable

import numpy as np


REVISION = 'wind_flight_output_braking_v1'
SAMPLES = 32
DRAG_FACTOR_KG_M = .5 * 1.2 * .47 * np.pi * .020 ** 2


def guidance_contract(*, quality_guard: bool = False, safety_guard: bool = False,
                       flight_alignment: bool = False, tail_precision: bool = False,
                       recovery_safety: bool = False, full_flight: bool = False) -> dict:
    contract = dict(revision=REVISION, apex_m=1.425, samples=SAMPLES,
        reference='six shooting iterations,five midpoint Picard iterations; native wind increment',
        endpoint='episode racket-anchorXY; native racket-Z midpoint plus calibrated reset ball/site offset',
        startup='future known1s ramp; no future OU noise or unstarted gust',
        outgoing_velocity_sigma_m_s=[.035, .035, .060], outgoing_penalty_per_release=4.,
        contact_height_sigma_m=.030, contact_height_penalty_per_release=1.,
        normal_advisory_weight_per_s=.5, normal_sigma_deg=3.,
        normal_model='frictionless impulse direction only; no imposed yaw/racket normal speed',
        brake_preview_s=.080, z_margin_m=.025, brake_weight_per_s=4.,
        onset_brake_multiplier='1+2exp(-max(confirmed_hits-5,0)/3)',
        scope='replace windy reward after five-hit shelter; clean/startup retain original reward',
        outcomes='all detached upward contacts,including unconfirmed weak bounces',
        cadence='fixed apex/return boundary imply a reference period; original real outcomes retained')
    if quality_guard:
        contract['quality_guard'] = dict(revision='center_view_guard_v1',
            center_sigma_m=.0055, clean_center_cost_per_hit=4., wind_center_cost_per_hit=2.,
            additional_view_failure_cost=200.,
            clean_scope='confirmed hits after five-hit shelter; no old-action supervision')
        if safety_guard:
            contract['quality_guard'].update(revision='center_view_braking_guard_v2',
                clean_center_cost_per_hit=.5, wind_center_cost_per_hit=.25,
                additional_braking_weight_per_s=4.,
                braking_scope='wind onset multiplier and clean after-five shelter; native hard bounds')
    if flight_alignment:
        from fixed_plane_wind_flight_outcomes import alignment_contract
        contract['outgoing_velocity_sigma_m_s'] = [.035, .035, .030]
        contract['physical_height_alignment'] = alignment_contract()
    if tail_precision:
        if not flight_alignment:
            raise ValueError('Tail precision requires observed flight-height alignment')
        from fixed_plane_wind_tail_precision import precision_contract
        contract['tail_precision'] = precision_contract()
    if recovery_safety:
        if not tail_precision:
            raise ValueError('Recovery safety requires independent vertical precision')
        from fixed_plane_wind_recovery_safety import recovery_contract
        contract['recovery_safety'] = recovery_contract(full_flight=full_flight)
    if full_flight and not recovery_safety:
        raise ValueError('Full-flight recovery requires the explicit recovery mode')
    return contract


def _loop(count: int, fn: Callable, carry: Any, xp: Any) -> Any:
    if xp is np:
        for index in range(count):
            carry = fn(index, carry)
        return carry
    import jax
    return jax.lax.fori_loop(0, count, fn, carry)


def integrate_flight(start_m: Any, outgoing_m_s: Any, duration_s: Any,
                     wind_m_s: Any, mass_kg: Any, *, xp: Any,
                     drag_factor_kg_m: float = DRAG_FACTOR_KG_M,
                     force_limit_n: float = .020, gravity_m_s2: float = 9.81) -> dict:
    """Midpoint flight for [...,3] states and [...,32,3] expected wind samples."""
    u = (xp.arange(SAMPLES, dtype=outgoing_m_s.dtype) + .5) / SAMPLES
    dt = duration_s / SAMPLES

    def acceleration(velocity):
        relative = velocity - wind_m_s
        force = drag_factor_kg_m * (
            xp.linalg.norm(velocity, axis=-1, keepdims=True) * velocity
            - xp.linalg.norm(relative, axis=-1, keepdims=True) * relative)
        force = xp.where(xp.all(wind_m_s == 0., axis=-1, keepdims=True), 0., force)
        force *= xp.minimum(1., force_limit_n / xp.maximum(
            xp.linalg.norm(force, axis=-1, keepdims=True), 1e-12))
        return force / mass_kg[..., None, None] + xp.asarray([0., 0., -gravity_m_s2])

    def refine(_, velocity):
        acc = acceleration(velocity)
        return outgoing_m_s[..., None, :] + dt[..., None, None] * (
            xp.cumsum(acc, axis=-2) - .5 * acc)

    initial = xp.broadcast_to(outgoing_m_s[..., None, :], wind_m_s.shape)
    velocity = _loop(5, refine, initial, xp)
    acc = acceleration(velocity)
    endpoint = start_m + outgoing_m_s * duration_s[..., None] + duration_s[..., None] ** 2 * xp.mean(
        (1. - u)[..., None] * acc, axis=-2)
    incoming = outgoing_m_s + duration_s[..., None] * xp.mean(acc, axis=-2)
    az, vz = acc[..., 2], velocity[..., 2]
    prefix = xp.cumsum(az, axis=-1)
    z_mid = (start_m[..., 2, None] + outgoing_m_s[..., 2, None] * duration_s[..., None] * u
             + dt[..., None] ** 2 * (xp.cumsum(prefix, axis=-1) - prefix + .125 * az))
    index = xp.argmin(xp.abs(vz), axis=-1)[..., None]
    pick = lambda values: xp.take_along_axis(values, index, axis=-1)[..., 0]
    apex = pick(z_mid - vz ** 2 / (2. * xp.minimum(az, -1e-6)))
    apex_time = pick(duration_s[..., None] * u - vz / xp.minimum(az, -1e-6))
    return dict(endpoint_m=endpoint, incoming_m_s=incoming, apex_m=apex,
                apex_time_s=apex_time)


def solve_flight_reference(start_m: Any, return_m: Any, wind_forecast: Callable,
                           mass_kg: Any, *, xp: Any, apex_m: float = 1.425,
                           drag_factor_kg_m: float = DRAG_FACTOR_KG_M,
                           force_limit_n: float = .020, gravity_m_s2: float = 9.81) -> dict:
    """Shoot a fixed return point and apex; wind forecast is causal, not sampled future truth."""
    vz = xp.sqrt(2. * gravity_m_s2 * xp.maximum(apex_m - start_m[..., 2], .05))
    period = (vz + xp.sqrt(xp.maximum(vz ** 2 + 2. * gravity_m_s2 * (
        start_m[..., 2] - return_m[..., 2]), 0.))) / gravity_m_s2
    velocity = xp.concatenate(((return_m[..., :2] - start_m[..., :2]) /
                               period[..., None], vz[..., None]), axis=-1)

    def flight(launch, duration):
        return integrate_flight(start_m, launch, duration, wind_forecast(duration), mass_kg,
            xp=xp, drag_factor_kg_m=drag_factor_kg_m, force_limit_n=force_limit_n,
            gravity_m_s2=gravity_m_s2)

    def refine(_, carry):
        launch, duration = carry
        result = flight(launch, duration)
        dvz = xp.clip((apex_m - result['apex_m']) /
                      xp.maximum(result['apex_time_s'], .05), -.5, .5)
        dt = xp.clip((return_m[..., 2] - result['endpoint_m'][..., 2] - duration * dvz) /
                     xp.minimum(result['incoming_m_s'][..., 2], -.5), -.1, .1)
        dvxy = (return_m[..., :2] - result['endpoint_m'][..., :2]
                 - result['incoming_m_s'][..., :2] * dt[..., None]) / duration[..., None]
        launch = xp.concatenate((launch[..., :2] + dvxy,
                                  (launch[..., 2] + dvz)[..., None]), axis=-1)
        return launch, xp.clip(duration + dt, .15, 1.2)

    velocity, period = _loop(6, refine, (velocity, period), xp)
    result = flight(velocity, period)
    valid = ((start_m[..., 2] < apex_m - .025) & (return_m[..., 2] < apex_m - .025)
             & (mass_kg > 0.) & xp.all(xp.isfinite(velocity), axis=-1)
             & (xp.linalg.norm(result['endpoint_m'] - return_m, axis=-1) < .005)
             & (xp.abs(result['apex_m'] - apex_m) < .003))
    return dict(outgoing_m_s=velocity, period_s=period, valid=valid, **result)


def expected_wind_samples(state: Any, cfg: Any, now_s: Any, duration_s: Any, *, xp: Any,
                           lead_s: Any = 0.) -> Any:
    """Forecast known startup/current gust and OU mean reversion; never future random draws."""
    offsets = xp.asarray(lead_s)[..., None] + duration_s[..., None] * (xp.arange(SAMPLES) + .5) / SAMPLES
    future = now_s[..., None] + offsets
    phase = (future - state.wind_gust_start_s[..., None]) / state.wind_gust_duration_s[..., None]
    active_now = ((now_s >= state.wind_gust_start_s)
                  & (now_s < state.wind_gust_start_s + state.wind_gust_duration_s))
    gust = xp.where(active_now[..., None] & (phase > 0.) & (phase < 1.),
                    xp.sin(xp.pi * phase) ** 2, 0.)
    wind = (state.wind_background_m_s[..., None, :]
            + state.wind_ou_m_s[..., None, :] * xp.exp(-offsets / cfg.ou_tau_s)[..., None]
            + gust[..., None] * state.wind_gust_vector_m_s[..., None, :])
    wind *= xp.minimum(1., cfg.max_speed_m_s / xp.maximum(
        xp.linalg.norm(wind, axis=-1, keepdims=True), 1e-12))
    from fixed_plane_wind_direction_domain import constrain_wind
    wind=constrain_wind(wind,cfg,xp=xp)
    if cfg.startup_no_wind_hits:
        active = state.wind_startup_started_s >= 0.
        phase = xp.clip((future - state.wind_startup_started_s[..., None]) /
                        max(cfg.startup_ramp_s, 1e-12), 0., 1.)
        gain = xp.sin(.5 * xp.pi * phase) ** 2 if cfg.startup_ramp_s else xp.ones_like(phase)
        wind *= xp.where(active[..., None], gain, 0.)[..., None]
    return xp.where((state.wind_mode > 0)[..., None, None], wind, 0.)


def ideal_contact_reference(incoming_m_s: Any, outgoing_m_s: Any, *, xp: Any,
                            restitution: float = .775) -> dict:
    """Frictionless impulse direction is advisory; contact twist/friction make it non-unique."""
    if not 0. <= restitution <= 1.:
        raise ValueError('Ideal restitution must be in [0,1]')
    change = outgoing_m_s - incoming_m_s
    normal = change / xp.maximum(xp.linalg.norm(change, axis=-1, keepdims=True), 1e-9)
    speed = xp.sum((outgoing_m_s + restitution * incoming_m_s) * normal,
                   axis=-1) / (1. + restitution)
    return dict(normal=normal, tilt_deg=xp.rad2deg(xp.arccos(xp.clip(normal[..., 2], -1., 1.))),
                normal_racket_m_s=speed)


def release_event(previous_contact: Any, current_contact: Any, raw_edge: Any,
                   outgoing_vz_m_s: Any, *, xp: Any) -> Any:
    """A detached upward physical bounce; no confirmed-apex selection bias."""
    return (~current_contact) & (previous_contact | raw_edge) & (outgoing_vz_m_s > 0.)


def braking_cost(z_m: Any, vz_m_s: Any, bounds_m: Any, *, xp: Any) -> Any:
    """Soft80ms constant-velocity preview, not a certified stopping-distance constraint."""
    lower, upper = bounds_m
    lower_safe_speed = xp.maximum(z_m - lower - .025, 0.) / .080
    upper_safe_speed = xp.maximum(upper - z_m - .025, 0.) / .080
    lower_excess = xp.maximum(-vz_m_s - lower_safe_speed, 0.) / .25
    upper_excess = xp.maximum(vz_m_s - upper_safe_speed, 0.) / .25
    return xp.minimum(lower_excess ** 2 + upper_excess ** 2, 4.)


@lru_cache(maxsize=12)
def _config_type(parent: type, quality_guard: bool, safety_guard: bool) -> type:
    fields = [('wind_dynamics_guidance', bool, field(default=True))]
    if quality_guard:
        fields.append(('wind_dynamics_quality_guard', bool, field(default=True)))
    if safety_guard:
        fields.append(('wind_dynamics_safety_guard', bool, field(default=True)))
    return make_dataclass('WindDynamicsGuidanceConfig', fields, bases=(parent,), frozen=True)


def with_guidance_config(cfg: Any, enabled: bool, *, quality_guard: bool = False,
                         safety_guard: bool = False) -> Any:
    if safety_guard and not quality_guard:
        raise ValueError('Safety refinement requires the explicit quality guard')
    if enabled and getattr(cfg, 'ball_flight_drag_coefficient_m_inv', 0.) != 0.:
        raise ValueError('Flight guidance requires the registered zero legacy-flight-drag path')
    return (_config_type(type(cfg), quality_guard, safety_guard)(**vars(cfg), wind_dynamics_guidance=True)
            if enabled else cfg)


def quality_guard_terms(metrics: dict, *, wind_gain: Any, clean_gain: Any, xp: Any,
                        safety_guard: bool = False, dt_s: float = .005) -> dict:
    """Direct real-contact center loss and view failures; retain every native termination."""
    hit = metrics['hit_event_count']
    center = metrics['hit_contact_center_dist_sum'] / xp.maximum(hit, 1.) / .0055
    loss = xp.minimum(.5 * xp.minimum(center, 1.) ** 2 + xp.maximum(center - 1., 0.), 4.)
    view = sum(value for key, value in metrics.items() if key.startswith('done/ball_view_'))
    result = {
        'reward/dynamics_clean_center': -(.5 if safety_guard else 4.) * clean_gain * hit * loss,
        'reward/dynamics_wind_center': -(.25 if safety_guard else 2.) * wind_gain * hit * loss,
        'reward/dynamics_view_failure': -200. * (wind_gain + clean_gain) * xp.minimum(view, 1.),
    }
    if safety_guard:
        onset = 1. + 2. * xp.exp(-xp.maximum(metrics['dynamics/confirmed_hits'] - 5., 0.) / 3.)
        result['reward/dynamics_guard_braking'] = -4. * dt_s * (
            wind_gain * onset + clean_gain) * metrics['dynamics/braking_cost']
    return result


def guidance_terms(native_reward: Any, metrics: dict, *, gain: Any, dt_s: float, xp: Any,
                    flight_alignment: bool = False, tail_precision: bool = False,
                    recovery_safety: bool = False) -> dict:
    """Bounded direct velocity loss plus real-hit credit; no positive per-release farming."""
    def huber(value):
        value = xp.abs(value)
        return .5 * xp.minimum(value, 1.) ** 2 + xp.maximum(value - 1., 0.)

    failed = metrics['terminated'] > .5
    active = metrics['dynamics/recent'] > .5
    velocity_error = metrics['dynamics/velocity_error_m_s'] / xp.asarray(
        [.035, .035, .030 if flight_alignment else .060])
    loss = xp.minimum(xp.sum(huber(velocity_error), axis=-1), 4.)
    if tail_precision:
        if not flight_alignment:
            raise ValueError('Tail precision requires observed flight-height alignment')
        from fixed_plane_wind_tail_precision import outgoing_precision_loss
        loss = outgoing_precision_loss(metrics['dynamics/velocity_error_m_s'], xp=xp)['total']
    loss = xp.where(metrics['dynamics/reference_valid'] > .5, loss, 12. if tail_precision else 4.)
    outgoing = -4. * metrics['dynamics/release_event'] * loss
    height = -metrics['dynamics/release_event'] * xp.minimum(huber((
        metrics['racket_z'] - metrics['dynamics/target_racket_z_m']) / .030), 4.)
    hit = metrics['hit_event_count']
    center = metrics['hit_contact_center_dist_sum'] / xp.maximum(hit, 1.)
    racket_xy = xp.sqrt(xp.maximum(metrics['hit_racket_vxy_sq_sum'], 0.) / xp.maximum(hit, 1.))
    steady = metrics['rapid_steady_steady_events']
    omega = xp.sqrt(xp.maximum(metrics['rapid_steady_steady_racket_omega_sq_sum'], 0.) /
                    xp.maximum(steady, 1.))
    quality = xp.exp(-.5 * ((center / .0055) ** 2 + (racket_xy / .055) ** 2 + (omega / .28) ** 2))
    margin = xp.clip(metrics['dynamics/z_margin_m'] / .025, 0., 1.)
    interval = xp.clip(metrics['dynamics/hit_interval_s'], 0., 1.2)
    cadence_error = (interval - metrics['dynamics/period_s']) / .060
    cadence = -.5 * hit * xp.minimum(huber(cadence_error), 4.)
    hit_credit = 4. * interval * hit * quality * margin
    normal = .5 * dt_s * metrics['dynamics/precontact'] * xp.exp(
        -.5 * (metrics['dynamics/normal_error_deg'] / 3.) ** 2)
    onset = 1. + 2. * xp.exp(-xp.maximum(metrics['dynamics/confirmed_hits'] - 5., 0.) / 3.)
    brake = -4. * dt_s * onset * metrics['dynamics/braking_cost']
    failure = xp.where(failed, xp.minimum(-100.,
        metrics['reward/racket_z_limit_termination_penalty']), 0.)
    if tail_precision:
        from fixed_plane_wind_tail_precision import failure_cost
        failure = failure_cost(failed, metrics['reward/racket_z_limit_termination_penalty'], xp=xp)
    valid = active & ~failed
    terms = {
        'reward/dynamics_replace': -gain * native_reward,
        'reward/dynamics_survival': gain * xp.where(valid, 2. * dt_s * margin, 0.),
        'reward/dynamics_hit': gain * xp.where(valid, hit_credit, 0.),
        'reward/dynamics_outgoing': gain * xp.where(~failed, outgoing, 0.),
        'reward/dynamics_contact_height': gain * xp.where(~failed, height, 0.),
        'reward/dynamics_cadence': gain * xp.where(valid, cadence, 0.),
        'reward/dynamics_normal': gain * xp.where(valid, normal, 0.),
        'reward/dynamics_braking': gain * xp.where(~failed, brake, 0.),
        'reward/dynamics_failure': gain * failure,
    }
    if tail_precision:
        from fixed_plane_wind_tail_precision import height_recovery_cost
        terms['reward/dynamics_height_recovery'] = -gain * xp.where(~failed,
            height_recovery_cost(metrics['racket_z'], metrics['dynamics/target_racket_z_m'],
                                 dt_s=dt_s, xp=xp), 0.)
    if recovery_safety:
        if not tail_precision:
            raise ValueError('Recovery safety requires independent vertical precision')
        from fixed_plane_wind_recovery_safety import safety_terms
        terms.update(safety_terms(metrics, gain=gain, dt_s=dt_s, xp=xp))
    return terms


def apply_guidance(env: Any, before: Any, after: Any, reward: Any, metrics: dict) -> tuple[Any, dict]:
    """Reward-only privileged references; actor/critic/state/reset remain unchanged."""
    import jax.numpy as jnp
    from fixed_plane_v325_wind import incremental_force, wind_velocity
    from fixed_plane_wind_pose import shelter_gain

    cfg = env.cfg.wind_config
    gain = shelter_gain(before, env.dt, xp=jnp) * (before.wind_mode > 0)
    ball_pos = after.data.qpos[:, env.ball_qadr:env.ball_qadr + 3]
    ball_vel = after.data.qvel[:, env.ball_vadr:env.ball_vadr + 3]
    released = release_event(before.prev_contact, after.prev_contact,
        metrics['raw_physical_contact_edge'] > .5, ball_vel[:, 2], xp=jnp)
    mass = after.model.body_mass[:, env.ball_body_id]
    bounds = env.cfg.fixed_contact_racket_z_bounds_m
    desired_site_z = .5 * (bounds[0] + bounds[1])
    ball_site_offset = before.reset_ball_pos[:, 2] - before.reset_ball_surface_gap - before.racket_anchor[:, 2]
    return_z = desired_site_z + ball_site_offset
    return_point = jnp.concatenate((before.racket_anchor[:, :2], return_z[:, None]), axis=-1)

    rotation = after.data.site_xmat[:, env.racket_site_id].reshape((-1, 3, 3))
    normal = rotation[:, :, 2]
    normal = jnp.where((normal[:, 2] < 0.)[:, None], -normal, normal)
    surface_z = after.data.geom_xpos[:, env.racket_geom_id, 2] + jnp.abs(normal[:, 2]) * (
        after.model.geom_size[:, env.racket_geom_id, int(env.cfg.fixed_contact_racket_surface_size_axis)])
    radius = after.model.geom_size[:, env.ball_geom_id, 0]
    contact_z = surface_z + radius
    gravity = float(env.mj_model.opt.gravity[2]) * -1.
    distance = jnp.maximum(ball_pos[:, 2] - contact_z, 0.)
    ttc = (ball_vel[:, 2] + jnp.sqrt(jnp.maximum(
        ball_vel[:, 2] ** 2 + 2. * gravity * distance, 0.))) / gravity
    now = after.step_count * env.dt
    force, _ = incremental_force(ball_vel, wind_velocity(after, now, cfg), cfg)
    acceleration = force / mass[:, None] + jnp.asarray([0., 0., -gravity])
    impact_position = ball_pos + ball_vel * ttc[:, None] + .5 * acceleration * ttc[:, None] ** 2
    impact_position = impact_position.at[:, 2].set(contact_z)
    lateral_distance = jnp.linalg.norm(impact_position[:, :2]
        - after.data.site_xpos[:, env.racket_site_id, :2], axis=-1)
    precontact = ((ball_vel[:, 2] < -.1) & (ttc > 0.) & (ttc < .12)
                  & (lateral_distance < .07) & ~after.prev_contact)
    start = jnp.where(precontact[:, None], impact_position,
                      jnp.where(released[:, None], ball_pos, return_point))
    aligned = getattr(env.cfg, 'wind_flight_alignment', False)
    if aligned:
        # At a raw physical edge the reference must use actual collision height,
        # not the arbitrary nominal return height used during inter-hit flight.
        edge = metrics['raw_physical_contact_edge'] > .5
        start = jnp.where((edge | released)[:, None], ball_pos, start)
    forecast = lambda duration: expected_wind_samples(after, cfg, now, duration, xp=jnp,
        lead_s=jnp.where(precontact, ttc, 0.))
    ref = solve_flight_reference(start, return_point, forecast, mass, xp=jnp,
        drag_factor_kg_m=cfg.drag_factor_kg_m, force_limit_n=cfg.force_limit_n,
        gravity_m_s2=gravity)
    incoming = jnp.where(precontact[:, None], ball_vel + acceleration * ttc[:, None], ref['incoming_m_s'])
    contact = ideal_contact_reference(incoming, ref['outgoing_m_s'], xp=jnp,
        restitution=float(env.cfg.fixed_contact_impact_effective_restitution))
    normal_error = jnp.rad2deg(jnp.arccos(jnp.clip(
        jnp.sum(normal * contact['normal'], axis=-1), -1., 1.)))
    available_tilt = jnp.rad2deg(jnp.min(env.wind_pose_limits(after)[1], axis=-1))
    normal_feasible = contact['tilt_deg'] <= available_tilt
    z_m = metrics['racket_z']
    z_margin = jnp.minimum(z_m - bounds[0], bounds[1] - z_m)
    age = after.step_count * env.dt - after.last_counted_hit_time
    telemetry = {
        'dynamics/recent': ((after.hit_count > 0) & (age >= 0.) & (age <= 1.2)).astype(jnp.float32),
        'dynamics/release_event': released.astype(jnp.float32),
        'dynamics/velocity_error_m_s': ball_vel - ref['outgoing_m_s'],
        'dynamics/reference_valid': ref['valid'].astype(jnp.float32),
        'dynamics/period_s': ref['period_s'],
        'dynamics/reference_tilt_deg': contact['tilt_deg'],
        'dynamics/normal_error_deg': normal_error,
        'dynamics/precontact': (precontact & ref['valid'] & normal_feasible).astype(jnp.float32),
        'dynamics/normal_reference_feasible': normal_feasible.astype(jnp.float32),
        'dynamics/z_margin_m': z_margin,
        'dynamics/braking_cost': braking_cost(z_m, after.prev_racket_vel[:, 2], bounds, xp=jnp),
        'dynamics/confirmed_hits': before.wind_startup_hit_count.astype(jnp.float32),
        'dynamics/hit_interval_s': after.step_count * env.dt - before.last_counted_hit_time,
        'dynamics/target_racket_z_m': jnp.full_like(z_m, desired_site_z),
        'dynamics/target_return_z_m': return_z,
        'dynamics/reference_endpoint_error_m': jnp.linalg.norm(ref['endpoint_m'] - return_point, axis=-1),
        'dynamics/reference_apex_error_m': ref['apex_m'] - 1.425,
    }
    metrics.update(telemetry)
    if getattr(env.cfg, 'wind_full_flight_recovery', False):
        # Protect a real approaching contact even if the advisory shooting
        # reference is invalid or its ideal normal exceeds current permission.
        metrics['dynamics/precontact_geometry'] = precontact.astype(jnp.float32)
    recovery_safety = getattr(env.cfg, 'wind_recovery_safety', False)
    if recovery_safety:
        from fixed_plane_wind_recovery_safety import recovery_phase
        metrics['dynamics/recovery_phase'] = recovery_phase(age, ball_vel[:, 2],
            after.prev_contact, before.wind_startup_hit_count, xp=jnp).astype(jnp.float32)
        metrics['dynamics/racket_vz_m_s'] = after.prev_racket_vel[:, 2]
    # PPO's compact telemetry expects scalar per-lane values, not vector leaves.
    terms = guidance_terms(reward, metrics, gain=gain, dt_s=env.dt, xp=jnp,
                           flight_alignment=aligned,
                           tail_precision=getattr(env.cfg, 'wind_tail_precision', False),
                           recovery_safety=recovery_safety and not getattr(
                               env.cfg, 'wind_full_flight_recovery', False))
    if getattr(env.cfg, 'wind_dynamics_quality_guard', False):
        clean_gain = ((before.wind_mode == 0) & (before.wind_startup_hit_count >= 5)).astype(jnp.float32)
        terms.update(quality_guard_terms(metrics, wind_gain=gain, clean_gain=clean_gain, xp=jnp,
            safety_guard=getattr(env.cfg, 'wind_dynamics_safety_guard', False), dt_s=env.dt))
    metrics.pop('dynamics/velocity_error_m_s')
    for axis, name in enumerate(('x', 'y', 'z')):
        metrics[f'dynamics/target_v{name}_m_s'] = ref['outgoing_m_s'][:, axis]
        if aligned:
            metrics[f'dynamics/target_incoming_v{name}_m_s'] = ref['incoming_m_s'][:, axis]
        metrics[f'dynamics/outgoing_error_v{name}_m_s'] = telemetry['dynamics/velocity_error_m_s'][:, axis]
        error = telemetry['dynamics/velocity_error_m_s'][:, axis]
        metrics[f'dynamics/release_error_v{name}_sum'] = released * error
        metrics[f'dynamics/release_error_v{name}_sq_sum'] = released * error ** 2
    reward = reward + sum(terms.values())
    metrics.update(terms)
    return reward, metrics
