"""Default-off causal contact-center preview; reward only, no state mutation."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

MODE = 'clean10_dynamics_reference_center_negy_v6'
BIRTH_MODE = 'clean10_dynamics_reference_center_negy_v2'
BIRTH_SHA256 = 'ce126cbfcf8df2ea16e57f683bf8e5ef2f7945eb832e4603bfb37cd24991e0ac'
BIRTH_UPDATE = 14000
BIRTH_STEP = 3444703232
BIRTH_ADAM_T = 950575
REVIEWED_BIRTH_SOURCES = {
    BIRTH_SHA256: (14000, 14, False),
    'fd9def8f5fdb511e1543cbcca264e0021738ff07c6425e8d393cb1c240918e97': (14400, 14, True),
}
STEPS = 32
HORIZON_S = .12
DRAG_FACTOR_KG_M = .5 * 1.2 * .47 * 3.141592653589793 * .020**2


def precontact_contract() -> dict:
    """Describe the extra bounded credit while retaining every original gate."""
    return dict(revision='causal_moving_local_center_v1', horizon_s=HORIZON_S,
        steps=STEPS, midpoint_picard_iterations=5,
        frame='future racket site local XY; actual linear/angular velocity',
        force='same capped quadratic wind increment; known causal wind forecast',
        score='exp(-0.5*(predicted_center/(0.0055+0.10*ttc_s))**2)',
        rate_per_s=dict(clean=80., wind=40.),
        timing='descending, approaching, recent/live, contact-free; original shelter/ramp',
        warmup_policy_steps=8192, base_reward_maturity_policy_steps=8192,
        source=dict(sha256=BIRTH_SHA256, update=BIRTH_UPDATE, true_step=BIRTH_STEP,
                    Adam_t=BIRTH_ADAM_T),
        additional_reviewed_births={sha: dict(update=u, source_stage_index=s, stage_converged=c)
            for sha, (u, s, c) in REVIEWED_BIRTH_SOURCES.items() if sha != BIRTH_SHA256},
        resume_origin_metadata='precontact_reward_birth_update; missing means original14000',
        preserved='all original rewards, observations, control, physics, PPO, sampling and gates')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindPrecontactCenterConfig', [
        ('wind_precontact_center', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_precontact_config(cfg: Any, enabled: bool) -> Any:
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def rotate_vector(vector: Any, rotation_vector: Any, *, xp: Any) -> Any:
    """Rotate [...,3] vectors by world rotation vectors, including zero angle."""
    theta = xp.linalg.norm(rotation_vector, axis=-1, keepdims=True)
    a = xp.sinc(theta / xp.pi)
    b = .5 * xp.sinc(theta / (2. * xp.pi))**2
    cross = xp.cross(rotation_vector, vector)
    return vector + a*cross + b*xp.cross(rotation_vector, cross)


def predict_contact(ball_pos_m: Any, ball_vel_m_s: Any, racket_pos_m: Any,
                    racket_rotation: Any, racket_velocity_m_s: Any,
                    racket_omega_rad_s: Any, plane_offset_m: Any,
                    wind_samples_m_s: Any, mass_kg: Any, *, xp: Any,
                    gravity_m_s2: Any = 9.81,
                    drag_factor_kg_m: float = DRAG_FACTOR_KG_M,
                    force_limit_n: float = .020) -> dict:
    """Preview [...,3] flight against a moving/rotating plane over120ms.

    Wind is [...,32,3], orientation [...,3,3]; mass/offset/gravity broadcast
    over the batch. The returned diagnostics are scalar per lane. This is
    advisory geometry, not a replacement for native contact or termination.
    """
    from fixed_plane_wind_dynamics_guidance import _loop
    dtype = ball_pos_m.dtype
    dt = HORIZON_S / STEPS
    mass = xp.asarray(mass_kg, dtype=dtype)
    g = xp.broadcast_to(xp.asarray(gravity_m_s2, dtype=dtype), ball_pos_m.shape[:-1])
    gravity = xp.stack((xp.zeros_like(g), xp.zeros_like(g), -g), axis=-1)

    def acceleration(velocity):
        relative = velocity - wind_samples_m_s
        force = drag_factor_kg_m * (
            xp.linalg.norm(velocity, axis=-1, keepdims=True)*velocity
            - xp.linalg.norm(relative, axis=-1, keepdims=True)*relative)
        force = xp.where(xp.all(wind_samples_m_s == 0., axis=-1, keepdims=True), 0., force)
        force = force*xp.minimum(1., force_limit_n/xp.maximum(
            xp.linalg.norm(force, axis=-1, keepdims=True), 1e-12))
        return force/xp.maximum(mass[..., None, None], 1e-6) + gravity[..., None, :]

    def refine(_, velocity):
        acc = acceleration(velocity)
        return ball_vel_m_s[..., None, :] + dt*(xp.cumsum(acc, axis=-2)-.5*acc)

    initial = xp.broadcast_to(ball_vel_m_s[..., None, :], wind_samples_m_s.shape)
    mid_velocity = _loop(5, refine, initial, xp)
    acc = acceleration(mid_velocity)
    prefix = xp.cumsum(acc, axis=-2)
    times = xp.arange(1, STEPS+1, dtype=dtype)*dt
    positions = (ball_pos_m[..., None, :] + ball_vel_m_s[..., None, :]*times[:, None]
                 + dt**2*(xp.cumsum(prefix, axis=-2)-.5*prefix))
    velocities = ball_vel_m_s[..., None, :] + dt*prefix
    positions = xp.concatenate((ball_pos_m[..., None, :], positions), axis=-2)
    velocities = xp.concatenate((ball_vel_m_s[..., None, :], velocities), axis=-2)
    times = xp.arange(STEPS+1, dtype=dtype)*dt
    normals = rotate_vector(racket_rotation[..., None, :, 2],
                            racket_omega_rad_s[..., None, :]*times[:, None], xp=xp)
    centers = racket_pos_m[..., None, :] + racket_velocity_m_s[..., None, :]*times[:, None]
    gaps = xp.sum((positions-centers)*normals, axis=-1)-plane_offset_m[..., None]
    crossing = (gaps[..., :-1] > 0.) & (gaps[..., 1:] <= 0.)
    index = xp.argmax(crossing, axis=-1)

    def scalar(values, offset=0):
        return xp.take_along_axis(values, (index+offset)[..., None], axis=-1)[..., 0]

    def vector(values, offset=0):
        return xp.take_along_axis(values, (index+offset)[..., None, None], axis=-2)[..., 0, :]

    fraction = xp.clip(scalar(gaps)/xp.maximum(scalar(gaps)-scalar(gaps, 1), 1e-9), 0., 1.)
    ttc = (index.astype(dtype)+fraction)*dt
    ball = vector(positions)+(vector(positions, 1)-vector(positions))*fraction[..., None]
    velocity = vector(velocities)+(vector(velocities, 1)-vector(velocities))*fraction[..., None]
    center = racket_pos_m+racket_velocity_m_s*ttc[..., None]
    relative = ball-center
    unrotated = rotate_vector(relative, -racket_omega_rad_s*ttc[..., None], xp=xp)
    local = xp.einsum('...ji,...j->...i', racket_rotation, unrotated)
    normal = rotate_vector(racket_rotation[..., :, 2], racket_omega_rad_s*ttc[..., None], xp=xp)
    closing = -xp.sum((velocity-racket_velocity_m_s
                       - xp.cross(racket_omega_rad_s, relative))*normal, axis=-1)
    initial_closing = -xp.sum((ball_vel_m_s-racket_velocity_m_s-xp.cross(
        racket_omega_rad_s, ball_pos_m-racket_pos_m))*racket_rotation[..., :, 2], axis=-1)
    valid = (xp.any(crossing, axis=-1) & (gaps[..., 0] > 0.)
             & (ball_vel_m_s[..., 2] < -.1) & (initial_closing > .1) & (closing > .1)
             & (racket_rotation[..., 2, 2] > .05) & (mass > 0.)
             & xp.all(xp.isfinite(local), axis=-1) & xp.isfinite(ttc))
    raw = dict(ttc_s=ttc, center_m=xp.linalg.norm(local[..., :2], axis=-1),
               local_x_m=local[..., 0], local_y_m=local[..., 1],
               racket_z_m=center[..., 2], closing_m_s=closing)
    return dict(valid=valid, **{k: xp.where(valid, v, xp.zeros_like(v)) for k, v in raw.items()})


def precontact_reward(prediction: dict, *, clean: Any, gain: Any, recent: Any,
                      terminated: Any, contact: Any, reward_steps: Any, dt_s: float,
                      racket_z_bounds_m: tuple, xp: Any) -> dict:
    """Extra bounded [N] credit; no old reward or failure term is replaced."""
    progress = xp.clip(xp.asarray(reward_steps, dtype=prediction['ttc_s'].dtype)/8192., 0., 1.)
    warmup = progress**2*(3.-2.*progress)
    active = (prediction['valid'] & recent & ~terminated & ~contact & (gain > 0.)
              & (prediction['racket_z_m'] >= racket_z_bounds_m[0])
              & (prediction['racket_z_m'] <= racket_z_bounds_m[1]))
    sigma = .0055+.10*prediction['ttc_s']
    score = xp.exp(-.5*(prediction['center_m']/sigma)**2)
    credit = dt_s*xp.where(clean, 80., 40.)*gain*warmup*xp.where(active, score, 0.)
    metrics = {'reward/precontact_center_credit': credit, 'precontact/warmup': warmup,
               'precontact/active': active.astype(prediction['ttc_s'].dtype)}
    metrics.update({'precontact/'+k: xp.where(active, v, 0.)
                    for k, v in prediction.items() if k != 'valid'})
    metrics['precontact/score'] = xp.where(active, score, 0.)
    return metrics


def environment_credit(env: Any, core: Any, metrics: dict, *, clean: Any,
                       gain: Any, reward_steps: Any) -> dict:
    """Use existing privileged physical quantities only for scalar rewards."""
    import jax.numpy as jnp
    from fixed_plane_wind_dynamics_guidance import expected_wind_samples
    rotation = core.data.site_xmat[:, env.racket_site_id].reshape((-1, 3, 3))
    site = core.data.site_xpos[:, env.racket_site_id]
    normal = rotation[:, :, 2]
    offset = (jnp.sum((core.data.geom_xpos[:, env.racket_geom_id]-site)*normal, axis=-1)
              + core.model.geom_size[:, env.racket_geom_id,
                  int(env.cfg.fixed_contact_racket_surface_size_axis)]
              + core.model.geom_size[:, env.ball_geom_id, 0])
    wind = expected_wind_samples(core, env.cfg.wind_config, core.step_count*env.dt,
        jnp.full_like(core.step_count, HORIZON_S, dtype=jnp.float32), xp=jnp)
    prediction = predict_contact(
        core.data.qpos[:, env.ball_qadr:env.ball_qadr+3],
        core.data.qvel[:, env.ball_vadr:env.ball_vadr+3], site, rotation,
        core.prev_racket_vel, core.data.cvel[:, env.racket_body_id, :3], offset,
        wind, core.model.body_mass[:, env.ball_body_id], xp=jnp,
        gravity_m_s2=jnp.abs(core.dr_gravity_z),
        drag_factor_kg_m=env.cfg.wind_config.drag_factor_kg_m,
        force_limit_n=env.cfg.wind_config.force_limit_n)
    return precontact_reward(prediction, clean=clean, gain=gain,
        recent=metrics['dynamics/recent'] > .5, terminated=metrics['terminated'] > .5,
        contact=core.prev_contact, reward_steps=reward_steps, dt_s=env.dt,
        racket_z_bounds_m=env.cfg.fixed_contact_racket_z_bounds_m, xp=jnp)


def source_resume_context(payload: dict, sha256: str) -> dict:
    """Authenticate this reward phase without changing any learner array."""
    args = payload.get('args', {})
    update, step, stage = (payload.get(k) for k in ('global_update', 'step', 'stage_index'))
    opt = payload.get('opt')
    try:
        adam = int(opt['t'] if isinstance(opt, dict) else opt.t)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError('Missing precontact source optimizer accounting') from error
    if (args.get('curriculum_profile') != 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v325'
            or type(update) is not int or update < BIRTH_UPDATE or type(step) is not int
            or type(stage) is not int or not 14 <= stage < 32):
        raise ValueError('Precontact reward requires its reviewed V325 continuation')
    elapsed = update-BIRTH_UPDATE
    if step != BIRTH_STEP+elapsed*131072 or adam != BIRTH_ADAM_T+elapsed*32:
        raise ValueError('Precontact source update/step/Adam counters disagree')
    if args.get('wind_training_mix') == BIRTH_MODE:
        proof = REVIEWED_BIRTH_SOURCES.get(sha256)
        if proof != (update, stage, bool(payload.get('stage_converged', False))):
            raise ValueError('Unreviewed precontact birth source')
        return dict(birth_update=update, policy_steps=0)
    if (args.get('wind_training_mix') != MODE
            or payload.get('env_cfg', {}).get('wind_precontact_center') is not True):
        raise ValueError('Only this exact precontact recipe can resume its reward phase')
    origin = payload.get('precontact_reward_birth_update', BIRTH_UPDATE)
    minimum_stage = {u: s+int(c) for u, s, c in REVIEWED_BIRTH_SOURCES.values()}
    resumed_stage = stage+int(bool(payload.get('stage_converged', False)))
    if (type(origin) is not int or origin not in minimum_stage or origin > update
            or resumed_stage < minimum_stage[origin]):
        raise ValueError('Unreviewed reward origin or curriculum rollback')
    return dict(birth_update=origin, policy_steps=min((update-origin)*128, 8192))


def source_resume_steps(payload: dict, sha256: str) -> int:
    """Retain the original reward-step interface and legacy source behavior."""
    return source_resume_context(payload, sha256)['policy_steps']


def bind_reward_resume(factory: Any, policy_steps: int) -> Any:
    """Keep base credit mature and resume new credit using reward context only."""
    if not 0 <= policy_steps <= 8192:
        raise ValueError('Invalid precontact continuation clock')

    def build(*args, **kwargs):
        return factory(*args, **kwargs, reward_resume_policy_steps=8192,
                       precontact_resume_policy_steps=policy_steps)

    return build
