"""V325/M12 and V326/M13: bounded horizontal wind-increment continuations."""
from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass, field, make_dataclass
from functools import lru_cache
import math
from typing import Any

import jax
import jax.numpy as jnp

from fixed_plane_v323_ball_obs_noise import spec_for as parent_spec_for
from fixed_plane_wind_direction_domain import constrain_wind

WIND_FIELDS = ('wind_mode', 'wind_background_m_s', 'wind_ou_m_s',
               'wind_gust_vector_m_s', 'wind_gust_start_s', 'wind_gust_duration_s',
               'wind_startup_hit_count', 'wind_startup_started_s', 'wind_sampled_s')
WindState = namedtuple('WindState', WIND_FIELDS)
CRITIC_EXTRA_DIM = 18
MODE_NAMES = ('clean', 'constant', 'slow', 'gust')
MODEL = 'quadratic_wind_increment_v1'
STARTUP_NO_WIND_HITS = 5
STARTUP_RAMP_S = 1.


@dataclass(frozen=True)
class WindConfig:
    """Candidate force/process bounds; air density and Cd are assumptions."""

    enabled: bool = True
    max_speed_m_s: float = .05
    mode_weights: tuple[float, ...] = (.3, .7, 0., 0.)
    ou_tau_s: float = 2.
    gust_duration_s: tuple[float, float] = (.5, 1.5)
    gust_wait_s: tuple[float, float] = (2., 6.)
    air_density_kg_m3: float = 1.2
    drag_coefficient: float = .47
    radius_m: float = .02
    force_limit_n: float = .002
    model: str = MODEL
    # Zero defaults preserve archived configurations/replay. Current V325/V326
    # explicitly select the user-requested five-hit shelter in make_config.
    startup_no_wind_hits: int = 0
    startup_ramp_s: float = 0.

    def __post_init__(self) -> None:
        scalars = (self.max_speed_m_s, self.ou_tau_s, self.air_density_kg_m3,
                   self.drag_coefficient, self.radius_m, self.force_limit_n)
        if (self.model != MODEL or not isinstance(self.enabled, bool)
                or not all(math.isfinite(x) for x in scalars)
                or self.max_speed_m_s < 0. or min(scalars[1:]) <= 0.
                or len(self.mode_weights) != 4
                or any(not math.isfinite(x) or x < 0. for x in self.mode_weights)
                or not math.isclose(sum(self.mode_weights), 1., abs_tol=1e-12)):
            raise ValueError('Invalid bounded horizontal wind contract')
        for low, high in (self.gust_duration_s, self.gust_wait_s):
            if not 0. < low <= high or not math.isfinite(high):
                raise ValueError('Wind durations must be finite ordered seconds')
        if (type(self.startup_no_wind_hits) is not int or self.startup_no_wind_hits < 0
                or not math.isfinite(self.startup_ramp_s) or self.startup_ramp_s < 0.):
            raise ValueError('Startup wind shelter requires nonnegative hits and ramp seconds')

    @property
    def drag_factor_kg_m(self) -> float:
        return .5 * self.air_density_kg_m3 * self.drag_coefficient * math.pi * self.radius_m**2


@dataclass(frozen=True)
class WindLesson:
    name: str
    seconds: float
    max_speed_m_s: float
    mode_weights: tuple[float, ...]
    ou_tau_s: float = 2.
    gust_duration_s: tuple[float, float] = (.5, 1.5)

    def domain(self) -> WindConfig:
        return WindConfig(max_speed_m_s=self.max_speed_m_s,
            mode_weights=self.mode_weights, ou_tau_s=self.ou_tau_s,
            gust_duration_s=self.gust_duration_s)


@dataclass(frozen=True)
class WindContinuation:
    version: int
    critic_obs_dim: int = 393 + CRITIC_EXTRA_DIM

    @property
    def origin(self):
        return parent_spec_for(self.version - 2)

    @property
    def parent_model(self) -> str:
        return self.origin.parent_model

    @property
    def profile(self) -> str:
        return f'fixed_plane_limit_cycle_qvel_rmp_gpu{self.origin.gpu_index}_v{self.version}'

    @property
    def course(self) -> tuple[WindLesson, ...]:
        from fixed_plane_wind_pose import pose_course
        return pose_course(self.version, reward_bridge_course(self.version))


def reward_bridge_course(version: int) -> tuple[WindLesson, ...]:
    """Archived seventeen-lesson reward bridge, for exact migration checks."""
    old = legacy_course(version)
    constant, slow, gust, mixed = ((.3, .7, 0., 0.), (.3, 0., .7, 0.),
                                 (.3, 0., 0., .7), (.3, .2, .2, .3))
    rows = (
        ('01_gust_reward_polish', 180., .40, gust, .75, (.25, .75)),
        ('02_constant_020_retention', 120., .20, constant, 2., (.5, 1.5)),
        ('03_constant_030', 180., .30, constant, 2., (.5, 1.5)),
        ('04_constant_040', 180., .40, constant, 2., (.5, 1.5)),
        ('05_slow_040', 180., .40, slow, .75, (.5, 1.5)),
        ('06_mixed_040_180s', 180., .40, mixed, .75, (.25, .75)),
        ('07_mixed_040_600s', 600., .40, mixed, .75, (.25, .75)),
        ('08_mixed_040_900s', 900., .40, mixed, .75, (.25, .75)),
    )
    bridge = tuple(WindLesson(f'fplc_v{version}_bridge{label}',
        seconds, speed, weights, tau, duration)
        for label, seconds, speed, weights, tau, duration in rows)
    return (*old[:7], *bridge[:6], old[7], *bridge[6:], old[8])


def legacy_course(version: int) -> tuple[WindLesson, ...]:
    """Keep archived course identities for authenticated migration and V327."""
    constant, slow, gust, mixed = ((.3, .7, 0., 0.), (.3, 0., .7, 0.),
                                 (.3, 0., 0., .7), (.3, .2, .2, .3))
    rows = (
        ('constant_005', 30., .05, constant, 2., (.5, 1.5)),
        ('constant_010', 30., .10, constant, 2., (.5, 1.5)),
        ('constant_020', 60., .20, constant, 2., (.5, 1.5)),
        ('slow_020', 60., .20, slow, 2., (.5, 1.5)),
        ('slow_030', 120., .30, slow, .75, (.5, 1.5)),
        ('gust_030', 120., .30, gust, .75, (.5, 1.5)),
        ('gust_040', 180., .40, gust, .75, (.25, .75)),
        ('mixed_040', 300., .40, mixed, .75, (.25, .75)),
        ('unchanged_consolidation', 1200., .40, mixed, .75, (.25, .75)),
    )
    return tuple(WindLesson(f'fplc_v{version}_s{i:02d}_{label}',
        seconds, speed, weights, tau, duration)
        for i, (label, seconds, speed, weights, tau, duration) in enumerate(rows, 1))


def spec_for(version: int) -> WindContinuation:
    if version not in (325, 326):
        raise ValueError('Only V325 and V326 are registered wind continuations')
    return WindContinuation(version)


SPEC = spec_for(325)
PROFILE = SPEC.profile
COURSE = SPEC.course


@lru_cache(maxsize=4)
def _config_type(parent_type: type) -> type:
    return make_dataclass('WindContinuationConfig',
        [('wind_config', WindConfig, field(default_factory=WindConfig))],
        bases=(parent_type,), frozen=True)


def make_config(parent: Any, lesson: WindLesson, *, enabled: bool = True,
                startup_shelter: bool = False) -> Any:
    if (parent.horizon_sec != 1200. or parent.ball_obs_rate_hz != 90.
            or parent.ball_flight_drag_coefficient_m_inv != 0.
            or hasattr(parent, 'wind_config')):
        raise ValueError('Wind continuation requires the unmodified drag-free parent')
    domain=lesson.domain()
    values = asdict(domain)
    values['enabled'] = enabled
    if startup_shelter:
        values.update(startup_no_wind_hits=STARTUP_NO_WIND_HITS,
                      startup_ramp_s=STARTUP_RAMP_S)
    return _config_type(type(parent))(**vars(parent), wind_config=type(domain)(**values))


def make_profile_config(parent: Any, lesson: WindLesson, version: int,
                        *, enabled: bool = True) -> Any:
    """Current V325/V326 use sheltered startup; archived V327 stays unchanged."""
    from fixed_plane_wind_reward import repair_scale, with_reward_repair
    cfg = make_config(parent, lesson, enabled=enabled,
                      startup_shelter=version in (325, 326))
    cfg = with_reward_repair(cfg, repair_scale(lesson, version))
    if version in (325, 326):
        from fixed_plane_wind_pose import with_pose_curriculum
        cfg = with_pose_curriculum(cfg, lesson, version)
        from fixed_plane_wind_history import with_history_config
        cfg = with_history_config(cfg, lesson)
    return cfg


def _bound(vector, limit):
    norm = jnp.linalg.norm(vector, axis=-1, keepdims=True)
    return vector * jnp.minimum(1., limit / jnp.maximum(norm, 1e-12))


def incremental_force(velocity_m_s, wind_m_s, cfg: WindConfig):
    """Return bounded/raw world forces (N,3); zero wind gives exactly zero.

    Horizontal wind can change vertical force through the relative-speed
    norm. This is an increment to legacy physics, not calibrated full drag.
    """
    if not cfg.enabled:
        zeros = jnp.zeros_like(velocity_m_s)
        return zeros, zeros
    relative = velocity_m_s - wind_m_s
    raw = cfg.drag_factor_kg_m * (
        jnp.linalg.norm(velocity_m_s, axis=-1, keepdims=True) * velocity_m_s
        - jnp.linalg.norm(relative, axis=-1, keepdims=True) * relative)
    # XLA may fuse only one product in the subtraction and leave a tiny
    # cancellation residual. Zero airspeed must mean exactly zero added force.
    raw = jnp.where(jnp.all(wind_m_s == 0., axis=-1, keepdims=True),
                    jnp.zeros_like(raw), raw)
    return _bound(raw, cfg.force_limit_n), raw


def _draw(keys, tag, size):
    return jax.vmap(lambda key: jax.random.uniform(
        jax.random.fold_in(key, tag), (size,), dtype=jnp.float32))(keys)


def _disk(uniforms, radius):
    angle = 2. * jnp.pi * uniforms[:, 0]
    length = radius * jnp.sqrt(uniforms[:, 1])
    return jnp.stack((length * jnp.cos(angle), length * jnp.sin(angle),
                      jnp.zeros_like(length)), axis=-1)


def initialize_wind(keys, cfg: WindConfig, time_s=0.) -> WindState:
    """Use independent fold-in streams without consuming native reset keys."""
    u = _draw(keys, 32501, 8)
    mode = jnp.minimum(jnp.sum(u[:, :1] >= jnp.cumsum(
        jnp.asarray(cfg.mode_weights))[None, :], axis=-1), 3).astype(jnp.int32)
    if not cfg.enabled:
        mode = jnp.zeros_like(mode)
    fraction = jnp.where(mode == 1, 1., jnp.where(mode == 2, .35, .15))
    background = _disk(u[:, 1:3], fraction * cfg.max_speed_m_s)
    peak_fraction=getattr(cfg,'peak_wind_fraction',0.)
    peak=(_draw(keys,32505,1)[:,0]<peak_fraction) if peak_fraction else jnp.zeros_like(mode,bool)
    direction=_disk(u[:,1:3].at[:,1].set(1.),1.)
    background=constrain_wind(background,cfg,xp=jnp)
    direction=constrain_wind(direction,cfg,xp=jnp)
    if peak_fraction:
        background=jnp.where((peak & (mode==1))[:,None],direction*cfg.max_speed_m_s,background)
        background=jnp.where((peak & (mode==2))[:,None],.75*direction*cfg.max_speed_m_s,background)
    normal = jax.vmap(lambda key: jax.random.normal(
        jax.random.fold_in(key, 32502), (3,), dtype=jnp.float32))(keys)
    ou = _bound(normal.at[:, 2].set(0.) * (.25 * cfg.max_speed_m_s),
                .65 * cfg.max_speed_m_s)
    if peak_fraction:
        ou=jnp.where((peak & (mode==2))[:,None],.25*direction*cfg.max_speed_m_s,ou)
    gust_vector=_disk(u[:,3:5],.85*cfg.max_speed_m_s)
    gust_vector=constrain_wind(gust_vector,cfg,xp=jnp)
    if peak_fraction:
        gust_vector=jnp.where(peak[:,None],direction*cfg.max_speed_m_s-background,gust_vector)
    sampled = jnp.broadcast_to(jnp.asarray(time_s, jnp.float32), mode.shape)
    started = jnp.full_like(sampled, -1.) if cfg.startup_no_wind_hits else sampled
    return WindState(mode, jnp.where((mode > 0)[:, None], background, 0.),
        jnp.where((mode == 2)[:, None], ou, 0.),
        jnp.where((mode == 3)[:, None], gust_vector, 0.),
        time_s + .5 + 2.5 * u[:, 5],
        cfg.gust_duration_s[0] + (cfg.gust_duration_s[1]-cfg.gust_duration_s[0]) * u[:, 6],
        jnp.zeros_like(mode), started, sampled)


def advance_wind(state, keys, time_s, dt_s: float, cfg: WindConfig):
    """Advance the bounded OU once per5ms; reschedule smooth gusts after quiet gaps."""
    if not cfg.enabled:
        return state
    normal = jax.vmap(lambda key: jax.random.normal(
        jax.random.fold_in(key, 32503), (3,), dtype=jnp.float32))(keys)
    alpha = math.exp(-dt_s / cfg.ou_tau_s)
    ou = alpha * state.wind_ou_m_s + math.sqrt(1.-alpha**2) * (
        .25 * cfg.max_speed_m_s) * normal.at[:, 2].set(0.)
    ou = jnp.where((state.wind_mode == 2)[:, None],
                  _bound(ou, .65 * cfg.max_speed_m_s), 0.)
    ended = (state.wind_mode == 3) & (
        time_s >= state.wind_gust_start_s + state.wind_gust_duration_s)
    u = _draw(keys, 32504, 4)
    start = time_s + cfg.gust_wait_s[0] + (
        cfg.gust_wait_s[1]-cfg.gust_wait_s[0]) * u[:, 0]
    duration = cfg.gust_duration_s[0] + (
        cfg.gust_duration_s[1]-cfg.gust_duration_s[0]) * u[:, 1]
    next_gust=_disk(u[:,2:4],.85*cfg.max_speed_m_s)
    next_gust=constrain_wind(next_gust,cfg,xp=jnp)
    peak_fraction=getattr(cfg,'peak_wind_fraction',0.)
    if peak_fraction:
        peak=_draw(keys,32506,1)[:,0]<peak_fraction
        direction=_disk(u[:,2:4].at[:,1].set(1.),1.)
        direction=constrain_wind(direction,cfg,xp=jnp)
        next_gust=jnp.where(peak[:,None],direction*cfg.max_speed_m_s-state.wind_background_m_s,next_gust)
    updated = state._replace(wind_ou_m_s=ou,
        wind_gust_vector_m_s=jnp.where(ended[:, None],
            next_gust, state.wind_gust_vector_m_s),
        wind_gust_start_s=jnp.where(ended, start, state.wind_gust_start_s),
        wind_gust_duration_s=jnp.where(ended, duration, state.wind_gust_duration_s))
    if cfg.startup_no_wind_hits:
        active = state.wind_startup_started_s >= 0.
        updated = updated._replace(**{name: jnp.where(
            active[:, None] if getattr(state, name).ndim == 2 else active,
            getattr(updated, name), getattr(state, name)) for name in (
                'wind_ou_m_s', 'wind_gust_vector_m_s',
                'wind_gust_start_s', 'wind_gust_duration_s')})
    return updated


def wind_clock_time(state, time_s, cfg: WindConfig):
    if not cfg.startup_no_wind_hits:
        return time_s
    return jnp.where(state.wind_startup_started_s >= 0., time_s, state.wind_sampled_s)


def startup_gain(state, time_s, cfg: WindConfig):
    """A smooth sin-squared airspeed ramp after five actual confirmed hits."""
    if not cfg.startup_no_wind_hits and cfg.startup_ramp_s == 0.:
        return jnp.ones_like(state.wind_startup_started_s)
    active = state.wind_startup_started_s >= 0.
    if cfg.startup_ramp_s == 0.:
        return active.astype(jnp.float32)
    phase = jnp.clip((time_s-state.wind_startup_started_s)/cfg.startup_ramp_s, 0., 1.)
    return jnp.where(active, jnp.sin(.5*jnp.pi*phase)**2, 0.)


def nominal_wind_velocity(state, time_s, cfg: WindConfig):
    time_s = wind_clock_time(state, time_s, cfg)
    phase = jnp.clip((time_s-state.wind_gust_start_s) / state.wind_gust_duration_s, 0., 1.)
    envelope = jnp.where((phase > 0.) & (phase < 1.), jnp.sin(jnp.pi * phase)**2, 0.)
    velocity = (state.wind_background_m_s + state.wind_ou_m_s
                + envelope[:, None] * state.wind_gust_vector_m_s)
    velocity=constrain_wind(velocity,cfg,xp=jnp)
    return jnp.where((state.wind_mode > 0)[:, None],
                     _bound(velocity, cfg.max_speed_m_s), 0.)


def wind_velocity(state, time_s, cfg: WindConfig):
    velocity = nominal_wind_velocity(state, time_s, cfg)
    if not cfg.startup_no_wind_hits and cfg.startup_ramp_s == 0.:
        return velocity
    return velocity * startup_gain(state, time_s, cfg)[:, None]


def critic_features(state, time_s, cfg: WindConfig):
    """18 training-only values: wind3/bg3/OU3/gust3/wait1/duration1/mode4."""
    scale = max(cfg.max_speed_m_s, .05)
    return jnp.concatenate((wind_velocity(state, time_s, cfg) / scale,
        state.wind_background_m_s / scale, state.wind_ou_m_s / scale,
        state.wind_gust_vector_m_s / scale,
        jnp.clip((state.wind_gust_start_s-wind_clock_time(state, time_s, cfg)) / 6., -1., 1.)[:, None],
        (state.wind_gust_duration_s / 1.5)[:, None],
        jax.nn.one_hot(state.wind_mode, 4)), axis=-1)


def append_wind_critic(train_state):
    """Zero only18 new critic rows and their moments, retaining all prior leaves."""
    def extend(tree):
        layer = dict(tree['v']['l1'])
        if layer['w'].shape[0] != 393:
            raise ValueError('Wind migration requires the exact393-D critic prefix')
        layer['w'] = jnp.pad(jnp.asarray(layer['w']), ((0, CRITIC_EXTRA_DIM), (0, 0)))
        return {**tree, 'v': {**tree['v'], 'l1': layer}}
    return train_state._replace(params=extend(train_state.params),
        opt=train_state.opt._replace(m=extend(train_state.opt.m), v=extend(train_state.opt.v)))
