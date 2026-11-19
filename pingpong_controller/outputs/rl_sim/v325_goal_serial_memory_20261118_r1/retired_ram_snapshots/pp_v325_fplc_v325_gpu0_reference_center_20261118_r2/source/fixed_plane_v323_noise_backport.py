"""Private camera-noise state adapters for the immutable M12/M13 environments.

Only V323 constructs this subclass. Frozen source files and native classes
remain untouched. Every replacement is anchored to the authenticated source.
V324 uses M13's native camera-noise path with an explicit combined state
schema retaining all six native V225 reset-cohort fields.
"""
from __future__ import annotations

from collections import namedtuple
import inspect
from pathlib import Path
import textwrap
from typing import Any

from fixed_plane_v323_ball_obs_noise import SPECS, digest_file


def _clone(function, replacements: tuple[tuple[str, str], ...], namespace: dict):
    source = textwrap.dedent(inspect.getsource(function))
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError('Frozen M12 camera-noise backport anchor changed')
        source = source.replace(old, new)
    scope = {**function.__globals__, **namespace}
    exec(compile(source, function.__code__.co_filename, 'exec'), scope)
    return scope[function.__name__]


def make_m12_noise_base(base: type) -> type:
    """Extend reset, state transport and pre-cache sampling without new physics."""
    import jax.numpy as jnp
    from ball_position_noise_ar1 import initialize_noise, advance_noise

    native_path = Path(inspect.getfile(base))
    if digest_file(native_path) != SPECS[323].native_env_sha256:
        raise ValueError('V323 backport requires the frozen M12 environment')
    native_state = base.reset.__globals__['EnvState']
    noise_state = namedtuple('M12CameraNoiseState',
        (*native_state._fields, 'ball_position_noise_epsilon_m'))
    namespace = dict(_NoiseState=noise_state, _initialize_noise=initialize_noise,
                     _advance_noise=advance_noise)
    # Match M10's stationary reset draw and its independent fold-in stream.
    reset = _clone(base.reset, ((
        '    if self.fixed_plane_v92_history_obs_enabled:\n',
        '    ball_position_noise_epsilon_m = jnp.zeros((n_envs, 3), dtype=jnp.float32)\n'
        '    if self.cfg.ball_position_noise_ar1_enabled:\n'
        '        noise_keys = jax.vmap(lambda key: jax.random.fold_in(key, 25401))(next_keys)\n'
        '        ball_position_noise_epsilon_m = _initialize_noise(\n'
        '            noise_keys, std_m=self.cfg.ball_position_noise_std_m)\n'
        '        reset_ball_obs_pos = reset_ball_obs_pos + jnp.asarray(\n'
        '            self.cfg.ball_position_noise_mean_m) + ball_position_noise_epsilon_m\n'
        '    if self.fixed_plane_v92_history_obs_enabled:\n'), (
        '    base_obs = self._make_obs(\n',
        '    state = _NoiseState(*state, ball_position_noise_epsilon_m)\n'
        '    base_obs = self._make_obs(\n')), namespace)
    step = _clone(base._step_impl, ((
        '    obs_state = next_state._replace(prev_racket_pos=state.prev_racket_pos)\n',
        '    next_state = _NoiseState(*next_state, state.ball_position_noise_epsilon_m)\n'
        '    obs_state = next_state._replace(prev_racket_pos=state.prev_racket_pos)\n'),), namespace)
    pipeline = _clone(base._apply_observation_pipeline, ((
        '    sampled_pos = true_bpos + pos_noise\n',
        '    if self.cfg.ball_position_noise_ar1_enabled:\n'
        '        epsilon_m = _advance_noise(state.ball_position_noise_epsilon_m, key_pos_noise,\n'
        '            dt_s=(state.step_count-state.last_ball_obs_step).astype(jnp.float32)*self.dt,\n'
        '            refresh=refresh, std_m=self.cfg.ball_position_noise_std_m,\n'
        '            rho_at_90hz=self.cfg.ball_position_noise_rho_at_90hz)\n'
        '        state = state._replace(ball_position_noise_epsilon_m=epsilon_m)\n'
        '        pos_noise = jnp.asarray(self.cfg.ball_position_noise_mean_m) + epsilon_m\n'
        '    sampled_pos = true_bpos + pos_noise\n'),), namespace)

    class M12NoiseEnv(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            if self.critic_obs_dim != 393:
                raise ValueError('V323 requires the unchanged 393-D native critic')
            self.critic_extra_dim += 3
            self.critic_obs_dim += 3

        def get_critic_obs(self, state, obs):
            prefix = super().get_critic_obs(state, obs)
            return jnp.concatenate((prefix, state.ball_position_noise_epsilon_m
                / jnp.asarray(self.cfg.ball_position_noise_std_m)), axis=-1)

    M12NoiseEnv.reset = reset
    M12NoiseEnv._step_impl = step
    M12NoiseEnv._apply_observation_pipeline = pipeline
    return M12NoiseEnv


def make_m13_noise_base(base: type) -> type:
    """Combine existing camera and V225 schemas without altering either path.

    The historical environment supports each schema separately. Both its
    reset and physical-step constructors append V225 fields after the noise
    state; bind that constructor privately to the combined field ordering.
    All original arithmetic and native cohort labels remain unchanged.
    """
    if digest_file(Path(inspect.getfile(base))) != SPECS[324].native_env_sha256:
        raise ValueError('V324 requires the frozen M13 environment')
    namespace = base.reset.__globals__
    fields = namespace['CameraPositionNoiseEnvState']._fields
    native_fields = namespace['EnvState']._fields
    cohort_fields = namespace['V225NearSideRecoveryEnvState']._fields[len(native_fields):]
    combined = namedtuple('V324CameraNearSideEnvState', (*fields, *cohort_fields))
    replacements = {'V225NearSideRecoveryEnvState': combined}

    class M13NoiseEnv(base):
        pass

    M13NoiseEnv.reset = _clone(base.reset, (), replacements)
    M13NoiseEnv._step_impl = _clone(base._step_impl, (), replacements)
    return M13NoiseEnv
