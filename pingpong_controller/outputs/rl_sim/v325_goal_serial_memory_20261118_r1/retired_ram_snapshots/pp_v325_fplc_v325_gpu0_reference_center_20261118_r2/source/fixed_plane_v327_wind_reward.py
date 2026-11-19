"""V327: reward-only continuation of the stopped V325 full learner."""
from dataclasses import dataclass, replace
import math
import re

from fixed_plane_v325_wind import WindContinuation, legacy_course

SOURCE_SHA256 = 'b0b059e28067f4920eaf224beeed05bc1c9427e1828b3cffbabf54602c17ed46'
SOURCE_STEP = 1676017664
SOURCE_UPDATE = 506
SOURCE_ADAM_T = 419067
REWARD_OVERRIDES = {
    'fixed_plane_v159_mean_center_penalty_weight': 4.,
    'fixed_plane_v92_signed_height_reward_scale': 9.,
    'fixed_plane_v156_adjacent_height_penalty_multiplier': 4.,
    'early_termination_remaining_horizon_penalty': 40.,
}


@dataclass(frozen=True)
class RewardContinuation(WindContinuation):
    base_version: int = 325

    @property
    def origin(self):
        return WindContinuation(self.base_version).origin

    @property
    def course(self):
        return legacy_course(self.version)


SPEC = RewardContinuation(327)


def reward_config(parent):
    """Change four reward weights; use the existing frozen native formulae."""
    return replace(parent, **REWARD_OVERRIDES)


def validate_source(sha256: str, payload: dict) -> None:
    """Pin the entire stopped learner before crossing the reward boundary."""
    if (sha256 != SOURCE_SHA256
            or payload.get('step') != SOURCE_STEP
            or payload.get('global_update') != SOURCE_UPDATE
            or payload.get('stage_index') != 0
            or payload.get('stage_converged', False)
            or payload.get('args', {}).get('curriculum_profile')
                != WindContinuation(325).profile
            or payload.get('opt') is None or int(payload['opt'].t) != SOURCE_ADAM_T):
        raise ValueError('V327 requires the pinned V325/u506 complete learner')


def expected_reward_delta(metrics):
    """Reconstruct the four weight changes from the unchanged native terms."""
    return (metrics['reward/fixed_plane_v159_mean_center_penalty'] * (4./1.5 - 1.)
        + (metrics['v92_absolute_height_reward']
           + metrics['v92_unresolved_contact_penalty']) * (9./6. - 1.)
        + metrics['v92_adjacent_height_penalty'] * (9.*4./(6.*3.) - 1.)
        + metrics['reward/early_termination_remaining_horizon_penalty'] * (40./4. - 1.))


def validate_reward_pair(report, binding):
    """Require paired transitions and the native reward-delta identity on CUDA."""
    from fixed_plane_v323_ball_obs_noise import canonical
    pair = report.get('reward_only_pair', {})
    if (canonical(report.get('binding')) != canonical(binding)
            or report.get('backend') != 'gpu'
            or pair.get('steps') != 1200 or pair.get('n_envs') != 64
            or not re.fullmatch('[a-f0-9]{64}', pair.get('reference_digest', ''))
            or pair.get('reference_digest') != pair.get('candidate_digest')
            or pair.get('reward_overrides') != REWARD_OVERRIDES
            or pair.get('reward_changed_steps', 0) < 1
            or len(pair.get('samples', [])) != 6):
        raise ValueError('V327 requires a matching six-second CUDA reward-only pair')
    for row in pair['samples']:
        if (len(row) != 4 or not all(math.isfinite(x) for x in row)
                or row[0] != 64 or not 0. <= row[1] <= 1e-3
                or row[2] != 0. or not 0. <= row[3] <= 64):
            raise ValueError('V327 reward pair has a numeric or transition mismatch')


def qualify_reward_pair(stack, args, train_state, binding, runtime, stop):
    """Use identical raw actions and wind to compare old/new native rewards."""
    import hashlib
    import json
    import subprocess
    import jax
    import jax.numpy as jnp
    import numpy as np
    from fixed_plane_v325_wind import make_config
    from train_fixed_plane_v321 import freeze_lanes
    from train_fixed_plane_wind import numeric_digest

    old = stack.env_class(stack.xml, n_envs=64,
        cfg=make_config(stack.legacy_native_cfg, stack.spec.course[0]))
    new = stack.env_class(stack.xml, n_envs=64, cfg=stack.cfg)
    rng, reset_key = jax.random.split(jax.random.PRNGKey(args.seed))
    keys = jax.random.split(reset_key, 64)
    a, ao = jax.jit(old.reset)(keys)
    b, bo = jax.jit(new.reset)(keys)
    reference, candidate = [numeric_digest((a, ao))], [numeric_digest((b, bo))]

    @jax.jit
    def chunk(a, ao, b, bo, active, rng):
        def tick(carry, _):
            a, ao, b, bo, active, rng = carry
            rng, key = jax.random.split(rng)
            mean = stack.ppo.policy_mean(train_state.params, ao)
            noise = jnp.exp(stack.ppo.effective_log_std(train_state.params['log_std'],
                args.min_log_std, args.max_log_std)) * jax.random.normal(key, mean.shape)
            action = mean + jnp.where((jnp.arange(64) >= 32)[:, None], noise, 0.)
            an, aon, ar, ad, am = old.step(a, action)
            bn, bon, br, bd, bm = new.step(b, action)
            failed = am['terminated'] > .5
            expected = expected_reward_delta(am)
            packet = dict(error=jnp.max(jnp.abs((br-ar)-expected)),
                done_mismatch=jnp.sum(ad != bd), changed=jnp.sum(active & (jnp.abs(br-ar) > 1e-6)),
                finite=jnp.all(jnp.isfinite(ar)) & jnp.all(jnp.isfinite(br)))
            return (freeze_lanes(a, an, active, jax, jnp),
                freeze_lanes(ao, aon, active, jax, jnp),
                freeze_lanes(b, bn, active, jax, jnp),
                freeze_lanes(bo, bon, active, jax, jnp), active & ~failed, rng), packet
        carry, packets = jax.lax.scan(tick, (a, ao, b, bo, active, rng), None, length=200)
        return carry, dict(error=jnp.max(packets['error']),
            done_mismatch=jnp.sum(packets['done_mismatch']), changed=jnp.sum(packets['changed']),
            finite=jnp.all(packets['finite']))

    active = jnp.ones(64, bool)
    samples, changed = [], 0
    for second in range(1, 7):
        if stop[0]:
            raise InterruptedError('V327 reward pair cancelled')
        (a, ao, b, bo, active, rng), packet = chunk(a, ao, b, bo, active, rng)
        packet = jax.device_get(packet)
        reference.append(numeric_digest((a, ao, old.get_critic_obs(a, ao))))
        candidate.append(numeric_digest((b, bo, new.get_critic_obs(b, bo))))
        samples.append([64 if packet['finite'] else 0, float(packet['error']),
                        int(packet['done_mismatch']), int(np.asarray(active).sum())])
        changed += int(packet['changed'])
        print(f'[V327 reward pair] t={second}/6s {samples[-1]}', flush=True)
        temperature = float(subprocess.check_output(['nvidia-smi', '-i',
            stack.spec.origin.gpu_uuid, '--query-gpu=temperature.gpu',
            '--format=csv,noheader,nounits'], text=True))
        if temperature >= 78.:
            raise RuntimeError('V327 reward pair reached78C')
    report = dict(binding=binding, backend=jax.default_backend(),
        scope='reward-only frozen transition/delta audit; not effectiveness evidence',
        reward_only_pair=dict(steps=1200, n_envs=64, samples=samples,
            reward_overrides=REWARD_OVERRIDES, reward_changed_steps=changed,
            reference_digest=hashlib.sha256(''.join(reference).encode()).hexdigest(),
            candidate_digest=hashlib.sha256(''.join(candidate).encode()).hexdigest()))
    (runtime/'wind_qualification_attempt.json').write_text(json.dumps(report, indent=2))
    validate_reward_pair(report, binding)
    (runtime/'wind_qualification.json').write_text(json.dumps(report, indent=2))
