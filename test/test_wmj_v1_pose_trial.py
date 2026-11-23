"""Opt-in reward integration preserves dynamics and full learning provenance."""

from collections import namedtuple
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def fixture_runtime():
    from world_model.pose_margin import PoseMarginContract
    transition = namedtuple('Transition', 'reward reward_terms valid model_truncated terminated')
    @dataclass(frozen=True)
    class Runtime:
        course_step: object
        control: object
    def step(angle):
        # Rotation around X; valid terminal is first row, second is padding.
        pose = jnp.stack((jnp.zeros_like(angle),) * 3 +
                         (jnp.ones_like(angle), jnp.zeros_like(angle), jnp.zeros_like(angle),
                          jnp.zeros_like(angle), jnp.cos(angle), jnp.sin(angle)), axis=-1)
        state = SimpleNamespace(world=SimpleNamespace(cached_racket_pose=pose))
        tr = transition(jnp.ones(3), {'base': jnp.ones(3)},
                        jnp.array([True, False, True]), jnp.array([False, False, True]),
                        jnp.array([True, False, False]))
        return state, tr
    runtime = Runtime(step, SimpleNamespace(projection=SimpleNamespace(reference_rotation=jnp.eye(3))))
    return runtime, replace(PoseMarginContract(), enabled=True, weight_per_s=4.)


def test_default_runtime_is_original_and_enabled_composition_keeps_state_masks():
    from world_model.pose_margin import PoseMarginContract
    from world_model.pose_trial import with_pose_reward
    runtime, contract = fixture_runtime()
    assert with_pose_reward(runtime, PoseMarginContract()) is runtime
    assert with_pose_reward(runtime, replace(contract, weight_per_s=0.)) is runtime
    angles = jnp.full((3,), contract.limit_rad)
    old_state, old = runtime.course_step(angles)
    state, tr = with_pose_reward(runtime, contract).course_step(angles)
    np.testing.assert_array_equal(state.world.cached_racket_pose, old_state.world.cached_racket_pose)
    for field in ('valid', 'model_truncated', 'terminated'):
        np.testing.assert_array_equal(getattr(tr, field), getattr(old, field))
    np.testing.assert_allclose(tr.reward, [.98, 1., 1.], atol=1e-7)
    np.testing.assert_allclose(tr.reward, sum(tr.reward_terms.values()), atol=1e-7)
    assert list(old.reward_terms) == ['base']


def test_actor_gradient_receives_terminal_margin_signal_only_on_trusted_interval():
    from world_model.pose_trial import with_pose_reward
    runtime, contract = fixture_runtime()
    step = with_pose_reward(runtime, contract).course_step
    gradient = jax.grad(lambda a:jnp.sum(step(a)[1].reward))(jnp.full((3,), contract.limit_rad))
    assert gradient[0] < 0.
    np.testing.assert_array_equal(gradient[1:], 0.)


def test_gate_rejects_wrong_reward_initial_pool_or_optimizer_authorization():
    from world_model.pose_trial import verify_gate
    ref = {'sha256': 'plan'}
    evidence = {'inherited_learning_sha256': 'learning', 'initial_session_sha256': 'pool'}
    gate = dict(status='POSE_TRIAL_NO_UPDATE_PASS', plan=ref, mode='fp32',
                selected_microbatch=256, runtime=evidence, runtime_signature='new',
                world_updates=0, actor_updates=0, critic_updates=0,
                initial_learning_unchanged=True, candidate_contract={'weight_per_s': 4.})
    verify_gate(gate, ref, evidence, 'new', 256, gate['candidate_contract'])
    for change in ({'runtime_signature': 'old'}, {'actor_updates': 1},
                   {'runtime': dict(evidence, initial_session_sha256='other')},
                   {'candidate_contract': {'weight_per_s': 0.}}):
        with pytest.raises(ValueError):
            verify_gate(dict(gate, **change), ref, evidence, 'new', 256, gate['candidate_contract'])


def test_pose_trial_storage_extension_is_exact_and_keeps_free_space_floor():
    import copy
    import json
    from world_model.contracts import repository_root, validate_config
    cfg = json.loads((repository_root() / 'pingpong_controller/tools/rl_2real/WMJ_V1_CONFIG.json').read_text())
    cfg['resources']['max_new_bytes'] = 6593445888
    validate_config(cfg)
    for altered in (6593445889, 6576668673):
        bad = copy.deepcopy(cfg); bad['resources']['max_new_bytes'] = altered
        with pytest.raises(ValueError, match='Storage budget'):
            validate_config(bad)
    bad = copy.deepcopy(cfg); bad.pop('pose_margin_trial')
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(bad)
    cfg['resources']['min_free_bytes'] = 8 * 2**30 - 1
    with pytest.raises(ValueError, match='Storage budget'):
        validate_config(cfg)
