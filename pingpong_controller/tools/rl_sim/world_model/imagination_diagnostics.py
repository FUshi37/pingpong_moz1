"""Frozen continuation and piecewise gradient checks, without optimization."""

from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .imagined_rollout import actor_window_objective
from .imagination_checkpoint import save_checkpoint, restore_checkpoint
from .imagination_learner import commit_policy_windows


def check_continuation_and_direction(session, gradient, windows, proposals, run, *,
        robot, auxiliary, options, signature, output_dir, guard):
    """Compare window boundaries and same-mode output-bias perturbations.

    Tests never assign a perturbed actor to the learner. The finite-difference
    actors are temporary diagnostic branches from the same complete state/RNG.
    """
    from .train_imagination import tree_difference, timed
    started = time.monotonic()
    state = session.pool.environment
    keys = session.pool.exploration_rng
    first, _ = timed(run, state, keys, proposals)
    second, _ = timed(run, first.environment, first.exploration_rng, proposals)
    joined, _ = timed(run, state, keys, jnp.concatenate((proposals, proposals)))
    continuous_difference = tree_difference(second.environment, joined.environment)
    if continuous_difference > 1e-4:
        raise ValueError('Forward state changed at an artificial gradient-window boundary')
    np.testing.assert_array_equal(second.exploration_rng, joined.exploration_rng)
    result = dict(cross_window_max_abs=continuous_difference,
        first_window_valid_steps=np.asarray(first.transitions.valid).sum(axis=0).tolist(),
        second_window_valid_steps=np.asarray(second.transitions.valid).sum(axis=0).tolist())
    guard.check()
    bias_gradient = np.asarray(gradient['out']['b'])
    norm = float(np.linalg.norm(bias_gradient))
    if norm <= 1e-10:
        result['finite_difference'] = dict(status='UNRESOLVED', reason='zero_output_bias_gradient')
    else:
        direction = jnp.asarray(bias_gradient/norm)

        @jax.jit
        def branch(offset):
            actor = dict(session.actor, out=dict(session.actor['out'],
                b=session.actor['out']['b']+offset*direction))
            loss, window = actor_window_objective(actor, session.target_critic, robot, auxiliary,
                state, keys, proposals, **options)
            events = (window.transitions.terminated, window.transitions.task_events['valid_release'],
                      window.transitions.task_events['valid_apex'])
            return loss, events

        rows = []
        for epsilon in (.0003, .001, .003, .01, .02):
            plus, _ = timed(branch, epsilon)
            minus, _ = timed(branch, -epsilon)
            same_mode = all(np.array_equal(a, b) for a, b in
                            zip(jax.tree.leaves(plus[1]), jax.tree.leaves(minus[1])))
            difference = float((plus[0]-minus[0])/(2*epsilon))
            rows.append(dict(epsilon=epsilon, automatic_derivative=norm,
                central_difference=difference, same_recorded_event_ticks=same_mode,
                relative_error=abs(difference-norm)/max(norm, abs(difference), 1e-8),
                direction_agrees=difference > 0.))
        # Timing/contact boundaries are reported, not silently forced smooth.
        result['finite_difference'] = dict(rows=rows, optimizer_updates=0,
            status='PASS' if any(r['same_recorded_event_ticks'] and r['relative_error']<=.1
                                 for r in rows) else 'UNRESOLVED',
            qualification='recorded200Hz events only; internal contact/projection branches may change')
    guard.check()
    # Capture a complete state each5 ms in a small diagnostic batch. This path
    # is separate from normal rollouts and never retained for the1024-lane pool.
    from .full_prior_world import world_actor_observation
    from .imagined_env import imagined_step
    from .migration import actor_mean

    def capture_tick(carry, goals):
        current, rng = carry
        split = jax.vmap(lambda k:jax.random.split(k, 2))(rng)
        noise = jax.vmap(lambda k:jax.random.normal(k, (7,)))(split[:, 1])
        action = actor_mean(session.actor, world_actor_observation(current.world, current.task.goal_xyz))
        action = action+options['exploration_std']*noise
        after, transition = imagined_step(robot, auxiliary, current, action, goals,
            **{k:v for k,v in options.items() if k != 'exploration_std'})
        return (after, split[:, 0]), (current, rng, transition.valid)

    @jax.jit
    def capture():
        return jax.lax.scan(capture_tick, (state, keys), jnp.concatenate((proposals, proposals)))[1]

    (states, rngs, valid), capture_s = timed(capture)
    contact = np.asarray(states.world.physics.ball.previous_contact)
    valid = np.asarray(valid)
    criteria = dict(flight=valid & ~contact,
        precontact=valid & ~contact & np.concatenate((contact[1:], np.zeros_like(contact[:1]))),
        incontact=valid & contact)
    snapshots = {}
    for phase, mask in criteria.items():
        indices = np.argwhere(mask)
        if not len(indices):
            snapshots[phase] = dict(status='NOT_OBSERVED', candidate_count=0)
            continue
        control_index, lane = map(int, indices[0])
        selected = jax.tree.map(lambda x:x[control_index, lane:lane+1], states)
        selected_keys = rngs[control_index, lane:lane+1]
        pool = session.pool._replace(environment=selected, exploration_rng=selected_keys)
        snapshot = dict(learner=session._replace(pool=pool), frozen_robot=robot,
                        frozen_auxiliary=auxiliary)
        ref = save_checkpoint(Path(output_dir)/('resume_'+phase+'.json'), snapshot,
            signature=signature, budget=guard.budget,
            metadata=dict(phase=phase, control_index=control_index, lane=lane,
                          actor_updates=0, critic_updates=0))
        restored = restore_checkpoint(ref, snapshot, signature=signature)['learner']
        task_goals = jnp.broadcast_to(selected.task.goal_xyz, (64, 1, 3))
        expected, _ = timed(run, selected, selected_keys, task_goals)
        replay, _ = timed(run, restored.pool.environment, restored.pool.exploration_rng, task_goals)
        difference = tree_difference(expected, replay)
        if difference != 0.:
            raise ValueError('Resume differs in '+phase)
        snapshots[phase] = dict(status='PASS', checkpoint=ref, max_abs=difference,
            control_index=control_index, lane=lane, candidate_count=len(indices),
            continued_valid_steps=int(np.asarray(replay.transitions.valid).sum()))
        guard.check()
    result.update(phase_snapshots=snapshots, capture_compile_and_run_s=capture_s,
                  wall_s=time.monotonic()-started)
    return result
