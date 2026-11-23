"""CPU replay of full seeded controls against coherent recorded V324 commands.

Measured robot states condition projection here, isolating known controls from
learned dynamics. This audit cannot qualify full own-feedback world behavior.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_support_replay import load_support_chain
from .contracts import config_hash, file_hash, load_config, repository_root
from .control_seed import read_control_seed
from .fit_robot import summarize_errors
from .native_stack import build_native_environment
from .resources import StorageBudget
from .support_data import SupportReplayReader
from .world_control import ControlContract, control_step


def audit(cfg, reader, *, seed: int, diagnostic_trials: int = 0, guard=None, component_diagnostics=False):
    bundle = build_native_environment(cfg, n_envs=1)
    contract = ControlContract.from_native(bundle.environment)
    _, chain = load_support_chain(cfg, reader.report['geometry'])
    rng = np.random.default_rng(seed)
    names = list(reader.records)
    results, all_errors = [], {}
    component_errors = {}
    if component_diagnostics:
        from .audit_control_components import make_component_replays
        component_replays = make_component_replays(chain, contract)
    metrics = ('reference_q_rad', 'accepted_normalized', 'delayed_q_rad',
               'delayed_dq_rad_s', 'delayed_ddq_rad_s2')

    @jax.jit
    def replay(initial, context, raw, measured):
        def tick(state, sample):
            action, q = sample
            after, trace = control_step(state, action, q, context, chain=chain, contract=contract)
            return after, dict(reference=after.reference_q_rad, accepted=after.previous_accepted_action,
                **trace['delayed_commands'])
        return jax.lax.scan(tick, initial, (raw, measured))

    count = len(names) if not diagnostic_trials else min(2*diagnostic_trials, len(names))
    for trial in range(0, count, 2):
        if guard is not None:
            guard.check()
        record = reader.records[names[trial]]
        validation = [i for i,g in enumerate(record['record']['groups']) if g['split'] == 'validation']
        if len(validation) < 32:
            raise ValueError('Not enough independent validation lanes')
        lanes = np.sort(rng.choice(validation, 32, replace=False))
        state, context = read_control_seed(record['first_seed_ref'],
            context_sha256=reader.report['context_sha256'], lanes=lanes)
        stream_names = ('action__raw_normalized', 'action__accepted_normalized',
            'state__q_ref_rad', 'state__q_rad', 'state__support_q_rad',
            'substep__delayed_q_command', 'substep__delayed_dq_command', 'substep__delayed_ddq_command')
        chunks = [reader.read(key, names=set(stream_names)) for key in names[trial:trial+2]]
        actions = np.concatenate([chunk['action__raw_normalized'][:, lanes] for chunk in chunks])
        measured = np.concatenate([np.concatenate((chunk['state__q_rad'][:-1, lanes],
            chunk['state__support_q_rad'][:-1, lanes]), axis=-1) for chunk in chunks])
        labels = [np.concatenate([chunk['state__q_ref_rad'][1:, lanes] for chunk in chunks]),
                  np.concatenate([chunk['action__accepted_normalized'][:, lanes] for chunk in chunks])]
        labels.extend(np.concatenate([chunk['substep__delayed_'+key+'_command'][:, :, lanes]
                                      for chunk in chunks]) for key in ('q', 'dq', 'ddq'))
        started = time.monotonic()
        _, trace = jax.device_get(replay(state, context, jnp.asarray(actions), jnp.asarray(measured)))
        if component_diagnostics:
            refs_before = np.concatenate([chunk['state__q_ref_rad'][:-1,lanes] for chunk in chunks])
            for index_name in ('target_buffer_index','output_buffer_index'):
                indices = np.asarray(getattr(state.rmp,index_name))
                if np.any(indices != indices[0]):
                    raise ValueError('Native batched diagnostic requires a genuinely common initial phase')
            for mode, function in component_replays.items():
                if mode == 'projection':
                    prediction = jax.device_get(function(context,jnp.asarray(actions),jnp.asarray(measured),jnp.asarray(refs_before)))
                    pairs = ((metrics[0],prediction['reference'],labels[0]),(metrics[1],prediction['accepted'],labels[1]))
                else:
                    prediction = jax.device_get(function(state,context,jnp.asarray(labels[0])))
                    pairs = tuple((name,prediction[key],label) for name,key,label in zip(metrics[2:],('q','dq','ddq'),labels[2:]))
                for name,predicted,actual in pairs:
                    component_errors.setdefault(mode,{}).setdefault(name,[]).append(np.abs(predicted-actual).ravel())
        errors = {}
        for name, predicted, actual in zip(metrics,
                (trace['reference'], trace['accepted'], trace['q'], trace['dq'], trace['ddq']), labels):
            value = np.abs(predicted-actual).ravel()
            all_errors.setdefault(name, []).append(value)
            errors[name] = summarize_errors([value])
        results.append(dict(trial=trial//2, lane_count=len(lanes), controls=len(actions),
            lanes=lanes.tolist(), root_seed_families=[record['record']['groups'][i]['root_seed_family'] for i in lanes],
            errors=errors, compile_included=trial == 0, replay_and_transfer_s=time.monotonic()-started))
        print(f'[CONTROL] trial{trial//2+1} maximums={dict((k,v["maximum"]) for k,v in errors.items())}', flush=True)
    return dict(overall={key:summarize_errors(values) for key,values in all_errors.items()},
                recorded_reference_diagnostics={mode:{key:summarize_errors(values) for key,values in entries.items()}
                                                for mode,entries in component_errors.items()},
                trials=results, population='validation_only', seed=seed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026110904)
    parser.add_argument('--diagnostic-trials', type=int, default=0, choices=(0, 1))
    args = parser.parse_args(argv)
    if jax.default_backend() != 'cpu':
        parser.error('Use explicit JAX_PLATFORMS=cpu for this bounded control audit')
    cfg, root = load_config(), repository_root()
    r = cfg['resources']
    budget = StorageBudget(root/r['output_root'], r['max_new_bytes'], r['min_free_bytes'])
    budget.check(additional_bytes=2*1024**2)
    if args.output.exists():
        raise FileExistsError('Existing control evidence is protected')
    reader = SupportReplayReader(args.support_replay, budget, max_bytes=0)
    started = time.monotonic()
    result = audit(cfg, reader, seed=args.seed, diagnostic_trials=args.diagnostic_trials)
    tolerance = dict(reference_q_rad=.0001, accepted_normalized=.002,
        delayed_q_rad=.0001, delayed_dq_rad_s=.001, delayed_ddq_rad_s2=.02)
    passed = all(value['nonfinite'] == 0 and value['maximum'] <= tolerance[key]
                 for key,value in result['overall'].items())
    result.update(schema='wmj_native_seed_control_audit_v1', training_version='WMJ-V1',
        status='CONTROL_REPLAY_PASS' if passed else 'CONTROL_REPLAY_NOT_PASSED',
        scope='native_recorded_q_feedback_not_learned_physics', reference_maximum_tolerances=tolerance,
        dataset_sha256=reader.identity, config_sha256=config_hash(cfg),
        timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), invocation=sys.argv,
        source_sha256={name:file_hash(Path(__file__).parent/name) for name in
                      ('world_control.py', 'control_seed.py', 'control_projection.py', 'control_rmp.py', 'audit_world_control.py')},
        wall_s=time.monotonic()-started, actor_updates=0, critic_updates=0, world_model_ready=False)
    print(dict(path=str(args.output.absolute()), sha256=write_json_unique(args.output, result, budget),
               status=result['status']), flush=True)


if __name__ == '__main__':
    main()
