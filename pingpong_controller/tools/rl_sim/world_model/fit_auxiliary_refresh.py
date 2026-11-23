"""Bounded auxiliary-only continuation on equal old/new training populations."""

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_array_archive_unique, write_json_unique
from .auxiliary_refresh import restore_auxiliary
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .fit_solver_auxiliary import evaluate, family_pool, load_labels, sample_pool, summarize_prediction
from .neural import adam_update
from .runtime_guard import RunGuard
from .solver_auxiliary import WIDTHS, supervised_loss
from .state_adapter import numeric_tree_digest, pack_numeric_state


def fit(cfg, args, guard, report):
    settings = cfg['auxiliary_refresh']
    if args.updates != settings['maximum_additional_updates'] or args.updates > 4096:
        raise ValueError('One fixed continuation of at most4096 updates is authorized')
    old_path = authenticate(repository_root(), settings['old_labels'])
    new_path = authenticate(repository_root(), settings['new_labels'])
    if args.labels.resolve() != new_path.parent:
        raise ValueError('Unexpected supplemental solver labels')
    datasets, labels = zip(*(load_labels(path.parent, cfg, guard) for path in (old_path, new_path)))
    roots = [{g['root_seed_family'] for record in label['records'] for g in record['groups']}
             for label in labels]
    if roots[0] & roots[1]:
        raise ValueError('Old/new acquisitions must have disjoint root families')
    restored, parent = restore_auxiliary(cfg, settings['parent_checkpoint'])
    parameters, optimizer, scale = (restored[k] for k in ('parameters', 'optimizer', 'scale'))
    if int(optimizer.step) != parent['updates']:
        raise ValueError('Parent Adam count disagrees with the checkpoint update count')
    before = numeric_tree_digest(parameters); normalization = numeric_tree_digest(scale)
    pools = [[family_pool(rows, (rows['split'] == 0) & mask)
              for mask in (rows['overlap'], ~rows['overlap'])] for rows in datasets]
    valid = [np.flatnonzero(rows['split'] == 1) for rows in datasets]
    if any(not len(index) for index in valid):
        raise ValueError('Both source populations require independent validation')
    rng = np.random.default_rng(settings['seed'])
    report.update(parent_checkpoint=settings['parent_checkpoint'], parent_updates=parent['updates'],
        label_references=dict(old=settings['old_labels'], new=settings['new_labels']),
        normalization=parent['normalization'], parent_parameter_sha256=before,
        parent_optimizer_sha256=numeric_tree_digest(optimizer), adam_step_before=int(optimizer.step),
        recipe=dict(updates=args.updates, batch_size=512, learning_rate=.0003, gradient_clip=1.,
            old_contact=128, old_noncontact=128, new_contact=128, new_noncontact=128,
            sampling='family_uniform_within_source_and_contact_bin',
            normalization='exact_parent', optimizer='parent_moments_and_count_restored',
            loss='unchanged_six_normalized_component_MSE'),
        source_counts={name:dict(train=int((rows['split']==0).sum()), validation=len(index),
            train_families=len(np.unique(rows['family_id'][rows['split']==0])),
            validation_families=len(np.unique(rows['family_id'][rows['split']==1])))
            for name,rows,index in zip(('old','new'),datasets,valid)},
        checkpoints=[], validation_history=[], actor_updates=0, critic_updates=0,
        robot_execution_proxy_updates=0, contact_residual_enabled=False, promoted=False)
    baseline, baseline_prediction = {}, {}
    for name, rows, index in zip(('old', 'new'), datasets, valid):
        baseline[name], baseline_prediction[name] = evaluate(parameters, scale, rows, index)
    report['parent_validation'] = baseline
    write_json_unique(args.output_dir/'prerun.json', report, guard.budget)

    @jax.jit
    def update(params, state, features, target):
        loss, gradient = jax.value_and_grad(supervised_loss)(params, scale, features, target)
        params, state, diagnostics = adam_update(params, gradient, state,
            learning_rate=.0003, max_gradient_norm=1.)
        return params, state, loss, diagnostics['gradient_norm']

    signature = config_hash(dict(parent=settings['parent_checkpoint'], labels=report['label_references'],
                                 recipe=report['recipe'], source=report['source_sha256']))
    began = time.monotonic(); sampled = dict(old=0, new=0)
    for step in range(1, args.updates+1):
        if step % 64 == 1:
            guard.check()
        indexes = [np.concatenate([sample_pool(rng, pool, 128) for pool in pair]) for pair in pools]
        for name, rows, index in zip(('old', 'new'), datasets, indexes):
            if not np.all(rows['split'][index] == 0) or rows['overlap'][index].sum() != 128:
                raise ValueError('Refresh sampler changed split or registered source/bin quotas')
            sampled[name] += len(index)
        features, targets = (np.concatenate([rows[key][index] for rows,index in zip(datasets,indexes)])
                             for key in ('features','target'))
        parameters, optimizer, loss, norm = update(parameters, optimizer,
            jnp.asarray(features), jnp.asarray(targets))
        if step % 256 == 0 or step == 1:
            loss_value, norm_value = float(loss), float(norm)
            if not np.isfinite([loss_value,norm_value]).all():
                raise FloatingPointError('Nonfinite auxiliary continuation')
            print(f'[REFIT] additional={step}/{args.updates} loss={loss_value:.6g} grad={norm_value:.6g}',flush=True)
        if step % 1024 == 0 or step == args.updates:
            guard.check(); metrics, predictions = {}, {}
            for name, rows, index in zip(('old','new'),datasets,valid):
                metrics[name], predictions[name] = evaluate(parameters,scale,rows,index)
            report['validation_history'].append(dict(additional_updates=step,metrics=metrics))
            arrays, state = pack_numeric_state(dict(parameters=parameters,optimizer=optimizer,scale=scale),
                                               context_sha256=signature)
            total = parent['updates']+step; filename=f'auxiliary_u{total:06d}'
            archive=write_array_archive_unique(args.output_dir/(filename+'.npz'),arrays,guard.budget)
            metadata=dict(schema='wmj_solver_auxiliary_checkpoint_v1',training_version='WMJ-V1',
                model_signature=signature,archive_file=filename+'.npz',archive=archive,state=state,
                numpy_rng=rng.bit_generator.state,updates=total,additional_updates=step,
                normalization=parent['normalization'],source_sha256=report['source_sha256'],
                label_result_sha256=report['label_references']['new']['sha256'],
                label_references=report['label_references'],parent_checkpoint=settings['parent_checkpoint'],
                recipe=report['recipe'],width=list(WIDTHS),integration_in_world=False)
            path=args.output_dir/(filename+'.json');digest=write_json_unique(path,metadata,guard.budget)
            report['checkpoints'].append(dict(path=str(path),sha256=digest))
            values=[metrics[k]['positive_force_scalar_velocity_error_m_s']['p95'] for k in ('old','new')]
            print(f'[VALIDATE] additional={step} old/new scalar-v P95={values}',flush=True)
    if numeric_tree_digest(scale) != normalization:
        raise ValueError('Inherited normalization changed during continuation')
    if int(optimizer.step) != parent['updates'] + args.updates:
        raise ValueError('Actual auxiliary Adam count disagrees with the authorized continuation')
    report.update(status='AUXILIARY_REFRESH_FIT_COMPLETE_NOT_PROMOTED',fit_wall_s=time.monotonic()-began,
        additional_updates=args.updates,total_updates=parent['updates']+args.updates,
        actual_sampled_rows=sampled,adam_step_after=int(optimizer.step),normalization_preserved=True,final_validation=metrics,
        final_parameter_sha256=numeric_tree_digest(parameters),world_qualified=False)
    saved = {}
    report['final_validation_by_region']={}
    for name, rows, index in zip(('old','new'),datasets,valid):
        selected={k:v[index] for k,v in rows.items()}
        report['final_validation_by_region'][name]={str(int(region)):summarize_prediction(
            predictions[name][selected['region_id']==region],
            {k:v[selected['region_id']==region] for k,v in selected.items()})
            for region in np.unique(selected['region_id'])}
        saved.update({name+'_prediction':predictions[name],name+'_parent_prediction':baseline_prediction[name],
                      name+'_label_row_index':index})
    report['validation_prediction_archive']=write_array_archive_unique(
        args.output_dir/'validation_predictions.npz',saved,guard.budget)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--updates',type=int,default=4096)
    parser.add_argument('--max-wall-s',type=float,default=900.)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=[Path(__file__).with_name(name) for name in
               ('fit_auxiliary_refresh.py','auxiliary_refresh.py','solver_auxiliary.py','neural.py')]
        report=dict(schema='wmj_v1_auxiliary_only_refresh',training_version='WMJ-V1',status='INCOMPLETE',
            config_sha256=config_hash(cfg),gpu_uuid=args.gpu_uuid,source_sha256={p.name:file_hash(p) for p in paths},
            actor_updates=0,critic_updates=0,world_qualified=False)
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report,sources={p.name:p.read_text() for p in paths}),guard.budget)
        write_json_unique(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            fit(cfg,args,guard,report)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
