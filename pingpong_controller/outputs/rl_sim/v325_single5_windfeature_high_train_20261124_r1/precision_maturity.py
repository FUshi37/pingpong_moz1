"""Restore only inherited precision-credit phase, never physical or wind time."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np

PRECISION_FIELDS=('center_precision/warmup','precision_context/physical_steps',
                  'precision_context/reward_steps','reward/center_mean_rms_credit',
                  'reward/clean_apex_preservation_credit')


def restore_precision_context(cfg,*,enabled=False):
    if not enabled or getattr(cfg,'wind_mature_precision_reward',False):return cfg
    from fixed_plane_wind_resume_precision_clock import with_mature_precision_config
    return with_mature_precision_config(cfg,True)


def verify_mature_batch(metrics):
    values=np.asarray(metrics['center_precision/warmup'])
    if not np.isfinite(values).all() or not np.all(values==1.):
        raise ValueError('Inherited precision reward unexpectedly rewound')


def verify_parent_maturity(plan):
    receipt=plan['reward_maturity']
    path=Path(receipt['parent_snapshot'])
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    if sha!=receipt['parent_snapshot_sha256']:raise ValueError('Maturity source hash mismatch')
    clocks=[]
    with ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if manifest['metadata']['actual_updates']!=receipt['parent_update']:
            raise ValueError('Wrong maturity source update')
        for index,spec in enumerate(manifest['trees']['runner']):
            if 'total_env_steps' in spec['path']:
                with archive.open(f'runner_{index:05d}.npy') as stream:
                    values=np.lib.format.read_array(stream,allow_pickle=False)
                clocks.append((int(values.min()),int(values.max())))
    if len(clocks)!=1 or clocks[0][0]<8192:raise ValueError('Precision source was not mature')
    return dict(source_sha256=sha,parent_min_steps=clocks[0][0],parent_max_steps=clocks[0][1],
        reward_only_offset_steps=8192,physical_and_wind_clocks_unchanged=True)
