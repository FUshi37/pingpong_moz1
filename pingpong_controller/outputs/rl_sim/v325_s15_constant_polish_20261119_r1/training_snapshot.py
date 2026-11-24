"""Stream complete numeric training state; preserve clocks, RNG and episode evidence."""
from pathlib import Path
import hashlib
import json
import os
import pickle
import re
import uuid
import zipfile

import jax
import numpy as np


def _canon(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def _digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def _spec(tree, require_finite=True):
    pairs,definition=jax.tree_util.tree_flatten_with_path(tree)
    specs=[]
    for path,leaf in pairs:
        value=np.asarray(jax.device_get(leaf))
        if value.dtype.kind not in 'biufc' or (require_finite and not np.isfinite(value).all()):
            raise ValueError('Training snapshot requires finite numeric leaves')
        specs.append(dict(path=jax.tree_util.keystr(path),shape=list(value.shape),dtype=value.dtype.str))
    return pairs,definition,specs


def save_snapshot(path: Path, learner, runner, auxiliary: dict, contract: dict, metadata: dict):
    """Write immutable archive and checksum commit, streaming one leaf at a time."""
    path=Path(path);commit=path.with_suffix(path.suffix+'.json')
    if path.exists() or commit.exists():raise FileExistsError(path)
    partial=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    auxiliary_bytes=pickle.dumps(auxiliary,protocol=5)
    manifest=dict(schema='v325_full_training_state_v1',contract=contract,metadata=metadata,
        auxiliary_sha256=hashlib.sha256(auxiliary_bytes).hexdigest(),trees={})
    try:
        with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
            for name,tree in [('learner',learner),('runner',runner)]:
                pairs,_,specs=_spec(tree,require_finite=name=='learner');manifest['trees'][name]=specs
                for index,(_,leaf) in enumerate(pairs):
                    value=np.asarray(jax.device_get(leaf))
                    with archive.open(f'{name}_{index:05d}.npy','w') as stream:
                        np.lib.format.write_array(stream,value,allow_pickle=False)
            archive.writestr('auxiliary.pkl',auxiliary_bytes)
            archive.writestr('manifest.json',_canon(manifest))
        if path.exists():raise FileExistsError(path)
        os.replace(partial,path)
        with commit.open('x') as stream:json.dump(dict(sha256=_digest(path),schema=manifest['schema']),stream)
    finally:
        if partial.exists():partial.unlink()
    return dict(path=str(path),sha256=_digest(path),bytes=path.stat().st_size)


def load_snapshot(path: Path, learner_template, runner_template, contract: dict):
    """Restore against authenticated same-layout templates before continuing simulation."""
    path=Path(path);commit=path.with_suffix(path.suffix+'.json')
    if path.is_symlink() or not commit.is_file():raise ValueError('Uncommitted snapshot')
    if _digest(path)!=json.loads(commit.read_text())['sha256']:raise ValueError('Snapshot checksum mismatch')
    restored={}
    with zipfile.ZipFile(path) as archive:
        manifest=json.loads(archive.read('manifest.json'))
        if manifest['schema']!='v325_full_training_state_v1' or _canon(manifest['contract'])!=_canon(contract):
            raise ValueError('Training state contract mismatch')
        for name,template in [('learner',learner_template),('runner',runner_template)]:
            _,definition,specs=_spec(template,require_finite=name=='learner')
            if specs!=manifest['trees'][name]:raise ValueError('Training state tree mismatch: '+name)
            leaves=[]
            for index,spec in enumerate(specs):
                with archive.open(f'{name}_{index:05d}.npy') as stream:value=np.lib.format.read_array(stream,allow_pickle=False)
                if (list(value.shape)!=spec['shape'] or value.dtype.str!=spec['dtype']
                        or (name=='learner' and not np.isfinite(value).all())):
                    raise ValueError('Invalid saved training leaf')
                leaves.append(jax.device_put(value))
            restored[name]=jax.tree_util.tree_unflatten(definition,leaves)
        auxiliary=archive.read('auxiliary.pkl')
        if hashlib.sha256(auxiliary).hexdigest()!=manifest['auxiliary_sha256']:
            raise ValueError('Auxiliary state checksum mismatch')
        # Only this task's owned, committed archives are admitted by the launcher.
        auxiliary=pickle.loads(auxiliary)
    return restored['learner'],restored['runner'],auxiliary,manifest['metadata']


def prune_owned_snapshots(directory: Path, keep: Path):
    """Keep the committed latest recovery state; policy-best retention is separate."""
    directory=Path(directory).resolve();keep=Path(keep).resolve()
    if keep.parent!=directory or not keep.with_suffix(keep.suffix+'.json').is_file():
        raise ValueError('Recovery snapshot must be committed before cleanup')
    for old in directory.glob('trainer_state_u[0-9]*.zip'):
        if not re.fullmatch(r'trainer_state_u\d{8,}\.zip',old.name):continue
        if old.resolve()!=keep:
            if old.is_symlink() or old.parent.resolve()!=directory:raise ValueError('Unsafe snapshot cleanup')
            old.unlink();old.with_suffix(old.suffix+'.json').unlink(missing_ok=True)


def state_digest(tree):
    """Include every numeric state byte, preserving masked simulator sentinels."""
    digest=hashlib.sha256()
    pairs,_=jax.tree_util.tree_flatten_with_path(jax.device_get(tree))
    for path,leaf in pairs:
        value=np.asarray(leaf)
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()
