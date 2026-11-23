"""Fixed evaluation cases are policy independent and preserve paired geometry."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.frozen_cases import generate_cases, validate_case_manifest
from world_model.contracts import load_config


def test_frozen_cases_preserve_groups_mirrors_distances_and_stage_pairing():
    cfg=load_config()
    manifest=generate_cases(cfg,seed=2026110925)
    validate_case_manifest(manifest,cfg)
    assert manifest==generate_cases(cfg,seed=2026110925)
    for stage,count in zip(('S01','S02','S03','S04','S05'),(1,2,2,4,4)):
        rows=manifest['stages'][stage]
        assert len(rows)==128
        nominal=[r for r in rows if r['group']=='nominal']
        mirror=[r for r in rows if r['group']=='mirror']
        for a,b in zip(nominal,mirror):
            np.testing.assert_array_equal(np.array(a['goals_xyz_m'])*[-1.,1.,1.],b['goals_xyz_m'])
        assert all(len(r['goals_xyz_m'])==count for r in rows)
        for r in rows:
            if r['group']=='near_side': assert -.38<=r['goals_xyz_m'][0][1]<=-.28
            if count>1:
                distance=np.linalg.norm(np.diff(r['goals_xyz_m'],axis=0),axis=1)
                low,high=(.02,.08) if stage=='S02' else ((.08,.2) if stage=='S03' else (0.,.2))
                assert np.all((distance>=low)&(distance<=high))
    assert manifest['stages']['S04']==manifest['stages']['S05']
    for stage in manifest['stages']:
        for a,b in zip(manifest['stages']['S01'],manifest['stages'][stage]):
            np.testing.assert_array_equal(a['goals_xyz_m'][0],b['goals_xyz_m'][0])


def test_changed_frozen_sequence_is_rejected():
    cfg=load_config(); manifest=generate_cases(cfg,seed=2026110925)
    manifest['stages']['S02'][0]['goals_xyz_m'][1][0]+=1.
    import pytest
    with pytest.raises(ValueError): validate_case_manifest(manifest,cfg)


def test_different_valid_seed_cannot_replace_the_frozen_population():
    import pytest
    cfg=load_config()
    replacement=generate_cases(cfg,seed=cfg['frozen_evaluation']['seed']+1)
    with pytest.raises(ValueError):
        validate_case_manifest(replacement,cfg)
