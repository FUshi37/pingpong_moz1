"""Expanded-motion fitting preserves source partitions and frozen inputs."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_selection_preserves_original_partitions_and_unique_root_families():
    from world_model.expanded_auxiliary import select_sources
    groups=[dict(root_seed_family=f'r{i//2}',split='train' if i<8 else 'validation',
        initial_geometry_admitted=True,region_id=i%3) for i in range(16)]
    groups.append(dict(root_seed_family='sealed',split='test',initial_geometry_admitted=True,region_id=0))
    chosen=select_sources(groups,np.ones(17,bool),excluded={'r0'},seed=7,counts=(3,2))
    assert [r['split'] for r in chosen]==['train']*3+['validation']*2
    assert len({r['root_seed_family'] for r in chosen})==5
    assert not {'r0','sealed'} & {r['root_seed_family'] for r in chosen}
    assert chosen==select_sources(groups,np.ones(17,bool),excluded={'r0'},seed=7,counts=(3,2))
    with pytest.raises(ValueError):select_sources(groups,np.zeros(17,bool),excluded=set(),seed=7,counts=(3,2))


def test_supervised_retention_covers_contact_neighborhood_without_failed_tail():
    from world_model.expanded_auxiliary import retained_label_mask
    overlap=np.zeros((100,2),bool);overlap[40,0]=True
    prefix=np.ones_like(overlap);prefix[50:,0]=False
    retained=retained_label_mask(overlap,prefix,np.random.default_rng(1),background_fraction=0.)
    assert retained[15:50,0].all()
    assert not retained[:15,0].any() and not retained[50:,0].any()
    assert not retained[:,1].any()


def test_family_filter_removes_all_siblings_and_never_uses_diagnostic_sources():
    from world_model.expanded_auxiliary import excluded_family_mask
    rows=dict(family_id=np.array([0,0,1,2]),split=np.array([0,0,1,0]))
    mask=excluded_family_mask(rows,{0:'diagnostic',1:'heldout',2:'ordinary'}, {'diagnostic'})
    np.testing.assert_array_equal(mask,[False,False,True,True])
    with pytest.raises(ValueError):excluded_family_mask(rows,{0:'diagnostic'},set())


def test_new_source_capture_keeps_pairing_and_full_native_state_restoration():
    import inspect
    from world_model.audit_free_imagination import audit
    from world_model.expanded_auxiliary import source_capture_code,make_source_capture
    source=source_capture_code(inspect.getsource(audit))
    assert 'load_training_bank(' not in source
    assert 'initial_course' not in source
    assert 'restore_native_sources(' in source and 'assess_native_pair(' in source
    assert 'range(128)' in source
    assert callable(make_source_capture(np.zeros((128,64,7),np.float32)))


def test_retention_audit_keeps_every_validation_case_and_drops_only_file_duplication():
    import inspect
    from world_model.audit_auxiliary_refresh import audit
    from world_model.audit_expanded_retention import retention_source
    source=retention_source(inspect.getsource(audit))
    assert 'validation_batches(groups)' in source
    assert 'for horizon in (1, 8, 32, 64, 128)' in source
    assert 'compare_releases(' in source and 'teacher_seeds(' in source
    assert 'write_array_archive_unique(' not in source
    assert 'write_json_unique(' not in source
    assert 'no_duplicate_step_arrays' in source
