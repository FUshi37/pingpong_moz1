"""Course activation, honest cold starts and unchanged strict legacy draws."""

import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.course_sampling import sample_course_starts
from world_model.sampling import SamplingShortage, sample_branch_starts


def rows(category,count=4):
    return [dict(category=category,split='train',time_to_contact_s=.12,
                 model_identity='model-a',stage_id='S04',ar1_enabled=False) for _ in range(count)]


def test_course_counts_and_explicit_missing_bucket_reassignment():
    population=rows('precontact')+rows('goal_switch')+rows('recoverable_disturbance')
    expected=[(256,0,0),(192,64,0),(192,64,0),(128,64,64)]
    for stage,counts in zip(('S01','S02','S03','S04'),expected):
        current=[dict(row,stage_id=stage) for row in population]
        draw=sample_course_starts(np.random.default_rng(5),current,stage_id=stage,
            model_identity='model-a',minimum_precontact_lead_s=.08)
        assert tuple(draw.report['actual_counts'].values())==counts
        assert len(draw.indices)==256 and np.all(draw.probability>0.)
    draw=sample_course_starts(np.random.default_rng(5),rows('precontact'),stage_id='S04',
        model_identity='model-a',minimum_precontact_lead_s=.08)
    assert draw.report['actual_counts']['precontact']==256
    assert draw.report['reallocated_to_precontact']==dict(goal_switch=64,recoverable_disturbance=64)
    assert np.all(draw.component=='precontact')
    with pytest.raises(SamplingShortage):
        sample_branch_starts(np.random.default_rng(5),rows('precontact'),minimum_precontact_lead_s=.08)


def test_no_future_model_noise_or_short_lead_seed_can_fill_a_missing_category():
    population=rows('precontact')
    population += [dict(rows('goal_switch')[0],model_identity='old-model')]
    population += [dict(rows('goal_switch')[0],ar1_enabled=True)]
    population += [dict(rows('precontact')[0],time_to_contact_s=.01)]
    draw=sample_course_starts(np.random.default_rng(9),population,stage_id='S04',
        model_identity='model-a',minimum_precontact_lead_s=.08)
    assert draw.indices.max()<4
    assert draw.report['excluded_rows']==3
    with pytest.raises(ValueError,match='partition'):
        sample_course_starts(np.random.default_rng(9),[dict(rows('precontact')[0],split='test')],
            stage_id='S04',model_identity='model-a',minimum_precontact_lead_s=.08)
    with pytest.raises(SamplingShortage):
        sample_course_starts(np.random.default_rng(9),[],stage_id='S04',
            model_identity='model-a',minimum_precontact_lead_s=.08)
