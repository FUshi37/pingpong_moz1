"""Approved course-dependent P_start with explicit cold-start reassignment."""

import numpy as np

from .sampling import SamplingShortage, _shuffle


COURSE_COUNTS = {
    'S01': (256, 0, 0),
    'S02': (192, 64, 0),
    'S03': (192, 64, 0),
    'S04': (128, 64, 64),
    'S05': (128, 64, 64),
}
CATEGORIES = ('precontact', 'goal_switch', 'recoverable_disturbance')


def sample_course_starts(rng, rows, *, stage_id, model_identity,
                         minimum_precontact_lead_s, diagnostic=False):
    """Draw256 full seeds; missing enabled buckets move only to precontact.

    Compatible task and AR1 state are required. No stage/model metadata is
    rewritten to manufacture compatible history. Final-test seeds never enter
    this sampler, including its explicit validation-only diagnostic mode.
    """
    if stage_id not in COURSE_COUNTS or not model_identity:
        raise ValueError('Registered course and authenticated model identity required')
    if not np.isfinite(minimum_precontact_lead_s) or minimum_precontact_lead_s<=0.:
        raise ValueError('A positive precontact lead is required')
    allowed_split='validation' if diagnostic else 'train'
    if any(row.get('split')!=allowed_split for row in rows):
        raise ValueError('Seed partition does not match this training/diagnostic mode')
    if any(row.get('category') not in CATEGORIES for row in rows):
        raise ValueError('Unknown branch category')
    requested=dict(zip(CATEGORIES,COURSE_COUNTS[stage_id]))
    actual=dict(requested)
    eligible={category:[] for category in CATEGORIES}
    excluded=[]
    for index,row in enumerate(rows):
        compatible=(row.get('model_identity')==model_identity and row.get('stage_id')==stage_id
                    and row.get('ar1_enabled')==(stage_id=='S05'))
        if row['category']=='precontact':
            lead=row.get('time_to_contact_s',np.nan)
            compatible=compatible and np.isfinite(lead) and lead>=minimum_precontact_lead_s
        if compatible:
            eligible[row['category']].append(index)
        else:
            excluded.append(index)
    if not eligible['precontact']:
        raise SamplingShortage('No compatible precontact seeds; cannot invent a fallback')
    reallocated={}
    for category in CATEGORIES[1:]:
        if requested[category] and not eligible[category]:
            reallocated[category]=requested[category]
            actual['precontact']+=requested[category]
            actual[category]=0
    probability=np.zeros(len(rows),np.float64)
    selected=[]
    components=[]
    for category,count in actual.items():
        if not count:
            continue
        indices=eligible[category]
        probability[indices]=count/256/len(indices)
        selected.extend(rng.choice(indices,count))
        components.extend([category]*count)
    return _shuffle(rng,selected,components,probability,dict(
        sampler='P_start_course',stage_id=stage_id,model_identity=model_identity,
        requested_counts=requested,actual_counts=actual,reallocated_to_precontact=reallocated,
        eligible_by_category={k:len(v) for k,v in eligible.items()},
        excluded_rows=len(excluded),excluded_indices=excluded,replacement=True,
        minimum_precontact_lead_s=minimum_precontact_lead_s,
        source='VALIDATION_DIAGNOSTIC' if diagnostic else 'TRAIN_IMAGINED'))
