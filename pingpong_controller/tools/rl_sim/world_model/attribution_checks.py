"""Attribution-only physical input comparison, independent of solver scratch."""

import numpy as np


def world_event_trace(task,scoring,labels):
    """Keep the same1ms clock alongside outgoing and apex labels."""
    return dict(scoring,release_position_m=labels['release_position_m'],
        release_velocity_m_s=labels['release_velocity_m_s'],
        apex_position_m=labels['apex_position_m'],elapsed_ms=task.elapsed_ms)


def compare_initial_interfaces(current,reference):
    """Keep control/history identity; accept existing native physical tolerances."""
    values={}
    for name,tolerance in (('qpos',1e-5),('qvel',1e-3),
                           ('racket_position_m',1e-5),('racket_rotation',1e-5)):
        a,b=np.asarray(current[name]),np.asarray(reference[name])
        if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Invalid physical initial interface')
        delta=float(np.max(np.abs(a-b)))
        values[name]=dict(max_abs=delta,tolerance=tolerance,passed=delta<=tolerance)
    memories={k:current[k]==reference[k] for k in ('control_sha256','camera_history_sha256')}
    return dict(physical=values,memories_exact=memories,
        passed=all(v['passed'] for v in values.values()) and all(memories.values()))
