"""Restore exact saved MJX action-fork inputs without compiling a new MJX reset."""
from pathlib import Path
import hashlib
import json
import jax.numpy as jnp
import numpy as np
from fixed_plane_v92 import rotation_matrix_to_6d
from .artifacts import read_array_archive
from .contracts import file_hash
from .control_seed import read_control_seed
from .full_prior_seeds import read_initial_camera, initial_full_history
from .physical_events import PhysicalEventState
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .full_prior_world import FullPriorWorldState
from .world_physics import PhysicsState


def read_action_seed(reference: dict, env, options: dict, context_sha256: str, lanes) -> tuple:
    """Authenticate all physical/controller/camera/event leaves of the same seed."""
    path=Path(reference['manifest']);metadata=json.loads(path.read_text())
    if file_hash(path)!=reference['sha256'] or metadata['context_sha256']!=context_sha256:
        raise ValueError('Action fork seed identity changed')
    lookup={row['path']:(f'leaf_{i:06d}',row) for i,row in enumerate(metadata['leaves'])}
    required=["['native'].data."+key for key in ('qpos','qvel','site_xpos','site_xmat')]
    required += ["['events']."+key for key in PhysicalEventState._fields]
    required += ["['ball_model_valid']","['goal']"]
    arrays=read_array_archive(path.parent/metadata['archive_file'],metadata['archive'],
                             names={lookup[k][0] for k in required})
    selected={}
    for key in required:
        name,spec=lookup[key];value=arrays[name]
        if (list(value.shape)!=spec['shape'] or value.dtype.str!=spec['dtype']
                or hashlib.sha256(value.tobytes()).hexdigest()!=spec['sha256']):
            raise ValueError('Saved fork leaf changed: '+key)
        selected[key]=jnp.asarray(value[np.asarray(lanes)])
    control,context=read_control_seed(reference,context_sha256=context_sha256,lanes=lanes)
    camera=read_initial_camera(reference,context_sha256=context_sha256,lanes=lanes)
    qpos=selected["['native'].data.qpos"];qvel=selected["['native'].data.qvel"]
    support=np.array([env.mj_model.joint(f'LegWaist-{i}').id for i in range(6)])
    qadr=np.r_[np.asarray(env.arm_qadr),env.mj_model.jnt_qposadr[support]]
    vadr=np.r_[np.asarray(env.arm_vadr),env.mj_model.jnt_dofadr[support]]
    q,dq=qpos[:,qadr],qvel[:,vadr]
    bq=qpos[:,env.ball_qadr:env.ball_qadr+7];bv=qvel[:,env.ball_vadr:env.ball_vadr+6]
    events=PhysicalEventState(*(selected["['events']."+key] for key in PhysicalEventState._fields))
    ball=ContactState(bq[:,:3],bv[:,:3],world_spin(bq[:,3:],bv[:,3:]),
                      events.contact_active,events.contact_duration_steps*.001)
    pose=jnp.concatenate((selected["['native'].data.site_xpos"][:,env.racket_site_id],
        rotation_matrix_to_6d(selected["['native'].data.site_xmat"][:,env.racket_site_id].reshape(-1,3,3))),axis=-1)
    history=initial_full_history(camera,q,chain=options['chain'])
    world=FullPriorWorldState(PhysicsState(q,dq,ball),control,history,pose)
    return world,context,events,selected["['ball_model_valid']"],selected["['goal']"]
