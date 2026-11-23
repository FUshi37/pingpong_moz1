"""Offline standard-drop admission; never called from a training/evaluation tick."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .artifacts import write_json_unique
from .contact_surface import surface_from_vertices
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .domain_atlas import collision_pairs, solve_ik, pure_vertical_speed_capacity
from .frozen_cases import generate_cases, validate_case_manifest, GROUPS
from .generators import load_pose_catalog
from .initialization_geometry import InitializationGeometryAuditor
from .resources import StorageBudget


def vertical_drop_geometry(surface,position,rotation,goal_xyz,ball_radius_m):
    """Intersect a vertical ball-center ray with the sphere-offset top plane."""
    axis=surface.normal_axis
    normal=np.asarray(rotation)[:,axis]*np.sign(rotation[2,axis])
    plane=np.asarray(position)+normal*surface.half_thickness_m
    center=np.asarray(goal_xyz,dtype=np.float64).copy()
    if normal[2]<=0.:
        raise ValueError('Expected a racket face with an upward normal')
    center[2]=(ball_radius_m+normal@plane-normal[:2]@center[:2])/normal[2]
    contact=center-ball_radius_m*normal
    local=np.asarray(rotation).T@(contact-position)
    planar=local[[(axis+1)%3,(axis+2)%3]]
    gap=np.asarray(surface.edge_offset_m)-np.asarray(surface.edge_outward_xy)@planar
    height=float(goal_xyz[2]-center[2])
    return dict(inside_face=bool(np.all(gap>=-1e-8)),positive_drop=height>0.,
        ball_center_at_contact_m=center.tolist(),contact_surface_point_m=contact.tolist(),
        minimum_face_halfspace_margin_m=float(gap.min()),drop_height_m=height,
        stationary_racket_contact_time_s=float(np.sqrt(max(2*height/9.81,0.))),
        scope='static_finite_face_geometry_not_dynamic_success')


class StandardDropAuditor:
    """Authenticate cached robot poses; solve additional poses only offline."""

    def __init__(self,cfg):
        self.cfg=cfg
        self.catalog=load_pose_catalog(cfg)
        self.atlas=json.loads(authenticate(repository_root(),cfg['collection']['initialization_atlas']).read_text())
        self.model=mujoco.MjModel.from_binary_path(str(authenticate(repository_root(),cfg['collection']['compiled_model_cache'])))
        self.data=mujoco.MjData(self.model)
        self.baseline=np.array(self.catalog.geometry['reset_fk']['qpos'])
        self.jids=np.array([self.model.joint(f'RightArm-{i}').id for i in range(7)])
        self.qadr=self.model.jnt_qposadr[self.jids]
        geometry=self.catalog.geometry
        self.site=geometry['racket']['site_id'];self.geom=geometry['racket']['geom_id']
        self.ball_body=self.model.body('pingpong_ball').id
        self.radius=geometry['ball']['radius_m']
        self.data.qpos[:]=self.baseline;mujoco.mj_forward(self.model,self.data)
        self.reference_rotation=self.data.site_xmat[self.site].reshape(3,3).copy()
        vertices=np.array(geometry['racket']['compiled_vertices_local_m'])
        self.surface=surface_from_vertices(vertices,normal_axis=int(np.argmin(np.ptp(vertices,axis=0))))
        self.auditor=InitializationGeometryAuditor(self.model,collision_pairs(self.model,
            moving_body_ids=[*self.model.jnt_bodyid[self.jids],geometry['racket']['body_id']],
            excluded_body_ids=[self.ball_body]))
        self.velocity_limits=np.deg2rad([210,210,240,240,300,300,300])
        self.cached=[]
        for q in self.catalog.q_rad:
            self.set_pose(q)
            self.cached.append((self.data.geom_xpos[self.geom].copy(),
                                self.data.geom_xmat[self.geom].reshape(3,3).copy()))

    def set_pose(self,q):
        self.data.qpos[:]=self.baseline
        self.data.qpos[self.qadr]=q
        mujoco.mj_forward(self.model,self.data)

    def audit_goal(self,goal):
        goal=np.asarray(goal)
        order=np.argsort(np.linalg.norm(self.catalog.position_m[:,:2]-goal[:2],axis=1)
                         +.02*np.abs(self.catalog.position_m[:,2]-1.075))
        for index in order:
            position,rotation=self.cached[index]
            contact=vertical_drop_geometry(self.surface,position,rotation,goal,self.radius)
            if contact['inside_face'] and contact['positive_drop']:
                return dict(status='STATIC_INITIALIZATION_ADMITTED',source='authenticated_atlas',
                    atlas_pose_index=int(index),atlas_region_id=int(self.catalog.pose_region_ids[index]),
                    q_rad=self.catalog.q_rad[index].tolist(),racket_site_m=self.catalog.position_m[index].tolist(),
                    drop_geometry=contact,warmup_admission='PENDING_NATIVE_WARMUP')
        attempts=[]
        nearby=self.catalog.position_m[order[0],:2]
        targets=[np.r_[xy,z] for xy in (goal[:2],.5*(goal[:2]+nearby)) for z in (1.075,1.025,1.125)]
        for target in targets:
            for index in order[:2]:
                result=solve_ik(self.model,self.baseline,self.jids,self.site,self.reference_rotation,
                    target,self.catalog.q_rad[index],self.model.jnt_range[self.jids,0],self.model.jnt_range[self.jids,1])
                attempt=dict(target_site_m=target.tolist(),ik=result)
                if result['status']=='IK_FEASIBLE':
                    self.set_pose(result['q_rad'])
                    contact=vertical_drop_geometry(self.surface,self.data.geom_xpos[self.geom],
                        self.data.geom_xmat[self.geom].reshape(3,3),goal,self.radius)
                    attempt['drop_geometry']=contact
                    if contact['inside_face'] and contact['positive_drop']:
                        attempt['self_collision']=self.auditor.audit(self.data)
                        attempt['velocity_capacity']=pure_vertical_speed_capacity(self.model,self.data,
                            self.jids,self.site,self.velocity_limits)
                        if (attempt['self_collision']['status']=='PASS'
                                and attempt['velocity_capacity']['status']=='PASS'):
                            return dict(status='STATIC_INITIALIZATION_ADMITTED',source='new_offline_IK_audit',
                                q_rad=result['q_rad'],racket_site_m=result['position_m'],drop_geometry=contact,
                                attempts=attempts+[attempt],warmup_admission='PENDING_NATIVE_WARMUP')
                attempts.append(attempt)
        return dict(status='UNKNOWN',reason='no_valid_stationary_drop_initialization_in_declared_search',
            attempts=attempts,proven_unreachable=False)

    def audit_warmed(self,qpos,qvel,goal,reference_rotation):
        """Audit the actual warmed full robot, once, before releasing the ball."""
        from .domain_atlas import _rpy_and_rate
        self.data.qpos[:]=qpos
        self.data.qvel[:]=qvel
        ball_joint=int(self.model.body_jntadr[self.ball_body])
        ball_qadr=int(self.model.jnt_qposadr[ball_joint])
        ball_vadr=int(self.model.jnt_dofadr[ball_joint])
        self.data.qpos[ball_qadr:ball_qadr+7]=[*goal,1.,0.,0.,0.]
        self.data.qvel[ball_vadr:ball_vadr+6]=0.
        mujoco.mj_forward(self.model,self.data)
        collision=self.auditor.audit(self.data)
        ball_geom=self.model.geom('ball').id
        ball_auditor=InitializationGeometryAuditor(self.model,
            [(ball_geom,g) for g in range(self.model.ngeom) if g!=ball_geom])
        ball_clearance=ball_auditor.audit(self.data)
        drop=vertical_drop_geometry(self.surface,self.data.geom_xpos[self.geom],
            self.data.geom_xmat[self.geom].reshape(3,3),goal,self.radius)
        position=self.data.site_xpos[self.site].copy()
        rpy,_=_rpy_and_rate(self.data.site_xmat[self.site].reshape(3,3),
            np.asarray(reference_rotation),np.zeros((3,7)))
        domain=self.cfg['domain']
        bounds=(np.all(position>=np.asarray(domain['racket_xyz_lower_m'])-1e-5)
            and np.all(position<=np.asarray(domain['racket_xyz_upper_m'])+1e-5)
            and np.all(np.abs(rpy)<=np.deg2rad(domain['roll_pitch_limit_deg'])+1e-5)
            and np.all(self.data.qpos[self.qadr]>=self.model.jnt_range[self.jids,0]-1e-5)
            and np.all(self.data.qpos[self.qadr]<=self.model.jnt_range[self.jids,1]+1e-5))
        admitted=(collision['status']=='PASS' and ball_clearance['status']=='PASS'
            and bounds and drop['inside_face'] and drop['positive_drop'])
        return dict(status='ADMITTED' if admitted else 'UNKNOWN',self_collision=collision,
            initial_ball_clearance=ball_clearance,drop_geometry=drop,bounds=bool(bounds),
            racket_site_m=position.tolist(),relative_roll_pitch_deg=np.rad2deg(rpy).tolist(),
            execution_scope='offline_after_native_control_warmup',policy_outputs_used=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();cfg=load_config();r=cfg['resources']
    budget=StorageBudget(repository_root()/r['output_root'],r['max_new_bytes'],r['min_free_bytes'])
    budget.check(additional_bytes=32*2**20)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    manifest=generate_cases(cfg,seed=cfg['frozen_evaluation']['seed'])
    validate_case_manifest(manifest,cfg)
    case_path=args.output_dir/'cases.json'
    case_hash=write_json_unique(case_path,manifest,budget)
    auditor=StandardDropAuditor(cfg)
    results=[]
    for i,case in enumerate(manifest['stages']['S01']):
        result=auditor.audit_goal(case['goals_xyz_m'][0])
        row=dict(case_id=case['case_id'],group=case['group'],goal_xyz_m=case['goals_xyz_m'][0],**result)
        results.append(row)
        write_json_unique(args.output_dir/(case['case_id']+'.json'),row,budget)
        print(f'[ADMISSION] {i+1}/128 {case["case_id"]} {result["status"]}',flush=True)
    report=dict(schema='wmj_v1_frozen_initialization',status='STATIC_AUDIT_COMPLETE_WARMUP_PENDING',
        cases_reference=dict(path=str(case_path.absolute()),sha256=case_hash),config_sha256=config_hash(cfg),
        physics_sha256=config_hash(cfg['physics']),control_sha256=config_hash(cfg['control']),
        atlas_reference=cfg['collection']['initialization_atlas'],model_reference=cfg['collection']['compiled_model_cache'],
        rows=results,groups={g:dict(requested=32,static_admitted=sum(x['group']==g and x['status']=='STATIC_INITIALIZATION_ADMITTED' for x in results),unknown=sum(x['group']==g and x['status']=='UNKNOWN' for x in results)) for g in GROUPS},
        runtime_geometry_queries=0,policy_outputs_used=False,wall_s=time.monotonic()-start,
        source_sha256={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')})
    write_json_unique(args.output_dir/'initialization.json',report,budget)


if __name__=='__main__': main()
