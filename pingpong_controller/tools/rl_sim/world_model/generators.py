"""Full-domain teacher experiment proposals; outcomes always come from MJX.

Offline admitted poses seed the control-chain warmup. Directed experiments
use an actual preview of the future racket pose, finite mesh offsets and a
backward ballistic proposal. Neither desired outgoing velocity nor desired
apex is applied as a physical state update. Every new ball starts without
spin; all subsequent native rotational dynamics are preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np

from .contracts import authenticate, repository_root, config_hash
from .contact_surface import proposed_ball_center, radial_boundary_distance
from .sampling import _counts


@dataclass(frozen=True)
class PoseCatalog:
    q_rad: np.ndarray
    pose_region_ids: np.ndarray
    region_ids: np.ndarray
    unknown_region_ids: np.ndarray
    position_m: np.ndarray
    geometry: dict
    atlas_sha256: str


def load_pose_catalog(cfg: dict) -> PoseCatalog:
    """Load cached admission only; no IK or self-collision work enters rollout."""
    reference = cfg['collection']['initialization_atlas']
    path = authenticate(repository_root(), reference)
    atlas = json.loads(path.read_text())
    if atlas['training_version'] != 'WMJ-V1' or atlas['execution_scope'] != 'offline_initialization_only':
        raise ValueError('A WMJ-V1 offline initialization atlas is required')
    for key in ('physics', 'control'):
        if config_hash(atlas['config'][key]) != config_hash(cfg[key]):
            raise ValueError(f'Atlas {key} differs from the current baseline')
    for key in ('racket_xyz_lower_m', 'racket_xyz_upper_m', 'roll_pitch_limit_deg'):
        if atlas['config']['domain'][key] != cfg['domain'][key]:
            raise ValueError('Initialization workspace changed since the offline audit')
    poses, regions, positions, unknown = [], [], [], []
    for row in atlas['rows']:
        found = False
        for attempt in row['attempts']:
            if (attempt['status'] == 'IK_FEASIBLE'
                    and attempt.get('self_collision', {}).get('status') == 'PASS'
                    and attempt.get('velocity_capacity', {}).get('status') == 'PASS'):
                poses.append(attempt['q_rad'])
                positions.append(attempt['position_m'])
                regions.append(row['id'])
                found = True
        if not found:
            unknown.append(row['id'])
    q = np.asarray(poses, np.float32)
    if q.ndim != 2 or q.shape[1] != 7 or not np.isfinite(q).all():
        raise ValueError('Atlas contains no finite admitted seven-joint poses')
    return PoseCatalog(q, np.asarray(regions, np.int32), np.unique(regions),
                       np.asarray(unknown, np.int32), np.asarray(positions, np.float32),
                       atlas['geometry'], reference['sha256'])


def select_pose_per_region(rng, catalog: PoseCatalog, region_indices):
    """Sample admitted IK alternatives inside each selected regional stratum."""
    region_indices = np.asarray(region_indices)
    if np.any((region_indices < 0) | (region_indices >= len(catalog.region_ids))):
        raise ValueError('Unknown regional index')
    ids = [int(rng.choice(np.flatnonzero(catalog.pose_region_ids == catalog.region_ids[i])))
           for i in region_indices]
    return np.asarray(ids, np.int32), catalog.q_rad[ids]


@dataclass(frozen=True)
class DirectedContactPlan:
    initial_position_m: np.ndarray
    initial_velocity_m_s: np.ndarray
    initial_angular_velocity_rad_s: np.ndarray
    proposed_contact_position_m: np.ndarray
    incoming_velocity_m_s: np.ndarray
    time_to_contact_s: np.ndarray
    offset_xy_m: np.ndarray
    offset_class: np.ndarray
    target_apex_m: np.ndarray
    desired_outgoing_velocity_m_s: np.ndarray


def sample_contact_lead_controls(rng, cfg: dict, n_envs: int) -> np.ndarray:
    """Use registered 80--160 ms endpoints of actual 200 Hz preview steps."""
    lower, upper = cfg['collection']['directed_conditions']['time_to_contact_s']
    dt = cfg['control']['policy_dt_s']
    return rng.integers(int(np.ceil(lower/dt)), int(np.floor(upper/dt))+1,
                        size=n_envs, dtype=np.int32)


def propose_directed_contacts(rng, cfg: dict, surface, future_geom_position_m,
                               future_geom_rotation_world, *, ball_radius_m,
                               contact_lead_controls=None) -> DirectedContactPlan:
    """Propose center/off-center/edge/near-miss trials over the whole goal domain.

    future_geom_* must come from the declared control preview. Near-miss is
    a requested condition, not an outcome label: a moving racket can still
    intercept it. Unmet targets and actual misses remain supervised data.
    """
    positions = np.asarray(future_geom_position_m)
    rotations = np.asarray(future_geom_rotation_world)
    n = len(positions)
    if positions.shape != (n, 3) or rotations.shape != (n, 3, 3):
        raise ValueError('Future racket poses must have shapes (N,3) and (N,3,3)')
    if not np.isfinite(positions).all() or not np.isfinite(rotations).all():
        raise ValueError('The actual racket preview must be finite')
    settings = cfg['collection']['directed_conditions']
    lead = (sample_contact_lead_controls(rng, cfg, n) if contact_lead_controls is None
            else np.asarray(contact_lead_controls))
    time = lead.astype(np.float64)*cfg['control']['policy_dt_s']
    low, high = settings['time_to_contact_s']
    if time.shape != (n,) or np.any((time < low-1e-9) | (time > high+1e-9)):
        raise ValueError('Contact preview lead is outside the confirmed sampling domain')
    classes = np.asarray(settings['offset_classes'])
    class_ids = np.arange(n) % len(classes)
    rng.shuffle(class_ids)
    angle = rng.uniform(-np.pi, np.pi, n)
    boundary = np.asarray(radial_boundary_distance(surface, angle))
    # These strata cover the finite face and both sides of its boundary. All
    # requested strata are retained separately from the teacher's hit labels.
    inner = np.array([0., .25, .90, 1.])[class_ids]
    outer = np.array([.25, .80, 1.05, 1.])[class_ids]
    radial = boundary*np.sqrt(inner**2+rng.random(n)*(outer**2-inner**2))
    radius = np.broadcast_to(np.asarray(ball_radius_m), (n,))
    radial = np.where(class_ids == 3, boundary+radius+rng.uniform(.002, .015, n), radial)
    offsets = radial[:, None]*np.column_stack((np.cos(angle), np.sin(angle)))
    contact = np.asarray(proposed_ball_center(surface, positions, rotations, offsets, radius))
    incoming_angle = rng.uniform(-np.pi, np.pi, n)
    incoming_speed = settings['incoming_horizontal_speed_max_m_s']*np.sqrt(rng.random(n))
    incoming = np.column_stack((incoming_speed*np.cos(incoming_angle), incoming_speed*np.sin(incoming_angle),
        rng.uniform(*settings['incoming_vertical_velocity_m_s'], n)))
    gravity = np.asarray(cfg['physics']['gravity_m_s2'], np.float64)
    initial_v = incoming-time[:, None]*gravity
    initial_p = contact-incoming*time[:, None]+.5*gravity*time[:, None]**2
    goals = rng.uniform(cfg['domain']['task_goal_xyz_lower_m'], cfg['domain']['task_goal_xyz_upper_m'], (n, 3))
    height = goals[:, 2]-contact[:, 2]
    if np.any(height <= 0.) or gravity[2] >= 0.:
        raise ValueError('Proposed target apex must be above the future contact point')
    outgoing_z = np.sqrt(2*(-gravity[2])*height)
    apex_time = outgoing_z/(-gravity[2])
    outgoing = np.column_stack(((goals[:, :2]-contact[:, :2])/apex_time[:, None], outgoing_z))
    return DirectedContactPlan(initial_p.astype(np.float32), initial_v.astype(np.float32),
        np.zeros((n, 3), np.float32), contact.astype(np.float32), incoming.astype(np.float32),
        time, offsets.astype(np.float32), classes[class_ids], goals.astype(np.float32), outgoing.astype(np.float32))


def generator_assignments(rng, n_envs: int, cfg: dict) -> np.ndarray:
    """Largest-remainder quotas preserve the declared 25/40/20/15 mixture."""
    fractions = cfg['collection']['generator_fractions']
    counts = _counts(n_envs, list(fractions.values()))
    names = np.concatenate([np.repeat(name, count) for name, count in zip(fractions, counts)])
    rng.shuffle(names)
    return names


def correlated_action_sequences(rng, steps: int, n_envs: int, *,
                                 amplitude_range=(.02, .25), dt_s=.005):
    """Generate constant, held-PRBS, chirp and correlated seven-joint raw actions.

    The native control chain still applies clip, integration, projection and
    RMP/PD. These are identification commands, never actor training targets.
    Boundary experiments may explicitly request amplitudes above one to expose
    the original clip; this function does not relabel clipping as execution.
    """
    if steps < 1 or n_envs < 1 or not 0 <= amplitude_range[0] <= amplitude_range[1]:
        raise ValueError('Invalid excitation population/amplitude')
    actions = np.zeros((steps, n_envs, 7), np.float32)
    modes = np.asarray(['constant', 'prbs', 'chirp', 'correlated'])[np.arange(n_envs) % 4]
    rng.shuffle(modes)
    times = np.arange(steps)*dt_s
    for lane, mode in enumerate(modes):
        amplitude = rng.uniform(*amplitude_range, 7)
        direction = rng.choice([-1., 1.], 7)
        if mode == 'constant':
            signal = np.broadcast_to(direction, (steps, 7))
        elif mode == 'prbs':
            hold = int(rng.integers(4, 13))
            levels = rng.choice([-1., 1.], ((steps+hold-1)//hold, 7))
            signal = np.repeat(levels, hold, axis=0)[:steps]
        else:
            phase = rng.uniform(-np.pi, np.pi, 7) if mode == 'chirp' else rng.uniform(-np.pi, np.pi)
            start_hz, end_hz = rng.uniform(.5, 1.5), rng.uniform(2., 5.)
            sweep = start_hz*times+.5*(end_hz-start_hz)*times**2/max(steps*dt_s, dt_s)
            signal = np.sin(2*np.pi*sweep[:, None]+phase)*direction
        actions[:, lane] = signal*amplitude
    return actions, modes
