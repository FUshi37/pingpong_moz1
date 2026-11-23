"""Plot original mesh surfaces and the convex-hull cross-section of an alarm."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np


def _section(triangles, height):
    edges = np.stack((triangles, np.roll(triangles, -1, axis=1)), axis=2)
    crossed = (edges[:, :, 0, 2] < height) != (edges[:, :, 1, 2] < height)
    ids = np.flatnonzero(crossed.sum(axis=1) == 2)
    selected = edges[ids][crossed[ids]]
    fraction = (height-selected[:, 0, 2])/(selected[:, 1, 2]-selected[:, 0, 2])
    points = selected[:, 0]+fraction[:, None]*(selected[:, 1]-selected[:, 0])
    return points[:, :2].reshape(-1, 2, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from scipy.spatial import ConvexHull
    from .contracts import load_config, repository_root, file_hash
    from .resources import StorageBudget
    from .audit_collision_geometry import _save_bytes

    report = json.loads(args.report.read_text())
    array_path = Path(report['arrays']['path'])
    if file_hash(array_path) != report['arrays']['sha256']:
        raise ValueError('Mesh artifact hash mismatch')
    arrays = np.load(array_path, allow_pickle=False)
    witnesses = np.array([row['convex_witness_m'] for row in report['pair']['surfaces']])
    height = float(witnesses[:, 2].mean())
    fig = plt.figure(figsize=(12, 5.5), layout='constrained')
    spatial, section = fig.add_subplot(121, projection='3d'), fig.add_subplot(122)
    colors = ('#248ca7', '#ed8c34')
    for geom, color, label in zip(report['pair']['geoms'], colors, ('waist03: torso mesh', 'right02: shoulder mesh')):
        vertices, faces = arrays[f'vertices_{geom}'], arrays[f'faces_{geom}']
        triangles = vertices[faces]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-30)
        light = .45+.55*np.abs(normal @ np.array([.5, -.7, .5]))
        from matplotlib.colors import to_rgb
        rgb = np.asarray(to_rgb(color))[None]*np.minimum(light, 1)[:, None]
        spatial.add_collection3d(Poly3DCollection(triangles, facecolors=rgb, linewidths=0))
        section.add_collection(LineCollection(_section(triangles, height), colors=color, linewidths=1.4, label=label))
        hull = vertices[ConvexHull(vertices).simplices]
        section.add_collection(LineCollection(_section(hull, height), colors=color,
                                             linewidths=1.2, linestyles='dashed'))
    spatial.set(xlim=(-.32, .17), ylim=(-.13, .14), zlim=(1.02, 1.49),
                xlabel='World X (m)', ylabel='World Y (m)', zlabel='World Z (m)',
                title='Actual reset: original triangle surfaces')
    spatial.set_box_aspect((.49, .27, .47))
    spatial.view_init(elev=15, azim=-115)
    spatial.scatter(*witnesses.T, c='crimson', s=25, depthshade=False)
    section.plot(*witnesses[:, :2].T, color='crimson', marker='x', linewidth=2, label='Convex query witnesses')
    section.set(xlim=(-.19, -.115), ylim=(-.065, .04), xlabel='World X (m)', ylabel='World Y (m)',
                title=f'Section Z = {height:.4f} m\nSolid: original surface; dashed: convex hull')
    section.set_aspect('equal')
    section.grid(alpha=.2)
    section.legend(fontsize=8, loc='upper left')
    fig.suptitle('WMJ-V1 diagnostic: convex overlap is not original-surface intersection', fontsize=12)
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=160)
    plt.close(fig)
    cfg = load_config()['resources']
    budget = StorageBudget(repository_root()/cfg['output_root'], cfg['max_new_bytes'], cfg['min_free_bytes'])
    _save_bytes(args.output, stream.getvalue(), budget)
    print(args.output)


if __name__ == '__main__':
    main()
