# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed P3 tet-only inputs, independent of the historical coupled fixtures."""

import numpy as np
import warp as wp

import newton


def build_model(device, *, cantilever=False, fixed=False, gravity=(0, 0, 0)):
    """Build the frozen tiny tet or G2H r1 geometry without any rigid actor."""
    builder = newton.ModelBuilder(gravity=gravity)
    mu, lam = 10000 / 2.6, 10000 * 0.3 / (1.3 * 0.4)
    if cantilever:
        builder.add_soft_grid(
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0),
            dim_x=3,
            dim_y=1,
            dim_z=1,
            cell_x=0.4,
            cell_y=0.3,
            cell_z=0.3,
            density=1000,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0,
            fix_left=True,
            add_surface_mesh_edges=False,
            particle_radius=0.001,
        )
    else:
        builder.add_soft_mesh(
            pos=(0, 0, 0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            vertices=[(0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02)],
            indices=[0, 1, 2, 3],
            density=1000,
            k_mu=mu,
            k_lambda=lam,
            k_damp=0,
            tri_ke=0,
            tri_ka=0,
            tri_kd=0,
            tri_drag=0,
            tri_lift=0,
            edge_ke=0,
            edge_kd=0,
            particle_radius=0.001,
        )
    mass = np.zeros(len(builder.particle_q))
    for ids, pose in zip(builder.tet_indices, builder.tet_poses, strict=True):
        volume = 1 / (6 * np.linalg.det(np.asarray(pose, dtype=float).reshape(3, 3)))
        np.add.at(mass, list(ids), 1000 * volume / 4)
    for i, x in enumerate(builder.particle_q):
        builder.particle_mass[i] = 0.0 if (cantilever and x[0] == 0) or (fixed and i == 0) else float(mass[i])
    return builder.finalize(device=device)
