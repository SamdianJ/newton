# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the small and nightly rigid-stack recipes without robot assets."""

from contextlib import nullcontext

import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from scripts.monolithic_reference.contact_routes import pooled_audit_route

DT = 0.001
STEPS = 120
HALF_EXTENT = 0.02
MASS = 0.01
GAP = 0.0005
PENETRATION_LIMIT = 0.002
BALANCE_RELATIVE = 1e-5
SUPPORT_RELATIVE = 0.05
HOLD_STEPS = 20
SETTLE_SPEED = GAP / (HOLD_STEPS * DT)
WEIGHT = MASS * 9.81


def stack_scene(device, *, columns=((-0.03, 0), (0.03, 0)), layers=3):
    """Build separated FREE box columns above a static plane."""
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
    cfg = builder.ShapeConfig(density=0, margin=0, has_particle_collision=False)
    for column, (x, y) in enumerate(columns):
        for layer in range(layers):
            z = HALF_EXTENT + GAP + layer * (2 * HALF_EXTENT + GAP)
            body = builder.add_link(
                xform=wp.transform((x, y, z), wp.quat_identity()),
                mass=MASS,
                inertia=wp.diag(wp.vec3(MASS * (2 * HALF_EXTENT) ** 2 / 6)),
                label=f"box_{column}_{layer}",
            )
            joint = builder.add_joint_free(body)
            builder.add_articulation([joint])
            builder.add_shape_box(body, hx=HALF_EXTENT, hy=HALF_EXTENT, hz=HALF_EXTENT, cfg=cfg)
    builder.add_shape_plane(body=-1, width=0, length=0, cfg=cfg)
    return builder.finalize(device=device)


def stack_solver(model, *, spatial, graph=False, query_capacity=4096, support_capacity=None):
    """Use the spatial route, or the complete-candidate pooled audit route, with fixed budgets."""
    options = {} if support_capacity is None else {"contact_factor_capacity": 3 * support_capacity}
    context = nullcontext()
    if spatial:
        options["contact_candidate_capacity"] = query_capacity
    else:
        context = pooled_audit_route()
    with context:
        run = SolverMonolithic(
            model,
            friction_velocity_scale=0.01,
            pcg_mode="production",
            use_cuda_graph=graph,
            precision="fp64",
            publish_policy="strict",
            **options,
        )
    return run
