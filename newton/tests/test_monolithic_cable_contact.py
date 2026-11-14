# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable self and inter-cable contact through the public coupled solver."""

import unittest

import numpy as np

import newton
from newton.solvers.experimental.monolithic import SolverMonolithic

R = 0.002  # cable radius [m]
L = 0.007  # rest edge length [m]
DENSITY = 0.015  # linear density [kg/m]


def cable_input(ids, *, radius=R):
    ids = tuple(int(i) for i in ids)
    return SolverMonolithic.CableInput(
        particle_ids=ids,
        EA=1000.0,
        bending=np.eye(2) * 1e-4,
        GJ=1e-4,
        radius=radius,
        linear_density=DENSITY,
        rest_theta=np.zeros(len(ids) - 1),
        rest_director=np.tile([0.0, 1.0, 0.0], (len(ids) - 1, 1)),
        fixed_theta=np.zeros(len(ids) - 1, dtype=bool),
    )


def cable_contact_solver(
    points,
    chains,
    *,
    config=None,
    radii=None,
    fixed=(),
    gravity=(0.0, 0.0, 0.0),
    device="cpu",
    precision=SolverMonolithic.Precision.FP64,
    pcg_mode=SolverMonolithic.PCGMode.DIAGNOSTIC,
    use_cuda_graph=False,
    **options,
):
    """A solver over cables through ``points`` (lumped masses from the linear density) with cable contact."""
    points = np.asarray(points, dtype=float)
    mass = np.zeros(len(points))
    for chain in chains:
        ids = list(chain)
        half = np.linalg.norm(np.diff(points[ids], axis=0), axis=1) * DENSITY / 2
        mass[ids[:-1]] += half
        mass[ids[1:]] += half
    mass[list(fixed)] = 0.0
    builder = newton.ModelBuilder(gravity=gravity)
    for point, value in zip(points, mass, strict=True):
        builder.add_particle(tuple(point), (0.0, 0.0, 0.0), mass=float(value))
    model = builder.finalize(device=device)
    radii = [R] * len(chains) if radii is None else radii
    solver = SolverMonolithic(
        model,
        cable_inputs=[cable_input(chain, radius=r) for chain, r in zip(chains, radii, strict=True)],
        cable_contact=config,
        friction_velocity_scale=0.01,
        mass_mode="lumped",
        precision=precision,
        pcg_mode=pcg_mode,
        use_cuda_graph=use_cuda_graph,
        **{"newton_max_iterations": 20, **options},
    )
    return solver, solver.create_state(dt=0.001)


def hairpin(leg=20, overlap=0.0004):
    """One cable folded into a hairpin whose legs press together; contact pairs are |i - j| >= 4 edges apart."""
    first = np.column_stack([L * np.arange(leg), np.zeros(leg), np.zeros(leg)])
    turn = np.array([[first[-1, 0] + 0.003, 0.0, 0.0018]])
    back = np.column_stack([first[::-1, 0], np.zeros(leg), np.full(leg, 2 * R - overlap)])
    points = np.vstack([first, turn, back])
    return points, [range(len(points))]


class TestMonolithicCableContact(unittest.TestCase):
    pass


if __name__ == "__main__":
    unittest.main()
