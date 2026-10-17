# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Closed-form tet benchmarks advanced through the public monolithic step."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .tet_analytic import (
    fixed_free_frequency,
    implicit_euler_oscillation,
    oscillation_period_and_decay,
    timoshenko_tip_deflection,
)

RECIPES = json.loads(Path(__file__).with_name("tet_recipes.json").read_text())
KINDS = ("patch", "cantilever", "axial_vibration")


class TetBenchmark:
    """Own one tet experiment and compare it with its closed-form reference."""

    def __init__(
        self,
        kind,
        *,
        case=None,
        cells=None,
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        device="cpu",
        pcg_mode="diagnostic",
        use_cuda_graph=False,
    ):
        if kind not in KINDS or (kind == "patch") != (case is not None):
            raise ValueError("Unknown tet benchmark or patch case")
        self.kind, self.case, self.recipe = kind, case, RECIPES[kind]
        self.mass_mode = mass_mode
        r = self.recipe
        self.size = np.asarray(r["size"], dtype=float)
        self.cells = tuple(cells or r["cells"][-1])
        self.dt, self.steps, self.history = r["dt"], 0, []
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        # The current constructor requires one articulation; this link has no shape or load.
        link = builder.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)), label="auxiliary")
        builder.add_articulation([builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, limit_ke=0, limit_kd=0)])
        nx, ny, nz = self.cells
        builder.add_soft_grid(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=nx,
            dim_y=ny,
            dim_z=nz,
            cell_x=self.size[0] / nx,
            cell_y=self.size[1] / ny,
            cell_z=self.size[2] / nz,
            density=r["density"],
            k_mu=r["mu"],
            k_lambda=r["lambda"],
            k_damp=0.0,
            add_surface_mesh_edges=False,
            particle_radius=r["particle_radius"],
        )
        self.rest = np.asarray(builder.particle_q, dtype=float)
        # The grid helper assigns each node a whole cell mass; use the tet row sums instead.
        self.nodal_mass = mass = np.zeros(len(self.rest))
        for ids, pose in zip(builder.tet_indices, builder.tet_poses, strict=True):
            volume = 1 / (6 * np.linalg.det(np.asarray(pose, dtype=float).reshape(3, 3)))
            np.add.at(mass, list(ids), r["density"] * volume / 4)
        tol = 1e-9 * self.size.max()
        self.cell = float(np.min(self.size / np.asarray(self.cells)))
        if kind == "patch":
            self.f = np.asarray(r["cases"][case], dtype=float)
            self.fixed = np.any((self.rest < tol) | (self.rest > self.size - tol), axis=1)
            self.target = self.rest @ self.f.T
            # Deterministic interior offset; the solution must return to the homogeneous state.
            wiggle = r["perturbation"] * self.cell * np.sin(np.arange(len(self.rest))[:, None] * [1.3, 2.1, 2.9])
            start = np.where(self.fixed[:, None], self.target, self.target + wiggle)
        else:
            self.fixed = self.rest[:, 0] < tol
            start = self.rest
        for i, x in enumerate(start):
            builder.particle_q[i] = wp.vec3(*x)
            builder.particle_mass[i] = 0.0 if self.fixed[i] else float(mass[i])
        self.model = builder.finalize(device=device)
        options = {}
        if mass_mode == "consistent":
            options["tet_rest_density"] = wp.full(
                self.model.tet_count, r["density"], dtype=float, device=self.model.device
            )
        self.solver = SolverMonolithic(
            self.model,
            contact_model="C-IE",
            globalization="ArmijoOrResNorm",
            allow_nonconservative_energy=True,
            friction_velocity_scale=0.01,
            material_model=material_model,
            mass_mode=mass_mode,
            publish_policy="strict",
            newton_max_iterations=r["newton_max_iterations"],
            pcg_mode=SolverMonolithic.PCGMode(pcg_mode),
            use_cuda_graph=use_cuda_graph,
            **options,
        )
        source = self.model.state()
        self.tip = np.abs(self.rest[:, 0] - self.size[0]) < tol
        if kind == "axial_vibration":
            velocity = np.zeros_like(self.rest)
            velocity[:, 0] = r["velocity_amplitude"] * np.sin(0.5 * np.pi * self.rest[:, 0] / self.size[0])
            velocity[self.fixed] = 0.0
            source.particle_qd.assign(velocity.astype(np.float32))
            self.omega = fixed_free_frequency(self.size[0], r["mu"], r["lambda"], r["density"])
            self.reference = implicit_euler_oscillation(self.omega, self.dt)
            self.total_steps = round(r["periods"] * self.reference[0] / self.dt)
        elif kind == "cantilever":
            self.reference = timoshenko_tip_deflection(
                r["force"], self.size[0], self.size[1], self.size[2], r["mu"], r["lambda"]
            )
            self.total_steps = r["ramp_steps"] + r["hold_steps"]
        else:
            self.total_steps = r["steps"]
        self.state = self.solver.create_state(dt=self.dt, source=source)
        self.control = self.solver.create_control()

    def positions(self):
        """Return native-precision particle positions [m]."""
        return self.state.particle_q.numpy().astype(float)

    def interior_error(self):
        """Return the largest interior distance [m] from the homogeneous patch solution."""
        x, interior = self.positions(), ~self.fixed
        return float(np.linalg.norm(x[interior] - self.target[interior], axis=1).max())

    def step(self):
        """Advance one prescribed step and require strict convergence."""
        r, step = self.recipe, self.steps + 1
        if self.kind == "cantilever":
            force = np.zeros((self.model.particle_count, 3), dtype=np.float32)
            force[self.tip, 2] = -r["force"] * min(step / r["ramp_steps"], 1.0) / self.tip.sum()
            self.state.particle_f.assign(force)
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        if not self.solver.last_stats.converged:
            raise RuntimeError(f"{self.kind} step {step}: {self.solver.last_stats}")
        self.steps = step
        x = self.positions()
        self.history.append((x[self.tip] - self.rest[self.tip]).mean(axis=0))

    def check_final(self):
        """Return metrics and the single-case status against the recipe gates."""
        g = self.recipe["gates"]
        velocity = float(np.linalg.norm(self.state.particle_qd.numpy(), axis=1).max())
        metrics = {"steps": self.steps, "complete": self.steps == self.total_steps, "velocity": velocity}
        if self.kind == "patch":
            metrics["interior_error_over_cell"] = self.interior_error() / self.cell
            passed = metrics["interior_error_over_cell"] <= g["interior_error_over_cell"]
            passed &= velocity <= g["velocity_m_s"]
        elif self.kind == "cantilever":
            deflection = -float(self.history[-1][2])
            metrics.update(deflection=deflection, reference=self.reference)
            metrics["relative_error"] = abs(deflection - self.reference) / self.reference
            passed = velocity <= g["velocity_m_s"]
        else:
            period, ratio = oscillation_period_and_decay(np.asarray(self.history)[:, 0], self.dt)
            metrics.update(
                mass_mode=self.mass_mode, period=period, decay_ratio=ratio, reference_period=self.reference[0]
            )
            metrics["period_error"] = period / self.reference[0] - 1
            metrics["decay_ratio_error"] = abs(ratio / self.reference[1] - 1)
            passed = metrics["decay_ratio_error"] <= g["decay_ratio_error"]
        values = [v for v in metrics.values() if isinstance(v, float)]
        passed &= metrics["complete"] and bool(np.isfinite(values).all())
        return {"status": "PASS" if passed else "FAIL", "metrics": metrics}
