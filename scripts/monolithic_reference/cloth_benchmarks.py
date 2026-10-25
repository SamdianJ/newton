# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""No-contact shell loading/unloading using the public monolithic transaction."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .cloth_oracle import shell_energy
from .material_recipes import shell_coefficients, shell_stencils
from .numerical_gates import rounding_floor, row_excess

RECIPE_PATH = Path(__file__).with_name("cloth_recipes.json")
RECIPES = json.loads(RECIPE_PATH.read_text())


class ClothBenchmark:
    """Exercise flat and naturally folded shells under frozen nodal loading."""

    def __init__(self, kind, *, device="cpu", pcg_mode="diagnostic", use_cuda_graph=False, precision="fp64"):
        self.kind = kind
        self.options = {
            "device": device,
            "pcg_mode": pcg_mode,
            "use_cuda_graph": use_cuda_graph,
            "precision": precision,
        }
        self.recipe = r = RECIPES["common"] | RECIPES[kind]
        self.dt = r["dt"]
        self.total_steps = sum(r[k] for k in ("ramp_steps", "hold_steps", "unload_steps", "recovery_steps"))
        n = r["cells"]
        uv = np.array([(i, j) for i in range(n + 1) for j in range(n + 1)]) / n * r["width"] - r["width"] / 2
        angle = r["fold_angle"] / 2
        positions = np.c_[uv[:, 0] * np.cos(angle), uv[:, 1], abs(uv[:, 0]) * np.sin(angle)]
        self.triangles = np.array(
            [
                face
                for i in range(n)
                for j in range(n)
                for face in (
                    (i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1),
                    (i * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1),
                )
            ]
        )
        self.fixed = np.any(np.isclose(abs(uv), r["width"] / 2), axis=1)
        self.center = (n // 2) * (n + 1) + n // 2
        self.material = shell_coefficients(
            {"young_modulus_Pa": r["young_modulus"], "poisson_ratio": r["poisson_ratio"], "thickness_m": r["thickness"]}
        )
        # Reference coordinates are exactly those that the builder publishes.
        positions = positions.astype(np.float32).astype(float)
        faces = positions[self.triangles]
        area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
        self.mass = np.zeros(len(positions))
        np.add.at(self.mass, self.triangles.ravel(), np.repeat(area * r["surface_density"] / 3, 3))
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for i, x in enumerate(positions):
            builder.add_particle(wp.vec3(x), wp.vec3(0), 0 if self.fixed[i] else self.mass[i])
        for face in self.triangles:
            builder.add_triangle(
                *face, tri_ke=self.material[1], tri_ka=self.material[0], tri_kd=0, tri_drag=0, tri_lift=0
            )
        self.model = builder.finalize(device=device)
        self.rest = self.model.particle_q.numpy().astype(float)
        self.rest_area = area
        self.solver = SolverMonolithic(
            self.model,
            friction_velocity_scale=0.1,
            cloth_inputs=[
                SolverMonolithic.ClothInput(
                    triangle_ids=tuple(range(len(self.triangles))),
                    bending=self.material[2:],
                    thickness=r["thickness"],
                    surface_density=r["surface_density"],
                )
            ],
            precision=SolverMonolithic.Precision(precision),
            pcg_mode=SolverMonolithic.PCGMode(pcg_mode),
            use_cuda_graph=use_cuda_graph,
            newton_max_iterations=60,
            linear_max_iterations=300,
            linear_tolerance=1e-4,
        )
        self.state = self.solver.create_state(dt=self.dt)
        self.control = self.solver.create_control()
        self.steps = 0
        self.records = []
        self.checkpoint_error = None
        self.peak_positions = self.rest.copy()
        self.reference_stencils = shell_stencils(self.rest, self.triangles)

    def load(self, step):
        """Apply a continuous ramp, hold, unload and zero-load recovery."""
        r = self.recipe
        if step <= r["ramp_steps"]:
            return step / r["ramp_steps"]
        if step <= r["ramp_steps"] + r["hold_steps"]:
            return 1.0
        return max(0.0, 1 - (step - r["ramp_steps"] - r["hold_steps"]) / r["unload_steps"])

    def step(self):
        """Require actual convergence and record independent physical quantities."""
        forces = np.zeros((self.model.particle_count, 3), dtype=np.float32)
        forces[self.center, 2] = self.recipe["force"] * self.load(self.steps + 1)
        self.state.particle_f.assign(forces)
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        if not self.solver.last_stats.converged:
            raise AssertionError(f"Step {self.steps + 1}: {self.solver.last_stats.status}")
        self.steps += 1
        x, v = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        np.testing.assert_array_equal(x[self.fixed], self.rest[self.fixed])
        displacement = float(np.max(np.linalg.norm(x - self.rest, axis=1)))
        if not self.records or displacement > max(row["displacement"] for row in self.records):
            self.peak_positions = x.copy()
        energy = sum(
            (
                shell_energy(rest, current, self.material, boundary=boundary)
                for (rest, boundary), (current, _) in zip(
                    self.reference_stencils, shell_stencils(x, self.triangles), strict=True
                )
            ),
            start=np.zeros(2),
        )
        faces = x[self.triangles]
        area_ratio = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / (
            2 * self.rest_area
        )
        rows = np.asarray(self.solver.last_stats.variational["residual"])
        residual = float(np.max(abs(rows)))
        floor = rounding_floor(self.solver)
        # The residual gate admits each row up to the solver's rounding floor.
        excess = row_excess(rows, floor, self.recipe["maximum_residual"])
        kinetic = float(np.sum(self.mass[~self.fixed, None] * v[~self.fixed] ** 2) / 2)
        n = self.recipe["cells"]
        left, right = x[self.center - n - 1] - x[self.center], x[self.center + n + 1] - x[self.center]
        fold = float(np.pi - np.arccos(np.clip(left @ right / np.linalg.norm(left) / np.linalg.norm(right), -1, 1)))
        self.records.append(
            {
                "step": self.steps,
                "load": self.load(self.steps),
                "displacement": displacement,
                "membrane": float(energy[0]),
                "bending": float(energy[1]),
                "kinetic": kinetic,
                "residual": residual,
                "residual_floor": float(floor.max(initial=0)),
                "residual_excess": excess,
                "area_min": float(area_ratio.min()),
                "area_max": float(area_ratio.max()),
                "fold": fold,
            }
        )
        if not all(np.isfinite(value) for value in self.records[-1].values()):
            raise AssertionError("Nonfinite physical measurement")
        if (
            excess > 1
            or area_ratio.min() < self.recipe["area_ratio_min"]
            or area_ratio.max() > self.recipe["area_ratio_max"]
        ):
            raise AssertionError(f"Physical step gate failed: {self.records[-1]}")

    def restore_checkpoint(self, path):
        """Compare one continuation step with a fresh solver, then continue that solver."""
        self.solver.save_checkpoint(self.state, path)
        other = ClothBenchmark(self.kind, **self.options)
        other.solver.load_checkpoint(other.state, path)
        other.steps = self.steps
        other.step()
        self.step()
        error = max(
            float(np.max(abs(self.state.particle_q.numpy() - other.state.particle_q.numpy()))),
            float(np.max(abs(self.state.particle_qd.numpy() - other.state.particle_qd.numpy()))),
        )
        if error > self.recipe["checkpoint_tolerance"]:
            raise AssertionError(f"Checkpoint continuation mismatch: {error}")
        self.checkpoint_error = error
        self.solver, self.state, self.control = other.solver, other.state, other.control

    def metrics(self):
        """Summarize measured recovery without changing acceptance thresholds."""
        if not self.records:
            return {"steps": self.steps}
        peak = max(row["displacement"] for row in self.records)
        peak_energy = max(row["membrane"] + row["bending"] for row in self.records)
        final = self.records[-1]
        return {
            "steps": self.steps,
            "peak_displacement": peak,
            "final_displacement_ratio": final["displacement"] / max(peak, 1e-30),
            "peak_energy": peak_energy,
            "final_energy_ratio": (final["membrane"] + final["bending"] + final["kinetic"]) / max(peak_energy, 1e-30),
            "final_fold_error": abs(final["fold"] - self.recipe["fold_angle"]),
            "maximum_residual": max(row["residual"] for row in self.records),
            "maximum_residual_excess": max(row["residual_excess"] for row in self.records),
            "checkpoint_error": self.checkpoint_error,
        }

    def check_final(self):
        """Check the complete recipe; stopped or partial trajectories cannot pass."""
        m, r = self.metrics(), self.recipe
        assert self.steps == self.total_steps, m
        assert m["peak_displacement"] >= r["minimum_peak_displacement"], m
        assert m["final_displacement_ratio"] <= r["final_displacement_ratio"], m
        assert m["final_energy_ratio"] <= r["final_energy_ratio"], m
        assert m["final_fold_error"] <= r["fold_error_radians"], m
        assert m["maximum_residual_excess"] <= 1, m
