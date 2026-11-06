# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run shared finite-force two-sheet and self-fold experiments in SI units."""

import argparse
import json

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .cloth_contact_run import add_arguments, create_solver, execute, publication
from .material_recipes import shell_coefficients


def sample_targets(times, targets, time_s):
    """Interpolate smooth position targets and their velocities on one physical clock."""
    times, targets = np.asarray(times, dtype=float), np.asarray(targets, dtype=float)
    if (
        times.ndim != 1
        or len(times) < 2
        or targets.shape != (len(times), 3)
        or not np.isfinite(times).all()
        or not np.isfinite(targets).all()
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
        or not np.isfinite(time_s)
        or time_s < 0
    ):
        raise ValueError("Require finite ordered keyframes and nonnegative time")
    i = min(int(np.searchsorted(times[1:], time_s, side="right")), len(times) - 2)
    duration = times[i + 1] - times[i]
    u = np.clip((time_s - times[i]) / duration, 0, 1)
    delta = targets[i + 1] - targets[i]
    return targets[i] + u * u * (3 - 2 * u) * delta, 6 * u * (1 - u) / duration * delta, i


def _grid(nx, ny, length, width):
    if type(nx) is not int or type(ny) is not int or min(nx, ny) < 2 or min(length, width) <= 0:
        raise ValueError("Require positive sheet dimensions and at least two cells per direction")
    x = np.array([(i * length / nx, (j / ny - 0.5) * width, 0) for i in range(nx + 1) for j in range(ny + 1)])
    faces = []
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            b, c, d = a + ny + 1, a + ny + 2, a + 1
            faces.extend(((a, b, c), (a, c, d)))
    return x, np.asarray(faces, dtype=np.int32)


class ClothContactScene:
    """Let actual sheet positions respond to finite external fixture forces."""

    def __init__(
        self, kind, recipe, *, device="cpu", precision="fp64", use_cuda_graph=False, method="off", method_config=None
    ):
        if recipe.get("units") != "SI" or kind not in ("pair", "self-fold"):
            raise ValueError("Require an SI pair/self-fold recipe")
        self.kind, self.recipe = kind, recipe
        self.dt = float(recipe["dt_s"])
        r, material = recipe[kind], recipe["material"]
        self.times = np.asarray(r["times_s"], dtype=float)
        self.targets = np.asarray(
            r["targets_gap_slide_scale" if kind == "pair" else "targets_z_slide_scale"], dtype=float
        )
        sample_targets(self.times, self.targets, 0)
        if (
            not np.isfinite(self.dt)
            or self.dt <= 0
            or not np.isclose(self.times[-1] / self.dt, round(self.times[-1] / self.dt))
        ):
            raise ValueError("Require a positive step and an integral planned step count")
        self.total_steps = round(self.times[-1] / self.dt)
        self.steps, self.phase = 0, r["phases"][0]
        if len(r["phases"]) != len(self.times) - 1:
            raise ValueError("Each keyframe interval must have a phase")
        if kind == "pair":
            sheet, faces = _grid(r["cells"], r["cells"], r["width_m"], r["width_m"])
            sheet[:, 0] -= r["width_m"] / 2
            offset = np.array([0, 0, self.targets[0, 0] / 2])
            rest = np.vstack((sheet - offset, sheet + offset))
            self.triangles = np.vstack((faces, faces + len(sheet)))
            self.owners = np.repeat(np.arange(2), len(faces))
            self.particle_owner = np.repeat(np.arange(2), len(sheet))
            initial = rest.copy()
            fixed, self.driven = np.zeros(len(rest), dtype=bool), np.ones(len(rest), dtype=bool)
        else:
            rest, self.triangles = _grid(r["length_cells"], r["width_cells"], r["length_m"], r["width_m"])
            self.owners = np.zeros(len(self.triangles), dtype=int)
            self.particle_owner = np.zeros(len(rest), dtype=int)
            radius = r["fold_radius_m"]
            a = (r["length_m"] - np.pi * radius) / 2
            if not 0 < radius or a <= r["length_m"] / r["length_cells"]:
                raise ValueError("Fold radius leaves no straight fixed end")
            s = rest[:, 0]
            theta = np.clip((s - a) / radius, 0, np.pi)
            initial = rest.copy()
            curved = s > a
            initial[curved, 0] = a + radius * np.sin(theta[curved])
            initial[curved, 2] = radius * (1 - np.cos(theta[curved]))
            upper = s > a + np.pi * radius
            initial[upper, 0] = a - (s[upper] - a - np.pi * radius)
            fixed = s <= r["length_m"] / r["length_cells"] + 1e-12
            self.driven = s >= r["length_m"] * (1 - 1 / r["length_cells"]) - 1e-12
        self.thicknesses = np.full(len(self.triangles), material["thickness_m"])
        rest = rest.astype(np.float32).astype(float)
        face_x = rest[self.triangles]
        areas = np.linalg.norm(np.cross(face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0]), axis=1) / 2
        self.mass = np.zeros(len(rest))
        np.add.at(self.mass, self.triangles.ravel(), np.repeat(areas * material["surface_density_kg_m2"] / 3, 3))
        coefficients = shell_coefficients(material)
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for i, x in enumerate(rest):
            builder.add_particle(wp.vec3(x), wp.vec3(0), 0.0 if fixed[i] else float(self.mass[i]))
        for face in self.triangles:
            builder.add_triangle(
                *face, tri_ke=coefficients[1], tri_ka=coefficients[0], tri_kd=0, tri_drag=0, tri_lift=0
            )
        self.model = builder.finalize(device=device)
        inputs = [
            SolverMonolithic.ClothInput(
                triangle_ids=tuple(np.flatnonzero(self.owners == owner).tolist()),
                bending=coefficients[2:],
                thickness=material["thickness_m"],
                surface_density=material["surface_density_kg_m2"],
            )
            for owner in np.unique(self.owners)
        ]
        solver_kwargs = dict(
            recipe["solver"],
            friction_coefficient=r["friction_coefficient"],
            precision=SolverMonolithic.Precision(precision),
            pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
            use_cuda_graph=use_cuda_graph,
        )
        self.solver = create_solver(self.model, inputs, solver_kwargs, method, method_config)
        source = self.model.state()
        initial[fixed] = rest[fixed]
        source.particle_q.assign(initial.astype(np.float32))
        self.state = self.solver.create_state(dt=self.dt, source=source)
        self.control = self.solver.create_control()
        self.initial = self.state.particle_q.numpy().copy()
        self.fixture_work = 0.0
        self.metadata = {
            "kind": kind,
            "recipe": recipe,
            "method": method,
            "method_config": method_config,
            "total_mass_kg": float(self.mass.sum()),
            "fixed_vertices": np.flatnonzero(fixed).tolist(),
            "driven_vertices": np.flatnonzero(self.driven).tolist(),
            "gravity_m_s2": [0, 0, 0],
            "initial_state": "two-separated-sheets" if kind == "pair" else "U-shape-with-flat-natural-reference",
            "fixture": "finite explicit external force; no position overwrite",
        }

    def step(self):
        """Freeze fixture forces for one request and advance only valid publications."""
        target, rate, phase = sample_targets(self.times, self.targets, (self.steps + 1) * self.dt)
        positions, velocities = self.state.particle_q.numpy(), self.state.particle_qd.numpy()
        desired, desired_v = self.initial.copy(), np.zeros_like(positions)
        if self.kind == "pair":
            signs = 2 * self.particle_owner - 1
            desired[:, 2] = signs * target[0] / 2
            desired_v[:, 2] = signs * rate[0] / 2
            desired[self.particle_owner == 1, 0] += target[1]
            desired_v[self.particle_owner == 1, 0] = rate[1]
        else:
            desired[self.driven, 2] += target[0]
            desired[self.driven, 0] += target[1]
            desired_v[self.driven, 2], desired_v[self.driven, 0] = rate[0], rate[1]
        omega, zeta = self.recipe["fixture"]["omega_rad_s"], self.recipe["fixture"]["damping_ratio"]
        force = self.mass[:, None] * (omega**2 * (desired - positions) + 2 * zeta * omega * (desired_v - velocities))
        force[~self.driven] = 0
        force *= target[2]
        applied_force = force.astype(np.float32)
        self.state.particle_f.assign(applied_force)
        previous = self.solver.last_stats.step_generation
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        stats = self.solver.last_stats
        self.last_forces = self.solver.get_contact_forces(self.state)
        if publication(stats, self.last_forces, previous_generation=previous) == "HARD_FAILURE":
            raise RuntimeError(f"{stats.status}: {stats.failure_reason}")
        self.fixture_work += float(np.sum(applied_force.astype(float) * (self.state.particle_q.numpy() - positions)))
        self.steps += 1
        self.phase = self.recipe[self.kind]["phases"][phase]
        return stats


def main():
    """Execute a shared recipe with explicit enabled-method or disabled controls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("pair", "self-fold"))
    add_arguments(parser)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    config = json.loads(args.method_config.read_text()) if args.method_config else None
    return execute(
        lambda: ClothContactScene(
            args.kind,
            recipe,
            device=args.device,
            precision=args.precision,
            use_cuda_graph=args.use_cuda_graph,
            method=args.method,
            method_config=config,
        ),
        args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
