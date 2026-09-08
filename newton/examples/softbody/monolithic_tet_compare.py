# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared G2H fixture and independent small-strain, float64 reference."""

import hashlib
import json

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton

MODES = (
    ("kim_stable_no_log", "lumped"),
    ("smith_log_stabilized", "lumped"),
    ("kim_stable_no_log", "consistent"),
    ("smith_log_stabilized", "consistent"),
)
FIXTURE = {
    "schema": "monolithic-g2h/v1",
    "length": 1.2,
    "width": 0.3,
    "height": 0.3,
    "density": 1000.0,
    "young_modulus": 10000.0,
    "poisson_ratio": 0.3,
    "gravity": [0, 0, 0],
    "dt": 0.02,
    "duration": 4.0,
    "stage_duration": 1.0,
    "axial_force": 10.0,
    "transverse_force": 0.5,
    "parameter_source": "simulation fixture; not measured material properties",
    "gates": {
        "converged_fraction": 0.99,
        "min_detF": 0.2,
        "linear_error_ratio": 0.03,
        "minimum_tip_displacement": 0.002,
        "device_tip_error": 0.00005,
        "refinement_ratio": 0.8,
    },
}


def load_fraction(time_s):
    """Evaluate a C1 loading/hold/unloading/free-response schedule."""
    stage = FIXTURE["stage_duration"]
    if time_s < stage:
        u = max(0, time_s / stage)
        return u * u * (3 - 2 * u)
    if time_s <= 2 * stage:
        return 1.0
    if time_s < 3 * stage:
        u = (time_s - 2 * stage) / stage
        return 1 - u * u * (3 - 2 * u)
    return 0.0


def build_case(device, *, material, mass, refinement=2, dt=0.02, direction="transverse", load_scale=1.0, density=None):
    """Construct one cantilever and an explicitly uncoupled one-DoF articulation."""
    if refinement not in range(1, 9) or type(refinement) is not int or direction not in ("axial", "transverse"):
        raise ValueError("Invalid G2H refinement or load direction")
    if not np.isfinite(dt) or dt <= 0 or abs(round(0.1 / dt) * dt - 0.1) > 1e-12 or dt > 0.1:
        raise ValueError("G2H dt must divide 100 ms")
    if not np.isfinite(load_scale) or not 0 < load_scale <= 1:
        raise ValueError("G2H load scale must be in (0,1]")
    density = FIXTURE["density"] if density is None else density
    if not np.isfinite(density) or density <= 0:
        raise ValueError("Density must be finite and positive")
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    link = b.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
    joint = b.add_joint_revolute(-1, link, limit_ke=0, limit_kd=0)
    b.add_articulation([joint])
    young, nu = FIXTURE["young_modulus"], FIXTURE["poisson_ratio"]
    mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
    b.add_soft_grid(
        pos=wp.vec3(0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0),
        dim_x=3 * refinement,
        dim_y=refinement,
        dim_z=refinement,
        cell_x=FIXTURE["length"] / (3 * refinement),
        cell_y=FIXTURE["width"] / refinement,
        cell_z=FIXTURE["height"] / refinement,
        density=density,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0,
        fix_left=True,
        add_surface_mesh_edges=False,
        particle_radius=0.001,
    )
    # add_soft_grid assigns cell mass to every node; use the same rho-derived
    # row sums in both modes so this fixture isolates mass discretization.
    nodal_mass = np.zeros(len(b.particle_q))
    for ids, pose in zip(b.tet_indices, b.tet_poses, strict=True):
        volume = 1 / (6 * np.linalg.det(np.asarray(pose, dtype=float).reshape(3, 3)))
        np.add.at(nodal_mass, list(ids), density * volume / 4)
    for i, x in enumerate(b.particle_q):
        b.particle_mass[i] = float(nodal_mass[i]) if x[0] > 1e-7 else 0.0
    model = b.finalize(device=device)
    density_array = wp.full(model.tet_count, density, dtype=float, device=device) if mass == "consistent" else None
    solver = SolverMonolithic(
        model,
        collision_pipeline=MonolithicCollisionPipeline(model),
        contact_stiffness=1e5,
        material_model=material,
        mass_mode=mass,
        tet_rest_density=density_array,
    )
    state, control = model.state(), model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    rest = model.particle_q.numpy().astype(float)
    faces = model.tri_indices.numpy()
    weights = np.zeros(model.particle_count)
    for face in faces:
        x = rest[face]
        if np.all(np.abs(x[:, 0] - FIXTURE["length"]) < 1e-7):
            area = np.linalg.norm(np.cross(x[1] - x[0], x[2] - x[0])) / 2
            np.add.at(weights, face, area / 3)
    weights /= weights.sum()
    axis = 0 if direction == "axial" else 2
    force = np.zeros_like(rest)
    force[:, axis] = weights * FIXTURE[direction + "_force"] * load_scale
    manifest = {
        **FIXTURE,
        "density": density,
        "material_model": material,
        "mass_mode": mass,
        "direction": direction,
        "refinement": refinement,
        "dt": dt,
        "load_scale": load_scale,
        "particle_count": model.particle_count,
        "tet_count": model.tet_count,
        "articulation": "independent unforced revolute, no shapes/contact",
        "mass_authority": "explicit rho*V row sums replace add_soft_grid uniform node mass",
        "tet_config_identity": solver.tet_config_identity,
        "topology_sha256": hashlib.sha256(model.tet_indices.numpy().tobytes() + rest.tobytes()).hexdigest(),
    }
    manifest["fixture_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return model, solver, state, control, rest, weights, force, manifest


def reference_matrices(model, mass_mode, *, density=None):
    """Assemble the small-strain isotropic K and full P1 M in NumPy float64."""
    n = model.particle_count
    density = FIXTURE["density"] if density is None else density
    stiffness = np.zeros((3 * n, 3 * n))
    mass = np.zeros((n, n))
    for ids, pose32, material in zip(
        model.tet_indices.numpy(), model.tet_poses.numpy(), model.tet_materials.numpy(), strict=True
    ):
        pose = pose32.astype(float)
        mu, lam = map(float, material[:2])
        volume = 1 / (6 * np.linalg.det(pose))
        gradients = np.vstack((-pose.sum(axis=0), pose))
        for a in range(4):
            for b in range(4):
                ga, gb = gradients[a], gradients[b]
                block = volume * (mu * np.dot(ga, gb) * np.eye(3) + mu * np.outer(gb, ga) + lam * np.outer(ga, gb))
                stiffness[3 * ids[a] : 3 * ids[a] + 3, 3 * ids[b] : 3 * ids[b] + 3] += block
                mass[ids[a], ids[b]] += density * volume / 20 * (2 if a == b else 1)
    if mass_mode == "lumped":
        mass = np.diag(mass.sum(axis=1))
    return stiffness, np.kron(mass, np.eye(3))


class Case:
    """Advance one configuration and collect final-state G2H measurements."""

    def __init__(self, device, **options):
        (self.model, self.solver, self.state, self.control, self.rest, self.weights, self.force, self.manifest) = (
            build_case(device, **options)
        )
        self.dt = options.get("dt", FIXTURE["dt"])
        self.steps = 0
        self.records = []
        self.work = 0.0
        self.indices = self.model.tet_indices.numpy()
        self.poses = self.model.tet_poses.numpy().astype(float)
        self.volumes = 1 / (6 * np.linalg.det(self.poses))
        self.materials = self.model.tet_materials.numpy().astype(float)
        self.fixed = self.model.particle_inv_mass.numpy() == 0
        self.linear_k, self.mass = reference_matrices(self.model, self.manifest["mass_mode"])
        self.dynamic = np.repeat(~self.fixed, 3)
        m = self.mass[np.ix_(self.dynamic, self.dynamic)]
        k = self.linear_k[np.ix_(self.dynamic, self.dynamic)]
        self.linear_inverse = np.linalg.inv(m + self.dt**2 * k)
        self.linear_mass = m
        self.linear_stiffness = k
        self.linear_u = np.zeros(len(m))
        self.linear_v = np.zeros(len(m))

    def step(self):
        time_s = (self.steps + 1) * self.dt
        fraction = load_fraction(time_s)
        old = self.state.particle_q.numpy().astype(float)
        load = self.force * fraction
        self.state.particle_f.assign(load.astype(np.float32))
        self.solver.step(self.state, self.state, self.control, None, self.dt)
        stats = self.solver.last_stats
        if stats.rolled_back:
            raise RuntimeError(f"G2H rollback at {time_s}: {stats.failure_reason}")
        x = self.state.particle_q.numpy().astype(float)
        v = self.state.particle_qd.numpy().astype(float)
        if not np.array_equal(x[self.fixed], self.rest[self.fixed]) or np.any(v[self.fixed] != 0):
            raise AssertionError("G2H fixed nodes moved")
        self.work += float(np.sum(load * (x - old)))
        self.linear_v = self.linear_inverse @ (
            self.linear_mass @ self.linear_v
            + self.dt * (load.ravel()[self.dynamic] - self.linear_stiffness @ self.linear_u)
        )
        self.linear_u += self.dt * self.linear_v
        reference = np.zeros_like(x)
        reference.ravel()[self.dynamic] = self.linear_u
        points = x[self.indices]
        f = np.einsum("tij,tjk->tik", (points[:, 1:] - points[:, 0, None]).transpose(0, 2, 1), self.poses)
        determinant = np.linalg.det(f)
        ic = np.sum(f * f, axis=(1, 2))
        mu, lam = self.materials[:, 0], self.materials[:, 1]
        d = determinant - 1
        if self.manifest["material_model"] == "smith_log_stabilized":
            energy = 0.5 * (4 * mu / 3) * (ic - 3 - np.log((ic + 1) / 4)) + 0.5 * (lam + 5 * mu / 6) * d * d - mu * d
        else:
            energy = 0.5 * mu * (ic - 3) - mu * d + 0.5 * (lam + mu) * d * d
        kinetic = float(0.5 * v.ravel() @ self.mass @ v.ravel())
        record = {
            "time": time_s,
            "load_fraction": fraction,
            "tip": (self.weights @ (x - self.rest)).tolist(),
            "load": load.sum(axis=0).tolist(),
            "linear_tip": (self.weights @ reference).tolist(),
            "min_detF": float(determinant.min()),
            "volume_ratio": float(self.volumes @ determinant / self.volumes.sum()),
            "elastic_energy": float(self.volumes @ energy),
            "kinetic_energy": kinetic,
            "external_work": self.work,
            "converged": stats.converged,
            "ratios": [stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x],
            "nonlinear_iterations": stats.nonlinear_iterations,
            "pcg_iterations": stats.linear_iterations,
            "step_ms": stats.timings["step"],
        }
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise AssertionError("Nonfinite G2H State")
        self.records.append(record)
        self.steps += 1

    def summary(self):
        records = self.records
        gates = FIXTURE["gates"]
        tip = np.array([r["tip"] for r in records])
        reference = np.array([r["linear_tip"] for r in records])
        peak = float(np.max(np.abs(tip))) if records else 0.0
        error = (
            float(np.max(np.abs(tip - reference)) / max(np.max(np.abs(reference)), 1e-30)) if records else float("inf")
        )
        return {
            "passed": bool(
                records
                and self.steps * self.dt >= FIXTURE["duration"] - 1e-10
                and np.mean([r["converged"] for r in records]) >= gates["converged_fraction"]
                and min(r["min_detF"] for r in records) >= gates["min_detF"]
                and peak >= gates["minimum_tip_displacement"] * self.manifest["load_scale"]
                and error <= gates["linear_error_ratio"]
                and all(not r["converged"] or max(r["ratios"]) <= 1 for r in records)
            ),
            "peak_tip_displacement": peak,
            "linear_reference_error_ratio": error,
            "steps": self.steps,
            "converged_fraction": float(np.mean([r["converged"] for r in records])) if records else 0,
            "contact_count": 0,
            "min_detF": min((r["min_detF"] for r in records), default=None),
        }
