# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent friction-on/off press, drag and release simulations."""

import hashlib
import json

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton

FIXTURE = {
    "schema": "monolithic-contact-friction/v1",
    "dimensions": [0.08, 0.04, 0.04],
    "cells": [4, 2, 2],
    "density": 1000.0,
    "young_modulus": 10000.0,
    "poisson_ratio": 0.3,
    "gravity": [0, 0, 0],
    "dt": 0.002,
    "duration": 4.0,
    "normal_stiffness": 1e7,
    "tangential_stiffness": 2e6,
    "smoothing_width": 0.0001,
    "friction_coefficient": 0.5,
    "pd_ke": 10000.0,
    "pd_kd": 100.0,
    "effort_limit": 200.0,
    "source": "Simulation calibration parameters, not measured hardware/material properties",
}


def target(time_s):
    """Return step-end position and velocity for horizontal and vertical drives."""
    times = np.array([0.0, 0.75, 1.5, 2.5, 3.0, 3.75, 4.0])
    positions = np.array(
        [[0, 0], [0, -0.006], [0.001, -0.006], [0.025, -0.006], [0.025, -0.006], [0.025, 0.008], [0.025, 0.008]]
    )
    segment = min(max(np.searchsorted(times, time_s, side="right") - 1, 0), len(times) - 2)
    duration = times[segment + 1] - times[segment]
    u = np.clip((time_s - times[segment]) / duration, 0, 1)
    blend = u * u * (3 - 2 * u)
    return positions[segment] + blend * (positions[segment + 1] - positions[segment]), 6 * u * (1 - u) * (
        positions[segment + 1] - positions[segment]
    ) / duration


class FrictionCase:
    """Run the shared coupled stepping path on one press-and-drag fixture."""

    def __init__(self, device, *, friction=True):
        b = newton.ModelBuilder(gravity=FIXTURE["gravity"])
        root = b.add_link(mass=1.0, inertia=wp.diag(wp.vec3(0.01)))
        plate = b.add_link(mass=0.5, inertia=wp.diag(wp.vec3(0.001)))
        params = {
            "armature": 0.0,
            "damping": 0.0,
            "friction": 0.0,
            "limit_ke": 0.0,
            "limit_kd": 0.0,
            "target_ke": FIXTURE["pd_ke"],
            "target_kd": FIXTURE["pd_kd"],
            "actuator_mode": newton.JointTargetMode.POSITION_VELOCITY,
            "effort_limit": FIXTURE["effort_limit"],
        }
        jx = b.add_joint_prismatic(-1, root, axis=newton.Axis.X, **params)
        jz = b.add_joint_prismatic(root, plate, axis=newton.Axis.Z, **params)
        b.add_articulation([jx, jz])
        self.plate_center = np.array([0.04, 0.02, 0.052])
        b.add_shape_box(
            plate,
            xform=wp.transform(self.plate_center, wp.quat_identity()),
            hx=0.07,
            hy=0.04,
            hz=0.01,
            cfg=b.ShapeConfig(margin=0.0),
        )
        young, nu = FIXTURE["young_modulus"], FIXTURE["poisson_ratio"]
        b.add_soft_grid(
            pos=wp.vec3(0),
            rot=wp.quat_identity(),
            vel=wp.vec3(0),
            dim_x=4,
            dim_y=2,
            dim_z=2,
            cell_x=0.02,
            cell_y=0.02,
            cell_z=0.02,
            density=FIXTURE["density"],
            k_mu=young / (2 * (1 + nu)),
            k_lambda=young * nu / ((1 + nu) * (1 - 2 * nu)),
            k_damp=0.0,
            particle_radius=0.001,
            add_surface_mesh_edges=False,
        )
        mass = np.zeros(len(b.particle_q))
        for ids, pose in zip(b.tet_indices, b.tet_poses, strict=True):
            volume = 1 / (6 * np.linalg.det(np.asarray(pose).reshape(3, 3)))
            np.add.at(mass, list(ids), FIXTURE["density"] * volume / 4)
        for i, x in enumerate(b.particle_q):
            b.particle_mass[i] = float(mass[i]) if x[2] > 1e-7 else 0.0
        self.model = b.finalize(device=device)
        self.model.request_contact_attributes("force")
        self.solver = SolverMonolithic(
            self.model,
            collision_pipeline=MonolithicCollisionPipeline(self.model, soft_contact_gap=0.002),
            contact_stiffness=FIXTURE["normal_stiffness"],
            normal_smoothing_width=FIXTURE["smoothing_width"],
            friction_coefficient=FIXTURE["friction_coefficient"] if friction else 0.0,
            tangential_stiffness=FIXTURE["tangential_stiffness"],
            material_model="smith_log_stabilized",
            mass_mode="consistent",
            tet_rest_density=wp.full(self.model.tet_count, FIXTURE["density"], dtype=float, device=device),
            joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
        )
        self.state, self.next = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.rest = self.model.particle_q.numpy()
        self.top = np.isclose(self.rest[:, 2], 0.04)
        self.dt = FIXTURE["dt"]
        self.steps = 0
        self.records = []
        self.label = "friction-on" if friction else "friction-off"
        self.manifest = {
            **FIXTURE,
            "friction_coefficient": FIXTURE["friction_coefficient"] if friction else 0.0,
            "tet_count": self.model.tet_count,
            "particle_count": self.model.particle_count,
            "material_model": "smith_log_stabilized",
            "mass_mode": "consistent",
            "topology_sha256": hashlib.sha256(
                self.model.tet_indices.numpy().tobytes() + self.rest.tobytes()
            ).hexdigest(),
        }
        self.manifest["fixture_sha256"] = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()

    def step(self):
        """Freeze controls, advance once, and stop without advancing time on rollback."""
        q, qd = target((self.steps + 1) * self.dt)
        self.control.joint_target_q.assign(q)
        self.control.joint_target_qd.assign(qd)
        self.solver.step(self.state, self.next, self.control, None, self.dt)
        stats = self.solver.last_stats
        if stats.rolled_back:
            raise RuntimeError(f"{self.label} rollback at {self.steps * self.dt}: {stats.failure_reason}")
        self.state, self.next = self.next, self.state
        self.steps += 1
        displacement = self.state.particle_q.numpy() - self.rest
        row = {
            "time": self.steps * self.dt,
            "converged": stats.converged,
            "rho": stats.rho,
            "rho_q": stats.rho_q,
            "rho_x": stats.rho_x,
            "top_displacement_x": float(displacement[self.top, 0].mean()),
            "top_displacement_z": float(displacement[self.top, 2].mean()),
            "q": self.state.joint_q.numpy().tolist(),
            "pd_force": stats.joint_pd_force,
            "min_det_f": stats.min_det_f,
            "iterations": stats.linear_iterations,
            **dict(stats.contact_history),
        }
        self.records.append(row)
        return row

    def summary(self):
        """Summarize actual completed trajectory, including release-history state."""
        r = self.records
        return {
            "steps": len(r),
            "converged_fraction": sum(x["converged"] for x in r) / max(1, len(r)),
            "peak_displacement_x": max((abs(x["top_displacement_x"]) for x in r), default=0),
            "peak_tangent_force": max((x["tangent_force_sum"] for x in r), default=0),
            "stick_samples": sum(x["stick_count"] for x in r),
            "slide_samples": sum(x["slide_count"] for x in r),
            "lost_samples": sum(x["history_lost_count"] for x in r),
            "final_history_energy": r[-1]["history_elastic_energy"] if r else None,
        }


def validate_comparison(cases):
    """Apply frozen component-demo gates; never retune a failed formal run."""
    summaries = [c.summary() for c in cases]
    gates = {
        "complete": all(s["steps"] == round(FIXTURE["duration"] / FIXTURE["dt"]) for s in summaries),
        "convergence": all(s["converged_fraction"] >= 0.99 for s in summaries),
        "deformation": summaries[0]["peak_displacement_x"] >= 0.005,
        "negative_control": summaries[1]["peak_displacement_x"] <= 0.0001 and summaries[1]["peak_tangent_force"] == 0,
        "stick_and_slide": summaries[0]["stick_samples"] >= 100 and summaries[0]["slide_samples"] >= 100,
        "release": summaries[0]["lost_samples"] > 0 and all(s["final_history_energy"] == 0 for s in summaries),
        "finite": all(
            all(np.isfinite([r["rho"], r["rho_q"], r["rho_x"], r["min_det_f"], r["top_displacement_x"]]))
            for c in cases
            for r in c.records
        ),
        "determinant": all(r["min_det_f"] > 0.5 for c in cases for r in c.records),
        "residual": all(
            max(r["rho"], r["rho_q"], r["rho_x"]) <= 1e-4 for c in cases for r in c.records if r["converged"]
        ),
    }
    return {"passed": all(gates.values()), "gates": gates, "summaries": summaries}
