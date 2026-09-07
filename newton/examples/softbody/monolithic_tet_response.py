# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demonstration fixtures for nonlinear stretching and released vibration."""

import hashlib
import json

import numpy as np

from newton.examples.softbody.monolithic_tet_compare import build_case, reference_matrices

DURATION = 12.0


def traction(time_s):
    """Return axial load [N]: four-second ramp, hold, then unloading."""
    u = np.clip(time_s / 4 if time_s <= 8 else (12 - time_s) / 4, 0, 1)
    return float(300 * u * u * (3 - 2 * u))


def vibration_peaks(records):
    """Interpolate positive tip maxima above 1 mm; require three for a period."""
    peaks = []
    for a, b, c in zip(records, records[1:], records[2:], strict=False):
        left, center, right = a["tip_m"], b["tip_m"], c["tip_m"]
        if center > 0.001 and center > left and center > right:
            delta = 0.5 * (left - right) / (left - 2 * center + right)
            peaks.append((b["time"] + delta * (c["time"] - b["time"]), center))
    return peaks


class ResponseCase:
    """Advance a separate simulation through the existing coupled solver."""

    def __init__(self, device, *, experiment, variant):
        if experiment not in ("material", "mass") or variant not in (0, 1):
            raise ValueError("Expected material/mass experiment and variant 0/1")
        self.experiment = experiment
        self.dt = 0.02 if experiment == "material" else 0.01
        material = "kim_stable_no_log" if experiment == "material" and variant == 0 else "smith_log_stabilized"
        mass = "lumped" if experiment == "mass" and variant == 0 else "consistent"
        self.label = ("Kim", "Smith")[variant] if experiment == "material" else mass
        self.model, self.solver, self.state, self.control, self.rest, self.weights, _, base = build_case(
            device,
            material=material,
            mass=mass,
            refinement=2 if experiment == "material" else 1,
            direction="axial",
            dt=self.dt,
        )
        self.fixed = self.model.particle_inv_mass.numpy() == 0
        self.indices = self.model.tet_indices.numpy()
        self.poses = self.model.tet_poses.numpy().astype(float)
        self.volumes = 1 / (6 * np.linalg.det(self.poses))
        self.materials = self.model.tet_materials.numpy().astype(float)
        _, self.mass = reference_matrices(self.model, mass)
        if experiment == "mass":
            initial = self.rest.copy()
            initial[:, 0] += 0.04 * np.sin(np.pi * self.rest[:, 0] / (2 * base["length"]))
            self.state.particle_q.assign(initial.astype(np.float32))
        self.manifest = {
            "schema": "monolithic-tet-response/v1",
            "experiment": experiment,
            "label": self.label,
            "material_model": material,
            "mass_mode": mass,
            "dt": self.dt,
            "duration": DURATION,
            "length_m": base["length"],
            "width_m": base["width"],
            "height_m": base["height"],
            "young_modulus_pa": base["young_modulus"],
            "poisson_ratio": base["poisson_ratio"],
            "density_kg_m3": base["density"],
            "tet_count": self.model.tet_count,
            "tet_config_identity": self.solver.tet_config_identity,
            "topology_sha256": base["topology_sha256"],
            "gravity": [0, 0, 0],
            "contact": False,
            "loading": "300 N axial smooth ramp/hold/unload, 4 s each"
            if experiment == "material"
            else "u_x=0.04*sin(pi*x/(2*L)) m initially; zero velocity and external force",
            "scope": "simulation demonstration; independent unforced articulation; not the frozen G2H gate",
        }
        self.manifest["fixture_sha256"] = hashlib.sha256(json.dumps(self.manifest, sort_keys=True).encode()).hexdigest()
        self.steps = 0
        self.records = [self.measure(0, 0, None)]

    def measure(self, time_s, force, stats):
        """Measure actual returned state, including each case's physical energy."""
        x, v = self.state.particle_q.numpy().astype(float), self.state.particle_qd.numpy().astype(float)
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise AssertionError("Nonfinite response State")
        if not np.array_equal(x[self.fixed], self.rest[self.fixed]) or np.any(v[self.fixed] != 0):
            raise AssertionError("Fixed nodes moved")
        points = x[self.indices]
        f = np.einsum("tij,tjk->tik", (points[:, 1:] - points[:, 0, None]).transpose(0, 2, 1), self.poses)
        det = np.linalg.det(f)
        ic = np.sum(f * f, axis=(1, 2))
        mu, lam = self.materials[:, 0], self.materials[:, 1]
        d = det - 1
        if self.manifest["material_model"] == "smith_log_stabilized":
            energy = 0.5 * (4 * mu / 3) * (ic - 3 - np.log((ic + 1) / 4)) + 0.5 * (lam + 5 * mu / 6) * d * d - mu * d
        else:
            energy = 0.5 * mu * (ic - 3) - mu * d + 0.5 * (lam + mu) * d * d
        return {
            "time": time_s,
            "tip_m": float(self.weights @ (x[:, 0] - self.rest[:, 0])),
            "force_n": force,
            "min_detF": float(det.min()),
            "volume_ratio": float(self.volumes @ det / self.volumes.sum()),
            "elastic_energy_j": float(self.volumes @ energy),
            "kinetic_energy_j": float(0.5 * v.ravel() @ self.mass @ v.ravel()),
            "converged": stats.converged if stats else True,
            "residual_ratios": [stats.convergence_ratio, stats.convergence_ratio_q, stats.convergence_ratio_x]
            if stats
            else [0, 0, 0],
        }

    def step(self):
        time_s = (self.steps + 1) * self.dt
        force = traction(time_s) if self.experiment == "material" else 0.0
        external = np.zeros_like(self.rest, dtype=np.float32)
        external[:, 0] = self.weights * force
        self.state.particle_f.assign(external)
        self.solver.step(self.state, self.state, self.control, None, self.dt)
        stats = self.solver.last_stats
        if stats.rolled_back:
            raise RuntimeError(f"{self.label} rollback at {time_s}: {stats.failure_reason}")
        self.records.append(self.measure(time_s, force, stats))
        self.steps += 1

    def summary(self):
        records = self.records
        peaks = vibration_peaks(records) if self.experiment == "mass" else []
        period = float(np.mean(np.diff([p[0] for p in peaks]))) if len(peaks) >= 3 else None
        return {
            "demo_verified": bool(
                self.steps * self.dt >= DURATION - 1e-10
                and np.mean([r["converged"] for r in records[1:]]) >= 0.99
                and min(r["min_detF"] for r in records) >= 0.2
                and all(not r["converged"] or max(r["residual_ratios"]) <= 1 for r in records)
                and (max(r["tip_m"] for r in records) >= 0.3 if self.experiment == "material" else len(peaks) >= 4)
            ),
            "steps": self.steps,
            "peak_tip_m": max(r["tip_m"] for r in records),
            "min_detF": min(r["min_detF"] for r in records),
            "converged_fraction": float(np.mean([r["converged"] for r in records[1:]])) if self.steps else None,
            "period_s": period,
            "positive_peaks": peaks,
            "last_first_peak_ratio": peaks[-1][1] / peaks[0][1] if len(peaks) >= 3 else None,
        }
