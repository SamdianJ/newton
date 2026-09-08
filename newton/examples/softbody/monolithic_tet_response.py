# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Demonstration fixtures for nonlinear stretching and released vibration."""

import hashlib
import json
from time import perf_counter

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_tet_compare import build_case, reference_matrices

DURATION = 12.0


def gravity_acceleration(time_s):
    """Return downward gravity magnitude [m/s²], ramping to 9.81 in two seconds."""
    u = np.clip(time_s / 2, 0, 1)
    return float(9.81 * u * u * (3 - 2 * u))


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

    def __init__(self, device, *, experiment, variant, refinement=None):
        if experiment not in ("material", "mass", "gravity") or variant not in range(
            3 if experiment == "gravity" else 2
        ):
            raise ValueError("Invalid experiment or comparison variant")
        if refinement is not None and (
            experiment != "gravity" or type(refinement) is not int or refinement not in range(1, 9)
        ):
            raise ValueError("Explicit refinement requires gravity and an integer in [1,8]")
        self.experiment = experiment
        self.dt = 0.01 if experiment == "mass" else 0.02
        self.duration = 36.0 if experiment == "gravity" else DURATION
        material = "kim_stable_no_log" if experiment == "material" and variant == 0 else "smith_log_stabilized"
        mass = "lumped" if experiment == "mass" and variant == 0 else "consistent"
        self.label = ("Kim", "Smith")[variant] if experiment == "material" else mass
        if experiment == "gravity":
            self.label = ("coarse", "medium", "fine")[variant] if refinement is None else f"r{refinement}"
        self.model, self.solver, self.state, self.control, self.rest, self.weights, _, base = build_case(
            device,
            material=material,
            mass=mass,
            refinement=(refinement or variant + 1)
            if experiment == "gravity"
            else (2 if experiment == "material" else 1),
            direction="axial",
            dt=self.dt,
            density=10.0 if experiment == "gravity" else None,
        )
        self.fixed = self.model.particle_inv_mass.numpy() == 0
        self.indices = self.model.tet_indices.numpy()
        self.poses = self.model.tet_poses.numpy().astype(float)
        self.volumes = 1 / (6 * np.linalg.det(self.poses))
        self.materials = self.model.tet_materials.numpy().astype(float)
        stiffness, self.mass = reference_matrices(self.model, mass, density=base["density"])
        self.total_mass = float(self.mass[::3, ::3].sum())
        if experiment == "gravity":
            dynamic = np.repeat(~self.fixed, 3)
            load = self.mass @ np.tile([0, 0, -9.81], self.model.particle_count)
            linear = np.zeros(3 * self.model.particle_count)
            linear[dynamic] = np.linalg.solve(stiffness[np.ix_(dynamic, dynamic)], load[dynamic])
            self.linear_static_tip = float(-self.weights @ linear.reshape(-1, 3)[:, 2])
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
            "duration": self.duration,
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
        if experiment == "gravity":
            self.manifest.update(
                gravity=[0, 0, -9.81],
                loading="2 s smooth gravity ramp, then hold to 36 s; no external nodal forces",
                total_mass_kg=self.total_mass,
                linear_static_tip_m=self.linear_static_tip,
                tip_convention="area-weighted end displacement, positive downward",
            )
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
        record = {
            "time": time_s,
            "tip_m": float(-self.weights @ (x[:, 2] - self.rest[:, 2]))
            if self.experiment == "gravity"
            else float(self.weights @ (x[:, 0] - self.rest[:, 0])),
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
        if self.experiment == "gravity":
            record.update(
                gravity_m_s2=gravity_acceleration(time_s), max_speed_m_s=float(np.max(np.linalg.norm(v, axis=1)))
            )
        return record

    def step(self, *, profile=False):
        """Advance one step; optionally separate synchronized solver and host measurement time."""
        start = perf_counter() if profile else 0
        time_s = (self.steps + 1) * self.dt
        force = traction(time_s) if self.experiment == "material" else 0.0
        external = np.zeros_like(self.rest, dtype=np.float32)
        external[:, 0] = self.weights * force
        self.state.particle_f.assign(external)
        if self.experiment == "gravity":
            self.model.set_gravity((0, 0, -gravity_acceleration(time_s)))
        if profile:
            wp.synchronize_device(self.model.device)
            solver_start = perf_counter()
        self.solver.step(self.state, self.state, self.control, None, self.dt)
        if profile:
            wp.synchronize_device(self.model.device)
            solver_end = perf_counter()
        stats = self.solver.last_stats
        if stats.rolled_back:
            raise RuntimeError(f"{self.label} rollback at {time_s}: {stats.failure_reason}")
        self.records.append(self.measure(time_s, force, stats))
        self.steps += 1
        if profile:
            end = perf_counter()
            return {
                "step_ms": 1000 * (end - start),
                "solver_ms": 1000 * (solver_end - solver_start),
                "measurement_ms": 1000 * (end - solver_end),
                "nonlinear_iterations": stats.nonlinear_iterations,
                "linear_iterations": stats.linear_iterations,
            }

    def summary(self):
        records = self.records
        peaks = vibration_peaks(records) if self.experiment == "mass" else []
        period = float(np.mean(np.diff([p[0] for p in peaks]))) if len(peaks) >= 3 else None
        result = {
            "demo_verified": bool(
                self.steps * self.dt >= self.duration - 1e-10
                and np.mean([r["converged"] for r in records[1:]]) >= 0.99
                and min(r["min_detF"] for r in records) >= 0.2
                and all(not r["converged"] or max(r["residual_ratios"]) <= 1 for r in records)
                and (
                    len(peaks) >= 4
                    if self.experiment == "mass"
                    else max(r["tip_m"] for r in records) >= (0.05 if self.experiment == "gravity" else 0.3)
                )
            ),
            "steps": self.steps,
            "peak_tip_m": max(r["tip_m"] for r in records),
            "min_detF": min(r["min_detF"] for r in records),
            "converged_fraction": float(np.mean([r["converged"] for r in records[1:]])) if self.steps else None,
            "period_s": period,
            "positive_peaks": peaks,
            "last_first_peak_ratio": peaks[-1][1] / peaks[0][1] if len(peaks) >= 3 else None,
        }
        if self.experiment == "gravity":
            tail = [r for r in records if r["time"] >= self.duration - 2]
            mean = float(np.mean([r["tip_m"] for r in tail])) if tail else None
            span = float(np.ptp([r["tip_m"] for r in tail])) if tail else None
            speed = max((r["max_speed_m_s"] for r in tail), default=None)
            settled = bool(tail and span < 0.0001 and speed < 0.001)
            result.update(
                tail_mean_tip_m=mean,
                tail_tip_range_m=span,
                tail_max_speed_m_s=speed,
                near_static=settled,
                linear_static_tip_m=self.linear_static_tip,
                self_weight_compliance_m_n=mean / (self.total_mass * 9.81) if settled else None,
            )
            result["demo_verified"] = bool(result["demo_verified"] and settled)
        return result
