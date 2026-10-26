# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Boundary-driven cable benchmarks using the public monolithic step transaction."""

import json
from pathlib import Path

import numpy as np
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .cable_analytic import cantilever_deflection, localized_helix
from .numerical_gates import block_norms, gate_floor

RECIPES = json.loads(Path(__file__).with_name("cable_recipes.json").read_text())


class CableBenchmark:
    """Own one prescribed physical experiment and its independent analytical comparison."""

    def __init__(
        self,
        kind,
        *,
        segments=None,
        axis=1,
        ea_multiplier=1,
        angle=0.0,
        device="cpu",
        pcg_mode="diagnostic",
        use_cuda_graph=False,
        precision="fp64",
    ):
        self.kind, self.axis = kind, axis
        self.ea_multiplier = ea_multiplier
        if not np.isfinite(angle) or (angle != 0 and kind != "cantilever"):
            raise ValueError("Only the cantilever may be rotated, by a finite angle")
        # Rotation about z of the whole experiment: geometry, frames and load.
        # At zero angle every array below is the axis-aligned recipe exactly.
        self.angle = float(angle)
        c, s_ = np.cos(np.radians(self.angle)), np.sin(np.radians(self.angle))
        self.rotation = np.array([[c, -s_, 0.0], [s_, c, 0.0], [0.0, 0.0, 1.0]]) if self.angle else np.eye(3)
        self.recipe = r = RECIPES[kind]
        self.segments = n = segments or r["segments"][-1]
        if n < 4 or axis not in (1, 2) or not np.isfinite(ea_multiplier) or ea_multiplier <= 0:
            raise ValueError("Require at least four segments, a transverse axis and positive EA multiplier")
        self.dt, self.steps = r["dt"], 0
        self.total_steps = (
            r.get("ramp_steps", 0) + r.get("rotation_steps", 0) + r.get("shortening_steps", 0) + r["hold_steps"]
        )
        self.s = np.linspace(0, r["length"], n + 1)
        # Model stores the reference geometry in FP32. Mass uses those same
        # lengths, while the simulation's unknowns and E/r use the solver precision.
        rest = np.zeros((n + 1, 3), dtype=np.float32)
        rest[:, 0] = self.s - r["length"] / 2
        length = np.diff(rest[:, 0]).astype(float)
        if self.angle:
            rest = (rest.astype(float) @ self.rotation.T).astype(np.float32)
            length = np.linalg.norm(np.diff(rest.astype(float), axis=0), axis=1)
        mass = np.r_[length[0] / 2, (length[:-1] + length[1:]) / 2, length[-1] / 2] * r["linear_density"]
        fixed = np.zeros(n + 1, dtype=bool)
        fixed[:2] = True
        fixed_theta = np.zeros(n, dtype=bool)
        fixed_theta[0] = True
        if kind == "localized_helix":
            fixed[-2:] = True
            fixed_theta[-1] = True
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for x, m, pin in zip(rest, mass, fixed, strict=True):
            builder.add_particle(x, (0, 0, 0), 0 if pin else float(m))
        self.model = builder.finalize(device=device)
        self.rest = self.model.particle_q.numpy().astype(float)
        bending = np.diag(r["bending"]) if kind == "cantilever" else np.eye(2) * r["bending"]
        director = np.tile(self.rotation @ np.array([0.0, 1.0, 0.0]), (n, 1))
        if self.angle:
            # Keep the rotated director exactly normal to the stored FP32 edges.
            edges = np.diff(self.rest, axis=0)
            edges /= np.linalg.norm(edges, axis=1)[:, None]
            director -= np.sum(director * edges, axis=1)[:, None] * edges
            director /= np.linalg.norm(director, axis=1)[:, None]
        cable = SolverMonolithic.CableInput(
            particle_ids=tuple(range(n + 1)),
            EA=r["EA"] * ea_multiplier,
            bending=bending,
            GJ=r["twisting"],
            radius=r["radius"],
            linear_density=r["linear_density"],
            rest_theta=np.zeros(n),
            rest_director=director,
            fixed_theta=fixed_theta,
        )
        self.solver_options = {
            "cable_inputs": (cable,),
            "friction_velocity_scale": 0.1,
            "mass_mode": "lumped",
            "newton_max_iterations": r["newton_max_iterations"],
            "line_search_max_iterations": 20,
            "linear_max_iterations": 400,
            "linear_tolerance": 0.5,
            "pcg_mode": SolverMonolithic.PCGMode(pcg_mode),
            "use_cuda_graph": use_cuda_graph,
            "precision": SolverMonolithic.Precision(precision),
        }
        self.solver = SolverMonolithic(self.model, **self.solver_options)
        self.state = self.solver.create_state(dt=self.dt)
        self.control = self.solver.create_control()
        if kind == "localized_helix":
            self.reference = localized_helix(
                self.s - r["length"] / 2,
                length=r["length"],
                bending=r["bending"],
                twisting=r["twisting"],
                rotation=2 * np.pi * r["rotation_turns"],
                shortening=r["shortening"],
            )
            # A tiny, deterministic transverse perturbation breaks exact straight
            # symmetry. It does not alter natural curvature or impose the answer.
            x = self.rest.copy()
            u = np.linspace(0, 1, n + 1)
            seed = r["seed_amplitude"] * np.sin(np.pi * u) ** 2
            seed[fixed] = 0
            x[:, 1] = seed * np.cos(6 * np.pi * u)
            x[:, 2] = seed * np.sin(6 * np.pi * u)
            t = np.diff(x, axis=0)
            t /= np.linalg.norm(t, axis=1)[:, None]
            old = np.tile([1.0, 0.0, 0.0], (n, 1))
            d = np.tile([0.0, 1.0, 0.0], (n, 1))
            v = np.cross(old, t)
            d += np.cross(v, d) + np.cross(v, np.cross(v, d)) / (1 + t[:, :1])
            v = np.cross(t[:-1], t[1:])
            transported = (
                d[:-1]
                + np.cross(v, d[:-1])
                + np.cross(v, np.cross(v, d[:-1])) / (1 + np.sum(t[:-1] * t[1:], axis=1))[:, None]
            )
            phi = np.arctan2(np.sum(t[1:] * np.cross(transported, d[1:]), axis=1), np.sum(transported * d[1:], axis=1))
            self.state.particle_q.assign(x)
            self.state._monolithic_coupled.cable_ref_director.assign(d)
            self.state._monolithic_coupled.cable_ref_twist.assign(phi)
            self.solver.reset_state(self.state, dt=self.dt)
        # Raw block L2 residuals and their ratios to max(gate, precision floor L2).
        self.maximum_residual = {"position": 0.0, "theta": 0.0, "position_excess": 0.0, "theta_excess": 0.0}

    def step(self):
        """Apply prescribed loading, solve all coordinates, and require convergence."""
        r, step = self.recipe, self.steps + 1
        if self.kind == "cantilever":
            force = np.zeros((self.segments + 1, 3), dtype=np.float32)
            force[-1] = r["force"] * min(step / r["ramp_steps"], 1) * self.rotation[:, self.axis]
            self.state.particle_f.assign(force)
        else:
            target = self.rest.copy()
            shortening = r["shortening"] * np.clip((step - r["rotation_steps"]) / r["shortening_steps"], 0, 1)
            target[:2, 0] += shortening / 2
            target[-2:, 0] -= shortening / 2
            self.control._monolithic_cable.target_position.assign(target)
            theta = np.zeros(self.segments)
            theta[-1] = 2 * np.pi * r["rotation_turns"] * min(step / r["rotation_steps"], 1)
            self.control._monolithic_cable.target_theta.assign(theta)
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        if not self.solver.last_stats.converged:
            raise RuntimeError(f"{self.kind} step {step}: {self.solver.last_stats}")
        self.steps = step
        residual = self.solver.last_stats.variational["residual"]
        floor = gate_floor(self.solver)
        for part, group in (("position", "x"), ("theta", "theta")):
            value, rounding = block_norms(self.solver, residual, floor, f"cable_0_{group}")
            gate = r["gates"][f"raw_{part}_residual"]
            self.maximum_residual[part] = max(self.maximum_residual[part], value)
            self.maximum_residual[part + "_excess"] = max(
                self.maximum_residual[part + "_excess"], value / max(gate, rounding)
            )

    def restore_checkpoint(self, path):
        """Replace the solver with a fresh instance and continue the same experiment."""
        self.solver.save_checkpoint(self.state, path)
        old_control = self.control._monolithic_cable
        solver = SolverMonolithic(self.model, **self.solver_options)
        state = solver.create_state(dt=self.dt)
        solver.load_checkpoint(state, path)
        control = solver.create_control()
        for name in ("target_position", "target_theta", "torque"):
            getattr(control._monolithic_cable, name).assign(getattr(old_control, name))
        self.solver, self.state, self.control = solver, state, control

    def metrics(self):
        """Compare physical coordinates and raw balances with the fixed reference."""
        r = self.recipe
        x = self.state.particle_q.numpy().copy()
        edges = np.diff(x, axis=0)
        lengths = np.linalg.norm(edges, axis=1)
        rest_lengths = np.linalg.norm(np.diff(self.rest, axis=0), axis=1)
        result = {
            "steps": self.steps,
            "complete": self.steps == self.total_steps,
            "maximum_axial_strain": float(np.max(abs(lengths / rest_lengths - 1))),
            "velocity": float(np.max(np.linalg.norm(self.state.particle_qd.numpy(), axis=1))),
            "raw_position_residual": self.maximum_residual["position"],
            "raw_theta_residual": self.maximum_residual["theta"],
            "raw_position_residual_excess": self.maximum_residual["position_excess"],
            "raw_theta_residual_excess": self.maximum_residual["theta_excess"],
        }
        # Representation limits of the stored coordinates in the solver precision:
        # one spacing per edge end bounds the measurable strain, one per step the
        # measurable velocity. Gates at or below them cannot be resolved (P8b2 D-1).
        native = np.float32 if self.solver.precision == SolverMonolithic.Precision.FP32 else np.float64
        spacing = float(np.spacing(native(np.abs(x).max())))
        result["axial_strain_resolution"] = 2 * spacing / float(rest_lengths.min())
        result["velocity_resolution"] = spacing / self.dt
        if self.kind == "cantilever":
            w = cantilever_deflection(self.s, length=r["length"], force=r["force"], bending=r["bending"][self.axis - 1])
            # Compare in the experiment's own frame; at zero angle this is the identity.
            x = x.astype(float) @ self.rotation
            reference = self.rest @ self.rotation
            reference[:, self.axis] = w
            result.update(
                relative_tip_error=float(abs(x[-1, self.axis] - w[-1]) / abs(w[-1])),
                relative_centerline_error=float(np.max(np.linalg.norm(x - reference, axis=1)) / abs(w[-1])),
            )
        else:
            ref = self.reference
            reference = ref["positions"].copy()
            # The localized branch has arbitrary axial phase. Only rotate its
            # transverse plane; do not fit scale, torque, tilt or localization.
            a, b = x[:, 1] + 1j * x[:, 2], reference[:, 1] + 1j * reference[:, 2]
            phase = np.angle(np.vdot(b, a))
            rotated = b * np.exp(1j * phase)
            reference[:, 1], reference[:, 2] = rotated.real, rotated.imag
            tangent = edges / lengths[:, None]
            mid = (self.s[:-1] + self.s[1:]) / 2 - r["length"] / 2
            theory = localized_helix(
                mid,
                length=r["length"],
                bending=r["bending"],
                twisting=r["twisting"],
                rotation=2 * np.pi * r["rotation_turns"],
                shortening=r["shortening"],
            )
            envelope = (tangent[:, 0] - np.cos(ref["max_tilt"])) / (1 - np.cos(ref["max_tilt"]))
            h = self.state._monolithic_coupled
            twist = np.diff(h.cable_theta.numpy()) + h.cable_ref_twist.numpy()
            torque = r["twisting"] * twist / ((rest_lengths[:-1] + rest_lengths[1:]) / 2)
            # Fuller solid angle of the tangent polygon, closed at +x. This
            # reference branch stays in the +x hemisphere, so no branch ambiguity.
            numerator = np.cross(tangent[:-1], tangent[1:])[:, 0]
            denominator = 1 + tangent[:-1, 0] + tangent[1:, 0] + np.sum(tangent[:-1] * tangent[1:], axis=1)
            writhe = np.sum(2 * np.arctan2(numerator, denominator)) / (2 * np.pi)
            result.update(
                relative_centerline_error=float(np.max(np.linalg.norm(x - reference, axis=1)) / (r["shortening"] / 2)),
                envelope_max_error=float(np.max(abs(envelope - theory["envelope"]))),
                relative_torque_error=float(np.max(abs(torque - ref["torque"])) / abs(ref["torque"])),
                writhe_turn_error=float(abs(writhe - ref["writhe_turns"])),
                rotation_balance_error=float(abs(twist.sum() + 2 * np.pi * writhe - 2 * np.pi * r["rotation_turns"])),
                finite_tail_shortening_error=float(abs(ref["finite_shortening"] - r["shortening"])),
                finite_tail_rotation_error=float(abs(ref["finite_end_rotation"] - 2 * np.pi * r["rotation_turns"])),
                torque=float(np.mean(torque)),
                writhe_turns=float(writhe),
                transverse_phase=float(phase),
            )
        return result

    def check_final(self, *, accuracy=True):
        """Require full duration and every applicable frozen single-case gate."""
        result = self.metrics()
        if not result["complete"]:
            raise AssertionError(f"Incomplete {self.kind}: {self.steps}/{self.total_steps}")
        if not all(np.isfinite(value) for value in result.values()):
            raise AssertionError("Nonfinite cable benchmark metrics")
        approximation = {
            "relative_tip_error",
            "relative_centerline_error",
            "envelope_max_error",
            "relative_torque_error",
            "writhe_turn_error",
        }
        # Residual gates admit each block up to the solver's precision floor.
        floored = {"raw_position_residual", "raw_theta_residual"}
        failures = {
            k: (result[k], limit)
            for k, limit in self.recipe["gates"].items()
            if k in result
            and (result[k + "_excess"] > 1 if k in floored else result[k] > limit)
            and (accuracy or k not in approximation)
        }
        if failures:
            raise AssertionError(failures)
        return result
