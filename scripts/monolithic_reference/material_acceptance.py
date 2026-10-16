# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure the frozen coupled material trajectories, including unsampled geometry."""

import hashlib

import numpy as np
import warp as wp

from .material_scene import MaterialScene


def segment_distance(point, start, end):
    """Return Euclidean distances to closed segments, including their endpoints."""
    edges = end - start
    t = np.clip(np.sum((point - start) * edges, axis=-1) / np.sum(edges * edges, axis=-1), 0, 1)
    return np.linalg.norm(point - start - t[..., None] * edges, axis=-1)


def triangle_distance(point, faces):
    """Use the interior projection or closest edge, independently of contact sampling."""
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    u, v, w = b - a, c - a, point - a
    uu, vv, uv = np.sum(u * u, axis=1), np.sum(v * v, axis=1), np.sum(u * v, axis=1)
    wu, wv = np.sum(w * u, axis=1), np.sum(w * v, axis=1)
    denominator = uu * vv - uv * uv
    s, t = (wu * vv - wv * uv) / denominator, (wv * uu - wu * uv) / denominator
    inside = (s >= 0) & (t >= 0) & (s + t <= 1)
    normal = np.cross(u, v)
    distance = abs(np.sum(w * normal, axis=1)) / np.linalg.norm(normal, axis=1)
    boundary = np.minimum.reduce(
        [segment_distance(point, a, b), segment_distance(point, b, c), segment_distance(point, c, a)]
    )
    return np.where(inside, distance, boundary)


def snapshot(state):
    """Copy every authoritative state/history array for exact checkpoint validation."""
    return {
        prefix + name: value.numpy().copy()
        for prefix, owner in (("", state), ("history.", state._monolithic_coupled))
        for name, value in vars(owner).items()
        if isinstance(value, wp.array)
    }


def natural_identity(solver):
    """Hash the immutable material references, including natural curvature and twist."""
    digest = hashlib.sha256()
    for material in (*solver._materials.cloth, *solver._materials.cable):
        for key, value in sorted(material.items()):
            digest.update(key.encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


class MaterialAcceptance(MaterialScene):
    """Record all committed frames and enforce the original physical gates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.positions, self.angles, self.coordinates = [], [], [], []
        self.natural_identity = natural_identity(self.solver)
        self.restart = None
        self.resume_position_error = self.resume_theta_error = 0.0

    def measure(self):
        """Measure native raw equations, geometry, energy and both force endpoints."""
        solver, state, r = self.solver, self.state, self.recipe
        x, theta = state.particle_q.numpy(), state._monolithic_coupled.cable_theta.numpy()
        residual = np.asarray(solver.last_stats.variational["residual"])
        rows = np.asarray(solver._actors.maps["theta_to_global"], dtype=int)
        force_mask = np.ones(len(residual), dtype=bool)
        force_mask[rows[rows >= 0]] = False
        torque = max(
            (
                np.max(abs(block.raw.numpy()[3 * block.ids.size :][~block.reference["fixed_theta"]]), initial=0)
                for block in solver._cables
            ),
            default=0,
        )
        energy = sum(block.objective.energy.numpy().sum() for block in solver._cloths)
        energy += sum(
            block.objective.stretch_energy.numpy().sum() + block.objective.bend_twist_energy.numpy().sum()
            for block in solver._cables
        )
        center = state.body_q.numpy()[0, :3]
        if r["material"] == "cable":
            ratio = np.linalg.norm(np.diff(x, axis=0), axis=1) / np.linalg.norm(np.diff(self.rest, axis=0), axis=1)
            distance = segment_distance(center, x[:-1], x[1:]).min()
            penetration = max(0, r["driver_radius_m"] + r["radius_m"] - distance)
        else:
            faces = x[self.triangles]
            ratio = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / (
                2 * self.rest_area
            )
            if r["driver_shape"] == "sphere":
                penetration = max(
                    0, r["driver_radius_m"] + r["thickness_m"] / 2 - triangle_distance(center, faces).min()
                )
            else:
                penetration = max(0, x[:, 2].max() + r["thickness_m"] / 2 - center[2])
        forces = solver.get_contact_forces(state)
        if r["material"] == "cable":
            samples = solver._contact._batch.material_surface.points.numpy()
            gap = np.linalg.norm(samples - center, axis=1) - r["driver_radius_m"]
        else:
            barycentric = np.full((3, 3), 1 / 6)
            np.fill_diagonal(barycentric, 2 / 3)
            samples = np.einsum("ij,fjk->fik", barycentric, x[self.triangles]).reshape(-1, 3)
            gap = (
                np.linalg.norm(samples - center, axis=1) - r["driver_radius_m"]
                if r["driver_shape"] == "sphere"
                else center[2] - samples[:, 2]
            ) - r["thickness_m"] / 2
        sampled_penetration = float(max(0, -gap.min()))
        assert forces.force_valid
        particle, body = forces.particle_force, forces.body_wrench
        balance = np.linalg.norm(particle.sum(axis=0) + body[:, :3].sum(axis=0))
        force_scale = np.linalg.norm(particle, axis=1).sum() + np.linalg.norm(body[:, :3], axis=1).sum()
        pairs = solver._contact.pair_diagnostics()
        wrench = np.asarray([pair["world_wrench"] for pair in pairs])
        pair_balance = np.max(np.linalg.norm(wrench.sum(axis=1), axis=1), initial=0) if pairs else 0
        moment_balance = np.max(np.linalg.norm(wrench[:, :, 3:].sum(axis=1), axis=1), initial=0) if pairs else 0
        moment_scale = np.linalg.norm(wrench[:, :, 3:], axis=2).sum() if pairs else 0
        displacement = np.linalg.norm(x[~self.fixed] - self.rest[~self.fixed], axis=1)
        result = {
            "step": self.steps,
            "force_residual": float(np.max(abs(residual[force_mask]), initial=0)),
            "theta_residual": float(torque),
            "energy": float(energy),
            "penetration": float(penetration),
            "displacement": float(displacement.max(initial=0)),
            "displacement_l2": float(np.linalg.norm(displacement)),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
            "force_balance": float(balance),
            "force_scale": float(force_scale),
            "pair_balance": float(pair_balance),
            "moment_balance": float(moment_balance),
            "moment_scale": float(moment_scale),
            "force_generation": int(forces.force_generation),
            "iterations": int(solver.last_stats.nonlinear_iterations),
        }
        self.records.append(result)
        result["continuous_penetration"] = result["penetration"]
        result["sampled_penetration"] = sampled_penetration
        if self.common.get("penetration_measure") == "sampled_gap":
            result["penetration"] = sampled_penetration
        self.positions.append(x.copy())
        self.angles.append(theta.copy())
        self.coordinates.append(state.joint_q.numpy().copy())
        assert all(np.isfinite(v) for v in result.values()), result
        np.testing.assert_array_equal(x[self.fixed], self.rest[self.fixed])
        assert natural_identity(solver) == self.natural_identity
        return result

    def check_step(self, m):
        """Apply gates to every measured step; no stopped or failing steps are filtered."""
        g, r = self.common["gates"], self.recipe
        assert m["force_residual"] <= g["force_residual_inf_N"], m
        assert m["theta_residual"] <= g["theta_residual_inf_Nm"], m
        if self.steps <= self.common["phases"][0][1]:
            assert m["force_residual"] <= g["rest_force_inf_N"], m
            assert m["theta_residual"] <= g["rest_theta_torque_inf_Nm"], m
        assert m["penetration"] <= r["penetration_max_m"], m
        assert (
            m["force_balance"]
            <= g["contact_force_balance_absolute_N"] + g["contact_balance_relative"] * m["force_scale"]
        ), m
        assert (
            m["moment_balance"]
            <= g["contact_torque_balance_absolute_Nm"] + g["contact_balance_relative"] * m["moment_scale"]
        ), m
        prefix = "edge_stretch" if r["material"] == "cable" else "area_ratio"
        assert r[prefix + "_min"] <= m["ratio_min"] <= m["ratio_max"] <= r[prefix + "_max"], m

    def restore_checkpoint(self, path):
        """Create a genuinely fresh solver and retain it for all remaining steps."""
        self.solver.save_checkpoint(self.state, path)
        other = MaterialAcceptance(self.name, **self.options)
        other.solver.load_checkpoint(other.state, path)
        other.steps = self.steps
        actual, expected = snapshot(other.state), snapshot(self.state)
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        assert other.solver._actors.identity == self.solver._actors.identity
        self.restart = other

    def step(self):
        """Advance and validate both the main and full continuation trajectories."""
        super().step()
        self.check_step(self.measure())
        if self.restart is not None:
            other = self.restart
            other.step()
            self.resume_position_error = max(
                self.resume_position_error,
                float(np.max(abs(self.positions[-1] - other.positions[-1]), initial=0)),
                float(np.max(abs(self.coordinates[-1] - other.coordinates[-1]), initial=0)),
            )
            self.resume_theta_error = max(
                self.resume_theta_error, float(np.max(abs(self.angles[-1] - other.angles[-1]), initial=0))
            )
            assert self.records[-1]["force_generation"] == other.records[-1]["force_generation"]
            assert self.resume_position_error <= self.common["gates"]["resume_position_error_m"]
            assert self.resume_theta_error <= self.common["gates"]["resume_theta_error_rad"]

    def check_final(self):
        """Require the entire loading, unloading and checkpoint experiment."""
        assert self.steps == self.common["steps"]
        assert self.restart is not None and self.restart.steps == self.steps
        g = self.common["gates"]
        assert sum(m["force_scale"] > 0 for m in self.records) >= g["active_contact_steps_min"]
        assert max(m["displacement"] for m in self.records) >= self.recipe["peak_displacement_min_m"]
        phases = {name: (first, last) for first, last, name in self.common["phases"]}
        hold = [m for m in self.records if phases["hold"][0] <= m["step"] - 1 < phases["hold"][1]]
        recover = [m for m in self.records if phases["recover"][0] <= m["step"] - 1 < phases["recover"][1]]
        assert min(m["displacement_l2"] for m in recover) < g["recover_min_displacement_over_peak_max"] * max(
            m["displacement_l2"] for m in hold
        )
        assert min(m["energy"] for m in recover) < max(m["energy"] for m in hold)
