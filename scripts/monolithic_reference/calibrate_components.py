# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure DRAFT tiny, no-contact float32 evidence; never freeze P0 tolerances.

Run from the repository root with ``uv run -m
scripts.monolithic_reference.calibrate_components --output evidence.json``.
The output separates repeated reduction noise from coordinate quantization:
the latter is not permission to increase line-search merit noise.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import warp as wp
from warp.utils import array_inner

from newton._src.solvers.monolithic.articulation import (
    MonolithicArticulationWorkspace,
    eval_articulation_actor_residual,
    eval_articulation_passive_candidate,
)
from newton._src.solvers.monolithic.solver_monolithic import _build_layout, _Candidate
from newton._src.solvers.monolithic.tet import (
    TetScatterBuffers,
    _compute_cofactor_derivative,
    _evaluate_stable_neo_hookean,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
    mat99,
)
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture


@wp.kernel
def _constitutive(
    deformation: wp.array[wp.mat33],
    energy: wp.array[float],
    stress: wp.array[wp.mat33],
    projected: wp.array[mat99],
    cofactor_derivative: wp.array[mat99],
):
    f = deformation[0]
    e, p, h = _evaluate_stable_neo_hookean(f, 1.0, 3846.15380859375, 5769.23095703125)
    energy[0] = e
    stress[0] = p
    projected[0] = h
    cofactor_derivative[0] = _compute_cofactor_derivative(f)


def _block_rms(values):
    values = np.asarray(values, dtype=np.float64)
    return [float(np.linalg.norm(block) / math.sqrt(len(block))) for block in (values, values[:2], values[2:])]


class _Probe:
    def __init__(self, device, dt):
        self.fixture = build_tiny_cpu_fixture(device=device)
        self.model = self.fixture.model
        self.layout = _build_layout(self.model)
        self.candidate = _Candidate(self.model, self.layout)
        self.candidate.load_predictor(self.fixture.state, dt)
        self.articulation = MonolithicArticulationWorkspace(self.model)
        self.residual_q = wp.empty(2, dtype=float, device=device)
        self.pattern = build_tet_triplet_pattern(
            self.model.tet_indices.numpy(), self.layout.particle_to_dynamic.numpy(), x_dof_start=2
        )
        self.scatter = TetScatterBuffers(
            wp.empty(len(self.pattern.ax_rows), dtype=wp.mat33, device=device),
            wp.empty(len(self.pattern.global_rows), dtype=float, device=device),
        )
        self.args = {
            "candidate_particle_q": self.candidate.state.particle_q,
            "particle_q_n": self.fixture.state.particle_q,
            "particle_qd_n": self.fixture.state.particle_qd,
            "frozen_particle_f": self.fixture.state.particle_f,
            "dynamic_particle_ids": self.layout.dynamic_particle_ids,
            "particle_to_dynamic": self.layout.particle_to_dynamic,
            "residual_x": wp.empty(4, dtype=wp.vec3, device=device),
            "dt": dt,
            "min_det_f_guard": 0.05,
            "workspace": create_tet_assembly_workspace(self.pattern, tet_count=1, device=device),
        }

    def evaluate(self, z=None):
        if z is not None:
            self.candidate.z.assign(z)
            self.candidate._recover(self.args["dt"])
        aw = self.articulation
        eval_articulation_passive_candidate(self.model, self.candidate.state, aw)
        aw.generalized_body_force.zero_()
        wp.launch(
            eval_articulation_actor_residual,
            2,
            inputs=[
                0,
                0,
                2,
                aw.M,
                self.candidate.qdd,
                aw.C,
                aw.g,
                self.fixture.control.joint_f,
                aw.generalized_body_force,
                self.residual_q,
            ],
            device=self.model.device,
        )
        assemble_tet_residual_tangent(self.model, **self.args, scatter=self.scatter)
        if self.args["workspace"].failure_flags.numpy()[0] != 0:
            raise AssertionError("The calibration candidate failed the production tet guards")
        residual = np.concatenate((self.residual_q.numpy(), self.args["residual_x"].numpy().ravel()))
        matrix = np.zeros((14, 14), dtype=np.float32)
        matrix[:2, :2] = aw.M.numpy()[0] * np.float32(1.0 / self.args["dt"] ** 2)
        np.add.at(matrix, (self.pattern.global_rows, self.pattern.global_columns), self.scatter.global_values.numpy())
        if not np.all(np.isfinite(matrix)) or np.any(np.diag(matrix) <= 0):
            raise AssertionError("The calibration tangent has an invalid diagonal")
        return residual, matrix


def _fd_sweep(device):
    arrays = [wp.empty(1, dtype=dtype, device=device) for dtype in (wp.mat33, float, wp.mat33, mat99, mat99)]

    def evaluate(f):
        arrays[0].assign(np.asarray([f], dtype=np.float32))
        wp.launch(_constitutive, 1, inputs=arrays, device=device)
        return [array.numpy()[0].copy() for array in arrays[1:]]

    f = np.array([[1.12, 0.07, -0.03], [0.02, 0.91, 0.08], [0.04, -0.02, 1.06]], dtype=np.float32)
    _, stress, projected, dc = evaluate(f)
    raw = (
        projected.astype(np.float64)
        + ((3846.15380859375 + 5769.23095703125) * (np.linalg.det(f.astype(np.float64)) - 1) - 3846.15380859375) * dc
    )
    records = []
    for h in (0.02, 0.006, 0.002, 0.0006, 0.0002):
        gradient, derivative = np.zeros(9), np.zeros((9, 9))
        for index in range(9):
            delta = np.zeros((3, 3), dtype=np.float32)
            delta[index % 3, index // 3] = h
            plus, minus = evaluate(f + delta), evaluate(f - delta)
            gradient[index] = (float(plus[0]) - float(minus[0])) / (2 * h)
            derivative[:, index] = ((plus[1].astype(np.float64) - minus[1]) / (2 * h)).flatten(order="F")
        ge = float(np.linalg.norm(gradient - stress.flatten(order="F")))
        he = float(np.linalg.norm(derivative - raw))
        records.append(
            {
                "step": h,
                "energy_gradient_absolute_error": ge,
                "energy_gradient_relative_error": float(ge / np.linalg.norm(stress)),
                "raw_tangent_absolute_error": he,
                "raw_tangent_relative_error": he / np.linalg.norm(raw),
            }
        )
    if min(row["energy_gradient_relative_error"] for row in records) > 5e-3:
        raise AssertionError("No energy-gradient finite-difference plateau meets the frozen gate")
    if min(row["raw_tangent_relative_error"] for row in records) > 5e-3:
        raise AssertionError("No raw-tangent finite-difference plateau meets the frozen gate")
    return {
        "deformation": f.tolist(),
        "determinant": float(np.linalg.det(f)),
        "variable_scale": 1.0,
        "projected_condition": float(np.linalg.cond(projected.astype(np.float64))),
        "records": records,
    }


def calibrate(device, *, dt, repeats):
    probe = _Probe(device, dt)
    residual, matrix = probe.evaluate()
    dynamic_diagonal = np.concatenate(
        (np.diag(probe.articulation.M.numpy()[0]), np.repeat(probe.model.particle_mass.numpy(), 3))
    ) / np.float32(dt * dt)
    diagonal = np.maximum(np.diag(matrix), np.float32(1e-6) * dynamic_diagonal)
    scale = np.float32(1) / np.sqrt(diagonal)
    arrays = [wp.empty(size, dtype=float, device=device) for size in (14, 2, 12)]
    reductions = [wp.empty(1, dtype=float, device=device) for _ in arrays]
    samples, errors = [], []
    for _ in range(repeats):
        repeated, _ = probe.evaluate()
        scaled = scale * repeated
        norms, reduction_errors = [], []
        for array, reduction, block in zip(arrays, reductions, (scaled, scaled[:2], scaled[2:]), strict=True):
            array.assign(block)
            array_inner(array, array, out=reduction)
            actual = math.sqrt(float(reduction.numpy()[0]))
            reference = float(np.linalg.norm(block.astype(np.float64)))
            norms.append(actual)
            reduction_errors.append(abs(actual - reference))
        samples.append(norms)
        errors.append(reduction_errors)
    samples, errors = np.asarray(samples), np.asarray(errors)
    z = probe.candidate.z.numpy().copy()
    quantization = []
    for axis in range(14):
        for direction in (-math.inf, math.inf):
            changed = z.copy()
            changed[axis] = np.nextafter(z[axis], np.float32(direction))
            changed_residual, _ = probe.evaluate(changed)
            quantization.append(
                {
                    "axis": axis,
                    "direction": direction > 0,
                    "delta": float(changed[axis] - z[axis]),
                    "scaled_step_rms": _block_rms((changed.astype(np.float64) - z) / scale),
                    "scaled_residual_change_rms": _block_rms(scale * (changed_residual - residual)),
                }
            )
    probe.evaluate(z)
    mass = probe.articulation.M.numpy()[0].astype(np.float64)
    predictor_roundoff = np.zeros(14)
    predictor_roundoff[:2] = scale[:2] * (mass @ probe.candidate.qdd.numpy().astype(np.float64))
    spread = np.ptp(samples, axis=0)
    return {
        "device": str(device),
        "device_name": wp.get_device(device).name,
        "dt": dt,
        "repeats": repeats,
        "block_order": ["global", "q", "x"],
        "candidate_z": z.tolist(),
        "initial_residual": residual.tolist(),
        "diagonal": diagonal.tolist(),
        "scale": scale.tolist(),
        "epsilon_d": 1e-6,
        "dynamic_diagonal": dynamic_diagonal.tolist(),
        "diagonal_floor_count": int(np.count_nonzero(diagonal != np.diag(matrix))),
        "scaled_condition": float(np.linalg.cond(scale[:, None] * matrix.astype(np.float64) * scale[None, :])),
        "scaled_norm_samples": samples.tolist(),
        "norm_peak_to_peak": spread.tolist(),
        "norm_float64_reference_absolute_error_max": errors.max(axis=0).tolist(),
        "merit_noise_observed_upper_bound": float(spread[0] / math.sqrt(14)),
        "merit_noise_proposal": float(2 * spread[0] / math.sqrt(14)),
        "merit_noise_margin_factor": 2,
        "predictor_coordinate_roundoff_scaled_merit": _block_rms(predictor_roundoff),
        "coordinate_quantization": quantization,
        "fd_sweep": _fd_sweep(device),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0 or args.repeats < 2:
        parser.error("dt must be positive and finite; repeats must be at least two")
    root = Path(__file__).resolve().parents[2]
    fixture = build_tiny_cpu_fixture()
    fixture_data = asdict(fixture.spec)
    for name in (
        "tet_materials",
        "tet_poses",
        "tet_indices",
        "tri_indices",
        "particle_mass",
        "particle_q",
        "particle_qd",
        "particle_radius",
        "body_mass",
        "body_com",
        "body_inertia",
        "joint_q",
        "joint_qd",
        "joint_parent",
        "joint_child",
        "joint_X_p",
        "joint_X_c",
        "joint_axis",
        "gravity",
    ):
        fixture_data[name] = getattr(fixture.model, name).numpy().tolist()
    fixture_data["source_sha256"] = hashlib.sha256(
        (root / "newton/tests/monolithic_test_utils.py").read_bytes()
    ).hexdigest()
    fixture_hash = hashlib.sha256(json.dumps(fixture_data, sort_keys=True).encode()).hexdigest()
    payload = {
        "status": "DRAFT",
        "scope": "tiny N=14, no shapes/contact, one dt, rest predictor and one-ULP neighbors",
        "excluded": [
            "active contact",
            "large assets",
            "solver convergence",
            "P0/C3 freeze",
            "production linear reduction parity",
        ],
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "fixture_sha256": fixture_hash,
        "fixture": fixture_data,
        "platform": platform.platform(),
        "warp_version": wp.__version__,
        "reduction": "warp.utils.array_inner float32, preallocated out, host sqrt; scale/residual multiply NumPy float32",
        "evidence": [calibrate(device, dt=args.dt, repeats=args.repeats) for device in args.devices],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
