# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure component precision and fail closed when aggregating freeze evidence.

Run from the repository root with ``uv run -m
scripts.monolithic_reference.calibrate_components --output evidence.json``.
The output separates repeated reduction noise from coordinate quantization:
the latter is not permission to increase line-search merit noise.
Use ``--matrix`` for the explicit PR-5 material/contact/coordinate sample matrix.
Without that flag, the original tiny no-contact CLI and evidence remain available.
The versioned freeze aggregator is intentionally separate from measurement: it
cannot freeze unless every declared device/case key is present, measured from one
clean source set, and passes its raw gates.  A frozen calibration sub-artifact
does not freeze the V0.1 product or trajectory fixture.
"""

import argparse
import hashlib
import itertools
import json
import math
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp
from warp.utils import array_inner

import newton
from newton._src.solvers.monolithic.articulation import (
    MonolithicArticulationWorkspace,
    articulation_point_jacobian_column,
    eval_articulation_actor_residual,
    eval_articulation_passive_candidate,
)
from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton._src.solvers.monolithic.contact import _evaluate as _evaluate_contact
from newton._src.solvers.monolithic.linear import MonolithicPcgConfig
from newton._src.solvers.monolithic.solver_monolithic import SolverMonolithic, _build_layout, _Candidate
from newton._src.solvers.monolithic.tet import (
    TetScatterBuffers,
    _compute_cofactor_derivative,
    _evaluate_elastic_residual,
    _evaluate_stable_neo_hookean,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
    mat99,
)
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from scripts.monolithic_reference.calibrate_p1q3 import refined_tetrahedron


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


def _finite_diagnostic(value):
    """Encode an unbounded/nonfinite diagnostic as JSON null, never Infinity."""
    value = float(value)
    return value if math.isfinite(value) else None


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
        "projected_condition": _finite_diagnostic(np.linalg.cond(projected.astype(np.float64))),
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
        "scaled_condition": _finite_diagnostic(
            np.linalg.cond(scale[:, None] * matrix.astype(np.float64) * scale[None, :])
        ),
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
    parser.add_argument("--matrix", action="store_true", help="Measure the explicit PR-5 discrete parameter matrix")
    parser.add_argument("--profile", choices=("quick", "freeze"), default="quick")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--batch-count", type=int, default=1)
    parser.add_argument("--aggregate", type=Path, nargs="+", help="Aggregate version-2 raw freeze batches")
    parser.add_argument(
        "--freeze", action="store_true", help="Require complete passing evidence and freeze calibration"
    )
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0 or args.repeats < 2:
        parser.error("dt must be positive and finite; repeats must be at least two")
    if args.aggregate:
        if args.matrix:
            parser.error("--aggregate and --matrix are mutually exclusive")
        payload = aggregate_calibration_evidence(
            [json.loads(path.read_text()) for path in args.aggregate], freeze=args.freeze
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return
    if args.freeze:
        parser.error("--freeze requires --aggregate")
    if args.matrix:
        if args.batch_count <= 0 or not 0 <= args.batch_index < args.batch_count:
            parser.error("batch-count must be positive and batch-index must select an existing batch")
        payload = calibrate_matrix(
            args.devices,
            seed=args.seed,
            repeats=args.repeats,
            profile=args.profile,
            batch_index=args.batch_index,
            batch_count=args.batch_count,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return
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


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """Describe one measured point, never an implicitly supported interval."""

    case_id: str
    young_modulus: float
    poisson_ratio: float
    dt: float
    coordinate_scale: float
    world_offset: float
    active_contact: bool
    seed: int
    direction_index: int = 0
    mesh_level: int = 0
    boundary_mode: str = "all_dynamic"
    contact_stiffness: float = 2e5
    requested_contact_cardinality: int = -1
    coordinate_role: str = "supported"
    coordinate_pattern: str = "diagonal_positive"

    def __post_init__(self):
        values = (self.young_modulus, self.dt, self.coordinate_scale, self.contact_stiffness)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("Material, dt and coordinate scale must be finite and positive")
        if not -1 < self.poisson_ratio < 0.5 or not math.isfinite(self.world_offset):
            raise ValueError("Poisson ratio or world offset is invalid")
        if self.direction_index not in (0, 1, 2) or self.mesh_level not in (0, 1, 2):
            raise ValueError("Direction and mesh refinement indices must be 0, 1 or 2")
        if self.boundary_mode not in ("all_dynamic", "fixed_opposite"):
            raise ValueError("Unknown boundary mode")
        if self.requested_contact_cardinality not in (-1, 0, 3, 12, 48):
            raise ValueError("Unsupported requested contact cardinality")
        if self.coordinate_role not in ("supported", "coordinate_sweep", "outside_control"):
            raise ValueError("Unknown coordinate role")
        if self.coordinate_pattern not in ("positive_axis", "negative_axis", "mixed", "diagonal_positive"):
            raise ValueError("Unknown coordinate pattern")

    @property
    def tet_count(self):
        return 8**self.mesh_level


def calibration_cases(*, seed):
    """Select 24 cross-product points and four central/translated controls."""
    parameters = [
        (young, poisson, dt, scale, 0.0, active)
        for young, poisson in ((1e3, 0.2), (1e4, 0.3), (1e5, 0.45))
        for dt in (0.001, 0.01)
        for scale in (0.5, 2.0)
        for active in (False, True)
    ]
    parameters += [(1e4, 0.3, 0.001, 1.0, offset, active) for offset in (0.0, 10.0) for active in (False, True)]
    return [CalibrationCase(f"case_{index:02d}", *values, seed) for index, values in enumerate(parameters)]


_FREEZE_SEEDS = (20260905, 20260917, 20260929)
_COORDINATE_SWEEP_MAGNITUDES = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
_OUTSIDE_WORLD_OFFSET = 10.0


def freeze_calibration_cases():
    """Return the finite C2/C3 freeze design; values are points, not intervals.

    The four core cases are a predeclared minimal coverage design rather than a
    Cartesian product.  Coordinate controls form a separate ordered sweep so
    that a 10 m failure cannot be averaged into, or skipped when claiming, a
    wider range.
    """
    cases = []
    core = (
        # seed, direction, level, boundary, stiffness, active, E, nu, dt, scale
        (_FREEZE_SEEDS[0], 0, 0, "all_dynamic", 2e5, True, 1e3, 0.2, 0.001, 0.5),
        (_FREEZE_SEEDS[1], 1, 1, "fixed_opposite", 1e7, True, 1e4, 0.3, 0.01, 1.0),
        (_FREEZE_SEEDS[2], 2, 2, "all_dynamic", 2e5, True, 1e5, 0.45, 0.001, 2.0),
        (_FREEZE_SEEDS[0], 2, 2, "fixed_opposite", 1e7, False, 1e5, 0.45, 0.01, 0.5),
    )
    for seed, direction, level, boundary, stiffness, active, young, poisson, dt, scale in core:
        cardinality = 3 * 4**level if active else 0
        cases.append(
            CalibrationCase(
                case_id=(f"core_s{seed}_d{direction}_l{level}_{boundary}_k{int(stiffness)}_c{cardinality}"),
                young_modulus=young,
                poisson_ratio=poisson,
                dt=dt,
                coordinate_scale=scale,
                world_offset=0.0,
                active_contact=active,
                seed=seed,
                direction_index=direction,
                mesh_level=level,
                boundary_mode=boundary,
                contact_stiffness=stiffness,
                requested_contact_cardinality=cardinality,
            )
        )
    for magnitude in _COORDINATE_SWEEP_MAGNITUDES:
        patterns = ("mixed",) if magnitude == 0 else ("positive_axis", "negative_axis", "mixed")
        for pattern in patterns:
            role = "outside_control" if magnitude == _OUTSIDE_WORLD_OFFSET else "coordinate_sweep"
            cases.append(
                CalibrationCase(
                    case_id=f"coordinate_{magnitude:g}_{pattern}",
                    young_modulus=1e4,
                    poisson_ratio=0.3,
                    dt=0.001,
                    coordinate_scale=1.0,
                    world_offset=magnitude,
                    active_contact=True,
                    seed=_FREEZE_SEEDS[0],
                    direction_index=0,
                    mesh_level=0,
                    boundary_mode="all_dynamic",
                    contact_stiffness=2e5,
                    requested_contact_cardinality=3,
                    coordinate_role=role,
                    coordinate_pattern=pattern,
                )
            )
    return cases


class CalibrationFreezeError(ValueError):
    """Raised when an artifact violates the fail-closed freeze contract."""


def _device_class(name):
    name = str(name)
    if name == "cpu":
        return "cpu"
    if name == "cuda" or name.startswith("cuda:"):
        return "cuda"
    raise CalibrationFreezeError(f"Unsupported calibration device: {name}")


def _case_key(case):
    return case.case_id


def _record_failure(record, case):
    if record.get("status") != "RAW_MEASUREMENT":
        return "record is not raw measurement"
    if len(record.get("fixture_sha256", "")) != 64:
        return "missing fixture identity"
    observed = record.get("observed", {})
    if observed.get("tet_count") != case.tet_count:
        return "observed tet count differs from declaration"
    if observed.get("active_sample_count") != case.requested_contact_cardinality:
        return "observed contact cardinality differs from declaration"
    gates = record.get("gates", {})
    if case.coordinate_role in ("coordinate_sweep", "outside_control"):
        required = ("finite_difference", "reduction", "linear")
        if any(gates.get(name) not in ("PASS", "FAIL") for name in required):
            return "coordinate sweep is missing a measured gate result"
        if record.get("coordinate_support_status") not in ("PASS", "OUTSIDE_MEASURED_GATE"):
            return "coordinate sweep is missing its measured support result"
        return None
    required = ("finite_difference", "reduction", "linear")
    if any(gates.get(name) != "PASS" for name in required):
        return "one or more required raw gates failed"
    return None


def _aggregate_parameters(records, max_coordinate):
    result = {}
    for device in ("cpu", "cuda"):
        supported = [
            record
            for record in records
            if record["device_class"] == device
            and (
                record["case"].coordinate_role == "supported"
                or (
                    record["case"].coordinate_role == "coordinate_sweep"
                    and max_coordinate is not None
                    and record["case"].world_offset <= max_coordinate
                    and record["raw"].get("coordinate_support_status") == "PASS"
                )
            )
        ]
        if not supported:
            continue
        draft = [record["raw"].get("draft_parameters", {}) for record in supported]

        def maxima(name, rows=draft):
            values = [row.get(name) for row in rows]
            if not values or any(value is None or len(value) != 3 for value in values):
                return None
            return np.max(np.asarray(values, dtype=np.float64), axis=0).tolist()

        force_floors = [
            record["raw"].get("force_noise", {}).get("recommended_force_detection_floor") for record in supported
        ]
        residual, merit, step = (maxima(name) for name in ("residual_floors", "merit_absolute", "small_scaled_step"))
        solver_internal_config = {
            "epsilon_d": 1e-6,
            "residual_floor_global": None if residual is None else residual[0],
            "residual_floor_q": None if residual is None else residual[1],
            "residual_floor_x": None if residual is None else residual[2],
            "merit_noise": max((row.get("merit_noise", 0.0) for row in draft), default=0.0),
            "merit_absolute_global": None if merit is None else merit[0],
            "merit_absolute_q": None if merit is None else merit[1],
            "merit_absolute_x": None if merit is None else merit[2],
            "merit_relative_global": 1e-4,
            "merit_relative_q": 1e-4,
            "merit_relative_x": 1e-4,
            "step_tolerance_global": None if step is None else step[0],
            "step_tolerance_q": None if step is None else step[1],
            "step_tolerance_x": None if step is None else step[2],
            "det_f_guard": 0.2,
            "regularization_values": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
        }
        result[device] = {
            "solver_internal_config": solver_internal_config,
            "acceptance": {
                "force_detection_floor_n": (
                    max(force_floors) if force_floors and all(value is not None for value in force_floors) else None
                )
            },
        }
    return result


def _coordinate_envelope(records):
    by_key = {
        (record["device_class"], record["case"].world_offset, record["case"].coordinate_pattern): record["raw"]
        for record in records
        if record["case"].coordinate_role == "coordinate_sweep"
    }
    passing_prefix = []
    first_failed = None
    passed_after_failure = []
    for magnitude in _COORDINATE_SWEEP_MAGNITUDES[:-1]:
        patterns = ("mixed",) if magnitude == 0 else ("positive_axis", "negative_axis", "mixed")
        rows = [by_key.get((device, magnitude, pattern)) for device in ("cpu", "cuda") for pattern in patterns]
        passed = bool(rows) and all(
            row is not None
            and row.get("coordinate_support_status") == "PASS"
            and all(row.get("gates", {}).get(name) == "PASS" for name in ("finite_difference", "reduction", "linear"))
            for row in rows
        )
        if first_failed is None and passed:
            passing_prefix.append(magnitude)
        elif first_failed is None:
            first_failed = magnitude
        elif passed:
            passed_after_failure.append(magnitude)
    return {
        "measured_magnitudes_m": list(_COORDINATE_SWEEP_MAGNITUDES),
        "required_patterns": ["positive_axis", "negative_axis", "mixed"],
        "max_supported_abs_coordinate_m": max(passing_prefix, default=None),
        "first_failed_magnitude_m": first_failed,
        "passing_points_after_first_failure_not_used": passed_after_failure,
        "minimum_freeze_envelope_m": 0.1,
        "method": "CPU/CUDA common contiguous passing prefix; no recovery after first failed magnitude",
    }


def _portable_config(per_device):
    if set(per_device) != {"cpu", "cuda"}:
        return None, None
    configs = [per_device[device]["solver_internal_config"] for device in ("cpu", "cuda")]
    fixed = ("epsilon_d", "det_f_guard", "regularization_values")
    if any(config[name] != configs[0][name] for config in configs[1:] for name in fixed):
        raise CalibrationFreezeError("Device candidates disagree on fixed solver policy")
    portable = {}
    for name in configs[0]:
        values = [config[name] for config in configs]
        if name in fixed:
            portable[name] = values[0]
        else:
            portable[name] = None if any(value is None for value in values) else max(values)
    force = [per_device[device]["acceptance"]["force_detection_floor_n"] for device in ("cpu", "cuda")]
    acceptance = {"force_detection_floor_n": None if any(value is None for value in force) else max(force)}
    return portable, acceptance


def aggregate_calibration_evidence(artifacts, *, freeze=False):
    """Aggregate raw batches and optionally freeze the calibration sub-state.

    This validates evidence completeness and identity.  It deliberately leaves
    ``v01_status`` as DRAFT because reference/E2E acceptance is a separate gate.
    """
    expected_cases = {_case_key(case): case for case in freeze_calibration_cases()}
    expected_keys = {(device, key) for device in ("cpu", "cuda") for key in expected_cases}
    sources, records, seen, trajectory = set(), [], set(), set()
    for artifact in artifacts:
        kind = artifact.get("artifact_kind")
        if kind not in ("raw_measurement", "trajectory_evidence") or artifact.get("calibration_schema_version") != 2:
            raise CalibrationFreezeError("Only version-2 raw or trajectory evidence artifacts may be aggregated")
        if artifact.get("profile") != "freeze" or artifact.get("git_dirty") is not False:
            raise CalibrationFreezeError("Freeze evidence must use the freeze profile from a clean tree")
        source = artifact.get("source_set_sha256", "")
        if len(source) != 64:
            raise CalibrationFreezeError("Missing source-set identity")
        sources.add(source)
        if kind == "trajectory_evidence":
            for row in artifact.get("evidence", []):
                device = _device_class(row.get("device", artifact.get("device", "")))
                role = row.get("role")
                if role not in ("normal_loading_1000", "c4_supported_motion"):
                    raise CalibrationFreezeError(f"Unexpected trajectory role: {(device, role)}")
                key = (device, role)
                if key in trajectory:
                    raise CalibrationFreezeError(f"Duplicate trajectory role: {key}")
                if (
                    row.get("status") != "PASS"
                    or len(row.get("fixture_sha256", "")) != 64
                    or row.get("finite_state") is not True
                    or row.get("convergence_gate") != "PASS"
                ):
                    raise CalibrationFreezeError(f"Trajectory gate failed: {key}")
                if role == "normal_loading_1000" and (
                    set(row.get("representative_states", ())) != {"free", "onset", "loading", "peak", "settled"}
                    or row.get("candidate_vs_strict_baseline") != "PASS"
                    or row.get("false_convergence_count") != 0
                ):
                    raise CalibrationFreezeError(f"Normal-loading evidence is incomplete: {key}")
                if role == "c4_supported_motion" and len(row.get("support_artifact_sha256", "")) != 64:
                    raise CalibrationFreezeError(f"C4 support identity is missing: {key}")
                trajectory.add(key)
            continue
        for raw in artifact.get("evidence", []):
            try:
                case = CalibrationCase(**raw["parameters"])
                device = _device_class(raw.get("device", artifact.get("device", "")))
            except (KeyError, TypeError, ValueError) as error:
                raise CalibrationFreezeError(f"Invalid raw record: {error}") from error
            key = (device, _case_key(case))
            if key not in expected_keys:
                raise CalibrationFreezeError(f"Unexpected freeze case: {key}")
            if case != expected_cases[case.case_id]:
                raise CalibrationFreezeError(f"Freeze case parameters differ from the declared design: {key}")
            if key in seen:
                raise CalibrationFreezeError(f"Duplicate freeze case: {key}")
            seen.add(key)
            records.append({"device_class": device, "case": case, "raw": raw})
    if len(sources) > 1:
        raise CalibrationFreezeError("Raw batches were measured from different source sets")
    failures = []
    for record in records:
        reason = _record_failure(record["raw"], record["case"])
        if reason:
            failures.append({"device": record["device_class"], "case_id": record["case"].case_id, "reason": reason})
    missing = sorted(f"{device}:{case}" for device, case in expected_keys - seen)
    expected_trajectory = {
        (device, role) for device in ("cpu", "cuda") for role in ("normal_loading_1000", "c4_supported_motion")
    }
    missing_trajectory = sorted(f"{device}:{case}" for device, case in expected_trajectory - trajectory)
    coordinate = _coordinate_envelope(records)
    coordinate_ready = (
        coordinate["max_supported_abs_coordinate_m"] is not None
        and coordinate["max_supported_abs_coordinate_m"] >= coordinate["minimum_freeze_envelope_m"]
    )
    per_device = _aggregate_parameters(records, coordinate["max_supported_abs_coordinate_m"])
    portable, portable_acceptance = _portable_config(per_device)
    ready = not missing and not missing_trajectory and not failures and len(sources) == 1 and coordinate_ready
    if freeze and not ready:
        raise CalibrationFreezeError("Calibration evidence is incomplete or contains failed gates")
    return {
        "artifact_kind": "calibration_freeze",
        "calibration_schema_version": 2,
        "calibration_status": "FROZEN" if freeze else "CANDIDATE" if ready else "INCOMPLETE",
        "v01_status": "DRAFT",
        "source_set_sha256": next(iter(sources), None),
        "coordinate_envelope": coordinate,
        "outside_coordinate_controls_m": [_OUTSIDE_WORLD_OFFSET],
        "expected_case_count": len(expected_keys),
        "observed_case_count": len(seen),
        "missing_case_keys": missing,
        "missing_trajectory_keys": missing_trajectory,
        "failed_records": failures,
        "candidate_private_config": per_device,
        "portable_solver_internal_config": portable,
        "portable_acceptance": portable_acceptance,
    }


def _hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _matrix_fixture(device, case):
    scale, magnitude = case.coordinate_scale, case.world_offset
    freeze_profile = case.requested_contact_cardinality >= 0
    if case.coordinate_pattern == "positive_axis":
        offset = np.array((magnitude, 0.0, 0.0))
    elif case.coordinate_pattern == "negative_axis":
        offset = np.array((-magnitude, 0.0, 0.0))
    elif case.coordinate_pattern == "mixed":
        offset = np.array((magnitude, -0.5 * magnitude, 0.25 * magnitude))
    else:
        offset = np.full(3, magnitude)
    builder = newton.ModelBuilder(gravity=(0, 0, -9.81), up_axis=newton.Axis.Z)
    passive = {
        "armature": 0.0,
        "damping": 0.0,
        "friction": 0.0,
        "limit_ke": 0.0,
        "limit_kd": 0.0,
        "target_ke": 0.0,
        "target_kd": 0.0,
        "actuator_mode": newton.JointTargetMode.NONE,
    }
    bodies = [
        builder.add_link(
            mass=mass,
            com=(0.01 * scale, 0, 0),
            inertia=wp.mat33(
                0.01 * mass * scale**2, 0, 0, 0, 0.012 * mass * scale**2, 0, 0, 0, 0.014 * mass * scale**2
            ),
        )
        for mass in (1.0, 0.5)
    ]
    joints = [
        builder.add_joint_revolute(
            -1,
            bodies[0],
            axis=newton.Axis.Y,
            parent_xform=wp.transform(offset, wp.quat_identity()),
            **passive,
        ),
        builder.add_joint_prismatic(bodies[0], bodies[1], axis=newton.Axis.Z, **passive),
    ]
    builder.add_articulation(joints)
    builder.joint_q[:] = [0.0, 0.0] if freeze_profile else [0.02, 0.001 * scale]
    builder.joint_qd[:] = [0.1, -0.02 * scale]
    margin = 0.006 * scale if case.requested_contact_cardinality < 0 else 0.0
    builder.add_shape_plane(body=bodies[1], width=0.0, length=0.0, cfg=builder.ShapeConfig(margin=margin))
    if freeze_profile:
        points, tets = refined_tetrahedron(case.mesh_level)
    else:
        points = np.asarray(((0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02)))
        tets = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    mu = case.young_modulus / (2 * (1 + case.poisson_ratio))
    lam = case.young_modulus * case.poisson_ratio / ((1 + case.poisson_ratio) * (1 - 2 * case.poisson_ratio))
    height = (
        (0.0005 if case.active_contact else 0.1) if freeze_profile else (0.002 if case.active_contact else 0.1)
    ) * scale
    builder.add_soft_mesh(
        pos=(offset[0], offset[1], offset[2] + height),
        rot=wp.quat_identity(),
        scale=scale,
        vel=(0, 0, 0),
        vertices=points.tolist(),
        indices=tets.ravel().tolist(),
        density=1000.0,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
        particle_radius=0.001 * scale,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    if case.boundary_mode == "fixed_opposite":
        fixed = np.isclose(points.sum(axis=1), 0.04, rtol=0, atol=1e-9)
        for index in np.flatnonzero(fixed):
            builder.particle_mass[index] = 0.0
    model = builder.finalize(device=device)
    state = model.state()
    deformation = np.array([[1.06, 0.02, -0.01], [0.01, 0.96, 0.03], [0.02, -0.01, 1.03]])
    rng = np.random.default_rng(case.seed + 1009 * case.direction_index)
    if freeze_profile:
        deformation[2, :2] = 0.0
        deformation[:2] += rng.normal(scale=0.002, size=(2, 3))
    else:
        deformation += rng.normal(scale=0.002, size=(3, 3))
    rest = state.particle_q.numpy().astype(np.float64)
    state.particle_q.assign(((rest - rest[0]) @ deformation.T + rest[0]).astype(np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    stored = {
        name: getattr(model, name).numpy().tolist()
        for name in (
            "tet_materials",
            "tet_poses",
            "tet_indices",
            "tri_indices",
            "particle_mass",
            "particle_q",
            "particle_radius",
            "body_mass",
            "body_com",
            "body_inertia",
            "joint_q",
            "joint_qd",
            "joint_parent",
            "joint_child",
            "joint_axis",
            "joint_X_p",
            "joint_X_c",
            "shape_transform",
            "shape_scale",
            "shape_margin",
            "gravity",
        )
    }
    stored["initial_particle_q"] = state.particle_q.numpy().tolist()
    stored["deformation_seed_input"] = deformation.tolist()
    return model, state, stored


@wp.kernel
def _force_resultant(force: wp.array[wp.vec3], moment: wp.array[wp.vec3], result: wp.array[float]):
    record = wp.tid()
    for axis in range(3):
        wp.atomic_add(result, axis, force[record][axis])
        wp.atomic_add(result, axis + 3, moment[record][axis])


@wp.kernel
def _matrix_constitutive(
    f: wp.array[wp.mat33],
    mu: float,
    lam: float,
    energy: wp.array[float],
    stress: wp.array[wp.mat33],
    projected: wp.array[mat99],
    dc: wp.array[mat99],
):
    index = wp.tid()
    e, p, h = _evaluate_stable_neo_hookean(f[index], 1.0, mu, lam)
    energy[index] = e
    stress[index] = p
    projected[index] = h
    dc[index] = _compute_cofactor_derivative(f[index])


@wp.kernel
def _fk_columns(
    jacobian: wp.array3d[float],
    poses: wp.array[wp.transform],
    com: wp.array[wp.vec3],
    point: wp.vec3,
    result: wp.array[wp.vec3],
):
    dof = wp.tid()
    result[dof] = articulation_point_jacobian_column(jacobian, 0, 1, dof, poses[1], com[1], point)


class _MatrixProbe:
    """Compose existing production evaluation paths without advancing a trajectory."""

    def __init__(self, device, case):
        self.case = case
        self.model, state, self.stored = _matrix_fixture(device, case)
        gap = (0.02 if case.requested_contact_cardinality < 0 else 0.0002) * case.coordinate_scale
        pipeline = MonolithicCollisionPipeline(self.model, soft_contact_gap=gap)
        self.solver = SolverMonolithic(
            self.model, collision_pipeline=pipeline, contact_stiffness=case.contact_stiffness
        )
        self.solver._transaction.begin(state, self.model.state(), self.model.control(), case.dt)
        self.solver._metrics = {"nonlinear_iterations": 0, "matrix_assembly_count": 0}
        self.candidate = self.solver._transaction.accepted
        self.force_result = wp.empty(6, dtype=float, device=device)
        self.force_residual = wp.zeros(self.solver._layout.scalar_dof_count, dtype=float, device=device)
        self.force_status = wp.zeros(1, dtype=int, device=device)

    def set_z(self, z):
        self.candidate.z.assign(z)
        self.candidate._recover(self.case.dt)

    def evaluate(self, z=None):
        if z is not None:
            self.set_z(z)
        solver = self.solver
        evaluation = solver._evaluate_current(self.candidate, self.case.dt)
        self.force_residual.zero_()
        self.force_status.zero_()
        # Mode 2 runs the same physical force kernel as final publication without
        # assigning a fictitious trajectory convergence/returned-state status.
        _evaluate_contact(
            2,
            self.model,
            self.candidate.state,
            solver._trial_contacts,
            solver.collision_pipeline,
            solver._articulation,
            solver._contact,
            self.force_residual,
            self.force_status,
        )
        if int(self.force_status.numpy()[0]):
            raise AssertionError("Production contact force evaluation failed")
        self.force_result.zero_()
        contact = solver._contact
        wp.launch(
            _force_resultant,
            contact.final_force_linear.size,
            [contact.final_force_linear, contact.final_force_moment, self.force_result],
            device=self.model.device,
        )
        return self.candidate.residual.numpy().copy(), self.force_result.numpy().copy(), evaluation


def _error(analytic, measured, *, absolute_scale=1e-30):
    analytic, measured = np.asarray(analytic, dtype=np.float64), np.asarray(measured, dtype=np.float64)
    difference = analytic - measured
    return {
        "relative_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(analytic), np.linalg.norm(measured), absolute_scale)
        ),
        "max_absolute_error": float(np.max(np.abs(difference), initial=0)),
    }


def _fd_status(records, names, device):
    gate = 5e-3 if wp.get_device(device).is_cpu else 1e-2
    smooth = [row for row in records if row.get("fixed_active_set", True)]
    minima = {name: min((row[name]["relative_error"] for row in smooth), default=1.0) for name in names}
    plateaus = {
        name: [
            [a["step"], b["step"]]
            for a, b in itertools.pairwise(records)
            if a.get("fixed_active_set", True)
            and b.get("fixed_active_set", True)
            and max(a[name]["relative_error"], b[name]["relative_error"]) <= gate
        ]
        for name in names
    }
    return {
        "status": "PASS" if all(plateaus.values()) else "OUTSIDE_MEASURED_GATE",
        "gate": gate,
        "adjacent_passing_step_pairs": plateaus,
        "best_relative_errors": minima,
        "records": records,
    }


def _material_sweep(device, case):
    arrays = [wp.empty(1, dtype=dtype, device=device) for dtype in (wp.mat33, float, wp.mat33, mat99, mat99)]
    mu = np.float32(case.young_modulus / (2 * (1 + case.poisson_ratio)))
    lam = np.float32(
        case.young_modulus * case.poisson_ratio / ((1 + case.poisson_ratio) * (1 - 2 * case.poisson_ratio))
    )

    def evaluate(f):
        arrays[0].assign(np.asarray([f], dtype=np.float32))
        wp.launch(_matrix_constitutive, 1, [arrays[0], mu, lam, *arrays[1:]], device=device)
        return [array.numpy()[0].copy() for array in arrays[1:]]

    rng = np.random.default_rng(case.seed)
    f = np.array([[1.12, 0.07, -0.03], [0.02, 0.91, 0.08], [0.04, -0.02, 1.06]], dtype=np.float32)
    f += rng.normal(scale=0.003, size=(3, 3)).astype(np.float32)
    _, stress, projected, dc = evaluate(f)
    raw = (
        projected.astype(np.float64)
        + ((float(mu) + float(lam)) * (np.linalg.det(f.astype(np.float64)) - 1) - float(mu)) * dc
    )
    direction = rng.normal(size=(3, 3)).astype(np.float32)
    direction /= np.linalg.norm(direction)
    records = []
    for h in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001):
        plus, minus = evaluate(f + h * direction), evaluate(f - h * direction)
        gradient = (float(plus[0]) - float(minus[0])) / (2 * h)
        derivative = ((plus[1].astype(np.float64) - minus[1]) / (2 * h)).flatten(order="F")
        records.append(
            {
                "step": h,
                "energy_gradient": _error(np.sum(stress * direction), gradient),
                "physical_force_raw_derivative": _error(-(raw @ direction.flatten(order="F")), -derivative),
            }
        )
    result = _fd_status(records, ("energy_gradient", "physical_force_raw_derivative"), device)
    result.update(
        deformation=f.tolist(),
        direction=direction.tolist(),
        seed=case.seed,
        condition_indicator=_finite_diagnostic(np.linalg.cond(projected.astype(np.float64))),
        variable_scale=1.0,
        first_piola_stress=stress.tolist(),
        physical_world_force_or_wrench="N/A: constitutive stress probe",
        force_sign_convention="Nodal physical force is the negative energy gradient",
    )
    return result


def _point(probe, local):
    pose = probe.candidate.state.body_q.numpy()[1]
    return np.asarray(wp.transform_point(wp.transform(*pose), wp.vec3(local)), dtype=np.float64)


def _tet_nodal_sweep(probe):
    """Differentiate the production elastic energy/force at actual world coordinates."""
    model, case, device = probe.model, probe.case, probe.model.device
    positions = probe.candidate.state.particle_q.numpy().copy()
    particle_count, tet_count = len(positions), len(model.tet_indices)
    dynamic_ids = probe.solver._layout.dynamic_particle_ids.numpy()
    dynamic_count = len(dynamic_ids)
    x = wp.empty(particle_count, dtype=wp.vec3, device=device)
    residual = wp.empty(dynamic_count, dtype=wp.vec3, device=device)
    energy = wp.empty(tet_count, dtype=float, device=device)
    minimum = wp.empty(1, dtype=float, device=device)
    flags = wp.empty(1, dtype=int, device=device)
    mapping = probe.solver._layout.particle_to_dynamic

    def evaluate(values):
        x.assign(values)
        residual.zero_()
        minimum.fill_(1e30)
        flags.zero_()
        wp.launch(
            _evaluate_elastic_residual,
            tet_count,
            [
                1e-6,
                x,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                mapping,
                residual,
                energy,
                minimum,
                flags,
            ],
            device=device,
        )
        if flags.numpy()[0]:
            raise AssertionError("Elastic FD probe left the valid determinant domain")
        return float(np.sum(energy.numpy().astype(np.float64))), residual.numpy().astype(np.float64)

    _, analytic = evaluate(positions)
    indices = model.tet_indices.numpy()
    rest_inverse = model.tet_poses.numpy().astype(np.float64)
    f = np.asarray(
        [
            (positions[tet[1:]] - positions[tet[0]]).astype(np.float64).T @ rest
            for tet, rest in zip(indices, rest_inverse, strict=True)
        ]
    )
    arrays = [
        wp.array(f, dtype=wp.mat33, device=device),
        wp.empty(tet_count, dtype=float, device=device),
        wp.empty(tet_count, dtype=wp.mat33, device=device),
        wp.empty(tet_count, dtype=mat99, device=device),
        wp.empty(tet_count, dtype=mat99, device=device),
    ]
    mu, lam, _ = model.tet_materials.numpy()[0]
    wp.launch(_matrix_constitutive, tet_count, [arrays[0], mu, lam, *arrays[1:]], device=device)
    raw = arrays[3].numpy().astype(np.float64)
    raw += ((float(mu) + float(lam)) * (np.linalg.det(f) - 1) - float(mu))[:, None, None] * arrays[4].numpy().astype(
        np.float64
    )
    direction = np.random.default_rng(case.seed + 1009 * case.direction_index).normal(size=(particle_count, 3))
    fixed = np.ones(particle_count, dtype=bool)
    fixed[dynamic_ids] = False
    direction[fixed] = 0.0
    direction /= np.linalg.norm(direction)
    derivative = np.zeros((dynamic_count, 3))
    for tet_index, (tet, rest) in enumerate(zip(indices, rest_inverse, strict=True)):
        gradients = np.vstack((-rest.sum(axis=0), rest))
        volume = 1 / (6 * np.linalg.det(rest))
        df = (direction[tet[1:]] - direction[tet[0]]).T @ rest
        stress_derivative = (raw[tet_index] @ df.flatten(order="F")).reshape((3, 3), order="F")
        nodal = volume * (stress_derivative @ gradients.T).T
        for local, particle in enumerate(tet):
            dynamic = int(probe.solver._layout.particle_to_dynamic.numpy()[particle])
            if dynamic >= 0:
                derivative[dynamic] += nodal[local]
    records = []
    for step in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001):
        h = step * 0.01 * case.coordinate_scale
        ep, rp = evaluate((positions + h * direction).astype(np.float32))
        em, rm = evaluate((positions - h * direction).astype(np.float32))
        records.append(
            {
                "step": step,
                "step_metres": h,
                "energy_gradient": _error(np.sum(analytic * direction[dynamic_ids]), (ep - em) / (2 * h)),
                "physical_force_raw_derivative": _error(-derivative, -(rp - rm) / (2 * h)),
            }
        )
    result = _fd_status(records, ("energy_gradient", "physical_force_raw_derivative"), device)
    result.update(
        variable_scale_metres=0.01 * case.coordinate_scale,
        seed=case.seed,
        direction=direction.tolist(),
        condition_indicator=_finite_diagnostic(np.max(np.linalg.cond(raw))),
        residual_contribution=analytic.tolist(),
        physical_world_force_or_wrench=(-analytic).tolist(),
        generalized_physical_force=(-analytic).tolist(),
        projection=("Identity on dynamic nodal world translations; fixed-node direction entries are exactly zero"),
        sign_error_q=0.0,
        sign_error_x=0.0,
        constitutive_probe=_material_sweep(device, case),
    )
    return result


def _fk_sweep(probe):
    solver, case, device = probe.solver, probe.case, probe.model.device
    base = probe.candidate.z.numpy().copy()
    local = np.array([0.013, 0.007, 0.003]) * case.coordinate_scale
    columns = wp.empty(2, dtype=wp.vec3, device=device)
    wp.launch(
        _fk_columns,
        2,
        [
            solver._articulation.scratch.J,
            probe.candidate.state.body_q,
            probe.model.body_com,
            wp.vec3(_point(probe, local)),
            columns,
        ],
        device=device,
    )
    analytic = columns.numpy().astype(np.float64).T
    records = []
    for h in (0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001):
        fd = np.zeros((3, 2))
        for dof, unit in enumerate((1.0, case.coordinate_scale)):
            delta = np.zeros_like(base)
            delta[dof] = h * unit
            probe.set_z(base + delta)
            plus = _point(probe, local)
            probe.set_z(base - delta)
            minus = _point(probe, local)
            fd[:, dof] = (plus - minus) / (2 * h * unit)
        records.append(
            {"step": h, "q_radians": _error(analytic[:, 0], fd[:, 0]), "q_metres": _error(analytic[:, 1], fd[:, 1])}
        )
    probe.set_z(base)
    result = _fd_status(records, ("q_radians", "q_metres"), device)
    result.update(
        condition_indicator=_finite_diagnostic(np.linalg.cond(analytic)),
        point_local=local.tolist(),
        variable_scales=[1.0, case.coordinate_scale],
        analytic=analytic.tolist(),
    )
    return result


def _frozen_contact_data(probe):
    contacts, model = probe.solver._trial_contacts, probe.model
    count = int(contacts.soft_contact_count.numpy()[0])
    ids = contacts.soft_contact_indices.numpy()[:count].copy()
    rest = model.particle_q.numpy().astype(np.float64)
    areas = np.linalg.norm(np.cross(rest[ids[:, 1]] - rest[ids[:, 0]], rest[ids[:, 2]] - rest[ids[:, 0]]), axis=1) * 0.5
    return {
        "ids": ids,
        "bary": contacts.soft_contact_barycentric.numpy()[:count].copy(),
        "normal": contacts.soft_contact_normal.numpy()[:count].copy(),
        "local": contacts.soft_contact_body_pos.numpy()[:count].copy(),
        "shape": contacts.soft_contact_shape.numpy()[:count].copy(),
        "weight": probe.case.contact_stiffness * areas / 3,
    }


def _frozen_gaps(probe, data):
    state, model = probe.candidate.state, probe.model
    poses, positions = state.body_q.numpy(), state.particle_q.numpy().astype(np.float64)
    bodies, margins = model.shape_body.numpy(), model.shape_margin.numpy()
    gaps = []
    for ids, bary, normal, local, shape in zip(
        data["ids"], data["bary"], data["normal"], data["local"], data["shape"], strict=True
    ):
        rigid = np.asarray(wp.transform_point(wp.transform(*poses[bodies[shape]]), wp.vec3(local)), dtype=np.float64)
        gaps.append(
            normal @ (bary.astype(np.float64) @ positions[ids] - rigid)
            - probe.solver.collision_pipeline.r_soft
            - margins[shape]
        )
    return np.asarray(gaps)


def _contact_sweep(probe):
    solver, device = probe.solver, probe.model.device
    nq, size = solver._layout.q_dof_count, solver._layout.scalar_dof_count
    count = int(solver._linear._factors.count.numpy()[0])
    if not count:
        return {
            "status": "NOT_APPLICABLE",
            "reason": "No active contact; this is not a force-floor measurement",
            "records": [],
        }
    base = probe.candidate.z.numpy().copy()
    data = _frozen_contact_data(probe)
    active_records = solver._contact.factor_contact_record.numpy()[:count].copy()
    gaps = _frozen_gaps(probe, data)
    factors = solver._linear._factors
    analytic = np.zeros((count, size))
    analytic[:, :nq] = factors.gq.numpy()[:count]
    for row in range(count):
        for column, value in zip(factors.gx_columns.numpy()[row], factors.gx_values.numpy()[row], strict=True):
            if column >= 0:
                analytic[row, nq + column] += value
    residual = solver._contact._residual[0].numpy().astype(np.float64)
    records = []
    units = np.r_[np.full(nq, 0.01), np.full(size - nq, 0.001 * probe.case.coordinate_scale)]
    for h in (0.2, 0.06, 0.02, 0.006, 0.002):
        gradient, derivative = np.zeros(size), np.zeros_like(analytic)
        smooth = True
        for dof in range(size):
            delta = np.zeros(size, dtype=np.float32)
            delta[dof] = h * units[dof]
            probe.set_z(base + delta)
            plus = _frozen_gaps(probe, data)
            probe.set_z(base - delta)
            minus = _frozen_gaps(probe, data)
            smooth &= bool(np.array_equal(plus < 0, gaps < 0) and np.array_equal(minus < 0, gaps < 0))
            gradient[dof] = (
                np.sum(0.5 * data["weight"] * np.minimum(plus, 0) ** 2)
                - np.sum(0.5 * data["weight"] * np.minimum(minus, 0) ** 2)
            ) / (2 * h * units[dof])
            derivative[:, dof] = (plus[active_records] - minus[active_records]) / (2 * h * units[dof])
        records.append(
            {
                "step": h,
                "fixed_active_set": smooth,
                "gap_q": _error(analytic[:, :nq], derivative[:, :nq]),
                "gap_x": _error(analytic[:, nq:], derivative[:, nq:]),
                "energy_q": _error(residual[:nq], gradient[:nq]),
                "energy_x": _error(residual[nq:], gradient[nq:]),
            }
        )
    probe.set_z(base)
    result = _fd_status(records, ("gap_q", "gap_x", "energy_q", "energy_x"), device)
    if not any(row["fixed_active_set"] for row in records):
        result["status"] = "OUTSIDE_MEASURED_GATE"
    result.update(
        variable_scales=units.tolist(),
        active_samples=count,
        condition_indicator=_finite_diagnostic(np.linalg.cond(analytic)),
        minimum_absolute_gap=float(np.min(np.abs(gaps))),
    )
    return result


def _project_world_forces(probe):
    """Independently project measured world forces with point J and barycentrics."""
    solver, device = probe.solver, probe.model.device
    data = _frozen_contact_data(probe)
    force = solver._contact.final_force_linear.numpy().astype(np.float64)
    nq, size = solver._layout.q_dof_count, solver._layout.scalar_dof_count
    particle_to_dynamic = solver._layout.particle_to_dynamic.numpy()
    result = np.zeros(size)
    columns = wp.empty(2, dtype=wp.vec3, device=device)
    for record, (ids, bary, local) in enumerate(zip(data["ids"], data["bary"], data["local"], strict=True)):
        wp.launch(
            _fk_columns,
            2,
            [
                solver._articulation.scratch.J,
                probe.candidate.state.body_q,
                probe.model.body_com,
                wp.vec3(_point(probe, local)),
                columns,
            ],
            device=device,
        )
        result[:nq] += columns.numpy().astype(np.float64) @ force[record]
        for particle, weight in zip(ids, bary, strict=True):
            dynamic = int(particle_to_dynamic[particle])
            if dynamic >= 0:
                start = nq + 3 * dynamic
                result[start : start + 3] -= float(weight) * force[record]
    return result


def calibrate_case(device, *, case, repeats, profile="quick"):
    """Measure production reductions, force resolution, cancellation and exact FD."""
    if repeats < 2:
        raise ValueError("At least two repeated measurements are required")
    probe = _MatrixProbe(device, case)
    solver = probe.solver
    nq, size = solver._layout.q_dof_count, solver._layout.scalar_dof_count
    residual, baseline_force, evaluation = probe.evaluate()
    base = probe.candidate.z.numpy().copy()
    scale = solver._linear.scale.numpy().copy()
    frozen_scale = wp.array(scale, dtype=float, device=device)
    merit_samples, norm_errors, force_samples = [], [], []
    for _ in range(repeats):
        _, force, _ = probe.evaluate(base)
        wp.launch(
            _scale_calibration_vector,
            size,
            [frozen_scale, probe.candidate.residual, solver._metric_vector],
            device=device,
        )
        merit = solver._rms(solver._metric_vector, scaled=False)
        merit_samples.append(merit)
        norm_errors.append(np.abs(np.asarray(merit) - _block_rms(solver._metric_vector.numpy())))
        force_samples.append(force)
    samples, forces = np.asarray(merit_samples), np.asarray(force_samples)
    reference_force = np.r_[
        solver._contact.final_force_linear.numpy().astype(np.float64).sum(axis=0),
        solver._contact.final_force_moment.numpy().astype(np.float64).sum(axis=0),
    ]
    quantization = []
    for dof in range(size):
        for direction in (-math.inf, math.inf):
            changed = base.copy()
            changed[dof] = np.nextafter(base[dof], np.float32(direction))
            changed_residual, force, _ = probe.evaluate(changed)
            quantization.append(
                {
                    "axis": dof,
                    "coordinate_unit": "rad" if dof == 0 else "m",
                    "direction": direction > 0,
                    "delta": float(changed[dof] - base[dof]),
                    "scaled_step_rms": _block_rms((changed.astype(np.float64) - base) / scale),
                    "scaled_residual_change_rms": _block_rms(scale * (changed_residual.astype(np.float64) - residual)),
                    "force_resultant_change": float(np.linalg.norm(force[:3].astype(np.float64) - baseline_force[:3])),
                    "active_sample_count": solver._contact._diagnostics(2)["active_sample_count"],
                }
            )
    probe.evaluate(base)
    generation, linear = solver._generation, solver._linear
    config = MonolithicPcgConfig(20, 5, 1e-4, 1e-30, 1e-30, 1e-30, 0.0, 1e-8, 0.0, 20, 0.001)
    cancellation = []
    for regularization in (0.0, 0.001, 1.0):
        linear.set_regularization(regularization, generation=generation)
        matrix = linear.densify_for_test(generation=generation).scaled_matrix
        for zero_block in ("none", "q", "x"):
            rhs = -(scale.astype(np.float64) * residual)
            if zero_block == "q":
                rhs[:nq] = 0
            elif zero_block == "x":
                rhs[nq:] = 0
            rhs = rhs.astype(np.float32)
            solver._rhs_hat.assign(rhs)
            solver._y.assign(np.linalg.solve(matrix, rhs.astype(np.float64)).astype(np.float32))
            errors = []
            for _ in range(repeats):
                linear._true_residual(solver._rhs_hat, solver._y, config)
                if int(linear._status.numpy()[0]):
                    raise AssertionError("Production true residual failed during calibration")
                errors.append(linear._true_norms.numpy()[:3].copy())
            cancellation.append(
                {
                    "lambda": regularization,
                    "zero_rhs_block": zero_block,
                    "true_norm_max": np.max(errors, axis=0).tolist(),
                    "true_norm_peak_to_peak": np.ptp(errors, axis=0).tolist(),
                    "scaled_condition": _finite_diagnostic(np.linalg.cond(matrix)),
                }
            )
    probe.evaluate(base)
    fd = {"tet": _tet_nodal_sweep(probe), "fk": _fk_sweep(probe), "contact": _contact_sweep(probe)}
    probe.evaluate(base)
    contact = solver._contact
    physical = solver._contact.final_force_linear.numpy().astype(np.float64)
    contact_residual = contact._residual[2].numpy().astype(np.float64)
    generalized = contact._projection[2].numpy().astype(np.float64)
    projected_world = _project_world_forces(probe)
    force_spread = float(np.linalg.norm(np.ptp(forces[:, :3], axis=0)))
    force_reduction_error = float(np.linalg.norm(baseline_force[:3] - reference_force[:3]))
    force_ulp = max(row["force_resultant_change"] for row in quantization)
    q_floor = 2e4 * max(row["true_norm_max"][1] for row in cancellation if row["zero_rhs_block"] == "q")
    x_floor = 2e4 * max(row["true_norm_max"][2] for row in cancellation if row["zero_rhs_block"] == "x")
    merit_jump = np.max([row["scaled_residual_change_rms"] for row in quantization], axis=0)
    step_jump = np.max([row["scaled_step_rms"] for row in quantization], axis=0)
    observed_contacts = contact._diagnostics(2)["active_sample_count"]
    fd_gate = "PASS" if all(value["status"] in ("PASS", "NOT_APPLICABLE") for value in fd.values()) else "FAIL"
    cardinality_gate = (
        "PASS"
        if case.requested_contact_cardinality < 0 or observed_contacts == case.requested_contact_cardinality
        else "FAIL"
    )
    reduction_gate = "PASS" if np.isfinite(samples).all() else "FAIL"
    linear_gate = "PASS" if all(np.isfinite(row["true_norm_max"]).all() for row in cancellation) else "FAIL"
    coordinate_support_status = (
        "PASS" if fd_gate == cardinality_gate == reduction_gate == linear_gate == "PASS" else "OUTSIDE_MEASURED_GATE"
    )
    return {
        "status": "RAW_MEASUREMENT" if profile == "freeze" else "DRAFT",
        "device": str(device),
        "parameters": asdict(case),
        "device_name": wp.get_device(device).name,
        "fixture_sha256": _hash_json({"parameters": asdict(case), "stored": probe.stored}),
        "stored_fixture": probe.stored,
        "block_order": ["global", "q", "x"],
        "candidate_z": base.tolist(),
        "min_det_f": evaluation.min_det_f,
        "diagonal": linear.diagonal.numpy().tolist(),
        "scale": scale.tolist(),
        "coordinate_quantization": quantization,
        "reduction_noise": {
            "merit_samples": samples.tolist(),
            "merit_peak_to_peak": np.ptp(samples, axis=0).tolist(),
            "float64_reduction_error_max": np.max(norm_errors, axis=0).tolist(),
            "merit_noise_proposal": float(2 * np.ptp(samples[:, 0])),
        },
        "linear_cancellation": cancellation,
        "finite_difference": fd,
        "force_noise": {
            "active_sample_count": observed_contacts,
            "physical_world_force_or_wrench": baseline_force.tolist(),
            "physical_rigid_sample_forces": physical.tolist(),
            "generalized_physical_force": generalized.tolist(),
            "residual_contribution": contact_residual.tolist(),
            "scaled_projection_error_q": _error(scale[:nq] * projected_world[:nq], scale[:nq] * generalized[:nq]),
            "scaled_projection_error_x": _error(scale[nq:] * projected_world[nq:], scale[nq:] * generalized[nq:]),
            "scaled_sign_error_q": _error(scale[:nq] * contact_residual[:nq], -scale[:nq] * generalized[:nq]),
            "scaled_sign_error_x": _error(scale[nq:] * contact_residual[nq:], -scale[nq:] * generalized[nq:]),
            "resultant_samples": forces.tolist(),
            "repeated_resultant_peak_to_peak": force_spread,
            "float64_sum_reference_error": force_reduction_error,
            "coordinate_ulp_resultant_change_max": force_ulp,
            "recommended_force_detection_floor": 2 * max(force_spread, force_reduction_error, force_ulp),
        },
        "draft_parameters": {
            "residual_floors": [math.hypot(q_floor, x_floor), q_floor, x_floor],
            "merit_absolute": (2 * merit_jump).tolist(),
            "small_scaled_step": (0.5 * step_jump).tolist(),
            "merit_noise": float(2 * np.ptp(samples[:, 0])),
            "linear_tolerance_target": 1e-4,
        },
        "observed": {"tet_count": len(probe.model.tet_indices), "active_sample_count": observed_contacts},
        "gates": {
            "finite_difference": fd_gate,
            "contact_cardinality": cardinality_gate,
            "reduction": reduction_gate,
            "linear": linear_gate,
            "trajectory": "NOT_MEASURED",
        },
        "coordinate_support_status": coordinate_support_status,
        "scope_note": (
            "Discrete component measurement; complete trajectory evidence is deliberately separate and required before freeze"
        ),
    }


@wp.kernel
def _scale_calibration_vector(scale: wp.array[float], values: wp.array[float], out: wp.array[float]):
    i = wp.tid()
    out[i] = scale[i] * values[i]


def calibrate_matrix(devices, *, seed, repeats, profile="quick", batch_index=0, batch_count=1):
    """Capture source identity and refuse mixed-source measurement artifacts."""
    if profile not in ("quick", "freeze"):
        raise ValueError("Unknown calibration profile")
    if batch_count <= 0 or not 0 <= batch_index < batch_count:
        raise ValueError("Invalid batch selection")
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__), *sorted((root / "newton/_src/solvers/monolithic").glob("*.py"))]
    before = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    evidence = []
    cases = calibration_cases(seed=seed) if profile == "quick" else freeze_calibration_cases()
    work = [(device, case) for device in devices for case in cases]
    selected = work[batch_index::batch_count]
    for device, case in selected:
        evidence.append(calibrate_case(device, case=case, repeats=repeats, profile=profile))
        print(
            f"Measured {device} {case.case_id}: {[value['status'] for value in evidence[-1]['finite_difference'].values()]}",
            flush=True,
        )
    after = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    if before != after:
        raise RuntimeError("Calibration sources changed during measurement")
    return {
        "status": "DRAFT" if profile == "quick" else "RAW_MEASUREMENT",
        "artifact_kind": "draft_measurement" if profile == "quick" else "raw_measurement",
        "calibration_schema_version": 2,
        "profile": profile,
        "scope": (
            "28 discrete material/dt/coordinate/contact points per requested device"
            if profile == "quick"
            else "finite C2/C3 freeze design; no continuous interval claim"
        ),
        "excluded": [
            "continuous parameter intervals",
            "larger meshes",
            "non-plane SDFs",
            "trajectory convergence",
            "V0.1 product freeze",
        ],
        "seed": seed,
        "repeats": repeats,
        "platform": platform.platform(),
        "warp_version": wp.__version__,
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)),
        "devices": list(devices),
        "batch_index": batch_index,
        "batch_count": batch_count,
        "selected_case_count": len(selected),
        "total_case_count": len(work),
        "source_sha256": before,
        "source_set_sha256": _hash_json(before),
        "evidence": evidence,
    }


if __name__ == "__main__":
    main()
