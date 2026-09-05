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

_INTERNAL_CONFIG_FIELDS = {
    "epsilon_d",
    "residual_floor_global",
    "residual_floor_q",
    "residual_floor_x",
    "merit_noise",
    "merit_absolute_global",
    "merit_absolute_q",
    "merit_absolute_x",
    "merit_relative_global",
    "merit_relative_q",
    "merit_relative_x",
    "step_tolerance_global",
    "step_tolerance_q",
    "step_tolerance_x",
    "det_f_guard",
    "regularization_values",
}
_PRODUCTION_SOURCE_PREFIX = "newton/_src/solvers/monolithic/"


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
    parser.add_argument("--trajectory-candidate", type=Path, nargs="+", help="Candidate normal-loading run directories")
    parser.add_argument("--trajectory-probe", type=Path, nargs="+", help="Unadjusted calibration candidate runs")
    parser.add_argument(
        "--trajectory-legacy-default-baseline",
        "--trajectory-strict",
        dest="trajectory_baseline",
        type=Path,
        nargs="+",
        help="Legacy-default normal-loading baseline directories",
    )
    parser.add_argument("--trajectory-c4", type=Path, help="Frozen C4 support artifact")
    parser.add_argument("--trajectory-calibration-raw", type=Path, help="Complete raw calibration artifact")
    parser.add_argument("--trajectory-calibration-compact", type=Path, help="Versioned compact calibration artifact")
    parser.add_argument(
        "--freeze", action="store_true", help="Require complete passing evidence and freeze calibration"
    )
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    if not math.isfinite(args.dt) or args.dt <= 0 or args.repeats < 2:
        parser.error("dt must be positive and finite; repeats must be at least two")
    trajectory_arguments = (
        args.trajectory_candidate,
        args.trajectory_probe,
        args.trajectory_baseline,
        args.trajectory_c4,
        args.trajectory_calibration_raw,
        args.trajectory_calibration_compact,
    )
    if any(value is not None for value in trajectory_arguments):
        if args.matrix or args.aggregate or not all(value is not None for value in trajectory_arguments):
            parser.error(
                "trajectory evidence requires probe, candidate, legacy-default baseline, C4 and calibration raw inputs"
            )
        c4_bytes = args.trajectory_c4.read_bytes()
        calibration_raw = json.loads(args.trajectory_calibration_raw.read_text())
        verified = build_trajectory_evidence(
            calibration_raw,
            [_load_normal_run(path) for path in args.trajectory_probe],
            [_load_normal_run(path) for path in args.trajectory_candidate],
            [_load_normal_run(path) for path in args.trajectory_baseline],
            c4_bytes,
            args.trajectory_calibration_compact.read_bytes(),
        )
        payload = (
            aggregate_calibration_evidence([calibration_raw], freeze=True, verified_trajectory=verified)
            if args.freeze
            else verified.audit_artifact
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        return
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
        parser.error("--freeze requires --aggregate or the complete trajectory input set")
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


_VERIFICATION_NONCE = object()


@dataclass(frozen=True, slots=True)
class _VerifiedTrajectoryEvidence:
    """In-process authority returned only after re-reading every upstream artifact."""

    audit_artifact: dict
    seal: tuple[int, str]


def _verified_trajectory_seal(artifact):
    return id(_VERIFICATION_NONCE), _hash_json(artifact)


def _device_class(name):
    name = str(name)
    if name == "cpu":
        return "cpu"
    if name == "cuda" or name.startswith("cuda:"):
        return "cuda"
    raise CalibrationFreezeError(f"Unsupported calibration device: {name}")


def _case_key(case):
    return case.case_id


def _finite_array(value, *, shape=None):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    return (shape is None or array.shape == shape) and bool(np.isfinite(array).all())


def _same_numbers(actual, expected):
    return (
        _finite_array(actual)
        and np.asarray(actual).shape == np.asarray(expected).shape
        and bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-15))
    )


def _record_failure(record, case, device, repeats):
    if record.get("status") != "RAW_MEASUREMENT":
        return "record is not raw measurement"
    if len(record.get("fixture_sha256", "")) != 64:
        return "missing fixture identity"
    stored = record.get("stored_fixture")
    if not isinstance(stored, dict) or record["fixture_sha256"] != _hash_json(
        {"parameters": asdict(case), "stored": stored}
    ):
        return "fixture identity does not match stored parameters"
    if (
        not _finite_array(stored.get("tet_indices"), shape=(case.tet_count, 4))
        or not _finite_array(stored.get("particle_q"))
        or np.asarray(stored["particle_q"]).ndim != 2
        or np.asarray(stored["particle_q"]).shape[1] != 3
        or not _finite_array(stored.get("initial_particle_q"), shape=np.asarray(stored["particle_q"]).shape)
        or not _finite_array(stored.get("joint_X_p"))
        or np.asarray(stored["joint_X_p"]).ndim != 2
        or np.asarray(stored["joint_X_p"]).shape[1] != 7
        or not _finite_array(stored.get("deformation_seed_input"))
        or np.asarray(stored["deformation_seed_input"]).shape[-1] != 3
    ):
        return "stored fixture geometry is incomplete or nonfinite"
    observed = record.get("observed", {})
    if observed.get("tet_count") != case.tet_count:
        return "observed tet count differs from declaration"
    if observed.get("active_sample_count") != case.requested_contact_cardinality:
        return "observed contact cardinality differs from declaration"
    finite_difference = record.get("finite_difference", {})
    definitions = {
        "tet": ("energy_gradient", "physical_force_raw_derivative"),
        "fk": ("q_radians", "q_metres"),
        "contact": ("gap_q", "gap_x", "energy_q", "energy_x"),
    }
    measured_fd = {}
    for name, metrics in definitions.items():
        block = finite_difference.get(name, {})
        if name == "contact" and case.requested_contact_cardinality == 0:
            if block.get("status") != "NOT_APPLICABLE" or block.get("records") != []:
                return "inactive contact FD is not a valid NOT_APPLICABLE record"
            measured_fd[name] = "NOT_APPLICABLE"
            continue
        records = block.get("records")
        if not isinstance(records, list) or len(records) < 5:
            return f"{name} FD records are incomplete"
        recomputed = _fd_status(records, metrics, device)["status"]
        if recomputed != block.get("status"):
            return f"{name} FD status does not match raw records"
        measured_fd[name] = recomputed
    fd_gate = "PASS" if all(value in ("PASS", "NOT_APPLICABLE") for value in measured_fd.values()) else "FAIL"
    reduction = record.get("reduction_noise", {})
    samples = np.asarray(reduction.get("merit_samples", ()), dtype=np.float64)
    reduction_gate = "PASS" if samples.shape == (repeats, 3) and np.isfinite(samples).all() else "FAIL"
    cancellation = record.get("linear_cancellation", ())
    expected_linear = {(value, block) for value in (0.0, 0.001, 1.0) for block in ("none", "q", "x")}
    observed_linear = {(row.get("lambda"), row.get("zero_rhs_block")) for row in cancellation}
    linear_gate = (
        "PASS"
        if len(cancellation) == 9
        and observed_linear == expected_linear
        and all(
            _finite_array(row.get("true_norm_max"), shape=(3,))
            and _finite_array(row.get("true_norm_peak_to_peak"), shape=(3,))
            for row in cancellation
        )
        else "FAIL"
    )
    quantization = record.get("coordinate_quantization", ())
    if not quantization or not all(
        _finite_array(row.get("scaled_step_rms"), shape=(3,))
        and _finite_array(row.get("scaled_residual_change_rms"), shape=(3,))
        and _finite_array([row.get("force_resultant_change")], shape=(1,))
        for row in quantization
    ):
        return "coordinate quantization records are incomplete or nonfinite"
    draft = record.get("draft_parameters", {})
    if not all(
        _finite_array(draft.get(name), shape=(3,))
        for name in ("residual_floors", "merit_absolute", "small_scaled_step")
    ) or not _finite_array([draft.get("merit_noise"), draft.get("linear_tolerance_target")], shape=(2,)):
        return "candidate parameter inputs are missing or nonfinite"
    force_floor = record.get("force_noise", {}).get("recommended_force_detection_floor")
    if not _finite_array([force_floor], shape=(1,)) or force_floor < 0:
        return "force detection floor is missing or invalid"
    q_floor = 2e4 * max(row["true_norm_max"][1] for row in cancellation if row["zero_rhs_block"] == "q")
    x_floor = 2e4 * max(row["true_norm_max"][2] for row in cancellation if row["zero_rhs_block"] == "x")
    residual = [math.hypot(q_floor, x_floor), q_floor, x_floor]
    merit = 2 * np.max([row["scaled_residual_change_rms"] for row in quantization], axis=0)
    step = 0.5 * np.max([row["scaled_step_rms"] for row in quantization], axis=0)
    merit_noise = 2 * np.ptp(samples[:, 0])
    force = record.get("force_noise", {})
    resultants = np.asarray(force.get("resultant_samples", ()), dtype=np.float64)
    rigid_forces = np.asarray(force.get("physical_rigid_sample_forces", ()), dtype=np.float64)
    baseline = np.asarray(force.get("physical_world_force_or_wrench", ()), dtype=np.float64)
    if (
        resultants.shape != (repeats, 6)
        or rigid_forces.ndim != 2
        or rigid_forces.shape[1] != 3
        or baseline.shape != (6,)
        or not np.isfinite(resultants).all()
        or not np.isfinite(rigid_forces).all()
        or not np.isfinite(baseline).all()
    ):
        return "force floor raw arrays are incomplete or nonfinite"
    force_spread = np.linalg.norm(np.ptp(resultants[:, :3], axis=0))
    force_reduction = np.linalg.norm(baseline[:3] - np.sum(rigid_forces, axis=0))
    force_ulp = max(row["force_resultant_change"] for row in quantization)
    expected_force_floor = 2 * max(force_spread, force_reduction, force_ulp)
    if (
        not _same_numbers(draft["residual_floors"], residual)
        or not _same_numbers(draft["merit_absolute"], merit)
        or not _same_numbers(draft["small_scaled_step"], step)
        or not _same_numbers(draft["merit_noise"], merit_noise)
        or not _same_numbers(draft["linear_tolerance_target"], 1e-4)
        or not _same_numbers(force_floor, expected_force_floor)
    ):
        return "candidate parameters do not match recomputed raw measurements"
    gates = record.get("gates", {})
    derived = {
        "finite_difference": fd_gate,
        "contact_cardinality": (
            "PASS" if observed.get("active_sample_count") == case.requested_contact_cardinality else "FAIL"
        ),
        "reduction": reduction_gate,
        "linear": linear_gate,
    }
    if any(gates.get(name) != value for name, value in derived.items()):
        return "stored component gate differs from recomputed raw evidence"
    coordinate_status = "PASS" if all(value == "PASS" for value in derived.values()) else "OUTSIDE_MEASURED_GATE"
    if (
        case.coordinate_role in ("coordinate_sweep", "outside_control")
        and record.get("coordinate_support_status") != coordinate_status
    ):
        return "coordinate support status differs from recomputed evidence"
    if case.coordinate_role in ("coordinate_sweep", "outside_control"):
        required = ("finite_difference", "reduction", "linear")
        if any(gates.get(name) not in ("PASS", "FAIL") for name in required):
            return "coordinate sweep is missing a measured gate result"
        if record.get("coordinate_support_status") not in ("PASS", "OUTSIDE_MEASURED_GATE"):
            return "coordinate sweep is missing its measured support result"
        return None
    if any(derived[name] != "PASS" for name in ("finite_difference", "reduction", "linear")):
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
    passing_rows = [
        record
        for record in records
        if record["case"].coordinate_role == "coordinate_sweep"
        and passing_prefix
        and record["case"].world_offset <= passing_prefix[-1]
        and record["raw"].get("coordinate_support_status") == "PASS"
    ]
    coordinate_values = []
    for record in passing_rows:
        stored = record["raw"].get("stored_fixture", {})
        coordinate_values.extend(np.asarray(stored.get("particle_q", ()), dtype=np.float64).reshape(-1).tolist())
        coordinate_values.extend(
            np.asarray(stored.get("initial_particle_q", ()), dtype=np.float64).reshape(-1).tolist()
        )
        for name in ("joint_X_p", "joint_X_c", "shape_transform"):
            transforms = np.asarray(stored.get(name, ()), dtype=np.float64)
            if transforms.ndim == 2 and transforms.shape[1] >= 3:
                coordinate_values.extend(transforms[:, :3].reshape(-1).tolist())
    maximum_world_coordinate = max((abs(value) for value in coordinate_values), default=None)
    return {
        "measured_magnitudes_m": list(_COORDINATE_SWEEP_MAGNITUDES),
        "required_patterns": ["positive_axis", "negative_axis", "mixed"],
        "maximum_passing_translation_offset_m": max(passing_prefix, default=None),
        "maximum_observed_absolute_world_coordinate_m": maximum_world_coordinate,
        "first_failing_translation_offset_m": first_failed,
        "passing_points_after_first_failure_not_used": passed_after_failure,
        "minimum_required_translation_offset_m": 0.1,
        "scope": "exact stored one-tet calibration fixture translations only; not a general world-coordinate range",
        "method": "CPU/CUDA common contiguous measured translation-offset prefix; no recovery after first failure",
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


def _config_failure(config, acceptance):
    if config is None or acceptance is None:
        return "CPU/CUDA portable config is incomplete"
    if set(config) != _INTERNAL_CONFIG_FIELDS:
        return "portable config does not contain exactly the solver internal fields"
    if set(acceptance) != {"force_detection_floor_n"}:
        return "portable acceptance has missing or unknown fields"
    fixed = {
        "epsilon_d": 1e-6,
        "det_f_guard": 0.2,
        "regularization_values": [0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0],
        "merit_relative_global": 1e-4,
        "merit_relative_q": 1e-4,
        "merit_relative_x": 1e-4,
    }
    if any(config[name] != value for name, value in fixed.items()):
        return "portable config changes a fixed solver policy"
    upper_bounds = {
        "residual_floor_global": 1e-2,
        "residual_floor_q": 1e-2,
        "residual_floor_x": 1e-2,
        "merit_noise": 1e-3,
        "merit_absolute_global": 1e-2,
        "merit_absolute_q": 1e-2,
        "merit_absolute_x": 1e-2,
        "step_tolerance_global": 1e-2,
        "step_tolerance_q": 1e-2,
        "step_tolerance_x": 1e-2,
    }
    nonnegative = {"merit_noise"}
    for name, value in config.items():
        if name == "regularization_values":
            if (
                not _finite_array(value)
                or len(value) < 2
                or value[0] != 0.0
                or any(right <= left for left, right in itertools.pairwise(value))
            ):
                return "regularization sequence is invalid"
        elif not _finite_array([value], shape=(1,)) or (value < 0 if name in nonnegative else value <= 0):
            return f"portable config field {name} is invalid"
        elif name in upper_bounds and value > upper_bounds[name]:
            return f"portable config field {name} exceeds its review bound"
    force_floor = acceptance.get("force_detection_floor_n")
    if not _finite_array([force_floor], shape=(1,)) or not 0 < force_floor <= 1e-2:
        return "portable force detection floor is invalid"
    return None


def _trajectory_config_failure(config, component, acceptance):
    reason = _config_failure(config, acceptance)
    if reason:
        return reason
    calibrated = _INTERNAL_CONFIG_FIELDS - {
        "epsilon_d",
        "det_f_guard",
        "regularization_values",
        "merit_relative_global",
        "merit_relative_q",
        "merit_relative_x",
    }
    if any(config[name] < component[name] for name in calibrated):
        return "trajectory candidate is less conservative than component calibration"
    return None


def _git_blob_sha256(git_sha, path):
    try:
        blob = subprocess.check_output(["git", "show", f"{git_sha}:{path}"], stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    return hashlib.sha256(blob).hexdigest()


def _source_set_matches(source_files, source_set):
    line_digest = hashlib.sha256(
        "".join(f"{path}:{source_files[path]}\n" for path in sorted(source_files)).encode()
    ).hexdigest()
    return source_set in (_hash_json(source_files), line_digest)


def _source_identity_failure(artifact, *, require_production=False):
    git_sha = artifact.get("git_sha", "")
    source_set = artifact.get("source_set_sha256", "")
    source_files = artifact.get("source_sha256")
    hexdigits = set("0123456789abcdef")
    if (
        len(git_sha) != 40
        or any(character not in hexdigits for character in git_sha.lower())
        or len(source_set) != 64
        or any(character not in hexdigits for character in source_set.lower())
        or not isinstance(source_files, dict)
        or not source_files
        or any(
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or not isinstance(value, str)
            or len(value) != 64
            or any(character not in hexdigits for character in value.lower())
            for path, value in source_files.items()
        )
        or not _source_set_matches(source_files, source_set)
    ):
        return "source-set identity is malformed or does not match its file hash map"
    production = [path for path in source_files if path.startswith(_PRODUCTION_SOURCE_PREFIX)]
    if require_production and not production:
        return "measurement source set does not include monolithic production code"
    for path, expected in source_files.items():
        if _git_blob_sha256(git_sha, path) != expected:
            return f"source hash does not match git revision: {path}"
    return None


def _production_source_failure(reference_sources, git_sha, candidate_sources=None, *, ignored_paths=()):
    expected = {
        path: digest
        for path, digest in reference_sources.items()
        if path.startswith(_PRODUCTION_SOURCE_PREFIX) and path not in ignored_paths
    }
    if not expected:
        return "calibration artifact has no production source identity"
    if candidate_sources is None:
        actual = {path: _git_blob_sha256(git_sha, path) for path in expected}
    else:
        overlap = expected.keys() & candidate_sources.keys()
        if not overlap:
            return "upstream artifact has no overlapping monolithic production source"
        actual = {path: candidate_sources[path] for path in overlap}
        expected = {path: expected[path] for path in overlap}
    if actual != expected:
        return "monolithic production sources differ from calibration measurement"
    return None


def aggregate_calibration_evidence(artifacts, *, freeze=False, verified_trajectory=None):
    """Aggregate raw batches and optionally freeze the calibration sub-state.

    This validates evidence completeness and identity.  It deliberately leaves
    ``v01_status`` as DRAFT because reference/E2E acceptance is a separate gate.
    """
    if verified_trajectory is not None and (
        type(verified_trajectory) is not _VerifiedTrajectoryEvidence
        or verified_trajectory.seal != _verified_trajectory_seal(verified_trajectory.audit_artifact)
    ):
        raise CalibrationFreezeError("verified_trajectory must be an intact in-process verified result")
    authoritative_trajectory = None if verified_trajectory is None else verified_trajectory.audit_artifact
    if freeze and any(artifact.get("artifact_kind") == "trajectory_evidence" for artifact in artifacts):
        raise CalibrationFreezeError("Serialized trajectory JSON is audit-only and cannot authorize freeze")
    artifacts = [*artifacts] + ([] if authoritative_trajectory is None else [authoritative_trajectory])
    expected_cases = {_case_key(case): case for case in freeze_calibration_cases()}
    expected_keys = {(device, key) for device in ("cpu", "cuda") for key in expected_cases}
    measurement_sources, trajectory_sources = set(), set()
    trajectory_measurement_sources, measurement_git_shas, trajectory_measurement_git_shas = set(), set(), set()
    records, seen, trajectory, artifact_failures, trajectory_configs = [], set(), set(), [], []
    for artifact in artifacts:
        kind = artifact.get("artifact_kind")
        if kind not in ("raw_measurement", "trajectory_evidence") or artifact.get("calibration_schema_version") != 2:
            raise CalibrationFreezeError("Only version-2 raw or trajectory evidence artifacts may be aggregated")
        if artifact.get("profile") != "freeze" or artifact.get("git_dirty") is not False:
            raise CalibrationFreezeError("Freeze evidence must use the freeze profile from a clean tree")
        source_error = _source_identity_failure(artifact, require_production=kind == "raw_measurement")
        if source_error:
            raise CalibrationFreezeError(source_error)
        source = artifact["source_set_sha256"]
        if kind == "raw_measurement":
            measurement_sources.add(source)
            measurement_git_shas.add(artifact["git_sha"])
        else:
            if artifact is not authoritative_trajectory:
                artifact_failures.append(
                    {"batch_index": None, "reason": "serialized trajectory evidence is audit-only"}
                )
                continue
            trajectory_sources.add(source)
            if set(artifact["source_sha256"]) != {"scripts/monolithic_reference/calibrate_components.py"}:
                raise CalibrationFreezeError("Trajectory evidence has an unexpected producer source set")
            measurement_source = artifact.get("measurement_source_set_sha256", "")
            if len(measurement_source) != 64:
                raise CalibrationFreezeError("Trajectory evidence is missing its measurement source dependency")
            trajectory_measurement_sources.add(measurement_source)
            trajectory_measurement_git_shas.add(artifact.get("measurement_git_sha", ""))
            upstream = artifact.get("upstream", {})
            compact_upstream = upstream.get("compact_calibration", {})
            normal_upstream = upstream.get("normal_loading", {})
            c4_upstream = upstream.get("c4", {})
            if (
                set(upstream) != {"compact_calibration", "normal_loading", "c4"}
                or set(compact_upstream) != {"artifact_sha256"}
                or len(compact_upstream.get("artifact_sha256", "")) != 64
                or set(normal_upstream) != {"cpu", "cuda"}
                or set(c4_upstream)
                != {
                    "git_sha",
                    "source_set_sha256",
                    "artifact_sha256",
                }
            ):
                raise CalibrationFreezeError("Trajectory upstream provenance is incomplete")
            validated_config = artifact.get("validated_solver_internal_config")
            derivation = artifact.get("trajectory_config_derivation", {})
            if (
                derivation.get("adjusted_field") != "merit_absolute_q"
                or derivation.get("safety_factor") != 1.5
                or derivation.get("round_up_quantum") != 0.5e-7
                or not _finite_array([derivation.get("probe_maximum"), derivation.get("result")], shape=(2,))
                or derivation["result"] != math.ceil(1.5 * derivation["probe_maximum"] / 0.5e-7) * 0.5e-7
                or not isinstance(validated_config, dict)
                or validated_config.get("merit_absolute_q") != derivation["result"]
            ):
                raise CalibrationFreezeError("Trajectory config derivation is invalid")
            trajectory_configs.append(validated_config)
        if kind == "raw_measurement" and artifact.get("repeats", 0) < 100:
            artifact_failures.append(
                {
                    "batch_index": artifact.get("batch_index"),
                    "reason": f"repeats {artifact.get('repeats', 0)} is below freeze minimum 100",
                }
            )
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
                    not isinstance(row.get("representative_states"), dict)
                    or set(row["representative_states"]) != {"free", "onset", "loading", "peak", "settled"}
                    or any(not isinstance(index, int) or index < 0 for index in row["representative_states"].values())
                    or len(row.get("candidate_artifact_sha256", "")) != 64
                    or len(row.get("legacy_default_baseline_artifact_sha256", "")) != 64
                    or not _finite_array(list(row.get("candidate_vs_legacy_default_relative_errors", {}).values()))
                    or max(row["candidate_vs_legacy_default_relative_errors"].values(), default=math.inf) > 0.05
                    or row.get("candidate_vs_legacy_default_baseline") != "PASS"
                    or row.get("false_convergence_count") != 0
                    or row.get("candidate_artifact_sha256")
                    != normal_upstream.get(device, {}).get("candidate_artifact_sha256")
                    or row.get("legacy_default_baseline_artifact_sha256")
                    != normal_upstream.get(device, {}).get("legacy_default_baseline_artifact_sha256")
                    or row.get("probe_artifact_sha256") != normal_upstream.get(device, {}).get("probe_artifact_sha256")
                    or row.get("probe_failed_final_merit_q_max")
                    != normal_upstream.get(device, {}).get("probe_failed_final_merit_q_max")
                ):
                    raise CalibrationFreezeError(f"Normal-loading evidence is incomplete: {key}")
                if role == "c4_supported_motion" and (
                    len(row.get("support_artifact_sha256", "")) != 64
                    or row.get("support_artifact_sha256") != c4_upstream.get("artifact_sha256")
                ):
                    raise CalibrationFreezeError(f"C4 support identity is missing or inconsistent: {key}")
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
            records.append(
                {
                    "device_class": device,
                    "case": case,
                    "raw": raw,
                    "repeats": artifact.get("repeats", 0),
                }
            )
    if len(measurement_sources) > 1:
        raise CalibrationFreezeError("Raw batches were measured from different source sets")
    if len(measurement_git_shas) > 1:
        raise CalibrationFreezeError("Raw batches were measured from different git revisions")
    if trajectory_measurement_sources and trajectory_measurement_sources != measurement_sources:
        raise CalibrationFreezeError("Trajectory evidence references a different calibration measurement source")
    if trajectory_measurement_git_shas and trajectory_measurement_git_shas != measurement_git_shas:
        raise CalibrationFreezeError("Trajectory evidence references a different calibration measurement revision")
    failures = []
    for record in records:
        reason = _record_failure(record["raw"], record["case"], record["device_class"], record["repeats"])
        if reason:
            failures.append({"device": record["device_class"], "case_id": record["case"].case_id, "reason": reason})
    missing = sorted(f"{device}:{case}" for device, case in expected_keys - seen)
    expected_trajectory = {
        (device, role) for device in ("cpu", "cuda") for role in ("normal_loading_1000", "c4_supported_motion")
    }
    missing_trajectory = sorted(f"{device}:{case}" for device, case in expected_trajectory - trajectory)
    coordinate = _coordinate_envelope(records)
    coordinate_ready = (
        coordinate["maximum_passing_translation_offset_m"] is not None
        and coordinate["maximum_passing_translation_offset_m"] >= coordinate["minimum_required_translation_offset_m"]
    )
    per_device = _aggregate_parameters(records, coordinate["maximum_passing_translation_offset_m"])
    component_portable, portable_acceptance = _portable_config(per_device)
    portable = component_portable
    config_error = _config_failure(component_portable, portable_acceptance)
    if trajectory_configs:
        if any(config != trajectory_configs[0] for config in trajectory_configs[1:]):
            config_error = "trajectory artifacts disagree on the validated solver config"
        else:
            config_error = _trajectory_config_failure(trajectory_configs[0], component_portable, portable_acceptance)
            if config_error is None:
                portable = trajectory_configs[0]
    if config_error:
        artifact_failures.append({"batch_index": None, "reason": config_error})
    ready = (
        not missing
        and not missing_trajectory
        and not failures
        and not artifact_failures
        and len(measurement_sources) == 1
        and coordinate_ready
    )
    if freeze and not ready:
        raise CalibrationFreezeError("Calibration evidence is incomplete or contains failed gates")
    return {
        "artifact_kind": "calibration_freeze",
        "calibration_schema_version": 2,
        "calibration_status": "FROZEN" if freeze else "CANDIDATE" if ready else "INCOMPLETE",
        "v01_status": "DRAFT",
        "measurement_source": {
            "git_sha": next(iter(measurement_git_shas), None),
            "source_set_sha256": next(iter(measurement_sources), None),
        },
        "trajectory_source_set_sha256": sorted(trajectory_sources),
        "exact_fixture_translation_offset_evidence": coordinate,
        "outside_coordinate_controls_m": [_OUTSIDE_WORLD_OFFSET],
        "expected_case_count": len(expected_keys),
        "observed_case_count": len(seen),
        "missing_case_keys": missing,
        "missing_trajectory_keys": missing_trajectory,
        "failed_records": failures,
        "artifact_failures": artifact_failures,
        "candidate_private_config": per_device,
        "component_portable_solver_internal_config": component_portable,
        "portable_solver_internal_config": portable,
        "portable_acceptance": portable_acceptance,
        "component_linear_gate_kind": "dense_true_residual_cancellation_not_runtime_pcg",
    }


def _load_normal_run(path):
    path = Path(path)
    required = {name: path / name for name in ("metadata.json", "summary.json", "steps.jsonl")}
    if any(not item.is_file() for item in required.values()):
        raise CalibrationFreezeError(f"Normal-loading run is incomplete: {path}")
    records = [json.loads(line) for line in required["steps.jsonl"].read_text().splitlines()]
    return {
        "metadata": json.loads(required["metadata.json"].read_text()),
        "summary": json.loads(required["summary.json"].read_text()),
        "records": records,
        "artifact_sha256": _hash_json(
            {name: hashlib.sha256(item.read_bytes()).hexdigest() for name, item in required.items()}
        ),
    }


def _normal_loading_failure(run):
    metadata, summary, records = run["metadata"], run["summary"], run["records"]
    limits = metadata.get("actual_parameters", {}).get("acceptance", {})
    minimum_steps = limits.get("minimum_substeps", 1000)
    if (
        not isinstance(records, list)
        or len(records) < minimum_steps
        or summary.get("substeps") != len(records)
        or not all(record.get("finite_state") is True for record in records)
    ):
        return "normal-loading records are incomplete or nonfinite"
    numerical_gates = {
        name: value for name, value in summary.get("gates", {}).items() if not name.startswith("frozen_")
    }
    if (
        not numerical_gates
        or numerical_gates.get("converged_linear_gates") is not True
        or any(value is not True for value in numerical_gates.values())
    ):
        return "normal-loading numerical gates did not all pass"
    converged_ratio = sum(record.get("converged") is True for record in records) / len(records)
    if not math.isclose(
        converged_ratio, summary.get("converged_ratio", math.nan), rel_tol=0.0, abs_tol=1e-12
    ) or converged_ratio < limits.get("minimum_converged_ratio", 0.99):
        return "normal-loading convergence ratio does not match raw records"
    streak = maximum_streak = linear_solve_count = 0
    for record in records:
        streak = 0 if record.get("converged") is True else streak + 1
        maximum_streak = max(maximum_streak, streak)
        if record.get("converged") is True:
            ratios = record.get("nonlinear_convergence_ratios")
            if not _finite_array(ratios, shape=(3,)) or max(ratios) > 1.0:
                return "normal-loading contains false nonlinear convergence"
            if record.get("linear_iterations", 0) > 0:
                linear_solve_count += 1
                tolerance = limits.get("linear_tolerance")
                values = [record.get("stats", {}).get(name) for name in ("rho", "rho_q", "rho_x")]
                if (
                    not _finite_array(values, shape=(3,))
                    or not _finite_array([tolerance], shape=(1,))
                    or max(values) > tolerance
                ):
                    return "normal-loading contains false linear convergence"
    if linear_solve_count == 0:
        return "normal-loading did not exercise runtime PCG"
    if maximum_streak != summary.get("maximum_consecutive_non_success") or maximum_streak > limits.get(
        "maximum_consecutive_non_success", 2
    ):
        return "normal-loading non-success streak does not match raw records"
    if summary.get("e2e_numerical_pass") is not True:
        return "normal-loading summary is not a numerical PASS"
    return None


def _normal_fixture_physics_sha(metadata):
    fixture = dict(metadata.get("actual_parameters", {}))
    fixture.pop("acceptance", None)
    for name in (
        "calibration_status",
        "support_status",
        "reference_status",
        "calibration_provenance",
        "support_provenance",
        "reference_provenance",
    ):
        fixture.pop(name, None)
    return _hash_json(fixture)


def _normal_loading_role(candidate, baseline, portable_config, reference_sources, probe):
    cm, bm = candidate["metadata"], baseline["metadata"]
    device = _device_class(cm.get("device"))
    if _device_class(bm.get("device")) != device:
        raise CalibrationFreezeError("Candidate/baseline normal-loading devices differ")
    if any(metadata.get("worktree_dirty") is not False for metadata in (cm, bm)):
        raise CalibrationFreezeError(f"Normal-loading evidence is dirty on {device}")
    if _normal_fixture_physics_sha(cm) != _normal_fixture_physics_sha(bm):
        raise CalibrationFreezeError(f"Candidate/baseline normal-loading physics fixtures differ on {device}")
    if (
        cm.get("requested_solver_internal_config") not in (None, portable_config)
        or cm.get("solver_internal_config") != portable_config
    ):
        raise CalibrationFreezeError(f"Candidate did not use the validated config on {device}")
    if bm.get("requested_solver_internal_config") is not None:
        raise CalibrationFreezeError(f"Legacy default baseline unexpectedly applied a candidate config on {device}")
    for run, ignored_paths in (
        (candidate, {"newton/_src/solvers/monolithic/solver_monolithic.py"}),
        (baseline, ()),
    ):
        source_error = _production_source_failure(
            reference_sources, run["metadata"].get("newton_sha", ""), ignored_paths=ignored_paths
        )
        if source_error:
            raise CalibrationFreezeError(f"{source_error} on {device}")
        numerical_error = _normal_loading_failure(run)
        if numerical_error:
            raise CalibrationFreezeError(f"{numerical_error} on {device}")
    metrics = ("peak_force_n", "settled_force_n", "secant_stiffness_n_m", "curve_work_j")
    parity = {}
    for name in metrics:
        actual, reference = candidate["summary"].get(name), baseline["summary"].get(name)
        if not _finite_array([actual, reference], shape=(2,)):
            raise CalibrationFreezeError(f"Normal-loading parity metric {name} is missing on {device}")
        error = abs(actual - reference) / max(abs(actual), abs(reference), 1e-12)
        parity[name] = error
    if max(parity.values()) > 0.05:
        raise CalibrationFreezeError(f"Candidate/legacy-default parity exceeds 5% on {device}")
    records = candidate["records"]
    onset = candidate["summary"]["contact_onset_step"]
    if onset is None:
        raise CalibrationFreezeError(f"Representative normal-loading onset is unavailable on {device}")
    peak = int(np.argmax([record["normal_compressive_force_n"] for record in records]))
    indices = {
        "free": 0,
        "onset": onset,
        "loading": (onset + peak) // 2,
        "peak": peak,
        "settled": len(records) - 1,
    }
    if any(not 0 <= index < len(records) for index in indices.values()):
        raise CalibrationFreezeError(f"Representative normal-loading states are unavailable on {device}")
    return {
        "device": device,
        "role": "normal_loading_1000",
        "status": "PASS",
        "fixture_sha256": cm["input_sha256"],
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "legacy_default_baseline_artifact_sha256": baseline["artifact_sha256"],
        "probe_artifact_sha256": probe["artifact_sha256"],
        "probe_failed_final_merit_q_max": probe["failed_final_merit_q_max"],
        "finite_state": True,
        "convergence_gate": "PASS",
        "representative_states": indices,
        "candidate_vs_legacy_default_baseline": "PASS",
        "candidate_vs_legacy_default_relative_errors": parity,
        "false_convergence_count": 0,
    }


def _derive_trajectory_config(probes, component_config, reference_sources):
    by_device = {_device_class(run["metadata"].get("device")): run for run in probes}
    if len(probes) != 2 or set(by_device) != {"cpu", "cuda"}:
        raise CalibrationFreezeError("Trajectory config probe requires exactly one CPU and CUDA run")
    maxima, provenance = [], {}
    for device, run in by_device.items():
        metadata, summary, records = run["metadata"], run["summary"], run["records"]
        if (
            metadata.get("worktree_dirty") is not False
            or metadata.get("requested_solver_internal_config") != component_config
            or metadata.get("solver_internal_config") != component_config
            or _production_source_failure(reference_sources, metadata.get("newton_sha", ""))
        ):
            raise CalibrationFreezeError(f"Unadjusted trajectory probe provenance differs on {device}")
        failed = {
            name
            for name, passed in summary.get("gates", {}).items()
            if not name.startswith("frozen_") and passed is not True
        }
        bad = [record for record in records if record.get("converged") is not True]
        if (
            summary.get("e2e_numerical_pass") is not False
            or summary.get("substeps") != 1000
            or failed != {"converged_ratio", "consecutive_non_success"}
            or not bad
            or not all(record.get("finite_state") is True for record in records)
            or any(record.get("rolled_back") is True for record in records)
            or not all(record.get("committed_unconverged") is True for record in bad)
        ):
            raise CalibrationFreezeError(f"Unadjusted probe did not isolate a convergence-only failure on {device}")
        merits = []
        for record in bad:
            ratios = record.get("nonlinear_convergence_ratios")
            merit_q = record.get("stats", {}).get("merit_q_final")
            if (
                not _finite_array(ratios, shape=(3,))
                or not (ratios[0] <= 1.0 < ratios[1] and ratios[2] <= 1.0)
                or not _finite_array([merit_q], shape=(1,))
                or merit_q <= 0
            ):
                raise CalibrationFreezeError(f"Unadjusted probe failure is not isolated to q merit on {device}")
            merits.append(merit_q)
        maximum = max(merits)
        maxima.append(maximum)
        provenance[device] = {
            "artifact_sha256": run["artifact_sha256"],
            "git_sha": metadata["newton_sha"],
            "failed_final_merit_q_max": maximum,
        }
    observed = max(maxima)
    adjusted = dict(component_config)
    quantum = 0.5e-7
    adjusted["merit_absolute_q"] = math.ceil(1.5 * observed / quantum) * quantum
    return adjusted, provenance


def _collect_nested_values(value, key):
    result = []
    if isinstance(value, dict):
        if key in value:
            result.append(value[key])
        for item in value.values():
            result.extend(_collect_nested_values(item, key))
    elif isinstance(value, list):
        for item in value:
            result.extend(_collect_nested_values(item, key))
    return result


def _current_producer_identity():
    root = Path(__file__).resolve().parents[2]
    paths = [Path(__file__).resolve()]
    sources = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    return {
        "git_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
        ),
        "source_sha256": sources,
        "source_set_sha256": _hash_json(sources),
    }


def build_trajectory_evidence(
    calibration_raw,
    probe_runs,
    candidate_runs,
    baseline_runs,
    c4_artifact_bytes,
    calibration_compact_bytes,
):
    """Derive trajectory roles from actual normal-loading and C4 artifacts."""
    try:
        c4_artifact = json.loads(c4_artifact_bytes)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CalibrationFreezeError("C4 artifact is not valid JSON bytes") from error
    c4_sha256 = hashlib.sha256(c4_artifact_bytes).hexdigest()
    try:
        calibration_compact = json.loads(calibration_compact_bytes)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise CalibrationFreezeError("Compact calibration artifact is not valid JSON bytes") from error
    calibration_compact_sha256 = hashlib.sha256(calibration_compact_bytes).hexdigest()
    component = aggregate_calibration_evidence([calibration_raw])
    if component["missing_case_keys"] or component["failed_records"] or component["artifact_failures"]:
        raise CalibrationFreezeError("Calibration component artifact is not complete")
    source_files = calibration_raw["source_sha256"]
    source_set = calibration_raw["source_set_sha256"]
    measurement_git_sha = calibration_raw.get("git_sha")
    c4_sources = c4_artifact.get("source_sha256")
    if (
        c4_artifact.get("status") != "FROZEN"
        or c4_artifact.get("git_dirty") is not False
        or len(c4_sha256) != 64
        or not isinstance(c4_sources, dict)
        or not c4_sources
        or not _source_set_matches(c4_sources, c4_artifact.get("source_set_sha256"))
    ):
        raise CalibrationFreezeError("C4 artifact is not frozen on the calibration source revision")
    source_error = _source_identity_failure(c4_artifact)
    if source_error:
        raise CalibrationFreezeError(f"C4 {source_error}")
    production_error = _production_source_failure(
        source_files,
        c4_artifact["git_sha"],
        c4_sources,
        ignored_paths={"newton/_src/solvers/monolithic/solver_monolithic.py"},
    )
    if production_error:
        raise CalibrationFreezeError(production_error)
    c4_rows = {_device_class(row.get("device")): row for row in c4_artifact.get("evidence", [])}
    candidate_config, probe_provenance = _derive_trajectory_config(
        probe_runs, component["component_portable_solver_internal_config"], source_files
    )
    candidates = {_device_class(run["metadata"].get("device")): run for run in candidate_runs}
    baseline = {_device_class(run["metadata"].get("device")): run for run in baseline_runs}
    if (
        len(candidate_runs) != 2
        or len(baseline_runs) != 2
        or len(c4_artifact.get("evidence", [])) != 2
        or set(candidates) != {"cpu", "cuda"}
        or set(baseline) != {"cpu", "cuda"}
        or set(c4_rows) != {"cpu", "cuda"}
    ):
        raise CalibrationFreezeError("Trajectory producer requires exactly one CPU and CUDA artifact per role")
    candidate_configs = [candidates[device]["metadata"].get("solver_internal_config") for device in ("cpu", "cuda")]
    if candidate_configs[0] != candidate_configs[1]:
        raise CalibrationFreezeError("CPU/CUDA trajectory candidates used different solver configs")
    if candidate_configs[0] != candidate_config:
        raise CalibrationFreezeError("Final candidate does not equal the config derived from failed probes")
    if (
        calibration_compact.get("calibration_status") != "FROZEN"
        or calibration_compact.get("solver_internal_config") != candidate_config
    ):
        raise CalibrationFreezeError("Compact calibration artifact does not contain the validated config")
    for device in ("cpu", "cuda"):
        metadata = candidates[device]["metadata"]
        if (
            metadata.get("calibration_status") != "FROZEN"
            or metadata.get("calibration_sha256") != calibration_compact_sha256
            or metadata.get("support_status") != "FROZEN"
            or metadata.get("support_sha256") != c4_sha256
        ):
            raise CalibrationFreezeError(f"Candidate fixture provenance hash differs on {device}")
    config_error = _trajectory_config_failure(
        candidate_config,
        component["component_portable_solver_internal_config"],
        component["portable_acceptance"],
    )
    if config_error:
        raise CalibrationFreezeError(config_error)
    output = []
    for device in ("cpu", "cuda"):
        output.append(
            _normal_loading_role(
                candidates[device], baseline[device], candidate_config, source_files, probe_provenance[device]
            )
        )
        c4 = c4_rows[device]
        c4_configs = _collect_nested_values(c4, "solver_internal_config")
        if (
            c4.get("c4_gate") != "PASS"
            or c4.get("supported_local_motion_gate") != "PASS"
            or c4.get("support_envelope", {}).get("status") != "FROZEN"
            or not c4_configs
            or any(config != candidate_config for config in c4_configs)
        ):
            raise CalibrationFreezeError(f"C4 supported motion gate failed on {device}")
        output.append(
            {
                "device": device,
                "role": "c4_supported_motion",
                "status": "PASS",
                "fixture_sha256": c4.get("support_envelope", {}).get("sha256") or c4_sha256,
                "support_artifact_sha256": c4_sha256,
                "finite_state": True,
                "convergence_gate": "PASS",
            }
        )
    producer = _current_producer_identity()
    artifact = {
        "artifact_kind": "trajectory_evidence",
        "calibration_schema_version": 2,
        "profile": "freeze",
        **producer,
        "measurement_git_sha": measurement_git_sha,
        "measurement_source_set_sha256": source_set,
        "validated_solver_internal_config": candidate_config,
        "trajectory_config_derivation": {
            "adjusted_field": "merit_absolute_q",
            "probe_maximum": max(row["failed_final_merit_q_max"] for row in probe_provenance.values()),
            "safety_factor": 1.5,
            "round_up_quantum": 0.5e-7,
            "result": candidate_config["merit_absolute_q"],
        },
        "upstream": {
            "compact_calibration": {"artifact_sha256": calibration_compact_sha256},
            "normal_loading": {
                device: {
                    "probe_git_sha": probe_provenance[device]["git_sha"],
                    "probe_artifact_sha256": probe_provenance[device]["artifact_sha256"],
                    "probe_failed_final_merit_q_max": probe_provenance[device]["failed_final_merit_q_max"],
                    "candidate_git_sha": candidates[device]["metadata"]["newton_sha"],
                    "candidate_artifact_sha256": candidates[device]["artifact_sha256"],
                    "legacy_default_baseline_git_sha": baseline[device]["metadata"]["newton_sha"],
                    "legacy_default_baseline_artifact_sha256": baseline[device]["artifact_sha256"],
                }
                for device in ("cpu", "cuda")
            },
            "c4": {
                "git_sha": c4_artifact["git_sha"],
                "source_set_sha256": c4_artifact["source_set_sha256"],
                "artifact_sha256": c4_sha256,
            },
        },
        "evidence": output,
    }
    return _VerifiedTrajectoryEvidence(artifact, _verified_trajectory_seal(artifact))


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
