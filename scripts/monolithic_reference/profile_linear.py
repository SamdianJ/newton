# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile real tiny assembled systems with an offline diagonal negative control.

Run from a checkout with ``uv run -m scripts.monolithic_reference.profile_linear
--output profile.json``. The diagonal inverse is injected only by this offline
tool; the solver's public runtime continues to require the actor preconditioner.
Timings include host orchestration and explicitly synchronize at measurement
boundaries. They are measurements of tiny systems, not an asset-scale speed claim.
"""

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator

from newton._src.solvers.monolithic import linear
from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton._src.solvers.monolithic.solver_monolithic import SolverMonolithic, _scale_vector
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.test_solver_monolithic_collision import _scene


@wp.kernel
def _build_diagonal_inverse(
    diagonal: wp.array[float],
    scale: wp.array[float],
    regularization: float,
    inverse: wp.array[float],
    status: wp.array[int],
):
    i = wp.tid()
    value = scale[i] * diagonal[i] * scale[i] + regularization
    if value <= 0.0 or not wp.isfinite(value):
        wp.atomic_max(status, 0, 1)
        return
    inverse[i] = 1.0 / value
    if not wp.isfinite(inverse[i]):
        wp.atomic_max(status, 0, 1)


@wp.kernel
def _apply_diagonal_inverse(
    inverse: wp.array[float],
    x: wp.array[float],
    y: wp.array[float],
    z: wp.array[float],
    alpha: float,
    beta: float,
):
    i = wp.tid()
    value = alpha * inverse[i] * x[i]
    if beta != 0.0:
        value += beta * y[i]
    z[i] = value


class _DiagonalInverse:
    """Offline inverse of the same scaled production BSR diagonal."""

    def __init__(self, workspace):
        self.workspace = workspace
        self.values = wp.empty(workspace.layout.scalar_dof_count, dtype=float, device=workspace.device)
        self.status = wp.zeros(1, dtype=int, device=workspace.device)
        self.operator = LinearOperator(workspace.operator.shape, wp.float32, workspace.device, self.matvec)

    def setup(self):
        self.status.zero_()
        w = self.workspace
        wp.launch(
            _build_diagonal_inverse,
            self.values.size,
            [w._matrix_diagonal, w.scale, w._lambda, self.values, self.status],
            device=w.device,
        )
        if self.status.numpy()[0]:
            raise ValueError("Offline diagonal inverse is not finite and positive")

    def matvec(self, x, y, z, alpha, beta):
        wp.launch(_apply_diagonal_inverse, x.size, [self.values, x, y, z, alpha, beta], device=self.workspace.device)


def _measure(device, function):
    wp.synchronize_device(device)
    start = time.perf_counter()
    result = function()
    wp.synchronize_device(device)
    return result, (time.perf_counter() - start) * 1000.0


def _percentiles(values):
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def _allocation_sample(device, function):
    allocator = wp.get_device(device).get_allocator()
    with patch.object(allocator, "allocate", wraps=allocator.allocate) as allocate:
        result, duration = _measure(device, function)
    return (
        result,
        duration,
        {"count": allocate.call_count, "requested_bytes": sum(int(call.args[0]) for call in allocate.call_args_list)},
    )


def _memory(workspace):
    groups = {
        "fixed_patterns": [array for pattern in workspace._fixed_patterns or () for array in pattern],
        "global_bsr": [
            workspace.k_global_scalar_bsr.offsets,
            workspace.k_global_scalar_bsr.columns,
            workspace.k_global_scalar_bsr.values,
        ],
        "global_triplets": [
            workspace._global.rows,
            workspace._global.columns,
            workspace._global.values,
            workspace._global.count,
            workspace._global.status,
        ],
        "actor_q": [workspace.aq_actor_dense],
        "internal_bsr3": [
            workspace.ax_internal_bsr3.offsets,
            workspace.ax_internal_bsr3.columns,
            workspace.ax_internal_bsr3.values,
        ],
        "internal_triplets": [
            workspace._internal.rows,
            workspace._internal.columns,
            workspace._internal.values,
            workspace._internal.count,
            workspace._internal.status,
        ],
        "contact_factors": [
            workspace._factors.gq,
            workspace._factors.gx_columns,
            workspace._factors.gx_values,
            workspace._factors.weights,
            workspace._factors.count,
            workspace._factors.status,
        ],
        "preconditioner": [
            workspace._q_preconditioner,
            workspace._q_cholesky,
            workspace._x_preconditioner,
            workspace._x_cholesky,
            workspace._preconditioned,
        ],
    }
    seen = set()
    output = {}
    for group, arrays in groups.items():
        unique = [a for a in arrays if a.ptr and a.ptr not in seen]
        seen.update(a.ptr for a in unique)
        output[group] = sum(a.capacity for a in unique)
    remaining = [a for a in vars(workspace).values() if isinstance(a, wp.array) and a.ptr and a.ptr not in seen]
    output["other_linear_workspace"] = sum(a.capacity for a in remaining)
    output["total_enumerated_capacity_bytes"] = sum(output.values())
    output["peak_device_bytes"] = None
    return output


def _factor_stage_times(workspace, generation, rhs, repeats):
    stages = {
        linear._assemble_monolithic_q_preconditioner: "q_setup",
        linear._factor_monolithic_q_cholesky: "q_factor",
        linear._assemble_monolithic_x_preconditioner: "particle_setup",
        linear._factor_monolithic_x_cholesky: "particle_factor",
        linear._apply_q_preconditioner: "q_apply",
        linear._apply_x_preconditioner: "particle_apply",
    }
    samples = {name: [] for name in stages.values()}
    launch = wp.launch

    def timed_launch(kernel, *args, **kwargs):
        if kernel not in stages:
            return launch(kernel, *args, **kwargs)
        result, elapsed = _measure(workspace.device, lambda: launch(kernel, *args, **kwargs))
        samples[stages[kernel]].append(elapsed)
        return result

    # This explicitly synchronized microprofile is separate from total PCG timings.
    with patch.object(wp, "launch", side_effect=timed_launch):
        for _ in range(repeats):
            workspace.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0)
            workspace.preconditioner.matvec(rhs, workspace._ap, workspace._ap, 1.0, 0.0)
    return {name: _percentiles(values) for name, values in samples.items()}


def _make_solver(device, *, active, young_modulus, poisson_ratio, contact_stiffness):
    if active:
        model, state = _scene(device, margin=0.006)
        model.joint_limit_ke.zero_()
        model.joint_limit_kd.zero_()
        model.joint_axis.assign([[0.0, 1.0, 0.0]])
        model.body_com.assign([[0.003, 0.004, 0.005]])
    else:
        fixture = build_tiny_cpu_fixture(device=device)
        model, state = fixture.model, fixture.state
    material = model.tet_materials.numpy()
    material[:, 0] = young_modulus / (2 * (1 + poisson_ratio))
    material[:, 1] = young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    model.tet_materials.assign(material)
    pipeline = MonolithicCollisionPipeline(model)
    solver = SolverMonolithic(model, collision_pipeline=pipeline, contact_stiffness=contact_stiffness)
    return solver, state


def _predictor_assembly(solver, state, dt):
    solver._transaction._finished = True
    solver._transaction.begin(state, solver._transaction.trial.state, solver._default_control, dt)
    solver._assembly_sequence = 0
    solver._metrics = {"nonlinear_iterations": 0, "matrix_assembly_count": 0}
    return solver._evaluate_current(solver._transaction.accepted, dt)


def profile_case(device, *, active, dt, young_modulus, poisson_ratio, contact_stiffness, repeats=10):
    """Compare real predictor systems without stepping or replacing any physical terms."""
    solver, state = _make_solver(
        device,
        active=active,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        contact_stiffness=contact_stiffness,
    )
    _predictor_assembly(solver, state, dt)
    assembly_samples, assembly_allocations = [], []
    for _ in range(repeats):
        _, duration, allocations = _allocation_sample(device, lambda: _predictor_assembly(solver, state, dt))
        assembly_samples.append(duration)
        assembly_allocations.append(allocations)
    w, generation = solver._linear, solver._generation
    wp.launch(
        _scale_vector,
        w.layout.scalar_dof_count,
        [solver._transaction.accepted.residual, w.scale, -1.0, solver._rhs_hat],
        device=device,
    )
    w.set_regularization(0.0, generation=generation)
    setup_status = w.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0)
    if setup_status != linear.MonolithicLinearStatus.SUCCESS:
        raise ValueError(f"Actor setup failed: {setup_status.name}")
    config = linear.MonolithicPcgConfig(
        200,
        10,
        solver.linear_tolerance,
        solver._config.residual_floor_global,
        solver._config.residual_floor_q,
        solver._config.residual_floor_x,
        0.0,
        1e-12,
        0.0,
        10,
        1e-3,
    )
    rhs_before = solver._rhs_hat.numpy().copy()
    bsr_before = w.k_global_scalar_bsr.values.numpy().copy()
    diagonal = _DiagonalInverse(w)
    diagonal.setup()
    actor_operator = w.preconditioner
    methods = {}
    try:
        for kind, inverse in [("actor_block", actor_operator), ("diagonal_negative_control", diagonal.operator)]:
            w.preconditioner = inverse

            def setup(kind=kind):
                if kind == "actor_block":
                    return w.factor_actor_preconditioner(generation=generation, pivot_tolerance=0.0)
                return diagonal.setup()

            def solve():
                return w.solve_pcg(
                    solver._rhs_hat,
                    solver._y,
                    generation=generation,
                    warm_start=linear.MonolithicPcgWarmStart.ZERO,
                    config=config,
                )

            setup()
            solve()
            inverse.matvec(solver._rhs_hat, w._ap, w._ap, 1.0, 0.0)
            setup_samples, apply_samples, solve_samples = [], [], []
            results, solve_allocations, setup_allocations = [], [], []
            for _ in range(repeats):
                _, elapsed, allocations = _allocation_sample(device, setup)
                setup_samples.append(elapsed)
                setup_allocations.append(allocations)
                _, elapsed = _measure(
                    device, lambda inverse=inverse: inverse.matvec(solver._rhs_hat, w._ap, w._ap, 1.0, 0.0)
                )
                apply_samples.append(elapsed)
                result, elapsed, allocations = _allocation_sample(device, solve)
                solve_samples.append(elapsed)
                solve_allocations.append(allocations)
                results.append({**asdict(result), "status": result.status.name, "warm_start": result.warm_start.name})
            methods[kind] = {
                "additional_inverse_capacity_bytes": diagonal.values.capacity + diagonal.status.capacity
                if kind == "diagonal_negative_control"
                else 0,
                "results": results,
                "pcg_iterations": _percentiles([r["iterations"] for r in results]),
                "pcg_ms": _percentiles(solve_samples),
                "setup_ms": _percentiles(setup_samples),
                "inverse_apply_ms": _percentiles(apply_samples),
                "pcg_ms_samples": solve_samples,
                "setup_ms_samples": setup_samples,
                "inverse_apply_ms_samples": apply_samples,
                "solve_allocator_samples": solve_allocations,
                "setup_allocator_samples": setup_allocations,
                "success_count": sum(r["status"] == "SUCCESS" for r in results),
            }
    finally:
        w.preconditioner = actor_operator
    np.testing.assert_array_equal(solver._rhs_hat.numpy(), rhs_before)
    np.testing.assert_array_equal(w.k_global_scalar_bsr.values.numpy(), bsr_before)
    stages = _factor_stage_times(w, generation, solver._rhs_hat, repeats)
    return {
        "fixture": "one_revolute_one_tet_plane_contact" if active else "monolithic_tiny_rev_pris_one_tet_v1",
        "device": str(w.device),
        "dt": dt,
        "young_modulus": young_modulus,
        "poisson_ratio": poisson_ratio,
        "contact_stiffness": contact_stiffness,
        "lambda": 0.0,
        "repeats": repeats,
        "dofs": w.layout.scalar_dof_count,
        "q_dofs": w.layout.q_dof_count,
        "dynamic_particles": w.layout.dynamic_particle_count,
        "tets": solver.model.tet_count,
        "active_samples": int(w._factors.count.numpy()[0]),
        "nnz": int(w.k_global_scalar_bsr.offsets.numpy()[-1]),
        "capacities": asdict(w.capacities),
        "linear_config": asdict(config),
        "memory": _memory(w),
        "predictor_assembly_ms": _percentiles(assembly_samples),
        "assembly_allocator_samples": assembly_allocations,
        "actor_stage_microprofile_ms": stages,
        "methods": methods,
        "total_step_ms": None,
        "dense_oracle_used": False,
    }


def _json_finite(value):
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--device", action="append", help="Repeat to choose devices; defaults to all available devices")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least 2")
    wp.init()
    devices = args.device or [str(device) for device in wp.get_devices()]
    cases = []
    for device in devices:
        for active in (False, True):
            for young_modulus, poisson_ratio in ((1e4, 0.3), (1e5, 0.45)):
                for dt in (1e-3, 1e-2):
                    for stiffness in (2e4, 2e5, 2e6) if active else (2e5,):
                        cases.append(
                            profile_case(
                                device,
                                active=active,
                                dt=dt,
                                young_modulus=young_modulus,
                                poisson_ratio=poisson_ratio,
                                contact_stiffness=stiffness,
                                repeats=args.repeats,
                            )
                        )
    sources = [Path(linear.__file__), Path(__file__)]
    report = {
        "status": "DRAFT_COMPONENT_PROFILE",
        "newton_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "warp_version": wp.__version__,
        "platform": platform.platform(),
        "devices": [{"alias": str(wp.get_device(device)), "name": wp.get_device(device).name} for device in devices],
        "source_sha256": {
            str(path.relative_to(Path.cwd())): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "measurement_scope": "Same actual predictor BSR and RHS; synchronized host-wall timings; no physical terms or thresholds changed; zero warm starts for both inverse preconditioners",
        "limitations": [
            "Tiny systems only; no trajectory, E2E, total-step, or asset-scale claim",
            "Allocator records cover the Warp Python device allocator; native sparse temporaries and driver peak memory are not measured",
            "Stage microtimings explicitly synchronize each observed kernel and are separate from normal PCG timing",
            "Current internal numerical configuration is recorded as DRAFT, not P0-frozen calibration",
        ],
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_finite(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
