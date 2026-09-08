# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Profile the frozen single-finger asset and a medium refined tet mesh.

This is an opt-in validation tool.  ``diagonal`` and ``cross_disabled`` are
negative controls injected only while this module is running; production
``SolverMonolithic`` remains actor-block with both contact cross blocks.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import time
from collections import Counter
from contextlib import ExitStack, nullcontext
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton._src.solvers.monolithic.solver_monolithic as solver_module
from newton._src.geometry.soft_contacts_sdf import SDF_FACE_ITERS, SDF_LS_ITERS, create_soft_face_contacts
from newton._src.solvers.monolithic import linear
from newton._src.solvers.monolithic.collision import create_monolithic_p1q3_face_contacts
from scripts.monolithic_reference.calibrate_p1q3 import _build_scene as build_refined_scene
from scripts.monolithic_reference.internal_envelope import FROZEN_FIXTURE
from scripts.monolithic_reference.normal_loading import (
    DEFAULT_FIXTURE,
    _json_value,
    assess_run,
    build_scene,
    command_at,
    load_fixture,
)
from scripts.monolithic_reference.normal_loading import (
    _measure as measure_normal,
)
from scripts.monolithic_reference.profile_linear import _DiagonalInverse, _memory


@wp.kernel
def _zero_qx_xq_triplets(rows: wp.array[int], columns: wp.array[int], values: wp.array[float], q_dof_count: int):
    i = wp.tid()
    row = rows[i]
    column = columns[i]
    if (row < q_dof_count and column >= q_dof_count) or (column < q_dof_count and row >= q_dof_count):
        values[i] = 0.0


def _percentiles(values):
    if not values:
        return None
    return {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}


def _finite_json(value):
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, np.generic):
        return _finite_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def summarize_records(records):
    """Recompute actual sparsity, solve quality and exit counts from saved steps."""
    result = {
        "record_count": len(records),
        "status_counts": dict(Counter(record["convergence_status"] for record in records)),
        "rollback_count": sum(bool(record["rolled_back"]) for record in records),
        "unconverged_count": sum(not record["converged"] for record in records),
        "nonfinite_state_count": sum(not record["finite_state"] for record in records),
    }
    for name in ("matrix_nnz", "triplet_count", "triplet_capacity", "active_sample_count"):
        result[name] = _percentiles([record["stats"][name] for record in records])
    for name in ("rho", "rho_q", "rho_x"):
        values = [record["stats"][name] for record in records if record["stats"][name] is not None]
        result[name] = {
            "measured_count": len(values),
            "unmeasured_count": len(records) - len(values),
            "percentiles": _percentiles(values),
            "maximum": max(values, default=None),
        }
    for name in ("matrix_assembly_count", "true_residual_recomputations", "residual_replacements"):
        result[name] = sum(record["stats"][name] for record in records)
    return result


def _save_trajectories(report, output):
    """Write every embedded trajectory once and bind exact JSONL bytes by hash."""
    hashes = {}

    def visit(value):
        if isinstance(value, dict):
            if "records" in value:
                records = value.pop("records")
                relative = f"trajectory-{len(hashes):03d}.steps.jsonl"
                payload = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in records)
                (output / relative).write_text(payload)
                digest = hashlib.sha256(payload.encode()).hexdigest()
                hashes[relative] = digest
                value["trajectory"] = {"path": relative, "sha256": digest, "record_count": len(records)}
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(report)
    return hashes


def _rss_bytes():
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, _unit = line.split()
            values[name.removesuffix(":").lower() + "_bytes"] = int(value) * 1024
    return values


def _cuda_process_memory_bytes():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    pid = str(os.getpid())
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 2 and fields[0] == pid:
            return int(fields[1]) * 1024 * 1024
    return None


class _AllocationAudit:
    def __init__(self, device):
        self.device = wp.get_device(device)
        self.allocator = self.device.get_allocator()
        self.original_allocate = self.allocator.allocate
        self.count = 0
        self.requested_bytes = 0
        self.sizes = Counter()
        self.reports = []
        self.tracker = wp.ScopedMemoryTracker("monolithic_e2e_steps", print=False, report_func=self.reports.append)
        self.before = None
        self.after = None
        self.driver_samples = []

    def _allocate(self, size):
        size = int(size)
        self.count += 1
        self.requested_bytes += size
        self.sizes[size] += 1
        return self.original_allocate(size)

    def __enter__(self):
        self.before = self.memory_snapshot()
        self.tracker.__enter__()
        self.allocator.allocate = self._allocate
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.allocator.allocate = self.original_allocate
        self.tracker.__exit__(exc_type, exc_value, traceback)
        self.tracker.report(max_items=50)
        self.after = self.memory_snapshot()

    def sample_driver(self):
        value = _cuda_process_memory_bytes() if self.device.is_cuda else _rss_bytes()["vmrss_bytes"]
        if value is not None:
            self.driver_samples.append(value)

    def memory_snapshot(self):
        result = _rss_bytes()
        result["process_driver_bytes"] = _cuda_process_memory_bytes() if self.device.is_cuda else result["vmrss_bytes"]
        if self.device.is_cuda and self.device.is_mempool_supported:
            result.update(
                mempool_current_bytes=wp.get_mempool_used_mem_current(self.device),
                mempool_lifetime_high_bytes=wp.get_mempool_used_mem_high(self.device),
            )
        return result

    def result(self, matrix_assembly_count):
        one_time_global_bsr_storage = 4
        sparse_builder_per_assembly = 4
        expected = one_time_global_bsr_storage + sparse_builder_per_assembly * matrix_assembly_count
        return {
            "python_device_allocator_count": self.count,
            "python_device_allocator_requested_bytes": self.requested_bytes,
            "python_device_allocator_size_histogram": {str(key): value for key, value in sorted(self.sizes.items())},
            "matrix_assembly_count": matrix_assembly_count,
            "expected_one_time_global_bsr_storage_allocations": one_time_global_bsr_storage,
            "expected_sparse_builder_allocations_per_assembly": sparse_builder_per_assembly,
            "expected_total_python_device_allocations": expected,
            "unexpected_python_device_allocations": self.count - expected,
            "allocation_gate": "PASS" if self.count == expected else "FAIL",
            "native_tracker_report": "\n".join(self.reports),
            "memory_before": self.before,
            "memory_after": self.after,
            "sampled_process_or_driver_peak_bytes": max(self.driver_samples) if self.driver_samples else None,
            "peak_semantics": (
                "nvidia-smi per-process used-memory sampled after synchronized steps"
                if self.device.is_cuda
                else "Linux current RSS sampled after synchronized steps; VmHWM is also recorded"
            ),
        }


class _StageProfile:
    def __init__(self, solver, variant):
        self.solver = solver
        self.variant = variant
        self.samples = {
            name: []
            for name in (
                "collision_candidate",
                "collision_final",
                "contact_assembly_current",
                "contact_evaluation_trial",
                "contact_evaluation_final",
                "bsr_build",
                "preconditioner_setup_factor",
                "preconditioner_q_setup",
                "preconditioner_x_setup",
                "preconditioner_q_factor",
                "preconditioner_x_factor",
                "preconditioner_q_apply",
                "preconditioner_x_apply",
                "pcg",
                "current_evaluation_overlapping_total",
                "trial_evaluation_overlapping_total",
            )
        }
        self.stack = ExitStack()
        self._actor_operator = solver._linear.preconditioner
        self._diagonal = None

    def _timed(self, name, function):
        def wrapped(*args, **kwargs):
            wp.synchronize_device(self.solver.model.device)
            start = time.perf_counter()
            result = function(*args, **kwargs)
            wp.synchronize_device(self.solver.model.device)
            self.samples[name].append((time.perf_counter() - start) * 1000.0)
            return result

        return wrapped

    def __enter__(self):
        try:
            return self._install()
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def _install(self):
        solver, workspace = self.solver, self.solver._linear
        original_collide = solver._collide
        original_launch = wp.launch
        kernel_stages = {
            linear._assemble_monolithic_q_preconditioner: "preconditioner_q_setup",
            linear._assemble_monolithic_x_preconditioner: "preconditioner_x_setup",
            linear._factor_monolithic_q_cholesky: "preconditioner_q_factor",
            linear._factor_monolithic_x_cholesky: "preconditioner_x_factor",
            linear._apply_q_preconditioner: "preconditioner_q_apply",
            linear._apply_x_preconditioner: "preconditioner_x_apply",
        }

        def launch(kernel, *args, **kwargs):
            stage = kernel_stages.get(kernel)
            if stage is None:
                return original_launch(kernel, *args, **kwargs)
            return self._timed(stage, original_launch)(kernel, *args, **kwargs)

        self.stack.enter_context(patch.object(wp, "launch", side_effect=launch))

        def collide(state, contacts):
            name = "collision_final" if contacts is solver.contacts else "collision_candidate"
            return self._timed(name, original_collide)(state, contacts)

        self.stack.enter_context(patch.object(solver, "_collide", side_effect=collide))
        self.stack.enter_context(
            patch.object(
                solver,
                "_evaluate_current",
                side_effect=self._timed("current_evaluation_overlapping_total", solver._evaluate_current),
            )
        )
        self.stack.enter_context(
            patch.object(
                solver,
                "_evaluate_trial",
                side_effect=self._timed("trial_evaluation_overlapping_total", solver._evaluate_trial),
            )
        )
        for function_name, stage in (
            ("assemble_current_contacts", "contact_assembly_current"),
            ("evaluate_trial_contacts", "contact_evaluation_trial"),
            ("evaluate_final_contacts", "contact_evaluation_final"),
        ):
            original = getattr(solver_module, function_name)
            if self.variant == "cross_disabled" and function_name == "assemble_current_contacts":

                def without_cross(*args, _original=original, **kwargs):
                    result = _original(*args, **kwargs)
                    triplets = args[6].global_scalar_triplets
                    wp.launch(
                        _zero_qx_xq_triplets,
                        triplets.values.size,
                        [triplets.rows, triplets.columns, triplets.values, solver._layout.q_dof_count],
                        device=solver.model.device,
                    )
                    return result

                original = without_cross
            self.stack.enter_context(
                patch.object(solver_module, function_name, side_effect=self._timed(stage, original))
            )
        self.stack.enter_context(
            patch.object(
                workspace,
                "finalize_assembly",
                side_effect=self._timed("bsr_build", workspace.finalize_assembly),
            )
        )
        if self.variant == "diagonal":
            self._diagonal = _DiagonalInverse(workspace)
            workspace.preconditioner = self._diagonal.operator

            def factor_diagonal(*, generation, pivot_tolerance):
                if (
                    not workspace._current(generation)
                    or not workspace._scaling_valid
                    or not math.isfinite(pivot_tolerance)
                    or pivot_tolerance < 0.0
                ):
                    return linear.MonolithicLinearStatus.INVALID_ARGUMENT
                workspace.factor_setup_count += 1
                try:
                    self._diagonal.setup()
                except ValueError:
                    workspace._factor_valid = False
                    workspace.factor_failure_count += 1
                    workspace.factor_last_status = linear.MonolithicLinearStatus.NON_POSITIVE_PIVOT
                    return workspace.factor_last_status
                workspace._factor_valid = True
                workspace.factor_last_status = linear.MonolithicLinearStatus.SUCCESS
                return workspace.factor_last_status

            factor = factor_diagonal
        else:
            factor = workspace.factor_actor_preconditioner
        self.stack.enter_context(
            patch.object(
                workspace,
                "factor_actor_preconditioner",
                side_effect=self._timed("preconditioner_setup_factor", factor),
            )
        )
        self.stack.enter_context(
            patch.object(workspace, "solve_pcg", side_effect=self._timed("pcg", workspace.solve_pcg))
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.stack.close()
        finally:
            self.solver._linear.preconditioner = self._actor_operator

    def summary(self):
        return {
            name: {"calls": len(values), "percentiles_ms": _percentiles(values), "samples_ms": values}
            for name, values in self.samples.items()
        }


def _launch_p1q3(model, state, pipeline, contacts):
    contacts.clear(bump_generation=True)
    contacts.soft_contact_tids.fill_(-1)
    pipeline._status.zero_()
    wp.launch(
        create_monolithic_p1q3_face_contacts,
        pipeline.soft_contact_pair_count,
        [
            pipeline._face_pairs,
            state.particle_q,
            model.tri_indices,
            model.shape_body,
            model.shape_type,
            model.shape_flags,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_margin,
            pipeline.r_soft,
            pipeline._soft_contact_gap,
            pipeline.soft_contact_max,
            contacts.soft_contact_count,
            contacts.soft_contact_tids,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
            pipeline._status,
            False,  # Preserve this profiler's original narrow-phase-only measurement.
            pipeline._bounds.face_lo,
            pipeline._bounds.face_hi,
            pipeline._bounds.shape_lo,
            pipeline._bounds.shape_hi,
            pipeline._bounds.aggregate,
            pipeline._query_counts,
        ],
        device=model.device,
    )


def _launch_adaptive_face(model, state, pipeline, contacts):
    contacts.clear(bump_generation=True)
    contacts.soft_contact_tids.fill_(-1)
    wp.launch(
        create_soft_face_contacts,
        pipeline.soft_contact_pair_count,
        [
            pipeline._face_pairs,
            state.particle_q,
            model.particle_radius,
            model.tri_indices,
            model.shape_body,
            model.shape_type,
            model.shape_flags,
            model.shape_transform,
            model.shape_scale,
            state.body_q,
            model._shape_sdf_index,
            model._texture_sdf_data,
            model.shape_margin,
            SDF_FACE_ITERS,
            SDF_LS_ITERS,
            pipeline._soft_contact_gap,
            0,
            pipeline.soft_contact_max,
            contacts.soft_contact_count,
            contacts.soft_contact_tids,
            contacts.soft_contact_particle,
            contacts.soft_contact_indices,
            contacts.soft_contact_barycentric,
            contacts.soft_contact_shape,
            contacts.soft_contact_body_pos,
            contacts.soft_contact_body_vel,
            contacts.soft_contact_normal,
        ],
        device=model.device,
    )


def _collision_asset(device, asset, medium_level):
    if asset == "formal_single_finger":
        fixture = load_fixture(FROZEN_FIXTURE)
        model, state, _control, solver = build_scene(fixture, device=device)
        manifest = {
            "fixture_id": fixture["fixture_id"],
            "fixture_sha256": hashlib.sha256(
                json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
    else:
        model, state, solver, source = build_refined_scene(device, medium_level, "plane")
        manifest = {
            "fixture_id": f"refined_tet_level_{medium_level}",
            "source_manifest_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        }
    return model, state, solver.collision_pipeline, manifest


def profile_collision_kernels(device, *, repeats=30, medium_level=3):
    """Compare face-only adaptive optimization and P1Q3 on identical pairs."""
    if repeats < 2:
        raise ValueError("repeats must be at least two")
    rows = []
    for asset in ("formal_single_finger", f"refined_tet_level_{medium_level}"):
        model, state, pipeline, manifest = _collision_asset(
            device, "formal_single_finger" if asset == "formal_single_finger" else "medium", medium_level
        )
        contacts = pipeline.contacts()
        methods = {}
        for name, launch in (("p1q3", _launch_p1q3), ("adaptive_face", _launch_adaptive_face)):
            launch(model, state, pipeline, contacts)
            wp.synchronize_device(model.device)
            samples, counts = [], []
            for _ in range(repeats):
                wp.synchronize_device(model.device)
                start = time.perf_counter()
                launch(model, state, pipeline, contacts)
                wp.synchronize_device(model.device)
                samples.append((time.perf_counter() - start) * 1000.0)
                counts.append(int(contacts.soft_contact_count.numpy()[0]))
            if len(set(counts)) != 1:
                raise RuntimeError(f"Nondeterministic contact count for {asset}/{name}: {counts}")
            methods[name] = {
                "record_count": counts[0],
                "samples_ms": samples,
                "percentiles_ms": _percentiles(samples),
            }
        rows.append(
            {
                "asset": asset,
                "device": str(model.device),
                "particle_count": model.particle_count,
                "tet_count": model.tet_count,
                "boundary_face_count": model.tri_count,
                "face_pair_count": pipeline.soft_contact_pair_count,
                "near_field_sdf_queries_per_pair": {
                    "p1q3": 4,
                    "adaptive_face": 2 + SDF_FACE_ITERS * (SDF_LS_ITERS + 4),
                    "basis": "one centroid cull plus the statically counted eval_shape_sdf calls in each near-field path",
                },
                **manifest,
                **methods,
            }
        )
    return rows


def _normal_fixture(stiffness=None):
    fixture = copy.deepcopy(load_fixture(FROZEN_FIXTURE if stiffness is None else DEFAULT_FIXTURE))
    if stiffness is not None:
        fixture["contact"]["stiffness_n_m3"] = float(stiffness)
    return fixture


def run_profiled_normal(
    device,
    *,
    variant="actor_block",
    substeps=1000,
    stiffness=None,
    allocation_audit=False,
    profile_stages=True,
):
    """Run one immutable normal trajectory with an explicitly named inverse/cross mode."""
    if variant not in ("actor_block", "diagonal", "cross_disabled"):
        raise ValueError(f"Unknown profile variant: {variant}")
    if not profile_stages and variant != "actor_block":
        raise ValueError("Negative controls require their stage injection; use actor_block for uninstrumented timing")
    fixture = _normal_fixture(stiffness)
    model, state, control, solver = build_scene(fixture, device=device)
    timer = _StageProfile(solver, variant) if profile_stages else nullcontext()
    audit = _AllocationAudit(device) if allocation_audit else nullcontext()
    records, step_samples, matrix_assemblies = [], [], 0
    with timer, audit:
        for step in range(substeps):
            time_s = (step + 1) * fixture["dt_s"]
            command = command_at(time_s, fixture)
            drive = fixture["drive"]
            force = drive["kp_n_m"] * (command[0] - float(state.joint_q.numpy()[0])) + drive["kd_n_s_m"] * (
                command[1] - float(state.joint_qd.numpy()[0])
            )
            control.joint_f.assign(np.asarray([force], dtype=np.float32))
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            solver.step(state, state, control, None, fixture["dt_s"])
            wp.synchronize_device(model.device)
            step_samples.append((time.perf_counter() - start) * 1000.0)
            record = measure_normal(model, state, solver, fixture, step, time_s, command, float(np.float32(force)))
            record["active_sample_count"] = solver.last_stats.active_sample_count
            records.append(record)
            matrix_assemblies += solver.last_stats.matrix_assembly_count
            if allocation_audit and (step % 25 == 0 or step + 1 == substeps):
                audit.sample_driver()
    assessment = assess_run(records, fixture)
    records = _finite_json(records)
    serialized = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    stages = timer.summary() if profile_stages else {}
    result = {
        "asset": fixture["fixture_id"],
        "fixture": fixture,
        "records": records,
        "diagnostics": summarize_records(records),
        "dt_s": fixture["dt_s"],
        "joint_count": model.joint_count,
        "particle_count": model.particle_count,
        "tet_count": model.tet_count,
        "scalar_dof_count": solver._layout.scalar_dof_count,
        "measurement_mode": {"stage_instrumentation": profile_stages, "allocation_tracker": allocation_audit},
        "fixture_sha256": hashlib.sha256(
            json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "device": str(model.device),
        "variant": variant,
        "substeps": substeps,
        "stiffness_n_m3": fixture["contact"]["stiffness_n_m3"],
        "step_ms_samples": step_samples,
        "percentiles_ms": {"whole_step": _percentiles(step_samples)},
        "stages": stages,
        "assessment": assessment,
        "all_finite": all(record["finite_state"] for record in records),
        "status_counts": dict(Counter(record["convergence_status"] for record in records)),
        "matrix_assembly_count": matrix_assemblies,
        "total_linear_iterations": sum(record["linear_iterations"] for record in records),
        "nonlinear_iterations": _percentiles([record["nonlinear_iterations"] for record in records]),
        "linear_iterations": _percentiles([record["linear_iterations"] for record in records]),
        "response": {
            "final_joint_q_m": records[-1]["joint_q"][0] if records else None,
            "final_node_positions_m": records[-1]["node_positions_m"] if records else None,
            "peak_force_n": assessment["peak_force_n"],
            "settled_force_n": assessment["settled_force_n"],
            "secant_stiffness_n_m": assessment["secant_stiffness_n_m"],
            "curve_work_j": assessment["curve_work_j"],
            "soft_response_m": records[-1]["soft_response_m"] if records else None,
            "maximum_penetration_m": max((record["penetration_m"] for record in records), default=None),
            "minimum_det_f": min((record["min_det_f"] for record in records), default=None),
        },
        "trajectory_sha256": hashlib.sha256(serialized).hexdigest(),
        "linear_workspace_memory": _memory(solver._linear),
        "allocation_audit": audit.result(matrix_assemblies) if allocation_audit else None,
    }
    result["percentiles_ms"].update(
        {name: value["percentiles_ms"] for name, value in stages.items() if value["percentiles_ms"] is not None}
    )
    return result


def run_profiled_medium(device, *, level=3, substeps=100, profile_stages=True):
    """Profile a 512-tet default medium fixture without changing its physical load."""
    model, state, solver, manifest = build_refined_scene(device, level, "plane")
    target, control = model.state(), model.control()
    control.joint_f.assign([0.1])
    timer = _StageProfile(solver, "actor_block") if profile_stages else nullcontext()
    samples, status, active, linear_iterations, records = [], [], [], [], []
    with timer:
        for step in range(substeps):
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            solver.step(state, target, control, None, 0.001)
            wp.synchronize_device(model.device)
            samples.append((time.perf_counter() - start) * 1000.0)
            status.append(solver.last_stats.status.value)
            active.append(solver.last_stats.active_sample_count)
            linear_iterations.append(solver.last_stats.linear_iterations)
            stats = solver.last_stats
            positions, joints = target.particle_q.numpy(), target.joint_q.numpy()
            records.append(
                {
                    "step": step,
                    "time_s": (step + 1) * 0.001,
                    "node_positions_m": positions.tolist(),
                    "joint_q": joints.tolist(),
                    "link_xform": target.body_q.numpy().tolist(),
                    "convergence_status": stats.status.value,
                    "converged": stats.converged,
                    "rolled_back": stats.rolled_back,
                    "finite_state": bool(
                        np.isfinite(positions).all()
                        and np.isfinite(joints).all()
                        and np.isfinite(target.particle_qd.numpy()).all()
                        and np.isfinite(target.body_q.numpy()).all()
                        and np.isfinite(target.body_qd.numpy()).all()
                        and np.isfinite(target.joint_qd.numpy()).all()
                    ),
                    "stats": {field.name: getattr(stats, field.name) for field in fields(stats)},
                }
            )
            state, target = target, state
    records = _json_value(records)
    stages = timer.summary() if profile_stages else {}
    return {
        "asset": f"refined_tet_level_{level}",
        "source_manifest": manifest,
        "records": records,
        "diagnostics": summarize_records(records),
        "dt_s": 0.001,
        "joint_count": model.joint_count,
        "measurement_mode": {"stage_instrumentation": profile_stages, "allocation_tracker": False},
        "source_manifest_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
        "device": str(model.device),
        "substeps": substeps,
        "particle_count": model.particle_count,
        "tet_count": model.tet_count,
        "boundary_face_count": model.tri_count,
        "scalar_dof_count": solver._layout.scalar_dof_count,
        "status_counts": dict(Counter(status)),
        "active_sample_count": _percentiles(active),
        "linear_iterations": _percentiles(linear_iterations),
        "step_ms_samples": samples,
        "percentiles_ms": {
            "whole_step": _percentiles(samples),
            **{name: value["percentiles_ms"] for name, value in stages.items() if value["percentiles_ms"]},
        },
        "stages": stages,
        "linear_workspace_memory": _memory(solver._linear),
    }


def _select_stiffness_lower_bound(rows):
    """Return the first bracketed PASS; missing evidence fails closed."""
    if not rows or any(row.get("physical_numerical_gate") not in ("PASS", "FAIL") for row in rows):
        return None
    ordered = sorted(rows, key=lambda row: row["stiffness_n_m3"])
    passed = [row for row in ordered if row["physical_numerical_gate"] == "PASS"]
    if not passed:
        return None
    candidate = passed[0]
    lower = [row for row in ordered if row["stiffness_n_m3"] < candidate["stiffness_n_m3"]]
    if not lower or not any(row["physical_numerical_gate"] == "FAIL" for row in lower):
        return None
    return float(candidate["stiffness_n_m3"])


def profile_stiffness(device, values, *, substeps=1000):
    rows = []
    for stiffness in sorted(set(values)):
        result = run_profiled_normal(
            device,
            variant="actor_block",
            substeps=substeps,
            stiffness=stiffness,
            profile_stages=False,
        )
        response = result["response"]
        passed = (
            result["assessment"]["e2e_numerical_pass"]
            and response["secant_stiffness_n_m"] is not None
            and response["peak_force_n"] is not None
        )
        rows.append(
            {
                "stiffness_n_m3": stiffness,
                "physical_numerical_gate": "PASS" if passed else "FAIL",
                "assessment": result["assessment"],
                "response": response,
                "status_counts": result["status_counts"],
                "linear_iterations": result["linear_iterations"],
                "whole_step_ms": result["percentiles_ms"]["whole_step"],
                "trajectory_sha256": result["trajectory_sha256"],
                "records": result["records"],
                "fixture": result["fixture"],
                "diagnostics": result["diagnostics"],
            }
        )
    return {
        "device": str(wp.get_device(device)),
        "values": rows,
        "fixture_bracketed_lower_bound_n_m3": _select_stiffness_lower_bound(rows),
        "scope": "Exact frozen single-finger fixture only; not a material- or asset-independent contact stiffness",
    }


def _response_difference(reference, candidate):
    scalar = {}
    for name in (
        "final_joint_q_m",
        "peak_force_n",
        "settled_force_n",
        "secant_stiffness_n_m",
        "curve_work_j",
        "soft_response_m",
        "maximum_penetration_m",
        "minimum_det_f",
    ):
        a, b = reference[name], candidate[name]
        scalar[name] = {
            "absolute": None if a is None or b is None else abs(b - a),
            "relative": None if a is None or b is None else abs(b - a) / max(abs(a), 1.0e-30),
        }
    a = np.asarray(reference["final_node_positions_m"])
    b = np.asarray(candidate["final_node_positions_m"])
    return {"scalars": scalar, "final_node_position_l2_m": float(np.linalg.norm(b - a))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", action="append", help="Repeat; default cpu and available CUDA devices")
    parser.add_argument("--collision-repeats", type=int, default=30)
    parser.add_argument("--normal-substeps", type=int, default=1000)
    parser.add_argument("--medium-substeps", type=int, default=100)
    parser.add_argument("--medium-level", type=int, default=3)
    parser.add_argument(
        "--stiffness-values",
        type=float,
        nargs="+",
        default=(1.0e5, 2.0e5, 5.0e5, 1.0e6, 2.0e6, 1.0e7),
    )
    args = parser.parse_args()
    if args.collision_repeats < 2 or min(args.normal_substeps, args.medium_substeps) <= 0:
        parser.error("repeat and substep counts must be positive; collision repeats must be at least two")
    wp.init()
    devices = args.device or [str(device) for device in wp.get_devices()]
    output = args.output
    output.mkdir(parents=True, exist_ok=False)

    # Compile every injected path on a disposable prefix before measured fresh-state runs.
    for device in devices:
        profile_collision_kernels(device, repeats=2, medium_level=min(args.medium_level, 1))
        for variant in ("actor_block", "diagonal", "cross_disabled"):
            run_profiled_normal(device, variant=variant, substeps=2, profile_stages=True)
        run_profiled_medium(device, level=args.medium_level, substeps=2)

    collision = {
        device: profile_collision_kernels(device, repeats=args.collision_repeats, medium_level=args.medium_level)
        for device in devices
    }
    trajectories = {}
    for device in devices:
        variants = {
            variant: run_profiled_normal(
                device,
                variant=variant,
                substeps=args.normal_substeps,
                profile_stages=True,
            )
            for variant in ("actor_block", "diagonal", "cross_disabled")
        }
        variants["actor_block_uninstrumented"] = run_profiled_normal(
            device, substeps=args.normal_substeps, profile_stages=False
        )
        variants["actor_block_allocation_audit"] = run_profiled_normal(
            device,
            variant="actor_block",
            substeps=args.normal_substeps,
            allocation_audit=True,
            profile_stages=False,
        )
        for variant in ("diagonal", "cross_disabled"):
            variants[variant]["difference_from_actor_block"] = _response_difference(
                variants["actor_block"]["response"], variants[variant]["response"]
            )
        trajectories[device] = variants
    medium = {
        device: run_profiled_medium(device, level=args.medium_level, substeps=args.medium_substeps)
        for device in devices
    }

    medium_uninstrumented = {
        device: run_profiled_medium(
            device, level=args.medium_level, substeps=args.medium_substeps, profile_stages=False
        )
        for device in devices
    }

    calibration_device = "cpu" if "cpu" in devices else devices[0]
    stiffness = profile_stiffness(calibration_device, args.stiffness_values, substeps=args.normal_substeps)
    selected = stiffness["fixture_bracketed_lower_bound_n_m3"]
    validation = {}
    if selected is not None:
        ordered = sorted(set(args.stiffness_values))
        predecessor = max(value for value in ordered if value < selected)
        for device in devices:
            validation[device] = profile_stiffness(device, (predecessor, selected), substeps=args.normal_substeps)

    root = Path(__file__).resolve().parents[2]
    report = {
        "status": "MEASURED_WITH_EXPLICIT_GATES",
        "newton_sha": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "worktree_dirty_at_publication": bool(
            subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip()
        ),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "warp_version": wp.__version__,
        "platform": platform.platform(),
        "devices": [{"alias": device, "name": wp.get_device(device).name} for device in devices],
        "measurement": {
            "collision": collision,
            "formal_single_finger_variants": trajectories,
            "medium_mesh": medium,
            "medium_mesh_uninstrumented": medium_uninstrumented,
            "stiffness_calibration": stiffness,
            "stiffness_cross_device_validation": validation,
        },
        "interpretation": {
            "production": "actor_block with complete qx/xq blocks",
            "negative_controls": [
                "offline scalar diagonal inverse",
                "contact qx/xq tangent triplets zeroed after the same physical contact assembly",
            ],
            "timing": (
                "Synchronized host-wall p50/p95; component timers overlap current/trial and factor/PCG totals "
                "and must not be added. Actor_block_uninstrumented and medium_mesh_uninstrumented have no stage "
                "patches or allocation trackers; only whole solver.step boundaries are synchronized/timed. "
                "Controls and record serialization are outside whole-step timing."
            ),
            "memory": "Scoped native allocation peak plus sampled process/driver resident memory; no unmeasured value is represented as zero",
            "superdex": "Not executed and not an exit gate",
        },
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    report = _finite_json(report)
    hashes = _save_trajectories(report, output)
    report_path = output / "release_profile.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    hashes["release_profile.json"] = digest
    (output / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": digest, "stiffness_lower_bound": selected}, indent=2))


if __name__ == "__main__":
    main()
