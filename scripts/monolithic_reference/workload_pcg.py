# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Snapshot and replay the existing production PCG core with explicit inputs."""

import dataclasses
import time

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.linear_types import MonolithicLinearLayout
from newton._src.solvers.monolithic.pcg_common import group_policy
from newton._src.solvers.monolithic.pcg_production import MaskedPcgConfig, MaskedPcgPlan
from newton.examples.softbody.monolithic_recording import serial
from scripts.monolithic_reference.profiling import provenance
from scripts.monolithic_reference.workload_scene import compare_arrays


def pcg_input(workspace, rhs, y, *, config, warm_start, delta, **kwargs):
    """Capture live inputs before the production adapter mutates its outputs."""
    if workspace.pcg_mode != "production" or workspace._study_soft is not None or delta is None:
        raise ValueError("Only ordinary production PCG with explicit delta is supported")
    layout = workspace.layout
    matrix = workspace.k_global_scalar_bsr
    offsets = matrix.offsets.numpy().copy()
    count = int(offsets[-1])
    arrays = {
        "offsets": offsets,
        "columns": matrix.columns.numpy()[:count].copy(),
        "values": matrix.values.numpy()[:count].copy(),
        "scale": workspace.scale.numpy().copy(),
        "q_cholesky": workspace._q_cholesky.numpy().copy(),
        "x_cholesky": workspace._x_cholesky.numpy().copy(),
        "rhs_hat": rhs.numpy().copy(),
        "y": y.numpy().copy(),
        "delta": delta.numpy().copy(),
        "limits": np.asarray(group_policy(layout, config)[1], dtype=np.float32),
    }
    if layout.rigid_ranges:
        arrays.update(
            rigid_cholesky=workspace._rigid_cholesky.numpy().copy(), rigid_rows=workspace.rigid_rows.numpy().copy()
        )
    metadata = {
        "config": dataclasses.asdict(MaskedPcgConfig.from_monolithic(config)),
        "regularization": workspace._lambda,
        "zero_initial_guess": int(warm_start) == 0,
        "graph": workspace.use_cuda_graph,
        "storage_capacity": matrix.values.size,
        "generation": dataclasses.asdict(workspace._generation),
        "boundary": "production_pcg_core",
        "layout": serial(dataclasses.asdict(layout)),
    }
    if layout.rigid_ranges or layout.q_scalar_offset != 0 or layout.x_scalar_offset != layout.q_dof_count:
        metadata.update(adapter="actor-pcg/1", source_hashes=provenance()["source_hashes"])
    return arrays, metadata


def pcg_output(workspace, y, delta):
    """Include actual work and validity flags, independent of wall-time noise."""
    return {"y": y.numpy().copy(), "delta": delta.numpy().copy()}, serial(workspace._last_masked_pcg_result)


def _layout(arrays, metadata):
    if metadata is not None and "layout" in metadata:
        data = dict(metadata["layout"])
        for name in ("actor_ranges", "rigid_ranges"):
            data[name] = tuple(tuple(row) for row in data[name])
        layout = MonolithicLinearLayout(**data)
        for kind, start, count in (
            ("articulation", layout.q_scalar_offset, layout.q_dof_count),
            ("tet", layout.x_scalar_offset, 3 * layout.dynamic_particle_count),
        ):
            if [row for row in layout.actor_ranges if row[0] == kind] != [(kind, start, count)]:
                raise ValueError("PCG actor ranges differ from factor offsets")
        return layout
    if "rigid_rows" in arrays or "rigid_cholesky" in arrays:
        raise ValueError("Mixed PCG input requires an explicit actor layout")
    return MonolithicLinearLayout(arrays["q_cholesky"].shape[0], arrays["x_cholesky"].shape[0])


def validate_pcg(arrays, metadata=None):
    """Reject corrupted sparse structure and nonfinite live input."""
    required = {"offsets", "columns", "values", "scale", "q_cholesky", "x_cholesky", "rhs_hat", "y", "delta", "limits"}
    if not required <= arrays.keys() or arrays.keys() - required - {"rigid_cholesky", "rigid_rows"}:
        raise ValueError("PCG live-in fields differ")
    layout = _layout(arrays, metadata)
    n = layout.scalar_dof_count
    offsets, columns = arrays["offsets"], arrays["columns"]
    if offsets.dtype != np.int32 or offsets.shape != (n + 1,) or offsets[0] != 0:
        raise ValueError("Invalid PCG offsets")
    if np.any(np.diff(offsets.astype(np.int64)) < 0) or offsets[-1] != len(columns):
        raise ValueError("Invalid PCG active count")
    if columns.dtype != np.int32 or np.any(columns < 0) or np.any(columns >= n):
        raise ValueError("Invalid PCG columns")
    shapes = {
        "values": (len(columns),),
        "scale": (n,),
        "rhs_hat": (n,),
        "y": (n,),
        "delta": (n,),
        "q_cholesky": (layout.q_dof_count, layout.q_dof_count),
        "x_cholesky": (layout.dynamic_particle_count, 3, 3),
        "limits": (len(layout.groups),),
    }
    if layout.rigid_ranges:
        shapes.update(rigid_cholesky=(len(layout.rigid_ranges), 6, 6), rigid_rows=(len(layout.rigid_ranges), 6))
        rows = np.asarray([np.arange(start, start + 6) for _, start in layout.rigid_ranges], dtype=np.int32)
        if "rigid_rows" not in arrays or not np.array_equal(arrays["rigid_rows"], rows):
            raise ValueError("PCG rigid rows differ from actor layout")
    elif "rigid_rows" in arrays or "rigid_cholesky" in arrays:
        raise ValueError("Unexpected rigid PCG fields")
    for name, shape in shapes.items():
        dtype = np.int32 if name == "rigid_rows" else np.float32
        if name not in arrays or arrays[name].shape != shape or arrays[name].dtype != dtype:
            raise ValueError(f"Invalid PCG shape/dtype: {name}")
    for name, value in arrays.items():
        if name != "limits" and not np.isfinite(value).all():
            raise ValueError(f"Nonfinite PCG input: {name}")
    if np.any(np.isnan(arrays["limits"])) or np.any(arrays["limits"] <= 0) or np.any(arrays["scale"] <= 0):
        raise ValueError("Invalid PCG limits")
    return layout


def true_residual(arrays, y, regularization):
    """Evaluate scaled CSR residual independently on the host in float64."""
    scaled = arrays["scale"].astype(np.float64) * y
    product = np.zeros_like(scaled)
    for i, (start, end) in enumerate(zip(arrays["offsets"][:-1], arrays["offsets"][1:], strict=True)):
        product[i] = arrays["values"][start:end].astype(np.float64) @ scaled[arrays["columns"][start:end]]
    return arrays["rhs_hat"].astype(np.float64) - arrays["scale"] * product - regularization * y


def replay_pcg(arrays, metadata, expected, expected_result, *, device, repeat=5):
    """Restore every live input before warmup and each timed core invocation."""
    if repeat < 1:
        raise ValueError("repeat must be positive")
    layout = validate_pcg(arrays, metadata)
    if layout.rigid_ranges or layout.q_scalar_offset != 0 or layout.x_scalar_offset != layout.q_dof_count:
        if metadata.get("adapter") != "actor-pcg/1" or metadata.get("source_hashes") != provenance()["source_hashes"]:
            raise ValueError("Mixed PCG adapter or source identity differs")
    types = {"offsets": wp.int32, "columns": wp.int32, "x_cholesky": wp.mat33, "rigid_rows": wp.int32}
    buffers = {
        name: wp.array(value, dtype=types.get(name, wp.float32), device=device)
        for name, value in arrays.items()
        if name != "limits"
    }
    plan = MaskedPcgPlan(**buffers, layout=layout, config=MaskedPcgConfig(**metadata["config"]))
    times, restore_times = [], []
    for index in range(repeat + 1):
        start = time.perf_counter()
        for name, buffer in buffers.items():
            buffer.assign(arrays[name])
        plan.residual_norm_limits.assign(arrays["limits"])
        plan.set_solve_parameters(
            regularization=metadata["regularization"], zero_initial_guess=metadata["zero_initial_guess"]
        )
        if metadata["graph"] and plan.graph is None:
            plan.capture()
        wp.synchronize_device(plan.device)
        restore = time.perf_counter() - start
        start = time.perf_counter()
        if metadata["graph"]:
            plan.replay()
        else:
            plan.enqueue_solve()
        result = plan.collect_result()
        elapsed = time.perf_counter() - start
        output = {"y": plan.y.numpy().copy(), "delta": plan.delta.numpy().copy()}
        compare_arrays(output, expected)
        actual_result = serial(result)
        for key in (
            "status",
            "output_valid",
            "iterations",
            "iteration_slots",
            "true_residual_checks",
            "residual_replacements",
        ):
            if actual_result[key] != expected_result[key]:
                raise ValueError(f"PCG execution differs: {key}")
        residual = true_residual(arrays, output["y"], metadata["regularization"])
        reference = true_residual(arrays, expected["y"], metadata["regularization"])
        np.testing.assert_allclose(residual, reference, rtol=2e-5, atol=2e-6)
        if index:
            times.append(elapsed)
            restore_times.append(restore)
    return {
        "status": "PASS",
        "boundary": metadata["boundary"],
        "seconds": times,
        "restore_seconds": restore_times,
        "result": actual_result,
        "true_residual_norm": float(np.linalg.norm(residual)),
        "warmup_samples": 1,
    }
