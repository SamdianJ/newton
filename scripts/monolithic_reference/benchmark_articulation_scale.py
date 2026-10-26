# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measure heterogeneous fixed-base trees under absent, sparse and denser contact.

This is a replay-warmed workload measurement, not a dexterous-task acceptance
gate. Runtime graph recaptures remain in the measured cost and are reported.
Run each case serially on the same idle GPU.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from scripts.monolithic_reference.benchmark_workload import check_gpu_idle
from scripts.monolithic_reference.profiling import capture_count, contact_pair_diagnostics, provenance

DT = 0.002


def build_model(count, contact, device):
    """Alternate one- and three-DoF trees without changing their local mass or control laws."""
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    amplitudes = []
    for tree in range(count):
        position = (0.2 * tree, 0, 0.1)
        if contact == "sparse":
            position = (0.019 * tree, 0, 0.1)
        elif contact == "dense":
            position = (0.017 * (tree % 4), 0.017 * (tree // 4), 0.1)
        root = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)))
        joints = [builder.add_joint_fixed(-1, root, parent_xform=wp.transform(position, wp.quat_identity()))]
        parent = root
        for dof in range(3 if tree % 2 else 1):
            body = builder.add_link(mass=0.2, inertia=wp.diag(wp.vec3(0.001)))
            method = builder.add_joint_revolute if dof == 2 else builder.add_joint_prismatic
            joints.append(
                method(
                    parent,
                    body,
                    axis=wp.vec3(np.eye(3)[dof]),
                    target_ke=300 if dof < 2 else 1,
                    target_kd=1 if dof < 2 else 0.01,
                    actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                    limit_ke=0,
                    limit_kd=0,
                    damping=0,
                    friction=0,
                )
            )
            amplitudes.append(0.0002 if dof < 2 else 0.003)
            parent = body
        builder.add_articulation(joints)
        builder.add_shape_box(
            body, hx=0.01, hy=0.01, hz=0.01, cfg=builder.ShapeConfig(density=0, has_particle_collision=False)
        )
    return builder.finalize(device=device), np.asarray(amplitudes)


def array_bytes(*owners):
    """Count the union of reachable project array ranges, excluding driver/library scratch."""
    seen, ranges = set(), {}

    def visit(value):
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, wp.array):
            if value.ptr and value.capacity:
                ranges.setdefault(str(value.device), []).append((value.ptr, value.ptr + value.capacity))
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif not isinstance(value, type):
            cls = type(value)
            if cls.__module__.startswith("newton.") or (
                cls.__module__.startswith("warp.") and ("sparse" in cls.__module__ or cls.__name__ == "StructInstance")
            ):
                for item in getattr(value, "__dict__", {}).values():
                    visit(item)
                for base in cls.__mro__:
                    for name in getattr(base, "__slots__", ()):
                        if name not in ("__dict__", "__weakref__"):
                            visit(getattr(value, name, None))

    visit(owners)
    total = 0
    for intervals in ranges.values():
        end = 0
        for lo, hi in sorted(intervals):
            total += max(0, hi - max(lo, end))
            end = max(end, hi)
    return total


def main():
    """Record warmed step timings, actual work, packed storage and source identity for one case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=int, choices=(1, 2, 4, 8), required=True)
    parser.add_argument("--contact", choices=("none", "sparse", "dense"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    device = wp.get_device(args.device)
    check_gpu_idle(device)
    model, amplitudes = build_model(args.trees, args.contact, device)
    run = SolverMonolithic(
        model,
        friction_velocity_scale=0.001,
        joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
        alignment_policy="mask",
        pcg_mode="production",
        use_cuda_graph=device.is_cuda,
        newton_max_iterations=80,
    )
    state, control = run.create_state(dt=DT), run.create_control()
    caches = {
        "solver": run._graphs,
        "contact_batch": run._contact._batch.graphs,
        "contact_pool": run._contact.pool.graphs,
        "contact_query": run._contact.query_graphs,
        "contact_reference": run._contact._batch.reference_rows.graphs,
        **{f"bsr_{i}": owner.graphs for i, owner in enumerate(run._linear._fixed_bsr or ())},
    }
    source = provenance()
    result = {
        "status": "RUNNING",
        "source": source,
        "trees": args.trees,
        "contact": args.contact,
        "dt": DT,
        "warm_steps": 20,
        "warm_window_steps": 40,
        "measured_steps": 40,
        "samples": [],
        "scope": "short workload scaling; not full task qualification; pair diagnostics disabled",
    }
    try:
        with contact_pair_diagnostics(run, enabled=False):
            checkpoint = args.output / "window-start.npz"
            for repeat in range(2):
                if repeat:
                    run.load_checkpoint(state, checkpoint)
                    evictions = {name: sum(cache.evictions.values()) for name, cache in caches.items()}
                for step in range(21 if repeat else 1, 61):
                    phase = 3 * step * DT + np.arange(len(amplitudes))
                    control.joint_target_q.assign(amplitudes * np.sin(phase))
                    control.joint_target_qd.assign(3 * amplitudes * np.cos(phase))
                    captures = capture_count(run)
                    wp.synchronize_device(device)
                    start = time.perf_counter()
                    run.step(state, state, control, dt=DT)
                    wp.synchronize_device(device)
                    seconds = time.perf_counter() - start
                    stats = run.last_stats
                    if not stats.converged or stats.rolled_back:
                        raise RuntimeError(f"Step {step}: {stats.status}: {stats.failure_reason}")
                    if not repeat and step == 20:
                        run.save_checkpoint(state, checkpoint)
                    if repeat:
                        result["samples"].append(
                            {
                                "seconds": seconds,
                                "newton": stats.nonlinear_iterations,
                                "pcg": stats.linear_iterations,
                                "line_search": stats.line_search_iterations,
                                "capture_affected": captures != capture_count(run),
                                "captures": capture_count(run) - captures,
                                "bsr_nnz": int(run._linear.k_global_scalar_bsr.offsets.numpy()[-1]),
                                "candidates": stats.variational["contact_spatial"]["candidates"],
                                "support_entries": stats.variational["contact_pool"]["support_count"],
                            }
                        )
        times = [row["seconds"] for row in result["samples"]]
        k, linear = run._kinematics, run._linear
        arms = run._articulation_actors
        expected_j = sum(len(a.body_indices) * a.dof_count for a in arms)
        expected_q = sum(a.dof_count**2 for a in arms)
        if k.jt.size != expected_j or linear._q_cholesky.size != expected_q:
            raise RuntimeError("Articulation storage is not packed per tree")
        result.update(
            status="MEASURED",
            capture_affected_steps=sum(row["capture_affected"] for row in result["samples"]),
            graph_captures=sum(row["captures"] for row in result["samples"]),
            graph_evictions={name: sum(cache.evictions.values()) - evictions[name] for name, cache in caches.items()},
            graph_cache_summaries={name: cache.summary() for name, cache in caches.items()},
            median_ms=1000 * float(np.median(times)),
            p95_ms=1000 * float(np.percentile(times, 95)),
            dofs=[a.dof_count for a in arms],
            jacobian_vector_entries=k.jt.size,
            internal_matrix_entries=linear.aq_actor_dense.size,
            cholesky_entries=linear._q_cholesky.size,
            full_contact_samples=run._contact.count,
            project_array_bytes=array_bytes(run, state, control),
            contact_buffer_bytes=sum(a.capacity for a in run._contact.buffers()),
        )
        if provenance()["source_hashes"] != source["source_hashes"]:
            raise RuntimeError("Source changed during measurement")
    except Exception as error:
        result.update(status="FAIL", error=repr(error))
        raise
    finally:
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(
                {key: result.get(key) for key in ("status", "trees", "contact", "median_ms", "p95_ms", "error")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
