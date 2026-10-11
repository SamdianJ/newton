# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Explicit P4 objective/assembly adapters; rebuild assets and process handles."""

import dataclasses
import time

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import serial
from scripts.monolithic_reference.profiling import SolverProfiler, capture_count, provenance
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays

# These are numerical owners, not a recursively serialized object graph.
OWNERS = ("_objective", "_kinematics", "_tet", "_contact", "_tet_assembly")
SOLVER_ARRAYS = ("_frozen_force", "_frozen_gravity", "_complete_r", "_diagonal", "_tet_r32")
HANDLES = {"table", "smooth_table"}
TOPOLOGY = {
    "_tet.ids",
    "_tet.mapping",
    "_tet_assembly.elastic_block_slots",
    "_contact.shapes",
    "_contact.mapping",
    "_contact.triangles",
    "_contact.types",
    "_contact.bodies",
    "_contact.sdf_indices",
}


def arrays_for(run):
    """Enumerate a bounded set of numerical buffers; exclude texture descriptors."""
    arrays = {"solver." + name: getattr(run, name) for name in SOLVER_ARRAYS}
    owners = {name: getattr(run, name) for name in OWNERS}
    if run._contact_pool:
        batch, pool = run._contact._batch, run._contact.pool
        owners.update({"contact.batch": batch, "contact.pool": pool})
        for name in ("all_samples", "samples", "support", "current", "start", "_start_chunk"):
            owners["contact.batch." + name] = getattr(batch, name)
        for name in ("candidates", "records", "_identity", "energy", "residual", "force"):
            owners["contact.pool." + name] = getattr(pool, name)
        for name, array in zip(("identities", "count", "base", "indices"), pool._stream, strict=True):
            arrays["contact.pool.stream." + name] = array
        if run._contact_spatial:
            selection = run._contact._spatial
            owners.update(
                {
                    "contact.spatial": selection,
                    "contact.spatial.sweep": selection.sweep,
                    "contact.spatial.query": selection.query,
                    "contact.reference": batch.reference_rows,
                }
            )
            for name, array in zip(("rows", "ids", "local", "values", "count"), batch.reference_inputs, strict=True):
                arrays["contact.reference.input." + name] = array
    else:
        owners.update({"contact.start": run._contact.start, "contact.current": run._contact.current})
    if not run._device_iteration or run._contact_pool:
        owners["_rigid"] = run._rigid
        actor_contact = run._contact.actor_contact
        if actor_contact is not None:
            owners.update(
                {
                    "actor_contact": actor_contact,
                    "actor_contact.start": actor_contact.start,
                    "actor_contact.current": actor_contact.current,
                }
            )
    for state_name in ("_original", "_accepted", "_trial"):
        state = getattr(run, state_name)
        owners[state_name] = state
        owners[state_name + ".history"] = state._monolithic_coupled
    for prefix, owner in owners.items():
        fields = (f.name for f in dataclasses.fields(owner)) if dataclasses.is_dataclass(owner) else vars(owner)
        for name in fields:
            value = getattr(owner, name)
            if isinstance(value, wp.array) and name not in HANDLES:
                # Struct arrays can contain process-specific device pointers.
                if hasattr(value.dtype, "vars"):
                    raise ValueError(f"Unsupported structured snapshot buffer: {prefix}.{name}")
                arrays[prefix + "." + name] = value
    return arrays


def capture_input(run, boundary, args, kwargs):
    """Save the exact iterate, frozen contact quantities and assembly dependencies."""
    state = args[0] if boundary == "objective" else run._current_state
    state_name = next((name for name in ("_original", "_accepted", "_trial") if state is getattr(run, name)), None)
    if state_name is None:
        raise ValueError("Unknown candidate state owner")
    metadata = {
        "boundary": boundary,
        "state": state_name,
        "dt": float(run._dt),
        "iteration": int(args[0]) if boundary == "assembly" else None,
        "build_tangent": kwargs.get("build_tangent", True),
        "device_result": kwargs.get("device_result", False),
        "step_generation": run._step_generation,
        "assembly_sequence": run._assembly_sequence,
        "contact_started": run._contact._batch._started if run._contact_pool else run._contact._started,
        "contact_evaluated": run._contact._evaluated,
        "physical_digest": run._kinematics.digest.hex(),
    }
    if not run._device_iteration or run._contact_pool:
        metadata.update(
            adapter="pool-objective/1" if run._contact_pool else "actor-objective/1",
            layout=serial(dataclasses.asdict(run._layout)),
            actor_maps=serial(run._actors.maps),
            rigid_has_torque=run._rigid.has_torque,
            source_hashes=provenance()["source_hashes"],
        )
    if run._contact_pool:
        c = run._contact
        metadata.update(
            contact_dataflow="pooled",
            pool={
                "started": c._batch._started,
                "candidate_generation": c.candidate_generation,
                "evaluation_generation": c.evaluation_generation,
                "admission_generation": c.evaluation_generation,
                "assembled_generation": c._assembled_generation,
                "tainted": c._tainted,
                "warned": c._warned,
            },
        )
    if run._contact_spatial:
        metadata["contact_dataflow"] = "spatial"
        metadata["spatial"] = {
            "candidate_capacity": c.candidate_capacity,
            "law_margin": c.law_margin,
            "selection_ready": c._selection_ready,
            "reference_mode": c._reference_mode,
            "refreshes": c._refreshes,
            "capacity_dropped": c._query_dropped,
            "warned": c._query_warned,
        }
    return {name: array.numpy().copy() for name, array in arrays_for(run).items()}, metadata


def capture_output(run, boundary, result):
    """Compare numerical outputs and the sparse operator, not capacity padding."""
    arrays = {
        "residual": run._complete_r.numpy().copy(),
        "tet_flags": run._tet.flags.numpy().copy(),
        "contact_flags": run._contact.flags.numpy().copy(),
        "contact_r": run._contact.residual.numpy().copy(),
    }
    if run._contact_pool:
        c, records = run._contact, run._contact.pool.records
        count = int(records.count.numpy()[0])
        if not 0 <= count <= c.pool.contact_capacity:
            raise ValueError("Invalid pooled output count")
        for name in (
            "accepted",
            "flags",
            "count",
            "used",
            "requested",
            "dropped",
            "support_count",
            "factor_count",
            "triplet_count",
        ):
            arrays["contact." + name] = getattr(records, name).numpy().copy()
        for name in ("identities", "source_slots", "pairs"):
            arrays["contact." + name] = getattr(records, name).numpy()[:count].copy()
        arrays["contact_force"] = c.pool.force.result.numpy().copy()
    else:
        arrays.update(
            contact_weights=run._contact.weights.numpy().copy(),
            contact_local_r=run._contact.local_residual.numpy().copy(),
        )
    if boundary == "objective":
        arrays.update(
            energy=np.asarray([result[0], result[2]]),
            returned_residual=result[1].numpy().copy() if isinstance(result[1], wp.array) else result[1].copy(),
            tangent=run._objective.tangent32.numpy().copy(),
            tet_parts=run._tet.parts.numpy().copy(),
            contact_energy=run._contact.energy.numpy().copy(),
        )
    else:
        for name, matrix in (("global", run._linear.k_global_scalar_bsr), ("internal", run._linear.ax_internal_bsr3)):
            offsets = matrix.offsets.numpy().copy()
            count = int(offsets[-1])
            arrays.update(
                {
                    name + ".offsets": offsets,
                    name + ".columns": matrix.columns.numpy()[:count].copy(),
                    name + ".values": matrix.values.numpy()[:count].copy(),
                }
            )
        arrays.update(
            scale=run._linear.scale.numpy().copy(),
            diagonal=run._diagonal.numpy().copy(),
            tet_failure=run._tet_assembly.failure_flags.numpy().copy(),
            actor_dense=run._linear.aq_actor_dense.numpy().copy(),
        )
    if not run._device_iteration or run._contact_pool:
        arrays.update(
            rigid_energy=run._rigid.energy.numpy().copy(),
            rigid_residual=run._rigid.residual.numpy().copy(),
            rigid_tangent=run._rigid.tangent.numpy().copy(),
        )
    return arrays


def compare_assembly(run, actual, expected):
    """Check parallel FP32 reductions against an independent FP64 triplet sum.

    Cancellation makes a relative tolerance on the reduced value unsuitable.
    Each reduction must lie within gamma_(n-1) * sum(abs(terms)); use the
    FP64 summation bound too, since the host reference is not exact arithmetic.
    """
    values = {"global.values", "internal.values"}
    compare_arrays(
        {k: v for k, v in actual.items() if k not in values}, {k: v for k, v in expected.items() if k not in values}
    )
    checks = {}
    for name, triplets in (("global", run._linear._global), ("internal", run._linear._internal)):
        key = name + ".values"
        got, wanted = actual[key], expected[key]
        if got.shape != wanted.shape or got.dtype != wanted.dtype:
            raise ValueError("Assembly output shape/dtype differs")
        count = int(triplets.count.numpy()[0])
        if not count:
            compare_arrays({key: got}, {key: wanted})
            continue
        indices = (
            triplets.rows.numpy()[:count].astype(np.int64) * triplets.column_count + triplets.columns.numpy()[:count]
        )
        order = np.argsort(indices)
        unique, first, counts = np.unique(indices[order], return_index=True, return_counts=True)
        terms = triplets.values.numpy()[:count][order].astype(np.float64)
        reference = np.add.reduceat(terms, first)
        magnitude = np.add.reduceat(np.abs(terms), first)
        additions = (counts - 1).reshape((-1,) + (1,) * (terms.ndim - 1))
        single = additions * (np.finfo(np.float32).eps / 2)
        double = additions * (np.finfo(np.float64).eps / 2)
        if np.any(single >= 1):
            raise ValueError("Reduction too large for a meaningful roundoff bound")
        bound = (single / (1 - single) + double / (1 - double)) * magnitude
        offsets = actual[name + ".offsets"]
        keys = (
            np.repeat(np.arange(len(offsets) - 1, dtype=np.int64), np.diff(offsets)) * triplets.column_count
            + actual[name + ".columns"]
        )
        positions = np.searchsorted(unique, keys)
        np.testing.assert_array_equal(unique[positions], keys)
        omitted = np.ones(len(unique), dtype=bool)
        omitted[positions] = False
        if np.any(np.abs(reference[omitted]) > bound[omitted]):
            raise AssertionError("Assembly omitted a nonzero triplet sum")
        reference, bound = reference[positions], bound[positions]
        for label, data in (("replayed", got), ("recorded", wanted)):
            if not np.all(np.abs(data.astype(np.float64) - reference) <= bound):
                raise AssertionError(f"{name} {label} BSR differs from the FP64 triplet sum beyond FP32 roundoff")
        checks[name] = {
            "raw_elementwise_mismatches": int(np.sum(~np.isclose(got, wanted, rtol=2e-5, atol=2e-6))),
            "max_roundoff_bound": float(np.max(bound, initial=0)),
            "max_reference_error": float(np.max(np.abs(got - reference), initial=0)),
        }
    return checks


def replay_objective(arrays, metadata, expected, *, scene, device, graph, repeat=5):
    """Reconstruct the scene once, restore input contents, and call the real boundary."""
    if metadata["boundary"] not in ("objective", "assembly") or repeat < 1:
        raise ValueError("Unsupported boundary or repeat count")
    if metadata["state"] not in ("_original", "_accepted", "_trial"):
        raise ValueError("Unsupported candidate state owner")
    model, run, state, _ = build_scene(
        scene,
        device=device,
        graph=graph,
        contact_dataflow=metadata.get("contact_dataflow", "legacy"),
        spatial={name: metadata.get("spatial", {}).get(name) for name in ("candidate_capacity", "law_margin")}
        if metadata.get("contact_dataflow") == "spatial"
        else None,
    )
    if run._kinematics.digest.hex() != metadata["physical_digest"]:
        raise ValueError("Hotspot physical identity differs")
    if not run._device_iteration or run._contact_pool:
        if (
            metadata.get("adapter") != ("pool-objective/1" if run._contact_pool else "actor-objective/1")
            or metadata.get("layout") != serial(dataclasses.asdict(run._layout))
            or metadata.get("actor_maps") != serial(run._actors.maps)
            or metadata.get("source_hashes") != provenance()["source_hashes"]
            or type(metadata.get("rigid_has_torque")) is not bool
        ):
            raise ValueError("Mixed hotspot adapter, source or layout identity differs")
    if run._contact_pool:
        pool = metadata.get("pool", {})
        if (
            type(pool) is not dict
            or set(pool)
            != {
                "started",
                "candidate_generation",
                "evaluation_generation",
                "admission_generation",
                "assembled_generation",
                "tainted",
                "warned",
            }
            or any(type(pool[name]) is not bool for name in ("started", "tainted", "warned"))
            or any(
                type(pool[name]) is not int or pool[name] < 0
                for name in ("candidate_generation", "evaluation_generation", "admission_generation")
            )
            or type(pool["assembled_generation"]) is not int
            or not -1 <= pool["assembled_generation"] <= pool["evaluation_generation"]
            or pool["admission_generation"] != pool["evaluation_generation"]
            or pool["started"] != metadata.get("contact_started")
            or type(metadata.get("contact_evaluated")) is not bool
            or (metadata["boundary"] == "assembly" and not metadata["contact_evaluated"])
        ):
            raise ValueError("Invalid pooled hotspot generations")
    for name in ("_original", "_accepted", "_trial"):
        setattr(run, name, run._states.copy(state))
    if run._contact_spatial:
        spatial = metadata["spatial"]
        if (
            set(spatial)
            != {
                "candidate_capacity",
                "law_margin",
                "selection_ready",
                "reference_mode",
                "refreshes",
                "capacity_dropped",
                "warned",
            }
            or any(type(spatial[n]) is not bool for n in ("selection_ready", "reference_mode", "warned"))
            or any(type(spatial[n]) is not int or spatial[n] < 0 for n in ("refreshes", "capacity_dropped"))
            or (not spatial["reference_mode"] and not spatial["selection_ready"])
        ):
            raise ValueError("Invalid spatial hotspot lifecycle")
    targets = arrays_for(run)
    if arrays.keys() != targets.keys():
        raise ValueError("Hotspot live-in schema changed")
    for name, target in targets.items():
        actual = target.numpy()
        if arrays[name].shape != actual.shape or arrays[name].dtype != actual.dtype:
            raise ValueError(f"Hotspot live-in shape/dtype differs: {name}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Nonfinite hotspot live-in: {name}")
        actor_contact = run._contact.actor_contact
        static_actor = actor_contact is not None and name in {"actor_contact." + key for key in actor_contact._bound}
        static_rigid = name in {
            "_rigid." + key for key in ("rows", "qs", "vs", "bodies", "mass", "second", "xp", "xc", "local_com")
        }
        static_pool = run._contact_pool and (
            name.startswith(("contact.batch.all_samples.", "contact.batch.support."))
            or name in {"_contact." + key for key in (*run._contact._batch.METADATA_ARRAYS, "_model_rows")}
        )
        spatial_mutable = {
            "q0",
            "x0",
            "poses0",
            "com0",
            "status",
            "count",
            "packet",
            "_ids",
            "_scratch_ids",
            "_indices",
            "_scratch_indices",
            "identities",
            "face_lo",
            "face_hi",
            "target_lo",
            "target_hi",
        }
        static_spatial = (
            run._contact_spatial
            and name.startswith("contact.spatial.")
            and (
                (name.count(".") == 2 and name.rsplit(".", 1)[1] not in spatial_mutable)
                or (
                    name.startswith("contact.spatial.sweep.")
                    and name.rsplit(".", 1)[1]
                    in {
                        "radii",
                        "incoming",
                        "parent_length",
                        "child_length",
                        "axis_length",
                    }
                )
                or (
                    name.startswith("contact.spatial.query.")
                    and name.rsplit(".", 1)[1]
                    in {
                        "face_sources",
                        "plane_targets",
                        "finite_targets",
                        "offsets",
                        "sources",
                    }
                )
            )
        )
        if (
            name in TOPOLOGY
            or static_actor
            or static_rigid
            or static_pool
            or static_spatial
            or (name.startswith("_kinematics.") and actual.dtype.kind in "iu")
        ):
            if not np.array_equal(arrays[name], actual):
                raise ValueError(f"Hotspot topology differs from reconstructed scene: {name}")
    run._dt = run._kinematics.timestep(metadata["dt"])
    run._step_generation = metadata["step_generation"]
    run._assembly_sequence = metadata["assembly_sequence"]
    run._metrics = {"matrix_assembly_count": 0}
    run._variational = {}
    times, restore_times, captures, checks = [], [], [], []
    for index in range(repeat + 1):
        start = time.perf_counter()
        for name, target in targets.items():
            target.assign(arrays[name])
        if not run._contact_pool:
            run._contact._started = metadata["contact_started"]
        run._contact._evaluated = metadata["contact_evaluated"]
        if run._contact_pool:
            c = run._contact
            c._batch._started = pool["started"]
            c.pool._stream_started = False
            c.candidate_generation = pool["candidate_generation"]
            c.evaluation_generation = pool["evaluation_generation"]
            c._assembled_generation = pool["assembled_generation"]
            c._tainted, c._warned = pool["tainted"], pool["warned"]
            # The fresh workspace has no previous append. New assembly tokens
            # still increase; recorded tokens/pointers are never restored.
            c._last_assembly = None
        if run._contact_spatial:
            c._selection_ready, c._reference_mode = spatial["selection_ready"], spatial["reference_mode"]
            if c._selection_ready:
                c._spatial.check(warn=False)
            c._refreshes, c._query_dropped, c._query_warned = (
                spatial["refreshes"],
                spatial["capacity_dropped"],
                spatial["warned"],
            )
        if not run._device_iteration or run._contact_pool:
            run._rigid.has_torque = metadata["rigid_has_torque"]
        run._current_state = getattr(run, metadata["state"])
        wp.synchronize_device(model.device)
        restore = time.perf_counter() - start
        before = capture_count(run)
        start = time.perf_counter()
        if metadata["boundary"] == "objective":
            result = run._evaluate(
                run._current_state,
                metadata["dt"],
                build_tangent=metadata["build_tangent"],
                device_result=metadata.get("device_result", False),
            )
        else:
            # Fresh assembly generations remain monotonic; never disable guards.
            result = run._assemble(metadata["iteration"], device_result=metadata.get("device_result", False))
        wp.synchronize_device(model.device)
        elapsed = time.perf_counter() - start
        output = capture_output(run, metadata["boundary"], result)
        if metadata["boundary"] == "assembly":
            checks.append(compare_assembly(run, output, expected))
        else:
            compare_arrays(output, expected)
        if index:
            times.append(elapsed)
            restore_times.append(restore)
            captures.append(before != capture_count(run))
    # Separate instrumented replay from the timing samples above.
    for name, target in targets.items():
        target.assign(arrays[name])
    with SolverProfiler(run) as profiler:
        if metadata["boundary"] == "objective":
            run._evaluate(
                run._current_state,
                metadata["dt"],
                build_tangent=metadata["build_tangent"],
                device_result=metadata.get("device_result", False),
            )
        else:
            run._assemble(metadata["iteration"], device_result=metadata.get("device_result", False))
    return {
        "status": "PASS",
        "boundary": metadata["boundary"],
        "seconds": times,
        "restore_seconds": restore_times,
        "capture_affected": captures,
        "profile": profiler.report(),
        "warmup_samples": 1,
        "assembly_reduction_checks": checks,
    }
