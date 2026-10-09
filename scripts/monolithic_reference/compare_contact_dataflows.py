# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare complete legacy/pool contacts using authenticated native actor inputs.

Both algorithms run in the imported source version. Each starts from its own
fresh state and keeps its own configuration identity. No checkpoint is loaded
across algorithms. Contact audits run on separate instances outside step timing.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.articulation import _configuration_digest
from newton._src.solvers.monolithic.coupled_kernels import contact_forces
from newton._src.solvers.monolithic.linear_types import MonolithicLinearGeneration, MonolithicLinearStatus
from newton.examples.softbody.monolithic_recording import serial, state_arrays
from scripts.monolithic_reference.actor_recording import _semantic, compare_semantic_fields
from scripts.monolithic_reference.contact_full_oracle import evaluate_stream
from scripts.monolithic_reference.profiling import capture_count, provenance
from scripts.monolithic_reference.record_workload import apply_inputs, inputs
from scripts.monolithic_reference.replay_workload import acceptance_trace
from scripts.monolithic_reference.run_actor_coupling import CASE, load_case, model_arrays, roles
from scripts.monolithic_reference.workload_io import digest, load_arrays, open_bundle, write_json
from scripts.monolithic_reference.workload_objective import replay_objective
from scripts.monolithic_reference.workload_scene import build_scene, compare_arrays, set_control

SCENES = ("actor-coupling", "actor-coupling-before", "actor-coupling-permuted")
ALGORITHM_FIELDS = {"contact_dataflow", "contact_resources"}


def _exact(a, b):
    try:
        compare_arrays(
            {k: np.asarray(v) for k, v in a.items()}, {k: np.asarray(v) for k, v in b.items()}, rtol=0, atol=0
        )
    except (AssertionError, TypeError) as error:
        raise ValueError("Physical input, policy or sampling differs") from error


def verify_contract(left, right):
    """Permit only the two declared complete-pool fields after exact physical checks."""
    if left._contact_pool or not right._contact_pool or left._contact.legacy_count or right._contact.legacy_count:
        raise ValueError("Comparison requires complete mixed-actor legacy/pool algorithms")
    expected = np.frombuffer(b"complete-pool/fixed-tree/v1", dtype=np.uint8)
    if not np.array_equal(right._contact.dataflow_identity, expected):
        raise ValueError("Unreviewed contact algorithm")
    if left.precision.value != "fp64" or right.precision.value != "fp64":
        raise ValueError("Comparison requires native FP64")
    _exact(model_arrays(left.model), model_arrays(right.model))
    _exact(left._physics_values, right._physics_values)
    _exact(left._sdf_fields, right._sdf_fields)
    _exact(left._kinematics._source_values, right._kinematics._source_values)
    _exact(left._contact.values, right._contact.values)
    _exact(left._contact.actor_samples.fields, right._contact.actor_samples.fields)
    _exact(left._actors.maps, right._actors.maps)
    if (
        left._actors.layout_payload() != right._actors.layout_payload()
        or left._layout != right._layout
        or left._config != right._config
        or left._joint_terms.implicit_pd != right._joint_terms.implicit_pd
        or left._joint_terms.allowed_properties != right._joint_terms.allowed_properties
        or left.pcg_mode != right.pcg_mode
        or left.use_cuda_graph != right.use_cuda_graph
        or left.model.device != right.model.device
        or left._contact.count != right._contact.count
        or left._contact.parameters != right._contact.parameters
        or left._contact.contact_model != right._contact.contact_model
        or left._contact.alignment_policy != right._contact.alignment_policy
    ):
        raise ValueError("Actor layout or numerical execution policy differs")
    _exact({"joint_parameters": left._joint_parameters_host}, {"joint_parameters": right._joint_parameters_host})
    policies = (
        "newton_max_iterations",
        "line_search_max_iterations",
        "linear_max_iterations",
        "linear_tolerance",
        "linear_absolute_tolerance",
        "_globalization",
    )
    _exact({name: getattr(left, name) for name in policies}, {name: getattr(right, name) for name in policies})
    a, b = left._coupled_config, right._coupled_config
    if a.keys() & ALGORITHM_FIELDS or b.keys() - a.keys() != ALGORITHM_FIELDS or a.keys() - b.keys():
        raise ValueError("Unreviewed configuration difference")
    _exact(a, {name: value for name, value in b.items() if name not in ALGORITHM_FIELDS})
    _exact({"identity": b["contact_dataflow"]}, {"identity": expected})
    c = right._contact
    resources = np.array(
        [c.pool.capacity, c.pool.max_support, c._batch.batch_capacity, a["guard_capacity"][1]], np.int64
    )
    _exact({"resources": b["contact_resources"]}, {"resources": resources})
    # Bind the exact compared values, rather than asserting two different
    # aggregate configuration hashes happen to represent the same physics.
    common = {"model." + name: value for name, value in model_arrays(left.model).items()}
    for prefix, values in (("config", a), ("sampling", left._contact.actor_samples.fields)):
        common.update({prefix + "." + name: value for name, value in values.items()})
    common["joint_parameters"] = left._joint_parameters_host
    common["layout"] = np.frombuffer(left._actors.layout_payload(), dtype=np.uint8)
    common.update({"effective_policy." + name: value for name, value in dataclasses.asdict(left._config).items()})
    return {
        "physical_contract_sha256": _configuration_digest(common).hex(),
        "algorithm_differences": {
            name: {"legacy": None, "pooled": serial(b[name])} for name in sorted(ALGORITHM_FIELDS)
        },
        "config_digests": [run._kinematics.digest.hex() for run in (left, right)],
    }


def _physical(state):
    h = state._monolithic_coupled
    return {
        "q": state.joint_q.numpy().copy(),
        "x": state.particle_q.numpy().copy(),
        "com": h.rigid_com.numpy().copy(),
        "rotation": h.rigid_rotation.numpy().copy(),
    }


def _sparse_bsr(matrix):
    """Copy only the used scalar raw BSR prefix, retaining every stored coordinate."""
    if matrix.block_shape != (1, 1) or matrix.nrow != matrix.ncol:
        raise ValueError("Contact audit requires square scalar raw BSR")
    offsets = matrix.offsets.numpy()
    if (
        offsets.shape != (matrix.nrow + 1,)
        or offsets[0] != 0
        or np.any(np.diff(offsets.astype(np.int64)) < 0)
        or offsets[-1] > min(matrix.columns.size, matrix.values.size)
    ):
        raise ValueError("Invalid contact BSR offsets")
    count = int(offsets[-1])
    columns = matrix.columns[:count].numpy().astype(np.int64) if count else np.empty(0, dtype=np.int64)
    values = matrix.values[:count].numpy().reshape(-1).astype(np.float64) if count else np.empty(0)
    rows = np.repeat(np.arange(matrix.nrow, dtype=np.int64), np.diff(offsets))
    if (
        np.any(columns < 0)
        or np.any(columns >= matrix.ncol)
        or not np.isfinite(values).all()
        or np.any((rows[1:] == rows[:-1]) & (columns[1:] <= columns[:-1]))
    ):
        raise ValueError("Invalid contact BSR coordinates or values")
    return {"shape": (matrix.nrow, matrix.ncol), "row": rows, "column": columns, "value": values}


def _contact_at(run, frame, start, dt, *, sparse=False):
    """Evaluate native contact inputs on an audit instance, without transplanting history."""
    c, k = run._contact, run._kinematics
    state = run.create_state(dt=dt)
    h = state._monolithic_coupled

    # Only geometry operands are needed by contact. This scratch state is not
    # a continuation or publication and must never be passed to solver.step.
    def query(values, begin):
        state.joint_q.assign(values["q"])
        state.particle_q.assign(values["x"])
        h.rigid_com.assign(values["com"])
        h.rigid_rotation.assign(values["rotation"])
        k.evaluate(k.gather_q(state))
        method = c.begin_step if begin else c.evaluate
        method(
            state.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            rigid_com=h.rigid_com,
            rigid_rotation=h.rigid_rotation,
            **({} if begin else {"dt": dt}),
        )

    query(start, True)
    query(frame, False)
    if c.flags.numpy()[0] or c.start_flags.numpy()[0]:
        raise ValueError("Invalid native contact audit")
    if run._contact_pool and c.evaluation_diagnostics()["tainted"]:
        raise ValueError("Capacity overflow cannot qualify complete contacts")
    sequence = run._assembly_sequence + 1
    run._assembly_sequence = sequence
    generation = MonolithicLinearGeneration(1, sequence, c.evaluation_generation if run._contact_pool else 1, sequence)
    assembly = run._linear.begin_assembly(generation)
    c.assemble_tangent(assembly)
    result = run._linear.finalize_assembly(generation=generation)
    if result.status != MonolithicLinearStatus.SUCCESS:
        raise ValueError("Contact audit assembly failed")
    tangent = (
        _sparse_bsr(run._linear.k_global_scalar_bsr)
        if sparse
        else run._linear.densify_for_test(generation=generation).raw_matrix
    )
    pairs = []
    if run._contact_pool:
        pairs = c.publish(state)
    else:
        if c.legacy_count:
            wp.launch(
                contact_forces,
                c.legacy_count,
                [
                    c.current,
                    c.local_residual,
                    c.shapes,
                    c.bodies,
                    c.body_local,
                    c.triangles,
                    state.particle_q,
                    k.com,
                    k.dof_map,
                    h.contact_joint_force,
                    h.contact_particle_force,
                    h.contact_body_force,
                ],
                device=run.model.device,
            )
        if c.actor_contact is not None:
            pairs = c.actor_contact.publish(state, k)
    static = np.zeros((run.model.shape_count, 6))
    bodies = run.model.shape_body.numpy()
    for pair in pairs:
        for shape, wrench in zip((pair["source"], pair["target"]), pair["world_wrench"], strict=True):
            if shape >= 0 and bodies[shape] < 0:
                static[shape] += wrench
    return {
        "energy": c.energy.numpy().reshape(-1, 2).sum(axis=0),
        "residual": c.residual.numpy().copy(),
        "tangent": tangent,
        "particle_force": h.contact_particle_force.numpy().copy(),
        "body_wrench": h.contact_body_force.numpy().copy(),
        "joint_force": h.contact_joint_force.numpy().copy(),
        "static_wrench": static,
    }


def _check_contact(actual, full):
    errors = {}
    for name in ("energy", "residual", "particle_force", "body_wrench", "joint_force"):
        got, wanted = actual[name], full[name]
        np.testing.assert_allclose(got, wanted, rtol=1e-8, atol=1e-12 if name == "energy" else 1e-10, err_msg=name)
        errors[name] = float(np.max(np.abs(got - wanted), initial=0))
    error = np.linalg.norm(actual["tangent"] - full["tangent"]) / max(1, np.linalg.norm(full["tangent"]))
    if not np.isfinite(error) or error >= 1e-5:
        raise ValueError("Contact tangent violates the original full-oracle gate")
    errors["tangent_relative"] = float(error)
    return errors


def _check_contact_sparse(actual, full):
    """Use the original numerical gates on every raw sparse B coordinate and endpoint."""
    errors = {}
    for name in ("energy", "residual", "particle_force", "body_wrench", "joint_force", "static_wrench"):
        got, wanted = actual[name], full[name]
        np.testing.assert_allclose(got, wanted, rtol=1e-8, atol=1e-12 if name == "energy" else 1e-10, err_msg=name)
        errors[name] = float(np.max(np.abs(got - wanted), initial=0))
    left, right = actual["tangent"], full["tangent"]
    if left["shape"] != right["shape"] or left["shape"][0] >= 2**31:
        raise ValueError("Contact tangent shape differs or exceeds the scalar layout")
    size = left["shape"][1]
    keys = [b["row"] * size + b["column"] for b in (left, right)]
    combined = np.union1d(*keys)
    values = []
    for b, ids in zip((left, right), keys, strict=True):
        if (
            ids.size != b["value"].size
            or not np.isfinite(b["value"]).all()
            or np.any(b["row"] < 0)
            or np.any(b["row"] >= left["shape"][0])
            or np.any(b["column"] < 0)
            or np.any(b["column"] >= size)
            or np.any(ids[1:] <= ids[:-1])
        ):
            raise ValueError("Invalid sparse contact tangent")
        row = np.zeros(combined.size)
        row[np.searchsorted(combined, ids)] = b["value"]
        values.append(row)
    error = np.linalg.norm(values[0] - values[1]) / max(1, np.linalg.norm(values[1]))
    if not np.isfinite(error) or error >= 1e-5:
        raise ValueError("Contact tangent violates the original full-oracle gate")
    errors["tangent_relative"] = float(error)
    return errors


def audit_contact_state(run, frame, start, dt, *, mesh_fields=None):
    """Audit one complete native contact state, not nonlinear convergence or a trajectory.

    The caller supplies a separate audit solver and true physical-step start.
    No checkpoint/history is transferred; scratch is never passed to step.
    """
    c = run._contact
    stiffness, mu, velocity, width, radius = c.parameters
    full = evaluate_stream(
        run.model,
        run._actors,
        frame,
        start,
        c.contact_model,
        radius=radius,
        stiffness=stiffness,
        mu=mu,
        width=width,
        falloff=velocity * dt,
        mesh_fields=mesh_fields,
        alignment_policy=c.alignment_policy,
    )
    if full["full_count"] != c.count:
        raise ValueError("Independent full enumeration disagrees with the declared complete set")
    actual = _contact_at(run, frame, start, dt, sparse=True)
    return {
        "status": "PASS_LOCAL_COMPLETE_CONTACT_STATE",
        "full_count": full["full_count"],
        "full_contributing_count": full["full_contributing_count"],
        "full_key_sha256": full["full_key_sha256"],
        "state_sha256": _configuration_digest(frame).hex(),
        "step_start_sha256": _configuration_digest(start).hex(),
        "max_penetration_m": max(0, -full["min_gap"]),
        "native_B_coordinates": actual["tangent"]["value"].size,
        "full_B_coordinates": full["full"]["tangent"]["value"].size,
        "errors": _check_contact_sparse(actual, full["full"]),
    }


def _audit_pair(audits, frame, start, dt, label):
    left = audits[0]
    c = left._contact
    stiffness, mu, velocity, width, radius = c.parameters
    full = evaluate_stream(
        left.model,
        left._actors,
        frame,
        start,
        c.contact_model,
        radius=radius,
        stiffness=stiffness,
        mu=mu,
        width=width,
        falloff=velocity * dt,
        alignment_policy=c.alignment_policy,
    )
    if full["full_count"] != c.count:
        raise ValueError("Independent full enumeration disagrees with the declared complete set")
    result = []
    for algorithm, audit in zip(("legacy", "pooled"), audits, strict=True):
        actual = _contact_at(audit, frame, start, dt, sparse=True)
        result.append(
            {
                **label,
                "algorithm": algorithm,
                "full_count": full["full_count"],
                "state_sha256": _configuration_digest(frame).hex(),
                "step_start_sha256": _configuration_digest(start).hex(),
                "max_penetration_m": max(0, -full["min_gap"]),
                "errors": _check_contact_sparse(actual, full["full"]),
            }
        )
    return result


def compare(root, *, device="cuda:0"):
    """Authenticate a complete input window and advance independently under frozen gates."""
    root = Path(root)
    manifest, source, case = open_bundle(root), provenance(), load_case()
    if (
        manifest.get("scene") not in SCENES
        or manifest.get("contact_dataflow", "legacy") != "legacy"
        or manifest.get("source", {}).get("source_hashes") != source["source_hashes"]
        or manifest.get("precision") != "fp64"
        or manifest.get("pcg_mode") != "production"
        or manifest.get("comparison") != {"rtol": 2e-5, "atol": 2e-6}
        or type(manifest.get("graph")) is not bool
        or type(manifest.get("steps")) is not int
        or not 1 <= manifest["steps"] <= case["steps"]
        or manifest.get("windows") != [[1, manifest["steps"]]]
        or manifest.get("dt") != case["dt"]
        or manifest.get("device") != str(wp.get_device(device))
    ):
        raise ValueError("Unreviewed algorithm, source, input range or execution identity")
    needed = {"checkpoint-1.npz", "trace.jsonl"} | {
        f"{name}-{step}.npz" for step in range(1, manifest["steps"] + 1) for name in ("inputs", "state")
    }
    if not needed <= manifest["files"].keys():
        raise ValueError("Unauthenticated comparison payload")
    traces = [json.loads(line) for line in (root / "trace.jsonl").read_text().splitlines()]
    if [row["step"] for row in traces] != list(range(1, manifest["steps"] + 1)):
        raise ValueError("Incomplete comparison trace")
    routes = [
        build_scene(manifest["scene"], device=device, graph=manifest["graph"], contact_dataflow=name)
        for name in ("legacy", "pooled")
    ]
    left, right = (route[1] for route in routes)
    contract = verify_contract(left, right)
    if left._kinematics.digest.hex() != manifest.get("physical_digest"):
        raise ValueError("Recorded physical identity differs")
    restored = left.create_state(dt=manifest["dt"])
    left.load_checkpoint(restored, root / "checkpoint-1.npz")
    _exact(state_arrays(restored), state_arrays(routes[0][2]))
    initial = [state_arrays(route[2]) for route in routes]
    _exact(
        {k: v for k, v in initial[0].items() if k != "history.config_digest"},
        {k: v for k, v in initial[1].items() if k != "history.config_digest"},
    )
    # The audit solvers never advance trajectories, so contact-only assembly
    # cannot disturb live PCG workspaces, accepted states or graph bindings.
    audits = [
        build_scene(manifest["scene"], device=device, graph=manifest["graph"], contact_dataflow=name)[1]
        for name in ("legacy", "pooled")
    ]
    verify_contract(*audits)
    controls = [route[0].control() for route in routes]
    errors, work, same_state = {}, [], []
    metadata = {"roles": roles(left.model, left._actors)}
    for step in range(1, manifest["steps"] + 1):
        key = f"inputs-{step}.npz"
        values = load_arrays(root / key, manifest["array_schemas"][key])
        starts, frames, row = [], [], {"step": step, "algorithms": []}
        for index, ((model, run, state, samples), control) in enumerate(zip(routes, controls, strict=True)):
            set_control(control, samples, step)
            # This allowlisted recipe has no external-force or gravity changes.
            # Changed controls remain errors even if payload hashes are rewritten.
            _exact(values, inputs(model, state, control))
            apply_inputs(values, model, state, control)
            starts.append(_physical(state))
            captures = capture_count(run)
            wp.synchronize_device(model.device)
            begin = time.perf_counter()
            run.step(state, state, control, dt=manifest["dt"])
            wp.synchronize_device(model.device)
            seconds = time.perf_counter() - begin
            stats = run.last_stats
            if not stats.converged or stats.rolled_back or stats.variational.get("contact_pool", {}).get("tainted"):
                raise ValueError("Stopped, failed or capacity-tainted trajectory cannot qualify complete contacts")
            run._states.validate(state)
            frames.append(_physical(state))
            row["algorithms"].append(
                {
                    "dataflow": ("legacy", "pooled")[index],
                    "step_seconds": seconds,
                    "new_captures": capture_count(run) - captures,
                    "converged": stats.converged,
                    "newton": stats.nonlinear_iterations,
                    "pcg": stats.linear_iterations,
                    "line_search": stats.line_search_iterations,
                    "acceptance": acceptance_trace(run._variational["trials"]),
                }
            )
        recorded = f"state-{step}.npz"
        compare_arrays(
            state_arrays(routes[0][2]),
            load_arrays(root / recorded, manifest["array_schemas"][recorded]),
            **manifest["comparison"],
        )
        expected = traces[step - 1]
        for name in ("converged", "rolled_back", "nonlinear_iterations", "linear_iterations", "line_search_iterations"):
            if getattr(left.last_stats, name) != expected["stats"][name]:
                raise ValueError("Legacy baseline execution differs from its authenticated trace")
        if acceptance_trace(left._variational["trials"]) != acceptance_trace(expected["trials"]):
            raise ValueError("Legacy baseline acceptance differs from its authenticated trace")
        if not compare_semantic_fields(
            _semantic(state_arrays(routes[0][2]), metadata),
            _semantic(state_arrays(routes[1][2]), metadata),
            case["equivalence"],
            errors,
        ):
            raise ValueError("Independent trajectories violate the frozen actor envelope")
        for origin, (frame, start) in enumerate(zip(frames, starts, strict=True)):
            checked = _audit_pair(
                audits, frame, start, manifest["dt"], {"step": step, "state_origin": ("legacy", "pooled")[origin]}
            )
            if checked[0]["max_penetration_m"] > case["gates"]["max_penetration_m"]:
                raise ValueError("Independent trajectory violates the frozen penetration gate")
            if routes[origin][1].last_stats.min_det_f <= case["gates"]["min_det_f"]:
                raise ValueError("Independent trajectory violates the frozen determinant gate")
            same_state.extend(checked)
        work.append(row)
    for name in manifest["snapshots"]:
        if not {name + "/" + file for file in ("call.json", "input.npz", "output.npz")} <= manifest["files"].keys():
            raise ValueError("Unauthenticated hotspot payload")
        path = root / name
        call = json.loads((path / "call.json").read_text())
        if call["boundary"] != "objective":
            continue
        if (
            call["state"] not in ("_original", "_accepted", "_trial")
            or call["step"] not in range(1, manifest["steps"] + 1)
            or call["dt"] != manifest["dt"]
            or call["physical_digest"] != manifest["physical_digest"]
            or call.get("contact_dataflow", "legacy") != "legacy"
        ):
            raise ValueError("Hotspot physical identity differs from its bundle")
        arrays = load_arrays(path / "input.npz", call["input_schema"])
        expected = load_arrays(path / "output.npz", call["output_schema"])
        replay_objective(
            arrays, call, expected, scene=manifest["scene"], device=device, graph=manifest["graph"], repeat=1
        )

        def physical(arrays, prefix):
            return {
                "q": arrays[prefix + ".joint_q"],
                "x": arrays[prefix + ".particle_q"],
                "com": arrays[prefix + ".history.rigid_com"],
                "rotation": arrays[prefix + ".history.rigid_rotation"],
            }

        same_state.extend(
            _audit_pair(
                audits,
                physical(arrays, call["state"]),
                physical(arrays, "_original"),
                call["dt"],
                {
                    "step": call["step"],
                    "snapshot": name,
                    "state_owner": call["state"],
                    "state_origin": "recorded_objective",
                },
            )
        )
    return {
        "schema": "monolithic-contact-dataflow-comparison/1",
        "status": "PASS_LOCAL_COMPLETE_CONTACT",
        "qualification": "Local complete-contact comparison; stage E2E/nightly/performance qualification NOT_RUN",
        "source": source,
        "bundle_manifest_sha256": digest(root / "manifest.json"),
        "fixture_sha256": digest(CASE),
        "scene": manifest["scene"],
        "steps": manifest["steps"],
        "asset_sha256": case["tet_asset_sha256"],
        "device": str(left.model.device),
        "graph": manifest["graph"],
        **contract,
        "trajectory_limits": case["equivalence"],
        "trajectory_fields": errors,
        "same_state_checks": len(same_state),
        "same_state": same_state,
        "work": work,
        "timing": "Diagnostic synchronized full steps, including captures; audit/restore outside timing",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.bundle, device=args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "receipt.json", result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
