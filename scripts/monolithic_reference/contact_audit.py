# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit spatial contact states and actor recordings against the independent full oracle.

Constructor contracts, native contact evaluation, full-system comparisons and
recording comparisons for the pooled spatial dataflow; no acceptance is granted.
"""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from newton._src.solvers.monolithic.articulation import _configuration_digest
from newton._src.solvers.monolithic.linear_types import MonolithicLinearGeneration, MonolithicLinearStatus
from newton.examples.softbody.monolithic_recording import serial
from scripts.monolithic_reference.actor_recording import _semantic, compare_semantic_fields
from scripts.monolithic_reference.contact_full_oracle import evaluate_stream
from scripts.monolithic_reference.profiling import provenance
from scripts.monolithic_reference.run_actor_coupling import load_case, model_arrays
from scripts.monolithic_reference.workload_scene import compare_arrays

ALGORITHM_FIELDS = {"contact_dataflow", "contact_resources"}


MODEL_IDENTITY_ADAPTER = "monolithic-owned-geometry/v1"


def _exact(a, b):
    try:
        compare_arrays(
            {k: np.asarray(v) for k, v in a.items()}, {k: np.asarray(v) for k, v in b.items()}, rtol=0, atol=0
        )
    except (AssertionError, TypeError) as error:
        raise ValueError("Physical input, policy or sampling differs") from error


def _legacy_sampling(contact):
    """Bind the actual legacy tet quadrature and endpoint metadata, including empty sets."""
    names = (
        "shapes",
        "triangles",
        "mapping",
        "types",
        "bodies",
        "body_local",
        "target_bodies",
        "sdf_indices",
        "transforms",
        "scales",
        "margins",
        "area",
    )
    values = {name: getattr(contact, name).numpy() for name in names}
    legacy_count = 3 * len(values["triangles"]) * len(values["shapes"])
    actor_count = contact.actor_samples.count if contact.actor_samples is not None else 0
    if contact.legacy_count != legacy_count or contact.count != legacy_count + actor_count:
        raise ValueError("Complete contact enumeration count differs from its sampling metadata")
    values["ancestors"] = np.frombuffer(json.dumps(contact._legacy_ancestors, sort_keys=True).encode(), dtype=np.uint8)
    return values


def _physical_model_arrays(run):
    """Compare owned geometry contents, validating live SDF binding before removing handles."""
    model, contact = run.model, run._contact
    values = model_arrays(model)
    pointers = values.pop("shape_source_ptr")
    meshes = {mesh.id: mesh for mesh in model._mesh_keep_alive}
    values["shape_source_present"] = pointers != 0
    for shape, pointer in enumerate(pointers):
        if not pointer:
            continue
        if int(pointer) not in meshes:
            raise ValueError("Shape source is not a Model-owned mesh")
        mesh = meshes[int(pointer)]
        for name in ("points", "indices", "velocities", "groups"):
            array = getattr(mesh, name)
            values[f"shape_source.{shape}.{name}.present"] = np.array(array is not None)
            if array is not None:
                values[f"shape_source.{shape}.{name}"] = array.numpy()
    # Keep the existing recording adapter's schema checks for unused coloring
    # and ray-BVH caches; their contents/tails can depend on GPU scheduling.
    for name in ("particle_colors", "bvh_shape_enabled"):
        if name in values:
            values[name] = np.zeros_like(values[name])
    values.pop("_texture_sdf_data", None)
    table = model._texture_sdf_data
    if table is not None:
        if table is not contact.table or contact._descriptors is None:
            raise ValueError("Model SDF descriptor binding changed")
        descriptors = table.numpy()
        _exact({"descriptors": descriptors}, {"descriptors": contact._descriptors})
        for name, owners in (
            ("coarse_texture", model._texture_sdf_coarse_textures),
            ("subgrid_texture", model._texture_sdf_subgrid_textures),
            ("subgrid_start_slots", model._texture_sdf_subgrid_start_slots),
        ):
            if len(owners) != table.size:
                raise ValueError("Model SDF requires every owned resource")
            for descriptor, owner in zip(descriptors[name], owners, strict=True):
                if owner is None or owner.device != model.device:
                    raise ValueError("Model SDF resource ownership differs")
                matches = (
                    descriptor["data"] == owner.ptr
                    if name == "subgrid_start_slots"
                    else descriptor.tobytes() == bytes(owner.__ctype__())
                )
                if not matches:
                    raise ValueError("Model SDF descriptor does not reference its owned resource")
    elif contact._descriptors is not None:
        raise ValueError("Model SDF descriptor binding changed")
    fields = run._freeze_sdf(model)
    _exact(fields, run._sdf_fields)
    values.update({"owned_sdf." + name: array for name, array in fields.items()})
    if contact.smooth_sdf is not None:
        contact.smooth_sdf.validate_identity()
    values["identity_adapter"] = np.frombuffer(MODEL_IDENTITY_ADAPTER.encode(), dtype=np.uint8)
    return values


def verify_contract(left, right):
    """Authenticate a complete-candidate pooled audit run against a spatial run.

    Only declared algorithm/resource fields may differ; all physics must match.
    """
    if not (left._contact_pool and not left._contact_spatial and right._contact_spatial):
        raise ValueError("Comparison requires pooled/spatial algorithms")
    algorithm_fields = ALGORITHM_FIELDS | {"contact_spatial"}
    labels = ("pooled", "spatial")
    if left.precision.value != "fp64" or right.precision.value != "fp64":
        raise ValueError("Comparison requires native FP64")
    model_inputs = _physical_model_arrays(left)
    _exact(model_inputs, _physical_model_arrays(right))
    _exact(left._physics_values, right._physics_values)
    _exact(left._sdf_fields, right._sdf_fields)
    _exact(left._kinematics._source_values, right._kinematics._source_values)
    _exact(left._contact.values, right._contact.values)
    _exact(left._contact.actor_samples.fields, right._contact.actor_samples.fields)
    legacy_sampling = _legacy_sampling(left._contact)
    _exact(legacy_sampling, _legacy_sampling(right._contact))
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
        or left._contact.alignment_policy != right._contact.alignment_policy
    ):
        raise ValueError("Actor layout or numerical execution policy differs")
    _exact({"joint_parameters": left._joint_parameters_host}, {"joint_parameters": right._joint_parameters_host})
    policies = (
        "newton_max_iterations",
        "line_search_max_iterations",
        "linear_max_iterations",
        "linear_tolerance",
        "_globalization",
    )
    _exact({name: getattr(left, name) for name in policies}, {name: getattr(right, name) for name in policies})
    a, b = left._coupled_config, right._coupled_config
    for run, config, fields in zip((left, right), (a, b), (ALGORITHM_FIELDS, algorithm_fields), strict=True):
        if config.keys() & algorithm_fields != fields:
            raise ValueError("Unreviewed configuration difference")
        c = run._contact
        identity = b"spatial-pool/newton-frozen/v1" if run._contact_spatial else b"complete-pool/fixed-tree/v1"
        _exact({"identity": config["contact_dataflow"]}, {"identity": np.frombuffer(identity, dtype=np.uint8)})
        _exact({"identity": c.dataflow_identity}, {"identity": config["contact_dataflow"]})
        resources = [c.pool.capacity, c.pool.max_support, c._batch.batch_capacity, config["guard_capacity"][1]]
        if run._contact_spatial:
            resources.append(c.candidate_capacity)
            _exact({"margin": config["contact_spatial"]}, {"margin": np.zeros(1, np.float64)})
        _exact({"resources": config["contact_resources"]}, {"resources": np.array(resources, np.int64)})
    common_config = {name: value for name, value in a.items() if name not in algorithm_fields}
    right_config = {name: value for name, value in b.items() if name not in algorithm_fields}
    differences = {
        name: {label: serial(config.get(name)) for label, config in zip(labels, (a, b), strict=True)}
        for name in sorted(algorithm_fields)
    }
    # The old combined field contains a physical guard and a contact resource
    # budget. Only the latter may vary in the explicit capacity experiment.
    for config in (common_config, right_config):
        if config["guard_capacity"].shape != (2,):
            raise ValueError("Invalid guard/resource configuration")
        config["guard_capacity"] = config["guard_capacity"][:1]
    differences["contact_factor_capacity"] = {
        label: int(config["guard_capacity"][1]) for label, config in zip(labels, (a, b), strict=True)
    }
    _exact(common_config, right_config)
    # Bind the exact compared values, rather than asserting two different
    # aggregate configuration hashes happen to represent the same physics.
    common = {"model." + name: value for name, value in model_inputs.items()}
    for prefix, values in (("config", common_config), ("sampling", left._contact.actor_samples.fields)):
        common.update({prefix + "." + name: value for name, value in values.items()})
    common.update({"legacy_sampling." + name: value for name, value in legacy_sampling.items()})
    common["joint_parameters"] = left._joint_parameters_host
    common["layout"] = np.frombuffer(left._actors.layout_payload(), dtype=np.uint8)
    common.update({"effective_policy." + name: value for name, value in dataclasses.asdict(left._config).items()})
    return {
        "comparison": "/".join(labels),
        "physical_contract_sha256": _configuration_digest(common).hex(),
        "model_identity_adapter": MODEL_IDENTITY_ADAPTER,
        "algorithm_differences": differences,
        "config_digests": [run._kinematics.digest.hex() for run in (left, right)],
        "enumeration": {
            "legacy": left._contact.legacy_count,
            "actor": left._contact.actor_samples.count,
            "complete": left._contact.count,
        },
    }


def compare_actor_contact_recordings(reference, actual, *, reference_run, actual_run, require_complete=True):
    """Compare independent B2/B3 recordings under the original actor envelope."""
    from scripts.monolithic_reference.actor_recording import actor_frames, check_actor  # noqa: PLC0415

    contract = verify_contract(reference_run, actual_run)
    if contract["comparison"] != "pooled/spatial":
        raise ValueError("Actor contact trajectories require B2/B3")
    paths = (Path(reference), Path(actual))
    audits = [
        check_actor(path, expected_run=run, require_complete=require_complete)
        for path, run in zip(paths, (reference_run, actual_run), strict=True)
    ]
    metadata = [json.loads((path / "configuration.json").read_text()) for path in paths]
    for name in ("fixture_sha256", "layout_digest", "input_scope", "start_step", "end_step"):
        if audits[0][name] != audits[1][name]:
            raise ValueError("Contact trajectory inputs, layout or clock differ")
    if metadata[0]["execution"] != metadata[1]["execution"] or any(
        m["input_scope"] == "frozen" and m["before"] for m in metadata
    ):
        raise ValueError("Contact trajectory execution or creation order differs")
    sources = [json.loads((path / "source-hashes.json").read_text()) for path in paths]
    if sources[0] != sources[1]:
        raise ValueError("Mixed source identities in contact trajectory comparison")
    current = provenance()["source_hashes"]
    required = {name for name in current if name.startswith("newton/_src/solvers/monolithic/")} | {
        "scripts/monolithic_reference/" + name + ".py"
        for name in (
            "run_actor_coupling",
            "check_recording",
            "actor_recording",
            "contact_system_audit",
            "contact_audit",
            "contact_routes",
        )
    }
    if sources[0].keys() != required or any(sources[0][name] != current[name] for name in required):
        raise ValueError("Contact trajectory recording differs from current source")
    errors, passed, count = {}, True, 0
    frames = [actor_frames(path, m) for path, m in zip(paths, metadata, strict=True)]
    for left, right in zip(*frames, strict=True):
        passed &= compare_semantic_fields(
            _semantic(left, metadata[0]), _semantic(right, metadata[1]), load_case()["equivalence"], errors
        )
        count += 1
    return {
        "status": "PASS_FROZEN_ENVELOPE" if passed else "FAIL",
        "frames": count,
        "fields": errors,
        "audits": audits,
        "contract": contract,
        "qualification": "INDEPENDENT_CONTACT_TRAJECTORIES",
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


def _contact_at(run, frame, start, dt, *, sparse=False, frozen=None, allow_tainted=False):
    """Evaluate native contact inputs on an audit instance, without transplanting history."""
    c, k = run._contact, run._kinematics
    state = run.create_state(dt=dt)
    h = state._monolithic_coupled

    # Only geometry operands are needed by contact. This scratch state is not
    # a continuation or publication and must never be passed to solver.step.
    def query(values, begin, *, refresh=False):
        state.joint_q.assign(values["q"])
        state.particle_q.assign(values["x"])
        h.rigid_com.assign(values["com"])
        h.rigid_rotation.assign(values["rotation"])
        k.evaluate(k.gather_q(state))
        if refresh:
            c.refresh(k.gather_q(state), state.particle_q, rigid_com=h.rigid_com)
            return
        method = c.begin_step if begin else c.evaluate
        method(
            state.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            rigid_com=h.rigid_com,
            rigid_rotation=h.rigid_rotation,
            **({"kinematics": k, "q0": k.gather_q(state)} if begin and run._contact_spatial else {}),
            **({} if begin else {"dt": dt}),
        )

    if frozen is not None and not run._contact_spatial:
        raise ValueError("Frozen query state requires a spatial audit")
    query(start, True)
    if run._contact_spatial:
        query(frame if frozen is None else frozen, False, refresh=True)
    query(frame, False)
    if c.flags.numpy()[0] or c.start_flags.numpy()[0]:
        raise ValueError("Invalid native contact audit")
    if c.evaluation_diagnostics()["tainted"] and not allow_tainted:
        raise ValueError("Capacity overflow cannot qualify complete contacts")
    sequence = run._assembly_sequence + 1
    run._assembly_sequence = sequence
    generation = MonolithicLinearGeneration(1, sequence, c.evaluation_generation, sequence)
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
    c.publish(state)
    static = np.zeros((run.model.shape_count, 6))
    bodies = run.model.shape_body.numpy()
    # Actor pair diagnostics omit T contacts; the committed ledger does not.
    records, samples = c.pool.records, c._batch.all_samples
    count = int(records.count.numpy()[0])
    identities, wrenches = records.identities.numpy()[:count], records.pairs.numpy()[:count]
    sources, targets = samples.sources.numpy(), samples.targets.numpy()
    for i in np.argsort(identities, kind="stable"):
        key = int(identities[i])
        for shape, wrench in zip((sources[key], targets[key]), wrenches[i], strict=True):
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


def audit_spatial_contact_state(run, frame, start, dt, *, frozen=None, mesh_fields=None):
    """Measure one frozen B3 state; selected correctness grants no full-system qualification.

    The separate audit instance must never be advanced after contact-only assembly.
    Full enumeration does not consume query bounds, margins or admission bitmaps.
    Query overflow makes candidate exclusion ambiguous; report it as such instead
    of attributing missing identities to spatial clipping. Unadmitted zero terms
    are counted separately from lost contributions and actual pool overflow.
    """
    if not run._contact_spatial or run.precision.value != "fp64":
        raise ValueError("Spatial audit requires the native FP64 experimental path")
    actual = _contact_at(run, frame, start, dt, sparse=True, frozen=frozen, allow_tainted=True)
    c, selection, records = run._contact, run._contact._spatial, run._contact.pool.records
    samples = c._batch.all_samples
    fields = np.column_stack([getattr(samples, name).numpy() for name in ("sources", "targets", "faces", "slots")])
    candidates = selection.identities.numpy()[: int(selection.count.numpy()[0])].astype(np.int64)
    admitted = records.identities.numpy()[: int(records.count.numpy()[0])].astype(np.int64)
    stiffness, mu, velocity, width, radius = c.parameters
    full = evaluate_stream(
        run.model,
        run._actors,
        frame,
        start,
        candidate_keys=fields[candidates],
        admitted_keys=fields[admitted],
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
    errors = _check_contact_sparse(actual, full["admitted"])
    query, pool = c.spatial_diagnostics(), c.evaluation_diagnostics()
    return {
        "status": "MEASURED_NOT_QUALIFIED",
        "tainted": pool["tainted"],
        "state_sha256": _configuration_digest(frame).hex(),
        "step_start_sha256": _configuration_digest(start).hex(),
        "frozen_sha256": _configuration_digest(frame if frozen is None else frozen).hex(),
        "candidate_key_sha256": hashlib.sha256(fields[candidates].astype("<i8").tobytes()).hexdigest(),
        "admitted_key_sha256": hashlib.sha256(fields[np.sort(admitted)].astype("<i8").tobytes()).hexdigest(),
        "query": query,
        "pool": pool,
        "native_selected_errors": errors,
        "candidate_zero_work": full["candidate_count"] - full["candidate_contributing_count"],
        "unadmitted_zero_work": full["capacity_omission"]["count"] - full["capacity_omission"]["contributing_count"],
        "spatial_omission": None if query["capacity_dropped"] else full["spatial_omission"],
        "candidate_exclusion": full["spatial_omission"],
        **{
            name: value
            for name, value in full.items()
            if name not in ("full", "candidate", "admitted", "spatial_omission")
        },
    }
