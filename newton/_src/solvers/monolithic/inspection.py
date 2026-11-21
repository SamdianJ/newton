# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only, versioned capture of a monolithic solver state for diagnostics.

:func:`capture` copies the physical state of one solver state and the values the
solver holds from its last evaluation into host arrays, with a schema that names
each field's unit, frame, binding and source. It never runs a solver phase, a
graph or a state validation, and never writes solver or state storage, so a
following step is bit for bit the same with or without it. Every call
synchronizes the host with the device; it is not part of :meth:`step`.

Binding of each field:

* ``state``: read from the state passed in.
* ``static``: fixed by the solver layout.
* ``last_evaluation``: held by the solver from its last evaluation. Exported only
  when the schema's ``role`` is ``published``: then that evaluation is the final
  one at the returned state. Off that path (a failed or rolled-back step, a state
  other than the published one) these values can come from different
  evaluations, so they are listed in ``absent`` instead.
* ``derived``: computed here from solver storage; currently the per-element
  det F of the published step, by the tet evaluation's formula and compile
  options.

Non-finite values in the JSON records are written as null; the schema's
``nonfinite`` maps each such path to its original value.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from . import tet_precision
from .convergence import Unit
from .numeric_parameters import scalar_value

# det F must round like tet_precision._elements, so compile with that module's arithmetic options.
ARITHMETIC_OPTIONS = ("fuse_fp", "fast_math")
wp.set_module_options({k: wp.get_module(tet_precision.__name__).options[k] for k in ARITHMETIC_OPTIONS})

SCHEMA_VERSION = 1

ROW_UNITS = tuple({Unit.FORCE: "N", Unit.TORQUE: "N m"}[unit] for unit in sorted(Unit))

# name: (binding, unit, frame, meaning, source, valid_when)
_FIELDS = {
    "joint_q": (
        "state",
        "joint",
        "joint",
        "joint coordinates; m or rad by joint type; a FREE joint is [position 3, quaternion x y z w] of the child joint frame relative to the parent joint frame, inv(X_p) * pose * X_c (rigid.py:88-98 _mirrors)",
        "state.joint_q",
        "always",
    ),
    "joint_qd": (
        "state",
        "joint_rate",
        "joint",
        "joint velocities; m/s or rad/s by DoF; a FREE joint is [child centre-of-mass velocity 3, angular velocity 3], both in its parent joint frame (rigid.py:91-96 _mirrors)",
        "state.joint_qd",
        "always",
    ),
    "particle_q": ("state", "m", "world", "particle positions in solver precision", "state.particle_q", "always"),
    "particle_qd": ("state", "m/s", "world", "particle velocities in solver precision", "state.particle_qd", "always"),
    "rigid_com": (
        "state",
        "m",
        "world",
        "FREE rigid centre of mass",
        "state._monolithic_coupled.rigid_com",
        "rigid actors present",
    ),
    "rigid_rotation": (
        "state",
        "quaternion",
        "world",
        "FREE rigid orientation (x, y, z, w)",
        "state._monolithic_coupled.rigid_rotation",
        "rigid actors present",
    ),
    "rigid_v_com": (
        "state",
        "m/s",
        "world",
        "FREE rigid centre-of-mass velocity",
        "state._monolithic_coupled.rigid_v_com",
        "rigid actors present",
    ),
    "rigid_omega": (
        "state",
        "rad/s",
        "world",
        "FREE rigid angular velocity, world-frame components (rigid.py:65 _initialize)",
        "state._monolithic_coupled.rigid_omega",
        "rigid actors present",
    ),
    "cable_theta": (
        "state",
        "rad",
        "material",
        "cable twist angle per edge",
        "state._monolithic_coupled.cable_theta",
        "cables present",
    ),
    "contact_particle_force": (
        "state",
        "N",
        "world",
        "published contact force on each particle: shape and material contact, plus the point-cloud cloth contact force "
        "when cloth_contact is enabled (state._monolithic_coupled.cloth_contact_particle_force, not exported) and the "
        "cable self and inter-cable contact force when cable_contact is enabled "
        "(state._monolithic_coupled.cable_contact_particle_force, not exported); the "
        "contact_* record fields cover only the shape and material contact pool",
        "state._monolithic_coupled.contact_particle_force",
        "force_valid == 1",
    ),
    "contact_body_force": (
        "state",
        "N|N m",
        "world",
        "published contact wrench on each body: [force 3, torque 3 about the body centre of mass] (contact_batch.py:426, contact_material.py:213)",
        "state._monolithic_coupled.contact_body_force",
        "force_valid == 1",
    ),
    "contact_joint_force": (
        "state",
        "joint_force",
        "joint",
        "published contact generalized force per joint DoF; with self_collision it includes same-tree (A-self) contact on the DoFs not shared by the two bodies (their common ancestors' DoFs carry none); a floating root's six rows are world-frame [force 3 at the root centre of mass, torque 3 about it] (the root unknowns are the world COM increment and the left rotation increment), unlike that root's joint_qd rows, which are in its parent joint frame",
        "state._monolithic_coupled.contact_joint_force",
        "force_valid == 1",
    ),
    "contact_cable_torque": (
        "state",
        "N m",
        "material",
        "published contact torque per cable edge",
        "state._monolithic_coupled.contact_cable_torque",
        "force_valid == 1 (cables present)",
    ),
    "force_generation": (
        "state",
        "1",
        "none",
        "step generation that published the contact forces",
        "state._monolithic_coupled.force_generation",
        "always",
    ),
    "force_valid": (
        "state",
        "1",
        "none",
        "1 when the contact forces were published",
        "state._monolithic_coupled.force_valid",
        "always",
    ),
    "residual": (
        "last_evaluation",
        "row_unit",
        "row",
        "force-balance residual per scalar row (RHS = -S r); the incremental-potential gradient only for conservative producers, not for C-IE contact or rigid torque rows",
        "Stats.variational['residual']",
        "role == published",
    ),
    "row_unit": (
        "static",
        "1",
        "row",
        "unit of each residual row, index into schema 'row_units'",
        "solver._criterion.rows.units",
        "always",
    ),
    "row_producer": (
        "static",
        "1",
        "row",
        "producer owning each row, index into schema 'producers'",
        "producer.rows",
        "always",
    ),
    "row_block": (
        "static",
        "1",
        "row",
        "q/x or actor block of each row, index into schema 'blocks' (the aggregate block is excluded)",
        "solver._problem.blocks",
        "always",
    ),
    "row_group": (
        "static",
        "1",
        "row",
        "physical convergence group of each row, index into schema 'row_groups'",
        "solver._criterion.rows.groups",
        "always",
    ),
    "energy_terms": (
        "last_evaluation",
        "J",
        "none",
        "row 0: energy terms, row 1: their magnitudes; contact is the C-IE energy, not a potential",
        "solver._problem.energy_terms",
        "role == published",
    ),
    "energy_term_producer": (
        "static",
        "1",
        "none",
        "producer of each energy term, index into schema 'producers'",
        "solver._problem._energy_bindings",
        "always",
    ),
    "producer_energy": (
        "last_evaluation",
        "J",
        "none",
        "per-producer sum of energy terms, exact host FP64 sum",
        "solver._problem.energy_terms",
        "role == published",
    ),
    "tet_min_det_f": (
        "last_evaluation",
        "1",
        "none",
        "minimum det F of the last evaluation",
        "Stats.min_det_f",
        "role == published and a tet is present",
    ),
    "tet_det_f": (
        "derived",
        "1",
        "element",
        "det F per element of the published step, det((Ds(x_n) + Ds(dx)) Dm^-1) as in the tet evaluation",
        "solver._original.particle_q, accepted step displacement",
        "role == published and a tet is present",
    ),
    "contact_identity": (
        "last_evaluation",
        "1",
        "none",
        "sample identity of each contact record",
        "solver._contact.pool.records.identities",
        "role == published",
    ),
    "contact_source": (
        "last_evaluation",
        "1",
        "none",
        "source Newton shape index of each contact record; -1: tet surface triangle sample, -2: cloth triangle sample, -3: cable edge sample (contact_sampling.py, contact_material_sampling.py, contact_batch.py)",
        "solver._contact._batch.all_samples.sources",
        "role == published",
    ),
    "contact_target": (
        "last_evaluation",
        "1",
        "none",
        "target Newton shape index of each contact record",
        "solver._contact._batch.all_samples.targets",
        "role == published",
    ),
    "contact_wrench": (
        "last_evaluation",
        "N|N m",
        "world",
        "per-record wrench, end 0 = source, end 1 = target, each [force 3, torque 3 about the world origin, cross(point, force)]; normal and friction combined (contact_batch.py:405-413, contact_material.py:203-211)",
        "solver._contact.pool.records.pairs",
        "role == published",
    ),
    "pair_source": (
        "last_evaluation",
        "1",
        "none",
        "first label of each contact pair: for a shape-shape pair the smaller Newton shape index (pairs are sorted, so not necessarily the record source), otherwise -1 tet surface, -2 cloth, -3 cable; key of actor_samples.pair_kinds (contact_pipeline.py:245)",
        "Stats.variational['actor_contact_pairs']",
        "role == published",
    ),
    "pair_target": (
        "last_evaluation",
        "1",
        "none",
        "second label of each contact pair: the larger shape index of a shape-shape pair, or the shape a -1/-2/-3 surface touches",
        "Stats.variational['actor_contact_pairs']",
        "role == published",
    ),
    "pair_wrench": (
        "last_evaluation",
        "N|N m",
        "world",
        "contact-pair wrench, fixed-tree sum of its records with ends in sorted shape order, each [force 3, torque 3 about the world origin]; records with identity below the legacy sample count are not included (contact_pipeline.py pair_diagnostics)",
        "Stats.variational['actor_contact_pairs']",
        "role == published",
    ),
}

_STATE_NAMES = ("joint_q", "joint_qd", "particle_q", "particle_qd")
_HISTORY_NAMES = (
    "rigid_com",
    "rigid_rotation",
    "rigid_v_com",
    "rigid_omega",
    "cable_theta",
    "contact_particle_force",
    "contact_body_force",
    "contact_joint_force",
    "contact_cable_torque",
    "force_generation",
    "force_valid",
)
_COORDINATES = ("joint_q", "particle_q", "rigid_com", "rigid_rotation", "cable_theta")


@wp.kernel
def _det_f(
    xn: wp.array[Any],
    dx: wp.array[Any],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    like: Any,
    det: wp.array[Any],
):
    zero = scalar_value(like)
    e = wp.tid()
    pose = wp.matrix(poses[e], shape=(3, 3), dtype=type(zero))
    n0, n1, n2, n3 = indices[e, 0], indices[e, 1], indices[e, 2], indices[e, 3]
    start = wp.matrix_from_cols(xn[n1] - xn[n0], xn[n2] - xn[n0], xn[n3] - xn[n0]) * pose
    step = wp.matrix_from_cols(dx[n1] - dx[n0], dx[n2] - dx[n0], dx[n3] - dx[n0]) * pose
    det[e] = wp.determinant(start + step)


class _Reader:
    """Copy device arrays to independent host arrays and record what was read."""

    def __init__(self):
        self.reads = []

    def __call__(self, source, array):
        self.reads.append(source)
        return np.array(array.numpy(), copy=True)


def _check_state(solver, state, read):
    schema = solver._states
    history = getattr(state, schema.namespace, None)
    if history is None:
        raise ValueError(f"state has no {schema.namespace} history; it was not created by this solver")
    actual = {name for name, value in vars(history).items() if isinstance(value, wp.array)}
    if actual != set(schema.schema):
        raise ValueError("state history does not match this solver's state schema")
    for name, (shape, dtype) in schema.schema.items():
        array = getattr(history, name)
        if (array.shape, array.dtype, array.device) != (shape, dtype, solver.model.device):
            raise ValueError(f"state history array {name} does not match this solver")
    expected = {
        "joint_q": (solver.model.joint_coord_count,),
        "joint_qd": (solver.model.joint_dof_count,),
        "particle_q": (solver.model.particle_count,),
        "particle_qd": (solver.model.particle_count,),
    }
    for name, shape in expected.items():
        array = getattr(state, name, None)
        if not isinstance(array, wp.array) or array.shape != shape or array.device != solver.model.device:
            raise ValueError(f"state array {name} does not match this solver")
    digest = read("state._monolithic_coupled.config_digest", history.config_digest).tobytes()
    if digest != solver._kinematics.digest:
        raise ValueError("state belongs to a solver with another configuration")
    if "layout_digest" in schema.schema:
        layout = read("state._monolithic_coupled.layout_digest", history.layout_digest).tobytes()
        if layout.hex() != solver._actors.identity:
            raise ValueError("state belongs to another actor layout")


def _blocks(problem, rows):
    """Split the reporting blocks into the aggregate over all rows and the partition of the rows."""
    (aggregate, whole), *partition = problem.blocks
    if (whole.start, whole.stop) != (0, rows):
        raise ValueError(f"First reporting block {aggregate} does not cover all {rows} rows")
    covered = np.zeros(rows, dtype=np.int32)
    for _, part in partition:
        covered[part] += 1
    if not (covered == 1).all():
        raise ValueError("Reporting blocks after the aggregate do not partition the rows")
    return aggregate, partition


def _energy_term_producers(problem, producers):
    """Return the producer index of every packed energy term."""
    index_of = {(id(owner), name): i for i, p in enumerate(producers) for owner, name, _ in p.energies}
    owner = []
    for (binding_owner, name, _), shape in zip(problem._energy_bindings, problem._energy_shapes, strict=True):
        owner.extend([index_of[(id(binding_owner), name)]] * math.prod(shape))
    return np.array(owner, dtype=np.int32)


def _role(solver, state, read):
    """Return (role, describes_state) of the solver-held evaluation."""
    stats = solver.last_stats
    current = getattr(solver, "_current_state", None)
    if solver._accepted is None or current is None or stats.status == solver.Status.NOT_RUN:
        return "none", False
    names = {id(solver._accepted): "accepted_iterate", id(solver._trial): "trial", id(solver._original): "original"}
    role = names.get(id(current), "unknown")
    if stats.rolled_back or current is not solver._accepted:
        return role, False
    history = getattr(state, solver._states.namespace)
    generation = int(read("state._monolithic_coupled.force_generation", history.force_generation)[0])
    if generation != stats.step_generation:
        return role, False
    accepted = solver._accepted
    for name in _COORDINATES:
        if name in ("joint_q", "particle_q"):
            mine, theirs = getattr(state, name), getattr(accepted, name)
        else:
            mine = getattr(history, name, None)
            theirs = getattr(getattr(accepted, solver._states.namespace), name, None)
            if mine is None:
                continue
        if not np.array_equal(read(f"state.{name}", mine), read(f"solver._accepted.{name}", theirs)):
            return role, False
    return "published", True


def _jsonable(value, path="", nonfinite=None):
    """Convert to JSON with non-finite numbers as null, recording each such path and value in ``nonfinite``."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v, f"{path}.{k}", nonfinite) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, f"{path}[{i}]", nonfinite) for i, v in enumerate(value)]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), path, nonfinite)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), path, nonfinite)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        if nonfinite is not None:
            nonfinite[path] = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return None
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)


def capture(solver, state, fields=None):
    """Copy the state and the solver's last-evaluation values into host arrays.

    Args:
        solver: A :class:`SolverMonolithic`.
        state: A state created by ``solver``.
        fields: Field names to capture, or None for every available field.

    Returns:
        ``{"schema": dict, "arrays": {name: ndarray}, "records": {name: list}}``.

    Raises:
        ValueError: On an unknown field, a requested field that is unavailable,
            or a state that does not belong to ``solver``.
        RuntimeError: When called while the device is capturing a CUDA graph.
    """
    device = wp.get_device(solver.model.device)
    if device.is_cuda and device.is_capturing:
        raise RuntimeError("capture cannot run during CUDA graph capture; it synchronizes with the host")
    requested = set(_FIELDS) if fields is None else set(fields)
    unknown = sorted(requested - set(_FIELDS))
    if unknown:
        raise ValueError(f"Unknown capture field(s): {', '.join(unknown)}")
    read = _Reader()
    _check_state(solver, state, read)
    role, describes = _role(solver, state, read)
    stats = solver.last_stats
    history = getattr(state, solver._states.namespace)
    arrays, absent = {}, {}

    def want(name):
        return name in requested

    for name in _STATE_NAMES:
        if want(name):
            arrays[name] = read(f"state.{name}", getattr(state, name))
    for name in _HISTORY_NAMES:
        if not want(name):
            continue
        array = getattr(history, name, None)
        if array is None:
            absent[name] = "not in this solver's state schema (no cables)"
        elif name.startswith("rigid_") and not solver._rigid.count:
            absent[name] = "no FREE rigid actor"
        else:
            arrays[name] = read(f"state._monolithic_coupled.{name}", array)

    problem = solver._problem
    producers = list(problem.producers)
    rows = solver._layout.scalar_dof_count
    if want("row_producer"):
        owner = np.full(rows, -1, dtype=np.int32)
        for index, producer in enumerate(producers):
            owner[producer.rows] = index
        arrays["row_producer"] = owner
    aggregate, partition = _blocks(problem, rows)
    if want("row_block"):
        block = np.full(rows, -1, dtype=np.int32)
        for index, (_, part) in enumerate(partition):
            block[part] = index
        arrays["row_block"] = block
    if want("row_unit"):
        arrays["row_unit"] = np.array(solver._criterion.rows.units, dtype=np.int32, copy=True)
    if want("row_group"):
        arrays["row_group"] = np.array(solver._criterion.rows.groups, dtype=np.int32, copy=True)
    term_owner = _energy_term_producers(problem, producers)
    if want("energy_term_producer"):
        arrays["energy_term_producer"] = term_owner

    evaluated = role == "published"
    unevaluated = (
        "no step has evaluated yet"
        if role == "none"
        else "not the published step: off that path the solver-held values can come from different evaluations"
    )
    residual = stats.variational.get("residual") if evaluated else None
    if want("residual"):
        if residual is None:
            absent["residual"] = unevaluated if not evaluated else "the published step recorded no residual"
        else:
            arrays["residual"] = np.array(residual, copy=True)
    if want("energy_terms") or want("producer_energy"):
        if not evaluated:
            for name in ("energy_terms", "producer_energy"):
                if want(name):
                    absent[name] = unevaluated
        else:
            terms = read("solver._problem.energy_terms", problem.energy_terms)
            if want("energy_terms"):
                arrays["energy_terms"] = terms
            if want("producer_energy"):
                values = terms[0].astype(np.float64)
                arrays["producer_energy"] = np.array(
                    [math.fsum(values[term_owner == i]) for i in range(len(producers))], dtype=np.float64
                )

    if want("tet_min_det_f"):
        if not solver._has_tet:
            absent["tet_min_det_f"] = "no tet actor"
        elif not evaluated:
            absent["tet_min_det_f"] = unevaluated
        else:
            arrays["tet_min_det_f"] = np.array([stats.min_det_f], dtype=solver._numpy_dtype)
    if want("tet_det_f"):
        if not solver._has_tet:
            absent["tet_det_f"] = "no tet actor"
        elif not evaluated:
            absent["tet_det_f"] = unevaluated
        else:
            physics = solver._tet.physics
            det = wp.empty(solver.model.tet_count, dtype=solver._dtype, device=solver.model.device)
            like = wp.zeros(1, dtype=solver._dtype, device=solver.model.device)
            wp.launch(
                _det_f,
                solver.model.tet_count,
                [
                    solver._original.particle_q,
                    solver._steps[id(solver._accepted)][0],
                    physics.indices,
                    physics.poses,
                    like,
                    det,
                ],
                device=solver.model.device,
            )
            arrays["tet_det_f"] = read("derived tet det F", det)

    contact_names = ("contact_identity", "contact_source", "contact_target", "contact_wrench")
    if any(want(n) for n in contact_names):
        if not evaluated:
            for name in contact_names:
                if want(name):
                    absent[name] = unevaluated
        else:
            records = solver._contact.pool.records
            count = int(read("solver._contact.pool.records.count", records.count)[0])
            if count:
                identities = read("solver._contact.pool.records.identities[:count]", records.identities[:count])
                wrenches = read("solver._contact.pool.records.pairs[:count]", records.pairs[:count])
            else:
                identities = np.empty(0, dtype=np.uint64)
                wrenches = np.empty((0, 2, 6), dtype=solver._numpy_dtype)
            samples = solver._contact._batch.all_samples
            sources = read("solver._contact._batch.all_samples.sources", samples.sources)
            targets = read("solver._contact._batch.all_samples.targets", samples.targets)
            keys = identities.astype(np.int64)
            for name, value in (
                ("contact_identity", identities),
                ("contact_source", sources[keys] if count else np.empty(0, dtype=sources.dtype)),
                ("contact_target", targets[keys] if count else np.empty(0, dtype=targets.dtype)),
                ("contact_wrench", wrenches),
            ):
                if want(name):
                    arrays[name] = np.array(value, copy=True)

    pair_names = ("pair_source", "pair_target", "pair_wrench")
    if any(want(n) for n in pair_names):
        pairs = stats.variational.get("actor_contact_pairs") if role == "published" else None
        if pairs is None:
            for name in pair_names:
                if want(name):
                    absent[name] = unevaluated
        else:
            values = {
                "pair_source": np.array([p["source"] for p in pairs], dtype=np.int32),
                "pair_target": np.array([p["target"] for p in pairs], dtype=np.int32),
                "pair_wrench": np.array([p["world_wrench"] for p in pairs], dtype=solver._numpy_dtype).reshape(
                    len(pairs), 2, 6
                ),
            }
            for name, value in values.items():
                if want(name):
                    arrays[name] = value

    if fields is not None:
        missing = sorted(requested & set(absent))
        if missing:
            raise ValueError(
                "Unavailable capture field(s): " + "; ".join(f"{name}: {absent[name]}" for name in missing)
            )

    field_specs = {}
    for name, value in arrays.items():
        binding, unit, frame, meaning, source, valid_when = _FIELDS[name]
        field_role = "published" if binding in ("last_evaluation", "derived") else binding
        field_specs[name] = {
            "binding": binding,
            "role": field_role,
            "unit": unit,
            "frame": frame,
            "meaning": meaning,
            "source": source,
            "valid_when": valid_when,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    records, nonfinite = {}, {}
    if role != "none":
        for name in ("trials", "linear_postchecks"):
            if name in stats.variational:
                records[name] = _jsonable(stats.variational[name], f"records.{name}", nonfinite)
    schema = {
        "schema_version": SCHEMA_VERSION,
        "precision": solver.precision.value,
        "device": str(solver.model.device),
        "configuration": solver._kinematics.digest.hex(),
        "layout": solver._actors.identity,
        "step_generation": int(stats.step_generation) if role != "none" else None,
        "status": stats.status.value,
        "publication_status": stats.variational.get("publication_status") if role != "none" else None,
        "role": role,
        "describes_state": describes,
        "producers": [p.name for p in producers],
        "aggregate_block": aggregate,
        "blocks": [name for name, _ in partition],
        "row_units": list(ROW_UNITS),
        "row_groups": list(solver._criterion.rows.names),
        "scalars": _jsonable(
            {
                "converged": stats.converged,
                "rolled_back": stats.rolled_back,
                "objective_total": stats.variational.get("objective_total") if evaluated else None,
                "min_det_f": stats.min_det_f if evaluated and solver._has_tet else None,
            },
            "scalars",
            nonfinite,
        ),
        "not_measured": ["matrix_nnz", "max_penetration", "timings", "contact_force_imbalance"],
        "nonfinite": nonfinite,
        "host_synchronizing": True,
        "reads": read.reads,
        "fields": field_specs,
        "absent": absent,
    }
    return {"schema": schema, "arrays": arrays, "records": records}


def save_capture(result, directory, *, metadata=None):
    """Write ``state.npz`` and ``schema.json`` into a new directory.

    Raises:
        FileExistsError: If ``directory`` already exists.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.savez(directory / "state.npz", **result["arrays"])
    nonfinite = dict(result["schema"]["nonfinite"])
    document = {
        **result["schema"],
        "records": result["records"],
        "metadata": _jsonable(metadata or {}, "metadata", nonfinite),
    }
    document["nonfinite"] = nonfinite
    (directory / "schema.json").write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def load_capture(directory):
    """Read a capture written by :func:`save_capture`.

    Raises:
        ValueError: On a schema version other than :data:`SCHEMA_VERSION`, or
            arrays that do not match the schema's fields.
    """
    directory = Path(directory)

    def reject(value):
        raise ValueError(f"Non-finite JSON constant in schema.json: {value}")

    document = json.loads((directory / "schema.json").read_text(), parse_constant=reject)
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported capture schema version: {document.get('schema_version')}")
    with np.load(directory / "state.npz") as data:
        arrays = {name: data[name] for name in data.files}
    if set(arrays) != set(document["fields"]):
        raise ValueError("state.npz arrays do not match schema.json fields")
    records = document.pop("records")
    document.pop("metadata", None)
    return {"schema": document, "arrays": arrays, "records": records}
