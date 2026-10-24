# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record complete monolithic states and unchanged diagnostic quantities."""

import dataclasses
import enum
import json
import math
import weakref
from pathlib import Path

import numpy as np
import warp as wp

import newton

REPO = Path(__file__).resolve().parents[3]
NORMAL = json.loads((REPO / "scripts/monolithic_reference/fixtures/normal_loading_frozen_v3.json").read_text())
# Private producers keep measurement independent of the solver's spatial selection and pool.
_MEASURE_BATCHES = weakref.WeakKeyDictionary()
# Grasp success, approved 2026-10-24: over the final 0.5 s at the top of the lift, every
# published step keeps the ground support within 5% of the object weight and the object
# center of mass at least a quarter of its smallest edge above rest.
GRASP_WINDOW_STEPS = 500
GRASP_GROUND_WEIGHT_FRACTION = 0.05
GRASP_RISE_EDGE_FRACTION = 0.25


def grasp_metrics(window, *, initial_com_z, weight, edge):
    """Measure the grasp window from ``(step, com_z, ground_force_z)`` rows [m, N].

    The window must be ``GRASP_WINDOW_STEPS`` consecutive published steps; otherwise the
    grasp is not held.
    """
    steps = [int(step) for step, _, _ in window]
    rise = min((float(z) - initial_com_z for _, z, _ in window), default=None)
    ground = max((abs(float(force)) for _, _, force in window), default=None)
    complete = len(steps) == GRASP_WINDOW_STEPS and steps == list(range(steps[0], steps[0] + len(steps)))
    return {
        "window_first_step": steps[0] if steps else None,
        "window_last_step": steps[-1] if steps else None,
        "window_steps": len(steps),
        "min_com_rise_m": rise,
        "max_ground_force_N": ground,
        "weight_N": weight,
        "object_min_edge_m": edge,
        "held": bool(
            complete and ground <= GRASP_GROUND_WEIGHT_FRACTION * weight and rise >= GRASP_RISE_EDGE_FRACTION * edge
        ),
    }


def object_weight_and_edge(model, positions):
    """Return the soft object weight [N] and smallest rest bounding-box edge [m]."""
    gravity = float(np.linalg.norm(model.gravity.numpy()[0]))
    mass = float(model.particle_mass.numpy().astype(float).sum())
    return mass * gravity, float(np.ptp(np.asarray(positions, dtype=float), axis=0).min())


def serial(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, np.generic):
        return serial(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if dataclasses.is_dataclass(value):
        return {f.name: serial(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, "items"):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def save(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def state_arrays(state):
    arrays = {name: value.numpy().copy() for name, value in vars(state).items() if isinstance(value, wp.array)}
    arrays.update(
        {
            "history." + name: value.numpy().copy()
            for name, value in vars(state._monolithic_coupled).items()
            if isinstance(value, wp.array)
        }
    )
    return arrays


def _native_inputs(run, state):
    k, h = run._kinematics, state._monolithic_coupled
    k.evaluate(k.gather_q(state))
    return state.particle_q, k.poses, k.com, k.jt, k.jw, h.rigid_com, h.rigid_rotation


def measure(run, state):
    """Measure the accepted step over every contact sample from its physical step start."""
    from newton._src.solvers.monolithic.contact_batch import _ContactBatch  # noqa: PLC0415

    contact, k = run._contact, run._kinematics
    batch = _MEASURE_BATCHES.get(contact)
    if batch is None:
        batch = _ContactBatch(contact, batch_capacity=min(contact.count, 8192), capacity=None)
        _MEASURE_BATCHES[contact] = batch
    start = _native_inputs(run, run._original)
    batch.begin_step(*start[:5], rigid_com=start[5], rigid_rotation=start[6])
    # Leaves the solver kinematics at the accepted state, as before measurement.
    args = batch._state(*_native_inputs(run, state))
    gaps, slips, load, work = [], [], 0.0, 0.0
    # Contacts the alignment mask omits still report their loads and penetration.
    masked_keys, masked_gaps, omitted_sum = [], [], np.zeros(4)
    for mode in batch._chunks():
        batch._stage(args, mode, run._dt, begin=False)
        if batch._flags.numpy()[0]:
            raise ValueError("Invalid contact measurement geometry")
        n = int(batch.pool.candidates.count.numpy()[0])
        gaps.append(batch.current.distance.numpy()[:n] - batch.samples.gaps.numpy()[:n])
        normal = batch.normal.numpy()[:n]
        slide = batch.current.slip.numpy()[:n]
        slip = slide - np.einsum("ni,ni->n", normal, slide)[:, None] * normal
        friction = batch.weights.numpy()[:n, 1, None] * slip
        gradient = batch.current.gradient.numpy()[:n]
        local = batch.local_r.numpy()[:n]
        load += float(
            (-np.einsum("ni,ni->n", local - friction, gradient) / np.maximum(np.sum(gradient**2, axis=1), 1e-300)).sum()
        )
        work += float(np.sum(friction * slide))
        slips.append(np.linalg.norm(slip, axis=1))
        omitted = batch.omitted.numpy()[:n]
        active = np.any(omitted[:, 2:] > 0, axis=1)
        masked_keys.extend(batch.pool.candidates.identities.numpy()[:n][active].tolist())
        masked_gaps.append(gaps[-1][active])
        omitted_sum += omitted.sum(axis=0)
    gaps, slips = np.concatenate(gaps or [np.zeros(0)]), np.concatenate(slips or [np.zeros(0)])
    masked_gaps = np.concatenate(masked_gaps or [np.zeros(0)])
    physical = state._monolithic_coupled
    pforce = physical.contact_particle_force.numpy()
    bforce = physical.contact_body_force.numpy()
    penetration = max(0.0, float(-gaps.min())) if len(gaps) else 0.0
    return {
        "penetration": penetration,
        "mask": {
            "masked_active_count": len(masked_keys),
            "masked_active_keys": masked_keys,
            "omitted_normal_energy": float(omitted_sum[0]),
            "omitted_friction_energy": float(omitted_sum[1]),
            "omitted_normal_load_sum": float(omitted_sum[2]),
            "omitted_friction_load_sum": float(omitted_sum[3]),
            "max_masked_penetration": max(0.0, float(-masked_gaps.min())) if len(masked_gaps) else 0.0,
            "max_all_penetration": penetration,
            "switch_history": "NOT_RECORDED_BY_BOUNDED_STREAM",
        },
        "contact_load": load,
        "friction_work": work,
        "slip": float(slips.max()) if len(slips) else 0.0,
        "force_balance": float(
            np.linalg.norm(pforce.sum(axis=0) + bforce[:, :3].sum(axis=0))
            / max(
                np.linalg.norm(pforce, axis=1).sum() + np.linalg.norm(bforce[:, :3], axis=1).sum(),
                NORMAL["acceptance"]["force_floor_n"],
            )
        ),
        "body_contact_force": bforce,
        "q": state.joint_q.numpy().copy(),
        "qd": state.joint_qd.numpy().copy(),
        "x": state.particle_q.numpy().copy(),
        "v": state.particle_qd.numpy().copy(),
        "com": k.com.numpy().copy(),
        "rotation": k.rotations.numpy().copy(),
    }


def contact_fields(run, state):
    """Return precision-generic penetration and ground/robot contact forces [m, N].

    Penetration covers every contact sample from the step start; forces sum the
    production pool's committed per-contact forces on the tet, split by whether the
    target shape is a plane. Requires canonical tet-to-shape samples.
    """
    c = run._contact
    records = c.pool.records
    n = int(records.count.numpy()[0])
    ids, pairs = records.identities.numpy()[:n], records.pairs.numpy()[:n]
    sources, targets = c._batch.all_samples.sources.numpy(), c._batch.all_samples.targets.numpy()
    plane = run.model.shape_type.numpy()[targets] == int(newton.GeoType.PLANE)
    forces = np.zeros((2, 3))
    for slot in np.argsort(ids, kind="stable"):
        key = int(ids[slot])
        if sources[key] != -1:
            raise ValueError("Contact fields require canonical tet-to-shape samples")
        forces[int(not plane[key])] += pairs[slot, 0, :3]
    measured = measure(run, state)
    return {
        "penetration": measured["penetration"],
        "contact_load": measured["contact_load"],
        "slip": measured["slip"],
        "ground_force": forces[0],
        "robot_force": forces[1],
    }
