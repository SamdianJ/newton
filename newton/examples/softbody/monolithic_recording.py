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

REPO = Path(__file__).resolve().parents[3]
NORMAL = json.loads((REPO / "scripts/monolithic_reference/fixtures/normal_loading_frozen_v3.json").read_text())
# Private producers keep measurement independent of the solver's spatial selection and pool.
_MEASURE_BATCHES = weakref.WeakKeyDictionary()


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
    gaps, slips = np.concatenate(gaps or [np.zeros(0)]), np.concatenate(slips or [np.zeros(0)])
    physical = state._monolithic_coupled
    pforce = physical.contact_particle_force.numpy()
    bforce = physical.contact_body_force.numpy()
    return {
        "penetration": max(0.0, float(-gaps.min())) if len(gaps) else 0.0,
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
