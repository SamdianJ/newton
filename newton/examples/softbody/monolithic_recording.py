# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record complete monolithic states and unchanged diagnostic quantities."""

import dataclasses
import enum
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

REPO = Path(__file__).resolve().parents[3]
NORMAL = json.loads((REPO / "scripts/monolithic_reference/fixtures/normal_loading_frozen_v3.json").read_text())


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


def measure(run, state):
    contact = run._contact
    k = run._kinematics
    gaps = contact.current.distance.numpy().copy()
    shape_ids = contact.shapes.numpy()
    if len(gaps):
        gaps -= contact.margins.numpy()[shape_ids[np.arange(len(gaps)) % len(shape_ids)]] + contact.parameters[4]
    local = contact.local_residual.numpy()
    normal = contact.normal.numpy().copy()
    slide = contact.current.y.numpy() - contact.start.y.numpy()
    slip = slide - np.einsum("ni,ni->n", normal, slide)[:, None] * normal
    friction = contact.weights.numpy()[:, 1, None] * slip
    gradient = contact.current.gradient.numpy()
    normal_load = -np.einsum("ni,ni->n", local - friction, gradient) / np.maximum(np.sum(gradient**2, axis=1), 1e-300)
    physical = state._monolithic_coupled
    pforce = physical.contact_particle_force.numpy()
    bforce = physical.contact_body_force.numpy()
    return {
        "penetration": max(0.0, float(-gaps.min())) if len(gaps) else 0.0,
        "contact_load": float(normal_load.sum()),
        "friction_work": float(np.sum(friction * slide)),
        "slip": float(np.linalg.norm(slip, axis=1).max()) if len(slip) else 0.0,
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
