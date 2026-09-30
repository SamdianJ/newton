# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the frozen Sharpa model and resolve its joint coordinates."""

import json
from pathlib import Path

import numpy as np
from newton.solvers.experimental.monolithic import SolverMonolithic

from newton.examples.softbody.monolithic_sharpa_assets import load_hand_cube
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture

REPO = Path(__file__).resolve().parents[3]
WORKSPACE = REPO.parents[1]
ARTIFACTS = WORKSPACE / "agents/integration/artifacts"
ASSETS = REPO / "asset/monolithic/data/inputs"


def joint_mapping(model, names, *, carriage=True):
    """Resolve coordinate, rate and target offsets independently by joint name."""
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    names = [*names, "monolithic_lift"] if carriage else list(names)
    if any(labels.count(name) != 1 for name in names):
        raise ValueError("Missing or duplicate hand/carriage joint")
    starts = [getattr(model, name).numpy() for name in ("joint_q_start", "joint_qd_start", "joint_target_q_start")]
    return {name: tuple(int(start[labels.index(name)]) for start in starts) for name in names}


def mask_statistics(contact):
    """Report omitted loads and penetration, including contacts excluded by the mask."""
    omitted = contact.omitted.numpy()
    keys = np.flatnonzero(np.any(omitted[:, 2:] > 0, axis=1))
    seen = contact.mask_seen.numpy()
    gap = contact.current.distance.numpy() - contact.parameters[4]
    gap -= contact.margins.numpy()[contact.shapes.numpy()[np.arange(contact.count) % contact.shapes.size]]
    return {
        "masked_active_count": len(keys),
        "masked_active_keys": keys,
        "switched_active_keys": np.flatnonzero(seen == 3),
        "omitted_normal_energy": float(omitted[:, 0].sum()),
        "omitted_friction_energy": float(omitted[:, 1].sum()),
        "omitted_normal_load_sum": float(omitted[:, 2].sum()),
        "omitted_friction_load_sum": float(omitted[:, 3].sum()),
        "max_masked_penetration": float(np.maximum(-gap[keys], 0).max()) if len(keys) else 0.0,
        "max_all_penetration": float(np.maximum(-gap, 0).max()) if len(gap) else 0.0,
    }


def build(position):
    """Reuse frozen assets, mount, imported inertia corrections and all joint terms."""
    old = ASSETS / f"sharpa/close-{position}"
    manifest = json.loads((old / "manifest.json").read_text())
    parameters = load_fixture(REPO / "newton/examples/softbody/sharpa_g1h.json")
    model, loaded = load_hand_cube(
        ASSETS / "sharpa/left_sharpa_wave",
        ASSETS / "sharpa/hand",
        REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz",
        device="cuda:0",
        parameters=parameters,
        position=manifest["object_position_m"],
        resolution=128,
        _mount=manifest["mount"],
    )
    if model.joint_dof_count != 23 or loaded["inertia_corrections"] != manifest["inertia_corrections"]:
        raise ValueError("Frozen DoF/inertia identity changed")
    mapping = joint_mapping(model, loaded["joint_names"], carriage=True)
    dofs = [mapping[n][1] for n in loaded["joint_names"]]
    curve = ClosureTrajectory(
        ASSETS / "sharpa/Ball_catch - Left_hand_motion_rad.csv",
        loaded["joint_names"],
        model.joint_limit_lower.numpy()[dofs],
        model.joint_limit_upper.numpy()[dofs],
        model.joint_velocity_limit.numpy()[dofs],
    )
    width, velocity = np.empty(23), np.empty(23)
    for name, (_, dof, _) in mapping.items():
        cfg = manifest["fixture"]["carriage"] if name == "monolithic_lift" else parameters["joints"][name]
        width[dof], velocity[dof] = cfg["limit_width"], cfg["friction_velocity_scale"]
    terms = SolverMonolithic.JointTerms(
        implicit_pd=True,
        limits=True,
        friction=True,
        limit_width=tuple(width),
        friction_velocity_scale=tuple(velocity),
    )
    return model, manifest, loaded, mapping, curve, terms, old
