# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read saved monolithic states, check them against a rebuilt scene and import them by field name.

States are copied by field name, never through the checkpoint path, so a newer
solver configuration can continue an older state. Snapshots are single saved
frames with the model arrays needed to prove they belong to the rebuilt scene.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import warp as wp

# Solver and layout identities belong to the new solver, never to the saved state.
IDENTITY_FIELDS = frozenset(
    "history." + name
    for name in ("schema_version", "config_digest", "precision", "layout_version", "layout_digest", "actor_layout")
)
MODEL_ARRAYS = (
    "particle_q",
    "particle_mass",
    "tet_indices",
    "tet_poses",
    "tri_indices",
    "edge_indices",
    "body_mass",
    "body_inertia",
    "body_com",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_X_p",
    "joint_X_c",
    "joint_axis",
    "shape_type",
    "shape_body",
    "shape_scale",
    "shape_transform",
)
# Scene inputs only; body poses are forward kinematics evaluated in the solver precision.
INITIAL_COORDINATES = ("particle_q", "joint_q")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path, step=None):
    """Return one flat saved state and the directory holding its recording sidecars.

    ``path`` is a checkpoint-like ``.npz`` file or a recording directory; a recording
    needs ``step``, where step 0 is its ``initial.npz``.
    """
    path = Path(path)
    if path.is_file():
        if step is not None:
            raise ValueError("A state file holds exactly one state")
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}, None
    if step is None or step < 0:
        raise ValueError("A recording requires a nonnegative step")
    if step == 0:
        return read_source(path / "initial.npz")[0], path
    index = path / "trajectory-index.json"
    if index.is_file():
        # Example recordings label every frame with its step.
        for chunk in json.loads(index.read_text())["chunks"]:
            with np.load(path / chunk["path"], allow_pickle=False) as data:
                hits = np.flatnonzero(data["frame_steps"] == step)
                if len(hits):
                    if _sha(path / chunk["path"]) != chunk["sha256"]:
                        raise ValueError("Recording chunk identity differs")
                    return {k: data[k][hits[0]] for k in data.files if not k.startswith("frame_")}, path
        raise ValueError(f"Recording lacks step {step}")
    # Native state chunks list consecutive published steps after the initial state.
    first = 1
    for chunk in json.loads((path / "receipt.json").read_text())["chunks"]:
        if first <= step < first + chunk["frames"]:
            if _sha(path / chunk["file"]) != chunk["sha256"]:
                raise ValueError("Recording chunk identity differs")
            with np.load(path / chunk["file"], allow_pickle=False) as data:
                return {k: data[k][step - first] for k in data.files}, path
        first += chunk["frames"]
    raise ValueError(f"Recording lacks step {step}")


def check_scene(solver, initial, source, directory):
    """Return the checks proving the saved state belongs to the rebuilt scene and solver layout."""
    checks = []
    layout = source.get("history.actor_layout")
    if layout is not None:
        if layout.tobytes() != solver._actors.layout_payload():
            raise ValueError("Saved actor layout differs from the rebuilt scene")
        checks.append("actor_layout")
    if directory is not None and (directory / "model.npz").is_file():
        with np.load(directory / "model.npz", allow_pickle=False) as saved:
            for name in MODEL_ARRAYS:
                actual = getattr(solver.model, name, None)
                if (name in saved.files) != isinstance(actual, wp.array) or (
                    name in saved.files and not np.array_equal(saved[name], actual.numpy())
                ):
                    raise ValueError(f"Recorded model differs from the rebuilt scene: {name}")
        checks.append("model_arrays")
    if directory is not None and (directory / "initial.npz").is_file():
        with np.load(directory / "initial.npz", allow_pickle=False) as saved:
            for name in INITIAL_COORDINATES:
                actual = getattr(initial, name).numpy()
                if saved[name].shape != actual.shape or not np.array_equal(saved[name].astype(actual.dtype), actual):
                    raise ValueError(f"Recorded initial state differs from the rebuilt scene: {name}")
        checks.append("initial_state")
    if not checks:
        raise ValueError("Saved state carries no layout, model or initial state to validate the scene")
    return checks


def _contact_free(name, source):
    """Whether an absent published contact force must be zero because no saved contact force is nonzero."""
    forces = [value for key, value in source.items() if key.startswith("history.contact_")]
    return name.startswith("history.contact_") and bool(forces) and not any(np.any(value) for value in forces)


def import_state(solver, state, source):
    """Write saved coordinates and history by name into a validated copy, then replace ``state``."""
    states = solver._states
    scratch = states.copy(state)
    history = getattr(scratch, states.namespace)
    targets = {name: value for name, value in vars(scratch).items() if isinstance(value, wp.array)}
    targets.update({"history." + name: getattr(history, name) for name in states.schema})
    unknown = source.keys() - targets.keys() - IDENTITY_FIELDS
    if unknown:
        raise ValueError(f"Saved state has unknown fields: {sorted(unknown)}")
    report = {"imported": [], "converted": [], "absent_empty": [], "zero_filled": []}
    for name, array in targets.items():
        if name in IDENTITY_FIELDS:
            continue
        expected = array.numpy()
        if name not in source:
            if not expected.size:
                report["absent_empty"].append(name)
            elif _contact_free(name, source):
                # Older schemas lack newer contact outputs; without any saved contact they are zero.
                array.zero_()
                report["zero_filled"].append(name)
            else:
                raise ValueError(f"Saved state lacks {name}")
            continue
        value = source[name]
        if value.shape != expected.shape or value.dtype.kind != expected.dtype.kind:
            raise ValueError(
                f"Saved field {name} has {value.dtype}{value.shape}, expected {expected.dtype}{expected.shape}"
            )
        if value.dtype != expected.dtype:
            report["converted"].append(name)
        array.assign(value.astype(expected.dtype))
        report["imported"].append(name)
    states.validate(scratch)
    state.assign(scratch)
    return report


SNAPSHOT_FORMAT = "monolithic-snapshot/v1"


def _frame_publication(recording, step):
    """Return the publication label of one frame of an example-format recording."""
    for chunk in json.loads((recording / "trajectory-index.json").read_text())["chunks"]:
        with np.load(recording / chunk["path"], allow_pickle=False) as data:
            hits = np.flatnonzero(data["frame_steps"] == step)
            if len(hits):
                return str(data["frame_publication"][hits[0]])
    raise ValueError(f"Recording lacks step {step}")


def write_snapshot(recording, step, destination, *, source_commit):
    """Write one recorded frame and its scene arrays to ``destination``.npz with a .json sidecar."""
    recording, destination = Path(recording), Path(destination)
    values, directory = read_source(recording, step)
    with np.load(directory / "model.npz", allow_pickle=False) as model:
        scene = {"model." + name: model[name] for name in MODEL_ARRAYS if name in model.files}
    if not scene:
        raise ValueError("Recording model carries none of the scene arrays")
    path = destination.with_suffix(".npz")
    np.savez_compressed(path, **values, **scene)
    meta = {
        "format": SNAPSHOT_FORMAT,
        "source_recording": str(recording),
        "step": int(step),
        "publication": "INITIAL" if step == 0 else _frame_publication(recording, step),
        "source_commit": source_commit,
        "sha256": _sha(path),
        "model_arrays": sorted(name[len("model.") :] for name in scene),
    }
    destination.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def load_snapshot(path, sha256):
    """Return (state fields, model arrays) after verifying the snapshot file identity."""
    path = Path(path)
    if _sha(path) != sha256:
        raise ValueError(f"Snapshot sha256 differs: {path}")
    with np.load(path, allow_pickle=False) as data:
        state = {k: data[k] for k in data.files if not k.startswith("model.")}
        model = {k[len("model.") :]: data[k] for k in data.files if k.startswith("model.")}
    if not model:
        raise ValueError("Snapshot carries no model arrays to validate the scene")
    return state, model


def check_snapshot_scene(solver, model_values):
    """Require every saved model array to equal the rebuilt scene; return the checked names."""
    for name, saved in model_values.items():
        actual = getattr(solver.model, name, None)
        if not isinstance(actual, wp.array) or not np.array_equal(saved, actual.numpy()):
            raise ValueError(f"Snapshot model differs from the rebuilt scene: {name}")
    return sorted(model_values)
