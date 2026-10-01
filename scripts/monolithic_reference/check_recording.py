# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify complete P3 state recordings and compare matching trajectories by field."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frames(directory):
    """Validate chunk identities and yield complete frame blocks without concatenation."""
    index = json.loads((directory / "trajectory-index.json").read_text())
    offset = 0
    schema = None
    required = {"joint_q", "joint_qd", "particle_q", "particle_qd", "history.force_generation", "history.force_valid"}
    with np.load(directory / "initial.npz", allow_pickle=False) as initial:
        required |= set(initial.files)
    for chunk in index["chunks"]:
        path = (directory / chunk["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != chunk["sha256"]:
            raise ValueError("Frame path/hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            if not required <= names or (schema is not None and schema != names):
                raise ValueError("Incomplete/changing state schema")
            schema = names
            n = len(data["frame_steps"])
            if n != chunk["count"] or not np.array_equal(data["frame_steps"], np.arange(offset, offset + n)):
                raise ValueError("Missing, shifted or duplicate frame")
            if not np.array_equal(data["frame_times"], np.arange(offset, offset + n) * 0.001):
                raise ValueError("Changed recorded time grid")
            for name in names:
                values = data[name]
                if len(values) != n or (values.dtype.kind in "fci" and not np.isfinite(values).all()):
                    raise ValueError(f"Invalid recorded field: {name}")
            if offset == 0:
                with np.load(directory / "initial.npz", allow_pickle=False) as initial:
                    if any(not np.array_equal(initial[name], data[name][0]) for name in initial.files):
                        raise ValueError("Initial snapshot and first frame disagree")
            generation = data["history.force_generation"].reshape(n)
            valid = data["history.force_valid"].reshape(n)
            if not np.array_equal(generation, np.arange(offset, offset + n)) or not np.array_equal(
                valid, generation > 0
            ):
                raise ValueError("Force validity/generation mismatch")
            publication = data["frame_publication"]
            expected_labels = (
                ("INITIAL", "CONVERGED", "STOPPED_ACCEPTED") if offset == 0 else ("CONVERGED", "STOPPED_ACCEPTED")
            )
            if not np.isin(publication, expected_labels).all() or (offset == 0 and publication[0] != "INITIAL"):
                raise ValueError("Invalid publication label")
            yield {name: data[name] for name in data.files}
            offset += n
    if not index["includes_initial"] or offset != index["total_frames"]:
        raise ValueError("Trajectory index totals disagree")


def check(directory, expected_steps):
    """Check every frame, history field and step label, plus the physical summary."""
    labels = []
    schema = None
    last = None
    for block in frames(directory):
        labels.extend(block["frame_publication"].tolist())
        schema = sorted(block)
        last = {name: values[-1] for name, values in block.items()}
    if len(labels) != expected_steps + 1:
        raise ValueError("Trajectory is not complete")
    with np.load(directory / "final.npz", allow_pickle=False) as final:
        if any(name not in last or not np.array_equal(final[name], last[name]) for name in final.files):
            raise ValueError("Final snapshot and last frame disagree")
    count = 0
    with gzip.open(directory / "steps.jsonl.gz", "rt") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            if row["step"] != count or row["time"] != count * 0.001 or row["publication"] != labels[count]:
                raise ValueError("Step log and trajectory disagree")
            if row["force_valid"] != 1 or row["force_generation"] != count or row["stats"]["rolled_back"]:
                raise ValueError("Invalid publication in step log")
            if row["stats"]["min_det_f"] < 0.2:
                raise ValueError("Published tet below its guard")
    if count != expected_steps:
        raise ValueError("Step log is not complete")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["complete"] or summary["published_steps"] != expected_steps:
        raise ValueError("Summary does not establish complete publication")
    if expected_steps == 5250 and not summary["held_cube"]:
        raise ValueError("Franka did not satisfy original grasp criteria")
    return {
        "status": "PASS",
        "frames": len(labels),
        "steps": count,
        "fields": schema,
        "counts": {label: labels.count(label) for label in ("CONVERGED", "STOPPED_ACCEPTED")},
        "index_sha256": sha(directory / "trajectory-index.json"),
        "steps_sha256": sha(directory / "steps.jsonl.gz"),
    }


_LAYOUT_FIELDS = {"layout_version", "layout_digest", "actor_layout"}
_RIGID_FIELDS = {
    "rigid_com": 3,
    "rigid_rotation": 4,
    "rigid_v_com": 3,
    "rigid_omega": 3,
    "rigid_vsym": 6,
    "rigid_bound_com": 3,
    "rigid_bound_rotation": 4,
}


def _recording_layout(reference, actual, *, native=False):
    """Authenticate canonical ownership from matching saved Model arrays."""
    import warp as wp  # noqa: PLC0415

    from newton._src.solvers.monolithic.actors import ActorTable  # noqa: PLC0415

    class ArrayView:
        def __init__(self, array):
            self.array, self.shape, self.size = array, array.shape, array.size
            self.dtype = wp.int32 if array.dtype == np.int32 else array.dtype

        def numpy(self):
            return self.array

    with np.load(actual / "model.npz", allow_pickle=False) as right:
        if native and not (reference / "model.npz").exists():
            # Native Sharpa references retain input digests and states, not Model
            # arrays. The original digest covers indexed joints, bodies, tet and
            # collision inputs; verify its declaration and the old Model dimensions.
            receipt = json.loads((reference / "receipt.json").read_text())
            with np.load(reference / "initial.npz", allow_pickle=False) as initial:
                if initial["history.config_digest"].tobytes().hex() != receipt["config_digest"]:
                    raise ValueError("Native reference physical identity differs")
                for state_name, model_name in (
                    ("joint_q", "joint_q"),
                    ("joint_qd", "joint_qd"),
                    ("body_q", "body_q"),
                    ("body_qd", "body_qd"),
                    ("particle_q", "particle_q"),
                    ("particle_qd", "particle_qd"),
                ):
                    if initial[state_name].shape != right[model_name].shape:
                        raise ValueError(f"Native reference Model dimensions differ: {state_name}")
        else:
            with np.load(reference / "model.npz", allow_pickle=False) as left:
                if set(left.files) != set(right.files):
                    raise ValueError("Recording Model schemas differ")
                for name in left.files:
                    a, b = left[name], right[name]
                    if a.shape != b.shape or a.dtype != b.dtype:
                        raise ValueError(f"Recording Model layouts differ: {name}")
                    if name == "shape_source_ptr":
                        # Mesh handles vary by process; the physical SDF digest
                        # below authenticates the geometry consumed by this solver.
                        same = True
                    elif name == "_texture_sdf_data":
                        fields = set(a.dtype.names) - {"coarse_texture", "subgrid_texture", "subgrid_start_slots"}
                        same = all(np.array_equal(a[field], b[field]) for field in fields)
                    else:
                        same = np.array_equal(a, b)
                    if not same:
                        raise ValueError(f"Recording physical Model differs: {name}")
        model = SimpleNamespace(**{name: ArrayView(right[name].copy()) for name in right.files})
    model.joint_count = len(model.joint_type.array)
    model.joint_coord_count, model.joint_dof_count = model.joint_q.size, model.joint_qd.size
    model.joint_target_q_start = (
        model.joint_q_start if model.joint_target_q.size == model.joint_coord_count else model.joint_qd_start
    )
    model.articulation_count = model.articulation_end.size
    model.body_count, model.particle_count = model.body_mass.size, model.particle_inv_mass.size
    model.shape_count = model.shape_body.size
    model.world_count = max(0, int(model.body_world.array.max()) + 1)
    model.device = wp.get_device("cpu")
    table = ActorTable.from_model(model)
    if tuple(actor.kind for actor in table.actors) != ("articulation", "tet"):
        raise ValueError("Recording migration requires canonical A+T without rigid actors")
    return table


def _migrate_actor_frame(left, right, table, physical_digest):
    """Audit every new field before projecting schema four onto the old physical fields."""
    added = {"history." + name for name in _LAYOUT_FIELDS | _RIGID_FIELDS.keys()}
    if not added | {"history.schema_version", "history.precision", "history.config_digest"} <= right.keys():
        raise ValueError("Missing actor recording metadata")
    version = left["history.schema_version"]
    if (
        version.dtype != np.int32
        or version.shape[1:] != (1,)
        or not np.all(version == version[0])
        or version[0, 0] not in (2, 3)
    ):
        raise ValueError("Invalid legacy actor recording schema")
    old_version = int(version[0, 0])
    dtype = left["joint_q"].dtype
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)) or (old_version == 2 and dtype != np.float64):
        raise ValueError("Invalid legacy actor recording precision")
    precision = 64 if dtype == np.float64 else 32
    n = len(version)
    if old_version == 3 and not np.array_equal(left["history.precision"], np.full((n, 1), precision, np.int32)):
        raise ValueError("Invalid schema three precision")
    metadata = {
        "history.schema_version": np.array([4], np.int32),
        "history.precision": np.array([precision], np.int32),
        "history.layout_version": np.array([1], np.int32),
        "history.layout_digest": np.frombuffer(bytes.fromhex(table.identity), dtype=np.uint8),
        "history.actor_layout": np.frombuffer(table.layout_payload(), dtype=np.uint8),
    }
    for name, expected in metadata.items():
        value = right[name]
        if value.dtype != expected.dtype or value.shape != (n, *expected.shape) or not np.all(value == expected):
            raise ValueError(f"Invalid actor recording metadata: {name}")
    for name, width in _RIGID_FIELDS.items():
        value = right["history." + name]
        if value.shape != (n, 0, width) or value.dtype != dtype:
            raise ValueError(f"Actor recording requires empty native rigid field: {name}")
    extra = {"history." + name for name in _LAYOUT_FIELDS | _RIGID_FIELDS.keys()}
    if old_version == 2:
        extra.add("history.precision")
    known = {
        "joint_q",
        "joint_qd",
        "body_q",
        "body_qd",
        "body_f",
        "particle_q",
        "particle_qd",
        "particle_f",
        "frame_steps",
        "frame_times",
        "frame_publication",
        "frame_target_q",
        "frame_target_qd",
    } | {
        "history." + name
        for name in (
            "joint_v",
            "joint_omega",
            "joint_vsym",
            "link_v_com",
            "link_omega",
            "link_vsym",
            "bound_q",
            "bound_qd",
            "last_dt",
            "schema_version",
            "valid",
            "config_digest",
            "bound_x",
            "bound_v",
            "contact_particle_force",
            "contact_body_force",
            "contact_joint_force",
            "force_generation",
            "force_valid",
            "precision",
        )
    }
    required = {name for name in known if not name.startswith("frame_") and name != "history.precision"}
    if not required <= left.keys() or not left.keys() <= known or right.keys() != left.keys() | extra:
        raise ValueError("Unexpected actor recording fields")
    digest = left["history.config_digest"]
    if (
        digest.dtype != np.uint8
        or digest.shape != (n, 32)
        or not np.all(digest == physical_digest)
        or not np.array_equal(digest, right["history.config_digest"])
    ):
        raise ValueError("Recording physical input digest differs")
    projected = {name: value for name, value in right.items() if name not in extra}
    projected["history.schema_version"] = version.copy()
    return projected, f"{old_version} -> 4 (canonical A+T, FP{precision})"


def compare(
    reference, actual, envelopes=None, *, legacy_fp64=False, native_controls=None, actor_layout_migration=False
):
    """Compare every same-configuration frame; never infer a trajectory tolerance."""
    if type(actor_layout_migration) is not bool:
        raise ValueError("actor_layout_migration must be bool")
    errors = {}
    exact = True
    within_envelope = True
    count = 0
    migrated = False
    actor_migration = None
    table = None
    if native_controls is None:
        reference_frames = frames(reference)
    else:
        from scripts.monolithic_reference.native_reference import frames as native_frames  # noqa: PLC0415

        reference_frames = native_frames(reference, native_controls)
    for left, right_block in zip(reference_frames, frames(actual), strict=True):
        right = right_block
        if (
            actor_layout_migration
            and np.all(right["history.schema_version"] == 4)
            and np.all(left["history.schema_version"] < 4)
        ):
            if table is None:
                table = _recording_layout(reference, actual, native=native_controls is not None)
                with np.load(reference / "initial.npz", allow_pickle=False) as initial:
                    physical_digest = initial["history.config_digest"].copy()
            right, actor_migration = _migrate_actor_frame(left, right, table, physical_digest)
        if legacy_fp64 and "history.precision" not in left:
            if right.keys() != left.keys() | {"history.precision"}:
                raise ValueError("Unexpected legacy FP64 comparison fields")
            if (
                not np.all(left["history.schema_version"] == 2)
                or not np.all(right["history.schema_version"] == 3)
                or right["history.precision"].dtype != np.int32
                or right["history.precision"].shape != right["history.schema_version"].shape
                or not np.all(right["history.precision"] == 64)
                or left["joint_q"].dtype != np.float64
                or right["joint_q"].dtype != np.float64
            ):
                raise ValueError("Invalid legacy FP64 recording migration")
            right = {name: value for name, value in right.items() if name != "history.precision"}
            right["history.schema_version"] = right["history.schema_version"] - np.int32(1)
            migrated = True
        if left.keys() != right.keys():
            raise ValueError("Comparison schemas differ")
        count += len(left["frame_steps"])
        for name in left:
            a, b = left[name], right[name]
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Comparison layouts differ: {name}")
            same = np.array_equal(a, b)
            exact &= same
            row = errors.setdefault(name, {"exact": True, "max_absolute": 0.0, "sum_squared": 0.0, "elements": 0})
            row["exact"] &= same
            if a.dtype.kind in "fci":
                delta = a.astype(np.float64) - b.astype(np.float64)
                row["max_absolute"] = max(row["max_absolute"], float(np.max(np.abs(delta), initial=0)))
                row["sum_squared"] += float(np.sum(delta * delta))
                row["elements"] += delta.size
            limit = (envelopes or {}).get(name, 0.0) if a.dtype.kind == "f" else 0.0
            row["absolute_tolerance"] = limit
            row["within_tolerance"] = row["exact"] or (a.dtype.kind == "f" and row["max_absolute"] <= limit)
            within_envelope &= row["within_tolerance"]
    for row in errors.values():
        row["rms"] = (row.pop("sum_squared") / max(1, row["elements"])) ** 0.5
    status = "PASS_EXACT" if exact else "PASS_FROZEN_ENVELOPE" if within_envelope else "NOT_READY_TOLERANCE"
    return {
        "status": status,
        "frames": count,
        "fields": errors,
        "schema_migration": actor_migration or ("2 -> 3 (FP64)" if migrated else None),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--protocol", type=Path, help="Previously frozen default-only comparison protocol")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--actor-layout-migration", action="store_true", help="Audit and bridge canonical A+T schema 2/3 to 4"
    )
    args = parser.parse_args()
    result = check(args.directory, args.steps)
    if args.reference:
        result["reference"] = check(args.reference, args.steps)
        envelopes = None
        if args.protocol:
            protocol = json.loads(args.protocol.read_text())
            scene = "franka" if args.steps == 5250 else "sharpa"
            envelopes = protocol["default_absolute_envelopes"][scene]
            result["protocol_sha256"] = sha(args.protocol)
        result["comparison"] = compare(
            args.reference, args.directory, envelopes, actor_layout_migration=args.actor_layout_migration
        )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "frames", "counts")}))
    return int(not result.get("comparison", {}).get("status", "PASS_EXACT").startswith("PASS"))


if __name__ == "__main__":
    raise SystemExit(main())
