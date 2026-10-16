# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit native mixed-actor recordings and explicitly compare semantic roles."""

import gzip
import json
from pathlib import Path

import numpy as np

from newton._src.solvers.monolithic.actors import ActorTable
from newton.examples.softbody.monolithic_recording import state_arrays
from scripts.monolithic_reference.check_recording import sha

FRAME_FIELDS = {
    "frame_steps",
    "frame_times",
    "frame_publication",
    "frame_target_q",
    "frame_target_qd",
    "frame_control_q",
    "frame_control_qd",
}
IDENTITY_FIELDS = {"history.config_digest", "history.layout_digest", "history.actor_layout"}


def _rotation(q):
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.moveaxis(
        np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        ),
        (0, 1),
        (-2, -1),
    )


def _equal(a, b, name):
    if np.asarray(a).shape != np.asarray(b).shape or not np.array_equal(a, b):
        raise ValueError(f"Recorded identity or binding differs: {name}")


def _close(a, b, name, *, atol=1e-10):
    if np.asarray(a).shape != np.asarray(b).shape or not np.allclose(a, b, atol=atol, rtol=1e-10):
        raise ValueError(f"Native state consistency differs: {name}")


def _context(directory, expected_run=None):
    from scripts.monolithic_reference import run_actor_coupling as scene  # noqa: PLC0415

    metadata = json.loads((directory / "configuration.json").read_text())
    schema = metadata.get("schema")
    pooled = schema == "actor-coupling-recording/v2"
    if metadata.keys() != {
        "schema",
        "case",
        "fixture_sha256",
        "input_scope",
        "before",
        "start_step",
        "end_step",
        "config_digest",
        "layout",
        "roles",
        "sampling",
        "execution",
    } | ({"algorithm"} if pooled else set()):
        raise ValueError("Unknown or missing actor recording metadata")
    if (
        schema not in ("actor-coupling-recording/v1", "actor-coupling-recording/v2")
        or metadata["case"] != scene.load_case()
        or metadata["fixture_sha256"] != sha(scene.CASE)
        or type(metadata["before"]) is not bool
        or type(metadata["start_step"]) is not int
        or type(metadata["end_step"]) is not int
        or not 0 <= metadata["start_step"] < metadata["end_step"] <= metadata["case"]["steps"]
    ):
        raise ValueError("Recorded actor input or clock differs")
    if metadata["input_scope"] not in ("frozen", "component") or (
        metadata["input_scope"] == "component" and expected_run is None
    ):
        raise ValueError("Component recording requires its explicit Model")
    execution = metadata["execution"]
    if (
        execution.keys() != {"device", "mode", "graph"}
        or execution["device"] not in ("cpu", "cuda:0")
        or execution["mode"] not in ("diagnostic", "production")
        or type(execution["graph"]) is not bool
        or (execution["graph"] and (execution["device"] == "cpu" or execution["mode"] != "production"))
    ):
        raise ValueError("Unsupported actor recording execution route")
    run = expected_run
    algorithm = metadata.get("algorithm", {})
    if pooled and (
        type(algorithm) is not dict
        or set(algorithm) != {"contact_dataflow", "contact_factor_capacity", "spatial"}
        or algorithm["contact_dataflow"] not in ("pooled", "spatial")
        or type(algorithm["contact_factor_capacity"]) is not int
        or algorithm["contact_factor_capacity"] < 0
    ):
        raise ValueError("Invalid recorded contact algorithm")
    if pooled and algorithm["contact_dataflow"] == "spatial":
        spatial = algorithm["spatial"]
        if (
            type(spatial) is not dict
            or set(spatial) != {"candidate_capacity", "law_margin"}
            or type(spatial["candidate_capacity"]) is not int
            or not 0 <= spatial["candidate_capacity"] < 2**31
            or type(spatial["law_margin"]) not in (int, float)
            or not np.isfinite(spatial["law_margin"])
            or spatial["law_margin"] < 0
        ):
            raise ValueError("Invalid recorded spatial parameters")
    elif pooled and algorithm["spatial"] is not None:
        raise ValueError("Complete-candidate recordings cannot declare spatial parameters")
    if run is None:
        # Batch/resource identity is device-specific for pooled routes. Rebuild
        # the recorded route explicitly rather than weakening its digest guard.
        device = execution["device"] if pooled else "cpu"
        run = scene.make_solver(
            scene.build_model(device=device, before=metadata["before"]),
            graph=execution["graph"] if pooled else False,
            mode=execution["mode"] if pooled else None,
            **(algorithm if pooled else {"contact_dataflow": "legacy"}),
        )
    if pooled != run._contact_pool or (pooled and algorithm != scene.contact_algorithm(run)):
        raise ValueError("Recorded contact algorithm or resources differ")
    model = run.model
    if run._states.precision_bits != 64:
        raise ValueError("Actor coupling recording requires native FP64")
    with np.load(directory / "model.npz", allow_pickle=False) as archive:
        expected = scene.model_arrays(model)
        if set(archive.files) != expected.keys():
            raise ValueError("Recorded Model schema differs")
        for name, value in expected.items():
            if archive[name].dtype != value.dtype:
                raise ValueError(f"Recorded Model dtype differs: {name}")
            _equal(archive[name], value, name)
    _equal(metadata["config_digest"], run._kinematics.digest.hex(), "physical digest")
    canonical = ActorTable.from_model(model)
    ids = [(a.kind, a.source_index) for a in canonical.actors]
    order = [ids.index((a["kind"], a["source_index"])) for a in metadata["layout"]["actors"]]
    table = ActorTable.from_model(model, actor_order=order)
    if metadata["layout"] != json.loads(table.layout_payload()) or metadata["roles"] != scene.roles(model, table):
        raise ValueError("Recorded Actor ownership or mapping differs")
    expected_sampling = scene.sampling(run)
    permutation = np.zeros(table.scalar_dof_count, dtype=np.int32)
    for actor in run._actors.actors:
        target = next(a for a in table.actors if (a.kind, a.source_index) == (actor.kind, actor.source_index))
        permutation[actor.dof_start : actor.dof_start + actor.dof_count] = np.arange(
            target.dof_start, target.dof_start + target.dof_count
        )
    if not pooled:
        expected_sampling["columns"] = permutation[expected_sampling["columns"]].tolist()
    if metadata["sampling"] != expected_sampling:
        raise ValueError("Recorded sampling identity or capacity differs")
    template = state_arrays(run.create_state(dt=metadata["case"]["dt"]))
    template["history.actor_layout"] = np.frombuffer(table.layout_payload(), dtype=np.uint8)
    template["history.layout_digest"] = np.frombuffer(bytes.fromhex(table.identity), dtype=np.uint8)
    return metadata, run, template, model.tet_poses.numpy().astype(np.float64)


def _validate_state(frame, metadata, run, template, inverse_rest):
    model, role_map = run.model, metadata["roles"]
    for name, value in template.items():
        actual = frame[name]
        if (
            actual.shape != value.shape
            or actual.dtype != value.dtype
            or (actual.dtype.kind in "fci" and not np.isfinite(actual).all())
        ):
            raise ValueError(f"Invalid recorded state field: {name}")
    for name in IDENTITY_FIELDS | {
        "history.schema_version",
        "history.precision",
        "history.layout_version",
        "history.valid",
        "history.last_dt",
    }:
        _equal(frame[name], template[name], name)
    for state, bound in (
        ("joint_q", "history.bound_q"),
        ("joint_qd", "history.bound_qd"),
        ("particle_q", "history.bound_x"),
        ("particle_qd", "history.bound_v"),
        ("history.rigid_com", "history.rigid_bound_com"),
        ("history.rigid_rotation", "history.rigid_bound_rotation"),
    ):
        _equal(frame[state], frame[bound], bound)
    for name in ("body_f", "particle_f"):
        _equal(frame[name], template[name], "external input." + name)
    step = int(frame["frame_steps"])
    _equal(frame["history.force_generation"], np.array([step], np.int64), "force generation")
    _equal(frame["history.force_valid"], np.array([step > 0], np.int32), "force validity")
    fixed = model.particle_inv_mass.numpy() == 0
    _equal(frame["particle_q"][fixed], model.particle_q.numpy().astype(np.float64)[fixed], "fixed positions")
    _equal(frame["particle_qd"][fixed], np.zeros((fixed.sum(), 3)), "fixed velocities")
    tets = model.tet_indices.numpy()
    x = frame["particle_q"]
    current = np.swapaxes(x[tets[:, 1:]] - x[tets[:, :1]], 1, 2)
    determinant = np.linalg.det(current @ inverse_rest)
    if np.min(determinant) <= metadata["case"]["gates"]["min_det_f"]:
        raise ValueError("Recorded tet violates its determinant guard")
    for role in role_map.values():
        if role.get("kind") == "articulation":
            position = model.joint_X_p.numpy()[role["joint"], :3].astype(float).copy()
            position[0] += frame["joint_q"][role["coord"]][0]
            velocity = np.array([frame["joint_qd"][role["dof"]][0], 0, 0])
            _close(frame["body_q"][role["body"], :3], position, "arm display position", atol=1e-7)
            _close(frame["body_qd"][role["body"], :3], velocity, "arm display velocity", atol=1e-6)
            _close(frame["history.link_v_com"][0], velocity, "arm native velocity")
        if role.get("kind") != "rigid":
            continue
        i, body, joint = role["rigid_row"], role["body"], role["joint"]
        q = frame["history.rigid_rotation"][i]
        if abs(np.linalg.norm(q) - 1) > 1e-10:
            raise ValueError("Invalid native rigid quaternion")
        r = _rotation(q)
        c = frame["history.rigid_com"][i]
        origin = c - r @ model.body_com.numpy()[body]
        parent, child = model.joint_X_p.numpy()[joint].astype(float), model.joint_X_c.numpy()[joint].astype(float)
        parent[3:] /= np.linalg.norm(parent[3:])
        child[3:] /= np.linalg.norm(child[3:])
        rp, rc = _rotation(parent[3:]), _rotation(child[3:])
        jq = frame["joint_q"][role["coord"]]
        _close(jq[:3], rp.T @ (origin + r @ child[:3] - parent[:3]), "FREE origin")
        _close(_rotation(jq[3:]), rp.T @ r @ rc, "FREE rotation")
        velocity, omega = frame["history.rigid_v_com"][i], frame["history.rigid_omega"][i]
        _close(frame["joint_qd"][role["dof"]], np.r_[rp.T @ velocity, rp.T @ omega], "FREE twist")
        _close(frame["body_q"][body, :3], origin, "body display origin", atol=1e-7)
        _close(_rotation(frame["body_q"][body, 3:]), r, "body display rotation", atol=1e-6)
        _close(frame["body_qd"][body], np.r_[velocity, omega], "body display velocity", atol=1e-6)
        sym = frame["history.rigid_vsym"][i]
        wx, wy, wz = omega
        rate = np.array(
            [
                [sym[0], sym[3] - wz, sym[4] + wy],
                [sym[3] + wz, sym[1], sym[5] - wx],
                [sym[4] - wy, sym[5] + wx, sym[2]],
            ]
        )
        relative = np.eye(3) + (1 if step == 0 else -1) * metadata["case"]["dt"] * rate
        _close(relative @ relative.T, np.eye(3), "rigid discrete history")
        _close(
            frame["history.contact_joint_force"][role["dof"]], frame["history.contact_body_force"][body], "FREE wrench"
        )
    return float(np.min(determinant))


def actor_frames(directory, metadata):
    """Read strict absolute-clock chunks without changing old recording semantics."""
    from scripts.monolithic_reference.run_actor_coupling import targets  # noqa: PLC0415

    index = json.loads((directory / "trajectory-index.json").read_text())
    with np.load(directory / "initial.npz", allow_pickle=False) as archive:
        required = set(archive.files) | FRAME_FIELDS
        initial = dict(archive)
    offset = metadata["start_step"]
    for chunk in index["chunks"]:
        path = (directory / chunk["path"]).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != chunk["sha256"]:
            raise ValueError("Actor frame path/hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError("Incomplete or unknown actor frame fields")
            n = chunk["count"]
            if type(n) is not int or n <= 0:
                raise ValueError("Invalid actor chunk count")
            steps = np.arange(offset, offset + n, dtype=np.int64)
            _equal(archive["frame_steps"], steps, "frame steps")
            _equal(archive["frame_times"], steps * metadata["case"]["dt"], "frame times")
            for name in FRAME_FIELDS:
                expected_dtype = (
                    np.dtype("U16")
                    if name == "frame_publication"
                    else np.dtype(np.int64 if name == "frame_steps" else np.float64)
                )
                if archive[name].dtype != expected_dtype:
                    raise ValueError("Invalid actor frame dtype")
            for name in archive.files:
                if archive[name].shape[0] != n:
                    raise ValueError("Invalid actor frame count")
            for i, step in enumerate(steps):
                frame = {name: archive[name][i] for name in archive.files}
                q, qd, _ = targets(metadata["case"], int(step))
                _equal(frame["frame_target_q"], q, "target position")
                _equal(frame["frame_target_qd"], qd, "target velocity")
                _equal(frame["frame_control_q"], float(np.float32(q)), "actual Control position")
                _equal(frame["frame_control_qd"], float(np.float32(qd)), "actual Control velocity")
                label = "INITIAL" if step == 0 else "RESTORED" if step == metadata["start_step"] else "CONVERGED"
                _equal(frame["frame_publication"], label, "publication")
                if step == metadata["start_step"]:
                    for name, value in initial.items():
                        _equal(frame[name], value, "initial." + name)
                yield frame
        offset += n
    if (
        not index["includes_initial"]
        or offset - metadata["start_step"] != index["total_frames"]
        or offset != metadata["end_step"] + 1
    ):
        raise ValueError("Actor trajectory is incomplete")


def _force_consistency(frame, row, metadata, model):
    body_force = np.zeros_like(frame["history.contact_body_force"])
    tet_wrench = np.zeros(6)
    for pair in row["pairs"]:
        for field in ("source", "target"):
            shape = pair[field]
            if type(shape) is not int or shape < (-1 if field == "source" else 0) or shape >= model.shape_count:
                raise ValueError("Invalid pair shape identity")
            role = "tet" if shape == -1 else model.body_label[int(model.shape_body.numpy()[shape])]
            _equal(pair[field + "_role"], role, "pair role")
        for name, values in zip((pair["source_role"], pair["target_role"]), pair["world_wrench"], strict=True):
            wrench = np.asarray(values)
            if name == "tet":
                tet_wrench += wrench
                continue
            role = metadata["roles"][name]
            if role["kind"] == "rigid":
                com = frame["history.rigid_com"][role["rigid_row"]]
            else:
                # The frozen scene has one x-prismatic link with identity rotation.
                j = role["joint"]
                com = model.joint_X_p.numpy()[j, :3].astype(float) + model.body_com.numpy()[role["body"]]
                com[0] += frame["joint_q"][role["coord"]][0]
            body_force[role["body"], :3] += wrench[:3]
            body_force[role["body"], 3:] += wrench[3:] - np.cross(com, wrench[:3])
    _close(frame["history.contact_body_force"], body_force, "pair/body wrench", atol=1e-9)
    force = frame["history.contact_particle_force"]
    _close(
        np.r_[force.sum(axis=0), np.cross(frame["particle_q"], force).sum(axis=0)],
        tet_wrench,
        "pair/tet wrench",
        atol=1e-9,
    )
    arm = metadata["roles"]["pusher"]
    _close(
        frame["history.contact_joint_force"][arm["dof"]], body_force[arm["body"], :1], "prismatic reaction", atol=1e-9
    )


def _penetration(frame, metadata, run):
    """Recompute analytic box gaps from native authority and authenticated samples."""
    model, role_map = run.model, metadata["roles"]
    transforms = model.shape_transform.numpy().astype(float)
    com = model.body_com.numpy().astype(float)
    poses = {}
    for role in role_map.values():
        if role.get("kind") == "rigid":
            i = role["rigid_row"]
            r = _rotation(frame["history.rigid_rotation"][i])
            origin = frame["history.rigid_com"][i] - r @ com[role["body"]]
        elif role.get("kind") == "articulation":
            r = np.eye(3)
            origin = model.joint_X_p.numpy()[role["joint"], :3].astype(float).copy()
            origin[0] += frame["joint_q"][role["coord"]][0]
        else:
            continue
        for shape in role["shape"]:
            poses[shape] = (origin + r @ transforms[shape, :3], r @ _rotation(transforms[shape, 3:]))
    fields = metadata["sampling"]["fields"]
    points = np.asarray(fields["actor_contact.points"])
    keys = np.asarray(fields["actor_contact.keys"], dtype=int)
    offsets = np.asarray(fields["actor_contact.offsets"])
    triangles = run._contact.triangles.numpy()
    scale = model.shape_scale.numpy().astype(float)
    distances = []
    for key, (source, target, face, slot) in enumerate(keys):
        if source == -1:
            bary = np.full(3, 1 / 6)
            bary[slot] = 2 / 3
            point = bary @ frame["particle_q"][triangles[face]]
        else:
            origin, rotation = poses[source]
            point = origin + rotation @ points[key]
        origin, rotation = poses[target]
        y = np.abs(rotation.T @ (point - origin)) - scale[target]
        distances.append(np.linalg.norm(np.maximum(y, 0)) + min(y.max(), 0) - offsets[key])
    return max(0.0, -min(distances, default=0.0))


def check_actor(directory, *, expected_run=None, require_complete=True):
    """Authenticate each Model/layout and audit every native field and logged step."""
    from scripts.monolithic_reference import run_actor_coupling as scene  # noqa: PLC0415

    directory = Path(directory)
    metadata, run, template, inverse_rest = _context(directory, expected_run)
    if require_complete and (
        metadata["input_scope"] != "frozen"
        or metadata["execution"] != {"device": "cuda:0", "mode": "production", "graph": True}
    ):
        raise ValueError("Full actor qualification requires frozen production/graph execution")
    rows, last, count, previous = [], None, 0, None
    with gzip.open(directory / "steps.jsonl.gz", "rt") as log:
        for frame in actor_frames(directory, metadata):
            if frame.keys() != template.keys() | FRAME_FIELDS:
                raise ValueError("Unknown or missing complete actor state fields")
            determinant = _validate_state(frame, metadata, run, template, inverse_rest)
            step = int(frame["frame_steps"])
            if step == 0 and metadata["input_scope"] == "frozen":
                for name, value in template.items():
                    _equal(frame[name], value, "frozen initial." + name)
            if previous is not None:
                h = metadata["case"]["dt"]
                _close(frame["particle_qd"], (frame["particle_q"] - previous["particle_q"]) / h, "tet velocity history")
                for role in metadata["roles"].values():
                    if role.get("kind") != "rigid":
                        continue
                    i = role["rigid_row"]
                    r, rn = (
                        _rotation(frame["history.rigid_rotation"][i]),
                        _rotation(previous["history.rigid_rotation"][i]),
                    )
                    rate = (r - rn) @ r.T / h
                    omega = np.array([rate[2, 1] - rate[1, 2], rate[0, 2] - rate[2, 0], rate[1, 0] - rate[0, 1]]) / 2
                    symmetric = (rate + rate.T) / 2
                    _close(frame["history.rigid_omega"][i], omega, "rotation rate history")
                    _close(
                        frame["history.rigid_vsym"][i],
                        symmetric[[0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]],
                        "symmetric rate history",
                    )
                    _close(
                        frame["history.rigid_v_com"][i],
                        (frame["history.rigid_com"][i] - previous["history.rigid_com"][i]) / h,
                        "COM velocity history",
                    )
            if step != metadata["start_step"]:
                line = log.readline()
                if not line:
                    raise ValueError("Actor step log is incomplete")
                row = json.loads(line)
                if row["step"] != step or row["time"] != step * metadata["case"]["dt"]:
                    raise ValueError("Actor log clock differs")
                q, qd, phase = scene.targets(metadata["case"], step)
                _equal([row["target_q"], row["target_qd"]], [q, qd], "logged control")
                _equal(
                    [row["control_target_q"], row["control_target_qd"]],
                    [float(np.float32(q)), float(np.float32(qd))],
                    "logged actual Control",
                )
                _equal(row["phase"], phase, "logged phase")
                if (run._contact_pool and "contact_pool" not in row["stats"]["variational"]) or (
                    run._contact_spatial and "contact_spatial" not in row["stats"]["variational"]
                ):
                    raise ValueError("Missing recorded contact capacity diagnostics")
                scene.validate_row(row, metadata["case"])
                _close(row["stats"]["min_det_f"], determinant, "logged determinant", atol=1e-8)
                displacement = np.linalg.norm(frame["particle_q"] - run.model.particle_q.numpy(), axis=1).max()
                _close(row["tet_displacement"], displacement, "logged deformation")
                penetration = _penetration(frame, metadata, run)
                _close(row["penetration"], penetration, "logged penetration", atol=1e-10)
                if penetration > metadata["case"]["gates"]["max_penetration_m"]:
                    raise ValueError("Native geometry exceeds its penetration guard")
                _force_consistency(frame, row, metadata, run.model)
                rows.append(row)
            last, count, previous = frame, count + 1, frame
        if log.readline():
            raise ValueError("Actor step log has extra rows")
    with np.load(directory / "final.npz", allow_pickle=False) as final:
        if set(final.files) != template.keys():
            raise ValueError("Incomplete final actor snapshot")
        for name in final.files:
            _equal(final[name], last[name], "final." + name)
    summary = json.loads((directory / "summary.json").read_text())
    expected_status = "PASS" if metadata["end_step"] == metadata["case"]["steps"] else "PASS_PREFIX"
    if (
        summary["status"] != expected_status
        or summary["error"] is not None
        or summary["published_steps"] != count - 1
        or summary["checked_steps"] != count - 1
        or summary["attempted_steps"] != count - 1
        or summary["qualification_failures"] != 0
        or summary["complete"] != (expected_status == "PASS")
        or summary["start_step"] != metadata["start_step"]
        or summary["end_step"] != metadata["end_step"]
        or summary["counts"] != {"CONVERGED": count - 1, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0}
    ):
        raise ValueError("Actor summary does not establish complete convergence")
    hold = scene.hold_gates(rows, metadata["case"]) if expected_status == "PASS" else None
    if require_complete and expected_status != "PASS":
        raise ValueError("A short prefix is not full actor qualification")
    if require_complete and run._contact_spatial and "full_system" not in summary:
        raise ValueError("Spatial actor qualification requires a complete system residual report")
    if "full_system" in summary:
        from scripts.monolithic_reference.contact_system_audit import check_system_report  # noqa: PLC0415

        check_system_report(directory, run, summary)
    if hold is not None and summary["hold"] != hold:
        raise ValueError("Actor hold summary differs")
    return {
        "status": expected_status,
        "frames": count,
        "steps": count - 1,
        "start_step": metadata["start_step"],
        "end_step": metadata["end_step"],
        "fixture_sha256": metadata["fixture_sha256"],
        "config_digest": metadata["config_digest"],
        "layout_digest": sha_bytes(metadata["layout"]),
        "hold": hold,
        "input_scope": metadata["input_scope"],
        "index_sha256": sha(directory / "trajectory-index.json"),
        "log_sha256": sha(directory / "steps.jsonl.gz"),
    }


def sha_bytes(value):
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _semantic(frame, metadata):
    result = {}
    role_map = metadata["roles"]
    names = sorted(name for name in role_map if name != "tet")
    dofs = [i for name in names for i in role_map[name]["dof"]]
    bodies = [role_map[name]["body"] for name in names]
    rigid_rows = [role_map[name]["rigid_row"] for name in names if role_map[name]["kind"] == "rigid"]
    for name, value in frame.items():
        if name in IDENTITY_FIELDS or name == "frame_publication":
            continue
        if name in {"joint_q", "history.bound_q"}:
            # FREE transforms are checked against their authority; compare rotations geometrically below.
            for role in names:
                q = value[role_map[role]["coord"]]
                result[name + "." + role] = q[:3] if len(q) == 7 else q
                if len(q) == 7:
                    result[name + "." + role + ".rotation"] = q[3:]
        elif name in {"joint_qd", "history.bound_qd", "history.contact_joint_force"}:
            result[name] = value[dofs]
        elif name in {"body_q", "body_qd", "body_f", "history.contact_body_force"}:
            selected = value[bodies]
            if name == "body_q":
                result[name] = selected[:, :3]
                result[name + ".rotation"] = selected[:, 3:]
            else:
                result[name] = selected
        elif name.startswith("history.rigid_"):
            result[name] = value[rigid_rows]
        else:
            result[name] = value
    return result


def compare_semantic_fields(aa, bb, limits, errors):
    """Apply the frozen actor envelope to one authenticated pair of semantic frames."""
    passed = True
    if aa.keys() != bb.keys():
        raise ValueError("Semantic actor fields differ")
    for name, value in aa.items():
        other = bb[name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Semantic actor shape/dtype differs")
        rotation = (
            name.endswith(".rotation") or name.endswith("rigid_rotation") or name.endswith("rigid_bound_rotation")
        )
        if rotation:
            q, r = np.atleast_2d(value), np.atleast_2d(other)
            q, r = q / np.linalg.norm(q, axis=-1)[:, None], r / np.linalg.norm(r, axis=-1)[:, None]
            sign = np.where(np.sum(q * r, axis=-1) < 0, -1, 1)
            delta = 4 * np.arcsin(np.minimum(1, np.linalg.norm(q - sign[:, None] * r, axis=-1) / 2))
            limit = limits["rotation_absolute_rad"]
        else:
            delta = (
                np.abs(value.astype(np.float64) - other.astype(np.float64))
                if value.dtype.kind in "fci"
                else np.array([not np.array_equal(value, other)], float)
            )
            if value.dtype.kind != "f":
                limit = 0
            elif "force" in name or name in {"body_f", "particle_f"}:
                limit = limits["force_absolute"] + limits["force_relative"] * np.abs(value)
            elif name.startswith("frame_") or name in {"history.last_dt"}:
                limit = 0
            elif any(
                token in name
                for token in ("qd", "_omega", "_vsym", "_v_com", "joint_v", "link_v", "particle_qd", "bound_v")
            ):
                limit = limits["velocity_absolute"]
            else:
                limit = limits["position_absolute_m"]
        ok = bool(np.all(delta <= limit))
        row = errors.setdefault(name, {"max_error": 0.0, "within_tolerance": True})
        row["max_error"] = max(row["max_error"], float(np.max(delta, initial=0)))
        row["within_tolerance"] &= ok
        passed &= ok
    return passed


def compare_actor(
    reference, actual, *, creation_order=False, reference_run=None, actual_run=None, require_complete=True
):
    """Compare authenticated native histories over the actual recorded clock range."""
    from scripts.monolithic_reference.run_actor_coupling import load_case  # noqa: PLC0415

    audits = [
        check_actor(path, expected_run=run, require_complete=require_complete)
        for path, run in ((Path(reference), reference_run), (Path(actual), actual_run))
    ]
    if type(creation_order) is not bool:
        raise ValueError("Creation-order comparison must be explicit")
    if not creation_order and any(audits[0][name] != audits[1][name] for name in ("config_digest", "layout_digest")):
        raise ValueError("Different actor layout requires explicit creation-order comparison")
    if audits[0]["fixture_sha256"] != audits[1]["fixture_sha256"] or audits[0]["end_step"] != audits[1]["end_step"]:
        raise ValueError("Actor physical input or trajectory range differs")
    if audits[0]["input_scope"] != audits[1]["input_scope"]:
        raise ValueError("Component recording cannot qualify the frozen E2E")
    if json.loads((Path(reference) / "source-hashes.json").read_text()) != json.loads(
        (Path(actual) / "source-hashes.json").read_text()
    ):
        raise ValueError("Mixed source identities in actor comparison")
    start = audits[1]["start_step"]
    if not audits[0]["start_step"] <= start:
        raise ValueError("Reference does not cover the restored clock range")
    left_meta, right_meta = [
        json.loads((Path(path) / "configuration.json").read_text()) for path in (reference, actual)
    ]
    if left_meta.get("algorithm") != right_meta.get("algorithm"):
        raise ValueError("Different contact algorithms require an explicit dataflow comparison")
    left = (f for f in actor_frames(Path(reference), left_meta) if int(f["frame_steps"]) >= start)
    right = actor_frames(Path(actual), right_meta)
    limits = load_case()["equivalence"]
    errors, passed, count = {}, True, 0
    for a, b in zip(left, right, strict=True):
        aa, bb = _semantic(a, left_meta), _semantic(b, right_meta)
        passed &= compare_semantic_fields(aa, bb, limits, errors)
        count += 1
    return {"status": "PASS_FROZEN_ENVELOPE" if passed else "FAIL", "frames": count, "fields": errors, "audits": audits}
