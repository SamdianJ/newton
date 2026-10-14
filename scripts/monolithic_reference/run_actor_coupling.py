# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen mixed articulation-rigid-tet acceptance scene explicitly."""

import argparse
import dataclasses
import gzip
import json
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from newton.examples.softbody.monolithic_recording import save, serial, state_arrays
from newton.examples.softbody.monolithic_soft_cube import load_cube
from scripts.monolithic_reference.check_recording import sha

REPO = Path(__file__).resolve().parents[2]
CASE = Path(__file__).with_name("fixtures") / "actor_coupling_v1.json"


def load_case():
    """Read the sole physical input and authenticate its asset."""
    case = json.loads(CASE.read_text())
    _finite_diagnostics(case)
    if (
        case["schema"] != "actor-coupling/v1"
        or case["dt"] != 0.001
        or case["steps"] != sum(case["phase_steps"])
        or case["phase_steps"] != [1000, 500, 500]
        or case["tet_fixed_face"] != "+x"
        or case["solver"]["precision"] != "fp64"
        or case["solver"]["publish_policy"] != "strict"
        or case["solver"]["contact_model"] != "C-IE"
        or case["solver"]["globalization"] != "ArmijoOrResNorm"
    ):
        raise ValueError("Invalid frozen actor coupling protocol")
    for name in ("pusher_mass_kg", "rigid_mass_kg", "density_kg_m3", "young_modulus_pa", "contact_stiffness"):
        if not np.isfinite(case[name]) or case[name] <= 0:
            raise ValueError(f"Invalid physical input: {name}")
    if not np.isfinite(case["box_dimensions_m"]).all() or np.any(np.array(case["box_dimensions_m"]) <= 0):
        raise ValueError("Invalid box dimensions")
    if (
        not -1 < case["poisson_ratio"] < 0.5
        or case["young_modulus_pa"] != 10000
        or case["poisson_ratio"] != 0.3
        or case["density_kg_m3"] != 1000
        or case["gravity"] != [0, 0, 0]
        or case["tet_edge_m"] != 0.04
        or len(case["box_dimensions_m"]) != 3
        or set(case["centers_m"]) != {"pusher", "rigid_1", "rigid_2", "tet"}
        or any(len(v) != 3 for v in case["centers_m"].values())
        or any(v <= 0 for v in case["pusher_pd"].values())
        or any(case[n] <= 0 for n in ("pusher_travel_m", "normal_smoothing_m", "friction_velocity_scale"))
        or any(case[n] < 0 for n in ("shape_margin_m", "particle_radius_m", "friction_coefficient"))
    ):
        raise ValueError("Invalid frozen material, environment or control input")
    asset = (REPO / case["tet_asset"]).resolve()
    if not asset.is_relative_to(REPO) or sha(asset) != case["tet_asset_sha256"]:
        raise ValueError("Frozen tet asset identity differs")
    return case


def targets(case, step):
    """Sample absolute step-end position [m] and velocity [m/s]."""
    if not np.isfinite(step) or not 0 <= step <= case["steps"]:
        raise ValueError("Control step is outside the frozen clock")
    load, hold, unload = case["phase_steps"]
    if step < load:
        u, duration, sign, base, phase = step / load, load, 1, 0, "load"
    elif step < load + hold:
        return case["pusher_travel_m"], 0.0, "hold"
    else:
        u, duration, sign, base, phase = (step - load - hold) / unload, unload, -1, 1, "unload"
    blend = u**3 * (10 - 15 * u + 6 * u**2)
    rate = 30 * u**2 * (1 - u) ** 2 / (duration * case["dt"])
    return case["pusher_travel_m"] * (base + sign * blend), sign * case["pusher_travel_m"] * rate, phase


def build_model(*, device="cuda:0", before=False, rigid_count=2):
    """Construct the frozen scene or extend its rigid chain for scaling experiments."""
    if type(rigid_count) is not int or rigid_count < 2:
        raise ValueError("Rigid chain requires at least two bodies")
    case = load_case()
    if rigid_count != 2:
        pitch = case["centers_m"]["rigid_2"][0] - case["centers_m"]["rigid_1"][0]
        end = case["centers_m"]["rigid_2"][0]
        for i in range(1, rigid_count + 1):
            case["centers_m"][f"rigid_{i}"] = [end - (rigid_count - i) * pitch, 0, 0]
        case["centers_m"]["pusher"] = [end - rigid_count * pitch, 0, 0]
    builder = newton.ModelBuilder(gravity=case["gravity"])
    dimensions = np.asarray(case["box_dimensions_m"])
    shapes = {}

    def body(name, mass, *, rigid):
        inertia = mass / 12 * (np.sum(dimensions**2) - dimensions**2)
        center = case["centers_m"][name]
        link = builder.add_link(
            xform=wp.transform(center, wp.quat_identity()),
            mass=mass,
            inertia=wp.diag(wp.vec3(inertia)),
            label=name,
        )
        if rigid:
            joint = builder.add_joint_free(link)
        else:
            pd = case["pusher_pd"]
            joint = builder.add_joint_prismatic(
                -1,
                link,
                axis=newton.Axis.X,
                parent_xform=wp.transform(center, wp.quat_identity()),
                target_ke=pd["ke"],
                target_kd=pd["kd"],
                effort_limit=pd["effort_limit"],
                velocity_limit=pd["velocity_limit"],
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                limit_ke=0,
                limit_kd=0,
                friction=0,
                damping=0,
                armature=0,
            )
        builder.add_articulation([joint])
        shapes[name] = builder.add_shape_box(
            link,
            hx=dimensions[0] / 2,
            hy=dimensions[1] / 2,
            hz=dimensions[2] / 2,
            cfg=builder.ShapeConfig(
                density=0,
                margin=case["shape_margin_m"],
                has_particle_collision=name == f"rigid_{rigid_count}",
            ),
        )

    rigid_names = tuple(f"rigid_{i}" for i in range(1, rigid_count + 1))
    names = (*rigid_names, "pusher") if before else ("pusher", *rigid_names)
    for name in names:
        body(name, case["pusher_mass_kg"] if name == "pusher" else case["rigid_mass_kg"], rigid=name != "pusher")
    for name in rigid_names[1:]:
        builder.shape_collision_filter_pairs.append((shapes["pusher"], shapes[name]))
    builder.add_soft_mesh(
        pos=case["centers_m"]["tet"],
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        mesh=load_cube(REPO / case["tet_asset"]),
        particle_radius=case["particle_radius_m"],
        add_surface_mesh_edges=False,
    )
    vertices = np.asarray(builder.particle_q)
    for index in np.flatnonzero(vertices[:, 0] == vertices[:, 0].max()):
        builder.particle_mass[index] = 0
    model = builder.finalize(device=device)
    if model.particle_count != 125 or model.tet_count != 384:
        raise ValueError("Unexpected frozen tet topology")
    return model


def make_solver(model, *, mode=None, graph=None, contact_dataflow=None, spatial=None, contact_factor_capacity=None):
    """Use the solver default unless an offline comparison route is explicitly requested."""
    if contact_dataflow not in (None, "legacy", "pooled", "spatial"):
        raise ValueError("Unsupported actor contact dataflow")
    if (contact_dataflow == "spatial") != (spatial is not None):
        raise ValueError("Spatial actor experiments require explicit capacity and law margin")
    context = nullcontext()
    if contact_dataflow == "legacy":
        from newton._src.solvers.monolithic.contact import ContactObjective  # noqa: PLC0415

        context = patch("newton._src.solvers.monolithic.solver_monolithic.ContactObjective", ContactObjective)
    elif contact_dataflow is not None:
        from newton._src.solvers.monolithic.contact_pipeline import (  # noqa: PLC0415
            _PooledContactObjective,
            _SpatialContactObjective,
        )

        objective = _PooledContactObjective
        if contact_dataflow == "spatial":
            if type(spatial) is not dict or set(spatial) != {"candidate_capacity", "law_margin"}:
                raise ValueError("Invalid spatial actor experiment parameters")
            objective = partial(_SpatialContactObjective, **spatial)
        context = patch("newton._src.solvers.monolithic.solver_monolithic.ContactObjective", objective)
    case = load_case()
    options = dict(case["solver"])
    if mode is not None:
        options["pcg_mode"] = mode
    if graph is not None:
        options["use_cuda_graph"] = graph
    with context:
        return SolverMonolithic(
            model,
            **options,
            contact_factor_capacity=contact_factor_capacity,
            contact_stiffness=case["contact_stiffness"],
            normal_smoothing=case["normal_smoothing_m"],
            friction_coefficient=case["friction_coefficient"],
            friction_velocity_scale=case["friction_velocity_scale"],
            joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
            tet_rest_density=wp.full(model.tet_count, case["density_kg_m3"], dtype=float, device=model.device),
        )


def contact_algorithm(run):
    """Record the exact private pooled constructor recipe, including its resource budget."""
    return {
        "contact_dataflow": "spatial" if run._contact_spatial else "pooled",
        "contact_factor_capacity": int(run._coupled_config["guard_capacity"][1]),
        "spatial": (
            {"candidate_capacity": run._contact.candidate_capacity, "law_margin": run._contact.law_margin}
            if run._contact_spatial
            else None
        ),
    }


def roles(model, table):
    """Bind named physical roles to Model indices and native rigid rows."""
    result, rigid_row = {}, 0
    for actor in table.actors:
        if actor.kind == "tet":
            result["tet"] = {
                "particle": list(actor.particle_indices),
                "global": list(range(actor.dof_start, actor.dof_start + actor.dof_count)),
            }
            continue
        if len(actor.body_indices) != 1:
            raise ValueError("Actor coupling requires single-body roles")
        name = model.body_label[actor.body_indices[0]]
        if not name or name in result:
            raise ValueError("Actor role labels must be unique")
        result[name] = {
            "kind": actor.kind,
            "body": actor.body_indices[0],
            "joint": actor.joint_indices[0],
            "coord": list(actor.coord_indices),
            "dof": list(actor.dof_indices),
            "target": list(actor.target_indices),
            "shape": list(actor.shape_indices),
            "global": list(range(actor.dof_start, actor.dof_start + actor.dof_count)),
        }
        if actor.kind == "rigid":
            result[name]["rigid_row"] = rigid_row
            rigid_row += 1
    return result


def sampling(run):
    """Record fixed keys, weights, support and allocation bounds once."""
    c = run._contact
    if run._contact_pool:
        # Compact metadata is authenticated directly; pooled execution has no
        # permanent per-contact expanded support table to archive.
        return {
            "fields": {name: value.tolist() for name, value in c.actor_samples.fields.items()},
            "keys": c.count,
            "triplets": c.triplet_capacity,
            # Pool payload only; lazy query/reference scratch is not part of
            # fixed sampling identity or a total-device-memory measurement.
            "pool_bytes": sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in c.pool.buffers()),
        }
    sample = c.actor_contact.samples
    return {
        "fields": {name: value.tolist() for name, value in sample.fields.items()},
        "offsets": c.factor_support[0].tolist(),
        "columns": c.factor_support[1].tolist(),
        "keys": c.count,
        "triplets": c.triplet_capacity,
        "bytes": sum(a.size * wp.types.type_size_in_bytes(a.dtype) for a in c.actor_contact.buffers()),
    }


def model_arrays(model):
    """Archive physical arrays; BVH roots and scratch geometry depend on the backend."""
    caches = {"bvh_shapes_group_roots", "bvh_particles_group_roots", "bvh_shape_bounds", "bvh_shape_world_transforms"}
    return {n: a.numpy() for n, a in vars(model).items() if isinstance(a, wp.array) and n not in caches}


def measure(run, state, step, *, metadata=None):
    """Measure accepted native geometry and both endpoint world-origin wrenches."""
    c = run._contact
    gaps = []
    if not run._contact_pool and c.legacy_count:
        shapes = c.shapes.numpy()
        gaps.extend(c.current.distance.numpy() - c.parameters[4] - c.margins.numpy()[np.resize(shapes, c.legacy_count)])
    if c.actor_contact is not None:
        gaps.extend(c.actor_contact.current.distance.numpy() - c.actor_contact.offsets.numpy())
    pairs = []
    role_map = roles(run.model, run._actors)
    if run._contact_pool:
        from scripts.monolithic_reference.actor_recording import _penetration  # noqa: PLC0415

        # This is an offline full-sampling geometry guard, independent of the
        # frozen query/admission set. It does not repair missing contributions.
        penetration = _penetration(state_arrays(state), metadata or {"roles": role_map, "sampling": sampling(run)}, run)
    else:
        penetration = max(0.0, -min(gaps, default=0.0))
    matrix = run._linear.densify_for_test(generation=run._generation).raw_matrix
    for row in run.last_stats.variational["actor_contact_pairs"]:
        pair = dict(row)
        pair["source_role"] = (
            "tet" if row["source"] < 0 else run.model.body_label[int(run.model.shape_body.numpy()[row["source"]])]
        )
        pair["target_role"] = run.model.body_label[int(run.model.shape_body.numpy()[row["target"]])]
        left = role_map[pair["source_role"]]["global"]
        right = role_map[pair["target_role"]]["global"]
        pair["cross_block_norms"] = [
            float(np.linalg.norm(matrix[np.ix_(left, right)])),
            float(np.linalg.norm(matrix[np.ix_(right, left)])),
        ]
        pairs.append(pair)
    unmeasured = [
        f.name
        for f in dataclasses.fields(run.last_stats)
        if isinstance(getattr(run.last_stats, f.name), float)
        and np.isnan(getattr(run.last_stats, f.name))
        and isinstance(f.default, float)
        and np.isnan(f.default)
    ]
    _finite_diagnostics(
        {
            f.name: getattr(run.last_stats, f.name)
            for f in dataclasses.fields(run.last_stats)
            if f.name not in unmeasured
        }
    )
    stats = serial(run.last_stats)
    return {
        "step": step,
        "time": step * load_case()["dt"],
        "publication": stats["variational"]["publication_status"],
        "stats": stats,
        "unmeasured_stats": unmeasured,
        "pairs": pairs,
        "penetration": penetration,
        "tet_displacement": float(
            np.linalg.norm(state.particle_q.numpy() - run.model.particle_q.numpy(), axis=1).max()
        ),
        "force_generation": int(state._monolithic_coupled.force_generation.numpy()[0]),
        "force_valid": int(state._monolithic_coupled.force_valid.numpy()[0]),
    }


def _finite_diagnostics(value):
    if hasattr(value, "items"):
        for entry in value.values():
            _finite_diagnostics(entry)
    elif isinstance(value, (list, tuple, np.ndarray)):
        for entry in value:
            _finite_diagnostics(entry)
    elif isinstance(value, (float, np.floating)) and not np.isfinite(value):
        raise ValueError("Nonfinite measured solver diagnostic")


def validate_row(row, case):
    """Reject incomplete convergence, publication, state guards or pair balance."""
    s = row["stats"]
    fields = dataclasses.fields(SolverMonolithic.Stats)
    defaults = {f.name for f in fields if isinstance(f.default, float) and np.isnan(f.default)}
    missing = {n for n in defaults if s.get(n) is None}
    if s.keys() != {f.name for f in fields} or set(row["unmeasured_stats"]) != missing:
        raise ValueError("Incomplete Stats or undeclared unmeasured diagnostics")
    _finite_diagnostics(s)
    pool = s["variational"].get("contact_pool")
    spatial = s["variational"].get("contact_spatial")
    if pool is not None and (pool["tainted"] is not False or pool["dropped"] != 0):
        raise ValueError("Contact capacity truncation cannot qualify an actor recording")
    if spatial is not None and (spatial["tainted"] is not False or spatial["capacity_dropped"] != 0):
        raise ValueError("Spatial capacity truncation cannot qualify an actor recording")
    if (
        s["converged"] is not True
        or s["rolled_back"]
        or row["publication"] != "CONVERGED"
        or s["status"] != "success"
        or s["variational"]["publish_policy"] != "strict"
        or s["step_generation"] != row["step"]
        or s["joint_force_generation"] != row["step"]
    ):
        raise ValueError("Actor step did not converge under strict publication")
    ratios = [s["convergence_ratio" + suffix] for suffix in ("", "_q", "_x")]
    blocks = s["variational"]["rigid_blocks"]
    if len(blocks) != 4:
        raise ValueError("Missing rigid convergence blocks")
    ratios.extend(v["convergence_ratio_" + name] for name, v in blocks.items())
    if any(v is None or not np.isfinite(v) or not 0 <= v <= 1 for v in ratios):
        raise ValueError("Invalid per-actor convergence ratio")
    for post in s["variational"]["linear_postchecks"]:
        for block in post["blocks"].values():
            if any(block[n] is None or not np.isfinite(block[n]) or block[n] < 0 for n in ("residual", "limit")):
                raise ValueError("Invalid measured linear posterior")
    if (
        row["force_valid"] != 1
        or row["force_generation"] != row["step"]
        or s["variational"]["contact_force_generation"] != row["step"]
        or s["min_det_f"] is None
        or not np.isfinite(s["min_det_f"])
        or s["min_det_f"] <= case["gates"]["min_det_f"]
        or not np.isfinite(row["penetration"])
        or row["penetration"] > case["gates"]["max_penetration_m"]
    ):
        raise ValueError("Invalid state guard or publication generation")
    expected = {frozenset(("pusher", "rigid_1")), frozenset(("rigid_1", "rigid_2")), frozenset(("rigid_2", "tet"))}
    actual = [frozenset((p["source_role"], p["target_role"])) for p in row["pairs"]]
    if len(actual) != 3 or set(actual) != expected:
        raise ValueError("Missing or duplicated force-chain interface")
    for pair in row["pairs"]:
        pair_roles = frozenset((pair["source_role"], pair["target_role"]))
        kind = "R-T" if "tet" in pair_roles else "A-R" if "pusher" in pair_roles else "R-R"
        if pair["kind"] != kind:
            raise ValueError("Pair kind differs from physical roles")
        wrench = np.asarray(pair["world_wrench"])
        if wrench.shape != (2, 6) or not np.isfinite(wrench).all() or pair["force_generation"] != row["step"]:
            raise ValueError("Invalid endpoint wrench")
        cross = pair["cross_block_norms"]
        if (
            len(cross) != 2
            or not np.isfinite(cross).all()
            or (np.linalg.norm(wrench[0, :3]) > 1e-4 and min(cross) <= 0)
        ):
            raise ValueError("Missing coupled cross block")
        for values, floor in ((wrench[:, :3], 1), (wrench[:, 3:], 1e-3)):
            if np.linalg.norm(values.sum(axis=0)) > case["gates"]["balance_relative_error"] * max(
                floor, np.linalg.norm(values, axis=1).sum()
            ):
                raise ValueError("Endpoint force or moment is unbalanced")
    return row


def hold_gates(rows, case):
    """Require the final hold window to carry force through all three interfaces."""
    last = sum(case["phase_steps"][:2])
    hold = [r for r in rows if last - 100 < r["step"] <= last]
    if [r["step"] for r in hold] != list(range(last - 99, last + 1)):
        raise ValueError("Incomplete final hold window")
    forces = {}
    for row in hold:
        validate_row(row, case)
        if row["tet_displacement"] < case["gates"]["hold_min_tet_displacement_m"]:
            raise ValueError("Hold tet deformation is insufficient")
        for p in row["pairs"]:
            label = "/".join(sorted((p["source_role"], p["target_role"])))
            force = float(np.linalg.norm(np.asarray(p["world_wrench"])[0, :3]))
            if force < case["gates"]["hold_min_pair_force_n"]:
                raise ValueError("Hold interface force is insufficient")
            forces.setdefault(label, []).append(force)
    noise = {name: float(np.std(v) / np.mean(v)) for name, v in forces.items()}
    if any(v > case["gates"]["hold_force_relative_noise"] for v in noise.values()):
        raise ValueError("Hold interface force is noisy")
    return {"steps": len(hold), "force_relative_noise": noise}


def save_midpoint(run, state, path, step, metadata):
    """Save a complete checkpoint and authenticated absolute control clock."""
    run.save_checkpoint(state, path)
    save(
        path.with_suffix(".json"),
        {
            "schema": "actor-coupling-checkpoint/v1",
            "step": step,
            "dt": load_case()["dt"],
            "checkpoint_sha256": sha(path),
            "fixture_sha256": metadata["fixture_sha256"],
            "config_digest": metadata["config_digest"],
            "layout_digest": run._actors.identity,
        },
    )


def restore_midpoint(run, state, path, step):
    """Validate the sidecar before loading without resetting history."""
    sidecar = json.loads(path.with_suffix(".json").read_text())
    expected = {
        "schema": "actor-coupling-checkpoint/v1",
        "step": step,
        "dt": load_case()["dt"],
        "checkpoint_sha256": sha(path),
        "fixture_sha256": sha(CASE),
        "config_digest": run._kinematics.digest.hex(),
        "layout_digest": run._actors.identity,
    }
    if sidecar != expected:
        raise ValueError("Checkpoint control clock or input identity differs")
    with np.load(path, allow_pickle=False) as archive:
        generation = archive["history.force_generation"]
        if generation.shape != (1,) or generation.dtype != np.int64 or generation[0] != step:
            raise ValueError("Checkpoint generation differs from control clock")
        if not np.array_equal(archive["history.last_dt"], np.array([load_case()["dt"]], np.float64)):
            raise ValueError("Checkpoint timestep differs from control clock")
    run.load_checkpoint(state, path)


def record_run(
    run,
    state,
    output,
    *,
    before=False,
    start_step=0,
    end_step=2000,
    checkpoint_step=1000,
    component=False,
    system_audit=None,
):
    """Record a bounded prefix or complete trajectory; stop on every failure."""
    case = load_case()
    if Path(newton.__file__).resolve().parent != REPO / "newton":
        raise ValueError("Actor recording must import Newton from this checkout")
    if type(start_step) is not int or type(end_step) is not int or not 0 <= start_step < end_step <= case["steps"]:
        raise ValueError("Invalid recording clock range")
    if system_audit is not None and system_audit.run is not run:
        raise ValueError("Complete system audit belongs to another solver")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema": "actor-coupling-recording/v1",
        "case": case,
        "fixture_sha256": sha(CASE),
        "input_scope": "component" if component else "frozen",
        "before": before,
        "start_step": start_step,
        "end_step": end_step,
        "config_digest": run._kinematics.digest.hex(),
        "layout": json.loads(run._actors.layout_payload()),
        "roles": roles(run.model, run._actors),
        "sampling": sampling(run),
        "execution": {"device": str(run.model.device), "mode": run.pcg_mode.value, "graph": run._use_cuda_graph},
    }
    if run._contact_pool:
        metadata.update(schema="actor-coupling-recording/v2", algorithm=contact_algorithm(run))
    save(output / "configuration.json", metadata)
    np.savez_compressed(output / "model.npz", **model_arrays(run.model))
    save(
        output / "source-hashes.json",
        {
            str(p.relative_to(REPO)): sha(p)
            for p in [
                *sorted((REPO / "newton/_src/solvers/monolithic").glob("*.py")),
                Path(__file__),
                REPO / "scripts/monolithic_reference/check_recording.py",
                REPO / "scripts/monolithic_reference/actor_recording.py",
                REPO / "scripts/monolithic_reference/contact_system_audit.py",
                REPO / "scripts/monolithic_reference/compare_contact_dataflows.py",
            ]
        },
    )
    save(
        output / "command.json",
        {
            "argv": sys.argv,
            "cwd": str(Path.cwd()),
            "newton_import": newton.__file__,
            "python": sys.version,
            "warp": wp.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()),
        },
    )
    control = run.model.control()
    a = run._articulation_actor

    def assign_control(step):
        q, v, phase = targets(case, step)
        tq, tv = control.joint_target_q.numpy(), control.joint_target_qd.numpy()
        tq[list(a.target_indices)] = q
        tv[list(a.dof_indices)] = v
        control.joint_target_q.assign(tq)
        control.joint_target_qd.assign(tv)
        return q, v, phase, float(tq[a.target_indices[0]]), float(tv[a.dof_indices[0]])

    q, v, _, cq, cv = assign_control(start_step)
    initial = state_arrays(state)
    np.savez_compressed(output / "initial.npz", **initial)
    frames = [(start_step, "INITIAL" if start_step == 0 else "RESTORED", initial, q, v, cq, cv)]
    chunks, rows, publications = [], [], []

    def flush():
        if not frames:
            return
        folder = output / "trajectory"
        folder.mkdir(exist_ok=True)
        path = folder / f"frames-{frames[0][0]:06d}-{frames[-1][0]:06d}.npz"
        data = {n: np.stack([f[2][n] for f in frames]) for n in initial}
        data.update(
            frame_steps=np.array([f[0] for f in frames], dtype=np.int64),
            frame_times=np.array([f[0] * case["dt"] for f in frames]),
            frame_publication=np.array([f[1] for f in frames], dtype="U16"),
            frame_target_q=np.array([f[3] for f in frames]),
            frame_target_qd=np.array([f[4] for f in frames]),
            frame_control_q=np.array([f[5] for f in frames]),
            frame_control_qd=np.array([f[6] for f in frames]),
        )
        np.savez_compressed(path, **data)
        chunks.append({"path": str(path.relative_to(output)), "count": len(frames), "sha256": sha(path)})
        save(
            output / "trajectory-index.json",
            {"chunks": chunks, "includes_initial": True, "total_frames": sum(c["count"] for c in chunks)},
        )
        frames.clear()

    error = None
    system_rows = []
    begin = time.perf_counter()
    with gzip.open(output / "steps.jsonl.gz", "wt") as log:
        try:
            for step in range(start_step + 1, end_step + 1):
                previous = state_arrays(state)
                q, v, phase, cq, cv = assign_control(step)
                run.step(state, state, control, dt=case["dt"])
                publications.append(
                    "CONVERGED"
                    if run.last_stats.converged
                    else "HARD_FAILURE"
                    if run.last_stats.rolled_back
                    else "STOPPED_ACCEPTED"
                )
                row = measure(run, state, step, metadata=metadata)
                row.update(target_q=q, target_qd=v, control_target_q=cq, control_target_qd=cv, phase=phase)
                log.write(json.dumps(serial(row), allow_nan=False) + "\n")
                log.flush()
                validate_row(row, case)
                if system_audit is not None:
                    system_rows.append(serial(system_audit.evaluate()))
                    # This gate belongs to offline acceptance, not trial selection.
                    if system_rows[-1]["status"] != "PASS_FULL_RESIDUAL":
                        raise ValueError("Final state fails the complete system residual gates")
                rows.append(row)
                frames.append((step, "CONVERGED", state_arrays(state), q, v, cq, cv))
                if len(frames) == 100:
                    flush()
                if step == checkpoint_step:
                    save_midpoint(run, state, output / "midpoint.npz", step, metadata)
        except Exception as failure:
            if len(publications) < step - start_step:
                publications.append("HARD_FAILURE")
            error = f"{type(failure).__name__}: {failure}"
            np.savez_compressed(output / "failure-input.npz", **previous)
            np.savez_compressed(output / "failure-current.npz", **state_arrays(state))
            for name in ("trial", "accepted"):
                candidate = getattr(run, "_" + name, None)
                if candidate is not None:
                    np.savez_compressed(output / ("failure-" + name + ".npz"), **state_arrays(candidate))
            save(
                output / "failure.json",
                {
                    "step": step,
                    "error": error,
                    "stats": run.last_stats,
                    "rollback_exact": all(np.array_equal(v, state_arrays(state)[n]) for n, v in previous.items()),
                },
            )
        finally:
            flush()
    summary = {
        "status": "FAIL" if error else "PASS_PREFIX",
        "start_step": start_step,
        "end_step": end_step,
        "published_steps": int(state._monolithic_coupled.force_generation.numpy()[0]) - start_step,
        "checked_steps": len(rows),
        "attempted_steps": len(publications),
        "complete": error is None and end_step == case["steps"],
        "counts": {name: publications.count(name) for name in ("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE")},
        "qualification_failures": int(error is not None),
        "error": error,
        "elapsed_seconds": time.perf_counter() - begin,
    }
    if system_audit is not None:
        report = output / "full-system.jsonl"
        report.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in system_rows))
        summary["full_system"] = {
            "file": report.name,
            "sha256": sha(report),
            "contract": system_audit.contract,
            "checked_steps": len(system_rows),
            "failed_steps": sum(row["status"] != "PASS_FULL_RESIDUAL" for row in system_rows),
        }
    if summary["complete"]:
        try:
            summary["hold"] = hold_gates(rows, case)
            summary["status"] = "PASS"
        except ValueError as failure:
            summary.update(status="FAIL", error=str(failure), complete=False, qualification_failures=1)
    save(output / "summary.json", summary)
    np.savez_compressed(output / "final.npz", **state_arrays(state))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for manual acceptance (5000 steps; spatial also runs a 2000-step full comparison)",
    )
    parser.add_argument("--retain-recordings", choices=("all", "failures"), default="failures")
    parser.add_argument("--contact-dataflow", choices=("legacy", "pooled", "spatial"), default="legacy")
    parser.add_argument("--candidate-capacity", type=int)
    parser.add_argument("--law-margin", type=float)
    parser.add_argument("--contact-factor-capacity", type=int)
    args = parser.parse_args()
    spatial = None
    if args.contact_dataflow == "spatial":
        if any(v is None for v in (args.candidate_capacity, args.law_margin, args.contact_factor_capacity)):
            parser.error("Spatial acceptance requires explicit candidate capacity, law margin and factor budget")
        spatial = {"candidate_capacity": args.candidate_capacity, "law_margin": args.law_margin}
    elif args.candidate_capacity is not None or args.law_margin is not None:
        parser.error("Spatial parameters require --contact-dataflow spatial")
    elif args.contact_dataflow == "legacy" and args.contact_factor_capacity is not None:
        parser.error("The frozen legacy command retains its original factor budget")
    args.output.mkdir(parents=True, exist_ok=False)
    from scripts.monolithic_reference.check_recording import check_actor, compare_actor  # noqa: PLC0415

    reports = {name: {"status": "NOT_RUN"} for name in ("primary", "restored", "reordered")}
    if args.contact_dataflow == "spatial":
        reports["complete"] = {"status": "NOT_RUN"}
    primary_run = None
    try:
        for name, before, start in (("primary", False, 0), ("restored", False, 1000), ("reordered", True, 0)):
            reports[name] = {"status": "IN_PROGRESS"}
            run = make_solver(
                build_model(before=before),
                contact_dataflow=args.contact_dataflow,
                spatial=spatial,
                contact_factor_capacity=args.contact_factor_capacity,
            )
            state = run.create_state(dt=load_case()["dt"])
            if start:
                restore_midpoint(run, state, args.output / "primary/recording/midpoint.npz", start)
            folder = args.output / name / "recording"
            system_audit = None
            if args.contact_dataflow == "spatial":
                from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit  # noqa: PLC0415

                complete = make_solver(build_model(before=before), contact_dataflow="pooled", contact_factor_capacity=0)
                system_audit = ContactSystemAudit(run, complete)
            summary = record_run(run, state, folder, before=before, start_step=start, system_audit=system_audit)
            reports[name] = {
                "status": summary["status"],
                "summary": summary,
                "provenance": json.loads((folder / "command.json").read_text()),
                "source_hashes": json.loads((folder / "source-hashes.json").read_text()),
                "configuration_sha256": sha(folder / "configuration.json"),
            }
            if summary["status"] != "PASS":
                raise ValueError(f"{name}: {summary['error']}")
            reports[name].update(status="PASS", audit=check_actor(folder))
            if name == "primary":
                primary_run = run
            if name != "primary":
                reports[name]["comparison"] = compare_actor(
                    args.output / "primary/recording",
                    folder,
                    creation_order=before,
                )
                if not reports[name]["comparison"]["status"].startswith("PASS"):
                    raise ValueError(f"{name}: physical comparison failed")
    except Exception as failure:
        reports[name].update(status="FAIL", error=f"{type(failure).__name__}: {failure}")
        reports["failure"] = {"status": "FAIL", "error": f"{type(failure).__name__}: {failure}"}
    # The complete counter-trajectory is independent of restored/reordered
    # qualification; retain it even when either of those comparisons fails.
    if args.contact_dataflow == "spatial" and reports["primary"]["status"] == "PASS":
        name = "complete"
        reports[name] = {"status": "IN_PROGRESS"}
        try:
            from scripts.monolithic_reference.compare_contact_dataflows import (  # noqa: PLC0415
                compare_actor_contact_recordings,
            )

            complete = make_solver(build_model(), contact_dataflow="pooled")
            folder = args.output / name / "recording"
            summary = record_run(complete, complete.create_state(dt=load_case()["dt"]), folder)
            reports[name] = {
                "status": summary["status"],
                "summary": summary,
                "provenance": json.loads((folder / "command.json").read_text()),
                "source_hashes": json.loads((folder / "source-hashes.json").read_text()),
                "configuration_sha256": sha(folder / "configuration.json"),
            }
            if summary["status"] != "PASS":
                raise ValueError(f"{name}: {summary['error']}")
            reports[name]["comparison"] = compare_actor_contact_recordings(
                folder,
                args.output / "primary/recording",
                reference_run=complete,
                actual_run=primary_run,
            )
            if reports[name]["comparison"]["status"] != "PASS_FROZEN_ENVELOPE":
                raise ValueError("Complete/spatial independent trajectories differ")
        except Exception as failure:
            reports[name].update(status="FAIL", error=f"{type(failure).__name__}: {failure}")
    cases = (
        ("primary", "restored", "reordered", "complete")
        if args.contact_dataflow == "spatial"
        else ("primary", "restored", "reordered")
    )
    status = "PASS" if all(reports[n]["status"] == "PASS" for n in cases) else "FAIL"
    retained = status != "PASS" or args.retain_recordings == "all"
    for name in cases:
        folder = args.output / name
        if folder.is_dir():
            reports[name]["recording_retained"] = retained
            save(folder / "receipt.json", reports[name])
    save(
        args.output / "summary.json",
        {
            "status": status,
            "scope": "frozen_actor_coupling",
            "cases": reports,
            "nightly": "NOT_RUN",
            "p5_status": "PARTIAL",
            "retain_recordings": args.retain_recordings,
        },
    )
    lines = [
        "# Mixed actor coupling",
        "",
        f"Result: {status}; P5: PARTIAL; nightly: NOT_RUN.",
        "",
        "| Case | Status | Converged | Checked | Recording retained |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("primary", "restored", "reordered"):
        row = reports[name]
        lines.append(
            f"| {name} | {row['status']} | {row.get('summary', {}).get('counts', {}).get('CONVERGED', 0)} | "
            f"{row.get('summary', {}).get('checked_steps', 0)} | {row.get('recording_retained', False)} |"
        )
    (args.output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if status == "PASS" and args.retain_recordings == "failures":
        import shutil  # noqa: PLC0415

        for name in ("primary", "restored", "reordered"):
            shutil.rmtree(args.output / name / "recording")
    print(json.dumps({"status": status, "nightly": "NOT_RUN"}))
    return int(status != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
