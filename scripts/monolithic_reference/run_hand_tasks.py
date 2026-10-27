# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Probe full-hand task feasibility with passive objects and independent physical gates."""

import argparse
import hashlib
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton
from newton._src.solvers.monolithic.smooth_sdf import sample_smooth
from newton.examples.softbody.monolithic_sharpa_assets import build_contact_hand
from newton.examples.softbody.monolithic_sharpa_scene import joint_mapping
from newton.examples.softbody.monolithic_soft_cube import load_cube
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture

from .actor_contact_oracle import box_samples
from .articulation_oracle import transform
from .hand_task_control import ContactHandControl, HandObservation, contact_loads, finger_index
from .run_dexterous_tasks import ramp, save_json

REPO = Path(__file__).resolve().parents[2]
TASKS = ("grasp-rigid", "grasp-soft", "pull-slider")
DT, DURATION = 0.002, 7.0


def schedule(task, t):
    """Return closure-source time/rate, carriage displacement/rate, and action phase."""
    close, close_v = ramp(t, 0.25, 2.75)
    release, release_v = ramp(t, 5.5, 6.5)
    move, move_v = ramp(t, 3.0, 4.0)
    lower, lower_v = ramp(t, 4.5, 5.25) if task != "pull-slider" else (0.0, 0.0)
    phase = (
        "settle"
        if t < 0.25
        else "close"
        if t < 3
        else "move"
        if t < 4
        else "hold"
        if t < 4.5
        else "lower/hold"
        if t < 5.5
        else "release"
        if t < 6.5
        else "observe"
    )
    return (
        3 * close * (1 - release),
        3 * (close_v * (1 - release) - close * release_v),
        0.05 * (move - lower),
        0.05 * (move_v - lower_v),
        phase,
    )


def build_model(task, assets, *, device, offset=0.0, surface_source="qualified", height=0.0):
    """Keep the imported hand intact; mount its existing scalar carriage along the task axis."""
    parameters = load_fixture(REPO / "newton/examples/softbody/sharpa_g1h.json")
    frozen = json.loads((assets / "close-18/manifest.json").read_text())
    mount = {**frozen["mount"], "support_height": 0.0}
    builder, identity = build_contact_hand(
        assets / "left_sharpa_wave", assets / "hand", device=device, parameters=parameters, _mount=mount
    )
    identity["surface_source"] = surface_source
    if surface_source == "cad":
        import trimesh

        surfaces, audit = {}, []
        for entry in identity["collision_mapping"]:
            name, shape = entry["source"], entry["shape"]
            if name not in surfaces:
                original = builder.shape_source[shape]
                cad = trimesh.load(assets / "left_sharpa_wave" / name, force="mesh", process=False)
                vertices = np.asarray(cad.vertices, dtype=np.float32)
                indices = np.asarray(cad.faces, dtype=np.int32)
                # Remove only exactly zero-area faces, in the eventual FP32 geometry.
                triangle = vertices.astype(float)[indices]
                keep = (
                    np.linalg.norm(np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]), axis=1)
                    > 0
                )
                surfaces[name] = newton.Mesh(vertices, indices[keep].ravel(), compute_inertia=False, sdf=original.sdf)
                audit.append(
                    {
                        "source": name,
                        "qualified_faces": len(original.indices) // 3,
                        "cad_faces": len(indices),
                        "zero_area_removed": int((~keep).sum()),
                    }
                )
            builder.shape_source[shape] = surfaces[name]
        identity["surface_adaptation"] = audit
    recorded = json.loads((assets / "recorded-control.json").read_text())["keyframes"][1]
    lift = builder.joint_label.index("monolithic_lift")
    builder.joint_X_p[lift] = wp.transform(np.add(recorded["position"], (0, 0, height)), wp.quat_identity())
    builder.joint_X_c[lift] = wp.transform((0, 0, 0), wp.quat_inverse(wp.quat(recorded["quaternion_xyzw"])))
    builder.joint_axis[builder.joint_qd_start[lift]] = wp.vec3(1, 0, 0) if task == "pull-slider" else wp.vec3(0, 0, 1)
    # A world support must also collide with the hand, unlike the old same-tree support.
    plane = next(i for i, kind in enumerate(builder.shape_type) if kind == newton.GeoType.PLANE)
    builder.body_shapes[builder.shape_body[plane]].remove(plane)
    builder.body_shapes[-1].append(plane)
    builder.shape_body[plane] = -1
    builder.shape_transform[plane] = wp.transform_identity()
    cfg = builder.ShapeConfig(density=0, margin=0)
    position = (0.0, offset, 0.0202 + height)
    object_body = object_dof = None
    if task == "grasp-soft":
        cube = REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r1.npz"
        builder.add_soft_mesh(
            pos=position,
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            mesh=load_cube(cube),
            particle_radius=0.0002,
            add_surface_mesh_edges=False,
        )
        identity["cube_sha256"] = hashlib.sha256(cube.read_bytes()).hexdigest()
    elif task == "grasp-rigid":
        object_body = builder.add_body(
            xform=wp.transform(position, wp.quat_identity()),
            mass=0.064,
            inertia=wp.diag(wp.vec3(0.064 * 0.04**2 / 6)),
            label="free_cube",
        )
        builder.add_shape_box(object_body, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)
    else:
        support = builder.add_link(mass=1, inertia=wp.diag(wp.vec3(0.01)), label="slider_support")
        fixed = builder.add_joint_fixed(-1, support, parent_xform=wp.transform(position, wp.quat_identity()))
        object_body = builder.add_link(
            mass=0.064, inertia=wp.diag(wp.vec3(0.064 * 0.04**2 / 6)), label="passive_handle"
        )
        joint = builder.add_joint_prismatic(
            support,
            object_body,
            axis=newton.Axis.X,
            target_ke=0,
            target_kd=0,
            friction=0.1,
            damping=0,
            limit_lower=0,
            limit_upper=0.06,
            limit_ke=1e4,
            limit_kd=20,
        )
        builder.add_articulation([fixed, joint])
        object_dof = builder.joint_qd_start[joint]
        builder.add_shape_box(object_body, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)
    model = builder.finalize(device=device)
    mapping = joint_mapping(model, identity["joint_names"])
    dofs = [mapping[n][1] for n in identity["joint_names"]]
    curve = ClosureTrajectory(
        assets / "Ball_catch - Left_hand_motion_rad.csv",
        identity["joint_names"],
        model.joint_limit_lower.numpy()[dofs],
        model.joint_limit_upper.numpy()[dofs],
        model.joint_velocity_limit.numpy()[dofs],
    )
    scalar_count = len(mapping) + int(task == "pull-slider")
    widths, speeds = np.full(scalar_count, 0.001), np.full(scalar_count, 0.001)
    for name, (_, dof, _) in mapping.items():
        cfg = mount["carriage"] if name == "monolithic_lift" else parameters["joints"][name]
        widths[dof], speeds[dof] = cfg["limit_width"], cfg["friction_velocity_scale"]
    terms = SolverMonolithic.JointTerms(
        implicit_pd=True, limits=True, friction=True, limit_width=tuple(widths), friction_velocity_scale=tuple(speeds)
    )
    identity.update(
        task=task,
        object_body=object_body,
        object_dof=object_dof,
        object_position_m=position,
        carriage_axis="X" if task == "pull-slider" else "Z",
        soft_refinement=1 if task == "grasp-soft" else None,
        hand_self_contact=False,
        hand_world_plane_contact_requested=True,
        height_m=height,
        input_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                REPO / "newton/examples/softbody/sharpa_g1h.json",
                assets / "close-18/manifest.json",
                assets / "recorded-control.json",
                assets / "Ball_catch - Left_hand_motion_rad.csv",
            )
        },
    )
    return model, mapping, curve, terms, identity


class HandTask:
    """Apply only hand actuator targets; preserve independent object dynamics."""

    def __init__(
        self,
        task,
        assets,
        *,
        device="cuda:0",
        friction=0.5,
        offset=0.0,
        surface_source="qualified",
        height=0.0,
        control_mode="open-loop",
        contact_stiffness=1e7,
    ):
        if control_mode not in ("open-loop", "contact") or not np.isfinite(contact_stiffness) or contact_stiffness <= 0:
            raise ValueError("Require open-loop/contact control and positive finite contact stiffness")
        self.task = task
        self.controller = self.feedback = self.feedback_geometry = self.control_before_step = None
        self.model, self.mapping, self.curve, terms, self.identity = build_model(
            task, assets, device=device, offset=offset, surface_source=surface_source, height=height
        )
        self.smooth = build_smooth_sdf(self.model, qualified_directory=assets / "geometry-rebuilt")
        self.solver = SolverMonolithic(
            self.model,
            precision="fp64",
            joint_terms=terms,
            smooth_sdf=self.smooth,
            mass_mode="consistent",
            material_model="smith_log_stabilized",
            tet_rest_density=wp.full(self.model.tet_count, 1000.0, dtype=float, device=device)
            if self.model.tet_count
            else None,
            contact_stiffness=contact_stiffness,
            friction_coefficient=friction,
            friction_velocity_scale=0.01,
            normal_smoothing=1e-4,
            alignment_policy="mask",
            publish_policy="strict",
            newton_max_iterations=80,
            linear_max_iterations=300,
            line_search_max_iterations=24,
            pcg_mode="production",
            use_cuda_graph=wp.get_device(device).is_cuda,
        )
        source = self.model.state()
        position, _, _ = self.targets(0)
        q = source.joint_q.numpy()
        for coord, _, slot in self.mapping.values():
            q[coord] = position[slot]
        source.joint_q.assign(q)
        newton.eval_fk(self.model, source.joint_q, source.joint_qd, source)
        self.state = self.solver.create_state(dt=DT, source=source)
        self.control = self.solver.create_control()
        self.hand_dofs = [self.mapping[n][1] for n in self.curve.names]
        self.hand_coords = [self.mapping[n][0] for n in self.curve.names]
        self.hand_bodies = self.solver._articulation_actors[0].body_indices
        pairs = self.solver._contact.actor_samples.pair_kinds
        planes = set(np.flatnonzero(self.model.shape_type.numpy() == int(newton.GeoType.PLANE)))
        hand_shapes = {r["shape"] for r in self.identity["collision_mapping"]}
        self.identity["hand_world_plane_pairs"] = [list(p) for p in pairs if set(p) & planes and set(p) & hand_shapes]
        self.identity["actor_contact_pairs"] = [{"shapes": list(p), "kind": kind} for p, kind in pairs.items()]
        self.identity["contact_samples"] = self.solver._contact.count
        errors, seen = {}, set()
        sdf_indices = self.model._shape_sdf_index.numpy()
        for entry in self.identity["collision_mapping"]:
            shape = entry["shape"]
            mesh = self.model.shape_source[shape]
            if id(mesh) in seen:
                continue
            seen.add(id(mesh))
            vertices = np.asarray(mesh.vertices, dtype=float)
            centroids = vertices[np.asarray(mesh.indices).reshape(-1, 3)].mean(axis=1)
            points = np.concatenate((vertices, centroids))
            values = wp.empty((len(points), 13), dtype=wp.float64, device=device)
            field = self.smooth.fields[int(sdf_indices[shape])]
            wp.launch(
                sample_smooth,
                dim=len(points),
                inputs=[field.data, wp.array(points, dtype=wp.vec3d, device=device), values],
                device=device,
            )
            errors[entry["source"]] = float(np.max(np.abs(values.numpy()[:, 0])))
        self.identity["surface_sdf_error_m"] = errors
        self.identity.update(control_mode=control_mode, contact_stiffness_N_m3=contact_stiffness)
        self.hand_shape_fingers = {r["shape"]: finger_index(r["link"]) for r in self.identity["collision_mapping"]}
        self.ground_shapes = planes
        body = self.identity["object_body"]
        self.object_shapes = set(np.flatnonzero(self.model.shape_body.numpy() == body)) if body is not None else set()
        if control_mode == "contact":
            rates = []
            for finger in range(5):
                slots = [i for i, name in enumerate(self.curve.names) if finger_index(name) == finger]
                slope = np.max(np.abs(self.curve.slopes[:, slots]), axis=0)
                limits = self.model.joint_velocity_limit.numpy()[np.array(self.hand_dofs)[slots]]
                rates.append(float(np.min(limits / np.maximum(slope, 1e-12))))
            mass = (
                float(self.model.body_mass.numpy()[body])
                if body is not None
                else float(self.model.particle_mass.numpy().sum())
            )
            self.controller = ContactHandControl(task, mass=mass, friction=friction, source_rates=rates)
            self.identity["contact_control_parameters"] = self.controller.parameters

    def penetration(self, *, by_finger=None):
        """Sample the published geometry independently of residual and active contact selection.

        Body output poses are FP32 even in an FP64 solve; this millimetre-scale
        audit is not an oracle for native-precision force roundoff.
        """
        model, state = self.model, self.state
        poses = [transform(p) for p in state.body_q.numpy()]
        body = self.identity["object_body"]
        if body is None:
            points = state.particle_q.numpy().astype(float)
            radius = 0.0002
        else:
            local, _ = box_samples(np.full(3, 0.02))
            points = local @ poses[body][:3, :3].T + poses[body][:3, 3]
            radius = 0.0
        penetration = {
            "object_ground_m": max(0.0, float(radius - points[:, 2].min())),
            "hand_ground_m": 0.0,
            "hand_object_m": 0.0,
        }
        indices = model._shape_sdf_index.numpy()
        for entry in self.identity["collision_mapping"]:
            shape, link = entry["shape"], entry["body"]
            frame = poses[link] @ transform(model.shape_transform.numpy()[shape])
            vertices = np.asarray(model.shape_source[shape].vertices, dtype=float)
            world = vertices @ frame[:3, :3].T + frame[:3, 3]
            penetration["hand_ground_m"] = max(penetration["hand_ground_m"], float(-world[:, 2].min()))
            if body is not None:
                local = (world - poses[body][:3, 3]) @ poses[body][:3, :3]
                d = np.abs(local) - 0.02
                distance = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(axis=1), 0)
                depth = max(0.0, float(-distance.min()))
            else:
                depth = 0.0
            local = (points - frame[:3, 3]) @ frame[:3, :3]
            field = self.smooth.fields[int(indices[shape])]
            values = wp.empty((len(local), 13), dtype=wp.float64, device=model.device)
            wp.launch(
                sample_smooth,
                dim=len(local),
                inputs=[field.data, wp.array(local, dtype=wp.vec3d, device=model.device), values],
                device=model.device,
            )
            depth = max(depth, float(radius - values.numpy()[:, 0].min()))
            penetration["hand_object_m"] = max(penetration["hand_object_m"], depth)
            finger = finger_index(entry["link"])
            if by_finger is not None and finger is not None:
                by_finger[finger] = max(by_finger[finger], depth)
        return penetration

    def targets(self, t):
        if self.controller is None:
            source, rate, move, velocity, phase = schedule(self.task, t)
            q, v = self.curve.sample(source)
            v *= rate
        else:
            q, v = np.zeros(22), np.zeros(22)
            for finger in range(5):
                values, speeds = self.curve.sample(self.controller.sources[finger])
                slots = [i for i, name in enumerate(self.curve.names) if finger_index(name) == finger]
                q[slots], v[slots] = values[slots], speeds[slots] * self.controller.rates[finger]
            move, velocity, phase = self.controller.carriage, self.controller.carriage_velocity, self.controller.phase
        position, speed = np.zeros(self.model.joint_target_q.shape[0]), np.zeros(self.model.joint_dof_count)
        for i, name in enumerate(self.curve.names):
            _, dof, slot = self.mapping[name]
            position[slot], speed[dof] = q[i], v[i]
        _, dof, slot = self.mapping["monolithic_lift"]
        position[slot], speed[dof] = move, velocity
        return position, speed, phase

    def step(self, t):
        if self.controller is not None:
            self.control_before_step = {
                "time": t,
                "controller": self.controller.snapshot(),
                "feedback": None if self.feedback is None else self.feedback.snapshot(),
            }
            self.controller.advance(t, self.feedback)
        q, v, phase = self.targets(t)
        self.control.joint_target_q.assign(q)
        self.control.joint_target_qd.assign(v)
        self.solver.step(self.state, self.state, self.control, dt=DT)
        if self.controller is not None and self.solver.last_stats.converged and not self.solver.last_stats.rolled_back:
            self.feedback = self.observe_contact()
        return phase

    def observe_contact(self):
        """Read only successfully published pair forces and geometry for the next step."""
        forces = self.solver.get_contact_forces(self.state)
        fingers, floor, support, hand_object = contact_loads(
            self.solver.last_stats.variational.get("actor_contact_pairs", ()),
            self.hand_shape_fingers,
            self.object_shapes,
            self.ground_shapes,
            material=self.identity["object_body"] is None,
            generation=forces.force_generation,
        )
        depths = np.zeros(5)
        self.feedback_geometry = self.penetration(by_finger=depths)
        body = self.identity["object_body"]
        if body is None:
            x = self.state.particle_q.numpy().astype(float)
            mass = self.model.particle_mass.numpy()
            position = np.average(x, axis=0, weights=mass)
            velocity = np.average(self.state.particle_qd.numpy(), axis=0, weights=mass)
            bottom = float(x[:, 2].min() - 0.0002)
        else:
            frame = transform(self.state.body_q.numpy()[body])
            position, velocity = frame[:3, 3], self.state.body_qd.numpy()[body, :3].astype(float)
            bottom = float(position[2] - 0.02 * np.abs(frame[2, :3]).sum())
        open_error = float(np.max(np.abs(self.state.joint_q.numpy()[self.hand_coords] - self.curve.q[0])))
        return HandObservation(
            forces.force_generation,
            fingers,
            floor,
            depths,
            support,
            position,
            velocity,
            bottom,
            valid=forces.force_valid,
            open_error_rad=open_error,
            hand_object_force=hand_object,
            hand_object_penetration=self.feedback_geometry["hand_object_m"],
        )

    def measure(self, t, phase):
        run, state, model = self.solver, self.state, self.model
        stats = run.last_stats
        forces = run.get_contact_forces(state)
        dofs = self.hand_dofs
        q, v = state.joint_q.numpy()[self.hand_coords], state.joint_qd.numpy()[dofs]
        lower, upper = model.joint_limit_lower.numpy()[dofs], model.joint_limit_upper.numpy()[dofs]
        pd = stats.joint_pd_force
        hand_object_force = None
        if stats.converged and not stats.rolled_back and forces.force_valid:
            pairs = stats.variational.get("actor_contact_pairs")
            if pairs is not None:
                try:
                    _, _, _, hand_object_force = contact_loads(
                        pairs,
                        self.hand_shape_fingers,
                        self.object_shapes,
                        self.ground_shapes,
                        material=self.identity["object_body"] is None,
                        generation=forces.force_generation,
                    )
                except ValueError:
                    pass  # Missing or invalid published object-force evidence stays unknown.
        row = {
            "time": t,
            "phase": phase,
            "converged": bool(stats.converged),
            "rolled_back": bool(stats.rolled_back),
            "newton": stats.nonlinear_iterations,
            "pcg": stats.linear_iterations,
            "contact_force_N": float(np.linalg.norm(forces.body_wrench[list(self.hand_bodies), :3], axis=1).sum()),
            "hand_object_force_N": hand_object_force,
            "joint_limit_violation_rad": float(max(0, np.max(lower - q), np.max(q - upper))),
            "speed_ratio": float(np.max(np.abs(v) / model.joint_velocity_limit.numpy()[dofs])),
            "effort_ratio": float(np.max(np.abs(np.asarray(pd)[dofs]) / model.joint_effort_limit.numpy()[dofs]))
            if pd is not None
            else None,
            "failure_reason": stats.failure_reason,
        }
        body = self.identity["object_body"]
        if body is None:
            x = state.particle_q.numpy().astype(float)
            mass = model.particle_mass.numpy()
            row["object_position"] = np.average(x, axis=0, weights=mass).tolist()
            tets = model.tet_indices.numpy()
            edge = np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1))
            row["min_det_f"] = float(np.linalg.det(edge @ model.tet_poses.numpy()).min())
        else:
            row["object_position"] = state.body_q.numpy()[body, :3].astype(float).tolist()
            row["min_det_f"] = 1.0
        object_dof = self.identity["object_dof"]
        if object_dof is None:
            row["object_active_force_N"] = 0.0
        elif pd is None:
            row["object_active_force_N"] = None
        else:
            row["object_active_force_N"] = float(abs(self.control.joint_f.numpy()[object_dof]) + abs(pd[object_dof]))
        if self.controller is not None:
            row["contact_control"] = {
                "sources": self.controller.sources.tolist(),
                "source_rates": self.controller.rates.tolist(),
                "carriage_m": self.controller.carriage,
                "carriage_speed_m_s": self.controller.carriage_velocity,
                "failure": self.controller.failure,
                "completed": self.controller.completed,
                "feedback": None if self.feedback is None else self.feedback.snapshot(),
            }
        return row


def physical_gates(task, rows, *, height=0.0):
    """Separate physical task success from solver convergence; never accept a static object."""
    position = np.array([r["object_position"] for r in rows])
    late = position[np.array([r["time"] for r in rows]) >= DURATION - 0.25]
    target = np.array([0.05 if task == "pull-slider" else 0, 0, 0.02 + height])
    metrics = {
        "lift_m": float(position[:, 2].max() - (0.0202 + height)),
        "final_position_error_m": float(np.linalg.norm(position[-1] - target)),
        "final_drift_m": float(np.max(np.linalg.norm(late - late[-1], axis=1))) if len(late) else None,
    }
    for key in ("contact_force_N", "joint_limit_violation_rad", "speed_ratio", "effort_ratio", "object_active_force_N"):
        values = [r[key] for r in rows]
        metrics["max_" + key] = max(values) if all(value is not None for value in values) else None
    object_forces = [r.get("hand_object_force_N") for r in rows]
    metrics["max_hand_object_force_N"] = (
        max(object_forces) if all(value is not None for value in object_forces) else None
    )
    metrics["min_det_f"] = min(r["min_det_f"] for r in rows)
    measurements = [r["sample_penetration_m"] for r in rows if "sample_penetration_m" in r]
    metrics["max_sample_penetration_m"] = max(measurements) if measurements else None
    gates = {
        "contact": metrics["max_hand_object_force_N"] is not None and metrics["max_hand_object_force_N"] > 0.01,
        "motion": metrics["final_position_error_m"] <= 0.01 if task == "pull-slider" else metrics["lift_m"] >= 0.03,
        "placement": metrics["final_position_error_m"] <= 0.01,
        "released_stable": metrics["final_drift_m"] is not None and metrics["final_drift_m"] <= 0.002,
        "joint_limits": metrics["max_joint_limit_violation_rad"] <= 0.02,
        "speed": metrics["max_speed_ratio"] <= 1.2,
        "effort": metrics["max_effort_ratio"] is not None and metrics["max_effort_ratio"] <= 1.2,
        "passive_object": metrics["max_object_active_force_N"] == 0,
        "material": metrics["min_det_f"] >= 0.2,
        "penetration": metrics["max_sample_penetration_m"] is not None
        and metrics["max_sample_penetration_m"] <= 0.0015,
    }
    return metrics, gates


def source_identity():
    """Bind runs to actual imported code, including uncommitted task implementations."""
    paths = [*sorted((REPO / "newton").rglob("*.py")), *sorted(Path(__file__).parent.glob("*.py"))]
    return {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_case(task, args):
    output = args.output / task
    output.mkdir()
    start = time.perf_counter()
    scene = HandTask(
        task,
        args.assets,
        friction=args.friction,
        offset=args.offset,
        surface_source=args.surface_source,
        height=args.height,
        control_mode=args.control_mode,
        contact_stiffness=args.contact_stiffness,
    )
    initial_penetration = scene.penetration()
    save_json(
        output / "scene.json",
        {
            **scene.identity,
            "smooth_sdf": scene.smooth.identity,
            "initial_sample_penetration_m": max(initial_penetration.values()),
            "initial_penetration": initial_penetration,
            "configuration_digest": scene.solver._kinematics.digest.hex(),
        },
    )
    frames = {name: [] for name in ("time", "body_q", "particle_q", "joint_q", "phase")}

    def record(t, phase):
        for key, value in zip(
            frames,
            (t, scene.state.body_q.numpy(), scene.state.particle_q.numpy(), scene.state.joint_q.numpy(), phase),
            strict=True,
        ):
            frames[key].append(value)

    record(0, "initial")
    rows, error = [], None
    limit = min(args.max_steps or round(DURATION / DT), round(DURATION / DT))
    try:
        with (output / "trace.jsonl").open("w") as stream:
            for step in range(1, limit + 1):
                phase = scene.step(step * DT)
                row = scene.measure(step * DT, phase)
                if step == 1 or step % 25 == 0 or step == limit or not row["converged"]:
                    row["penetration"] = (
                        scene.feedback_geometry if scene.feedback_geometry is not None else scene.penetration()
                    )
                    row["sample_penetration_m"] = max(*initial_penetration.values(), *row["penetration"].values())
                if scene.controller is not None and scene.feedback_geometry is not None:
                    row["penetration"] = scene.feedback_geometry
                    row["sample_penetration_m"] = max(*initial_penetration.values(), *row["penetration"].values())
                rows.append(row)
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                if step % 25 == 0 or step == limit or not row["converged"]:
                    record(step * DT, phase)
                if not row["converged"] or row["rolled_back"]:
                    error = f"Step {step}: {scene.solver.last_stats.status}: {row['failure_reason']}"
                    scene.solver.save_checkpoint(scene.state, output / "last-published-state.npz")
                    if scene.controller is not None:
                        save_json(output / "control-before-failed-step.json", scene.control_before_step)
                    break
                if step % 100 == 0:
                    print(
                        f"{task} {step}/{limit}: Newton={row['newton']}, force={row['contact_force_N']:.3g} N",
                        flush=True,
                    )
    except Exception:
        error = traceback.format_exc()
    np.savez_compressed(output / "trajectory.npz", **{key: np.array(value) for key, value in frames.items()})
    metrics, gates = physical_gates(task, rows, height=args.height) if rows else ({}, {})
    gates["surface_sdf_consistency"] = max(scene.identity["surface_sdf_error_m"].values()) <= 0.0004
    if scene.controller is not None:
        gates["controller_completed"] = scene.controller.completed and scene.controller.failure is None
        save_json(output / "controller-final.json", scene.controller.snapshot())
    complete = len(rows) == round(DURATION / DT) and error is None
    result = {
        "task": task,
        "status": "PASS" if complete and all(gates.values()) else "FAIL" if complete or error else "PARTIAL",
        "complete": complete,
        "numerical_pass": complete and all(r["converged"] and not r["rolled_back"] for r in rows),
        "steps": len(rows),
        "expected_steps": round(DURATION / DT),
        "error": error,
        "metrics": metrics,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - start,
    }
    save_json(output / "summary.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Retained inputs/sharpa directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=(*TASKS, "all"), default="all")
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--offset", type=float, default=0, help="Initial object offset in world Y [m]")
    parser.add_argument("--height", type=float, default=0, help="Raise hand and slider together [m], pull-slider only")
    parser.add_argument("--control-mode", choices=("open-loop", "contact"), default="open-loop")
    parser.add_argument("--contact-stiffness", type=float, default=1e7, help="Normal contact stiffness [N/m^3]")
    parser.add_argument(
        "--surface-source",
        choices=("qualified", "cad"),
        default="qualified",
        help="CAD surface quadrature with unchanged qualified SDFs is an explicit task-input adaptation",
    )
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if (
        not np.isfinite([args.friction, args.offset, args.height, args.contact_stiffness]).all()
        or args.friction < 0
        or args.height < 0
        or args.contact_stiffness <= 0
        or (args.height != 0 and args.task != "pull-slider")
        or (args.max_steps is not None and args.max_steps <= 0)
    ):
        parser.error(
            "Require finite nonnegative friction/height, positive contact-stiffness, finite offset and positive max-steps; "
            "height requires pull-slider"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    before = source_identity()
    save_json(
        args.output / "manifest.json",
        {
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "newton_import": newton.__file__,
            "source_sha256": before,
            "assets": str(args.assets.resolve()),
            "task": args.task,
            "friction": args.friction,
            "offset": args.offset,
            "height": args.height,
            "control_mode": args.control_mode,
            "contact_stiffness_N_m3": args.contact_stiffness,
            "surface_source": args.surface_source,
            "precision": "fp64",
            "device": "cuda:0",
            "dt": DT,
            "duration": DURATION,
            "max_steps": args.max_steps,
            "input_kind": "scripted full Sharpa hand with scalar fixed-base carriage; passive task object",
        },
    )
    results = []
    for task in TASKS if args.task == "all" else (args.task,):
        try:
            results.append(run_case(task, args))
        except Exception:
            result = {"task": task, "status": "ERROR", "error": traceback.format_exc()}
            save_json(args.output / task / "summary.json", result)
            results.append(result)
            print(result["error"], flush=True)
    save_json(args.output / "summary.json", {"cases": results, "source_unchanged": source_identity() == before})
    raise SystemExit(0 if all(r["status"] == "PASS" for r in results) else 1)


if __name__ == "__main__":
    main()
