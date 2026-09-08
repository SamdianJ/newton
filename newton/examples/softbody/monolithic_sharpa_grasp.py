# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exploratory PR-7A assembly and controls; G6/G7 calibration remains separate."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline, SolverMonolithic

import newton
from newton.examples.softbody.monolithic_sharpa_assets import load_hand_ball
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture, sha256

FIXTURE = {
    "schema": "sharpa-grasp-exploratory/v1",
    "status": "G6_G7_NOT_VALIDATED",
    "dt": 0.001,
    "frame_dt": 0.01,
    "duration": 9.0,
    "ball_position": [0.025, -0.03, 0.19],
    "particle_radius": 0.0002,
    "sdf_query_error": 0.0004,
    "sdf_resolution": 128,
    "gap": 0.002,
    "mu": 0.5,
    "tangential_stiffness": 2e6,
    "smoothing_width": 0.0001,
    "lift_height": 0.05,
    "stage_ends": [0.5, 2.5, 4.5, 5.5, 6.5, 8.5, 9.0],
    "carriage": {
        "target_ke": 10000.0,
        "target_kd": 250.0,
        "limit_ke": 100000.0,
        "limit_kd": 250.0,
        "lower": -0.01,
        "upper": 0.07,
        "effort": 100.0,
        "velocity": 0.1,
        "friction": 0.1,
        "limit_width": 0.001,
        "friction_velocity_scale": 0.001,
    },
    "parameter_source": "simulation integration defaults, not hardware identification or a frozen grasp gate",
}


def stage_target(trajectory, time_s):
    """Sample continuous positions and same-segment velocities at physical step end."""
    if not np.isfinite(time_s) or time_s < 0 or time_s > FIXTURE["duration"] + 1e-12:
        raise ValueError("Time outside the fixed nine-second schedule")
    if time_s <= 0.5:
        stage, source, rate = "prepare", 0.0, 0.0
    elif time_s <= 2.5:
        stage, source, rate = "close", time_s - 0.5, 1.0
    elif time_s <= 4.5:
        stage, source, rate = "hold", time_s - 0.5, 1.0
    elif time_s <= 5.5:
        stage, source, rate = "lift", 4.0, 0.0
    elif time_s <= 6.5:
        stage, source, rate = "hold_high", 4.0, 0.0
    elif time_s <= 8.5:
        stage, source, rate = "release", 2.0 - (time_s - 6.5), -1.0
    else:
        stage, source, rate = "observe", 0.0, 0.0
    q, v = trajectory.sample(source)
    u = min(max(time_s - 4.5, 0.0), 1.0)
    lift = FIXTURE["lift_height"] * u**3 * (10 - 15 * u + 6 * u * u)
    lift_v = FIXTURE["lift_height"] * 30 * u * u * (1 - u) ** 2
    return stage, q, rate * v, lift, lift_v


def joint_mapping(model, names):
    """Resolve coordinate, rate and target offsets independently by joint name."""
    labels = [label.rsplit("/", 1)[-1] for label in model.joint_label]
    if any(labels.count(name) != 1 for name in [*names, "monolithic_lift"]):
        raise ValueError("Missing or duplicate hand/carriage joint")
    starts = [getattr(model, name).numpy() for name in ("joint_q_start", "joint_qd_start", "joint_target_q_start")]
    return {name: tuple(int(start[labels.index(name)]) for start in starts) for name in [*names, "monolithic_lift"]}


def load_calibration(path):
    """Require a complete frozen three-grid scan and its unchanged protocol."""
    result = json.loads(Path(path).read_text())
    if result.get("status") != "FROZEN" or result.get("pcg_iteration_p95_budget") != 64:
        raise ValueError("Freeze the PR-7A calibration before running the grasp stages")
    protocol = result.get("protocol", {})
    digest = hashlib.sha256((json.dumps(protocol, indent=2) + "\n").encode()).hexdigest()
    if digest != result.get("protocol_sha256"):
        raise ValueError("Calibration protocol hash mismatch")
    rows = result.get("results", [])
    grid = protocol.get("refinements", [])
    scan = protocol.get("stiffness_scan_n_m3", [])
    if grid != [1, 2, 3] or len(rows) != len(grid) * len(scan):
        raise ValueError("Incomplete calibration scan")
    passing = []
    for stiffness in scan:
        cells = [
            [r for r in rows if r.get("refinement") == refinement and r.get("stiffness") == stiffness]
            for refinement in grid
        ]
        if any(len(c) != 1 for c in cells):
            raise ValueError("Missing or duplicate calibration point")
        if all(c[0].get("passed") is True for c in cells):
            passing.append(stiffness)
    floor = result.get("stiffness_floor_n_m3")
    if not passing or floor != min(passing) or not np.isfinite(floor) or floor <= 0:
        raise ValueError("Invalid frozen contact stiffness floor")
    if result.get("initial_grasp_stiffness_n_m3") != max(1e7, floor):
        raise ValueError("Grasp stiffness does not match calibration")
    return result


class GraspCase:
    """Run all stages through the existing coupled solver and its state transaction."""

    def __init__(self, args):
        device = wp.get_device(args.device)
        if not device.is_cuda:
            raise ValueError("Sharpa mesh SDF grasp requires CUDA; CPU components have independent tests")
        calibration = load_calibration(args.calibration)
        floor = calibration.get("stiffness_floor_n_m3")
        if not isinstance(floor, (int, float)) or not np.isfinite(floor) or floor <= 0:
            raise ValueError("Invalid frozen contact stiffness floor")
        self.parameters = load_fixture(Path(__file__).with_name("sharpa_g1h.json"))
        with np.load(args.ball, allow_pickle=False) as ball:
            # The archive format is checked again by load_hand_ball/load_ball.
            rest_positions = ball["vertices"] if "vertices" in ball else ball["rest_positions"]
            height = FIXTURE["ball_position"][2] + float(rest_positions[:, 2].min()) - FIXTURE["particle_radius"]
        mount = {"support_height": height, "carriage": FIXTURE["carriage"]}
        self.model, self.manifest = load_hand_ball(
            args.asset_dir,
            args.contact_dir,
            args.ball,
            device=device,
            parameters=self.parameters,
            position=FIXTURE["ball_position"],
            resolution=128,
            _mount=mount,
        )
        model = self.model
        self.mapping = joint_mapping(model, self.manifest["joint_names"])
        dofs = [self.mapping[n][1] for n in self.manifest["joint_names"]]
        self.trajectory = ClosureTrajectory(
            args.trajectory,
            self.manifest["joint_names"],
            model.joint_limit_lower.numpy()[dofs],
            model.joint_limit_upper.numpy()[dofs],
            model.joint_velocity_limit.numpy()[dofs],
        )
        model.request_contact_attributes("force")
        pipeline = MonolithicCollisionPipeline(
            model,
            soft_contact_gap=FIXTURE["gap"],
            _sdf_query_error=FIXTURE["sdf_query_error"],
            _enable_aabb=not args.disable_aabb,
        )
        width, velocity = np.empty(model.joint_dof_count), np.empty(model.joint_dof_count)
        for name, (_, dof, _) in self.mapping.items():
            cfg = FIXTURE["carriage"] if name == "monolithic_lift" else self.parameters["joints"][name]
            width[dof], velocity[dof] = cfg["limit_width"], cfg["friction_velocity_scale"]
        self.solver = SolverMonolithic(
            model,
            collision_pipeline=pipeline,
            contact_stiffness=max(1e7, floor),
            normal_smoothing_width=FIXTURE["smoothing_width"],
            friction_coefficient=0.0 if args.friction_off else FIXTURE["mu"],
            tangential_stiffness=FIXTURE["tangential_stiffness"],
            material_model="smith_log_stabilized",
            mass_mode="consistent",
            tet_rest_density=wp.full(model.tet_count, 1000.0, device=device),
            joint_terms=SolverMonolithic.JointTerms(
                implicit_pd=True,
                limits=True,
                friction=True,
                limit_width=tuple(width),
                friction_velocity_scale=tuple(velocity),
            ),
        )
        self.state, self.control = model.state(), model.control()
        self.target_q = np.zeros(model.joint_coord_count, dtype=np.float32)
        self.target_v = np.zeros(model.joint_dof_count, dtype=np.float32)
        self.set_target(0.0)
        initial = self.state.joint_q.numpy()
        for coord, _, target in self.mapping.values():
            initial[coord] = self.target_q[target]
        self.state.joint_q.assign(initial)
        self.state.joint_qd.zero_()
        newton.eval_fk(model, self.state.joint_q, self.state.joint_qd, self.state)
        self.weights = model.particle_mass.numpy().astype(float)
        self.shape_bodies = model.shape_body.numpy()
        self.lower = model.joint_limit_lower.numpy()
        self.upper = model.joint_limit_upper.numpy()
        self.effort = model.joint_effort_limit.numpy()
        self.rest = self.state.particle_q.numpy().astype(float)
        self.rest -= np.average(self.rest, axis=0, weights=self.weights)
        self.step_count = 0
        self.records = []
        self.snapshots = {}
        self.failure = None
        self._pcg_calls = []
        solve = self.solver._linear.solve_pcg

        def measure_pcg(*a, **kw):
            start = time.perf_counter()
            result = solve(*a, **kw)
            self._pcg_calls.append(
                {"iterations": result.iterations, "status": result.status.name, "seconds": time.perf_counter() - start}
            )
            return result

        self.solver._linear.solve_pcg = measure_pcg
        self.manifest.update(
            fixture=FIXTURE,
            calibration_sha256=sha256(args.calibration),
            trajectory_sha256=sha256(args.trajectory),
            mapping=self.mapping,
            friction_off=args.friction_off,
            aabb=not args.disable_aabb,
            normal_stiffness=max(1e7, floor),
            pcg_iteration_p95_budget=64,
            force_convention="world force on rigid body; negate for force on the ball",
            scope="PR-7A exploratory integration; G6/G7 NOT_VALIDATED",
        )

    def set_target(self, time_s):
        """Freeze the complete hand and carriage control for one physical step."""
        stage, q, qd, lift, lift_v = stage_target(self.trajectory, time_s)
        for i, name in enumerate(self.trajectory.names):
            _, dof, target = self.mapping[name]
            self.target_q[target], self.target_v[dof] = q[i], qd[i]
        _, dof, target = self.mapping["monolithic_lift"]
        self.target_q[target], self.target_v[dof] = lift, lift_v
        self.control.joint_target_q.assign(self.target_q)
        self.control.joint_target_qd.assign(self.target_v)
        return stage

    def step(self):
        """Advance one accepted step, stopping without clock advancement on rollback."""
        if self.failure or self.step_count >= 9000:
            return
        time_s = (self.step_count + 1) * FIXTURE["dt"]
        stage = self.set_target(time_s)
        self._pcg_calls.clear()
        start = time.perf_counter()
        try:
            self.solver.step(self.state, self.state, self.control, None, FIXTURE["dt"])
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            return
        elapsed = time.perf_counter() - start
        stats = self.solver.last_stats
        if stats.rolled_back:
            self.failure = f"{stats.status.name}: {stats.failure_reason}"
        else:
            self.step_count += 1
        x = self.state.particle_q.numpy().astype(float)
        com = np.average(x, axis=0, weights=self.weights)
        centered = x - com
        u, _, vt = np.linalg.svd(self.rest.T @ (self.weights[:, None] * centered))
        rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
        deformation = float(
            np.sqrt(np.average(np.sum((centered - self.rest @ rotation) ** 2, axis=1), weights=self.weights))
        )
        contacts = self.solver.contacts
        count = int(contacts.soft_contact_count.numpy()[0])
        shape = contacts.soft_contact_shape.numpy()[:count]
        body = self.shape_bodies[shape]
        forces = self.solver._contact.final_force_linear.numpy()[:count]
        per_finger = {}
        for name in ("thumb", "index", "middle", "ring", "pinky"):
            mask = [name in self.model.body_label[b] for b in body]
            per_finger[name] = np.sum(forces[np.asarray(mask, dtype=bool)], axis=0).tolist()
        support = np.sum(forces[body == 0], axis=0)
        q = self.state.joint_q.numpy()
        qd = self.state.joint_qd.numpy()
        hand_root = self.state.body_q.numpy()[1, :3]
        tracking = np.zeros_like(qd)
        violation = np.zeros_like(qd)
        for coord, dof, target in self.mapping.values():
            tracking[dof] = q[coord] - self.target_q[target]
            violation[dof] = max(self.lower[dof] - q[coord], q[coord] - self.upper[dof], 0)
        if self.step_count in (500, 2500, 4500, 5500, 6500, 8500, 9000):
            self.snapshots[str(self.step_count)] = {"q": q.copy(), "x": x.copy()}
        self.records.append(
            {
                "time": self.step_count * FIXTURE["dt"],
                "target_time": time_s,
                "stage": stage,
                "converged": stats.converged,
                "rollback": stats.rolled_back,
                "status": stats.status.name,
                "reason": stats.failure_reason,
                "step_seconds": elapsed,
                "pcg_calls": list(self._pcg_calls),
                "q_target": self.target_q.tolist(),
                "qd_target": self.target_v.tolist(),
                "q": q.tolist(),
                "qd": qd.tolist(),
                "tracking_error": tracking.tolist(),
                "limit_violation": violation.tolist(),
                "hand_root": hand_root.tolist(),
                "pd_force": stats.joint_pd_force,
                "pd_saturated": stats.joint_pd_saturated,
                "limit_force": stats.joint_limit_force,
                "friction_force": stats.joint_friction_force,
                "joint_friction_power": stats.joint_friction_power,
                "ball_com": com.tolist(),
                "deformation_rms": deformation,
                "support_force": support.tolist(),
                "finger_forces": per_finger,
                "support_present": bool(np.linalg.norm(support) > 1e-6),
                "penetration": stats.max_penetration,
                "min_det_f": stats.min_det_f,
                "residual_ratio": stats.convergence_ratio,
                "q_residual_ratio": stats.convergence_ratio_q,
                "x_residual_ratio": stats.convergence_ratio_x,
                "history": dict(stats.contact_history),
                "query_counts": self.solver.collision_pipeline._query_counts.numpy().tolist(),
            }
        )

    def save(self, output):
        """Publish actual execution and performance outcomes without asserting a grasp gate."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        pcg = {}
        for stage in ("close", "hold", "lift", "hold_high"):
            calls = [c["iterations"] for r in self.records if r["stage"] == stage for c in r["pcg_calls"]]
            pcg[stage] = float(np.percentile(calls, 95)) if calls else None
        summary = {
            "complete_schedule": self.step_count == 9000 and self.failure is None,
            "accepted_steps": self.step_count,
            "failure": self.failure,
            "pcg_iteration_p95": pcg,
            "patch_schur_trigger": any(v is not None and v > 64 for v in pcg.values()),
            "converged_fraction": float(np.mean([r["converged"] for r in self.records])) if self.records else 0,
            "G6": "NOT_VALIDATED",
            "G7": "NOT_VALIDATED",
        }
        high = [r for r in self.records if r["stage"] == "hold_high"]
        if high:
            relative = np.array([np.array(r["ball_com"]) - r["hand_root"] for r in high])
            summary["high_hold_relative_drift_m"] = float(np.max(np.linalg.norm(relative - relative[0], axis=1)))
            summary["high_hold_support_absent"] = all(not r["support_present"] for r in high)
        lift = [r for r in self.records if r["stage"] == "lift"]
        if lift:
            summary["lift_com_rise_m"] = lift[-1]["ball_com"][2] - lift[0]["ball_com"][2]
            summary["lift_support_absent"] = all(not r["support_present"] for r in lift)
        for name, value in (("manifest", self.manifest), ("trace", self.records), ("summary", summary)):
            (output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
        np.savez(
            output / "final_state.npz",
            q=self.state.joint_q.numpy(),
            qd=self.state.joint_qd.numpy(),
            x=self.state.particle_q.numpy(),
            v=self.state.particle_qd.numpy(),
        )
        for step, snapshot in self.snapshots.items():
            np.savez(output / f"state_{step}.npz", **snapshot)
        return summary
