# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the internal, collision-free Sharpa G1H diagnostic for four seconds."""

import json
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
import newton.examples
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_hand, sha256


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.parameters = json.loads(Path(args.fixture).read_text())
        self.model, self.manifest = load_hand(args.asset_dir, device=args.device, parameters=self.parameters)
        self.trajectory = ClosureTrajectory(
            args.trajectory,
            self.manifest["joint_names"],
            self.model.joint_limit_lower.numpy(),
            self.model.joint_limit_upper.numpy(),
            self.model.joint_velocity_limit.numpy(),
        )
        self.state = self.model.state()
        q, _ = self.trajectory.sample(0)
        self.state.joint_q.assign(q.astype(np.float32))
        newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.control = self.model.control()
        self.control.joint_target_q.assign(q.astype(np.float32))
        self.control.joint_target_qd.zero_()
        configs = [self.parameters["joints"][n] for n in self.trajectory.names]
        self.solver = SolverMonolithic._create_joint_diagnostic(
            self.model,
            joint_terms=SolverMonolithic.JointTerms(
                implicit_pd=True,
                limits=True,
                friction=True,
                limit_width=tuple(c["limit_width"] for c in configs),
                friction_velocity_scale=tuple(c["friction_velocity_scale"] for c in configs),
            ),
        )
        self.sim_dt = float(self.parameters["dt"])
        self.frame_dt = 0.01
        self.sim_substeps = round(self.frame_dt / self.sim_dt)
        if self.sim_dt <= 0 or abs(self.sim_substeps * self.sim_dt - self.frame_dt) > 1e-12:
            raise ValueError("Fixture dt must divide 10 ms")
        self.substeps = 0
        self.sim_time = 0.0
        self.records = []
        self.output = Path(args.output) if args.output else None
        self.manifest.update(
            trajectory_sha256=sha256(args.trajectory),
            fixture_sha256=sha256(args.fixture),
            mapping=self.trajectory.mapping,
            parameters=self.parameters,
            device=str(self.model.device),
            scope="G1H q-only; x/tet/contact metrics NOT_APPLICABLE",
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.35, -0.35, 0.25), pitch=-20.0, yaw=130.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--asset-dir", required=True)
        parser.add_argument("--trajectory", required=True)
        parser.add_argument("--fixture", default=str(Path(__file__).with_name("sharpa_g1h.json")))
        parser.add_argument("--output")
        parser.set_defaults(num_frames=400)
        return parser

    def step(self):
        for _ in range(self.sim_substeps):
            if self.substeps * self.sim_dt >= 4.0 - 1e-12:
                break
            time_s = (self.substeps + 1) * self.sim_dt
            q, qd = self.trajectory.sample(time_s)
            self.control.joint_target_q.assign(q.astype(np.float32))
            self.control.joint_target_qd.assign(qd.astype(np.float32))
            self.solver.step(self.state, self.state, self.control, None, self.sim_dt)
            stats = self.solver.last_stats
            self.records.append(
                {
                    "time": time_s if not stats.rolled_back else self.substeps * self.sim_dt,
                    "target_time": time_s,
                    "q_target": q.tolist(),
                    "qd_target": qd.tolist(),
                    "q": self.state.joint_q.numpy().tolist(),
                    "qd": self.state.joint_qd.numpy().tolist(),
                    "pd_force": stats.joint_pd_force,
                    "limit_force": stats.joint_limit_force,
                    "friction_force": stats.joint_friction_force,
                    "friction_power": stats.joint_friction_power,
                    "saturated": stats.joint_pd_saturated,
                    "status": stats.status.value,
                    "converged": stats.converged,
                    "rolled_back": stats.rolled_back,
                    "body_finite": bool(
                        np.isfinite(self.state.body_q.numpy()).all() and np.isfinite(self.state.body_qd.numpy()).all()
                    ),
                    "ratio_global": stats.convergence_ratio,
                    "ratio_q": stats.convergence_ratio_q,
                    "step_ms": stats.timings["step"],
                    "nonlinear_iterations": stats.nonlinear_iterations,
                }
            )
            if stats.rolled_back:
                self.write_results()
                raise RuntimeError(f"G1H rollback at target time {time_s}: {stats.failure_reason}")
            self.substeps += 1
        self.sim_time = self.substeps * self.sim_dt
        if self.sim_time >= 4.0:
            self.write_results()

    def summarize(self):
        if not self.records:
            return {"passed": False, "reason": "No simulated steps"}
        r = self.records
        q, qd, qt = (np.asarray([v[k] for v in r]) for k in ("q", "qd", "q_target"))
        error = np.abs(q - qt)
        hold = np.asarray([v["time"] >= 3.0 for v in r])
        gates = self.parameters["gates"]
        displacement = self.trajectory.q[-1] - self.trajectory.q[0]
        moving = np.abs(displacement) >= gates["moving_threshold"]
        completion = np.ones(22)
        completion[moving] = (q[-1, moving] - self.trajectory.q[0, moving]) / displacement[moving]
        limit = np.maximum(
            np.maximum(self.model.joint_limit_lower.numpy() - q, q - self.model.joint_limit_upper.numpy()), 0
        ).max(axis=0)
        velocity_ratio = (np.abs(qd) / self.model.joint_velocity_limit.numpy()).max(axis=0)
        effort_ratio = (np.abs(np.asarray([v["pd_force"] for v in r])) / self.model.joint_effort_limit.numpy()).max(
            axis=0
        )
        drift = np.ptp(q[hold], axis=0) if hold.any() else np.full(22, np.inf)
        hold_speed = np.abs(qd[hold]).max(axis=0) if hold.any() else np.full(22, np.inf)
        max_error, final_error = error.max(axis=0), error[-1]
        passed = (
            (max_error <= gates["max_tracking_error"])
            & (final_error <= gates["final_tracking_error"])
            & (drift <= gates["hold_drift"])
            & (hold_speed <= gates["hold_speed"])
            & (limit <= gates["limit_violation"])
            & (velocity_ratio <= gates["velocity_ratio"])
            & (effort_ratio <= gates["effort_ratio"])
            & (~moving | (completion >= gates["completion_min"]))
        )
        fraction = sum(v["converged"] for v in r) / len(r)
        residual_pass = all(not v["converged"] or max(v["ratio_q"], v["ratio_global"]) <= 1 for v in r)
        finite = all(v["body_finite"] for v in r) and all(
            np.isfinite(np.asarray([v[k] for v in r])).all()
            for k in ("q", "qd", "pd_force", "limit_force", "friction_force", "friction_power")
        )
        power_pass = bool(np.all(np.asarray([v["friction_power"] for v in r]) <= 1e-12))
        return {
            "passed": bool(
                passed.all()
                and fraction >= gates["converged_fraction"]
                and finite
                and power_pass
                and residual_pass
                and self.sim_time >= 4.0
                and not any(v["rolled_back"] for v in r)
            ),
            "steps": len(r),
            "sim_time": self.sim_time,
            "converged_fraction": fraction,
            "finite": bool(finite),
            "residual_gates_passed": residual_pass,
            "friction_dissipative": power_pass,
            "x": "NOT_APPLICABLE",
            "tet": "NOT_APPLICABLE",
            "contact": "NOT_APPLICABLE",
            "contact_count": 0,
            "sdf_queries": 0,
            "joints": [
                {
                    "name": name,
                    "moving": bool(moving[i]),
                    "passed": bool(passed[i]),
                    "max_error": float(max_error[i]),
                    "final_error": float(final_error[i]),
                    "completion": float(completion[i]) if moving[i] else None,
                    "overshoot": float(
                        max(0.0, np.max(np.sign(displacement[i]) * (q[:, i] - self.trajectory.q[-1, i])))
                    )
                    if moving[i]
                    else None,
                    "hold_drift": float(drift[i]),
                    "hold_speed": float(hold_speed[i]),
                    "limit_violation": float(limit[i]),
                    "velocity_ratio": float(velocity_ratio[i]),
                    "effort_ratio": float(effort_ratio[i]),
                    "saturation_fraction": float(np.asarray([v["saturated"] for v in r])[:, i].mean()),
                }
                for i, name in enumerate(self.trajectory.names)
            ],
        }

    def write_results(self):
        def serializable(value):
            if isinstance(value, dict):
                return {k: serializable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [serializable(v) for v in value]
            return None if isinstance(value, float) and not np.isfinite(value) else value

        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
            for name, value in (("manifest", self.manifest), ("summary", self.summarize())):
                (self.output / f"{name}.json").write_text(
                    json.dumps(serializable(value), indent=2, allow_nan=False) + "\n"
                )
            with (self.output / "trace.jsonl").open("w") as stream:
                for record in self.records:
                    stream.write(json.dumps(serializable(record), allow_nan=False) + "\n")

    def test_post_step(self):
        """Require finite states and the production q/global gates on converged steps."""
        assert np.isfinite(self.state.joint_q.numpy()).all()
        assert np.isfinite(self.state.joint_qd.numpy()).all()
        stats = self.solver.last_stats
        assert not stats.rolled_back
        if stats.converged:
            assert stats.convergence_ratio <= 1 and stats.convergence_ratio_q <= 1

    def test_final(self):
        """Require the complete per-joint G1H acceptance."""
        self.write_results()
        summary = self.summarize()
        assert summary["passed"], summary

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
