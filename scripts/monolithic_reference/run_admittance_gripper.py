# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise physical-axis admittance by squeezing and releasing a fixed block.

Two independently actuated spherical fingers observe only published contact
reactions. This is a force-control experiment, not a free-object grasp qualifier.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from scripts.monolithic_reference.admittance_control import AxisAdmittance

DT = 0.002
STEPS = 1000
REPO = Path(__file__).resolve().parents[2]


class AdmittanceGripper:
    """Drive two prismatic fingers in SI coordinates using signed reaction forces."""

    def __init__(self, *, device="cpu", precision="fp64"):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        cfg = builder.ShapeConfig(density=0, margin=0, has_particle_collision=False)
        self.fingers, self.origins, self.axes = [], [], []
        for side in (-1, 1):
            body = builder.add_link(mass=0.01, inertia=wp.diag(wp.vec3(1e-6)))
            origin, axis = (side * 0.031, 0, 0), (-side, 0, 0)
            joint = builder.add_joint_prismatic(
                -1,
                body,
                axis=wp.vec3(axis),
                parent_xform=wp.transform(origin, wp.quat_identity()),
                target_ke=1000.0,
                target_kd=6.0,
                effort_limit=10.0,
                actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
                limit_ke=0,
                limit_kd=0,
                armature=0,
                damping=0,
                friction=0,
            )
            builder.add_articulation([joint])
            builder.add_shape_sphere(body, radius=0.01, cfg=cfg)
            self.fingers.append(body)
            self.axes.append(axis)
        self.box = builder.add_shape_box(-1, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)
        self.model = builder.finalize(device=device)
        self.axes = np.asarray(self.axes)
        self.origins = self.model.joint_X_p.numpy()[:, 0].astype(float)
        self.radius = self.model.shape_scale.numpy()[:2, 0].astype(float)
        self.half_width = float(self.model.shape_scale.numpy()[self.box, 0])
        self.solver = SolverMonolithic(
            self.model,
            precision=precision,
            joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
            contact_stiffness=1e8,
            friction_coefficient=0,
            friction_velocity_scale=0.001,
            normal_smoothing=1e-4,
            alignment_policy="frozen",
            alignment_smoothing=0.1,
            box_smoothing=25e-6,
            publish_policy="strict",
            pcg_mode="production",
            use_cuda_graph=wp.get_device(device).is_cuda,
            newton_max_iterations=80,
            line_search_max_iterations=24,
        )
        self.state, self.control = self.solver.create_state(dt=DT), self.solver.create_control()
        self.controllers = [
            AxisAdmittance(mass=0.2, damping=20.0, position_limits=(0.0, 0.005), max_speed=0.01) for _ in self.fingers
        ]
        self.feedback, self.last_generation, self.steps = None, -1, 0

    def step(self):
        """Consume one published observation and publish one new physical step."""
        t = (self.steps + 1) * DT
        phase = "low-force" if self.steps < 350 else "high-force" if self.steps < 700 else "release"
        target = 0.1 if phase == "low-force" else 0.2 if phase == "high-force" else 0.0
        before = [(controller.position, controller.velocity) for controller in self.controllers]
        guarded = [False, False]
        if self.feedback is not None:
            if not self.feedback["valid"] or self.feedback["generation"] <= self.last_generation:
                raise ValueError("Admittance requires fresh published contact feedback")
            for i, controller in enumerate(self.controllers):
                guarded[i] = bool(self.feedback["depth_m"][i] >= 0.001)
                bounds = (-0.01, -0.01) if phase == "release" else (-0.01, -0.005) if guarded[i] else None
                controller.advance(target - self.feedback["load_N"][i], DT, velocity_bounds=bounds)
        self.control.joint_target_q.assign(np.array([c.position for c in self.controllers]))
        self.control.joint_target_qd.assign(np.array([c.velocity for c in self.controllers]))
        try:
            self.solver.step(self.state, self.state, self.control, dt=DT)
            stats = self.solver.last_stats
            if not stats.converged or stats.rolled_back:
                raise RuntimeError(f"Step {self.steps + 1}: {stats.failure_reason}")
        except Exception:
            for controller, command in zip(self.controllers, before, strict=True):
                controller.reset(*command)
            raise
        self.last_generation = -1 if self.feedback is None else self.feedback["generation"]
        forces = self.solver.get_contact_forces(self.state)
        if not forces.force_valid:
            raise RuntimeError("Solver did not publish valid contact forces")
        load = -np.einsum("ij,ij->i", forces.body_wrench[self.fingers, :3], self.axes)
        q = self.state.joint_q.numpy().astype(float)
        centers = self.origins + self.axes[:, 0] * q
        gap = np.abs(centers) - self.half_width - self.radius
        self.feedback = {
            "generation": forces.force_generation,
            "valid": True,
            "load_N": load,
            "depth_m": np.maximum(0, -gap),
        }
        self.steps += 1
        return {
            "step": self.steps,
            "time_s": t,
            "phase": phase,
            "target_force_N": target,
            "force_N": load.tolist(),
            "gap_m": gap.tolist(),
            "joint_velocity_m_s": self.state.joint_qd.numpy().tolist(),
            "target_position_m": [c.position for c in self.controllers],
            "target_velocity_m_s": [c.velocity for c in self.controllers],
            "guard_active": guarded,
            "force_generation": forces.force_generation,
            "converged": stats.converged,
            "newton_iterations": stats.nonlinear_iterations,
        }


def evaluate(rows):
    """Apply predeclared force, geometry, release and completion gates."""
    metrics = {}
    for phase, start, end in (("low-force", 0.55, 0.7), ("high-force", 1.25, 1.4)):
        window = [row for row in rows if round(start / DT) < row["step"] <= round(end / DT)]
        metrics[phase] = {
            "samples": len(window),
            "maximum_force_error_N": max(
                (max(abs(f - r["target_force_N"]) for f in r["force_N"]) for r in window), default=None
            ),
            "maximum_speed_m_s": max((max(abs(v) for v in r["joint_velocity_m_s"]) for r in window), default=None),
        }
    metrics["maximum_penetration_m"] = max((max(0, -min(r["gap_m"])) for r in rows), default=None)
    release = [r for r in rows if r["step"] > 925]
    metrics["release_force_N"] = max((max(abs(f) for f in r["force_N"]) for r in release), default=None)
    metrics["release_minimum_gap_m"] = min((min(r["gap_m"]) for r in release), default=None)
    gates = {"complete": len(rows) == STEPS and all(r["converged"] for r in rows)}
    for phase in ("low-force", "high-force"):
        values = metrics[phase]
        gates[phase] = (
            values["samples"] == 75
            and values["maximum_force_error_N"] <= 0.005
            and values["maximum_speed_m_s"] <= 0.001
        )
    gates["penetration"] = metrics["maximum_penetration_m"] is not None and metrics["maximum_penetration_m"] <= 0.001
    gates["release"] = (
        len(release) == 75 and metrics["release_force_N"] <= 0.005 and metrics["release_minimum_gap_m"] >= 0.0005
    )
    return metrics, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("admittance_control.py")]
    paths += sorted((REPO / "newton/_src/solvers/monolithic").glob("*.py"))
    hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    rows, error = [], None
    try:
        scene = AdmittanceGripper(device=args.device, precision=args.precision)
        with (args.output / "trace.jsonl").open("w") as stream:
            for _ in range(STEPS):
                row = scene.step()
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    metrics, gates = evaluate(rows)
    unchanged = all(hashlib.sha256((REPO / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    summary = {
        "passed": all(gates.values()) and error is None and unchanged,
        "error": error,
        "device": args.device,
        "precision": args.precision,
        "dt_s": DT,
        "steps": len(rows),
        "metrics": metrics,
        "gates": gates,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "newton_import": str(Path(newton.__file__).resolve()),
        "source_sha256": hashes,
        "source_unchanged": unchanged,
        "scope": "Two prismatic fingers against a fixed block; not a free-object or full-hand grasp qualification",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("passed", "error", "precision", "metrics", "gates")}, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
