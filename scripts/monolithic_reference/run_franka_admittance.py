# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the full Franka soft-cube trajectory with per-finger translational admittance."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_franka import CUBE_MATERIALS, DT, REPO, TIMES, FrankaGrasp, sha, targets
from newton.examples.softbody.monolithic_paper_box import PaperBoxMetrics
from newton.examples.softbody.monolithic_recording import (
    grasp_metrics,
    object_weight_and_edge,
    save,
    serial,
    state_arrays,
)
from scripts.monolithic_reference.franka_admittance import FrankaFingerAdmittance
from scripts.monolithic_reference.robot_contact_acceptance import FrozenRobotFactory

STEPS = round(TIMES[-1] / DT)


class FrankaAdmittance(FrankaGrasp):
    """Keep the original scene and arm trajectory while closing fingers from contact feedback."""

    def __init__(
        self,
        cache,
        *,
        device="cuda:0",
        precision="fp64",
        material="reference",
        paper_stiffness_scale=1.0,
        linear_max_iterations=200,
        newton_max_iterations=10,
        box_smoothing=0.0,
    ):
        self.factory = FrozenRobotFactory("franka-cube", dataflow="default", precision=precision)

        super().__init__(
            cache,
            device,
            precision=precision,
            solver_factory=self.factory,
            material=material,
            paper_stiffness_scale=paper_stiffness_scale,
            linear_max_iterations=linear_max_iterations,
            newton_max_iterations=newton_max_iterations,
            box_smoothing=box_smoothing,
        )
        self.control = self.solver.create_control()
        self.admittance = FrankaFingerAdmittance(
            self.model.joint_limit_lower.numpy()[7:], self.model.joint_limit_upper.numpy()[7:]
        )
        self.metadata.update(controller="per-finger SI admittance, 1 N, M=0.2 kg, B=20 N*s/m")
        self.fingers = self.model.joint_child.numpy()[7:]
        self.parents = self.model.joint_parent.numpy()[7:]
        self.joint_frames = self.model.joint_X_p.numpy()[7:]
        self.joint_axes = self.model.joint_axis.numpy()[7:]

    def step(self):
        """Accept converged or STOPPED publications; restore the command on hard failure."""
        if self.failed or self.steps >= STEPS:
            return None
        q, v, self.phase = targets(self.high, self.low, (self.steps + 1) * DT)
        before = self.admittance.snapshot()
        previous_q, previous_v = self.control.joint_target_q.numpy(), self.control.joint_target_qd.numpy()
        try:
            if self.steps:
                q[7:], v[7:] = self.admittance.command(self.phase, self.solver.get_contact_forces(self.state), DT)
            self.control.joint_target_q.assign(q)
            self.control.joint_target_qd.assign(v)
            self.solver.step(self.state, self.state, self.control, dt=DT)
            stats = self.solver.last_stats
            if stats.rolled_back or stats.variational.get("publication_status") not in (
                "CONVERGED",
                "STOPPED_ACCEPTED",
            ):
                raise RuntimeError(f"Step {self.steps + 1}: {stats.status}; reason={stats.failure_reason}")
        except Exception:
            self.admittance.restore(before)
            self.control.joint_target_q.assign(previous_q)
            self.control.joint_target_qd.assign(previous_v)
            self.failed = True
            raise
        self.steps += 1
        return stats

    def finger_projection(self, forces):
        """Independently project world reactions onto current opening axes [N]."""
        poses, projected = self.state.body_q.numpy(), []
        for body, parent, frame, axis in zip(
            self.fingers, self.parents, self.joint_frames, self.joint_axes, strict=True
        ):
            joint = wp.transform_multiply(wp.transform(*poses[parent]), wp.transform(*frame))
            outward = np.array(wp.transform_vector(joint, wp.vec3(axis)), dtype=float)
            projected.append(float(np.dot(forces.body_wrench[body, :3], outward)))
        return np.array(projected)


def evaluate(rows, *, initial_com_z, weight, edge, material="reference"):
    """Keep grasp, force tracking, numerical convergence and release gates distinct."""
    hold = [r for r in rows if r["phase"] == "hold"]
    steady = [r for r in hold if r["step"] >= 3300]
    release = [r for r in rows if r["step"] > STEPS - 200]
    grasp = grasp_metrics(
        [(r["step"], r["cube_com_m"][2], r["ground_force_N"][2]) for r in hold],
        initial_com_z=initial_com_z,
        weight=weight,
        edge=edge,
    )
    metrics = {
        "counts": {
            label: sum(r["publication"] == label for r in rows)
            for label in ("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE")
        },
        "grasp": grasp,
        "maximum_penetration_m": max((r["penetration_m"] for r in rows), default=None),
        "maximum_field_penetration_m": max(
            (r.get("field_penetration_m", r["penetration_m"]) for r in rows), default=None
        ),
        "minimum_det_f": min((r["min_det_f"] for r in rows), default=None) if material != "paper-box" else None,
        "hold_force_error_N": max((max(abs(f - 1.0) for f in r["force_N"]) for r in steady), default=None),
        "hold_speed_m_s": max((max(abs(v) for v in r["finger_velocity_m_s"]) for r in steady), default=None),
        "release_force_N": max((max(abs(f) for f in r["force_N"]) for r in release), default=None),
        "release_opening_m": min((min(r["finger_position_m"]) for r in release), default=None),
        "release_height_error_m": max((abs(r["cube_com_m"][2] - initial_com_z) for r in release), default=None),
        "release_ground_force_N": min((r["ground_force_N"][2] for r in release), default=None),
        "maximum_force_projection_error_N": max((r["force_projection_error_N"] for r in rows), default=None),
    }
    if material == "paper-box":
        metrics.update(
            minimum_area_ratio=min((r["area_ratio_min"] for r in rows), default=None),
            maximum_area_ratio=max((r["area_ratio_max"] for r in rows), default=None),
            maximum_self_intersections=max((r["self_intersections"] for r in rows), default=None),
        )
    gates = {
        "complete_published": [r["step"] for r in rows] == list(range(1, STEPS + 1))
        and all(
            r["publication"] in ("CONVERGED", "STOPPED_ACCEPTED")
            and r["converged"] == (r["publication"] == "CONVERGED")
            for r in rows
        ),
        "grasp": grasp["held"],
        "penetration": bool(rows) and metrics["maximum_penetration_m"] <= 0.002,
        "deformation": bool(rows)
        and (
            metrics["minimum_area_ratio"] >= 0.2 and metrics["maximum_area_ratio"] <= 2.0
            if material == "paper-box"
            else metrics["minimum_det_f"] >= 0.2
        ),
        "force_projection": bool(rows) and metrics["maximum_force_projection_error_N"] <= 1e-4,
        "force_tracking": len(steady) == 200
        and metrics["hold_force_error_N"] <= 0.1
        and metrics["hold_speed_m_s"] <= 0.001,
        "release": len(release) == 200
        and metrics["release_force_N"] <= 0.05
        and metrics["release_opening_m"] >= 0.039
        and metrics["release_height_error_m"] <= 0.005
        and metrics["release_ground_force_N"] >= 0.9 * weight,
    }
    if material == "paper-box":
        gates["no_self_intersections"] = bool(rows) and metrics["maximum_self_intersections"] == 0
    return metrics, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp32", "fp64"), default="fp64")
    parser.add_argument("--material", choices=(*CUBE_MATERIALS, "paper-box"), default="reference")
    parser.add_argument(
        "--paper-stiffness-scale",
        type=float,
        default=1.0,
        help="Multiply paper Young's modulus; keep mass and thickness",
    )
    parser.add_argument("--linear-max-iterations", type=int, default=200)
    parser.add_argument("--newton-max-iterations", type=int, default=10)
    parser.add_argument(
        "--box-smoothing",
        type=float,
        default=0.0,
        help="BOX LogSumExp length [m]; positive selects frozen alignment width 0.1 (contact model change)",
    )
    parser.add_argument("--asset-cache", type=Path, default=REPO / "asset/monolithic/data/inputs/franka/sdf-cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-steps", type=int, default=STEPS, help="Short diagnostic window; cannot qualify as complete"
    )
    args = parser.parse_args()
    if not np.isfinite(args.paper_stiffness_scale) or args.paper_stiffness_scale <= 0:
        parser.error("Paper stiffness scale must be finite and positive")
    if args.material != "paper-box" and args.paper_stiffness_scale != 1.0:
        parser.error("Paper stiffness scale requires --material paper-box")
    if args.linear_max_iterations <= 0 or args.newton_max_iterations <= 0:
        parser.error("Iteration budgets must be positive integers")
    if not np.isfinite(args.box_smoothing) or args.box_smoothing < 0:
        parser.error("BOX smoothing must be finite and nonnegative")
    if not 1 <= args.max_steps <= STEPS or not wp.get_device(args.device).is_cuda:
        parser.error("Require CUDA mesh SDFs and max-steps in [1, 5250]")
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {
        str(p.relative_to(REPO)): sha(p)
        for folder in ("newton", "scripts/monolithic_reference")
        for p in (REPO / folder).rglob("*.py")
    }
    save(args.output / "source-hashes.json", hashes)
    save(
        args.output / "command.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )
    rows, frames, scene, error = [], [], None, None
    hard_failure_steps = []
    initial_com_z, weight, edge = 0.0, 1.0, 0.04
    begin = time.perf_counter()
    try:
        scene = FrankaAdmittance(
            args.asset_cache,
            device=args.device,
            precision=args.precision,
            material=args.material,
            paper_stiffness_scale=args.paper_stiffness_scale,
            linear_max_iterations=args.linear_max_iterations,
            newton_max_iterations=args.newton_max_iterations,
            box_smoothing=args.box_smoothing,
        )
        save(args.output / "configuration.json", scene.metadata)
        shell_metrics = PaperBoxMetrics(scene.model) if args.material == "paper-box" else None
        if shell_metrics is not None:
            np.savez_compressed(
                args.output / "paper-box-mesh.npz",
                positions=scene.model.particle_q.numpy(),
                triangles=scene.model.tri_indices.numpy(),
            )
        weights = scene.model.particle_mass.numpy().astype(float)
        initial_com_z = float(np.average(scene.state.particle_q.numpy(), axis=0, weights=weights)[2])
        weight, edge = object_weight_and_edge(scene.model, scene.state.particle_q.numpy())
        np.savez_compressed(args.output / "initial.npz", **state_arrays(scene.state))
        with (args.output / "trace.jsonl").open("w") as stream:
            for _ in range(args.max_steps):
                before = state_arrays(scene.state)
                try:
                    stats = scene.step()
                except Exception:
                    if scene.solver.last_stats.rolled_back:
                        hard_failure_steps.append(scene.steps + 1)
                    np.savez_compressed(args.output / "failure-input.npz", **before)
                    np.savez_compressed(args.output / "failure-iterate.npz", **state_arrays(scene.solver._accepted))
                    save(args.output / "failure.json", scene.solver.last_stats)
                    raise
                forces = scene.solver.get_contact_forces(scene.state)
                if not forces.force_valid:
                    raise RuntimeError("Missing published contact forces")
                measured = scene.factory.measure(scene.solver, scene.state)
                row = {
                    "step": scene.steps,
                    "time_s": scene.steps * DT,
                    "phase": scene.phase,
                    "force_N": forces.joint_force[7:].tolist(),
                    "force_generation": forces.force_generation,
                    "force_projection_error_N": float(
                        np.max(np.abs(scene.finger_projection(forces) - forces.joint_force[7:]))
                    ),
                    "finger_position_m": scene.state.joint_q.numpy()[7:].tolist(),
                    "finger_velocity_m_s": scene.state.joint_qd.numpy()[7:].tolist(),
                    "target_position_m": scene.control.joint_target_q.numpy()[7:].tolist(),
                    "cube_com_m": np.average(scene.state.particle_q.numpy(), axis=0, weights=weights).tolist(),
                    "ground_force_N": measured["ground_force"].tolist(),
                    "penetration_m": measured["penetration"],
                    "field_penetration_m": measured["field_penetration"],
                    "min_det_f": stats.min_det_f if shell_metrics is None else None,
                    "converged": stats.converged,
                    "publication": stats.variational["publication_status"],
                    "solver_status": stats.status.value,
                    "stop_reason": stats.failure_reason,
                    "convergence_ratio": stats.convergence_ratio,
                    "newton_iterations": stats.nonlinear_iterations,
                }
                if shell_metrics is not None:
                    row.update(shell_metrics.measure(scene.state.particle_q))
                rows.append(row)
                stream.write(json.dumps(serial(row), allow_nan=False) + "\n")
                stream.flush()
                if scene.steps % 10 == 0:
                    frames.append((scene.steps, scene.state.body_q.numpy(), scene.state.particle_q.numpy()))
                if scene.steps % 250 == 0:
                    print(json.dumps(serial(row)), flush=True)
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"
    if scene is not None:
        np.savez_compressed(args.output / "final.npz", **state_arrays(scene.state))
    if frames:
        np.savez_compressed(
            args.output / "frames.npz",
            steps=np.array([f[0] for f in frames]),
            body_q=np.stack([f[1] for f in frames]),
            particle_q=np.stack([f[2] for f in frames]),
        )
    metrics, gates = evaluate(rows, initial_com_z=initial_com_z, weight=weight, edge=edge, material=args.material)
    metrics["counts"]["HARD_FAILURE"] += len(hard_failure_steps)
    unchanged = all(sha(REPO / path) == value for path, value in hashes.items())
    summary = {
        "passed": all(gates.values()) and error is None and unchanged,
        "error": error,
        "steps": len(rows),
        "counts": metrics["counts"],
        "stopped_steps": [r["step"] for r in rows if r["publication"] == "STOPPED_ACCEPTED"],
        "hard_failure_steps": hard_failure_steps,
        "precision": args.precision,
        "material": args.material,
        "paper_stiffness_scale": args.paper_stiffness_scale,
        "device": args.device,
        "metrics": metrics,
        "gates": gates,
        "elapsed_seconds": time.perf_counter() - begin,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "newton_import": str(Path(newton.__file__).resolve()),
        "source_unchanged": unchanged,
        "source_hashes_sha256": sha(args.output / "source-hashes.json"),
        "scope": "Franka finger admittance experiment; not the frozen nightly qualification",
    }
    save(args.output / "summary.json", summary)
    print(json.dumps(serial(summary), indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
