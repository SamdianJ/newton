# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagnose PR2 Sharpa trajectories without certifying continuous accuracy."""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

from newton.examples.softbody.example_monolithic_sharpa_close import Example
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture, load_hand, sha256


def configure_stopping(solver, scale=1.0):
    """Scale the four nonlinear goals relative to the current production defaults."""
    if isinstance(scale, bool) or scale not in (1.0, 0.1, 0.01):
        raise ValueError("Stopping scale must be 1, 0.1 or 0.01")
    fields = ("merit_absolute_global", "merit_absolute_q", "merit_relative_global", "merit_relative_q")
    solver._config = replace(solver._config, **{name: getattr(solver._config, name) * scale for name in fields})
    return asdict(solver._config)


class SharpaDiagnostic(Example):
    """Reuse the original target sampling and physical metric definitions."""

    def __init__(self, args, device, mode, *, pcg_mode=None, use_cuda_graph=None, stopping_scale=1.0):
        pcg_mode = pcg_mode or ("diagnostic" if device == "cpu" else "production")
        if use_cuda_graph is None:
            use_cuda_graph = device != "cpu" and pcg_mode == "production"
        self.parameters = load_fixture(args.fixture)
        self.model, self.manifest = load_hand(
            args.asset_root / "left_sharpa_wave", device=device, parameters=self.parameters
        )
        trajectory_path = args.asset_root / "trajectory/Ball_catch - Left_hand_motion_rad.csv"
        self.trajectory = ClosureTrajectory(
            trajectory_path,
            self.manifest["joint_names"],
            self.model.joint_limit_lower.numpy(),
            self.model.joint_limit_upper.numpy(),
            self.model.joint_velocity_limit.numpy(),
        )
        configs = [self.parameters["joints"][name] for name in self.trajectory.names]
        self.solver = SolverMonolithic._create_articulation_only(
            self.model,
            globalization=mode,
            joint_terms=SolverMonolithic.JointTerms(
                implicit_pd=True,
                limits=True,
                friction=True,
                limit_width=tuple(c["limit_width"] for c in configs),
                friction_velocity_scale=tuple(c["friction_velocity_scale"] for c in configs),
            ),
            pcg_mode=pcg_mode,
            use_cuda_graph=use_cuda_graph,
        )
        stopping_config = configure_stopping(self.solver, stopping_scale)
        self.sim_dt = self.parameters["dt"] / args.refinement
        self.state = self.solver._create_articulation_state(dt=self.sim_dt)
        q, qd = self.trajectory.sample(0.0)
        self.state.joint_q.assign(q)
        self.state.joint_qd.assign(qd)
        self.solver._reset_articulation_state(self.state, dt=self.sim_dt)
        self.control = self.model.control()
        self.sim_substeps = 1
        self.substeps = 0
        self.sim_time = 0.0
        self.records = []
        self.output = args.output_dir / f"{device.replace(':', '_')}-{mode}-r{args.refinement}"
        self.manifest.update(
            schema="p3-pr2-sharpa-diagnostic/v1",
            scope="No-contact diagnostic; continuous accuracy NOT_CERTIFIED",
            globalization=mode,
            device=device,
            pcg_mode=pcg_mode,
            graph=use_cuda_graph,
            stopping_scale=stopping_scale,
            stopping_config=stopping_config,
            dt=self.sim_dt,
            parameters=self.parameters,
            fixture_sha256=sha256(args.fixture),
            trajectory_sha256=sha256(trajectory_path),
            mapping=self.trajectory.mapping,
            strict_success_required=True,
            warp_version=wp.__version__,
        )

    def step(self):
        target_time = (self.substeps + 1) * self.sim_dt
        q, qd = self.trajectory.sample(target_time)
        self.control.joint_target_q.assign(q.astype(np.float32))
        self.control.joint_target_qd.assign(qd.astype(np.float32))
        start = time.perf_counter()
        self.solver.step(self.state, self.state, self.control, dt=self.sim_dt)
        elapsed = time.perf_counter() - start
        stats = self.solver.last_stats
        self.solver._kinematics.evaluate(self.state.joint_q)
        self.records.append(
            {
                "time": self.sim_time if stats.rolled_back else target_time,
                "target_time": target_time,
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
                "step_ms": 1000 * elapsed,
                "nonlinear_iterations": stats.nonlinear_iterations,
                "linear_iterations": stats.linear_iterations,
                "linear_scheduled_slots": stats.linear_scheduled_slots,
                "goal_global": self.solver._config.merit_absolute_global
                + self.solver._config.merit_relative_global * stats.merit_reference,
                "goal_q": self.solver._config.merit_absolute_q
                + self.solver._config.merit_relative_q * stats.merit_reference,
                "com": self.solver._kinematics.com.numpy().tolist(),
                "rotation": self.solver._kinematics.rotations.numpy().tolist(),
                "trials": stats.variational["trials"],
                "linear_postchecks": stats.variational["linear_postchecks"],
                "failure_reason": stats.failure_reason,
            }
        )
        if not stats.converged or stats.status.value != "success" or stats.rolled_back:
            self.output.mkdir(parents=True, exist_ok=True)
            self.solver._states.save_checkpoint(self.state, self.output / "failed-input.npz")
            np.savez(
                self.output / "failed-control.npz",
                **{
                    name: getattr(self.control, name).numpy()
                    for name in ("joint_target_q", "joint_target_qd", "joint_f")
                },
            )
            raise RuntimeError(f"Strict step failed at {target_time}: {stats.status.value}: {stats.failure_reason}")
        self.substeps += 1
        self.sim_time = self.substeps * self.sim_dt

    def summarize(self):
        result = super().summarize()
        complete = self.substeps == round(self.parameters["duration"] / self.sim_dt)
        success = complete and all(r["status"] == "success" and r["converged"] for r in self.records)
        result.update(
            diagnostic_passed=bool(result.pop("passed") and success),
            complete_schedule=complete,
            all_steps_success=success,
            physical_tracking_passed=bool(complete and all(row["passed"] for row in result.get("joints", []))),
            continuous_accuracy="NOT_CERTIFIED",
            same_discrete_oracle="NOT_RUN_FOR_FULL_HAND",
            pr2_acceptance="INCOMPLETE",
            pcg_mode=self.manifest["pcg_mode"],
            graph=self.manifest["graph"],
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=Path("newton/examples/softbody/sharpa_g1h.json"))
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    parser.add_argument("--modes", nargs="+", default=["Armijo", "ResNorm", "ArmijoOrResNorm"])
    parser.add_argument("--refinement", type=int, choices=(1, 2, 4), default=1)
    parser.add_argument("--stopping-scale", type=float, choices=(1.0, 0.1, 0.01), default=1.0)
    parser.add_argument("--pcg-mode", choices=("diagnostic", "production"))
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = sorted(Path("newton").rglob("*.py")) + sorted(Path("scripts/monolithic_reference").glob("*.py"))
    hashes = {str(path): sha256(path) for path in sources}
    (args.output_dir / "execution-source-hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    results = []
    for device in args.devices:
        for mode in args.modes:
            case = SharpaDiagnostic(
                args,
                device,
                mode,
                pcg_mode=args.pcg_mode,
                use_cuda_graph=args.cuda_graph,
                stopping_scale=args.stopping_scale,
            )
            failure = None
            try:
                for _ in range(round(case.parameters["duration"] / case.sim_dt)):
                    case.step()
                    if case.substeps % 250 == 0:
                        print(device, mode, case.substeps, case.records[-1]["ratio_q"], flush=True)
            except RuntimeError as error:
                failure = str(error)
            case.write_results()
            result = case.summarize()
            result.update(device=device, globalization=mode, failure=failure, output=str(case.output))
            results.append(result)
            (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(device, mode, "complete", result["steps"], result["diagnostic_passed"], failure, flush=True)
    if hashes != {str(path): sha256(path) for path in sources}:
        raise RuntimeError("Execution sources changed during diagnostics; results need revalidation")


if __name__ == "__main__":
    main()
