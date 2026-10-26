# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental P3 Franka grasp of a 40 mm soft cube; run from the repository checkout.

Uses dynamic arm/fingers with symmetric independent finger PD, C-IE contact,
C2 mesh SDFs and example-local Stopped publication including zero accepted updates.
"""

import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples.softbody.monolithic_franka import DT, REPO, TIMES, FrankaGrasp, sha
from newton.examples.softbody.monolithic_recording import contact_fields as measure_contact_fields
from newton.examples.softbody.monolithic_recording import (
    grasp_metrics,
    object_weight_and_edge,
    save,
    serial,
    state_arrays,
)


class Example:
    def __init__(self, viewer, args):
        if wp.get_device(args.device).is_cpu:
            raise ValueError("This example requires CUDA mesh SDFs")
        self.output = Path(args.output) if args.output else None
        if self.output:
            self.output.mkdir(parents=True, exist_ok=False)
        solver_factory = getattr(args, "solver_factory", None)
        self.contact_recorder = getattr(args, "contact_recorder", None)
        precision = getattr(args, "precision", "fp64")
        if solver_factory is None:
            from scripts.monolithic_reference.robot_contact_acceptance import FrozenRobotFactory  # noqa: PLC0415

            solver_factory = FrozenRobotFactory("franka-cube", dataflow="default", precision=precision)
            self.contact_recorder = solver_factory.measure
        self.case = FrankaGrasp(
            args.asset_cache,
            device=args.device,
            precision=precision,
            solver_factory=solver_factory,
        )
        self.step_limit = min(getattr(args, "max_steps", None) or round(TIMES[-1] / DT), round(TIMES[-1] / DT))
        self.viewer = viewer
        self.viewer.set_model(self.case.model)
        self.viewer.set_camera(pos=wp.vec3(1.35, -1.2, 0.8), pitch=-22.0, yaw=135.0)
        self.display = self.case.model.state()
        self.sim_time = 0.0
        self.rows = []
        self.frames = []
        self.chunks = []
        self.closed = False
        self.weights = self.case.model.particle_mass.numpy().astype(float)
        self.initial_com = np.average(self.case.state.particle_q.numpy(), axis=0, weights=self.weights)
        self.object_weight, self.object_edge = object_weight_and_edge(
            self.case.model, self.case.state.particle_q.numpy()
        )
        if self.output:
            self.log = gzip.open(self.output / "steps.jsonl.gz", "wt")
            save(self.output / "configuration.json", self.case.metadata)
            save(self.output / "command.json", {"argv": sys.argv, "cwd": str(Path.cwd())})
            save(
                self.output / "source-hashes.json",
                {
                    str(p.relative_to(REPO)): sha(p)
                    for folder in ("newton", "scripts/monolithic_reference")
                    for p in (REPO / folder).rglob("*.py")
                },
            )
            np.savez_compressed(self.output / "initial.npz", **state_arrays(self.case.state))
            np.savez_compressed(
                self.output / "model.npz",
                **{name: value.numpy() for name, value in vars(self.case.model).items() if isinstance(value, wp.array)},
            )
            self.record("INITIAL")
        self.begin = time.perf_counter()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--asset-cache",
            type=Path,
            default=REPO / "asset/monolithic/data/inputs/franka/sdf-cache",
        )
        parser.add_argument("--output", type=Path, help="New directory for complete trajectory and diagnostics")
        parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64", help="Solver precision")
        parser.set_defaults(device="cuda:0", num_frames=525)
        return parser

    def record(self, publication):
        self.frames.append((self.case.steps, publication, state_arrays(self.case.state)))
        if len(self.frames) >= 100:
            self.flush()

    def flush(self):
        if not self.frames:
            return
        folder = self.output / "trajectory"
        folder.mkdir(exist_ok=True)
        path = folder / f"frames-{self.frames[0][0]:06d}-{self.frames[-1][0]:06d}.npz"
        arrays = {k: np.stack([r[2][k] for r in self.frames]) for k in self.frames[0][2]}
        arrays.update(
            frame_steps=np.array([r[0] for r in self.frames]),
            frame_times=np.array([r[0] * DT for r in self.frames]),
            frame_publication=np.array([r[1] for r in self.frames], dtype="U16"),
        )
        np.savez_compressed(path, **arrays)
        self.chunks.append({"path": str(path.relative_to(self.output)), "count": len(self.frames), "sha256": sha(path)})
        save(
            self.output / "trajectory-index.json",
            {"chunks": self.chunks, "total_frames": sum(r["count"] for r in self.chunks), "includes_initial": True},
        )
        self.frames.clear()

    def step(self):
        if self.closed:
            return
        for _ in range(10):
            if self.case.steps >= self.step_limit:
                break
            before = state_arrays(self.case.state) if self.output else None
            start = time.perf_counter()
            stats = self.case.step()
            elapsed = time.perf_counter() - start
            if stats is None:
                break
            detail = stats.variational
            row = {
                "step": self.case.steps + int(stats.rolled_back),
                "time": (self.case.steps + int(stats.rolled_back)) * DT,
                "phase": self.case.phase,
                "publication": detail["publication_status"],
                "accepted_count": detail["accepted_count"],
                "zero_accepted": detail["zero_accepted_publication"],
                "solve_seconds": elapsed,
                "stats": stats,
            }
            if not stats.rolled_back:
                state = self.case.state
                recorder = self.contact_recorder or measure_contact_fields
                contact_fields = recorder(self.case.solver, state)
                q = state.joint_q.numpy()
                row.update(
                    cube_com=np.average(state.particle_q.numpy(), axis=0, weights=self.weights),
                    min_cube_z=float(state.particle_q.numpy()[:, 2].min()),
                    **contact_fields,
                    finger_positions=q[7:].copy(),
                    finger_asymmetry=abs(float(q[7] - q[8])),
                    force_valid=int(state._monolithic_coupled.force_valid.numpy()[0]),
                    force_generation=int(state._monolithic_coupled.force_generation.numpy()[0]),
                )
                if self.output:
                    self.record(row["publication"])
            elif self.output:
                np.savez_compressed(self.output / "failure-input.npz", **before)
                np.savez_compressed(self.output / "failure-trial.npz", **state_arrays(self.case.solver._trial))
                np.savez_compressed(self.output / "failure-iterate.npz", **state_arrays(self.case.solver._accepted))
                save(
                    self.output / "failure.json",
                    {
                        "row": row,
                        "rollback_exact": all(
                            np.array_equal(v, state_arrays(self.case.state)[k]) for k, v in before.items()
                        ),
                    },
                )
            if self.output:
                self.log.write(json.dumps(serial(row), allow_nan=False) + "\n")
                self.log.flush()
            self.rows.append(
                {k: v for k, v in row.items() if k != "stats"}
                | {"min_det_f": stats.min_det_f, "newton": stats.nonlinear_iterations}
            )
            self.sim_time = self.case.steps * DT
            if self.case.failed:
                self.close()
                raise RuntimeError(f"Franka step {row['step']} rolled back: {stats.failure_reason}")
            if self.case.steps % 250 == 0:
                print(self.case.steps, self.case.phase, flush=True)
        if self.case.steps >= self.step_limit:
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        successful = [r for r in self.rows if r["publication"] != "HARD_FAILURE"]
        hold = [r for r in successful if r["phase"] == "hold"]
        self.summary = {
            "complete": self.case.steps == round(TIMES[-1] / DT),
            "published_steps": self.case.steps,
            "counts": {
                label: sum(r["publication"] == label for r in self.rows)
                for label in ("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE")
            },
            "zero_accepted_steps": [r["step"] for r in successful if r["zero_accepted"]],
            "final_cube_com": successful[-1]["cube_com"] if successful else self.initial_com,
            "final_ground_force": successful[-1]["ground_force"] if successful else None,
            "maximum_com_rise_m": max((r["cube_com"][2] - self.initial_com[2] for r in successful), default=0.0),
            # The hold phase is the window at the top of the lift.
            "grasp": grasp_metrics(
                [(r["step"], r["cube_com"][2], r["ground_force"][2]) for r in hold],
                initial_com_z=float(self.initial_com[2]),
                weight=self.object_weight,
                edge=self.object_edge,
            ),
            "maximum_penetration_m": max((r["penetration"] for r in successful), default=0.0),
            "minimum_det_f": min((r["min_det_f"] for r in successful), default=1.0),
            "max_finger_asymmetry_m": max((r["finger_asymmetry"] for r in successful), default=0.0),
            "elapsed_seconds": time.perf_counter() - self.begin,
            "solve_seconds": sum(r["solve_seconds"] for r in self.rows),
            "acceptance_scope": "single-scene demonstration; no general precision certification",
        }
        if self.output:
            self.flush()
            self.log.close()
            save(self.output / "summary.json", self.summary)
            np.savez_compressed(self.output / "final.npz", **state_arrays(self.case.state))
        print(json.dumps(serial(self.summary)), flush=True)

    def render(self):
        self.display.body_q.assign(self.case.state.body_q.numpy())
        self.display.particle_q.assign(self.case.state.particle_q.numpy().astype(np.float32))
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.display)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Franka: {self.case.phase} | {self.sim_time:.3f} / {TIMES[-1]:.2f} s")
        ui.text("Independent dynamic fingers; shared PD targets")
        ui.text("Experimental Stopped rule allows zero accepted updates")

    def test_final(self):
        """Require a complete run and a sustained off-ground grasp during the hold."""
        self.close()
        assert self.summary["complete"] and not self.case.failed
        assert self.summary["grasp"]["held"], self.summary


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()
