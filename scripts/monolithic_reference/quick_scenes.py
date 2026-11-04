# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Robot scene windows for quick regression.

Each window builds its scene exactly like the nightly regression (the frozen robot
recipe with the spatial dataflow), imports a snapshot and then publishes one step
at a time with the recorded control program. Rows use the nightly contact recorder.
"""

from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import object_weight_and_edge
from scripts.monolithic_reference.retained_assets import INPUTS
from scripts.monolithic_reference.robot_contact_acceptance import FrozenRobotFactory
from scripts.monolithic_reference.state_snapshots import check_snapshot_scene, import_state


class _Window:
    def start(self, values, model_values, step):
        """Check the scene, import the saved state and resume the control program after ``step``."""
        self.scene_checks = check_snapshot_scene(self.solver, model_values)
        self.import_report = import_state(self.solver, self.state, values)
        self.steps = step

    def advance(self):
        """Publish one step with the recorded control and return its row."""
        self._control(self.steps + 1)
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        stats = self.solver.last_stats
        row = {
            "step": self.steps + 1,
            "publication": stats.variational["publication_status"],
            "penetration": None,
            "min_det_f": float(stats.min_det_f),
            "ground_force_z": None,
            "newton": int(stats.nonlinear_iterations),
        }
        if not stats.rolled_back:
            fields = self.factory.measure(self.solver, self.state)
            row.update(penetration=float(fields["penetration"]), ground_force_z=float(fields["ground_force"][2]))
            self.steps += 1
        return row


class SharpaWindow(_Window):
    """Recorded Sharpa close-and-lift scene."""

    def __init__(self, precision, device):
        from newton.examples.softbody.monolithic_sharpa_recorded import setup, targets  # noqa: PLC0415

        self.factory = FrozenRobotFactory("sharpa-recorded-10mm", dataflow="spatial", precision=precision)
        args = SimpleNamespace(
            trajectory=INPUTS / "sharpa/recorded-control.json",
            precision=precision,
            solver_factory=self.factory,
            contact_recorder=self.factory.measure,
        )
        with wp.ScopedDevice(device):
            scene = setup(args)
        self.solver, self.state, self.control = scene.solver, scene.state, scene.control
        self.dt = scene.trajectory.dt
        self.weight, _ = object_weight_and_edge(scene.model, scene.model.particle_q.numpy())

        def control(step):
            q, v, _, _ = targets(scene.trajectory, scene.curve, scene.mapping, step * self.dt)
            self.control.joint_target_q.assign(q)
            self.control.joint_target_qd.assign(v)

        self._control = control


class FrankaWindow(_Window):
    """Franka grasp scene with its endpoint controller."""

    def __init__(self, precision, device):
        from newton.examples.softbody.monolithic_franka import DT, FrankaGrasp, targets  # noqa: PLC0415

        self.factory = FrozenRobotFactory("franka-cube", dataflow="spatial", precision=precision)
        case = FrankaGrasp(INPUTS / "franka/sdf-cache", device=device, precision=precision, solver_factory=self.factory)
        self.solver, self.state, self.control, self.dt = case.solver, case.state, case.control, DT
        self.weight, _ = object_weight_and_edge(case.model, case.model.particle_q.numpy())

        def control(step):
            q, v, _ = targets(case.high, case.low, step * DT)
            self.control.joint_target_q.assign(np.asarray(q, dtype=np.float32))
            self.control.joint_target_qd.assign(np.asarray(v, dtype=np.float32))

        self._control = control


WINDOWS = {"sharpa": SharpaWindow, "franka": FrankaWindow}
