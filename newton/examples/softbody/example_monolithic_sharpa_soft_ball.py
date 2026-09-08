# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Display a palm-anchored soft ball during Sharpa closure, or exploratory grasp stages."""

from pathlib import Path

import warp as wp

import newton.examples
from newton.examples.softbody.monolithic_sharpa_grasp import GraspCase


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.case = GraspCase(args)
        self.model, self.state = self.case.model, self.case.state
        self.output = args.output
        self.sim_time = 0.0
        self._headless = args.viewer == "null" or args.headless
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.28, -0.28, 0.23), pitch=-20.0, yaw=130.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--asset-dir", required=True)
        parser.add_argument("--contact-dir", required=True)
        parser.add_argument("--trajectory", required=True)
        parser.add_argument("--calibration", required=True)
        parser.add_argument("--experiment", choices=("anchored-close", "grasp"), default="anchored-close")
        parser.add_argument("--ball", default="scripts/monolithic_reference/fixtures/soft_ball/ball_r2.npz")
        parser.add_argument("--output", type=Path, default=Path("output/monolithic-sharpa-soft-ball"))
        parser.add_argument("--friction-off", action="store_true")
        parser.add_argument("--disable-aabb", action="store_true", help="Run the internal full-table collision oracle")
        parser.set_defaults(num_frames=450, device="cuda:0")
        return parser

    def step(self):
        if self.case.step_count >= self.case.total_steps:
            return
        for _ in range(10):
            self.case.step()
            if self.case.failure:
                self.case.save(self.output)
                raise RuntimeError(f"Grasp integration stopped: {self.case.failure}")
        self.sim_time = self.case.step_count * self.case.fixture["dt"]
        if self._headless and self.case.step_count % 500 == 0:
            record = self.case.records[-1]
            print(
                f"Sharpa: {self.sim_time:.3f} / {self.case.fixture['duration']:.3f} s; "
                f"contacts={record['active_contacts']}, penetration={1000 * record['penetration']:.3f} mm, "
                f"min(detF)={record['min_det_f']:.3f}",
                flush=True,
            )
        if self.case.step_count >= self.case.total_steps:
            self.case.save(self.output)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        """Require successful execution of the entire schedule, not G6 grasp success."""
        result = self.case.save(self.output)
        assert result["complete_schedule"], result
        assert result["converged_fraction"] >= 0.99, result
        if self.case.anchored:
            assert result["stability_passed"], result


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
