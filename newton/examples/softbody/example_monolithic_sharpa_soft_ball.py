# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Display the exploratory Sharpa Close-Hold-Lift-Release integration."""

from pathlib import Path

import warp as wp

import newton.examples
from newton.examples.softbody.monolithic_sharpa_grasp import FIXTURE, GraspCase


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.case = GraspCase(args)
        self.model, self.state = self.case.model, self.case.state
        self.output = args.output
        self.sim_time = 0.0
        self._headless = args.viewer == "null" or args.headless
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.4, -0.4, 0.3), pitch=-20.0, yaw=130.0)

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--asset-dir", required=True)
        parser.add_argument("--contact-dir", required=True)
        parser.add_argument("--trajectory", required=True)
        parser.add_argument("--calibration", required=True)
        parser.add_argument("--ball", default="scripts/monolithic_reference/fixtures/soft_ball/ball_r2.npz")
        parser.add_argument("--output", type=Path, default=Path("output/monolithic-sharpa-soft-ball"))
        parser.add_argument("--friction-off", action="store_true")
        parser.add_argument("--disable-aabb", action="store_true", help="Run the internal full-table collision oracle")
        parser.set_defaults(num_frames=900, device="cuda:0")
        return parser

    def step(self):
        for _ in range(10):
            self.case.step()
            if self.case.failure:
                self.case.save(self.output)
                raise RuntimeError(f"Grasp integration stopped: {self.case.failure}")
        self.sim_time = self.case.step_count * FIXTURE["dt"]
        if self._headless and self.case.step_count % 500 == 0:
            print(f"Sharpa: {self.sim_time:.3f} / 9.000 s", flush=True)
        if self.case.step_count >= 9000:
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


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
