# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Load and unload a contact-free flat or naturally folded monolithic shell."""

import tempfile
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from scripts.monolithic_reference.cloth_benchmarks import RECIPES, ClothBenchmark


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.case = ClothBenchmark(
            args.kind, device=args.device, pcg_mode=args.pcg_mode, use_cuda_graph=args.use_cuda_graph
        )
        self.frame_dt = self.case.dt
        self.sim_time = 0.0
        self.points = wp.array(self.case.rest, dtype=wp.vec3, device=self.case.model.device)
        self.indices = wp.array(self.case.triangles.ravel(), dtype=int, device=self.case.model.device)
        self.edges = np.unique(
            np.sort(
                np.concatenate(
                    [self.case.triangles[:, [0, 1]], self.case.triangles[:, [1, 2]], self.case.triangles[:, [2, 0]]]
                ),
                axis=1,
            ),
            axis=0,
        )
        viewer.set_model(self.case.model)
        viewer.set_camera(wp.vec3(0.29, -0.35, 0.25), pitch=-30, yaw=130)

    @classmethod
    def create_parser(cls):
        """Select the frozen patch or pre-fold loading recipe and linear backend."""
        parser = newton.examples.create_parser()
        parser.add_argument("--kind", choices=("patch", "prefold"), default="prefold")
        parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
        parser.add_argument("--use-cuda-graph", action="store_true")
        r = RECIPES["common"]
        parser.set_defaults(
            device="cpu", num_frames=sum(r[k] for k in ("ramp_steps", "hold_steps", "unload_steps", "recovery_steps"))
        )
        return parser

    def step(self):
        if self.case.steps == self.case.recipe["checkpoint_step"]:
            with tempfile.TemporaryDirectory() as directory:
                self.case.restore_checkpoint(Path(directory) / "cloth.npz")
        else:
            self.case.step()
        self.sim_time = self.case.steps * self.frame_dt

    def render(self):
        """Draw physical shell coordinates and the unchanged natural shape in orange."""
        self.points.assign(self.case.state.particle_q.numpy().astype(np.float32))
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_mesh("shell", self.points, self.indices, color=(0.2, 0.6, 0.9), backface_culling=False)
        self.viewer.log_lines(
            "reference",
            wp.array(self.case.rest[self.edges[:, 0]], dtype=wp.vec3, device=self.case.model.device),
            wp.array(self.case.rest[self.edges[:, 1]], dtype=wp.vec3, device=self.case.model.device),
            (1.0, 0.6, 0.1),
        )
        self.viewer.end_frame()

    def test_final(self):
        """Require complete loading/unloading and fresh-solver checkpoint agreement."""
        self.case.check_final()
        assert self.case.checkpoint_error is not None


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
