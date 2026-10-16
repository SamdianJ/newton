# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental monolithic anisotropic cantilever with Euler-Bernoulli comparison.

Run each principal bending direction using --axis 1 or --axis 2. Simulation
coordinates are physical; --display-amplification changes only the drawing.
"""

import numpy as np
import warp as wp

import newton.examples
from scripts.monolithic_reference.cable_analytic import cantilever_deflection
from scripts.monolithic_reference.cable_benchmarks import RECIPES, CableBenchmark


class Example:
    kind = "cantilever"

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.case = CableBenchmark(
            self.kind,
            segments=args.segments,
            axis=args.axis,
            ea_multiplier=args.ea_multiplier,
            device=args.device,
            pcg_mode=args.pcg_mode,
            use_cuda_graph=args.use_cuda_graph,
        )
        self.amplification = args.display_amplification
        self.sim_time = 0.0
        self.frame_dt = self.case.dt
        viewer.set_model(self.case.model)
        length = self.case.recipe["length"]
        if self.kind == "cantilever" and args.axis == 1:
            viewer.set_camera(wp.vec3(0, 0.05, 1.3 * length), pitch=-89, yaw=90)
        elif self.kind == "cantilever":
            viewer.set_camera(wp.vec3(0, -1.3 * length, 0.05), pitch=0, yaw=90)
        else:
            viewer.set_camera(wp.vec3(0.14 * length, -0.19 * length, 0.13 * length), pitch=-28, yaw=126)

    @classmethod
    def create_parser(cls):
        """Expose physical resolution and execution mode without changing frozen gates."""
        parser = newton.examples.create_parser()
        r = RECIPES[cls.kind]
        parser.add_argument("--segments", type=int, default=r["segments"][-1])
        parser.add_argument("--axis", type=int, choices=(1, 2), default=1)
        parser.add_argument("--ea-multiplier", type=float, default=1)
        parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="diagnostic")
        parser.add_argument("--use-cuda-graph", action="store_true")
        parser.add_argument("--display-amplification", type=float, default=100 if cls.kind == "cantilever" else 1)
        total = r.get("ramp_steps", 0) + r.get("rotation_steps", 0) + r.get("shortening_steps", 0) + r["hold_steps"]
        parser.set_defaults(device="cpu", num_frames=total)
        return parser

    def step(self):
        self.case.step()
        self.sim_time = self.case.steps * self.case.dt

    def render(self):
        """Overlay production centerline and independent continuum reference."""
        case, r = self.case, self.case.recipe
        x = case.state.particle_q.numpy().copy()
        if self.kind == "cantilever":
            reference = case.rest.copy()
            reference[:, case.axis] = cantilever_deflection(
                case.s, length=r["length"], force=r["force"], bending=r["bending"][case.axis - 1]
            )
        else:
            reference = case.reference["positions"].copy()
            transverse = reference[:, 1] + 1j * reference[:, 2]
            phase = np.angle(np.vdot(transverse, x[:, 1] + 1j * x[:, 2]))
            transverse *= np.exp(1j * phase)
            reference[:, 1], reference[:, 2] = transverse.real, transverse.imag
        self.viewer.begin_frame(self.sim_time)
        for name, positions, color in (("simulation", x, (0.2, 0.7, 1.0)), ("analytic", reference, (1.0, 0.6, 0.1))):
            positions[:, 1:] *= self.amplification
            self.viewer.log_lines(
                name,
                wp.array(positions[:-1], dtype=wp.vec3, device=case.model.device),
                wp.array(positions[1:], dtype=wp.vec3, device=case.model.device),
                color,
            )
        self.viewer.end_frame()

    def test_final(self):
        """Require all single-case physical gates after the full loading and hold."""
        self.case.check_final()


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
