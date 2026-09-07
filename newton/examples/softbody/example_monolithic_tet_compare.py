# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare Kim/Smith and lumped/consistent cantilever responses (G2H)."""

import json
from pathlib import Path

import warp as wp

import newton.examples
from newton.examples.softbody.monolithic_tet_compare import FIXTURE, MODES, Case


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.cases = [
            Case(
                args.device,
                material=m,
                mass=w,
                refinement=args.refinement,
                dt=args.dt,
                direction=args.direction,
                load_scale=args.load_scale,
            )
            for m, w in MODES
        ]
        self.sim_time = 0.0
        self.sim_dt = args.dt
        self.substeps = round(0.1 / args.dt)
        if self.substeps < 1 or abs(self.substeps * args.dt - 0.1) > 1e-12:
            raise ValueError("dt must divide 100 ms")
        self.output = Path(args.output) if args.output else None
        self.viewer.set_model(self.cases[0].model)
        self.viewer.set_camera(pos=wp.vec3(3, -3.8, 3), pitch=-32, yaw=125)
        self.display = [wp.empty_like(c.state.particle_q) for c in self.cases]
        self.faces = [wp.array(c.model.tri_indices.numpy().ravel(), dtype=int, device=args.device) for c in self.cases]

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--direction", choices=("axial", "transverse"), default="transverse")
        parser.add_argument("--refinement", type=int, choices=(1, 2, 3), default=2)
        parser.add_argument("--dt", type=float, default=FIXTURE["dt"])
        parser.add_argument("--load-scale", type=float, default=1.0)
        parser.add_argument("--output")
        parser.set_defaults(num_frames=40)
        return parser

    def step(self):
        for _ in range(self.substeps):
            if self.sim_time >= FIXTURE["duration"] - 1e-10:
                break
            for case in self.cases:
                case.step()
            self.sim_time = self.cases[0].steps * self.sim_dt
        if self.sim_time >= FIXTURE["duration"] - 1e-10:
            self.write_results()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        colors = ((0.25, 0.55, 0.95), (0.95, 0.55, 0.25), (0.25, 0.8, 0.55), (0.75, 0.45, 0.9))
        for i, case in enumerate(self.cases):
            wp.launch(
                _offset,
                case.model.particle_count,
                [case.state.particle_q, wp.vec3(0, i * 0.6, 0), self.display[i]],
                device=case.model.device,
            )
            self.viewer.log_mesh(f"{MODES[i][0]}-{MODES[i][1]}", self.display[i], self.faces[i], color=colors[i])
        self.viewer.end_frame()

    def write_results(self):
        if self.output is None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        for case in self.cases:
            directory = self.output / (case.manifest["material_model"] + "-" + case.manifest["mass_mode"])
            directory.mkdir(exist_ok=True)
            (directory / "manifest.json").write_text(json.dumps(case.manifest, indent=2) + "\n")
            (directory / "summary.json").write_text(json.dumps(case.summary(), indent=2) + "\n")
            (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in case.records))

    def test_final(self):
        for case in self.cases:
            if not case.summary()["passed"]:
                raise AssertionError(case.summary())


@wp.kernel
def _offset(source: wp.array[wp.vec3], offset: wp.vec3, destination: wp.array[wp.vec3]):
    i = wp.tid()
    destination[i] = source[i] + offset


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
