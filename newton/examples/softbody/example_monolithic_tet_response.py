# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Show nonlinear material differences and mass-dependent released vibration."""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples.softbody.monolithic_tet_response import DURATION, ResponseCase


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.experiment = args.experiment
        self.cases = [ResponseCase(args.device, experiment=args.experiment, variant=i) for i in range(2)]
        self.dt = self.cases[0].dt
        self.substeps = round(0.04 / self.dt)
        self.sim_time = 0.0
        self.display_scale = (
            args.display_scale if args.display_scale is not None else (1 if args.experiment == "material" else 5)
        )
        if not np.isfinite(self.display_scale) or self.display_scale <= 0:
            raise ValueError("display-scale must be finite and positive")
        self.output = Path(args.output) if args.output else None
        self.written = False
        self.viewer.set_model(self.cases[0].model)
        self.viewer.set_camera(pos=wp.vec3(2.8, -2.6, 2.3), pitch=-28, yaw=125)
        self.buffers = []
        for i, case in enumerate(self.cases):
            faces = case.model.tri_indices.numpy()
            edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
            rest = case.rest + np.array([0, i * 0.8, 0])
            self.buffers.append(
                (
                    wp.array(case.rest, dtype=wp.vec3, device=args.device),
                    wp.empty_like(case.state.particle_q),
                    wp.array(faces.ravel(), dtype=int, device=args.device),
                    wp.array(edges, dtype=int, device=args.device),
                    wp.empty(len(edges), dtype=wp.vec3, device=args.device),
                    wp.empty(len(edges), dtype=wp.vec3, device=args.device),
                    wp.array(rest[edges[:, 0]], dtype=wp.vec3, device=args.device),
                    wp.array(rest[edges[:, 1]], dtype=wp.vec3, device=args.device),
                )
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--experiment", choices=("material", "mass"), default="material")
        parser.add_argument("--display-scale", type=float, help="Visual deformation multiplier; physics is unchanged")
        parser.add_argument("--output", help="Write physical traces and fixture manifests")
        parser.set_defaults(num_frames=300)
        return parser

    def step(self):
        for _ in range(self.substeps):
            if self.sim_time >= DURATION - 1e-10:
                break
            for case in self.cases:
                case.step()
            self.sim_time = self.cases[0].steps * self.dt
        if self.sim_time >= DURATION - 1e-10 and not self.written:
            self.write_results()
            self.written = True

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for i, (case, buffers) in enumerate(zip(self.cases, self.buffers, strict=True)):
            rest, points, faces, edges, starts, ends, rest_starts, rest_ends = buffers
            wp.launch(
                _display,
                len(case.rest),
                [case.state.particle_q, rest, self.display_scale, wp.vec3(0, i * 0.8, 0), points],
                device=case.model.device,
            )
            wp.launch(_edges, len(edges), [points, edges, starts, ends], device=case.model.device)
            color = (0.2, 0.55, 0.95) if i == 0 else (0.95, 0.5, 0.2)
            self.viewer.log_mesh(case.label, points, faces, color=color)
            self.viewer.log_lines(case.label + " mesh", starts, ends, (0.15, 0.2, 0.25))
            self.viewer.log_lines(case.label + " undeformed", rest_starts, rest_ends, (0.75, 0.75, 0.75))
            self.viewer.log_scalar(case.label + " physical tip [mm]", case.records[-1]["tip_m"] * 1000)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Nonlinear stretch" if self.experiment == "material" else "Released axial vibration")
        ui.text(f"Deformation display: {self.display_scale:g}x (physics: 1x)")
        ui.text("Blue: " + self.cases[0].label + " / Orange: " + self.cases[1].label)
        ui.text(f"Time: {self.sim_time:.2f} / {DURATION:g} s")
        ui.text(f"External load: {self.cases[0].records[-1]['force_n']:.1f} N")
        for case in self.cases:
            ui.text(f"{case.label} tip: {1000 * case.records[-1]['tip_m']:.2f} mm")
            data = np.full(round(DURATION / case.dt) + 1, np.nan, dtype=np.float32)
            data[: len(case.records)] = [r["tip_m"] * 1000 for r in case.records]
            ui.plot_lines(
                case.label + " [mm]",
                data,
                scale_min=0 if self.experiment == "material" else -45,
                scale_max=550 if self.experiment == "material" else 45,
                graph_size=ui.ImVec2(0, 90),
            )
            if self.experiment == "mass":
                period = case.summary()["period_s"]
                ui.text(f"Measured period: {period:.3f} s" if period else "Collecting 3 positive peaks...")

    def write_results(self):
        if self.output is None:
            return
        for case in self.cases:
            directory = self.output / self.experiment / case.label
            directory.mkdir(parents=True, exist_ok=True)
            for name, data in (("manifest", case.manifest), ("summary", case.summary())):
                (directory / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
            (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in case.records))

    def test_final(self):
        summaries = [c.summary() for c in self.cases]
        if not all(s["demo_verified"] for s in summaries):
            raise AssertionError(summaries)
        if self.experiment == "material":
            difference = abs(summaries[0]["peak_tip_m"] - summaries[1]["peak_tip_m"])
            if difference < 0.005:
                raise AssertionError("Material demonstration has less than 5 mm peak separation")
        elif abs(summaries[0]["period_s"] / summaries[1]["period_s"] - 1) < 0.01:
            raise AssertionError("Mass demonstration has less than 1% period separation")


@wp.kernel
def _display(
    source: wp.array[wp.vec3], rest: wp.array[wp.vec3], scale: float, offset: wp.vec3, points: wp.array[wp.vec3]
):
    i = wp.tid()
    points[i] = rest[i] + scale * (source[i] - rest[i]) + offset


@wp.kernel
def _edges(points: wp.array[wp.vec3], indices: wp.array2d[int], starts: wp.array[wp.vec3], ends: wp.array[wp.vec3]):
    i = wp.tid()
    starts[i] = points[indices[i, 0]]
    ends[i] = points[indices[i, 1]]


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
