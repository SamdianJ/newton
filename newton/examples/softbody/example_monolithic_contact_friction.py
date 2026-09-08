# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare elastic sticking, sliding and release with and without contact friction."""

import json
from pathlib import Path

import numpy as np
import warp as wp

import newton.examples
from newton.examples.softbody.monolithic_contact_friction import FIXTURE, FrictionCase, validate_comparison


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.cases = [FrictionCase(args.device, friction=enabled) for enabled in (True, False)]
        self.sim_time = 0.0
        self.output = Path(args.output) if args.output else None
        self.written = False
        viewer.set_model(self.cases[0].model)
        viewer.set_camera(pos=wp.vec3(0.27, -0.32, 0.25), pitch=-18, yaw=125)
        self.buffers = []
        self.edges = []
        self.plate_vertices = np.array([[x, y, z] for z in (-0.01, 0.01) for y in (-0.04, 0.04) for x in (-0.07, 0.07)])
        plate_faces = np.array(
            [
                [0, 2, 1],
                [1, 2, 3],
                [4, 5, 6],
                [5, 7, 6],
                [0, 1, 4],
                [1, 5, 4],
                [2, 6, 3],
                [3, 6, 7],
                [0, 4, 2],
                [2, 4, 6],
                [1, 3, 5],
                [3, 7, 5],
            ]
        )
        for case in self.cases:
            faces = case.model.tri_indices.numpy()
            edges = np.unique(np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
            self.edges.append(edges)
            self.buffers.append(
                (
                    wp.empty_like(case.state.particle_q),
                    wp.array(case.model.tri_indices.numpy().ravel(), dtype=int, device=case.model.device),
                    wp.empty(8, dtype=wp.vec3, device=case.model.device),
                    wp.array(plate_faces.ravel(), dtype=int, device=case.model.device),
                    wp.empty(len(edges), dtype=wp.vec3, device=case.model.device),
                    wp.empty(len(edges), dtype=wp.vec3, device=case.model.device),
                )
            )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--output", help="Write frozen manifests, summaries and physical traces")
        parser.set_defaults(num_frames=200)
        return parser

    def step(self):
        for _ in range(10):
            if self.sim_time >= FIXTURE["duration"] - 1e-10:
                break
            for case in self.cases:
                case.step()
            self.sim_time = self.cases[0].steps * FIXTURE["dt"]
        if self.output is not None and round(self.sim_time * 1000) in (1500, 2500, 3000):
            for c in self.cases:
                directory = self.output / c.label
                directory.mkdir(parents=True, exist_ok=True)
                np.savez(
                    directory / f"pose-{self.sim_time:.2f}.npz",
                    particle_q=c.state.particle_q.numpy(),
                    joint_q=c.state.joint_q.numpy(),
                )
        if self.sim_time >= FIXTURE["duration"] - 1e-10 and not self.written:
            self.write_results()
            self.written = True

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        for i, (case, buffers) in enumerate(zip(self.cases, self.buffers, strict=True)):
            points, faces, plate, plate_faces, starts, ends = buffers
            offset = np.array([0, i * 0.13, 0])
            positions = case.state.particle_q.numpy() + offset
            points.assign(positions)
            starts.assign(positions[self.edges[i][:, 0]])
            ends.assign(positions[self.edges[i][:, 1]])
            q = case.state.joint_q.numpy()
            plate.assign(self.plate_vertices + case.plate_center + np.array([q[0], 0, q[1]]) + offset)
            self.viewer.log_mesh(case.label, points, faces, color=(0.2, 0.55, 0.95) if i == 0 else (0.95, 0.5, 0.2))
            self.viewer.log_lines(case.label + " mesh", starts, ends, (0.1, 0.15, 0.2))
            self.viewer.log_mesh(case.label + " plate", plate, plate_faces, color=(0.5, 0.55, 0.6), opacity=0.25)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Press / drag / hold / lift")
        ui.text("Blue: on | Orange: off")
        ui.text(f"Time: {self.sim_time:.2f} / 4 s; deformation: 1x")
        for c in self.cases:
            if not c.records:
                continue
            r = c.records[-1]
            ui.text(f"{c.label}: {1000 * r['top_displacement_x']:.2f} mm")
            ui.text(f"Normal {r['normal_force_sum']:.2f} N; tangent {r['tangent_force_sum']:.2f} N")
            ui.text(
                f"Stick {r['stick_count']} | Slide {r['slide_count']} | E {1e3 * r['history_elastic_energy']:.3f} mJ"
            )
            data = np.array([x["top_displacement_x"] * 1000 for x in c.records], dtype=np.float32)
            ui.plot_lines("##" + c.label, data, scale_min=-1, scale_max=12, graph_size=ui.ImVec2(0, 80))

    def write_results(self):
        if self.output is None:
            return
        self.output.mkdir(parents=True, exist_ok=True)
        for c in self.cases:
            directory = self.output / c.label
            directory.mkdir(exist_ok=True)
            for name, data in (("manifest", c.manifest), ("summary", c.summary())):
                (directory / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")
            (directory / "trace.jsonl").write_text("".join(json.dumps(r) + "\n" for r in c.records))
            np.savez(
                directory / "final-state.npz", particle_q=c.state.particle_q.numpy(), joint_q=c.state.joint_q.numpy()
            )
        (self.output / "comparison.json").write_text(json.dumps(validate_comparison(self.cases), indent=2) + "\n")

    def test_final(self):
        """Require completed dynamics and the frozen on/off and release gates."""
        result = validate_comparison(self.cases)
        if not result["passed"]:
            raise AssertionError(result)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
