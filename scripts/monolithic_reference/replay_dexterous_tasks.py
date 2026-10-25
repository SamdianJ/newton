# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render recorded task states; never advance physics during replay."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import warp as wp

from newton.viewer import ViewerGL

from .run_dexterous_tasks import TaskScene


def render(case):
    """Save a labeled animation and storyboard from the actual published states."""
    from PIL import Image, ImageDraw

    summary = json.loads((case / "summary.json").read_text())
    manifest = json.loads((case.parent / "manifest.json").read_text())
    data = np.load(case / "trajectory.npz", allow_pickle=False)
    task = summary["task"]
    with wp.ScopedDevice("cpu"):
        scene = TaskScene(task, device="cpu", precision=manifest["precision"], friction=manifest["friction"])
        model = scene.model
        colors = np.tile((0.32, 0.39, 0.46), (model.shape_count, 1))
        colors[:2] = (0.96, 0.55, 0.32)
        if scene.object_body is not None:
            colors[model.shape_body.numpy() == scene.object_body] = (0.12, 0.62, 0.77)
        model.shape_color.assign(colors)
        state = model.state()
        viewer = ViewerGL(width=800, height=600, headless=True, enable_cuda_interop=ViewerGL.CudaInterop.NONE)
        viewer.set_model(model)
        viewer.show_ground = False
        viewer.show_joints = False
        viewer.set_camera(pos=wp.vec3(0.18, -0.20, 0.16), pitch=-22.0, yaw=130.0)
        if task == "cable-pull":
            viewer.set_camera(pos=wp.vec3(0.22, -0.24, 0.23), pitch=-15.0, yaw=130.0)
        viewer.hide_loading_splash()
        images = []
        try:
            for i, t in enumerate(data["time"]):
                state.body_q.assign(data["body_q"][i].astype(np.float32))
                if model.particle_count:
                    state.particle_q.assign(data["particle_q"][i].astype(np.float32))
                viewer.begin_frame(float(t))
                viewer.log_state(state)
                if task == "cable-pull":
                    points = data["particle_q"][i]
                    viewer.log_lines(
                        "cable",
                        wp.array(points[:-1], dtype=wp.vec3),
                        wp.array(points[1:], dtype=wp.vec3),
                        colors=(0.1, 0.65, 0.8),
                    )
                    viewer.log_points(
                        "cable_nodes",
                        wp.array(points, dtype=wp.vec3),
                        radii=0.003,
                        colors=wp.full(len(points), wp.vec3(0.1, 0.65, 0.8), dtype=wp.vec3),
                    )
                if task in ("pinch-transfer", "pinch-place"):
                    viewer.log_lines(
                        "placement",
                        wp.array(((0.05, 0, 0.0001), (0.07, -0.02, 0.0001)), dtype=wp.vec3),
                        wp.array(((0.09, 0, 0.0001), (0.07, 0.02, 0.0001)), dtype=wp.vec3),
                        colors=(0.2, 0.8, 0.3),
                    )
                viewer.end_frame()
                frame = Image.fromarray(viewer.get_frame().numpy()).convert("RGB")
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 800, 55), fill=(17, 23, 32))
                draw.text(
                    (18, 10),
                    f"{task} | {manifest['precision'].upper()} | {summary['status']} | t={t:.3f}s",
                    fill="white",
                )
                draw.text(
                    (18, 31),
                    "Dynamic fingertip proxies / scripted controls / simulated object response",
                    fill=(180, 199, 215),
                )
                draw.rectangle((0, 568, 800, 600), fill=(17, 23, 32))
                draw.text((18, 578), f"Phase: {data['phase'][i]} | orange: finger pads | blue: object", fill="white")
                images.append(frame)
        finally:
            viewer.close()
        durations = np.maximum(10, np.r_[np.diff(data["time"]), 0.5] * 1000).astype(int).tolist()
        images[0].save(case / "preview.webp", save_all=True, append_images=images[1:], duration=durations, loop=0)
        images[0].save(case / "preview.gif", save_all=True, append_images=images[1:], duration=durations, loop=0)
        selected = np.linspace(0, len(images) - 1, 6).astype(int)
        sheet = Image.new("RGB", (1200, 600), (17, 23, 32))
        for slot, index in enumerate(selected):
            sheet.paste(images[index].resize((400, 300)), ((slot % 3) * 400, (slot // 3) * 300))
        sheet.save(case / "storyboard.jpg", quality=92)
        print(case / "preview.gif", flush=True)


def main():
    """Render one recorded case or all available cases in a run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    cases = (
        [args.path]
        if (args.path / "trajectory.npz").exists()
        else sorted(p.parent for p in args.path.glob("*/trajectory.npz"))
    )
    if not cases:
        parser.error("No recorded trajectories found")
    if (args.path / "trajectory.npz").exists():
        render(args.path)
    else:
        # OpenGL caches cannot be reused after closing the first viewer context.
        for case in cases:
            subprocess.run(
                [sys.executable, "-m", "scripts.monolithic_reference.replay_dexterous_tasks", str(case)], check=True
            )


if __name__ == "__main__":
    main()
