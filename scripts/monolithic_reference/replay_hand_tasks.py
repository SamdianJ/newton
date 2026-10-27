# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay full-hand task states without advancing or correcting the physics."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton.viewer import ViewerGL

from .run_hand_tasks import build_model


def render(case):
    from PIL import Image, ImageDraw

    manifest = json.loads((case.parent / "manifest.json").read_text())
    summary = json.loads((case / "summary.json").read_text())
    scene = json.loads((case / "scene.json").read_text())
    data = np.load(case / "trajectory.npz", allow_pickle=False)
    with wp.ScopedDevice("cpu"):
        model, _, _, _, _ = build_model(
            summary["task"],
            Path(manifest["assets"]),
            device="cpu",
            offset=manifest["offset"],
            surface_source=manifest.get("surface_source", "qualified"),
            height=manifest.get("height", 0.0),
        )
        colors = np.tile((0.72, 0.75, 0.80), (model.shape_count, 1))
        if scene["object_body"] is not None:
            colors[model.shape_body.numpy() == scene["object_body"]] = (0.05, 0.65, 0.85)
        model.shape_color.assign(colors)
        state = model.state()
        viewer = ViewerGL(width=800, height=600, headless=True, enable_cuda_interop=ViewerGL.CudaInterop.NONE)
        viewer.set_model(model)
        viewer.show_ground = False
        viewer.show_joints = False
        viewer.set_camera(pos=wp.vec3(-0.22, -0.28, 0.20 + manifest.get("height", 0.0)), pitch=-22.0, yaw=50.0)
        viewer.hide_loading_splash()
        images = []
        try:
            for i, t in enumerate(data["time"]):
                state.body_q.assign(data["body_q"][i].astype(np.float32))
                if model.particle_count:
                    state.particle_q.assign(data["particle_q"][i].astype(np.float32))
                viewer.begin_frame(float(t))
                viewer.log_state(state)
                grid = np.linspace(-0.15, 0.20, 8)
                starts = [(a, -0.15, 0) for a in grid] + [(-0.15, a, 0) for a in grid]
                ends = [(a, 0.20, 0) for a in grid] + [(0.20, a, 0) for a in grid]
                viewer.log_lines(
                    "world_plane_z0",
                    wp.array(starts, dtype=wp.vec3),
                    wp.array(ends, dtype=wp.vec3),
                    colors=(0.3, 0.4, 0.5),
                )
                viewer.end_frame()
                frame = Image.fromarray(viewer.get_frame().numpy()).convert("RGB")
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 800, 58), fill=(17, 23, 32))
                draw.text((16, 10), f"{summary['task']} | FP64 | {summary['status']} | t={t:.3f}s", fill="white")
                draw.text(
                    (16, 33),
                    f"Recorded dynamics | full Sharpa / fixed-base carriage | {data['phase'][i]}",
                    fill="white",
                )
                images.append(frame)
        finally:
            viewer.close()
        durations = np.maximum(10, np.r_[np.diff(data["time"]), 0.5] * 1000).astype(int).tolist()
        images[0].save(case / "preview.webp", save_all=True, append_images=images[1:], duration=durations, loop=0)
        selected = np.linspace(0, len(images) - 1, 6).astype(int)
        sheet = Image.new("RGB", (1200, 600), (17, 23, 32))
        for slot, index in enumerate(selected):
            sheet.paste(images[index].resize((400, 300)), ((slot % 3) * 400, (slot // 3) * 300))
        sheet.save(case / "storyboard.jpg", quality=92)
        images[0].save(case / "initial.png")
        print(case / "preview.webp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    render(parser.parse_args().case)
