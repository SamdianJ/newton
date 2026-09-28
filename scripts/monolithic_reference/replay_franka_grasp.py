# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay verified Franka grasp frames without running the physics solver."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_franka import REPO, sha
from newton.examples.softbody.monolithic_soft_cube import load_cube


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--step", type=int, default=3250)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.case / "configuration.json").read_text())
    index = json.loads((args.case / "trajectory-index.json").read_text())
    data = {k: [] for k in ("body_q", "particle_q", "frame_steps", "frame_publication")}
    for chunk in index["chunks"]:
        path = args.case / chunk["path"]
        if sha(path) != chunk["sha256"]:
            raise ValueError("Recorded trajectory hash mismatch")
        with np.load(path) as arrays:
            for name, values in data.items():
                values.append(arrays[name])
    data = {k: np.concatenate(v) for k, v in data.items()}
    steps = data["frame_steps"]
    if not np.array_equal(steps, np.arange(len(steps))):
        raise ValueError("Recorded trajectory has missing or duplicate frames")
    print(json.dumps({"frames": len(steps), "last_step": int(steps[-1])}), flush=True)
    if args.check_only:
        return
    urdf = Path(config["urdf"])
    if sha(urdf) != config["urdf_sha256"]:
        raise ValueError("Franka URDF changed")
    from newton.viewer import ViewerGL  # noqa: PLC0415 - Keep checks independent of OpenGL.

    with wp.ScopedDevice("cpu"):
        builder = newton.ModelBuilder()
        builder.add_urdf(str(urdf), floating=False, collapse_fixed_joints=True, enable_self_collisions=False)
        builder.add_soft_mesh(
            pos=config["cube_position_m"],
            rot=wp.quat_identity(),
            scale=1.0,
            vel=(0, 0, 0),
            mesh=load_cube(REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz"),
            particle_radius=0.0002,
            add_surface_mesh_edges=False,
        )
        builder.add_ground_plane()
        builder.shape_flags = [
            int(f) & ~int(newton.ShapeFlags.COLLIDE_PARTICLES | newton.ShapeFlags.COLLIDE_SHAPES)
            for f in builder.shape_flags
        ]
        model = builder.finalize(device="cpu")
        state = model.state()
        viewer = ViewerGL(
            width=args.size,
            height=args.size,
            headless=bool(args.snapshot),
            enable_cuda_interop=ViewerGL.CudaInterop.NONE,
        )
        viewer.set_model(model)
        viewer.set_camera(pos=wp.vec3(1.35, -1.2, 0.8), pitch=-22.0, yaw=135.0)
        viewer.hide_loading_splash()
        cursor = {"step": min(args.step, len(steps) - 1) if args.snapshot else 0, "playing": False, "clock": 0.0}

        def controls(ui):
            ui.text(f"Franka grasp: {cursor['step'] * 0.001:.3f} s")
            ui.text(str(data["frame_publication"][cursor["step"]]))
            if ui.button("Pause" if cursor["playing"] else "Play"):
                if cursor["step"] == steps[-1]:
                    cursor["clock"] = 0.0
                cursor["playing"] = not cursor["playing"]
            changed, value = ui.slider_int("Step", cursor["step"], 0, int(steps[-1]))
            if changed:
                cursor["step"] = value
                cursor["clock"] = value * 0.001

        viewer.register_ui_callback(controls)

        def draw(step):
            state.body_q.assign(data["body_q"][step])
            state.particle_q.assign(data["particle_q"][step].astype(np.float32))
            viewer.begin_frame(step * 0.001)
            viewer.log_state(state)
            viewer.end_frame()

        try:
            if args.snapshot:
                for _ in range(3):
                    draw(cursor["step"])
                from PIL import Image

                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(viewer.get_frame().numpy()).save(args.snapshot)
            else:
                previous = time.perf_counter()
                while viewer.is_running():
                    now = time.perf_counter()
                    if cursor["playing"]:
                        cursor["clock"] = min(float(steps[-1]) * 0.001, cursor["clock"] + now - previous)
                        cursor["step"] = min(int(steps[-1]), int(cursor["clock"] / 0.001))
                        if cursor["step"] == steps[-1]:
                            cursor["playing"] = False
                    previous = now
                    draw(cursor["step"])
                    time.sleep(0.005)
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
