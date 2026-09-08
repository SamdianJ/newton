# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a recorded physical state without advancing or replacing the simulation."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_sharpa_assets import load_hand_ball
from newton.examples.softbody.sharpa_close import load_fixture, sha256
from newton.viewer import ViewerGL


def main():
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--contact-dir", required=True)
    parser.add_argument("--ball", default="scripts/monolithic_reference/fixtures/soft_ball/ball_r2.npz")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--state", default="state_500.npz")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest["ball_sha256"] != sha256(args.ball):
        raise ValueError("Snapshot ball asset mismatch")
    model, _ = load_hand_ball(
        args.asset_dir,
        args.contact_dir,
        args.ball,
        ball_radius=manifest.get("ball_radius_m", 0.020),
        device="cpu",
        parameters=load_fixture("newton/examples/softbody/sharpa_g1h.json"),
        position=manifest["ball_position_m"],
        _mount=manifest.get("mount"),
    )
    state = model.state()
    with np.load(args.run / args.state, allow_pickle=False) as data:
        state.joint_q.assign(data["q"])
        state.particle_q.assign(data["x"])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    viewer = ViewerGL(width=640, height=640, headless=True, enable_cuda_interop=ViewerGL.CudaInterop.NONE)
    viewer.set_model(model)
    viewer.set_camera(pos=wp.vec3(0.28, -0.28, 0.23), pitch=-20.0, yaw=130.0)
    if hasattr(viewer, "hide_loading_splash"):
        viewer.hide_loading_splash()
    for _ in range(3):
        viewer.begin_frame(0.0)
        viewer.log_state(state)
        viewer.end_frame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    png = args.output.with_suffix(".png")
    Image.fromarray(viewer.get_frame().numpy()).save(png)
    with Image.open(png) as picture:
        picture.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS).save(args.output, quality=92)
    viewer.close()


if __name__ == "__main__":
    main()
