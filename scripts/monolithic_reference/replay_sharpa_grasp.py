# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay recorded Close poses without running a physics solver."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parents[1]


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_trajectory(case):
    """Verify chunk hashes and load exact recorded display poses and labels."""
    index = json.loads((case / "trajectory-index.json").read_text())
    fields = ("frame_steps", "frame_times", "frame_publication", "body_q", "particle_q")
    frames = {key: [] for key in fields}
    for chunk in index["chunks"]:
        path = case / chunk["path"]
        if sha(path) != chunk["sha256"]:
            raise ValueError(f"Recording hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            for key in fields:
                frames[key].append(data[key])
    frames = {key: np.concatenate(values) for key, values in frames.items()}
    steps = frames["frame_steps"]
    if not np.array_equal(steps, np.arange(len(steps))):
        raise ValueError("Recording has missing or duplicated physical steps")
    if not np.allclose(frames["frame_times"], steps * 0.001, rtol=0, atol=1e-12):
        raise ValueError("Recording time grid changed")
    if frames["frame_publication"][0] != "INITIAL":
        raise ValueError("Recording does not start at the original initial state")
    if not set(frames["frame_publication"][1:]) <= {"CONVERGED", "STOPPED_ACCEPTED"}:
        raise ValueError("Recording includes an unpublished state")
    return frames


def build_display_model(config):
    """Load frozen visual geometry on CPU, with no SDF or physics solver."""
    from newton.examples.softbody.monolithic_sharpa_assets import load_hand_cube  # noqa: PLC0415
    from newton.examples.softbody.sharpa_close import load_fixture  # noqa: PLC0415

    manifest = config["old_manifest"]
    model, _ = load_hand_cube(
        WORKSPACE / "assets/left_sharpa_wave",
        WORKSPACE / "agents/integration/artifacts/p1/pr6d-assets/hand",
        REPO / "scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz",
        device="cpu",
        parameters=load_fixture(REPO / "newton/examples/softbody/sharpa_g1h.json"),
        position=manifest["object_position_m"],
        resolution=128,
        _mount=manifest["mount"],
    )
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path, help="Recorded close-18-C-EE etc. directory")
    parser.add_argument("--snapshot", type=Path, help="Save one PNG headlessly and exit")
    parser.add_argument("--step", type=int, default=-1, help="Snapshot physical step; -1 means last")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    case = args.case.resolve()
    frames = load_trajectory(case)
    config = json.loads((case / "configuration.json").read_text())
    last = len(frames["frame_steps"]) - 1
    print(
        json.dumps(
            {
                "case": case.name,
                "frames": last + 1,
                "last_step": last,
                "last_time": float(frames["frame_times"][-1]),
                "complete": last == config["expected_steps"],
                "stopped_frames": int(np.sum(frames["frame_publication"] == "STOPPED_ACCEPTED")),
            }
        )
    )
    if args.check_only:
        return

    import warp as wp  # noqa: PLC0415

    from newton.viewer import ViewerGL  # noqa: PLC0415

    with wp.ScopedDevice("cpu"):
        model = build_display_model(config)
        state = model.state()
        viewer = ViewerGL(
            width=1000, height=760, headless=bool(args.snapshot), enable_cuda_interop=ViewerGL.CudaInterop.NONE
        )
        viewer.set_model(model)
        # The infinite support plane occludes the fingers from this close-up view.
        viewer.show_ground = False
        viewer.set_camera(pos=wp.vec3(0.28, -0.28, 0.23), pitch=-20.0, yaw=130.0)
        viewer.hide_loading_splash()
        play = {"step": 0, "playing": False, "speed": 1.0, "clock": 0.0}

        def controls(imgui):
            imgui.text(f"Recorded Sharpa: {case.name}")
            imgui.text(f"Recorded through {last / 1000:.3f} s")
            changed, value = imgui.slider_int("Physical step", play["step"], 0, last)
            if changed:
                play.update(step=value, clock=value * 0.001, playing=False)
            imgui.text(f"t = {play['step'] * 0.001:.3f} s | {frames['frame_publication'][play['step']]}")
            if imgui.button("Pause recording" if play["playing"] else "Play recording"):
                if play["step"] == last:
                    play.update(step=0, clock=0.0)
                play["playing"] = not play["playing"]
            imgui.same_line()
            if imgui.button("Previous"):
                play.update(step=max(0, play["step"] - 1), playing=False)
                play["clock"] = play["step"] * 0.001
            imgui.same_line()
            if imgui.button("Next"):
                play.update(step=min(last, play["step"] + 1), playing=False)
                play["clock"] = play["step"] * 0.001
            _, play["speed"] = imgui.slider_float("Playback speed", play["speed"], 0.05, 2.0)
            imgui.text("Recorded poses only. Stopped frames are not converged.")

        viewer.register_ui_callback(controls, "side")

        def draw(step):
            # Display buffers are FP32; the saved physical states remain FP64.
            state.body_q.assign(frames["body_q"][step].astype(np.float32))
            state.particle_q.assign(frames["particle_q"][step].astype(np.float32))
            viewer.begin_frame(float(frames["frame_times"][step]))
            viewer.log_state(state)
            viewer.end_frame()

        try:
            if args.snapshot:
                selected = last if args.step == -1 else args.step
                if not 0 <= selected <= last:
                    raise ValueError(f"Requested frame is outside [0, {last}]")
                play["step"] = selected
                for _ in range(3):
                    draw(selected)
                from PIL import Image

                args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(viewer.get_frame().numpy()).save(args.snapshot)
            else:
                previous = time.perf_counter()
                while viewer.is_running():
                    now = time.perf_counter()
                    if play["playing"]:
                        play["clock"] = min(last * 0.001, play["clock"] + (now - previous) * play["speed"])
                        play["step"] = min(last, int(play["clock"] / 0.001 + 1e-9))
                        if play["step"] == last:
                            play["playing"] = False
                    previous = now
                    draw(play["step"])
                    time.sleep(0.005)
        finally:
            viewer.close()


if __name__ == "__main__":
    main()
