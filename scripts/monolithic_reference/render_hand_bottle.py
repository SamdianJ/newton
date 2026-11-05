# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render recorded hand/bottle states without rerunning or correcting physics."""

import argparse
import json

import numpy as np

import newton
from scripts.monolithic_reference.graspit_hand import build_hand
from scripts.monolithic_reference.run_graspit_hand import shape_vertices


def render(asset, result, *, video=True):
    """Show actual states with an independently fitted deformation and contact trace."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415 - loads optional matplotlib

    recording = np.load(result / "trajectory.npz")
    rows = [json.loads(line) for line in (result / "steps.jsonl").read_text().splitlines()]
    summary = json.loads((result / "summary.json").read_text())
    builder, _ = build_hand(asset)
    model = builder.finalize(device="cpu")
    shapes = [i for i, flags in enumerate(model.shape_flags.numpy()) if not flags & int(newton.ShapeFlags.VISIBLE)]
    faces = recording["triangles"]
    rest = recording["particle_q"][0]
    # A colored label is visual only and follows the same recorded triangles.
    label = np.abs(rest[faces].mean(1)[:, 1]) < 0.025
    colors = np.tile([0.40, 0.77, 0.80, 0.82], (len(faces), 1))
    colors[label] = [0.77, 0.08, 0.10, 1]
    fig = plt.figure(figsize=(11, 7), facecolor="#f6f7f9")
    ax = fig.add_axes([0, 0.12, 0.74, 0.82], projection="3d")
    ax.set_facecolor("#f6f7f9")
    ax.set(xlim=(-0.255, 0.01), ylim=(-0.14, 0.045), zlim=(-0.09, 0.1))
    ax.set_box_aspect((1.4, 1, 1))
    ax.view_init(elev=18, azim=-110)
    ax.set_axis_off()
    collection = Poly3DCollection(
        np.zeros((len(faces), 3, 3)), facecolors=colors, edgecolors=(0.1, 0.3, 0.3, 0.16), linewidths=0.25
    )
    ax.add_collection3d(collection)
    hands = []
    for _shape in shapes:
        mesh = Poly3DCollection([], facecolors="#d9aa8e", edgecolors=(0.35, 0.2, 0.1, 0.12), linewidths=0.2)
        ax.add_collection3d(mesh)
        hands.append(mesh)
    floor_low = np.minimum([-0.26, -0.14], recording["particle_q"][:, :, [0, 2]].min(axis=(0, 1)) - 0.04)
    floor_high = np.maximum([0, 0.04], recording["particle_q"][:, :, [0, 2]].max(axis=(0, 1)) + 0.04)
    table = np.array(
        [
            [
                [x, y, -0.080125]
                for x, y in (floor_low, (floor_high[0], floor_low[1]), floor_high, (floor_low[0], floor_high[1]))
            ]
        ]
    )
    ax.add_collection3d(Poly3DCollection(table, facecolors="#d7dce2", alpha=0.55))
    fig.text(0.05, 0.95, "Human hand | admittance squeeze | empty open bottle", fontsize=17, weight="bold")
    fig.text(
        0.05, 0.90, f"Complete task gates: {'PASS' if summary['passed'] else 'FAIL — see summary.json'}", fontsize=11
    )
    info = fig.text(0.73, 0.81, "", fontsize=12, va="top", linespacing=1.7, family="monospace")
    fig.text(
        0.05,
        0.04,
        "Recorded collision hulls, fixed palm, four controlled fingers. Elastic proxy; no liquid, gas or plasticity.",
        fontsize=10,
    )
    trace = fig.add_axes([0.74, 0.18, 0.23, 0.25])
    times = np.array([r["time_s"] for r in rows])
    for i in range(4):
        trace.plot(times, [r["load_N"][i] for r in rows], lw=1, label=("index", "middle", "ring", "little")[i])
    trace.set(xlabel="Time [s]", ylabel="Virtual tendon load [N]")
    trace.legend(fontsize=7, ncol=2)
    cursor = trace.axvline(0, color="black", lw=1)

    def update(frame):
        x, poses = recording["particle_q"][frame], recording["body_q"][frame]
        # Widen the view when the released bottle rolls away; never hide its motion.
        low = np.minimum([-0.255, -0.14, -0.09], x[:, [0, 2, 1]].min(axis=0) - 0.015)
        high = np.maximum([0.01, 0.045, 0.1], x[:, [0, 2, 1]].max(axis=0) + 0.015)
        ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
        ax.set_box_aspect(high - low)
        collection.set_verts(x[faces][:, :, [0, 2, 1]])
        for shape, mesh in zip(shapes, hands, strict=True):
            vertices = shape_vertices(model, poses, shape)
            mesh.set_verts(vertices[model.shape_source[shape].indices.reshape(-1, 3)][:, :, [0, 2, 1]])
        if frame:
            r = rows[frame - 1]
            cursor.set_xdata([r["time_s"], r["time_s"]])
            info.set_text(
                f"{summary['precision'].upper()}  {r['time_s']:.2f} s\n{r['phase'].upper()}\n{r['publication']}\n\nWall motion\n{r['deformation_max_m'] * 1000:.2f} mm\n\nSampled overlap\n{r['sampled_penetration_m'] * 1000:.2f} mm"
            )
        else:
            info.set_text("INITIAL STATE")

    peak = 1 + int(np.argmax([r["deformation_max_m"] for r in rows]))
    update(peak)
    fig.savefig(result / "preview.png", dpi=160)
    if video:
        writer = FFMpegWriter(fps=30, metadata={"title": "Recorded human hand bottle admittance"}, bitrate=3000)
        indices = np.unique(np.linspace(0, len(rows), max(2, round(times[-1] * 30))).astype(int))
        with writer.saving(fig, str(result / "demo.mp4"), dpi=100):
            for frame in indices:
                update(frame)
                writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--image-only", action="store_true")
    args = parser.parse_args()
    render(args.asset_dir, args.result, video=not args.image_only)
