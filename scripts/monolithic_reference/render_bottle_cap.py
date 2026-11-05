# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay the cap's recorded rigid poses; the bottle is explicitly a fixed fixture."""

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.bottle_cap import rotation


def render(result, *, video=True):
    """Show unscrewing, lift, lateral removal and a short drive-free flight."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    data = np.load(result / "trajectory.npz")
    rows = [json.loads(line) for line in (result / "steps.jsonl").read_text().splitlines()]
    summary = json.loads((result / "summary.json").read_text())
    fig = plt.figure(figsize=(12, 7), facecolor="#f4f7fa")
    ax = fig.add_axes([0, 0.1, 0.58, 0.8], projection="3d")
    zoom = fig.add_axes([0.56, 0.43, 0.43, 0.47], projection="3d")
    body = data["bottle_vertices"][data["bottle_faces"]][:, :, [0, 2, 1]]
    colors = np.tile([0.38, 0.73, 0.77, 0.85], (len(body), 1))
    colors[np.abs(body[:, :, 2].mean(1)) < 0.025] = [0.76, 0.08, 0.12, 1]
    caps, marks = [], []
    for view in (ax, zoom):
        view.set_facecolor("#f4f7fa")
        view.add_collection3d(
            Poly3DCollection(body, facecolors=colors, edgecolors=(0.1, 0.3, 0.3, 0.12), linewidths=0.2)
        )
        cap = Poly3DCollection([], facecolors="#2372ba", edgecolors=(0.03, 0.12, 0.3, 0.2), linewidths=0.3)
        view.add_collection3d(cap)
        caps.append(cap)
        marks.append(view.plot([], [], [], color="white", lw=3)[0])
        view.view_init(elev=22, azim=-65)
        view.set_axis_off()
    ax.set(xlim=(-0.05, 0.10), ylim=(-0.045, 0.045), zlim=(-0.09, 0.15))
    ax.set_box_aspect((0.15, 0.09, 0.24))
    floor = [
        [[-0.05, -0.045, -0.080125], [0.10, -0.045, -0.080125], [0.10, 0.045, -0.080125], [-0.05, 0.045, -0.080125]]
    ]
    ax.add_collection3d(Poly3DCollection(floor, facecolors="#cbd3df", alpha=0.6))
    zoom.set(xlim=(-0.022, 0.085), ylim=(-0.023, 0.023), zlim=(0.060, 0.14))
    zoom.set_box_aspect((0.107, 0.046, 0.08))
    fig.text(0.04, 0.94, "Removable bottle cap | torque-driven thread surrogate", fontsize=17, weight="bold")
    fig.text(
        0.04,
        0.90,
        "Fixed neck fixture. Recorded FREE rigid body; no prescribed cap poses or hand control.",
        fontsize=11,
    )
    info = fig.text(0.04, 0.78, "", fontsize=11, linespacing=1.6)
    fig.text(0.58, 0.86, "Neck detail / cap removal", fontsize=11)
    fig.text(
        0.04,
        0.04,
        "3 mm/turn, 2-turn disengagement. Simplified compliant thread; no resolved teeth, seal or tamper band.",
        fontsize=10,
    )
    trace = fig.add_axes([0.62, 0.15, 0.33, 0.22])
    times = np.array([r["time_s"] for r in rows])
    trace.plot(times, [1000 * r["rise_m"] for r in rows], label="actual cap rise")
    trace.plot(
        times,
        [3 * r["theta_rad"] / (2 * np.pi) if not r["released"] else np.nan for r in rows],
        "--",
        label="thread lead x turns",
    )
    trace.set(xlabel="Time [s]", ylabel="Rise [mm]")
    trace.legend(fontsize=8)
    cursor = trace.axvline(0, color="#333333", lw=1)
    local_mark = np.array(
        [[0.002, data["cap_vertices"][:, 1].max() + 0.0001, 0], [0.012, data["cap_vertices"][:, 1].max() + 0.0001, 0]]
    )

    def update(frame):
        pose = data["body_q"][frame]
        r = rotation(pose[3:])
        vertices = data["cap_vertices"] @ r.T + pose[:3]
        mark = local_mark @ r.T + pose[:3]
        for mesh, line in zip(caps, marks, strict=True):
            mesh.set_verts(vertices[data["cap_faces"]][:, :, [0, 2, 1]])
            line.set_data_3d(mark[:, 0], mark[:, 2], mark[:, 1])
        if frame:
            row = rows[frame - 1]
            cursor.set_xdata([row["time_s"], row["time_s"]])
            info.set_text(
                f"{summary['precision'].upper()} | checks {'PASS' if summary['passed'] else 'FAIL'}\n"
                f"{row['time_s']:.2f} s | {row['phase'].replace('_', ' ').upper()}\n"
                f"Turns: {row['theta_rad'] / (2 * np.pi):.2f}\n"
                f"Rise: {row['rise_m'] * 1000:.2f} mm\n"
                f"Thread: {'DISENGAGED' if row['released'] else 'ENGAGED'}"
            )
        else:
            info.set_text("INITIAL | cap seated")

    candidates = [i + 1 for i, r in enumerate(rows) if r["phase"] == "move_aside"]
    update(candidates[len(candidates) // 2] if candidates else len(rows))
    fig.savefig(result / "preview.png", dpi=150)
    if video:
        writer = FFMpegWriter(fps=30, bitrate=3000)
        with writer.saving(fig, str(result / "demo.mp4"), dpi=100):
            for frame in np.unique(np.linspace(0, len(rows), max(2, round(times[-1] * 30))).astype(int)):
                update(frame)
                writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--image-only", action="store_true")
    args = parser.parse_args()
    render(args.result, video=not args.image_only)
