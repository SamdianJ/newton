# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot frozen G5 asset geometry and measured contact/SDF resolution evidence."""

import argparse
import json
from pathlib import Path

from newton.examples.softbody.monolithic_soft_ball import load_ball


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--balls", type=Path, default=Path(__file__).with_name("fixtures") / "soft_ball")
    args = parser.parse_args()
    fig = plt.figure(figsize=(13, 8))
    colors = ["#3683ba", "#ed9a32", "#398f71"]
    traces = []
    for i in range(3):
        mesh = load_ball(args.balls / f"ball_r{i + 1}.npz")
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        boundary = mesh.surface_tri_indices.reshape(-1, 3)
        ax.add_collection3d(
            Poly3DCollection(
                mesh.vertices[boundary] * 1000, facecolor=colors[i], edgecolor="white", linewidth=0.3, alpha=0.85
            )
        )
        ax.set(
            xlim=(-22, 22),
            ylim=(-22, 22),
            zlim=(-22, 22),
            xlabel="x [mm]",
            ylabel="y [mm]",
            zlabel="z [mm]",
            title=f"{len(mesh.vertices)} nodes / {len(mesh.tet_indices) // 4} tets / {len(boundary)} faces",
        )
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        traces.append(json.loads((args.artifacts / f"postcommit-r{i + 1}" / "trace.json").read_text()))
    for column, (key, label, multiplier) in enumerate(
        (
            ("normal_force_n", "Normal force [N]", 1),
            ("deformation_rms_m", "Deformation RMS [mm]", 1000),
            ("step_seconds", "Step cost [ms]", 1000),
        )
    ):
        ax = fig.add_subplot(2, 3, 4 + column)
        for i, trace in enumerate(traces):
            ax.plot(
                [r["time"] * 1000 for r in trace],
                [r[key] * multiplier for r in trace],
                color=colors[i],
                label=f"r{i + 1}",
            )
        ax.set(xlabel="Time [ms]", ylabel=label)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle("Sharpa + free soft balls (20 mm radius): short contact, not a grasp gate")
    fig.tight_layout()
    fig.savefig(args.artifacts / "ball-resolution.png", dpi=170)
    plt.close(fig)
    data = json.loads((args.artifacts / "hand/manifest.json").read_text())
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, resolution in enumerate((32, 64, 128)):
        values = [row["sdf"][i]["max_error_m"] * 1000 for row in data["meshes"]]
        ax.plot(range(len(values)), values, "o-", label=str(resolution), color=colors[i])
    ax.axhline(0.4, color="#b83838", linestyle="--", label="Frozen error limit 0.4 mm")
    ax.set_xticks(range(len(data["meshes"])), [Path(r["source"]).stem for r in data["meshes"]], rotation=30, ha="right")
    ax.set(ylabel="Maximum probe distance error [mm]", title="SDF resolution: 11 collision meshes / 2560 probes each")
    ax.legend(title="Max resolution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.artifacts / "sdf-resolution.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
