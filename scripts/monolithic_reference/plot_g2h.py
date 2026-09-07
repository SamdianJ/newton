# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot G2H physical traces and an actual simulated cantilever thumbnail."""

import argparse
import json
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_tet_compare import MODES, Case


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415 - optional plotting dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thumbnail", action="store_true")
    args = parser.parse_args()
    colors = ("#408de8", "#ef9140", "#40b982", "#ba72de")
    if args.thumbnail:
        fig = plt.figure(figsize=(3.2, 3.2), dpi=100)
        ax = fig.add_subplot(projection="3d")
        for i, (material, mass) in enumerate(MODES):
            case = Case("cpu", material=material, mass=mass)
            for _ in range(100):
                case.step()
            points = case.state.particle_q.numpy() + np.array([0, i * 0.6, 0])
            ax.add_collection3d(
                Poly3DCollection(
                    points[case.model.tri_indices.numpy()], facecolor=colors[i], edgecolor="#304050", linewidth=0.2
                )
            )
        ax.set(xlim=(0, 1.25), ylim=(0, 2.1), zlim=(-0.05, 0.4))
        ax.set_box_aspect((1.25, 2.1, 0.45))
        ax.view_init(elev=34, azim=-48)
        ax.set_axis_off()
        fig.text(0.5, 0.96, "G2H: material / mass", ha="center", va="top", fontsize=12)
        fig.subplots_adjust(left=0, right=1, top=0.87, bottom=0)
        fig.savefig(args.output, dpi=100)
        plt.close(fig)
        return
    if args.results is None:
        parser.error("--results is required for trace plots")
    args.output.mkdir(parents=True, exist_ok=True)
    for direction, axis in (("axial", 0), ("transverse", 2)):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for i, (material, mass) in enumerate(MODES):
            path = args.results / (direction + "-" + material + "-" + mass) / "trace.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            time = [r["time"] for r in records]
            tip = [r["tip"][axis] * 1000 for r in records]
            label = material.split("_")[0] + "/" + mass
            axes[0, 0].plot(time, tip, color=colors[i], label=label)
            axes[0, 1].plot(tip, [r["load"][axis] for r in records], color=colors[i])
            axes[1, 0].plot(time, [r["elastic_energy"] + r["kinetic_energy"] for r in records], color=colors[i])
            axes[1, 1].plot(time, [r["volume_ratio"] for r in records], color=colors[i])
        axes[0, 0].set(xlabel="Time [s]", ylabel="Tip displacement [mm]")
        axes[0, 0].legend()
        axes[0, 1].set(xlabel="Tip displacement [mm]", ylabel="Applied load [N]")
        axes[1, 0].set(xlabel="Time [s]", ylabel="Elastic + kinetic energy [J]")
        axes[1, 1].set(xlabel="Time [s]", ylabel="Volume / rest volume")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        fig.suptitle("G2H " + direction + ": load / hold / unload / free response")
        fig.tight_layout()
        fig.savefig(args.output / (direction + ".png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
