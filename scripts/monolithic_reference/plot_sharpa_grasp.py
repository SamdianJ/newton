# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot actual PR-7A traces and the predeclared stiffness calibration."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--aabb-report", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.run / "trace.json").read_text())
    manifest = json.loads((args.run / "manifest.json").read_text())
    t = np.array([r["time"] for r in rows])
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
    anchored = manifest.get("experiment") == "anchored-close"
    fig.suptitle(
        "Sharpa anchored closure: collision stability"
        if anchored
        else "Sharpa / soft ball: exploratory integration (G6/G7 not validated)"
    )
    axes[0, 0].plot(t, [r["ball_com"][2] * 1000 for r in rows], label="Ball COM")
    lift_target = manifest["mapping"].get("monolithic_lift", (0, 0, None))[2]
    axes[0, 0].plot(
        t,
        [
            1000 * (rows[0]["ball_com"][2] + (r["q_target"][lift_target] if lift_target is not None else 0))
            for r in rows
        ],
        ls="--",
        label="Initial ball height" if anchored else "Initial ball height + commanded lift",
    )
    axes[0, 0].set_ylabel("World height [mm]")
    axes[0, 1].plot(
        t,
        [np.linalg.norm(r["palm_force"]) if anchored else -r["support_force"][2] for r in rows],
        label="Palm on ball" if anchored else "Support on ball",
    )
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        axes[0, 1].plot(
            t,
            [np.linalg.norm(r["finger_forces"][finger]) if anchored else -r["finger_forces"][finger][2] for r in rows],
            label=finger,
        )
    axes[0, 1].set_ylabel("Contact force magnitude [N]" if anchored else "Vertical force on ball [N]")
    axes[1, 0].plot(t, [1000 * r["penetration"] for r in rows], label="Penetration")
    axes[1, 0].plot(t, [1000 * r["deformation_rms"] for r in rows], label="Rigid-fit deformation RMS")
    axes[1, 0].set_ylabel("Distance [mm]")
    axes[1, 1].plot(t, [r["min_det_f"] for r in rows], label="min(det F)")
    axes[1, 1].plot(t, [r["residual_ratio"] for r in rows], label="Global convergence ratio")
    axes[2, 0].plot(
        t, [max((c["iterations"] for c in r["pcg_calls"]), default=np.nan) for r in rows], label="Max PCG call"
    )
    axes[2, 0].axhline(64, color="tab:red", ls="--", label="Stage p95 budget: 64")
    axes[2, 0].set_ylabel("Iterations")
    axes[2, 1].plot(t, [r["history"]["history_elastic_energy"] for r in rows], label="Elastic history")
    axes[2, 1].set_ylabel("Energy [J]")
    for ax in axes.flat:
        for boundary in manifest["fixture"]["stage_ends"][:-1]:
            ax.axvline(boundary, color="gray", alpha=0.25, lw=0.7)
        ax.set_xlabel("Physical time [s]")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(args.run / "stages.png", dpi=160)
    plt.close(fig)
    calibration = json.loads(args.calibration.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for refinement in (1, 2, 3):
        points = [r for r in calibration["results"] if r["refinement"] == refinement]
        x = [r["stiffness"] for r in points]
        axes[0].semilogx(x, [1000 * r["maximum_penetration"] for r in points], "o-", label=f"r{refinement}")
        axes[1].loglog(x, [r["force_error"] for r in points], "o-", label=f"r{refinement}")
    axes[0].axhline(2, color="tab:red", ls="--", label="2 mm gate")
    axes[1].axhline(0.05, color="tab:red", ls="--", label="5% gate")
    axes[0].set_ylabel("Maximum penetration [mm]")
    axes[1].set_ylabel("Mean final reaction relative error")
    for ax in axes:
        ax.set_xlabel("Normal stiffness [N/m³]")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(args.run / "stiffness-calibration.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(6, 4, figsize=(15, 15), constrained_layout=True)
    for ax, (name, (coord, _dof, target)) in zip(axes.flat, manifest["mapping"].items(), strict=False):
        ax.plot(t, [r["q_target"][target] for r in rows], ls="--", label="Target")
        ax.plot(t, [r["q"][coord] for r in rows], label="Actual")
        ax.set_title(name, fontsize=9)
        ax.set_ylabel("m" if name == "monolithic_lift" else "rad")
        ax.set_xlabel("s")
        ax.grid(alpha=0.2)
    for ax in list(axes.flat)[len(manifest["mapping"]) :]:
        ax.axis("off")
    axes.flat[0].legend()
    fig.savefig(args.run / "joint-tracking.png", dpi=130)
    plt.close(fig)
    if args.aabb_report:
        report = json.loads(args.aabb_report.read_text())
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for ax, key, title in zip(
            axes, ("collision_seconds_p50_p95", "step_seconds_p50_p95"), ("Collision", "Complete step"), strict=True
        ):
            for mode, label in enumerate(("AABB on", "Full table")):
                x = np.arange(len(report)) + (mode - 0.5) * 0.35
                values = np.array([r[key][mode] for r in report]) * 1000
                ax.bar(x, values[:, 0], width=0.35, label=label)
                ax.errorbar(
                    x,
                    values[:, 0],
                    yerr=[np.zeros(len(x)), values[:, 1] - values[:, 0]],
                    fmt="none",
                    color="black",
                    capsize=3,
                )
            ax.set_xticks(range(len(report)), [f"r{r['refinement']}" for r in report])
            ax.set_ylabel("ms (bar: p50; whisker: p95)")
            ax.set_title(title)
            ax.legend()
        fig.savefig(args.run / "aabb-performance.png", dpi=160)


if __name__ == "__main__":
    main()
