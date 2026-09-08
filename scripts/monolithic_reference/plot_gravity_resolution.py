# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot measured spatial refinement and headless solver/diagnostic cost."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for directory in args.results:
        protocol = json.loads((directory / "protocol.json").read_text())
        rows = json.loads((directory / "results.json").read_text())["cases"]
        # Cost remains meaningful for a complete dynamic run even if its tail
        # has not passed the stricter near-static gate.
        rows = [r for r in rows if all(run["failure"] is None and run["summary"]["steps"] == 1800 for run in r["runs"])]
        label = protocol["device"]
        n = [r["tet_count"] for r in rows]
        tips = [1000 * np.mean([run["summary"]["tail_mean_tip_m"] for run in r["runs"]]) for r in rows]
        (line,) = axes[0, 0].plot(n, tips, "--", label=f"{label} tail mean")
        for count, tip, row in zip(n, tips, rows, strict=True):
            span = 1000 * max(run["summary"]["tail_tip_range_m"] for run in row["runs"])
            axes[0, 0].errorbar(count, tip, yerr=span, marker="o" if row["verified"] else "x", color=line.get_color())
        if directory == args.results[0]:
            axes[0, 0].plot(n, [1000 * r["linear_static_tip_m"] for r in rows], "x--", label="linear static")
        linear = np.array([r["linear_static_tip_m"] for r in rows])
        if directory == args.results[0]:
            axes[0, 1].plot(n[1:], 100 * np.abs(np.diff(linear)) / linear[1:], "x-", label="linear static")
        axes[0, 1].plot(n[1:], 100 * np.abs(np.diff(tips)) / np.array(tips[1:]), "o--", label=f"{label} tail means")
        for key, ax, suffix in (
            ("solver_ms", axes[0, 2], "solver"),
            ("step_ms", axes[0, 2], "incl. diagnostics"),
            ("linear_iterations", axes[1, 1], "PCG iterations"),
        ):
            means = np.array([[run["performance"][key]["mean"] for run in row["runs"]] for row in rows])
            middle = np.median(means, axis=1)
            ax.errorbar(
                n,
                middle,
                yerr=[middle - means.min(axis=1), means.max(axis=1) - middle],
                marker="o",
                label=f"{label} {suffix}",
            )
        axes[1, 0].plot(
            n,
            [np.median([run["performance"]["solver_real_time_factor"] for run in row["runs"]]) for row in rows],
            "o-",
            label=label,
        )
        cost = [np.median([run["performance"]["solver_ms"]["mean"] for run in row["runs"]]) for row in rows]
        axes[1, 2].plot(cost, 100 * np.abs(linear - linear[-1]) / linear[-1], "o-", label=label)
    labels = [
        ("Tet count", "Tail-mean downward deflection [mm]"),
        ("Tet count", "Change from previous mesh [%]"),
        ("Tet count", "Mean wall time per physical step [ms]"),
        ("Tet count", "Solver simulated seconds / wall second"),
        ("Tet count", "Mean PCG iterations per physical step"),
        ("Mean solver time per step [ms]", "Linear static difference to finest mesh [%]"),
    ]
    for ax, (x, y) in zip(axes.flat, labels, strict=True):
        ax.set(xlabel=x, ylabel=y)
        if x == "Tet count":
            ax.set_xscale("log")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 2].set_yscale("log")
    axes[1, 0].axhline(1, color="gray", linestyle="--", linewidth=1)
    fig.suptitle(
        "Gravity cantilever: fixed material/density/dt; finest tested mesh is not continuum truth\n"
        "x on tail curve: near-static gate failed; tip bars: full tail range; timing bars: repeat mean range"
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    refinements = [r["refinement"] for r in rows[-3:]]
    fig, axes = plt.subplots(2, len(refinements), figsize=(15, 7), squeeze=False)
    for directory in args.results:
        device = json.loads((directory / "protocol.json").read_text())["device"]
        for col, refinement in enumerate(refinements):
            path = directory / f"r{refinement}" / "repeat-1" / "trace.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            for row, start in ((0, 0), (1, 34)):
                selected = [r for r in records if r["time"] >= start]
                axes[row, col].plot([r["time"] for r in selected], [1000 * r["tip_m"] for r in selected], label=device)
                axes[row, col].set(
                    title=f"r{refinement}: {15 * refinement**3} tets", xlabel="Time [s]", ylabel="Downward tip [mm]"
                )
                axes[row, col].grid(alpha=0.25)
                axes[row, col].legend()
    fig.suptitle(
        "Refined meshes: full trajectory and final two seconds (first repetition)\n"
        "Small residuals alone do not establish near-static equilibrium"
    )
    fig.tight_layout()
    fig.savefig(args.output.with_name(args.output.stem + "-dynamics.png"), dpi=150)


if __name__ == "__main__":
    main()
