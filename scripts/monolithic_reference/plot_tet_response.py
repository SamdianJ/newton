# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot physical (unmagnified) material and vibration comparison traces."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    found = False
    for experiment, labels in (
        ("material", ("Kim", "Smith")),
        ("mass", ("lumped", "consistent")),
        ("gravity", ("coarse", "medium", "fine")),
    ):
        if not (args.results / experiment).exists():
            continue
        found = True
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for label, color in zip(labels, ("#328bd9", "#e67e36", "#32ae78")[: len(labels)], strict=True):
            directory = args.results / experiment / label
            records = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
            summary = json.loads((directory / "summary.json").read_text())
            time = np.array([r["time"] for r in records])
            tip = np.array([r["tip_m"] * 1000 for r in records])
            energy = np.array([r["elastic_energy_j"] + r["kinetic_energy_j"] for r in records])
            legend = f"{label}: T={summary['period_s']:.4f} s" if experiment == "mass" else label
            axes[0, 0].plot(time, tip, label=legend, color=color)
            if experiment == "material":
                axes[0, 1].plot(tip, [r["force_n"] for r in records], color=color)
                axes[0, 1].set(xlabel="Physical tip displacement [mm]", ylabel="Applied traction [N]")
            elif experiment == "mass":
                peaks = np.array(summary["positive_peaks"])
                axes[0, 1].plot(peaks[:, 0], peaks[:, 1] * 1000, "o-", color=color)
                axes[0, 1].set(xlabel="Time [s]", ylabel="Positive peak envelope [mm]")
            else:
                manifest = json.loads((directory / "manifest.json").read_text())
                count = manifest["tet_count"]
                axes[0, 1].plot(count, summary["tail_mean_tip_m"] * 1000, "o", color=color)
                axes[0, 1].plot(count, summary["linear_static_tip_m"] * 1000, "x", color=color)
                axes[0, 1].set(
                    xlabel="Tet count (o: nonlinear tail mean; x: linear static)", ylabel="Downward tip deflection [mm]"
                )
                axes[0, 0].axhline(summary["linear_static_tip_m"] * 1000, color=color, linestyle="--", alpha=0.4)
            axes[1, 0].plot(time, energy / energy[0] if experiment == "mass" else energy, color=color)
            volume = np.array([r["volume_ratio"] for r in records])
            axes[1, 1].plot(time, 100 * (volume - 1) if experiment == "gravity" else volume, color=color)
        axes[0, 0].set(
            xlabel="Time [s]",
            ylabel="Downward tip deflection [mm]" if experiment == "gravity" else "Physical tip displacement [mm]",
        )
        axes[0, 0].legend()
        axes[1, 0].set(
            xlabel="Time [s]",
            ylabel="Energy / initial energy" if experiment == "mass" else "Total elastic + kinetic energy [J]",
        )
        axes[1, 1].set(
            xlabel="Time [s]", ylabel="Volume change [%]" if experiment == "gravity" else "Volume / rest volume"
        )
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        fig.suptitle(
            {
                "material": "Nonlinear axial stretch (consistent mass)",
                "mass": "Released vibration (Smith): physical scale, BE numerical decay",
                "gravity": "Self-weight mesh comparison: same Smith material and consistent mass",
            }[experiment]
        )
        fig.tight_layout()
        fig.savefig(args.output / (experiment + ".png"), dpi=150)
        plt.close(fig)
    if not found:
        parser.error("No material or mass trace directory found")


if __name__ == "__main__":
    main()
