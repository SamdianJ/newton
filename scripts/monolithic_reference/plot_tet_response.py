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
    for experiment, labels in (("material", ("Kim", "Smith")), ("mass", ("lumped", "consistent"))):
        if not (args.results / experiment).exists():
            continue
        found = True
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for label, color in zip(labels, ("#328bd9", "#e67e36"), strict=True):
            directory = args.results / experiment / label
            records = [json.loads(line) for line in (directory / "trace.jsonl").read_text().splitlines()]
            summary = json.loads((directory / "summary.json").read_text())
            time = np.array([r["time"] for r in records])
            tip = np.array([r["tip_m"] * 1000 for r in records])
            energy = np.array([r["elastic_energy_j"] + r["kinetic_energy_j"] for r in records])
            legend = label if experiment == "material" else f"{label}: T={summary['period_s']:.4f} s"
            axes[0, 0].plot(time, tip, label=legend, color=color)
            if experiment == "material":
                axes[0, 1].plot(tip, [r["force_n"] for r in records], color=color)
                axes[0, 1].set(xlabel="Physical tip displacement [mm]", ylabel="Applied traction [N]")
            else:
                peaks = np.array(summary["positive_peaks"])
                axes[0, 1].plot(peaks[:, 0], peaks[:, 1] * 1000, "o-", color=color)
                axes[0, 1].set(xlabel="Time [s]", ylabel="Positive peak envelope [mm]")
            axes[1, 0].plot(time, energy / energy[0] if experiment == "mass" else energy, color=color)
            axes[1, 1].plot(time, [r["volume_ratio"] for r in records], color=color)
        axes[0, 0].set(xlabel="Time [s]", ylabel="Physical tip displacement [mm]")
        axes[0, 0].legend()
        axes[1, 0].set(
            xlabel="Time [s]",
            ylabel="Energy / initial energy" if experiment == "mass" else "Total elastic + kinetic energy [J]",
        )
        axes[1, 1].set(xlabel="Time [s]", ylabel="Volume / rest volume")
        for ax in axes.flat:
            ax.grid(alpha=0.25)
        fig.suptitle(
            "Nonlinear axial stretch (consistent mass)"
            if experiment == "material"
            else "Released vibration (Smith): physical scale, BE numerical decay"
        )
        fig.tight_layout()
        fig.savefig(args.output / (experiment + ".png"), dpi=150)
        plt.close(fig)
    if not found:
        parser.error("No material or mass trace directory found")


if __name__ == "__main__":
    main()
