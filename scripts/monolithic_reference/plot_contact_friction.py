# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Plot physical friction demonstration traces from a completed example run."""

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    columns = [
        ("top_displacement_x", 1000, "Top displacement [mm]"),
        ("tangent_force_sum", 1, "Tangential force magnitude sum [N]"),
        ("normal_force_sum", 1, "Normal force magnitude sum [N]"),
        ("relative_slip_max", 1000, "Maximum relative speed [mm/s]"),
        ("history_elastic_energy", 1000, "Elastic friction energy [mJ]"),
        ("slide_count", 1, "Sliding samples"),
    ]
    for name, color in (("friction-on", "tab:blue"), ("friction-off", "tab:orange")):
        rows = [json.loads(line) for line in (args.input / name / "trace.jsonl").read_text().splitlines()]
        for ax, (key, scale, label) in zip(axes.flat, columns, strict=True):
            ax.plot([r["time"] for r in rows], [scale * r[key] for r in rows], label=name, color=color)
            ax.set_ylabel(label)
            ax.grid(alpha=0.2)
            for t in (0.75, 1.5, 2.5, 3.0, 3.75):
                ax.axvline(t, color="gray", alpha=0.2, linewidth=0.6)
    for ax in axes[-1]:
        ax.set_xlabel("Time [s]")
    axes[0, 0].legend()
    fig.suptitle("Press → small drag → large drag → hold → lift | Smith + consistent mass")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)


if __name__ == "__main__":
    main()
