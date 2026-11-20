# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render a recorded cable-contact run from its saved frames; never reruns physics."""

import argparse
from pathlib import Path

import numpy as np

from .cable_contact_criteria import load_run


def render(directory, *, fps=25, output=None, dpi=100):
    """Write ``render.gif`` (sampled at ``fps`` of simulated time) and ``preview.png`` (last frame)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import PillowWriter

    run = load_run(directory)
    topology, frames = run.topology, run.frames
    dt = float(topology["dt"][0])
    stride = max(1, round(1 / (fps * dt)))
    picks = list(range(0, len(frames["step"]), stride))
    if picks[-1] != len(frames["step"]) - 1:
        picks.append(len(frames["step"]) - 1)
    x = frames["x"]
    lower, upper = x.reshape(-1, 3).min(axis=0), x.reshape(-1, 3).max(axis=0)
    centre, half = (lower + upper) / 2, max(float(np.max(upper - lower)) / 2, 0.02)
    width = max(1.0, 2 * float(np.max(topology["radius"])) / (2 * half) * 600)
    colours = plt.cm.tab10(np.arange(len(topology["vertex_ranges"])) % 10)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="3d")
    scene = run.manifest["scene"]["recipe"]["scene"]
    precision = run.manifest["arguments"]["precision"]

    def draw(k):
        ax.cla()
        for colour, (a, b) in zip(colours, topology["vertex_ranges"], strict=True):
            ax.plot(*x[k, a:b].T, color=colour, linewidth=width, solid_capstyle="round")
        height = float(topology["ground_height"][0])
        if np.isfinite(height):
            grid = np.linspace(-half, half, 2)
            gx, gy = np.meshgrid(centre[0] + grid, centre[1] + grid)
            ax.plot_surface(gx, gy, np.full_like(gx, height), color="0.85", alpha=0.4)
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_zlim(centre[2] - half, centre[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(f"{scene} {precision} t = {frames['step'][k] * dt:.3f} s ({run.manifest['source_sha'][:8]})")

    target = Path(output or directory)
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, str(target / "render.gif"), dpi=dpi):
        for k in picks:
            draw(k)
            writer.grab_frame()
    draw(len(frames["step"]) - 1)
    fig.savefig(target / "preview.png", dpi=2 * dpi)
    plt.close(fig)
    return target / "render.gif"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(render(args.run, fps=args.fps, output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
