# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Form a clove hitch with two opposite-handed half knots around a seven-wire bundle."""

import numpy as np

from .cable_hitch_fixture import arc_lengths, bezier, sample, smooth

EDGE = 0.0054
WIRE_RADIUS = 0.025 / 6
CORD_RADIUS = 0.0016
WIRE_COUNT = 7
WIRE_NODES = 17
HALF_SPAN = 0.042
PROOF_FORCES_MN = (12, 16, 8)
PROOF_START = 6.12
PROOF_PERIOD = 4.98  # Ramp 0.04 s, hold 4 s, unload 0.80 s, rest 0.14 s.
RELEASE = 20.92
SERVICE_START = 21.06
END = 24.14


def grip_tensions(t):
    """Dress both tails, release the free tail first, then proof-draw the standing end."""
    close = smooth((t - 1.58) / 0.18)
    transfer = smooth((t - 2.12) / 0.50)
    open_grips = 1 - smooth((t - (1.78 + 1.8 + 2.4 - 0.30)) / 0.30)
    loaded = 0.016 * (1 + transfer / 14) * close * open_grips
    free = 0.016 * (1 - transfer + 1.5 / 28 * transfer) * close * open_grips
    free *= 1 - smooth((t - 2.62) / 0.24)
    grips = np.array([[0, loaded, -0.12 * loaded * (1 - transfer)], [0, free, 0.12 * free]])
    for i, magnitude in enumerate(PROOF_FORCES_MN):
        age = t - (PROOF_START + i * PROOF_PERIOD)
        grips[0, 1] += 0.001 * magnitude * smooth(age / 0.04) * (1 - smooth((age - 4.04) / 0.8))
    return grips


def route(tight):
    # Work in mm. Z runs along the bundle, +Y points out of its front face.
    radius, pitch, rise = (13.65, 4.0, 6.7) if tight else (16.7, 5.6, 6.4)
    span, coil, low, high = (8.4, 2.55, 26.0, 35.2) if tight else (13.5, 4.6, 28.2, 44.0)
    angle = 0.83
    turn_extent = pitch + (11.0 if tight else 16.0)
    u = np.linspace(0, 1, 181)

    def on_face(phi, z, r):
        return np.column_stack((r * np.sin(phi), r * np.cos(phi), np.broadcast_to(z, u.shape)))

    def half(upper, strand):
        # Two antiparallel half knots with opposite chirality around the Z axis.
        theta = 3 * np.pi * u
        side = 1 if strand == "a" else -1
        x = -side * coil * np.sin(theta)
        y = (high if upper else low) + (1 if upper else -1) * side * coil * np.cos(theta)
        return np.column_stack((x, y, span * (2 * u - 1)))

    a_upper, a_lower = half(True, "a"), half(False, "a")
    b_upper, b_lower = half(True, "b"), half(False, "b")

    def outside_link(a, b, side):
        # Keep the bight outboard of the adjacent standing leg.
        offset = np.array([0, 0, side * 7])
        return bezier(a, a + offset, b + offset, b)

    enter = on_face(angle * u, -turn_extent + (turn_extent + pitch) * u, radius)
    lap1 = on_face(angle + (2 * np.pi - 2 * angle) * u, pitch, radius)
    bridge = on_face(-angle + 2 * angle * u, pitch * (1 - 2 * u), radius + rise * np.sin(np.pi * u) ** 0.6)
    lap2 = on_face(angle + (2 * np.pi - 2 * angle) * u, -pitch, radius)
    leave = on_face(-angle * (1 - u), -pitch + (turn_extent + pitch) * u, radius)
    tail_offset = 12 if tight else 8
    head = np.array([0, 100.0, -span - tail_offset])
    end = np.array([0, 100.0, span + tail_offset])
    parts = [
        ("tail_a", bezier(head, head + np.array([0, -26, 0]), a_upper[0] + [0, 12, -7], a_upper[0])),
        ("a_upper", a_upper),
        ("a_bight", outside_link(a_upper[-1], a_lower[-1], 1)),
        ("a_lower", a_lower[::-1]),
        ("a_lead", bezier(a_lower[0], a_lower[0] + [0, -2, -7], enter[0] + [0, 0, -6], enter[0])),
        ("clove_in", enter),
        ("lap1", lap1),
        ("bridge", bridge),
        ("lap2", lap2),
        ("clove_out", leave),
        ("b_lead", bezier(leave[-1], leave[-1] + [0, 0, 6], b_lower[-1] + [0, -2, 7], b_lower[-1])),
        ("b_lower", b_lower[::-1]),
        ("b_bight", outside_link(b_lower[0], b_upper[0], -1)),
        ("b_upper", b_upper),
        ("tail_b", bezier(b_upper[-1], b_upper[-1] + [0, 12, 7], end + np.array([0, -26, 0]), end)),
    ]
    pieces = [parts[0][1], *[arr[1:] for _, arr in parts[1:]]]
    curve = np.vstack(pieces) * 0.001
    ends = np.cumsum([len(p) for p in pieces]) - 1
    marks = dict(zip((name for name, _ in parts), zip(np.r_[0, ends[:-1]], ends, strict=True), strict=True))
    return curve, marks


class SpotTie:
    def __init__(self):
        self.loose, self.marks = route(False)
        self.tight, _ = route(True)
        self.lengths = arc_lengths(self.loose)
        self.length = self.lengths[-1]
        self.offsets = EDGE * np.arange(int(np.ceil((self.length + 0.023) / EDGE)), -1, -1)
        self.initial_points = sample(self.loose, self.lengths, -self.offsets)

    def feeding(self, t):
        seating = smooth((t - 1.26) / 0.20)
        shape = self.loose if not seating else (1 - seating) * self.loose + seating * self.tight
        grid = self.lengths if not seating else arc_lengths(shape)
        tip = grid[-1] * smooth(t / 1.25) + 0.5 * (self.length - grid[-1])
        return shape, grid, tip - self.offsets

    def targets(self, t):
        shape, grid, material = self.feeding(t)
        target = sample(shape, grid, material)
        pull = 0.016 * smooth((t - 1.37) / 0.19)
        target[0] += pull * np.array([0, 0.65, -0.76])
        target[-1] += pull * np.array([0, 0.65, 0.76])
        strength = (
            smooth((material + 0.006) / 0.010)
            * smooth((self.length + 0.006 - material) / 0.010)
            * (1 - smooth((t - 1.48) / 0.15))
        )
        strength[[0, -1]] = 1 - smooth((t - 1.62) / 0.16)
        return target, strength

    def material_span(self, t, name):
        _, grid, material = self.feeding(t)
        first, last = self.marks[name]
        return (grid[[first, last]] - material[0]) / EDGE
