# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A post loop with three turns around its two legs and a tag fed back through the turns."""

import numpy as np

DT = 0.003
END = 5.0
RADIUS = 0.0016
CORE_RADIUS = 0.0125
HALF_LENGTH = 0.042


EDGE = 0.0054
TURN_COUNT = 3
NEAR = 0.032
FEED_END = 1.25


def smooth(x):
    u = np.clip(x, 0, 1)
    return u * u * (3 - 2 * u)


def bezier(a, b, c, d):
    t = np.linspace(0, 1, 121)[:, None]
    return (
        (1 - t) ** 3 * np.asarray(a)
        + 3 * t * (1 - t) ** 2 * np.asarray(b)
        + 3 * t * t * (1 - t) * np.asarray(c)
        + t**3 * np.asarray(d)
    )


def guide_curve(radius, pitch, separation, center_x, center_z, tag_height, tail_length):
    far = NEAR + TURN_COUNT * pitch
    post = 0.016
    leading = np.linspace([0, 0.089, 0], [0, 0.030, 0], 100)
    enter = bezier(leading[-1], [0, 0.023, 0], [-post, 0.022, 0], [-post, 0, 0])
    angle = np.linspace(np.pi, 2 * np.pi, 121)
    around = np.column_stack((post * np.cos(angle), post * np.sin(angle), np.zeros_like(angle)))
    leave = bezier(around[-1], [post, 0.010, 0], [separation, 0.018, 0], [separation, 0.025, 0])
    returning = np.linspace(leave[-1], [separation, far, 0], 101)
    outer = center_x + radius
    start_turn = bezier(
        returning[-1], [separation, far + 0.012, -0.002], [outer, far + 0.012, -0.005], [outer, far, center_z]
    )
    theta = np.linspace(0, TURN_COUNT * 2 * np.pi, 641)
    turns = np.column_stack(
        (
            center_x + radius * np.cos(theta),
            far - pitch * theta / (2 * np.pi),
            center_z + radius * np.sin(theta),
        )
    )
    tuck = bezier(
        turns[-1],
        [outer, NEAR - 0.011, center_z + 0.004],
        [center_x, NEAR - 0.011, tag_height],
        [center_x, NEAR, tag_height],
    )
    tag = np.linspace(tuck[-1], [center_x, far + tail_length, tag_height], 121)
    return np.vstack([leading, *(part[1:] for part in (enter, around, leave, returning, start_turn, turns, tuck, tag))])


def arc_lengths(curve):
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]


def sample(curve, lengths, material):
    targets = np.column_stack([np.interp(material, lengths, axis) for axis in curve.T])
    targets[:, 1] -= np.minimum(material, 0)
    targets[:, 1] += np.maximum(material - lengths[-1], 0)
    return targets


class LockingHitch:
    def __init__(self):
        self.loose = guide_curve(0.0085, 0.0062, 0.0062, 0.0031, 0.0018, 0.0053, 0.080)
        self.tight = guide_curve(0.0057, 0.0038, 0.00325, 0.001625, 0.00095, 0.00284, 0.080)
        self.lengths = arc_lengths(self.loose)
        self.length = self.lengths[-1]
        count = int(np.ceil((self.length + 0.023) / EDGE)) + 1
        self.offsets = EDGE * np.arange(count - 1, -1, -1)
        self.initial_points = sample(self.loose, self.lengths, -self.offsets)
        # Fixed positions on the loose guide identify each measured winding during threading.
        bounds = [[0.0031 + 0.0085, NEAR + (TURN_COUNT - k) * 0.0062, 0.0018] for k in range(TURN_COUNT + 1)]
        self.turn_lengths = [self.lengths[np.linalg.norm(self.loose - p, axis=1).argmin()] for p in bounds]

    def targets(self, t):
        seating = smooth((t - 1.26) / 0.20)
        if seating:
            curve = (1 - seating) * self.loose + seating * self.tight
            grid = arc_lengths(curve)
        else:
            curve, grid = self.loose, self.lengths
        tip = grid[-1] * smooth(t / FEED_END) + 0.82 * (self.length - grid[-1])
        material = tip - self.offsets
        target = sample(curve, grid, material)
        # Take 82% of the released material from the working end, then draw it another 28 mm.
        pull = 0.028 * smooth((t - 1.37) / 0.19)
        target[0, 1] += 0.22 * pull
        target[-1, 1] += pull
        withdraw = 1 - smooth((t - 1.48) / 0.15)
        strength = smooth((material + 0.006) / 0.010) * smooth((self.length + 0.006 - material) / 0.010) * withdraw
        strength[[0, -1]] = 1 - smooth((t - 1.62) / 0.16)
        return target, strength

    def turn_passages(self, times, positions, radius, core_radius):
        # Arclength on the feed guide labels each turn. Its rim and spanning surface
        # below are built from the measured rod, including the two ends of the seam.
        previous = [None] * TURN_COUNT
        passages = []
        for t, q in zip(times, positions, strict=True):
            if not 0.80 < t <= FEED_END:
                continue
            first = self.length * smooth(t / FEED_END) - self.offsets[0]
            bounds = (np.asarray(self.turn_lengths) - first) / EDGE
            for k in range(TURN_COUNT):
                observation = turn_surface(q, bounds[k], bounds[k + 1])
                old = previous[k]
                previous[k] = None if observation is None else (t, *observation)
                if old is None or observation is None or not old[1] < 0 <= observation[0]:
                    continue
                rim = min(old[2], observation[1])
                seam = max(old[3], observation[2])
                core_clearance = min(old[4], observation[3]) - core_radius
                if rim >= 2 * radius and seam <= 8 * radius and core_clearance >= radius:
                    passages.append(
                        {
                            "turn_from_start_of_wrap": k + 1,
                            "interval_s": [float(old[0]), float(t)],
                            "min_tip_to_rim_mm": 1000 * rim,
                            "max_virtual_closure_mm": 1000 * seam,
                            "min_tip_to_core_surface_mm": 1000 * core_clearance,
                        }
                    )
        return passages


def turn_surface(q, start, end):
    if start < 0 or end > len(q) - 4:
        return None
    indices = np.r_[start, np.arange(int(np.floor(start)) + 1, int(np.ceil(end))), end]
    rim = np.column_stack([np.interp(indices, np.arange(len(q)), axis) for axis in q.T])
    tip = q[-1]
    a, b = rim, np.roll(rim, -1, axis=0)
    center = rim.mean(axis=0)
    u, v = (a - center)[:, [0, 2]], (b - center)[:, [0, 2]]
    d = (tip - center)[[0, 2]]

    def cross(x, y):
        return x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]

    area = cross(u, v)
    denominator = np.where(np.abs(area) > 1e-15, area, np.inf)
    v1, v2 = cross(d, v) / denominator, cross(u, d) / denominator
    inside = np.flatnonzero((np.abs(area) > 1e-15) & (v1 >= 0) & (v2 >= 0) & (v1 + v2 <= 1))
    if len(inside) != 1:
        return None
    j = inside[0]
    height = center[1] + v1[j] * (a[j, 1] - center[1]) + v2[j] * (b[j, 1] - center[1])
    edges = b - a
    along = np.clip(((tip - a) * edges).sum(axis=1) / np.maximum((edges * edges).sum(axis=1), 1e-30), 0, 1)
    distance = np.linalg.norm(tip - a - along[:, None] * edges, axis=1).min()
    return (
        float(tip[1] - height),
        float(distance),
        float(np.linalg.norm(a[-1] - a[0])),
        float(np.linalg.norm(tip[:2])),
    )
