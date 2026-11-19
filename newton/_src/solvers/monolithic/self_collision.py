# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Same-tree (self) contact selection and its initial-overlap check.

``SolverMonolithic(self_collision={"exclude": [[label_a, label_b], ...]})`` turns on contact between
shapes of different bodies of one articulation tree. The pairs are the Model's ``shape_contact_pairs``
whose shapes lie on two bodies of the same tree, both with COLLIDE_SHAPES, minus excluded body pairs.
Each pair becomes an ``A-self`` actor contact; the shared columns on the root path of both bodies'
lowest common ancestor are dropped (an ancestor DoF moves both ends rigidly and changes no gap), so the
self-contact force on the root and shared ancestor rows is exactly zero.

Rest overlap is checked in FP64 on the host, independent of the solver precision: every directed
P1Q3 sample of every pair, g = phi_target - (margin_a + margin_b); g < -1e-6 m fails and names the
bodies, shapes and depth; -1e-6 m <= g < normal_smoothing is reported as pre-compression.

Box sources of same-tree pairs are sampled on a face grid of ``box_sample_spacing`` (default 4 mm)
instead of their 12 triangles: with 36 samples, two hand-sized boxes crossed edge to edge by 4 mm
with no sample inside (an Allegro hand making a fist).
"""

import numpy as np
import warp as wp

from ...geometry import GeoType, ShapeFlags
from .contact_sampling import _surface_samples

OVERLAP_LIMIT_M = -1e-6
DEFAULT_BOX_SAMPLE_SPACING_M = 4e-3
# Self pairs need surfaces both directions can sample and fields the host can evaluate exactly.
SUPPORTED_TYPES = (GeoType.SPHERE, GeoType.BOX, GeoType.CAPSULE)


def _short(label):
    return label.rsplit("/", 1)[-1]


def _quat_matrix(q):
    x, y, z, w = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _field(geo, scale, points):
    """Exact FP64 signed distance of local points to an analytic shape."""
    if geo == GeoType.SPHERE:
        return np.linalg.norm(points, axis=1) - scale[0]
    if geo == GeoType.CAPSULE:
        offset = points.copy()
        offset[:, 2] -= np.clip(points[:, 2], -scale[1], scale[1])
        return np.linalg.norm(offset, axis=1) - scale[0]
    q = np.abs(points) - np.asarray(scale[:3])
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(q.max(axis=1), 0.0)


class SelfCollision:
    """Resolve the self-collision option into shape pairs; check rest overlap in FP64."""

    def __init__(self, model, actors, option, *, normal_smoothing):
        self.model, self.enabled = model, option is not None
        self.pairs, self.excluded, self.report = frozenset(), (), []
        self.box_spacing = None
        self.normal_smoothing = float(normal_smoothing)
        if option is None:
            return
        if not isinstance(option, dict):
            raise ValueError("self_collision must be None or a dict such as {'exclude': [['body_a', 'body_b']]}")
        unknown = set(option) - {"exclude", "box_sample_spacing"}
        if unknown:
            raise ValueError(
                f"Unknown self_collision keys: {sorted(unknown)}; the keys are 'exclude' and 'box_sample_spacing'"
            )
        spacing = option.get("box_sample_spacing", DEFAULT_BOX_SAMPLE_SPACING_M)
        if (
            isinstance(spacing, bool)
            or not isinstance(spacing, (int, float))
            or not np.isfinite(spacing)
            or spacing <= 0
        ):
            raise ValueError(f"self_collision box_sample_spacing must be a positive finite length [m], got {spacing!r}")
        self.box_spacing = float(spacing)
        labels = list(model.body_label)
        body_owner = actors.maps["body_owner"]
        kinds = [a.kind for a in actors.actors]

        def body(label):
            if not isinstance(label, str):
                raise ValueError(f"self_collision exclusion labels must be strings, got {label!r}")
            matches = [b for b, full in enumerate(labels) if label in (full, _short(full))]
            if len(matches) != 1:
                reason = "no body" if not matches else f"{len(matches)} bodies"
                raise ValueError(f"self_collision exclusion label {label!r} matches {reason}; labels must be unique")
            return matches[0]

        excluded = []
        for entry in option.get("exclude", ()):
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(f"self_collision exclusions are body label pairs, got {entry!r}")
            a, b = body(entry[0]), body(entry[1])
            if a == b or body_owner[a] != body_owner[b] or kinds[body_owner[a]] != "articulation":
                raise ValueError(f"self_collision exclusion {list(entry)} is not a pair of bodies of one tree")
            excluded.append(tuple(sorted((a, b))))
        flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
        types, scales = model.shape_type.numpy(), model.shape_scale.numpy()
        owner = actors.maps["shape_owner"]
        pairs = model.shape_contact_pairs.numpy() if model.shape_contact_pair_count else np.zeros((0, 2), int)
        collide = int(ShapeFlags.COLLIDE_SHAPES)
        candidates, body_pairs = set(), set()
        for a, b in (tuple(sorted(map(int, p))) for p in pairs):
            ba, bb = int(bodies[a]), int(bodies[b])
            if ba < 0 or bb < 0 or ba == bb or owner[a] != owner[b] or kinds[owner[a]] != "articulation":
                continue
            # Assets may clear COLLIDE_SHAPES after finalization without rebuilding the pair list.
            if not (flags[a] & collide and flags[b] & collide):
                continue
            key = tuple(sorted((ba, bb)))
            body_pairs.add(key)
            if key in excluded:
                continue
            for s in (a, b):
                if GeoType(int(types[s])) not in SUPPORTED_TYPES:
                    raise ValueError(
                        f"Self-collision shape {model.shape_label[s]!r} ({GeoType(int(types[s])).name}, body "
                        f"{labels[bodies[s]]!r}) is not supported: same-tree contact samples spheres, boxes and "
                        "capsules; clear its COLLIDE_SHAPES flag or exclude its body pair"
                    )
            candidates.add((a, b))
        for key in excluded:
            if key not in body_pairs:
                raise ValueError(
                    f"self_collision exclusion [{labels[key[0]]!r}, {labels[key[1]]!r}] has no same-tree Model "
                    "contact pair between those bodies"
                )
        if not candidates:
            raise ValueError(
                "self_collision is enabled but the Model has no same-tree contact pairs; build the articulation "
                "with enable_self_collisions=True (Newton filters parent-child pairs)"
            )
        self.pairs, self.excluded = frozenset(candidates), tuple(sorted(set(excluded)))
        # Directed samples in each source shape's frame, as the solver's sampler places them.
        self._samples = {}
        for a, b in sorted(self.pairs):
            for s in (a, b):
                if s not in self._samples:
                    spacing = self.box_spacing if GeoType(int(types[s])) == GeoType.BOX else None
                    points = _surface_samples(
                        int(types[s]), scales[s].astype(float), model.shape_source[s], box_spacing=spacing
                    )[0]
                    self._samples[s] = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        self._kinematics = None

    @property
    def identity(self):
        """Checkpoint identity: the resolved pairs, exclusions and box sample spacing [nm]."""
        pairs = np.asarray(sorted(self.pairs), dtype=np.int32).reshape(-1, 2)
        excluded = np.asarray(self.excluded, dtype=np.int32).reshape(-1, 2)
        spacing = np.full((1, 2), round(self.box_spacing * 1e9), dtype=np.int32)
        separator = np.full((1, 2), -1, dtype=np.int32)
        return np.concatenate((pairs, separator, excluded, separator, spacing))

    def bind(self, articulation_actors):
        """Build the FP64 kinematics used for every overlap check (independent of the solver precision).

        Its construction evaluates FK at the Model coordinates, which the construction-time check uses.
        """
        if not self.enabled:
            return
        from .articulation import _TreeKinematics  # noqa: PLC0415

        self._kinematics = _TreeKinematics(self.model, actors=articulation_actors, dtype=wp.float64)
        self._model_poses = self._read_poses()

    def _read_poses(self):
        k = self._kinematics
        values = k.poses.numpy().astype(np.float64)
        poses = {}
        for local, body in enumerate(k.body_map.numpy()):
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = _quat_matrix(values[local, 3:]), values[local, :3]
            poses[int(body)] = m
        return poses

    def _poses(self, state):
        """World body poses (4x4, FP64) at the Model coordinates (``state=None``) or a coupled state's."""
        if state is None:
            return self._model_poses
        k = self._kinematics
        roots = None
        if k.floating_count:
            history = state._monolithic_coupled
            roots = (
                wp.array(history.floating_com.numpy().astype(np.float64), dtype=wp.vec3d, device=k.device),
                wp.array(history.floating_rotation.numpy().astype(np.float64), dtype=wp.quatd, device=k.device),
            )
        k.evaluate(k.gather_q(state), roots=roots)
        return self._read_poses()

    def check(self, state=None):
        """Fail on rest overlap (g < -1e-6 m); record pre-compression (-1e-6 m <= g < normal_smoothing)."""
        if not self.enabled:
            return []
        model = self.model
        poses = self._poses(state)
        bodies, types = model.shape_body.numpy(), model.shape_type.numpy()
        scales, margins = model.shape_scale.numpy().astype(np.float64), model.shape_margin.numpy().astype(np.float64)
        transforms = model.shape_transform.numpy().astype(np.float64)

        def frame(s):
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = _quat_matrix(transforms[s, 3:]), transforms[s, :3]
            return poses[int(bodies[s])] @ m

        rows, overlaps = [], []
        for a, b in sorted(self.pairs):
            gap = np.inf
            for source, target in ((a, b), (b, a)):
                world = frame(source) @ np.c_[self._samples[source], np.ones(len(self._samples[source]))].T
                local = (np.linalg.inv(frame(target)) @ world)[:3].T
                g = _field(GeoType(int(types[target])), scales[target], local) - (margins[source] + margins[target])
                gap = min(gap, float(g.min()))
            labels = [model.body_label[int(bodies[s])] for s in (a, b)]
            row = {"shapes": [model.shape_label[a], model.shape_label[b]], "bodies": labels, "gap_m": gap}
            if gap < OVERLAP_LIMIT_M:
                overlaps.append(row)
            elif gap < self.normal_smoothing:
                rows.append(row)
        self.report = rows
        if overlaps:
            details = "; ".join(f"{r['bodies']} shapes {r['shapes']} depth {-r['gap_m']:.3e} m" for r in overlaps)
            raise ValueError(
                f"Self-collision rest overlap (g < {OVERLAP_LIMIT_M} m) for {len(overlaps)} pairs: {details}. "
                "Exclude these body pairs (self_collision={'exclude': [...]}) or change the pose"
            )
        return rows
