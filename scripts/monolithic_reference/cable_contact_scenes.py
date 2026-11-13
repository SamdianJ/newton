# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cable self- and inter-contact scenes in SI units: lowering, winding, twisting and crossing."""

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .cloth_contact_run import publication

KINDS = ("coil", "plectoneme", "cross", "helix-pile", "spiral-stack")
DRIVEN_KINDS = ("coil", "plectoneme")
COMMON_FIELDS = ("scene", "kind", "dt_s", "duration_s", "gravity_m_s2", "material", "cables", "solver")
KIND_FIELDS = {
    "coil": {"ground": None, "coil": ("bottom_clearance_m", "stop_height_m", "lower_speed_m_s", "ramp_s")},
    "plectoneme": {"plectoneme": ("height_m", "approach_per_end_m", "approach_s", "twist_s", "turns")},
    "cross": {"ground": None, "cross": ("angles_deg", "heights_m", "wave_amplitude_m", "wavelength_m")},
    "helix-pile": {"ground": None, "helix": ("radius_m", "pitch_m", "bottom_clearance_m")},
    "spiral-stack": {
        "ground": None,
        "spiral": ("layers", "inner_radius_m", "radial_gap_m", "vertical_gap_m", "transition_turns"),
    },
}


def validate_recipe(recipe):
    """Name the first missing recipe field instead of failing later with a bare KeyError."""
    for name in COMMON_FIELDS:
        if name not in recipe:
            raise ValueError(f"Recipe lacks field {name}")
    kind = recipe["kind"]
    if kind not in KINDS:
        raise ValueError(f"Recipe field kind must be one of {KINDS}")
    for block, fields in KIND_FIELDS[kind].items():
        if recipe.get(block) is None:
            raise ValueError(f"Recipe kind {kind} lacks field {block}")
        for name in fields or ():
            if name not in recipe[block]:
                raise ValueError(f"Recipe lacks field {block}.{name}")
    for name in ("radius_m", "young_modulus_Pa", "poisson_ratio", "density_kg_m3"):
        if name not in recipe["material"]:
            raise ValueError(f"Recipe lacks field material.{name}")
    for index, cable in enumerate(recipe["cables"]):
        for name in ("vertices", "segment_length_m"):
            if name not in cable:
                raise ValueError(f"Recipe lacks field cables[{index}].{name}")


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Recipe field {name} must be finite and positive")
    return value


def cable_section(material):
    """Return a solid circular section [m, N, N m^2, N m^2, kg/m] from isotropic material data."""
    radius = _positive(material["radius_m"], "material.radius_m")
    young = _positive(material["young_modulus_Pa"], "material.young_modulus_Pa")
    density = _positive(material["density_kg_m3"], "material.density_kg_m3")
    poisson = float(material["poisson_ratio"])
    if not 0 <= poisson < 0.5:
        raise ValueError("Recipe field material.poisson_ratio must lie in [0, 0.5)")
    area, inertia = np.pi * radius**2, np.pi * radius**4 / 4
    return {
        "radius_m": radius,
        "EA_N": young * area,
        "EI_Nm2": young * inertia,
        "GJ_Nm2": young / (2 * (1 + poisson)) * 2 * inertia,
        "linear_density_kg_m": density * area,
    }


def smoothstep(s):
    """C1 ramp from 0 to 1 on [0, 1]."""
    s = float(np.clip(s, 0.0, 1.0))
    return s * s * (3 - 2 * s)


def trapezoid(t, distance, speed, ramp):
    """Travel [m or rad] at time t of a trapezoidal velocity profile that reaches ``distance``."""
    if distance <= 0:
        return 0.0
    ramp = min(ramp, distance / speed)
    duration = distance / speed + ramp
    t = float(np.clip(t, 0.0, duration))
    accel = speed / ramp
    if t < ramp:
        return 0.5 * accel * t * t
    if t <= duration - ramp:
        return 0.5 * speed * ramp + speed * (t - ramp)
    return distance - 0.5 * accel * (duration - t) ** 2


def edge_directors(points, hint):
    """Unit directors perpendicular to each edge, the projection of a per-edge hint."""
    tangents = np.diff(points, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1)[:, None]
    hint = np.broadcast_to(np.asarray(hint, dtype=float), tangents.shape)
    directors = hint - np.sum(hint * tangents, axis=1)[:, None] * tangents
    norms = np.linalg.norm(directors, axis=1)
    if np.any(norms < 1e-3):
        raise ValueError("Director hint is nearly parallel to a cable edge")
    return directors / norms[:, None]


class CableContactScene:
    """Build one recipe scene and advance it through the public monolithic step transaction."""

    def __init__(
        self, recipe, *, device="cpu", precision="fp64", use_cuda_graph=False, method="off", method_config=None
    ):
        if recipe.get("units") != "SI" or recipe.get("schema_version") != 1:
            raise ValueError("Require an SI cable-contact recipe with schema_version 1")
        validate_recipe(recipe)
        self.kind = recipe["kind"]
        if method != "off":
            raise ValueError("Cable contact is not available before PR2b; only method 'off' runs")
        if method_config:
            raise ValueError("Method-off runs must not silently consume an enabled configuration")
        self.recipe = recipe
        self.dt = _positive(recipe["dt_s"], "dt_s")
        duration = _positive(recipe["duration_s"], "duration_s")
        if not np.isclose(duration / self.dt, round(duration / self.dt), rtol=0, atol=1e-9):
            raise ValueError("Recipe duration_s must be an integral number of dt_s steps")
        self.total_steps, self.steps, self.phase = round(duration / self.dt), 0, "start"
        self.section = cable_section(recipe["material"])
        gravity = np.asarray(recipe["gravity_m_s2"], dtype=float)
        if gravity.shape != (3,) or not np.isfinite(gravity).all():
            raise ValueError("Recipe field gravity_m_s2 must be three finite numbers")
        self.gravity = gravity
        builder = newton.ModelBuilder(gravity=tuple(gravity))
        cfg = builder.ShapeConfig(density=0.0, margin=0.0)
        ground = recipe.get("ground")
        if ground is not None:
            if ground.get("type") != "plane":
                raise ValueError("Recipe field ground.type must be 'plane'")
            self.ground_height = float(ground["height_m"])
            builder.add_shape_plane(
                xform=wp.transform(wp.vec3(0.0, 0.0, self.ground_height), wp.quat_identity()),
                width=0.0,
                length=0.0,
                cfg=cfg,
            )
        else:
            self.ground_height = None
        build = getattr(self, f"_build_{self.kind.replace('-', '_')}")
        curves, fixed, fixed_theta, hints = build(recipe)
        self.vertex_ranges, cables, start = [], [], 0
        r = self.section
        self.rest = []
        for curve, pinned, clamp, hint in zip(curves, fixed, fixed_theta, hints, strict=True):
            points = np.asarray(curve, dtype=np.float32).astype(float)
            if len(points) < 4:
                raise ValueError("Each cable needs at least four vertices")
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            if lengths.min() < r["radius_m"]:
                raise ValueError("Cable segments must not be shorter than the cable radius")
            mass = np.zeros(len(points))
            mass[:-1] += lengths * r["linear_density_kg_m"] / 2
            mass[1:] += lengths * r["linear_density_kg_m"] / 2
            ids = []
            for i, point in enumerate(points):
                ids.append(
                    builder.add_particle(
                        wp.vec3(point), wp.vec3(0.0), 0.0 if i in pinned else float(mass[i]), radius=r["radius_m"]
                    )
                )
            theta_fixed = np.zeros(len(points) - 1, dtype=bool)
            theta_fixed[list(clamp)] = True
            cables.append(
                SolverMonolithic.CableInput(
                    particle_ids=tuple(ids),
                    EA=r["EA_N"],
                    bending=np.eye(2) * r["EI_Nm2"],
                    GJ=r["GJ_Nm2"],
                    radius=r["radius_m"],
                    linear_density=r["linear_density_kg_m"],
                    rest_theta=np.zeros(len(points) - 1),
                    rest_director=edge_directors(points, hint),
                    fixed_theta=theta_fixed,
                )
            )
            self.vertex_ranges.append((start, start + len(points)))
            self.rest.append(points)
            start += len(points)
        self.fixed_vertices = sorted(
            start_ + i for (start_, _), pinned in zip(self.vertex_ranges, fixed, strict=True) for i in pinned
        )
        self.fixed_edges = [list(clamp) for clamp in fixed_theta]
        self.model = builder.finalize(device=device)
        self.cables = cables
        options = dict(
            recipe["solver"],
            cable_inputs=tuple(cables),
            precision=SolverMonolithic.Precision(precision),
            pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
            use_cuda_graph=use_cuda_graph,
        )
        self.solver = SolverMonolithic(self.model, **options)
        self.state = self.solver.create_state(dt=self.dt)
        self.control = self.solver.create_control()
        self.initial_q = self.model.particle_q.numpy().astype(float)
        self.target = self.initial_q.copy()
        self.target_theta = self.control._monolithic_cable.target_theta.numpy().astype(float)
        self.theta_offsets = np.cumsum([0] + [len(c.particle_ids) - 1 for c in cables])
        perturbation = recipe.get("perturbation")
        if perturbation is not None:
            self._perturb(perturbation)
        self.metadata = {
            "kind": f"cable-contact-{self.kind}",
            "scene": recipe.get("scene"),
            "recipe": recipe,
            "method": method,
            "method_config": method_config,
            "section": self.section,
            "cables": [{"vertices": b - a, "first_particle": a} for a, b in self.vertex_ranges],
            "fixed_vertices": self.fixed_vertices,
            "fixed_theta_edges": self.fixed_edges,
            "total_mass_kg": float(self.model.particle_mass.numpy().sum()),
            "gravity_m_s2": gravity.tolist(),
            "ground": "analytic plane" if ground is not None else "none",
        }

    # ---- scene geometry -------------------------------------------------------------------

    def _segments(self, spec):
        count = int(spec["vertices"])
        if count < 4:
            raise ValueError("Recipe cables need at least four vertices")
        return count, _positive(spec["segment_length_m"], "cables.segment_length_m")

    def _build_coil(self, recipe):
        (spec,), coil = recipe["cables"], recipe["coil"]
        count, length = self._segments(spec)
        bottom = self.ground_height + _positive(coil["bottom_clearance_m"], "coil.bottom_clearance_m")
        z = bottom + length * np.arange(count)[::-1]
        points = np.column_stack([np.zeros(count), np.zeros(count), z])
        self._top0 = points[0].copy()
        self._stop = self.ground_height + _positive(coil["stop_height_m"], "coil.stop_height_m")
        if self._stop >= self._top0[2]:
            raise ValueError("Recipe field coil.stop_height_m must lie below the initial top")
        self._speed = _positive(coil["lower_speed_m_s"], "coil.lower_speed_m_s")
        self._ramp = _positive(coil["ramp_s"], "coil.ramp_s")
        travel = self._top0[2] - self._stop
        self.drive_end_s = travel / self._speed + self._ramp
        return [points], [[0]], [[0]], [(1.0, 0.0, 0.0)]

    def _build_plectoneme(self, recipe):
        (spec,), twist = recipe["cables"], recipe["plectoneme"]
        count, length = self._segments(spec)
        height = float(twist["height_m"])
        x = length * (np.arange(count) - (count - 1) / 2)
        self._approach = _positive(twist["approach_per_end_m"], "plectoneme.approach_per_end_m")
        if 2 * self._approach >= x[-1] - x[0]:
            raise ValueError("Recipe field plectoneme.approach_per_end_m closes the clamps")
        self._phases = [_positive(twist[name], f"plectoneme.{name}") for name in ("approach_s", "twist_s")]
        self._twist_turns = _positive(twist["turns"], "plectoneme.turns")
        self.drive_end_s = sum(self._phases)
        clamp = [0, 1, count - 2, count - 1]
        return (
            [np.column_stack([x, np.zeros(count), np.full(count, height)])],
            [clamp],
            [[0, count - 2]],
            [(0.0, 0.0, 1.0)],
        )

    def _build_cross(self, recipe):
        cross, specs = recipe["cross"], recipe["cables"]
        angles, heights = cross["angles_deg"], cross["heights_m"]
        if not len(specs) == len(angles) == len(heights) >= 2:
            raise ValueError("Recipe cross needs one angle and one height per cable, at least two cables")
        amplitude, wavelength = float(cross["wave_amplitude_m"]), _positive(cross["wavelength_m"], "cross.wavelength_m")
        curves = []
        for spec, angle, height in zip(specs, angles, heights, strict=True):
            count, length = self._segments(spec)
            total = length * (count - 1)
            # Equal-arc vertices along a planar sine centred on the origin, then rotated.
            u = np.linspace(-total, total, 400 * count)
            v = amplitude * np.sin(2 * np.pi * u / wavelength + np.deg2rad(angle))
            arc = np.r_[0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]
            centre = np.interp(0.0, u, arc)
            s = centre + length * (np.arange(count) - (count - 1) / 2)
            u, v = np.interp(s, arc, u), np.interp(s, arc, v)
            c, w = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            curves.append(np.column_stack([u * c - v * w, u * w + v * c, np.full(count, self.ground_height + height)]))
        self.drive_end_s = 0.0
        return curves, [[] for _ in curves], [[] for _ in curves], [(0.0, 0.0, 1.0)] * len(curves)

    def _build_helix_pile(self, recipe):
        (spec,), helix = recipe["cables"], recipe["helix"]
        count, length = self._segments(spec)
        radius = _positive(helix["radius_m"], "helix.radius_m")
        pitch = _positive(helix["pitch_m"], "helix.pitch_m")
        rise = pitch / (2 * np.pi)
        step = length / np.hypot(radius, rise)
        u = step * np.arange(count)
        bottom = self.ground_height + _positive(helix["bottom_clearance_m"], "helix.bottom_clearance_m")
        points = np.column_stack([radius * np.cos(u), radius * np.sin(u), bottom + rise * u])
        middle = (u[:-1] + u[1:]) / 2
        hints = np.column_stack([-np.cos(middle), -np.sin(middle), np.zeros(count - 1)])
        self.drive_end_s = 0.0
        return [points], [[]], [[]], [hints]

    def _build_spiral_stack(self, recipe):
        """Flat Archimedean spiral layers, alternately outward and inward, resting on the table.

        Neighbouring turns and layers are ``radial_gap_m`` and ``vertical_gap_m`` apart at the
        surface; the bottom layer touches the plane. The rest shape is the stack itself, so it
        is stress free: a fixed state for timing one step with and without cable contact.
        """
        (spec,), spiral = recipe["cables"], recipe["spiral"]
        count, length = self._segments(spec)
        layers = int(spiral["layers"])
        if layers < 1:
            raise ValueError("Recipe field spiral.layers must be a positive integer")
        radius = self.section["radius_m"]
        inner = _positive(spiral["inner_radius_m"], "spiral.inner_radius_m")
        # Chords of a turn of radius R bow inward by L^2 / (8 R); add the largest sagitta
        # so that the polygonal turns, not only their vertices, keep the requested gap.
        pitch = 2 * radius + float(spiral["radial_gap_m"]) + length**2 / (8 * inner)
        rise = 2 * radius + float(spiral["vertical_gap_m"])
        transition = _positive(spiral["transition_turns"], "spiral.transition_turns") * 2 * np.pi
        if min(pitch, rise) < 2 * radius:
            raise ValueError("Recipe spiral gaps must not be negative")
        per_layer = length * (count - 1) / layers
        pieces, start_angle, start_radius = [], 0.0, inner
        for layer in range(layers):
            sign = 1.0 if layer % 2 == 0 else -1.0
            angle = np.linspace(0.0, 200 * np.pi, 200_000)
            r = start_radius + sign * pitch * angle / (2 * np.pi)
            arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(r), r[1:] * np.diff(angle)))]
            stop = int(np.searchsorted(arc, per_layer)) + 1
            if stop >= len(angle) or np.any(r[:stop] <= 0):
                raise ValueError("Recipe spiral does not fit its layers; enlarge spiral.inner_radius_m")
            angle, r = angle[:stop], r[:stop]
            z = np.full(stop, self.ground_height + radius + layer * rise)
            if layer + 1 < layers:
                ramp = angle >= angle[-1] - transition
                z[ramp] += rise * (angle[ramp] - (angle[-1] - transition)) / transition
            phi = start_angle + angle
            pieces.append(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))
            start_angle, start_radius = phi[-1], r[-1]
        dense = np.vstack(pieces)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))]
        s = length * np.arange(count)
        if s[-1] > arc[-1]:
            raise ValueError("Recipe spiral is shorter than the cable")
        points = np.column_stack([np.interp(s, arc, dense[:, k]) for k in range(3)])
        self.drive_end_s = 0.0
        return [points], [[]], [[]], [(0.0, 0.0, 1.0)]

    def _perturb(self, perturbation):
        """Add a small deterministic helical velocity field to break planar symmetry."""
        speed = float(perturbation["velocity_m_s"])
        waves = float(perturbation["waves"])
        velocity = np.zeros_like(self.initial_q)
        for a, b in self.vertex_ranges:
            s = np.linspace(0, 1, b - a)
            velocity[a:b, 0] = speed * np.cos(2 * np.pi * waves * s)
            velocity[a:b, 1] = speed * np.sin(2 * np.pi * waves * s)
        velocity[self.fixed_vertices] = 0
        self.state.particle_qd.assign(velocity)
        self.solver.reset_state(self.state, dt=self.dt)

    # ---- prescribed boundary motion -------------------------------------------------------

    def targets(self, t):
        """Return prescribed vertex positions and edge angles at time t [s]."""
        target, theta = self.initial_q.copy(), self.target_theta.copy()
        if self.kind == "coil":
            target[0, 2] = self._top0[2] - trapezoid(t, self._top0[2] - self._stop, self._speed, self._ramp)
        elif self.kind == "plectoneme":
            approach, twist = self._phases
            shift = self._approach * smoothstep(t / approach)
            turns = self._twist_turns * smoothstep((t - approach) / twist)
            a, b = self.vertex_ranges[0]
            target[a : a + 2, 0] += shift
            target[b - 2 : b, 0] -= shift
            theta[0] -= np.pi * turns
            theta[b - a - 2] += np.pi * turns
        return target, theta

    def phase_at(self, t):
        if self.kind in DRIVEN_KINDS:
            if self.kind == "plectoneme":
                approach, twist = self._phases
                return "approach" if t <= approach else "twist" if t <= approach + twist else "hold"
            return "drive" if t <= self.drive_end_s else "hold"
        return "free"

    def step(self):
        """Prescribe end-of-step boundary targets, solve, and require a valid publication."""
        t = (self.steps + 1) * self.dt
        target, theta = self.targets(t)
        self.control._monolithic_cable.target_position.assign(target)
        self.control._monolithic_cable.target_theta.assign(theta)
        self.phase = self.phase_at(t)
        previous = self.solver.last_stats.step_generation
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        stats = self.solver.last_stats
        self.last_forces = self.solver.get_contact_forces(self.state)
        if publication(stats, self.last_forces, previous_generation=previous) == "HARD_FAILURE":
            raise RuntimeError(f"{stats.status}: {stats.failure_reason}")
        self.steps += 1
        return stats
