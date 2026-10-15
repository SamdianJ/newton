# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Immutable Model ownership and scalar layouts; no actor dynamics or scene API."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from types import MappingProxyType

import numpy as np
import warp as wp

from ...sim import BodyFlags, JointType
from .collision import _validate_and_collect_boundary_faces


def _checked_dof_count(counts):
    if any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 0 for n in counts):
        raise ValueError("Actor DoF counts must be nonnegative integers")
    total = sum(map(int, counts))
    if total >= 2**31:
        raise ValueError("Actor layout capacity exceeds int32")
    return total


def _joint_layout(model):
    """Validate Model prefixes before distinguishing scalar joints from FREE coordinates."""
    types = model.joint_type.numpy()
    if (
        types.shape != (model.joint_count,)
        or not np.isin(types, [JointType.FIXED, JointType.REVOLUTE, JointType.PRISMATIC, JointType.FREE]).all()
    ):
        raise ValueError("Monolithic supports only FIXED, REVOLUTE, PRISMATIC and single-body FREE joints")
    free, moving = types == JointType.FREE, types != JointType.FIXED
    widths = np.where(free, 7, moving.astype(int))
    speeds = np.where(free, 6, moving.astype(int))
    prefixes = []
    for name, sizes, count in (
        ("joint_q_start", widths, model.joint_coord_count),
        ("joint_qd_start", speeds, model.joint_dof_count),
    ):
        expected = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
        _checked_dof_count(sizes)
        array = getattr(model, name)
        if array.dtype != wp.int32 or expected[-1] != count or not np.array_equal(array.numpy(), expected):
            raise ValueError(f"Invalid joint coordinate prefix layout: {name}")
        prefixes.append(expected)
    dimensions = np.stack(
        (np.where(free, 3, types == JointType.PRISMATIC), np.where(free, 3, types == JointType.REVOLUTE)), axis=1
    )
    actual_dimensions = model.joint_dof_dim.numpy()
    if not np.array_equal(actual_dimensions, dimensions) and not (
        model.joint_count == 0 and actual_dimensions.size == 0
    ):
        raise ValueError("Invalid joint_dof_dim")
    targets = model.joint_target_q_start
    if targets is None or targets.shape != (model.joint_count + 1,) or targets.dtype != wp.int32:
        raise ValueError("Invalid joint_target_q_start")
    target_count = model.joint_target_q.size
    expected = prefixes[0] if target_count == model.joint_coord_count else prefixes[1]
    if targets.dtype != wp.int32 or target_count != expected[-1] or not np.array_equal(targets.numpy(), expected):
        raise ValueError("Invalid scalar joint target layout")
    return types, prefixes[0], prefixes[1], targets.numpy()


@dataclass(frozen=True, slots=True)
class Actor:
    kind: str
    source_index: int
    body_indices: tuple[int, ...] = ()
    joint_indices: tuple[int, ...] = ()
    coord_indices: tuple[int, ...] = ()
    dof_indices: tuple[int, ...] = ()
    target_indices: tuple[int, ...] = ()
    particle_indices: tuple[int, ...] = ()
    dynamic_particle_indices: tuple[int, ...] = ()
    shape_indices: tuple[int, ...] = ()
    dof_start: int = 0
    dof_count: int = 0
    state_count: int = 0
    theta_indices: tuple[int, ...] = ()
    free_theta_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ActorTable:
    actors: tuple[Actor, ...]
    maps: Mapping[str, tuple[int, ...]]
    device_maps: Mapping[str, wp.array[wp.int32]]
    scalar_dof_count: int
    identity: str
    static_shape_indices: tuple[int, ...]

    @classmethod
    def from_model(cls, model, *, actor_order=None, materials=None):
        """Classify at most one tree/tet and independently owned rigid/material actors."""
        types, qs, vs, ts = _joint_layout(model)
        starts, ends = model.articulation_start.numpy(), model.articulation_end.numpy()
        if (
            starts.shape != (model.articulation_count + 1,)
            or ends.shape != (model.articulation_count,)
            or starts[0] != 0
            or starts[-1] != model.joint_count
            or not np.array_equal(ends, starts[1:])
            or np.any(np.diff(starts) <= 0)
        ):
            raise ValueError("Every joint must have exactly one articulation owner")
        parents, children = model.joint_parent.numpy(), model.joint_child.numpy()
        flags, worlds = model.body_flags.numpy(), model.body_world.numpy()
        shape_bodies = model.shape_body.numpy()
        if np.any(shape_bodies < -1) or np.any(shape_bodies >= model.body_count):
            raise ValueError("Invalid shape body owner")
        masses, inertias = model.body_mass.numpy(), model.body_inertia.numpy().astype(float)
        if not model.joint_enabled.numpy().all() or not np.isin(flags, [BodyFlags.DYNAMIC, BodyFlags.KINEMATIC]).all():
            raise ValueError("Actors require enabled joints and supported body flags")
        material_mode = materials is not None and bool(materials.cloth or materials.cable)
        actor_worlds = np.r_[worlds, model.particle_world.numpy()] if material_mode else worlds
        if (
            len(actor_worlds) == 0
            or len(np.unique(actor_worlds)) != 1
            or actor_worlds[0] < -1
            or actor_worlds[0] >= model.world_count
        ):
            raise ValueError("Actors must share one valid world")
        if not np.all(model.joint_world.numpy() == actor_worlds[0]):
            raise ValueError("Joint and body actors must share one world")
        actors, owned = [], set()
        for source, (start, end) in enumerate(zip(starts[:-1], ends, strict=True)):
            joints = tuple(range(int(start), int(end)))
            bodies, seen, roots, movable = [], set(), 0, set()
            for j in joints:
                parent, child = int(parents[j]), int(children[j])
                if child < 0 or child >= model.body_count or child in owned or (parent != -1 and parent not in seen):
                    raise ValueError("Actors require unique bodies in a parent-before-child world-anchored tree")
                roots += int(parent == -1)
                seen.add(child)
                owned.add(child)
                bodies.append(child)
                if types[j] != JointType.FIXED or parent in movable:
                    movable.add(child)
                if flags[child] == BodyFlags.KINEMATIC and child in movable:
                    raise ValueError("Kinematic bodies with movable ancestors are unsupported")
            if roots != 1:
                raise ValueError("Each actor requires one world root")
            free = types[list(joints)] == JointType.FREE
            if free.any() and (len(joints) != 1 or parents[start] != -1):
                raise ValueError("FREE requires a single-body world-root actor; floating bases are unsupported")
            kind = "rigid" if free.any() else "articulation"
            if kind == "rigid":
                mass = float(masses[bodies[0]])
                inertia = inertias[bodies[0]]
                second = np.trace(inertia) / 2 * np.eye(3) - inertia
                if (
                    not np.isfinite(mass)
                    or mass <= 0
                    or not np.isfinite(inertia).all()
                    or not np.allclose(inertia, inertia.T, rtol=1e-6, atol=0)
                    or np.linalg.eigvalsh(inertia).min() <= 0
                    or np.linalg.eigvalsh(second).min() < 0
                ):
                    raise ValueError("FREE actors require positive physical mass and inertia")
            coords = tuple(range(int(qs[start]), int(qs[end])))
            dofs = tuple(range(int(vs[start]), int(vs[end])))
            targets = tuple(range(int(ts[start]), int(ts[end])))
            if not dofs:
                raise ValueError("Monolithic requires a nonempty mechanical tree")
            actors.append(
                Actor(
                    kind,
                    int(start) if free.any() else source,
                    tuple(sorted(bodies)),
                    joints,
                    coords,
                    dofs,
                    targets,
                    shape_indices=tuple(map(int, np.flatnonzero(np.isin(shape_bodies, bodies)))),
                    dof_count=len(dofs),
                    state_count=len(coords),
                )
            )
        arm_count = sum(a.kind == "articulation" for a in actors)
        if len(owned) != model.body_count or arm_count > 1 or (not material_mode and arm_count != 1):
            raise ValueError("Monolithic requires uniquely owned bodies and at most one mechanical articulation")
        inv_mass = model.particle_inv_mass.numpy()
        dynamic = np.flatnonzero(inv_mass > 0)
        actors.sort(key=lambda a: (a.kind != "articulation", a.source_index))
        if model.tet_count or not material_mode:
            if material_mode:
                # Validate only the native tet boundary; cloth faces are independently owned.
                selected = np.concatenate([obj["triangle_ids"] for obj in materials.cloth]) if materials.cloth else []
                _, particles = _validate_and_collect_boundary_faces(model, excluded_triangle_ids=selected)
            else:
                _, particles = _validate_and_collect_boundary_faces(model)
            tet_dynamic = dynamic[np.isin(dynamic, particles)]
            if not material_mode and np.any(~np.isin(dynamic, particles)):
                raise ValueError("Tet particles must belong to the actor world and connected body")
            if not np.all(model.particle_world.numpy()[particles] == actor_worlds[0]):
                raise ValueError("Tet particles must belong to the actor world and connected body")
            actors.append(
                Actor(
                    "tet",
                    0,
                    particle_indices=tuple(map(int, particles)),
                    dynamic_particle_indices=tuple(map(int, tet_dynamic)),
                    dof_count=3 * len(tet_dynamic),
                    state_count=3 * len(particles),
                )
            )
        if material_mode:
            if not model.tet_count:
                faces = [int(i) for obj in materials.cloth for i in obj["triangle_ids"]]
                if sorted(faces) != list(range(model.tri_count)):
                    raise ValueError("Every triangle requires a cloth or tet owner")
            for kind, objects in (("cloth", materials.cloth), ("cable", materials.cable)):
                for source, obj in enumerate(objects):
                    ids = tuple(map(int, obj["particle_ids"]))
                    free = tuple(i for i, fixed in zip(ids, obj["fixed"], strict=True) if not fixed)
                    theta, free_theta = (), ()
                    if kind == "cable":
                        offset = int(obj["theta_offset"])
                        theta = tuple(range(offset, offset + len(obj["rest_theta"])))
                        free_theta = tuple(offset + int(i) for i in obj["free_theta"])
                    actors.append(
                        Actor(
                            kind,
                            source,
                            particle_indices=ids,
                            dynamic_particle_indices=free,
                            dof_count=3 * len(free) + len(free_theta),
                            state_count=3 * len(ids) + len(theta),
                            theta_indices=theta,
                            free_theta_indices=free_theta,
                        )
                    )
            owned_particles = {i for a in actors for i in a.particle_indices}
            if owned_particles != set(range(model.particle_count)):
                raise ValueError("Every material-scene particle must have exactly one actor owner")
        if actor_order is not None:
            if any(type(i) is not int for i in actor_order) or sorted(actor_order) != list(range(len(actors))):
                raise ValueError("actor_order must be a complete permutation")
            actors = [actors[i] for i in actor_order]
        total = _checked_dof_count(tuple(a.dof_count for a in actors))
        if not total:
            raise ValueError("Monolithic requires at least one dynamic degree of freedom")
        offset = 0
        for i, actor in enumerate(actors):
            actors[i] = replace(actor, dof_start=offset)
            offset += actor.dof_count
        maps = {}
        for stem, count in (
            ("body", model.body_count),
            ("joint", model.joint_count),
            ("coord", model.joint_coord_count),
            ("dof", model.joint_dof_count),
            ("target", model.joint_target_q.size),
            ("particle", model.particle_count),
            ("shape", model.shape_count),
        ):
            owners, local = [-1] * count, [-1] * count
            for actor_id, actor in enumerate(actors):
                for i, index in enumerate(getattr(actor, stem + "_indices")):
                    if owners[index] != -1:
                        raise ValueError(f"Duplicate {stem} owner")
                    owners[index], local[index] = actor_id, i
            maps[stem + "_owner"], maps[stem + "_local"] = tuple(owners), tuple(local)
        dofs, particles = [-1] * model.joint_dof_count, [-1] * model.particle_count
        for actor in actors:
            for i, index in enumerate(actor.dof_indices):
                dofs[index] = actor.dof_start + i
            for i, index in enumerate(actor.dynamic_particle_indices):
                particles[index] = actor.dof_start + 3 * i
        maps.update(joint_dof_to_global=tuple(dofs), particle_to_global=tuple(particles))
        if material_mode:
            owners, local, rows = ([-1] * materials.theta_count for _ in range(3))
            for actor_id, actor in enumerate(actors):
                for i, index in enumerate(actor.theta_indices):
                    owners[index], local[index] = actor_id, i
                for i, index in enumerate(actor.free_theta_indices):
                    rows[index] = actor.dof_start + 3 * len(actor.dynamic_particle_indices) + i
            maps.update(theta_owner=tuple(owners), theta_local=tuple(local), theta_to_global=tuple(rows))
        identity = hashlib.sha256(cls._payload(actors, maps)).hexdigest()
        device_maps = {name: wp.array(value, dtype=wp.int32, device=model.device) for name, value in maps.items()}
        return cls(
            tuple(actors),
            MappingProxyType(maps),
            MappingProxyType(device_maps),
            total,
            identity,
            tuple(map(int, np.flatnonzero(shape_bodies == -1))),
        )

    def layout_payload(self):
        """Return the complete versioned host description used by the layout digest."""
        return self._payload(self.actors, dict(self.maps))

    @staticmethod
    def _payload(actors, maps):
        descriptions = [asdict(a) for a in actors]
        material_mode = "theta_to_global" in maps
        if not material_mode:
            for actor in descriptions:
                del actor["theta_indices"], actor["free_theta_indices"]
        return json.dumps(
            {"version": 2 if material_mode else 1, "actors": descriptions, "maps": maps}, sort_keys=True
        ).encode()
