# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static authoring of the experimental monolithic material attribute convention."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .material_inputs import _MaterialInputs
from .solver_types import CableInput, ClothInput

if TYPE_CHECKING:
    from ...sim.builder import ModelBuilder
    from ...sim.model import Model

_PREFIX = "newton:monolithic:"


def import_materials_usd(builder: ModelBuilder, source: str | Path) -> dict:
    """Append independently owned rest materials from a USD file to a builder.

    .. experimental::

        This callable implements version 1 of the project-specific attribute
        convention, not a registered USD schema. It may change without notice.

    Args:
        builder: Destination builder; imported objects use its current world.
        source: USD file with triangular Mesh or open linear BasisCurves prims.

    Returns:
        Keyword arguments ``cloth_inputs`` and ``cable_inputs`` for
        :class:`SolverMonolithic`, with IDs in the destination builder.

    Raises:
        ValueError: Missing/invalid attributes, topology, animated material data,
            or nonuniform, reflected or sheared transforms. Validation finishes
            before modifying the destination builder.

    Points and translations are converted using stage metersPerUnit. Other
    material values are SI, independent of stage units. World coordinates retain
    the authored axes. Directors rotate with the world transform. Rest data do
    not contain velocities, contact history or dynamic checkpoint state.
    """
    from pxr import Usd, UsdGeom

    from ...sim.builder import ModelBuilder  # noqa: PLC0415

    stage = Usd.Stage.Open(str(source))
    if stage is None:
        raise ValueError("Cannot open material USD")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    if not np.isfinite(units) or units <= 0:
        raise ValueError("Invalid metersPerUnit")
    # add_builder also copies gravity into the destination's active world.
    gravity = builder.world_gravity[builder.current_world] if builder.current_world >= 0 else builder.gravity
    scratch = ModelBuilder(gravity=gravity)
    cloth, cable = [], []

    def read(prim, name):
        attribute = prim.GetAttribute(_PREFIX + name)
        value = attribute.Get() if attribute else None
        if value is None:
            raise ValueError(f"Missing material attribute {prim.GetPath()}: {name}")
        expected = {
            "kind": "token",
            "version": "int",
            "fixedVertices": "int[]",
            "membrane": "double2",
            "restTheta": "double[]",
            "restDirector": "vector3d[]",
            "fixedTheta": "bool[]",
        }.get(name, "double")
        if name == "bending":
            expected = "double2" if prim.GetAttribute(_PREFIX + "kind").Get() == "cloth" else "double4"
        if str(attribute.GetTypeName()) != expected:
            raise ValueError(f"Material attribute {name} requires USD type {expected}")
        return value

    for prim in stage.Traverse():
        if not prim.HasAttribute(_PREFIX + "kind"):
            continue
        ancestor = prim
        while ancestor and not ancestor.IsPseudoRoot():
            if any(attribute.GetNumTimeSamples() for attribute in ancestor.GetAttributes()):
                raise ValueError("Material USD requires static rest data and transforms")
            ancestor = ancestor.GetParent()
        if read(prim, "version") != 1:
            raise ValueError("Unsupported material USD version")
        kind = read(prim, "kind")
        geometry = UsdGeom.PointBased(prim)
        if not geometry:
            raise ValueError("Material USD requires point-based geometry")
        points = np.asarray(geometry.GetPointsAttr().Get(), dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("Invalid material USD points")
        transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        linear = transform[:3, :3]
        scale = np.linalg.norm(linear[0])
        if (
            not np.isfinite(transform).all()
            or scale <= 0
            or np.linalg.det(linear) <= 0
            or not np.allclose(linear @ linear.T, np.eye(3) * scale**2, rtol=1e-8, atol=1e-12 * scale**2)
            or not np.array_equal(transform[:, 3], [0, 0, 0, 1])
        ):
            raise ValueError("Material USD requires a positive uniform scale and rigid transform")
        points = ((points @ linear + transform[3, :3]) * units).astype(np.float32).astype(float)
        fixed = np.asarray(read(prim, "fixedVertices"))
        if (
            fixed.ndim != 1
            or fixed.dtype.kind not in "iu"
            or len(np.unique(fixed)) != len(fixed)
            or np.any(fixed < 0)
            or np.any(fixed >= len(points))
        ):
            raise ValueError("Invalid fixedVertices")
        masses = np.zeros(len(points))
        first, face_start = scratch.particle_count, scratch.tri_count
        if kind == "cloth" and prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
            if (
                counts.ndim != 1
                or not len(counts)
                or np.any(counts != 3)
                or indices.shape != (3 * len(counts),)
                or np.any(indices < 0)
                or np.any(indices >= len(points))
                or not np.array_equal(np.unique(indices), np.arange(len(points)))
            ):
                raise ValueError("Cloth USD requires triangles using every point")
            if mesh.GetOrientationAttr().Get() != UsdGeom.Tokens.rightHanded or mesh.GetHoleIndicesAttr().Get():
                raise ValueError("Cloth USD requires right-handed faces without holes")
            triangles = indices.reshape(-1, 3)
            faces = points[triangles]
            area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
            density = read(prim, "surfaceDensity")
            np.add.at(masses, triangles.ravel(), np.repeat(area * density / 3, 3))
            membrane = np.asarray(read(prim, "membrane"), dtype=float)
            if membrane.shape != (2,):
                raise ValueError("Cloth membrane requires (mu, lambda) [N/m]")
            item = ClothInput(
                triangle_ids=tuple(range(face_start, face_start + len(triangles))),
                bending=tuple(read(prim, "bending")),
                thickness=read(prim, "thickness"),
                surface_density=density,
            )
            cloth.append(item)
        elif kind == "cable" and prim.IsA(UsdGeom.BasisCurves):
            curves = UsdGeom.BasisCurves(prim)
            if (
                list(curves.GetCurveVertexCountsAttr().Get() or []) != [len(points)]
                or curves.GetTypeAttr().Get() != UsdGeom.Tokens.linear
                or curves.GetWrapAttr().Get() != UsdGeom.Tokens.nonperiodic
            ):
                raise ValueError("Cable USD requires one open linear curve")
            density = read(prim, "linearDensity")
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            masses[:-1] += lengths * density / 2
            masses[1:] += lengths * density / 2
            director = np.asarray(read(prim, "restDirector"), dtype=float)
            cable.append(
                CableInput(
                    particle_ids=tuple(range(first, first + len(points))),
                    EA=read(prim, "EA"),
                    bending=np.asarray(read(prim, "bending")).reshape(2, 2),
                    GJ=read(prim, "GJ"),
                    radius=read(prim, "radius"),
                    linear_density=density,
                    rest_theta=np.asarray(read(prim, "restTheta")),
                    rest_director=director @ (linear / scale),
                    fixed_theta=np.asarray(read(prim, "fixedTheta")),
                )
            )
        else:
            raise ValueError("Material kind and USD geometry type disagree")
        masses[fixed] = 0
        for point, mass in zip(points, masses, strict=True):
            scratch.add_particle(wp.vec3(point), wp.vec3(0), float(mass))
        if kind == "cloth":
            for face in triangles:
                scratch.add_triangle(
                    *(face + first), tri_ke=membrane[0], tri_ka=membrane[1], tri_kd=0, tri_drag=0, tri_lift=0
                )
    if not cloth and not cable:
        raise ValueError("No monolithic material prims found")
    _MaterialInputs(scratch.finalize(device="cpu"), cloth, cable)
    vertex_offset, triangle_offset = builder.particle_count, builder.tri_count
    result = {
        "cloth_inputs": tuple(
            replace(item, triangle_ids=tuple(i + triangle_offset for i in item.triangle_ids)) for item in cloth
        ),
        "cable_inputs": tuple(
            replace(item, particle_ids=tuple(i + vertex_offset for i in item.particle_ids)) for item in cable
        ),
    }
    builder.add_builder(scratch)
    return result


def export_materials_usd(path: str | Path, model: Model, *, cloth_inputs: tuple = (), cable_inputs: tuple = ()) -> None:
    """Export selected uniform rest materials using the experimental USD convention.

    .. experimental::

        This callable writes project-specific version 1 attributes; it may change
        without notice. The output represents rest geometry, not a checkpoint.

    Args:
        path: Destination USD file.
        model: Model containing the selected reference particles and triangles.
        cloth_inputs: Uniform membrane/bending objects to export.
        cable_inputs: Open DER objects to export, including unwrapped angles.

    Raises:
        ValueError: Invalid material ownership/data or nonuniform cloth membrane.

    The exported stage uses meters and baked world coordinates. Each object owns
    its points, preserving their local order; other Model actors are not exported.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    materials = _MaterialInputs(model, cloth_inputs, cable_inputs)
    for item in materials.cloth:
        if not np.all(item["membrane"] == item["membrane"][0]):
            raise ValueError("Material USD version 1 requires uniform cloth membrane")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    objects = sorted(
        [("cloth", item) for item in materials.cloth] + [("cable", item) for item in materials.cable],
        key=lambda entry: int(entry[1]["particle_ids"].min()),
    )
    if not objects:
        raise ValueError("No materials to export")
    for index, (kind, item) in enumerate(objects):
        prim_path = f"/material_{index:04d}"
        if kind == "cloth":
            geometry = UsdGeom.Mesh.Define(stage, prim_path)
            ids = item["particle_ids"]
            geometry.CreateFaceVertexCountsAttr([3] * len(item["triangles"]))
            geometry.CreateFaceVertexIndicesAttr(np.searchsorted(ids, item["triangles"]).ravel().tolist())
            geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            attributes = {
                "membrane": (Sdf.ValueTypeNames.Double2, Gf.Vec2d(*item["membrane"][0])),
                "bending": (Sdf.ValueTypeNames.Double2, Gf.Vec2d(*item["bending"])),
                "thickness": (Sdf.ValueTypeNames.Double, float(item["thickness"])),
                "surfaceDensity": (Sdf.ValueTypeNames.Double, float(item["surface_density"])),
            }
        else:
            geometry = UsdGeom.BasisCurves.Define(stage, prim_path)
            geometry.CreateCurveVertexCountsAttr([len(item["particle_ids"])])
            geometry.CreateTypeAttr(UsdGeom.Tokens.linear)
            geometry.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
            attributes = {key: (Sdf.ValueTypeNames.Double, float(item[key])) for key in ("EA", "GJ", "radius")}
            attributes.update(
                linearDensity=(Sdf.ValueTypeNames.Double, float(item["linear_density"])),
                bending=(Sdf.ValueTypeNames.Double4, Gf.Vec4d(*item["bending"].ravel())),
                restTheta=(Sdf.ValueTypeNames.DoubleArray, Vt.DoubleArray(item["rest_theta"].tolist())),
                restDirector=(Sdf.ValueTypeNames.Vector3dArray, Vt.Vec3dArray.FromNumpy(item["rest_director"])),
                fixedTheta=(Sdf.ValueTypeNames.BoolArray, Vt.BoolArray(item["fixed_theta"].tolist())),
            )
        geometry.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(item["positions"].astype(np.float32)))
        attributes.update(
            kind=(Sdf.ValueTypeNames.Token, kind),
            version=(Sdf.ValueTypeNames.Int, 1),
            fixedVertices=(Sdf.ValueTypeNames.IntArray, Vt.IntArray(np.flatnonzero(item["fixed"]).tolist())),
        )
        for name, (value_type, value) in attributes.items():
            geometry.GetPrim().CreateAttribute(_PREFIX + name, value_type, custom=True).Set(value)
    if not stage.GetRootLayer().Export(str(path)):
        raise ValueError("Cannot export material USD")
