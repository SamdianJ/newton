# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Create the P9b coarse shirt with Blender's collapse decimator.

Run with Blender, not the project Python interpreter::

    blender --background --factory-startup -noaudio --python-exit-code 1 \
        --python scripts/monolithic_reference/prepare_cloth_shirt.py -- \
        --evidence-dir /absolute/path/to/asset-evidence

The source coordinates and pose are retained, with world transforms and USD
units baked into SI coordinates. Blender performs the actual mesh simplification;
pxr writes that result to an intentionally minimal, independent USD mesh.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _components(neighbors):
    remaining = set(neighbors)
    result = []
    while remaining:
        reached = {min(remaining)}
        pending = list(reached)
        while pending:
            for neighbor in neighbors[pending.pop()]:
                if neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        remaining.difference_update(reached)
        result.append(sorted(reached))
    return result


def _statistics(points, faces):
    """Measure topology and static, nonincident triangle intersections."""
    points = np.asarray(points, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    triangles = points[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    edges = defaultdict(list)
    neighbors = {i: set() for i in range(len(points))}
    for face_index, face in enumerate(faces):
        for first, second in zip(face, np.roll(face, -1), strict=True):
            a, b = int(first), int(second)
            edges[tuple(sorted((a, b)))].append((face_index, a, b))
            neighbors[a].add(b)
            neighbors[b].add(a)
    boundary = defaultdict(set)
    for (a, b), incident in edges.items():
        if len(incident) == 1:
            boundary[a].add(b)
            boundary[b].add(a)
    loops = []
    for vertices in _components(boundary):
        vertex_set = set(vertices)
        perimeter = sum(
            np.linalg.norm(points[a] - points[b])
            for a, b in edges
            if a in vertex_set and b in vertex_set and len(edges[(a, b)]) == 1
        )
        loops.append(
            {
                "vertices": len(vertices),
                "closed": all(len(boundary[v]) == 2 for v in vertices),
                "centroid_m": points[vertices].mean(axis=0).tolist(),
                "bbox_min_m": points[vertices].min(axis=0).tolist(),
                "bbox_max_m": points[vertices].max(axis=0).tolist(),
                "perimeter_m": float(perimeter),
            }
        )
    if len(loops) == 4:
        ordered = sorted(loops, key=lambda loop: loop["centroid_m"][1])
        ordered[0]["opening"] = "hem"
        ordered[-1]["opening"] = "neck"
        for loop in ordered[1:-1]:
            loop["opening"] = "left_cuff" if loop["centroid_m"][0] < 0 else "right_cuff"
    bvh = BVHTree.FromPolygons(points.tolist(), faces.tolist(), all_triangles=True, epsilon=0.0)
    overlap = {(i, j) for i, j in bvh.overlap(bvh) if i < j and not set(faces[i]).intersection(faces[j])}
    edge_lengths = np.array([np.linalg.norm(points[a] - points[b]) for a, b in edges])
    probes = []
    for fraction in (0.3, 0.5, 0.7):
        x = float((points[:, 0].min() + points[:, 0].max()) / 2)
        y = float(points[:, 1].min() + fraction * np.ptp(points[:, 1]))
        origin = Vector((x, y, float(points[:, 2].min() - 0.1)))
        hits = []
        for _ in range(10):
            hit, _, _, _ = bvh.ray_cast(origin, Vector((0, 0, 1)))
            if hit is None:
                break
            hits.append(float(hit.z))
            origin = hit + Vector((0, 0, 1.0e-6))
        probes.append({"xy_m": [x, y], "surface_z_m": hits})
    return {
        "vertices": len(points),
        "triangles": len(faces),
        "edges": len(edges),
        "area_m2": float(areas.sum()),
        "min_triangle_area_m2": float(areas.min()),
        "bbox_min_m": points.min(axis=0).tolist(),
        "bbox_max_m": points.max(axis=0).tolist(),
        "dimensions_m": np.ptp(points, axis=0).tolist(),
        "edge_length_min_m": float(edge_lengths.min()),
        "edge_length_median_m": float(np.median(edge_lengths)),
        "edge_length_max_m": float(edge_lengths.max()),
        "components": len(_components(neighbors)),
        "isolated_vertices": sum(not n for n in neighbors.values()),
        "euler_characteristic": len(points) - len(edges) + len(faces),
        "boundary_edges": sum(len(incident) == 1 for incident in edges.values()),
        "boundary_loops": loops,
        "nonmanifold_edges": sum(len(incident) > 2 for incident in edges.values()),
        "inconsistent_winding_edges": sum(
            len(incident) == 2 and incident[0][1:] == incident[1][1:] for incident in edges.values()
        ),
        "duplicate_faces": len(faces) - len({tuple(sorted(face)) for face in faces}),
        "degenerate_triangles": int(np.count_nonzero(areas <= 1.0e-14)),
        "finite_coordinates": bool(np.isfinite(points).all()),
        "body_front_back_ray_probes": probes,
        "intersection_check": {
            "algorithm": "Blender mathutils BVHTree self-overlap, epsilon=0; unique nonincident triangle pairs",
            "nonincident_intersecting_pairs": len(overlap),
            "first_pairs": [list(pair) for pair in sorted(overlap)[:20]],
            "coverage": "Static triangle overlap on the complete mesh, excluding pairs sharing any vertex.",
            "limitations": "Floating-point diagnostic, not exact predicates or CCD; excludes adjacent-fold overlap; does not certify finite-thickness separation.",
        },
    }


def _write_usd(path, points, faces, up_axis):
    from pxr import Gf, Usd, UsdGeom, Vt

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/root/shirt")
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*point) for point in points]))
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(faces, dtype=np.int32).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.rightHanded)
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    stage.GetRootLayer().Save()


def _preview(obj, evidence_dir):
    """Save inspectable front/back/oblique views without changing the mesh."""
    scene = bpy.context.scene
    # This changes only this background process; no user preferences are saved.
    bpy.context.preferences.filepaths.file_preview_type = "NONE"
    bpy.context.preferences.filepaths.save_version = 0
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 12
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 4
    scene.render.resolution_x = 1000
    scene.render.resolution_y = 1000
    scene.render.resolution_percentage = 100
    scene.world.color = (0.3, 0.3, 0.3)
    scene.view_settings.view_transform = "Standard"
    scene.render.image_settings.file_format = "PNG"
    material = bpy.data.materials.new("Preview blue fabric")
    material.diffuse_color = (0.12, 0.38, 0.65, 1.0)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = material.diffuse_color
    bsdf.inputs["Roughness"].default_value = 0.85
    obj.data.materials.append(material)
    points = np.array([tuple(v.co) for v in obj.data.vertices])
    center = Vector((points.min(axis=0) + points.max(axis=0)) / 2)
    camera_data = bpy.data.cameras.new("Evidence camera")
    camera = bpy.data.objects.new("Evidence camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = float(np.ptp(points, axis=0).max() * 1.25)
    light_data = bpy.data.lights.new("Evidence key", "AREA")
    light_data.energy = 65
    light_data.shape = "DISK"
    light_data.size = 1.2
    light = bpy.data.objects.new("Evidence key", light_data)
    scene.collection.objects.link(light)
    paths = []
    for name, direction in (
        ("front", (0, 0, 1)),
        ("back", (0, 0, -1)),
        ("oblique", (0.8, -0.7, 1)),
        ("openings", (0, -1, 0)),
    ):
        camera.location = center + Vector(direction).normalized() * 1.8
        camera.rotation_euler = (center - camera.location).to_track_quat("-Z", "Y").to_euler()
        light.location = camera.location + Vector((-0.4, 0.0, 0.7))
        light.rotation_euler = (center - light.location).to_track_quat("-Z", "Y").to_euler()
        path = evidence_dir / f"shirt-{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        paths.append(str(path))
    obj.show_wire = True
    obj.show_all_edges = True
    bpy.ops.wm.save_as_mainfile(filepath=str(evidence_dir / "unisex_shirt_coarse.blend"))
    return paths


def main():
    from pxr import Gf, Usd, UsdGeom

    repo = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=repo / "newton/examples/assets/unisex_shirt.usd")
    parser.add_argument("--output", type=Path, default=repo / "newton/examples/assets/unisex_shirt_coarse.usd")
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--target-triangles", type=int, default=2500)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    if args.source.resolve() == args.output.resolve():
        raise ValueError("The output must not overwrite the source shirt.")
    if not 2000 <= args.target_triangles <= 3000:
        raise ValueError("P9b expects a target between 2,000 and 3,000 triangles.")
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(args.source)
    stage = Usd.Stage.Open(str(args.source))
    prim = stage.GetPrimAtPath("/root/shirt")
    source = UsdGeom.Mesh(prim)
    if not source or any(a.GetNumTimeSamples() for a in prim.GetAttributes()):
        raise ValueError("Expected a static /root/shirt USD mesh.")
    counts = list(source.GetFaceVertexCountsAttr().Get())
    if any(count != 3 for count in counts):
        raise ValueError("Expected the original triangle mesh.")
    units = UsdGeom.GetStageMetersPerUnit(stage)
    up_axis = UsdGeom.GetStageUpAxis(stage)
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    points = [tuple(transform.Transform(Gf.Vec3d(*point)) * units) for point in source.GetPointsAttr().Get()]
    faces = np.asarray(source.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3).tolist()
    before = _statistics(points, faces)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    mesh = bpy.data.meshes.new("shirt")
    mesh.from_pydata(points, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("shirt", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    modifier = obj.modifiers.new("P9b collapse simplification", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = args.target_triangles / len(faces)
    modifier.use_collapse_triangulate = True
    ratio = modifier.ratio
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    points = [tuple(v.co) for v in obj.data.vertices]
    faces = [tuple(p.vertices) for p in obj.data.polygons]
    after = _statistics(points, faces)
    receipt = {
        "generator": str(Path(__file__).resolve()),
        "generator_sha256": _sha256(Path(__file__)),
        "invocation_argv": sys.argv,
        "blender_version": bpy.app.version_string,
        "blender_build_hash": bpy.app.build_hash.decode(),
        "execution": "Isolated Blender background; MCP unavailable; no GUI or global configuration changed.",
        "source": str(args.source.resolve()),
        "source_sha256": source_hash,
        "source_meters_per_unit": units,
        "source_up_axis": up_axis,
        "source_units_authored": stage.HasAuthoredMetadata("metersPerUnit"),
        "source_up_axis_authored": stage.HasAuthoredMetadata("upAxis"),
        "source_authored_attributes": [a.GetName() for a in prim.GetAttributes() if a.HasAuthoredValueOpinion()],
        "output": str(args.output.resolve()),
        "output_prim": "/root/shirt",
        "output_meters_per_unit": 1.0,
        "output_up_axis": up_axis,
        "operations": [
            "Read USD world points and multiply by metersPerUnit; retain original coordinate axes/position/pose.",
            f"Blender DECIMATE COLLAPSE, ratio={ratio}, triangulation enabled, symmetry disabled.",
            "Write evaluated mesh only to independent USD; do not add shell thickness or flatten the garment.",
        ],
        "before": before,
        "after": after,
        "area_relative_change": after["area_m2"] / before["area_m2"] - 1,
        "simulation_performed": False,
    }
    errors = []
    if not 2000 <= after["triangles"] <= 3000:
        errors.append("triangle count outside P9b range")
    for key in ("components", "euler_characteristic"):
        if after[key] != before[key]:
            errors.append(f"changed {key}")
    if len(after["boundary_loops"]) != len(before["boundary_loops"]) or not all(
        loop["closed"] for loop in after["boundary_loops"]
    ):
        errors.append("changed or invalid boundary loops")
    for key in (
        "nonmanifold_edges",
        "inconsistent_winding_edges",
        "duplicate_faces",
        "degenerate_triangles",
        "isolated_vertices",
    ):
        if after[key]:
            errors.append(f"nonzero {key}")
    if not after["finite_coordinates"] or after["intersection_check"]["nonincident_intersecting_pairs"]:
        errors.append("nonfinite geometry or detected static intersections")
    receipt["validation_errors"] = errors
    receipt_path = args.evidence_dir / "asset-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if errors:
        raise ValueError("; ".join(errors))
    _write_usd(args.output, points, faces, up_axis)
    reopened = Usd.Stage.Open(str(args.output))
    result = UsdGeom.Mesh(reopened.GetPrimAtPath("/root/shirt"))
    valid, reason = UsdGeom.Mesh.ValidateTopology(
        result.GetFaceVertexIndicesAttr().Get(),
        result.GetFaceVertexCountsAttr().Get(),
        len(result.GetPointsAttr().Get()),
    )
    if not valid:
        raise ValueError(reason)
    readback = np.asarray(result.GetPointsAttr().Get(), dtype=np.float64)
    if (
        not np.array_equal(readback, points)
        or list(result.GetFaceVertexIndicesAttr().Get()) != np.asarray(faces).ravel().tolist()
    ):
        raise ValueError("USD readback changed the evaluated geometry.")
    receipt["usd_readback_topology_valid"] = valid
    receipt["usd_readback_geometry_equal"] = True
    receipt["output_sha256"] = _sha256(args.output)
    receipt["previews"] = _preview(obj, args.evidence_dir)
    receipt["preview_sha256"] = {Path(path).name: _sha256(Path(path)) for path in receipt["previews"]}
    receipt["blend"] = str(args.evidence_dir / "unisex_shirt_coarse.blend")
    receipt["blend_sha256"] = _sha256(Path(receipt["blend"]))
    receipt["source_unchanged"] = source_hash == _sha256(args.source)
    if not receipt["source_unchanged"]:
        raise ValueError("Source changed while generating the derivative.")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "triangles": after["triangles"],
                "vertices": after["vertices"],
                "receipt": str(receipt_path),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
