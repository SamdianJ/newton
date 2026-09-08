# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hash-checked hand/ball assembly for internal PR-6D asset validation."""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import ShapeFlags
from newton.examples.softbody.monolithic_soft_ball import RADIUS, load_ball
from newton.examples.softbody.sharpa_close import build_hand, finalize_hand, sha256


def sdf_build_identity():
    """Bind cached SDFs to the build/query implementation and Warp version."""
    root = Path(newton.__file__).parent
    sources = (
        "_src/geometry/types.py",
        "_src/geometry/sdf_utils.py",
        "_src/geometry/sdf_texture.py",
        "_src/sim/builder.py",
    )
    return {"warp_version": wp.__version__, "source_sha256": {name: sha256(root / name) for name in sources}}


def build_contact_hand(asset_dir, contact_dir, *, device, parameters, resolution=128, _mount=None):
    """Load validated collision meshes; keep visual shapes and all fixed links.

    CPU imports geometry with collision flags disabled for import-only checks.
    The CUDA path explicitly builds every SDF and enables particle contact.
    """
    asset_dir, contact_dir = Path(asset_dir).resolve(), Path(contact_dir).resolve()
    manifest_path = contact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "sharpa-contact-assets/v1" or not manifest.get("complete"):
        raise ValueError("Incomplete Sharpa contact manifest")
    if manifest["urdf_sha256"] != sha256(asset_dir / "left_sharpa_wave.urdf"):
        raise ValueError("Stale Sharpa URDF hash")
    if manifest.get("sdf_build_identity") != sdf_build_identity():
        raise ValueError("Stale SDF build identity; regenerate the contact assets")
    for name, expected in manifest.get("cache_sha256", {}).items():
        path = (contact_dir / "sdf-cache" / name).resolve()
        if not path.is_relative_to(contact_dir / "sdf-cache"):
            raise ValueError("Invalid SDF cache path")
        if path.exists() and sha256(path) != expected:
            raise ValueError("Stale SDF cache hash")
    rows = {row["source"]: row for row in manifest["meshes"]}
    if len(rows) != 11 or len(manifest["references"]) != 26:
        raise ValueError("Incomplete Sharpa collision mapping")
    meshes, build_times = {}, {}
    device = wp.get_device(device)
    for name, row in rows.items():
        source, derived = (asset_dir / name).resolve(), (contact_dir / row["derived"]).resolve()
        if not source.is_relative_to(asset_dir) or not derived.is_relative_to(contact_dir):
            raise ValueError("Invalid contact asset path")
        if sha256(source) != row["source_sha256"] or sha256(derived) != row["derived_sha256"]:
            raise ValueError("Stale collision mesh hash")
        support = [s for s in row["sdf"] if s["max_resolution"] == resolution]
        if not support or not support[0]["passed"]:
            raise ValueError(f"Unvalidated SDF resolution {resolution}: {name}")
        with np.load(derived, allow_pickle=False) as data:
            mesh = newton.Mesh(data["vertices"], data["indices"].flatten(), compute_inertia=False)
        start = time.perf_counter()
        if device.is_cuda:
            mesh.build_sdf(
                device=device,
                max_resolution=resolution,
                margin=manifest["sdf_margin_m"],
                narrow_band_range=tuple(manifest["sdf_narrow_band_m"]),
                texture_format=manifest["texture_format"],
                sign_method=manifest["sign_method"],
                cache_dir=contact_dir / "sdf-cache",
            )
            wp.synchronize_device(device)
        build_times[name] = time.perf_counter() - start
        meshes[name] = mesh
    builder, hand = build_hand(asset_dir, device=device, parameters=parameters, _mount=_mount)
    refs_by_link = {}
    for ref in manifest["references"]:
        refs_by_link.setdefault(ref["link"], []).append(ref)
    root = ET.parse(asset_dir / "left_sharpa_wave.urdf").getroot()
    count = 0
    shape_mapping = []
    for body, label in enumerate(builder.body_label):
        if _mount is not None and body == 0 and label == "monolithic_support":
            continue
        name = label.rsplit("/", 1)[-1]
        refs = sorted(refs_by_link.get(name, []), key=lambda r: r["ordinal"])
        shapes = [
            i
            for i, b in enumerate(builder.shape_body)
            if b == body and int(builder.shape_flags[i]) & int(ShapeFlags.COLLIDE_PARTICLES)
        ]
        collisions = root.find(f"link[@name='{name}']").findall("collision")
        if len(shapes) != len(refs) or len(refs) != len(collisions):
            raise ValueError(f"Skipped collision shapes on {name}")
        for shape, ref, xml in zip(shapes, refs, collisions, strict=True):
            uri = xml.find("geometry/mesh").get("filename").removeprefix("package://left_sharpa_wave/")
            if ref["source"] != uri:
                raise ValueError("Incorrect collision source mapping")
            origin = xml.find("origin")
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()] if origin is not None else [0, 0, 0]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
            expected = wp.transform(xyz, wp.quat_rpy(*rpy))
            if not np.allclose(builder.shape_transform[shape], expected, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Collision pose changed on {name}")
            if not np.allclose(builder.shape_scale[shape], [1, 1, 1], rtol=0, atol=0):
                raise ValueError("Unexpected Sharpa collision scale")
            builder.shape_source[shape] = meshes[ref["source"]]
            builder.shape_flags[shape] = (
                int(ShapeFlags.COLLIDE_PARTICLES | ShapeFlags.COLLIDE_SHAPES) if device.is_cuda else 0
            )
            shape_mapping.append({**ref, "shape": shape, "body": body, "transform": list(expected)})
            count += 1
    if count != 26:
        raise ValueError("Expected all 26 collision shapes")
    hand.update(
        contact_manifest_sha256=sha256(manifest_path),
        collision_mapping=shape_mapping,
        sdf_resolution=resolution,
        sdf_build_or_cache_seconds=build_times,
        cpu_sdf="NOT_REQUIRED",
        carriage=_mount is not None,
    )
    return builder, hand


def load_hand_ball(
    asset_dir, contact_dir, ball_path, *, device, parameters, position, resolution=128, ball_radius=RADIUS, _mount=None
):
    """Assemble a free soft sphere and the unchanged 22-DoF hand for diagnostics."""
    mesh = load_ball(ball_path, radius=ball_radius)
    builder, manifest = build_contact_hand(
        asset_dir, contact_dir, device=device, parameters=parameters, resolution=resolution, _mount=_mount
    )
    builder.add_soft_mesh(
        pos=position,
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        mesh=mesh,
        particle_radius=0.0002,
        add_surface_mesh_edges=False,
    )
    model, manifest = finalize_hand(builder, manifest, device=device)
    # Builder attaches prebuilt mesh SDFs through shape-collision registration.
    # Disable rigid collision before returning the model or constructing a pipeline.
    flags = model.shape_flags.numpy().copy()
    flags &= ~int(ShapeFlags.COLLIDE_SHAPES)
    model.shape_flags.assign(flags)
    if not np.all(model.particle_inv_mass.numpy() > 0):
        raise ValueError("Soft sphere must have all dynamic nodes")
    manifest.update(
        ball_sha256=sha256(ball_path),
        ball_radius_m=ball_radius,
        ball_position_m=list(position),
        particle_radius_m=0.0002,
        gravity=parameters["gravity"],
        all_ball_nodes_dynamic=True,
    )
    return model, manifest
