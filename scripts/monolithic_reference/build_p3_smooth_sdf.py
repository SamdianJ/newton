# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build FP64 Hermite jets or natural cubic B-spline coefficients offline."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from newton._src.solvers.monolithic.smooth_sdf import SmoothSDF, SmoothSDFSet


def build_coefficients(values, spacing, representation):
    """Preserve node values on a z/y/x grid without changing its resolution."""
    values = np.asarray(values, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=np.float64)
    if values.ndim != 3 or min(values.shape) < 3 or not np.isfinite(values).all():
        raise ValueError("Require a finite 3D grid with at least three nodes per axis")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Require positive finite x/y/z spacing")
    if representation == "c1-hermite":
        jets = [values]
        for mask in range(1, 8):
            axis = (mask & -mask).bit_length() - 1
            jets.append(np.gradient(jets[mask ^ (1 << axis)], spacing[axis], axis=2 - axis, edge_order=2))
        return np.stack(jets, axis=-1)
    if representation != "c2-bspline":
        raise ValueError("Unknown smooth SDF representation")
    result = values.copy()
    for axis in range(3):
        rhs = np.moveaxis(result, axis, 0).copy()
        count = len(rhs)
        diagonal = np.full(count, 4.0)
        diagonal[[0, -1]] = 1.0
        rhs[1:-1] *= 6.0
        # Natural ends imply c_0=f_0 and c_(n-1)=f_(n-1); interior rows are 1,4,1.
        for i in range(1, count - 1):
            factor = 1.0 / diagonal[i - 1]
            diagonal[i] -= factor if i > 1 else 0.0
            rhs[i] -= factor * rhs[i - 1]
        for i in range(count - 2, 0, -1):
            rhs[i] = (rhs[i] - rhs[i + 1]) / diagonal[i]
        result = np.moveaxis(rhs, 0, axis)
    for axis in range(3):
        data = np.moveaxis(result, axis, 0)
        data = np.concatenate(((2 * data[0] - data[1])[None], data, (2 * data[-1] - data[-2])[None]), axis=0)
        result = np.moveaxis(data, 0, axis)
    return np.ascontiguousarray(result[..., None])


def load_qualified_fields(model, directory, representation):
    """Bind the frozen common nodes to a diagnostic solver, deduplicated by SDF index."""

    directory = Path(directory)
    begin = time.perf_counter()
    certificate = json.loads((directory / "candidate-qualification.json").read_text())
    if not certificate["passed"].get(representation, False):
        raise ValueError("Candidate has not passed the complete geometry and derivative checks")
    if certificate["source_textures"] != texture_identity(model):
        raise ValueError("Qualified SDF source textures do not match model")
    summary = json.loads((directory / "summary.json").read_text())
    if not summary["geometry_qualified"].get(representation, False):
        raise ValueError("Candidate has not passed the frozen geometry qualification")
    rows = json.loads((directory / "geometry-results.json").read_text())
    table = model._texture_sdf_data.numpy()
    if len(rows) != len(table):
        raise ValueError("Qualified SDF table does not match model")
    fields = []
    for index, row in enumerate(rows):
        path = directory / f"{index:02d}-{Path(row['source']).stem}" / "nodes.npz"
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != row["node_sha256"]:
                raise ValueError("Qualified node field changed")
        sdf = table[index]
        if (
            row["sdf_index"] != index
            or not np.array_equal(row["lower"], sdf["sdf_box_lower"])
            or not np.array_equal(row["upper"], sdf["sdf_box_upper"])
            or not np.array_equal(row["original_inv_dx"], sdf["inv_sdf_dx"])
        ):
            raise ValueError("Qualified SDF coordinates do not match model")
        with np.load(path) as nodes:
            values = nodes["values"]
            coefficients = build_coefficients(
                values, (np.asarray(row["upper"]) - row["lower"]) / (np.asarray(row["dims"]) - 1), representation
            )
        field = SmoothSDF(coefficients, row["lower"], row["upper"], row["dims"], representation, device=model.device)
        if field.identity != row["candidates"][representation]["identity"]:
            raise ValueError("Qualified coefficients do not reproduce exactly")
        fields.append(field)
    result = SmoothSDFSet(
        fields,
        source_identity={
            "nodes": [row["node_sha256"] for row in rows],
            "scale": model.shape_scale.numpy().tolist(),
            "scale_baked": table["scale_baked"].tolist(),
            "source_textures": certificate["source_textures"],
        },
    )
    result.build_seconds = time.perf_counter() - begin
    return result


def texture_identity(model):
    """Bind actual source texels and slot tables, independently of CUDA handles."""
    hashes = {}
    for name in ("_texture_sdf_coarse_textures", "_texture_sdf_subgrid_textures"):
        for index, texture in enumerate(getattr(model, name)):
            shape = (texture.depth, texture.height, texture.width)
            if texture.num_channels != 1:
                shape = (*shape, texture.num_channels)
            pixels = np.empty(shape, dtype=np.float32)
            texture.copy_to(pixels)
            hashes[f"{name}.{index}"] = hashlib.sha256(pixels.tobytes()).hexdigest()
    for index, slots in enumerate(model._texture_sdf_subgrid_start_slots):
        hashes[f"slots.{index}"] = hashlib.sha256(slots.numpy().tobytes()).hexdigest()
    return hashes
