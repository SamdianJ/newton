# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build FP64 Hermite jets or natural cubic B-spline coefficients offline."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from ...geometry.sdf_texture import TextureSDFData
from .contact_geometry import _mesh64
from .smooth_sdf import SmoothSDF, SmoothSDFSet, sample_smooth


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
    result.build_metadata = {
        "source": "qualified_directory",
        "directory": str(directory.resolve()),
        "qualified": True,
        "checks": rows,
    }
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


@wp.kernel
def _nodes(table: wp.array[TextureSDFData], sdf_index: int, dims: wp.vec3i, output: wp.array[wp.float64]):
    i = wp.tid()
    index = wp.vec3d(wp.float64(i % dims[0]), wp.float64((i / dims[0]) % dims[1]), wp.float64(i / (dims[0] * dims[1])))
    sdf = table[sdf_index]
    lower, upper = wp.vec3d(sdf.sdf_box_lower), wp.vec3d(sdf.sdf_box_upper)
    dx = wp.cw_div(upper - lower, wp.vec3d(dims - wp.vec3i(1)))
    value, _gradient = _mesh64(sdf, lower + wp.cw_mul(index, dx))
    output[i] = value


def sample(field, points):
    points = wp.array(points, dtype=wp.vec3d, device=field.device)
    output = wp.empty((len(points), 13), dtype=wp.float64, device=field.device)
    wp.launch(sample_smooth, len(points), [field.data, points, output], device=field.device)
    return output.numpy()


def _sample_fields(model, representation):
    """Reconstruct C2 fields from the actual source nodes; report component checks, not certification."""
    table = model._texture_sdf_data.numpy()
    fields, reports = [], []
    for i, sdf in enumerate(table):
        dims = np.array(model._texture_sdf_subgrid_start_slots[i].shape) * sdf["subgrid_size"] + 1
        lower, upper = sdf["sdf_box_lower"].astype(float), sdf["sdf_box_upper"].astype(float)
        nodes = wp.empty(int(np.prod(dims)), dtype=wp.float64, device=model.device)
        wp.launch(_nodes, nodes.size, [model._texture_sdf_data, i, wp.vec3i(dims), nodes], device=model.device)
        values = nodes.numpy().reshape(tuple(dims[::-1]))
        dx = (upper - lower) / (dims - 1)
        coefficients = build_coefficients(values, dx, representation)
        field = SmoothSDF(coefficients, lower, upper, dims, representation, device=model.device)
        rng = np.random.default_rng(1701 + i)
        indices = np.column_stack([rng.integers(2, int(d) - 2, 32) for d in dims])
        points = lower + indices * dx
        got = sample(field, points)
        expected = values[indices[:, 2], indices[:, 1], indices[:, 0]]
        node_error = float(abs(got[:, 0] - expected).max())
        fd = []
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = dx[axis] * 1e-4
            numerical = (sample(field, points + offset)[:, 0] - sample(field, points - offset)[:, 0]) / (
                2 * offset[axis]
            )
            fd.append(float(abs(numerical - got[:, axis + 1]).max()))
        if node_error > 1e-10 or max(fd) > 1e-5 or not np.isfinite(got).all():
            raise ValueError("Smooth SDF node/gradient check failed")
        reports.append({"index": i, "identity": field.identity, "node_error": node_error, "gradient_fd_error": fd})
        fields.append(field)
    result = SmoothSDFSet(
        fields,
        source_identity={"source_textures": texture_identity(model), "scale": model.shape_scale.numpy().tolist()},
    )
    result.build_metadata = {"source": "model", "qualified": False, "checks": reports}
    return result


def build_smooth_sdf(model, *, representation="c2-bspline", qualified_directory=None) -> SmoothSDFSet:
    """Build smooth fields from the model's existing mesh SDF textures.

    .. experimental::

    Args:
        model: Model with attached mesh SDF textures on its device.
        representation: ``c2-bspline`` or ``c1-hermite`` interpolation.
        qualified_directory: Frozen nodes and qualification certificates. Any
            validation failure raises without falling back to model sampling.

    Returns:
        Fields bound to the model's textures. ``build_metadata`` records source,
        checks and qualification separately from the unchanged physical identity.
        Model sampling runs component checks but does not certify geometry.
    """
    if representation not in ("c2-bspline", "c1-hermite"):
        raise ValueError("Unknown smooth SDF representation")
    if qualified_directory is not None:
        return load_qualified_fields(model, qualified_directory, representation)
    return _sample_fields(model, representation)
