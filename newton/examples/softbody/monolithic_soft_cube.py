# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed 40 mm Freudenthal volume assets for PR8H."""

import itertools

import numpy as np

import newton
from newton.examples.softbody.monolithic_soft_ball import (
    DENSITY,
    POISSON_RATIO,
    YOUNG_MODULUS,
    _grid_tets,
    _validate_volume,
)
from newton.examples.softbody.sharpa_close import sha256

EDGE = 0.040
CELLS = (4, 6, 8)


def generate_cube(refinement):
    """Generate one of the three approved grids without a geometry mapping."""
    if isinstance(refinement, bool) or not isinstance(refinement, int) or refinement not in (1, 2, 3):
        raise ValueError("Cube refinement must be 1, 2, or 3")
    n = CELLS[refinement - 1]
    points = np.asarray(list(itertools.product(np.linspace(-EDGE / 2, EDGE / 2, n + 1), repeat=3)), dtype=np.float32)
    tets = _grid_tets(n)
    validate_cube(points, tets)
    mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
    lam = YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
    return newton.TetMesh(points, tets.flatten(), density=DENSITY, k_mu=mu, k_lambda=lam, k_damp=0.0)


def validate_cube(vertices, tets):
    """Require the frozen geometry, topology, mass and conforming grid."""
    report, surface = _validate_volume(vertices, tets)
    x = np.asarray(vertices, dtype=np.float64)
    if np.any(np.abs(x) > EDGE / 2 + 1e-9) or not np.allclose(np.max(np.abs(x[surface]), axis=1), EDGE / 2, rtol=1e-6):
        raise ValueError("Cube boundary mismatch")
    n = next((n for n in CELLS if (n + 1) ** 3 == len(x)), None)
    if n is None or not np.array_equal(np.asarray(tets), _grid_tets(n)):
        raise ValueError("Cube grid connectivity mismatch")
    expected = np.asarray(list(itertools.product(np.linspace(-EDGE / 2, EDGE / 2, n + 1), repeat=3)))
    if not np.allclose(x, expected, rtol=1e-6, atol=1e-10) or not np.isclose(report["volume_m3"], EDGE**3, rtol=1e-6):
        raise ValueError("Cube grid or volume mismatch")
    return {**report, "object_type": "cube", "edge_m": EDGE, "cells_per_axis": n, "refinement": CELLS.index(n) + 1}


def load_cube(path):
    """Reject changed geometry/materials before constructing the solver."""
    with np.load(path, allow_pickle=False) as data:
        validate_cube(data["vertices"], data["tet_indices"].reshape(-1, 4))
        mu = YOUNG_MODULUS / (2 * (1 + POISSON_RATIO))
        lam = YOUNG_MODULUS * POISSON_RATIO / ((1 + POISSON_RATIO) * (1 - 2 * POISSON_RATIO))
        for name, expected in (("density", DENSITY), ("k_mu", mu), ("k_lambda", lam), ("k_damp", 0)):
            if name not in data or not np.allclose(data[name], expected, rtol=1e-6, atol=0):
                raise ValueError(f"Frozen cube material mismatch: {name}")
    return newton.TetMesh.create_from_file(str(path))


def validate_cube_calibration(result, path, *, dt):
    """Bind a complete physical envelope to this cube, material and timestep."""

    mesh = load_cube(path)
    report = validate_cube(mesh.vertices, mesh.tet_indices.reshape(-1, 4))
    protocol = result.get("protocol", {})
    expected = {
        "schema": "sharpa-cube-contact-calibration/v1",
        "object_type": "cube",
        "edge_m": EDGE,
        "cells_per_axis": list(CELLS),
        "dt": dt,
        "material_model": "smith_log_stabilized",
        "mass_mode": "consistent",
        "density": DENSITY,
        "young_modulus": YOUNG_MODULUS,
        "poisson_ratio": POISSON_RATIO,
        "mu": 0.5,
        "gates": {"penetration_m": 0.002, "min_det_f": 0.1, "converged_fraction": 0.99, "force_error": 0.05},
        "ramp_seconds": 0.5,
        "hold_seconds": 1.0,
        "measurement_seconds": 0.2,
        "gravity": [0, 0, 0],
        "tangential_stiffness": 2e6,
        "smoothing_width": 0.0001,
        "gap": 0.002,
        "particle_radius": 0.0002,
        "plate_mass": 0.01,
        "plate_velocity_gain": 10.0,
        "stiffness_scan_n_m3": [5e5, 1e6, 2e6, 5e6, 1e7, 2e7, 5e7, 1e8],
    }
    if result.get("status") != "FROZEN" or any(protocol.get(k) != v for k, v in expected.items()):
        raise ValueError("Cube calibration physical binding mismatch")
    if protocol.get("assets", {}).get(str(report["refinement"])) != sha256(path):
        raise ValueError("Cube calibration asset hash mismatch")
    for row in result["results"]:
        if row.get("cube_sha256") != protocol["assets"].get(str(row["refinement"])):
            raise ValueError("Cube calibration row asset mismatch")
