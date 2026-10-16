# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build deterministic material acceptance geometry; do not run a simulation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

RECIPE_PATH = Path(__file__).parent / "fixtures/material_acceptance_v1.json"


def build_case(name, *, recipe=RECIPE_PATH):
    """Return the frozen recipe plus rest geometry, masses and constrained node IDs."""
    specification = json.loads(Path(recipe).read_text())
    recipe = specification["cases"][name]
    if recipe["material"] == "cable":
        count = recipe["vertices"]
        positions = np.zeros((count, 3))
        positions[:, 0] = np.linspace(0, recipe["length_m"], count)
        segment_mass = recipe["linear_density_kg_m"] * recipe["length_m"] / (count - 1)
        masses = np.full(count, segment_mass)
        masses[[0, -1]] /= 2
        triangles = np.empty((0, 3), dtype=int)
        fixed = np.array(recipe["fixed_vertices"])
    else:
        n = recipe["cells_per_side"]
        if name == "precreased-box":
            faces = [
                [(-n, -n, 0), (-n, n, 0), (n, n, 0), (n, -n, 0)],
                [(-n, -n, 0), (n, -n, 0), (n, -n, n), (-n, -n, n)],
                [(n, -n, 0), (n, n, 0), (n, n, n), (n, -n, n)],
                [(n, n, 0), (-n, n, 0), (-n, n, n), (n, n, n)],
                [(-n, n, 0), (-n, -n, 0), (-n, -n, n), (-n, n, n)],
            ]
        else:
            faces = [[(-n, -n, 0), (n, -n, 0), (n, n, 0), (-n, n, 0)]]
        vertices, indices = {}, []
        for face in np.array(faces):
            grid = np.empty((n + 1, n + 1), dtype=int)
            for i in range(n + 1):
                for j in range(n + 1):
                    point = tuple(face[0] + i * (face[1] - face[0]) // n + j * (face[3] - face[0]) // n)
                    grid[i, j] = vertices.setdefault(point, len(vertices))
            for i in range(n):
                for j in range(n):
                    a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
                    indices.extend([(a, b, c), (a, c, d)])
        lattice = np.array(list(vertices))
        positions = lattice * [recipe["width_m"] / (2 * n), recipe["depth_m"] / (2 * n), recipe.get("height_m", 0) / n]
        triangles = np.array(indices, dtype=int)
        area = (
            np.linalg.norm(
                np.cross(
                    positions[triangles[:, 1]] - positions[triangles[:, 0]],
                    positions[triangles[:, 2]] - positions[triangles[:, 0]],
                ),
                axis=1,
            )
            / 2
        )
        masses = np.zeros(len(positions))
        np.add.at(masses, triangles.ravel(), np.repeat(recipe["surface_density_kg_m2"] * area / 3, 3))
        fixed = (
            np.flatnonzero(lattice[:, 2] == 0)
            if name == "precreased-box"
            else np.flatnonzero(np.any(np.abs(lattice[:, :2]) == n, axis=1))
        )
    return {
        "recipe": recipe,
        "common": specification["common"],
        "positions": positions,
        "triangles": triangles,
        "masses": masses,
        "fixed": fixed,
    }


def shell_coefficients(recipe):
    """Map the recipe's small-strain plane-stress E/nu/h to membrane and curvature units."""
    young, nu, thickness = recipe["young_modulus_Pa"], recipe["poisson_ratio"], recipe["thickness_m"]
    mu = young * thickness / (2 * (1 + nu))
    lam = young * thickness * nu / (1 - nu**2)
    return lam, mu, lam * thickness**2 / 12, 2 * mu * thickness**2 / 12


def shell_stencils(positions, triangles):
    """Construct oriented six-node stencils and missing-neighbor flags from triangle adjacency."""
    edges = {}
    for face, triangle in enumerate(triangles):
        for opposite in range(3):
            a, b = triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3]
            edges.setdefault(tuple(sorted((a, b))), []).append((face, opposite, int(a), int(b)))
    for neighbors in edges.values():
        if len(neighbors) > 2 or (len(neighbors) == 2 and neighbors[0][2:] != neighbors[1][2:][::-1]):
            raise ValueError("Shell mesh must have consistently oriented manifold edges")
    result = []
    for face, triangle in enumerate(triangles):
        ids, missing = list(triangle), []
        for opposite in range(3):
            edge = tuple(sorted((triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3])))
            neighbors = [entry for entry in edges[edge] if entry[0] != face]
            missing.append(not neighbors)
            ids.append(triangles[neighbors[0][0], neighbors[0][1]] if neighbors else triangle[opposite])
        result.append((positions[ids].copy(), tuple(missing)))
    return result


def main():
    """Print an auditable geometry/recipe description without importing Newton or Warp."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["cable-bend-twist", "cloth-press", "precreased-box"], required=True)
    parser.add_argument("--describe", action="store_true", required=True)
    args = parser.parse_args()
    case = build_case(args.case)
    payload = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in case.items()}
    payload.update(
        status="RECIPE_ONLY_NOT_SIMULATED", recipe_sha256=hashlib.sha256(RECIPE_PATH.read_bytes()).hexdigest()
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
