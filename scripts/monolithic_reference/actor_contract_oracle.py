# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent NumPy references and short calibration; never run a solver trajectory."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.articulation_oracle import skew
from scripts.monolithic_reference.tet_oracle import material

CASE = Path(__file__).with_name("fixtures") / "actor_coupling_v1.json"


def rotation(v):
    """Evaluate a world-frame rotation vector using Rodrigues' formula."""
    angle = np.linalg.norm(v)
    k = skew(np.asarray(v))
    if angle < 1e-8:
        return np.eye(3) + k + 0.5 * k @ k
    return np.eye(3) + np.sin(angle) / angle * k + (1 - np.cos(angle)) / angle**2 * k @ k


def rigid_terms(com, rot, predicted_com, predicted_rot, mass, inertia, dt, force, *, initial_com=None):
    """Return conservative COM/rotation inertia and the independently projected PSD B."""
    second = np.trace(inertia) / 2 * np.eye(3) - inertia
    delta, dr = com - predicted_com, rot - predicted_rot
    initial_com = predicted_com if initial_com is None else initial_com
    energy = (mass * delta @ delta + np.sum((dr @ second) * dr)) / (2 * dt**2) - force @ (com - initial_com)
    a = dr @ second @ rot.T / dt**2
    residual = np.r_[mass * delta / dt**2 - force, [a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]]]
    c = rot @ second @ predicted_rot.T
    raw = (np.trace(c) * np.eye(3) - (c + c.T) / 2) / dt**2
    values, vectors = np.linalg.eigh(raw)
    b = np.zeros((6, 6))
    b[:3, :3] = mass / dt**2 * np.eye(3)
    b[3:, 3:] = (vectors * np.maximum(values, 0)) @ vectors.T
    return energy, residual, b


def sphere_samples(radius, subdivisions=32):
    """Project P1Q3 points to a UV sphere and use exact spherical triangle areas."""
    theta = np.linspace(0, np.pi, subdivisions + 1)
    phi = np.arange(subdivisions) * (2 * np.pi / subdivisions)
    grid = np.stack(
        (
            np.sin(theta[:, None]) * np.cos(phi),
            np.sin(theta[:, None]) * np.sin(phi),
            np.broadcast_to(np.cos(theta[:, None]), (subdivisions + 1, subdivisions)),
        ),
        axis=-1,
    )
    triangles = []
    for i in range(subdivisions):
        for j in range(subdivisions):
            a, b, c, d = (
                grid[i, j],
                grid[i, (j + 1) % subdivisions],
                grid[i + 1, j],
                grid[i + 1, (j + 1) % subdivisions],
            )
            if i > 0:
                triangles.append((a, b, c))
            if i < subdivisions - 1:
                triangles.append((b, d, c))
    tri = np.asarray(triangles)
    u, v, w = tri[:, 0], tri[:, 1], tri[:, 2]
    area = (
        2
        * radius**2
        * np.arctan2(np.abs(np.einsum("ij,ij->i", u, np.cross(v, w))), 1 + np.sum(u * v + v * w + w * u, axis=1))
    )
    bary = np.eye(3) * 0.5 + np.full((3, 3), 1 / 6)
    points = np.einsum("sn,fni->fsi", bary, tri)
    points *= radius / np.linalg.norm(points, axis=-1)[..., None]
    return points.reshape(-1, 3), np.repeat(area / 3, 3)


def calibration():
    """Check immutable accuracy budgets with short independent physical fixtures."""
    case = json.loads(CASE.read_text())
    dt, mass = case["dt"], case["rigid_mass_kg"]
    dimensions = np.array(case["box_dimensions_m"])
    inertia = mass / 12 * np.diag(np.sum(dimensions**2) - dimensions**2)
    rot = rotation(np.array([0.21, -0.12, 0.31]))
    predicted = rotation(np.array([0.2, -0.1, 0.3]))
    com, cp, force = np.array([0.002, -0.001, 0.0015]), np.zeros(3), np.array([0.1, -0.2, 0.15])
    _, residual, b = rigid_terms(com, rot, cp, predicted, mass, inertia, dt, force)
    numerical = []
    step = 1e-7
    for i in range(6):
        perturb = np.eye(6)[i] * step
        upper = rigid_terms(com + perturb[:3], rotation(perturb[3:]) @ rot, cp, predicted, mass, inertia, dt, force)[0]
        lower = rigid_terms(com - perturb[:3], rotation(-perturb[3:]) @ rot, cp, predicted, mass, inertia, dt, force)[0]
        numerical.append((upper - lower) / (2 * step))
    derivative_error = np.max(np.abs(residual - numerical)) / max(1, np.max(np.abs(residual)))
    # Linear impulse fixture includes anisotropic inertia rotated out of its principal frame.
    world = rot @ inertia @ rot.T
    operator = np.zeros((6, 6))
    operator[:3, :3], operator[3:, 3:] = mass / dt**2 * np.eye(3), world / dt**2
    load = np.array([0.1, -0.2, 0.15, 0.0002, -0.0003, 0.0001])
    scale = 1 / np.sqrt(np.diag(operator))
    scaled = operator * scale[:, None] * scale[None, :]
    y = np.linalg.solve(scaled.astype(np.float32), (scale * load).astype(np.float32)).astype(float)
    posterior = scale * (operator @ (scale * y) - load)
    translation_noise, rotation_noise = (
        np.linalg.norm(posterior[:3]) / np.sqrt(3),
        np.linalg.norm(posterior[3:]) / np.sqrt(3),
    )
    # Sphere against plane: integrate the unsmoothed penalty over a cap analytically.
    radius, penetration = 0.02, 0.002
    points, weights = sphere_samples(radius)
    sampled = np.sum(weights * np.maximum(penetration - radius - points[:, 2], 0))
    exact = np.pi * radius * penetration**2
    sampling_error = abs(sampled / exact - 1)
    cap_errors, moment_errors = [], []
    for depth in (0.001, 0.002, 0.004):
        for angle in (0, 0.37, 0.81):
            rotated = points @ rotation(np.array([angle, angle * 0.6, 0.2])).T
            loads = weights * np.maximum(depth - radius - rotated[:, 2], 0)
            reference = np.pi * radius * depth**2
            cap_errors.append(abs(loads.sum() / reference - 1))
            moment = np.sum(np.cross(rotated, np.array([0, 0, 1])) * loads[:, None], axis=0)
            moment_errors.append(np.linalg.norm(moment) / (radius * reference))
    # Homogeneous, laterally constrained one-cell compression, not an E2E prediction.
    young, poisson = case["young_modulus_pa"], case["poisson_ratio"]
    mu, lam = young / (2 * (1 + poisson)), young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    edge, area = case["tet_edge_m"], dimensions[1] * dimensions[2]

    def compression_force(displacement):
        _, stress, _ = material(np.diag([1 - displacement / edge, 1, 1]), mu, lam, True)
        return -area * stress[0, 0]

    lo, hi = 0.0, case["pusher_travel_m"]
    for _ in range(60):
        mid = (lo + hi) / 2
        f = compression_force(mid)
        total = mid + f / case["pusher_pd"]["ke"] + 3 * f / (case["contact_stiffness"] * area)
        if total < case["pusher_travel_m"]:
            lo = mid
        else:
            hi = mid
    compression = (lo + hi) / 2
    chain_force = compression_force(compression)
    penetration_estimate = chain_force / (case["contact_stiffness"] * area)
    metrics = {
        "rigid_derivative_relative_error": float(derivative_error),
        "rigid_psd_min_eigenvalue": float(np.linalg.eigvalsh(b).min()),
        "translation_fp32_posterior_rms": float(translation_noise),
        "rotation_fp32_posterior_rms": float(rotation_noise),
        "sphere_area_relative_error": float(abs(weights.sum() / (4 * np.pi * radius**2) - 1)),
        "sphere_plane_force_relative_error": float(sampling_error),
        "sphere_rotated_caps_max_force_relative_error": float(max(cap_errors)),
        "sphere_rotated_caps_max_moment_relative_error": float(max(moment_errors)),
        "compression_fixture_displacement_m": float(compression),
        "compression_fixture_force_n": float(chain_force),
        "compression_fixture_penetration_m": float(penetration_estimate),
    }
    checks = {
        "derivative": derivative_error < 1e-8,
        "PSD": np.linalg.eigvalsh(b).min() >= 0,
        "symmetric_B": np.allclose(b, b.T, rtol=1e-12, atol=1e-12),
        "translation_noise": translation_noise < case["rigid_goals"]["translation_absolute"] / 10,
        "rotation_noise": rotation_noise < case["rigid_goals"]["rotation_absolute"] / 10,
        "sphere_area": abs(weights.sum() / (4 * np.pi * radius**2) - 1) < 1e-12,
        "sampling": sampling_error < case["gates"]["sampling_relative_error"],
        "rotated_sampling": max(cap_errors + moment_errors) < case["gates"]["sampling_relative_error"],
        "loaded_fixture": compression > case["gates"]["hold_min_tet_displacement_m"]
        and chain_force > case["gates"]["hold_min_pair_force_n"],
        "penetration_budget": penetration_estimate + 2 * case["normal_smoothing_m"]
        < case["gates"]["max_penetration_m"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "case_sha256": hashlib.sha256(CASE.read_bytes()).hexdigest(),
        "metrics": metrics,
        "checks": {k: bool(v) for k, v in checks.items()},
        "scope": "independent short fixtures; no mixed-actor trajectory certification",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = calibration()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
