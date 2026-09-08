# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate frozen PR-6D volume assets; no tetrahedralizer runtime dependency."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_soft_ball import generate_ball, validate_ball


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for refinement in (1, 2, 3):
        mesh = generate_ball(refinement)
        path = args.output / f"ball_r{refinement}.npz"
        mesh.save(str(path))
        rows.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "refinement": refinement,
                **validate_ball(mesh.vertices, mesh.tet_indices.reshape(-1, 4)),
            }
        )
    manifest = {
        "schema": "monolithic-soft-ball/v1",
        "units": "m, kg, s",
        "generator": "spherified-cube/Freudenthal-v1",
        "numpy_version": np.__version__,
        "seed": None,
        "parameter_source": "simulation_config",
        "material_model": "smith_log_stabilized",
        "mass_mode": "consistent",
        "density_kg_m3": 1000.0,
        "young_modulus_pa": 10000.0,
        "poisson_ratio": 0.3,
        "damping": 0.0,
        "refinement_scope": "boundary and interior refined together; coarse geometry is not canonical",
        "assets": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
