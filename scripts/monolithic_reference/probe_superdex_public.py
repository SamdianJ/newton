# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproduce pinned public-API frame/mass/plane limitations without changing physics."""

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from adapter import world_nodes
from manifest import require_reference_worktree


def main():
    """Write independent DRAFT probes; run with the isolated native interpreter."""
    identity = require_reference_worktree(Path.cwd(), expected_sha="54ae749a042709e897cad12da66822c71bcd1b97")
    sys.dont_write_bytecode = True
    import superdex.physics as p  # noqa: PLC0415 - Preserve the guarded source tree on import.

    p.initialize(num_worker_threads=0)
    scene = p.create_scene("reference_public_probe")
    result = {
        "status": "DRAFT",
        "precision": "float64" if p.uses_double_precision() else "float32",
        "superdex_sha": identity.sha,
        "hardware": platform.platform(),
        "num_worker_threads": 0,
    }
    try:
        rest = np.array([[0, 0, 0], [0.04, 0, 0], [0, 0.03, 0], [0, 0, 0.02]], np.float32)
        shape = p.create_tet_mesh_shape(rest.ravel(), np.array([0, 1, 2, 3], np.uint32))
        rotation = [0, 0, 2**-0.5, 2**-0.5]
        translation = np.array([0.3, -0.4, 0.5])
        actor = scene.create_soft_actor(
            name="rotated",
            shape=shape,
            has_gravity=False,
            world_from_local=p.TransformRT(rotation=rotation, translation=translation),
        )
        actor.register_query_and_compute(p.QueryType.NODE_POSITIONS)
        root = actor.get_root_transform()
        measured = world_nodes(
            actor.get_node_positions_local(), np.asarray(root.rotation), np.asarray(root.translation)
        )
        expected = rest[:, [1, 0, 2]].astype(np.float64)
        expected[:, 0] *= -1
        expected += translation
        inputs = {
            "rest_positions_m": rest.tolist(),
            "rotation_xyzw": rotation,
            "translation_m": translation.tolist(),
            "mass_kg": 0.1,
            "inertia_kg_m2": [0.001, 0, 0, 0.001, 0, 0.001],
            "dynamic_plane_normal": [0, 0, 1],
        }
        result["actual_parameters"] = inputs
        result["fixture_sha256"] = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
        error = float(np.max(np.abs(measured - expected)))
        np.testing.assert_allclose(measured, expected, rtol=0, atol=1e-7)
        result["rotated_translated_world_error_m"] = error
        for name, carrier in (("shapeless", None), ("with_carrier", shape)):
            kwargs = {} if carrier is None else {"shape": carrier}
            art = scene.create_articulated_actor(
                name=name,
                joints=[p.ArticulatedJointParams(type=p.ArticulatedJointType.PRISMATIC, axis=[0, 0, 1])],
                links=[
                    p.ArticulatedLinkParams(
                        parent_link=-1,
                        collider_type=p.ColliderType.NONE,
                        mass=0.1,
                        center_of_mass=[0, 0, 0],
                        moment_of_inertia=[0.001, 0, 0, 0.001, 0, 0.001],
                        **kwargs,
                    )
                ],
            )
            link = scene.get_actor(art.get_nested_link_actors()[0])
            result[name] = {
                "mass_kg": link.get_mass(),
                "inertia_kg_m2": np.asarray(link.get_rigid_moment_of_inertia_local()).tolist(),
            }
        np.testing.assert_allclose(result["with_carrier"]["mass_kg"], 0.1, rtol=1e-6)
        if result["shapeless"]["mass_kg"] != 0:
            raise AssertionError("Pinned shapeless behavior changed; re-audit the mapping")
        try:
            scene.create_articulated_actor(
                name="dynamic_plane",
                joints=[p.ArticulatedJointParams(type=p.ArticulatedJointType.PRISMATIC, axis=[0, 0, 1])],
                links=[
                    p.ArticulatedLinkParams(
                        parent_link=-1,
                        shape=p.create_plane_shape([0, 0, 1], 0),
                        mass=0.1,
                        center_of_mass=[0, 0, 0],
                        moment_of_inertia=[0.001, 0, 0, 0.001, 0, 0.001],
                    )
                ],
            )
        except p.Error as error:
            result["dynamic_plane_unmapped"] = str(error)
        else:
            raise AssertionError("Pinned dynamic plane behavior changed; re-audit the mapping")
    finally:
        p.destroy_scene(scene)
        p.shutdown()
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
