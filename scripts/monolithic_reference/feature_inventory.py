# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inventory frozen scenes for shapes whose contact changes once capsules, convex meshes and F1/F2 land.

Builds each scene's Model without stepping and reports, per scene:

* ``f1``: link shapes of a type the solver cannot sample, paired with static
  geometry. Today these pairs are skipped silently (contact_sampling.py, A-static).
* ``capsule_convex``: CAPSULE or CONVEX_MESH shapes with COLLIDE_SHAPES or
  COLLIDE_PARTICLES. Today they are skipped or rejected; after PR1 they collide.
* ``particle_flag_without_particles``: COLLIDE_PARTICLES shapes in scenes without
  particles; F2 removes them from the particle-target checks.
* ``same_tree_pairs``: Model pairs inside one articulation (unchanged while
  self-collision is off).

The asset data directory must be unchanged afterwards; the run fails otherwise.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

import newton
from scripts.monolithic_reference.retained_assets import ASSET_ROOT, INPUTS, ROOT

SAMPLED = (newton.GeoType.SPHERE, newton.GeoType.BOX, newton.GeoType.MESH)
NEW_TYPES = (newton.GeoType.CAPSULE, newton.GeoType.CONVEX_MESH)


def _tree(model):
    """Per body: articulation index, whether any moving joint lies on its root path, and FREE single bodies."""
    parent_joint = {int(c): j for j, c in enumerate(model.joint_child.numpy())}
    parents, types = model.joint_parent.numpy(), model.joint_type.numpy()
    articulation = np.full(model.joint_count, -1)
    starts = model.articulation_start.numpy()
    for a in range(model.articulation_count):
        articulation[starts[a] : starts[a + 1]] = a
    moving, owner, free = {}, {}, set()
    for body in range(model.body_count):
        j = parent_joint.get(body)
        owner[body] = int(articulation[j]) if j is not None else -1
        if j is not None and types[j] == newton.JointType.FREE and parents[j] == -1:
            free.add(body)
        any_moving, b = False, body
        while b >= 0 and b in parent_joint:
            jj = parent_joint[b]
            any_moving |= types[jj] != newton.JointType.FIXED
            b = int(parents[jj])
        moving[body] = any_moving
    return moving, owner, free


def inventory(model):
    flags, types, bodies = model.shape_flags.numpy(), model.shape_type.numpy(), model.shape_body.numpy()
    collide = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    particles = (flags & int(newton.ShapeFlags.COLLIDE_PARTICLES)) != 0
    moving, owner, free = _tree(model)
    name = lambda s: newton.GeoType(int(types[s])).name  # noqa: E731
    label = lambda s: model.shape_label[s] if s < len(model.shape_label) else str(s)  # noqa: E731
    dynamic_link = lambda s: bodies[s] >= 0 and moving[int(bodies[s])] and int(bodies[s]) not in free  # noqa: E731
    static = lambda s: bodies[s] < 0 or (not moving[int(bodies[s])] and int(bodies[s]) not in free)  # noqa: E731
    pairs = model.shape_contact_pairs.numpy() if model.shape_contact_pair_count else np.zeros((0, 2), int)
    f1, same_tree = [], 0
    for a, b in map(tuple, pairs):
        if not (collide[a] and collide[b]):
            continue
        if bodies[a] >= 0 and bodies[b] >= 0 and owner[int(bodies[a])] == owner[int(bodies[b])] >= 0:
            same_tree += 1
            continue
        for arm, other in ((a, b), (b, a)):
            if dynamic_link(arm) and static(other) and types[arm] not in SAMPLED:
                f1.append({"link_shape": label(arm), "type": name(arm), "static_shape": label(other)})
    new = [
        {
            "shape": label(s),
            "type": name(s),
            "collide_shapes": bool(collide[s]),
            "collide_particles": bool(particles[s]),
        }
        for s in range(model.shape_count)
        if types[s] in NEW_TYPES and (collide[s] or particles[s])
    ]
    flagged = []
    if model.particle_count == 0:
        flagged = [
            {"shape": label(s), "type": name(s)}
            for s in np.flatnonzero(particles)
            if types[s] not in (newton.GeoType.PLANE, *SAMPLED)
        ]
    return {
        "shapes": int(model.shape_count),
        "particles": int(model.particle_count),
        "types": {
            k: int(v)
            for k, v in zip(*np.unique([name(s) for s in range(model.shape_count)], return_counts=True), strict=True)
        },
        "f1": f1,
        "capsule_convex": new,
        "particle_flag_without_particles": flagged,
        "same_tree_pairs": same_tree,
    }


def _scenes(cache):
    """Scene name -> zero-argument Model builder. Robot scenes need CUDA; GraspIt needs its external asset."""
    import json as _json  # noqa: PLC0415

    def franka():
        from newton.examples.softbody.monolithic_franka import build_model  # noqa: PLC0415

        return build_model(cache / "franka", device="cuda:0")[0]

    def franka_paper():
        from newton.examples.softbody.monolithic_franka import build_model  # noqa: PLC0415

        return build_model(cache / "franka", device="cuda:0", material="paper-box")[0]

    def sharpa():
        from newton.examples.softbody.monolithic_sharpa_recorded import build_recorded  # noqa: PLC0415

        return build_recorded(_json.loads((INPUTS / "sharpa/recorded-control.json").read_text()))[0]

    def rigid64():
        from scripts.monolithic_reference.rigid_stack_scene import stack_scene  # noqa: PLC0415

        pitch = 0.0405
        columns = tuple((pitch * (i - 1.5), pitch * (j - 1.5)) for i in range(4) for j in range(4))
        return stack_scene("cpu", columns=columns, layers=4)

    def remote_tree():
        from scripts.monolithic_reference.run_dexterous_tasks import TaskScene  # noqa: PLC0415

        return TaskScene("turn-knob", remote_tree=True, device="cpu", precision="fp64").model

    def materials(name):
        def build():
            from scripts.monolithic_reference.material_scene import MaterialScene  # noqa: PLC0415

            return MaterialScene(name, device="cpu").model

        return build

    def cloth_stack(recipe, device="cpu"):
        def build():
            from scripts.monolithic_reference.run_cloth_stack import ClothStackScene  # noqa: PLC0415

            fixtures = ROOT / "scripts/monolithic_reference/fixtures"
            config = _json.loads((fixtures / "cloth_stack_point_cloud_v1.json").read_text())
            recipe_data = _json.loads((fixtures / recipe).read_text())
            return ClothStackScene(recipe_data, method="point-cloud", method_config=config, device=device).model

        return build

    def cloth_contact(kind):
        def build():
            from scripts.monolithic_reference.run_cloth_contact import ClothContactScene  # noqa: PLC0415

            fixtures = ROOT / "scripts/monolithic_reference"
            return ClothContactScene(
                kind,
                _json.loads((fixtures / "fixtures/cloth_contact_v1.json").read_text()),
                method="point-cloud",
                method_config=_json.loads((fixtures / "cloth_contact_point_cloud_v1.json").read_text()),
            ).model

        return build

    def hand_bottle():
        asset = os.environ.get("NEWTON_GRASPIT_ASSET_DIR")
        if not asset:
            raise FileNotFoundError("NEWTON_GRASPIT_ASSET_DIR is not set")
        from scripts.monolithic_reference.run_hand_bottle import build_scene  # noqa: PLC0415

        return build_scene(Path(asset), "cuda:0", "fp32")[0]

    return {
        "quick/franka-grasp+release; nightly franka-cube; example franka grasp; admittance": franka,
        "example paper box; admittance paper-box": franka_paper,
        "quick/sharpa-close+lift; nightly sharpa-recorded": sharpa,
        "nightly rigid-stack-64": rigid64,
        "quick/remote-tree": remote_tree,
        "stage materials cable-bend-twist": materials("cable-bend-twist"),
        "stage materials cloth-press": materials("cloth-press"),
        "stage materials precreased-box": materials("precreased-box"),
        "P9b E1 pair": cloth_contact("pair"),
        "P9b E2 self-fold": cloth_contact("self-fold"),
        "P9b E4 sphere": cloth_stack("cloth_stack_sphere_v1.json"),
        "P9b E5 table": cloth_stack("cloth_stack_table_v1.json"),
        "P9b E6 mesh sphere": cloth_stack("cloth_stack_sphere_mesh_v1.json", device="cuda:0"),
        "admittance hand bottle (external GraspIt asset)": hand_bottle,
    }


def _tree_snapshot(path):
    return {str(p.relative_to(path)): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.rglob("*"))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    data = ASSET_ROOT / "data"
    before = _tree_snapshot(data)
    report = {"scenes": {}, "not_run": {}}
    with tempfile.TemporaryDirectory(prefix="p9d-inventory-") as tmp:
        for scene, build in _scenes(Path(tmp)).items():
            try:
                report["scenes"][scene] = inventory(build())
            except Exception as error:  # Report and continue; the scene is NOT_RUN, not clean.
                report["not_run"][scene] = f"{type(error).__name__}: {error}"
    after = _tree_snapshot(data)
    if before != after:
        changed = sorted(set(before.items()) ^ set(after.items()))
        raise RuntimeError(f"Inventory changed the asset data directory: {changed[:5]}")
    report["hits"] = {
        kind: sorted(s for s, r in report["scenes"].items() if r[kind])
        for kind in ("f1", "capsule_convex", "particle_flag_without_particles")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"hits": report["hits"], "not_run": report["not_run"]}, indent=2))


if __name__ == "__main__":
    main()
