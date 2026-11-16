# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Survey same-tree contact pairs of the Allegro hand: sample and column counts, exact gaps at named poses.

Counts use the solver's P1Q3 sampler; gaps use the FP64 convex criteria
(``floating_hand_criteria``), so they are independent of the solver's fields.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.contact_sampling import _surface_samples
from scripts.monolithic_reference import floating_hand_criteria as fc
from scripts.monolithic_reference.floating_hand_scenes import (
    MOUNTS,
    _short,
    add_allegro,
    finger_pose,
    hand_keyframes,
    load_recipe,
    set_finger_coordinates,
)

REST_OVERLAP_LIMIT = -1e-6
PROFILE_EXCLUDE = (("palm_link", "link_13.0"),)


def root_paths(model):
    """Moving DoFs from the world root to each body, in root-to-body order."""
    parent_joint = {int(c): j for j, c in enumerate(model.joint_child.numpy())}
    parents, types, starts = model.joint_parent.numpy(), model.joint_type.numpy(), model.joint_qd_start.numpy()
    paths = {}
    for body in range(model.body_count):
        dofs, b = [], body
        while b >= 0:
            j = parent_joint[b]
            if types[j] != newton.JointType.FIXED:
                dofs = list(range(int(starts[j]), int(starts[j + 1]))) + dofs
            b = int(parents[j])
        paths[body] = dofs
    return paths


def _shared_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def survey(model, poses):
    """Return the per-pair table and totals for the same-tree pairs of ``model``."""
    flags = model.shape_flags.numpy()
    bodies, types, scales = model.shape_body.numpy(), model.shape_type.numpy(), model.shape_scale.numpy()
    collide = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    pairs = model.shape_contact_pairs.numpy() if model.shape_contact_pair_count else np.zeros((0, 2), int)
    paths = root_paths(model)
    labels = [_short(model.body_label[b]) for b in range(model.body_count)]
    cores = {}
    states = {}
    for name, joint_q in poses.items():
        state = model.state()
        state.joint_q.assign(np.asarray(joint_q, dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        states[name] = state.body_q.numpy().astype(np.float64)
    rows, filtered_same_tree = [], 0
    tree_shapes = [s for s in range(model.shape_count) if collide[s] and bodies[s] >= 0]
    in_pairs = {tuple(sorted(map(int, p))) for p in pairs}
    for i, a in enumerate(tree_shapes):
        for b in tree_shapes[i + 1 :]:
            if bodies[a] != bodies[b] and (a, b) not in in_pairs:
                filtered_same_tree += 1
    for a, b in sorted(in_pairs):
        if not (collide[a] and collide[b]) or bodies[a] < 0 or bodies[b] < 0 or bodies[a] == bodies[b]:
            continue
        ba, bb = int(bodies[a]), int(bodies[b])
        pa, pb = paths[ba], paths[bb]
        shared = _shared_prefix(pa, pb)
        k = (len(pa) - shared) + (len(pb) - shared)
        k_unreduced = len(pa) + len(pb)
        samples = {}
        for s in (a, b):
            samples[s] = len(_surface_samples(int(types[s]), scales[s].astype(float), model.shape_source[s])[0])
        for s in (a, b):
            if s not in cores:
                cores[s] = fc.model_core(model, s)
        gaps = {
            pose: fc.signed_distance(fc.world_core(cores[a], q[ba]), fc.world_core(cores[b], q[bb]))
            for pose, q in states.items()
        }
        excluded = tuple(sorted((labels[ba], labels[bb]))) in {tuple(sorted(p)) for p in PROFILE_EXCLUDE}
        rows.append(
            {
                "shapes": [int(a), int(b)],
                "bodies": [labels[ba], labels[bb]],
                "types": [newton.GeoType(int(types[a])).name, newton.GeoType(int(types[b])).name],
                "directed_samples": samples[a] + samples[b],
                "columns_lca": k,
                "columns_unreduced": k_unreduced,
                "gap_m": gaps,
                "profile_excluded": excluded,
            }
        )
    totals = {
        "same_tree_pairs": len(rows),
        "same_tree_pairs_filtered_by_model": filtered_same_tree,
        "directed_samples": sum(r["directed_samples"] for r in rows),
        "factor_demand_lca": sum(r["directed_samples"] * r["columns_lca"] for r in rows),
        "triplet_demand_lca": sum(r["directed_samples"] * r["columns_lca"] ** 2 for r in rows),
        "triplet_demand_unreduced": sum(r["directed_samples"] * r["columns_unreduced"] ** 2 for r in rows),
        "rest_overlaps": [
            {"bodies": r["bodies"], "gap_m": r["gap_m"]["rest"], "profile_excluded": r["profile_excluded"]}
            for r in rows
            if r["gap_m"].get("rest", 0.0) < REST_OVERLAP_LIMIT
        ],
    }
    for pose in states:
        totals[f"overlapping_pairs_{pose}"] = sum(r["gap_m"][pose] < 0 for r in rows)
    return {"pairs": rows, "totals": totals}


def build(mount, recipes=()):
    """Rest, open and fist poses, plus each grasp recipe's open-at-grasp and squeezed closed finger targets."""
    builder = newton.ModelBuilder()
    layout = add_allegro(builder, mount=mount, palm_pose=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    model = builder.finalize(device="cpu")
    rest = model.joint_q.numpy().astype(np.float64)
    poses = {
        "rest": rest,
        "open": set_finger_coordinates(model, rest, finger_pose(model, layout, "open")),
        "fist": set_finger_coordinates(model, rest, finger_pose(model, layout, "fist")),
    }
    for path in recipes:
        recipe = load_recipe(path)
        names, _, open_, closed = hand_keyframes(model, recipe)
        for label, values in (("open_at_grasp", open_), ("closed", closed)):
            fingers = dict(zip(names[6:], values[6:], strict=True))
            poses[f"{recipe['scene']}_{label}"] = set_finger_coordinates(model, rest, fingers)
    return model, poses


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mount", choices=MOUNTS, default="wrist")
    parser.add_argument(
        "--recipe", type=Path, action="append", default=[], help="grasp recipe whose finger targets to add"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    model, poses = build(args.mount, args.recipe)
    result = survey(model, poses)
    result["mount"] = args.mount
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["totals"], indent=2))


if __name__ == "__main__":
    main()
