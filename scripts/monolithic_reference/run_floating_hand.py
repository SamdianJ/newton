# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one floating-hand recipe scene and record every published step for the offline criteria.

    python -m scripts.monolithic_reference.run_floating_hand E1a \\
        --recipe scripts/monolithic_reference/fixtures/floating_hand_e1a_capsule_v1.json \\
        --device cuda:0 --precision fp32 --use-cuda-graph --output <new directory>

Each published step is CONVERGED or STOPPED_ACCEPTED; a rollback is a HARD_FAILURE
and ends the run (the clock does not advance past it). STOPPED is never counted
as converged. ``model.json`` holds the geometry and inertia the criteria need, so
``floating_hand_criteria`` never rebuilds the solver.
"""

import argparse
import hashlib
import importlib.metadata
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference.cloth_contact_run import _memory, publication, write_json
from scripts.monolithic_reference.floating_hand_scenes import (
    MOUNTS,
    SCENES,
    FloatingHandScene,
    convex_hull_asset,
    load_recipe,
)

SOURCE = Path(__file__).resolve().parents[2]
CHUNK = 500
STATUS_CODE = {"CONVERGED": 0, "STOPPED_ACCEPTED": 1, "HARD_FAILURE": 2}


def _git(*command):
    return subprocess.check_output(["git", *command], cwd=SOURCE, text=True).strip()


def model_record(scene):
    """Geometry, inertia and pair sets of the scene Model, in plain JSON."""
    model = scene.model
    flags, bodies = model.shape_flags.numpy(), model.shape_body.numpy()
    collide = (flags & int(newton.ShapeFlags.COLLIDE_SHAPES)) != 0
    shapes = []
    for s in np.flatnonzero(collide):
        geo = newton.GeoType(int(model.shape_type.numpy()[s]))
        row = {
            "index": int(s),
            "label": model.shape_label[s],
            "body": int(bodies[s]),
            "type": geo.name,
            "scale": model.shape_scale.numpy()[s].tolist(),
            "transform": model.shape_transform.numpy()[s].tolist(),
        }
        if geo == newton.GeoType.CONVEX_MESH:
            row["hull_points"] = convex_hull_asset(scene.recipe["object"])[0].tolist()
        shapes.append(row)
    articulation = np.full(model.joint_count, -1)
    starts = model.articulation_start.numpy()
    for a in range(model.articulation_count):
        articulation[starts[a] : starts[a + 1]] = a
    body_tree = {int(c): int(articulation[j]) for j, c in enumerate(model.joint_child.numpy())}
    pairs = model.shape_contact_pairs.numpy() if model.shape_contact_pair_count else np.zeros((0, 2), int)
    same_tree = [
        [int(a), int(b)]
        for a, b in pairs
        if collide[a]
        and collide[b]
        and bodies[a] >= 0
        and bodies[b] >= 0
        and bodies[a] != bodies[b]
        and body_tree[int(bodies[a])] == body_tree[int(bodies[b])]
    ]
    labels = [model.body_label[b].rsplit("/", 1)[-1] for b in range(model.body_count)]
    return {
        "body_labels": labels,
        "body_mass": model.body_mass.numpy().tolist(),
        "body_com": model.body_com.numpy().tolist(),
        "body_inertia": model.body_inertia.numpy().tolist(),
        "joint_labels": [model.joint_label[j].rsplit("/", 1)[-1] for j in range(model.joint_count)],
        "joint_type": model.joint_type.numpy().tolist(),
        "joint_q_start": model.joint_q_start.numpy().tolist(),
        "joint_qd_start": model.joint_qd_start.numpy().tolist(),
        "gravity": list(map(float, scene.recipe.get("gravity", (0.0, 0.0, -9.81)))),
        "shapes": shapes,
        "same_tree_pairs": same_tree,
        "object_body": scene.object_body,
        "palm_body": getattr(scene, "palm_body", None),
        "hand_bodies": sorted(
            b for n, b in (scene.layout or {}).get("bodies", {}).items() if not n.startswith("wrist_")
        ),
        "table": scene.recipe.get("table"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", choices=SCENES)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp32")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--mount", choices=MOUNTS)
    parser.add_argument("--self-collision", choices=("recipe", "off"), default="recipe")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if Path(newton.__file__).resolve() != SOURCE / "newton/__init__.py":
        raise RuntimeError(f"Newton import {newton.__file__} does not belong to the tested worktree {SOURCE}")
    recipe = load_recipe(args.recipe)
    if recipe["scene"] != args.scene:
        raise ValueError(f"Scene argument {args.scene} differs from the recipe scene {recipe['scene']}")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError(f"--max-steps must be positive, got {args.max_steps}")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "dirty.patch").write_text(_git("diff", "--binary", "HEAD"))
    files = sorted((SOURCE / "newton/_src/solvers/monolithic").glob("*.py")) + sorted(
        Path(__file__).parent.glob("*.py")
    )
    manifest = {
        "schema_version": 1,
        "status": "STARTING",
        "source_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--porcelain"),
        "arguments": vars(args),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "source_files_sha256": {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warp": wp.__version__,
            "newton": newton.__file__,
            "scipy": importlib.metadata.version("scipy"),
            "trimesh": importlib.metadata.version("trimesh"),
        },
    }
    write_json(args.output / "manifest.json", manifest)
    counts = dict.fromkeys(STATUS_CODE, 0)
    recorded, planned = 0, None
    stopped, frames, chunks, error = [], [], [], None

    def flush():
        if frames:
            path = args.output / f"states-{len(chunks):04d}.npz"
            np.savez_compressed(path, **{k: np.stack([f[k] for f in frames]) for k in frames[0]})
            chunks.append(path.name)
            frames.clear()

    start = time.perf_counter()
    try:
        scene = FloatingHandScene(
            recipe,
            device=args.device,
            precision=args.precision,
            use_cuda_graph=args.use_cuda_graph,
            mount=args.mount,
            self_collision=args.self_collision,
        )
        record = model_record(scene)
        write_json(args.output / "model.json", record)
        others = [s["index"] for s in record["shapes"] if s["body"] != record["object_body"]]
        object_shapes = {s["index"] for s in record["shapes"] if s["body"] == record["object_body"]}
        stop = min(args.max_steps or scene.total_steps, scene.total_steps)
        planned = scene.total_steps
        manifest.update(
            status="RUNNING",
            planned_steps=scene.total_steps,
            requested_steps=stop,
            dt_s=scene.dt,
            construction_seconds=time.perf_counter() - start,
            object_pair_shapes=others,
        )
        write_json(args.output / "manifest.json", manifest)
        rollout = time.perf_counter()
        while scene.steps < stop:
            previous, step = scene.solver.last_stats.step_generation, scene.steps
            try:
                scene.step()
            except NotImplementedError:
                raise
            except Exception as failure:  # The step failed outright: record and stop.
                error = f"step {step}: {type(failure).__name__}: {failure}"
                counts["HARD_FAILURE"] += 1
                break
            stats = scene.solver.last_stats
            forces = scene.solver.get_contact_forces(scene.state) if not stats.rolled_back else None
            label = "HARD_FAILURE" if forces is None else publication(stats, forces, previous_generation=previous)
            counts[label] += 1
            if label == "HARD_FAILURE":
                error = f"step {step}: {stats.status} {stats.failure_reason}"
                break
            if label == "STOPPED_ACCEPTED":
                stopped.append({"step": step, "status": str(stats.status), "reason": stats.failure_reason})
            pair_force = np.zeros((len(others), 3))
            for pair in (stats.variational or {}).get("actor_contact_pairs", ()):
                a, b = pair["source"], pair["target"]
                for end, (mine, other) in enumerate(((a, b), (b, a))):
                    if mine in object_shapes and other in others:
                        pair_force[others.index(other)] += np.asarray(pair["world_wrench"][end][:3])
            target = scene.palm_target(step)
            recorded += 1
            frames.append(
                {
                    "step": np.int32(step),
                    "status": np.int8(STATUS_CODE[label]),
                    # Copies: on CPU .numpy() is a view of the live state, so every frame of a chunk
                    # would otherwise hold the chunk's last step.
                    "body_q": scene.state.body_q.numpy().copy(),
                    "body_qd": scene.state.body_qd.numpy().copy(),
                    "joint_q": scene.state.joint_q.numpy().copy(),
                    "joint_qd": scene.state.joint_qd.numpy().copy(),
                    "contact_joint_force": forces.joint_force.astype(np.float64),
                    "contact_body_wrench": forces.body_wrench.astype(np.float64),
                    "object_pair_force": pair_force,
                    "palm_target": np.zeros((4, 4)) if target is None else target,
                    "newton_iterations": np.int32(stats.nonlinear_iterations),
                    "linear_iterations": np.int32(stats.linear_iterations),
                    "line_search_iterations": np.int32(stats.line_search_iterations),
                    "regularization_retries": np.int32(stats.regularization_retries),
                }
            )
            if len(frames) == CHUNK:
                flush()
        flush()
        manifest.update(
            rollout_seconds=time.perf_counter() - rollout,
            published_steps=recorded,
            memory=_memory(scene.model.device),
        )
    except NotImplementedError as missing:
        error = f"NOT_APPLICABLE: {missing}"
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    flush()
    complete = error is None and counts["HARD_FAILURE"] == 0
    if not complete:
        status = "NOT_APPLICABLE" if error and error.startswith("NOT_APPLICABLE") else "FAILED"
    else:
        # A run shortened with --max-steps is PARTIAL, never COMPLETE (AGENTS.md: filtered runs remain PARTIAL).
        status = "COMPLETE" if recorded == planned else "PARTIAL"
    manifest.update(
        status=status,
        error=error,
        counts=counts,
        stopped_steps=stopped,
        chunks=chunks,
        total_seconds=time.perf_counter() - start,
    )
    write_json(args.output / "manifest.json", manifest)
    print({"status": manifest["status"], "counts": counts, "error": error, "output": str(args.output)})
    # 0 only for a complete run; a PARTIAL (--max-steps) run exits 2 so wrappers cannot mistake it for complete.
    return {"COMPLETE": 0, "PARTIAL": 2}.get(status, 1)


if __name__ == "__main__":
    raise SystemExit(main())
