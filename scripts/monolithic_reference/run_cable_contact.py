# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record a cable-contact E2E scene (E1-E4 or a performance variant) with full per-frame geometry.

Every published frame stores cable positions, velocities, material angles and the
reference frame, so ``cable_contact_criteria`` can judge penetration, crossing,
link conservation and energy offline. Runs are observations; the criteria decide gates.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .cable_contact_scenes import CableContactScene
from .cloth_contact_run import _memory, _plain, publication, write_json

STAT_FIELDS = (
    "nonlinear_iterations",
    "linear_iterations",
    "linear_scheduled_slots",
    "line_search_iterations",
    "regularization_retries",
    "matrix_assembly_count",
    "factor_setup_count",
    "factor_failure_count",
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", help="scene id; must equal the recipe's scene field")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--method", choices=("off", "cable-contact"), default="off")
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _graph_totals(variational):
    graphs = (variational or {}).get("graphs") or {}
    segments = graphs.get("segments") or {}
    return {
        "graph_captures": int(sum(s["captures"] for s in segments.values())),
        "graph_replays": int(sum(s["replays"] for s in segments.values())),
    }


def _topology(scene):
    """Static arrays the offline criteria need: ranges, rest geometry, stiffness and masses."""
    ranges = np.asarray(scene.vertex_ranges, dtype=np.int64)
    return {
        "vertex_ranges": ranges,
        "rest_positions": np.vstack(scene.rest),
        "rest_directors": np.vstack([np.asarray(c.rest_director, dtype=float) for c in scene.cables]),
        "theta_offsets": np.asarray(scene.theta_offsets, dtype=np.int64),
        "radius": np.asarray([c.radius for c in scene.cables], dtype=float),
        "EA": np.asarray([c.EA for c in scene.cables], dtype=float),
        "EI": np.asarray([np.asarray(c.bending, dtype=float)[0, 0] for c in scene.cables], dtype=float),
        "GJ": np.asarray([c.GJ for c in scene.cables], dtype=float),
        "particle_mass": scene.model.particle_mass.numpy().astype(float),
        "fixed_vertices": np.asarray(scene.fixed_vertices, dtype=np.int64),
        "gravity": np.asarray(scene.gravity, dtype=float),
        "ground_height": np.asarray([np.nan if scene.ground_height is None else scene.ground_height]),
        "dt": np.asarray([scene.dt]),
    }


def execute(args):
    """Run one recipe, keeping every attempt, publication and failure."""
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Requested steps must be positive")
    recipe = json.loads(args.recipe.read_text())
    if recipe.get("scene") != args.scene:
        raise ValueError(f"Scene argument {args.scene!r} differs from the recipe scene {recipe.get('scene')!r}")
    config = json.loads(args.method_config.read_text()) if args.method_config else None
    source = Path(__file__).resolve().parents[2]
    if Path(newton.__file__).resolve() != source / "newton/__init__.py":
        raise RuntimeError("Newton import does not belong to the tested worktree")
    args.output.mkdir(parents=True, exist_ok=False)

    def git(*command):
        return subprocess.check_output(["git", *command], cwd=source, text=True).strip()

    files = sorted((source / "newton/_src/solvers/monolithic").glob("*.py"))
    files += [*sorted(Path(__file__).parent.glob("*.py")), args.recipe.resolve()]
    if args.method_config is not None:
        files.append(args.method_config.resolve())
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output / "dirty.patch").write_text(git("diff", "--binary", "HEAD"))
    manifest = {
        "schema_version": 1,
        "status": "STARTING",
        "source": source,
        "source_sha": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "arguments": vars(args),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "source_files_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warp": wp.__version__,
            "newton": newton.__file__,
        },
        "frames": "every published step: cable positions, velocities, material angles, reference frame",
    }
    write_json(args.output / "manifest.json", manifest)
    scene, error, stage = None, None, "construction"
    counts = dict.fromkeys(("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE"), 0)
    frames, chunks, stopped, published, attempted = [], [], [], 0, 0
    # Per-run peaks of the per-step cable diagnostics (plan section 2: used shift, contact and triplet counts).
    cable = {"used_shift_max": 0.0}
    if args.method == "cable-contact":
        cable.update(dict.fromkeys(("triplets_max", "inter_candidates_max", "inter_active_max"), 0))
        cable.update(dict.fromkeys(("self_candidates_max", "self_active_max"), 0))
    rollout_start = rollout_seconds = None
    construction_start = time.perf_counter()

    def flush():
        if frames:
            path = args.output / f"states-{len(chunks):04d}.npz"
            np.savez_compressed(path, **{key: np.stack([f[key] for f in frames]) for key in frames[0]})
            chunks.append(path.name)
            frames.clear()

    try:
        scene = CableContactScene(
            recipe,
            device=args.device,
            precision=args.precision,
            use_cuda_graph=args.use_cuda_graph,
            method=args.method,
            method_config=config,
        )
        manifest.update(
            status="RUNNING",
            scene=scene.metadata,
            planned_steps=scene.total_steps,
            dt_s=scene.dt,
            construction_seconds=time.perf_counter() - construction_start,
            device=str(scene.model.device),
            timing_scope="cold construction and rollout; lazy JIT and graph capture included",
        )
        write_json(args.output / "manifest.json", manifest)
        np.savez_compressed(args.output / "topology.npz", **_topology(scene))
        initial = scene.state._monolithic_coupled
        np.savez_compressed(
            args.output / "initial.npz",
            x=scene.state.particle_q.numpy(),
            v=scene.state.particle_qd.numpy(),
            theta=initial.cable_theta.numpy(),
            director=initial.cable_ref_director.numpy(),
            twist=initial.cable_ref_twist.numpy(),
            target=scene.control._monolithic_cable.target_position.numpy(),
            target_theta=scene.control._monolithic_cable.target_theta.numpy(),
        )
        stop = min(args.max_steps or scene.total_steps, scene.total_steps)
        wp.synchronize_device(scene.model.device)
        rollout_start, stage = time.perf_counter(), "rollout"
        with (args.output / "steps.jsonl").open("w") as stream:
            for request in range(stop):
                attempted += 1
                previous = scene.solver.last_stats.step_generation
                before = scene.steps
                start = time.perf_counter()
                step_error = None
                try:
                    scene.step()
                except Exception as failure:
                    step_error = f"{type(failure).__name__}: {failure}"
                stats = scene.solver.last_stats
                row = {
                    "request": request,
                    "time_after_s": (before + 1) * scene.dt,
                    "phase": scene.phase,
                    "generation": stats.step_generation,
                    "status": stats.status,
                    "failure_reason": stats.failure_reason,
                }
                fresh = stats.step_generation > previous
                if fresh:
                    row.update({name: getattr(stats, name) for name in STAT_FIELDS})
                    row.update(_graph_totals(stats.variational))
                    row["contact_pool"] = (stats.variational or {}).get("contact_pool")
                    row["cable_used_shift_max"] = (stats.variational or {}).get("cable_used_shift_max")
                    cable["used_shift_max"] = max(cable["used_shift_max"], row["cable_used_shift_max"] or 0.0)
                    if args.method == "cable-contact":
                        # Triplets of the last cable contact evaluation; the solver has no public accessor.
                        records = scene.solver._cable_contact.pool.records
                        row["cable_contact_triplets"] = int(records.triplet_count.numpy()[0])
                        cable["triplets_max"] = max(cable["triplets_max"], row["cable_contact_triplets"])
                try:
                    forces = scene.solver.get_contact_forces(scene.state)
                    label = publication(stats, forces, previous_generation=previous)
                except Exception as failure:
                    label, forces = "HARD_FAILURE", None
                    step_error = step_error or f"{type(failure).__name__}: {failure}"
                counts[label] += 1
                row["publication_status"] = label
                if label == "HARD_FAILURE":
                    error = step_error or f"Invalid publication: {stats.status}"
                    row["error"] = error
                else:
                    published += 1
                    if label == "STOPPED_ACCEPTED":
                        stopped.append({"step": before + 1, "reason": stats.failure_reason})
                    history = scene.state._monolithic_coupled
                    if args.method == "cable-contact":
                        diagnostics = scene.solver.get_cable_contact_diagnostics(scene.state)
                        row["cable_contact"] = {name: diagnostics[name] for name in ("inter", "self")}
                        for name in ("inter", "self"):
                            for key in ("candidates", "active"):
                                peak = f"{name}_{key}_max"
                                cable[peak] = max(cable[peak], diagnostics[name][key])
                    x, v = scene.state.particle_q.numpy(), scene.state.particle_qd.numpy()
                    if not (np.isfinite(x).all() and np.isfinite(v).all()):
                        error = "Nonfinite published particle state"
                        row["error"] = error
                    frames.append(
                        {
                            "step": np.int64(before + 1),
                            "x": x.copy(),
                            "v": v.copy(),
                            "theta": history.cable_theta.numpy().copy(),
                            "director": history.cable_ref_director.numpy().copy(),
                            "twist": history.cable_ref_twist.numpy().copy(),
                            "target": scene.control._monolithic_cable.target_position.numpy().copy(),
                            "target_theta": scene.control._monolithic_cable.target_theta.numpy().copy(),
                            "particle_force": forces.particle_force.copy(),
                            "cable_torque": forces.cable_torque.copy(),
                        }
                    )
                row["host_seconds"] = time.perf_counter() - start
                stream.write(json.dumps(_plain(row), allow_nan=False) + "\n")
                stream.flush()
                if len(frames) >= 64:
                    flush()
                if error:
                    break
                if published % 100 == 0:
                    print(
                        json.dumps({"published_steps": published, "phase": scene.phase, "counts": counts}), flush=True
                    )
        flush()
        wp.synchronize_device(scene.model.device)
        rollout_seconds = time.perf_counter() - rollout_start
    except Exception as failure:
        error = error or f"{type(failure).__name__}: {failure}"
    finally:
        try:
            flush()
        except Exception as failure:
            error = error or f"State recording failure: {failure}"
        if rollout_start is not None and rollout_seconds is None:
            rollout_seconds = time.perf_counter() - rollout_start
        if scene is not None and error:
            try:
                scene.solver.save_checkpoint(scene.state, args.output / "failed-state.npz")
                write_json(args.output / "failure-stats.json", scene.solver.last_stats)
            except Exception as failure:
                manifest["failure_capture_error"] = str(failure)
        planned = scene.total_steps if scene is not None else None
        requested = min(args.max_steps or planned, planned) if planned else None
        status = "HARD_FAILURE" if error else "COMPLETE" if published == planned else "PARTIAL"
        summary = {
            "status": status,
            "error": error,
            "failure_stage": stage if error else None,
            "scene": args.scene,
            "precision": args.precision,
            "device": args.device,
            "use_cuda_graph": args.use_cuda_graph,
            "method": args.method,
            "attempted_requests": attempted,
            "requested_steps": requested,
            "published_steps": published,
            "planned_steps": planned,
            "publication_counts": counts,
            "stopped_steps": stopped,
            "cable": cable,
            "construction_seconds": manifest.get("construction_seconds"),
            "rollout_seconds": rollout_seconds,
            "memory": _memory(None if scene is None else scene.model.device),
            "chunks": chunks,
        }
        manifest["source_files_changed_during_run"] = [
            name
            for name, digest in hashes.items()
            if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
        ]
        manifest["status"] = status
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "summary.json", summary)
        with (args.output / "files.sha256").open("w") as stream:
            for path in sorted(args.output.iterdir()):
                if path.is_file() and path.name != "files.sha256":
                    stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
        print(json.dumps(_plain(summary), allow_nan=False), flush=True)
    return 1 if error else 0


def main(argv=None):
    return execute(parse_arguments(argv))


if __name__ == "__main__":
    raise SystemExit(main())
