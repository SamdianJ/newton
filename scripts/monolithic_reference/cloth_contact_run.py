# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared local receipts for exploratory cloth contact and Franka rollouts."""

import dataclasses
import enum
import hashlib
import importlib
import json
import platform
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .cloth_contact_geometry import inspect_geometry


def _plain(value):
    if dataclasses.is_dataclass(value):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    """Write strict JSON; unavailable or nonfinite measurements remain null."""
    Path(path).write_text(json.dumps(_plain(value), indent=2, allow_nan=False) + "\n")


def publication(stats, forces, *, previous_generation):
    """Classify a fresh publication without inferring success from absence of rollback."""
    label = (stats.variational or {}).get("publication_status")
    if (
        stats.step_generation <= previous_generation
        or stats.rolled_back
        or not forces.force_valid
        or forces.force_generation != stats.step_generation
        or label not in ("CONVERGED", "STOPPED_ACCEPTED")
        or stats.converged != (label == "CONVERGED")
    ):
        return "HARD_FAILURE"
    return label


def add_arguments(parser):
    """Add the common execution controls without changing the physical recipe."""
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--method", choices=("off", "point-cloud", "vfee"), default="off")
    parser.add_argument("--method-config", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--geometry-stride", type=int, default=25)
    parser.add_argument("--output", type=Path, required=True)


def create_solver(model, cloth_inputs, solver_kwargs, method, method_config):
    """Select only the explicit local method adapter; disabled means the old solver."""
    if method == "off":
        if method_config:
            raise ValueError("Method-off runs must not silently consume an enabled configuration")
        return SolverMonolithic(model, cloth_inputs=cloth_inputs, **solver_kwargs)
    if method_config is None or method_config.get("method") != method:
        raise ValueError("Enabled contact requires an explicitly matching method configuration")
    adapter = importlib.import_module(".cloth_contact_method", __package__)
    parameters = {key: value for key, value in method_config.items() if key != "method"}
    return adapter.create_solver(
        model, cloth_inputs=cloth_inputs, solver_kwargs=solver_kwargs, method_config=parameters
    )


def execute(scene_factory, args):
    """Retain each attempt, publication and failure; never award exploratory task PASS."""
    if args.geometry_stride < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError("Geometry stride and requested steps must be positive")
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
        "source_files_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "warp": wp.__version__,
            "newton": newton.__file__,
        },
        "geometry_observer": {"stride": args.geometry_stride, "continuous_check": False, "affects_dynamics": False},
    }
    write_json(args.output / "manifest.json", manifest)
    scene, error, rollout_start, rollout_seconds = None, None, None, None
    counts = dict.fromkeys(("CONVERGED", "STOPPED_ACCEPTED", "HARD_FAILURE"), 0)
    frames, chunks, publications, stopped, construction_start = [], [], 0, [], time.perf_counter()

    def flush():
        if not frames:
            return
        path = args.output / f"states-{len(chunks):04d}.npz"
        np.savez_compressed(path, **{key: np.stack([f[key] for f in frames]) for key in frames[0]})
        chunks.append(path)
        frames.clear()

    try:
        scene = scene_factory()
        observer_radius = scene.metadata.get("recipe", {}).get("geometry_search_distance_m", 0.01)
        if scene.total_steps <= 0 or not np.isfinite(scene.dt) or scene.dt <= 0:
            raise ValueError("Scene must declare a finite positive planned duration")
        manifest.update(
            status="RUNNING",
            scene=scene.metadata,
            planned_steps=scene.total_steps,
            dt_s=scene.dt,
            construction_seconds=time.perf_counter() - construction_start,
        )
        write_json(args.output / "manifest.json", manifest)
        np.savez_compressed(
            args.output / "mesh.npz",
            triangles=scene.triangles,
            owners=scene.owners,
            thicknesses=scene.thicknesses,
            reference=scene.model.particle_q.numpy(),
        )
        initial = scene.state.particle_q.numpy()
        triangle_ids = np.asarray(scene.triangles)
        reference_faces = scene.model.particle_q.numpy()[triangle_ids].astype(float)
        reference_area2 = np.linalg.norm(
            np.cross(reference_faces[:, 1] - reference_faces[:, 0], reference_faces[:, 2] - reference_faces[:, 0]),
            axis=1,
        )
        masses = scene.model.particle_mass.numpy() if hasattr(scene.model, "particle_mass") else None
        write_json(
            args.output / "initial-geometry.json",
            inspect_geometry(
                initial, scene.triangles, scene.owners, scene.thicknesses, search_distance=observer_radius
            ),
        )
        stop = min(args.max_steps or scene.total_steps, scene.total_steps)
        wp.synchronize_device(scene.model.device)
        rollout_start = time.perf_counter()
        with (args.output / "steps.jsonl").open("w") as stream:
            for request in range(stop):
                previous = scene.solver.last_stats.step_generation
                before_steps = scene.steps
                step_start = time.perf_counter()
                row = {"request": request, "requested_dt_s": scene.dt, "time_before_s": before_steps * scene.dt}
                step_error = None
                try:
                    scene.step()
                except Exception as failure:
                    step_error = f"{type(failure).__name__}: {failure}"
                stats = scene.solver.last_stats
                row.update(
                    stats_fresh=stats.step_generation > previous,
                    generation=stats.step_generation,
                    status=stats.status,
                    failure_reason=stats.failure_reason,
                    phase=scene.phase,
                )
                try:
                    forces = getattr(scene, "last_forces", None)
                    if forces is None or forces.force_generation != stats.step_generation:
                        forces = scene.solver.get_contact_forces(scene.state)
                    label = publication(stats, forces, previous_generation=previous)
                except Exception as failure:
                    label, forces = "HARD_FAILURE", None
                    step_error = step_error or f"{type(failure).__name__}: {failure}"
                counts[label] += 1
                row["publication_status"] = label
                row["force_generation"] = forces.force_generation if forces is not None else None
                if label == "HARD_FAILURE":
                    error = step_error or f"Invalid publication: {stats.status}"
                    row.update(error=error, time_after_s=before_steps * scene.dt)
                else:
                    publications += 1
                    row["time_after_s"] = (before_steps + 1) * scene.dt
                    if label == "STOPPED_ACCEPTED":
                        stopped.append({"step": before_steps + 1, "reason": stats.failure_reason})
                    if step_error or scene.steps != before_steps + 1:
                        error = step_error or "Scene clock disagrees with a valid solver publication"
                        row["execution_error_after_publication"] = error
                try:
                    if label == "HARD_FAILURE":
                        raise RuntimeError(error)
                    x, velocity = scene.state.particle_q.numpy(), scene.state.particle_qd.numpy()
                    if not np.isfinite(x).all() or not np.isfinite(velocity).all():
                        raise RuntimeError("Nonfinite published particle state")
                    current_faces = x[triangle_ids].astype(float)
                    ratios = (
                        np.linalg.norm(
                            np.cross(
                                current_faces[:, 1] - current_faces[:, 0], current_faces[:, 2] - current_faces[:, 0]
                            ),
                            axis=1,
                        )
                        / reference_area2
                    )
                    row["area_ratio_min"] = float(ratios.min())
                    row["area_ratio_max"] = float(ratios.max())
                    row["kinetic_energy_J"] = (
                        float(np.sum(masses[:, None] * velocity.astype(float) ** 2) / 2) if masses is not None else None
                    )
                    row["solver_stats"] = {
                        name: getattr(stats, name, None)
                        for name in (
                            "nonlinear_iterations",
                            "linear_iterations",
                            "line_search_iterations",
                            "regularization_retries",
                            "triplet_count",
                            "triplet_capacity",
                            "matrix_nnz",
                            "merit_initial",
                            "merit_final",
                            "convergence_ratio",
                        )
                    }
                    diagnostics = {"method": "off", "active": 0, "compressive_load_N": 0.0}
                    if args.method != "off":
                        adapter = importlib.import_module(".cloth_contact_method", __package__)
                        diagnostics = adapter.diagnostics(scene.solver, scene.state)
                    row["method_diagnostics"] = diagnostics
                    row["contact_spatial"] = (stats.variational or {}).get("contact_spatial")
                    row["contact_pool"] = (stats.variational or {}).get("contact_pool")
                    frames.append(
                        {
                            "step": before_steps + 1,
                            "x": x,
                            "v": velocity,
                            "q": scene.state.joint_q.numpy(),
                            "qd": scene.state.joint_qd.numpy(),
                            "particle_force": forces.particle_force,
                            "joint_force": forces.joint_force,
                            "external_particle_force": scene.state.particle_f.numpy(),
                        }
                    )
                    if hasattr(scene, "fixture_work"):
                        row["fixture_external_work_J"] = scene.fixture_work
                except Exception as failure:
                    error = error or f"{type(failure).__name__}: {failure}"
                    if label != "HARD_FAILURE":
                        row["recording_error_after_publication"] = error
                row["host_attempt_with_feedback_seconds"] = time.perf_counter() - step_start
                stream.write(json.dumps(_plain(row), allow_nan=False) + "\n")
                stream.flush()
                if len(frames) >= 64:
                    flush()
                if error:
                    break
                if publications % 100 == 0:
                    print(
                        json.dumps({"published_steps": publications, "phase": scene.phase, "counts": counts}),
                        flush=True,
                    )
        flush()
        wp.synchronize_device(scene.model.device)
        rollout_seconds = time.perf_counter() - rollout_start
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
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
        observer_start = time.perf_counter()
        checked, observer_errors = 0, []
        if scene is not None:
            with (args.output / "quality.jsonl").open("w") as stream:
                for chunk in chunks:
                    with np.load(chunk) as data:
                        for step, x in zip(data["step"], data["x"], strict=True):
                            result = {"step": int(step), "geometry_checked": False, "continuous_check": False}
                            if step == 1 or step % args.geometry_stride == 0 or step == publications:
                                try:
                                    result.update(
                                        inspect_geometry(
                                            x,
                                            scene.triangles,
                                            scene.owners,
                                            scene.thicknesses,
                                            search_distance=observer_radius,
                                        ),
                                        geometry_checked=True,
                                    )
                                    checked += 1
                                except Exception as failure:
                                    result["observer_error"] = f"{type(failure).__name__}: {failure}"
                                    observer_errors.append({"step": int(step), "error": result["observer_error"]})
                            stream.write(json.dumps(_plain(result), allow_nan=False) + "\n")
        summary = {
            "status": "HARD_FAILURE"
            if error
            else "COMPLETE"
            if scene is not None and publications == scene.total_steps
            else "PARTIAL",
            "task_verdict": "OBSERVATION_ONLY",
            "error": error,
            "published_steps": publications,
            "planned_steps": scene.total_steps if scene is not None else None,
            "publication_counts": counts,
            "stopped_steps": stopped,
            "rollout_seconds": rollout_seconds,
            "published_time_s": publications * scene.dt if scene is not None else 0,
            "offline_geometry_seconds": time.perf_counter() - observer_start,
            "geometry_checked_frames": checked,
            "geometry_unchecked_frames": publications - checked,
            "continuous_check": False,
            "observer_errors": observer_errors,
        }
        manifest["status"] = summary["status"]
        write_json(args.output / "manifest.json", manifest)
        write_json(args.output / "summary.json", summary)
        with (args.output / "files.sha256").open("w") as stream:
            for path in sorted(args.output.iterdir()):
                if path.is_file() and path.name != "files.sha256":
                    stream.write(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n")
        print(json.dumps(_plain(summary), allow_nan=False), flush=True)
    return 1 if error else 0
