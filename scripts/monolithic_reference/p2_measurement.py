# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Measurement boundaries and evidence for the P2 runners, outside solver code."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

from scripts.monolithic_reference.profile_release import _percentiles, _StageProfile, solver_module

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "monolithic-p2-measurement/v2"


@contextmanager
def solver_timing(solver, *, before_solve=None):
    """Measure completed solves and steps; collect each retry without array readbacks."""
    record = SimpleNamespace(seconds=0.0, calls=[])
    step, solve = solver.step, solver._linear.solve_pcg

    def timed_solve(*args, **kwargs):
        if before_solve is not None:
            before_solve()
        wp.synchronize_device(solver.model.device)
        start = time.perf_counter()
        result = solve(*args, **kwargs)
        wp.synchronize_device(solver.model.device)
        record.calls.append(
            {
                "iterations": result.iterations,
                "status": result.status.name,
                "seconds": time.perf_counter() - start,
                "rho": result.rho,
                "rho_q": result.rho_q,
                "rho_x": result.rho_x,
            }
        )
        return result

    def timed_step(*args, **kwargs):
        record.calls.clear()
        wp.synchronize_device(solver.model.device)
        start = time.perf_counter()
        try:
            return step(*args, **kwargs)
        finally:
            wp.synchronize_device(solver.model.device)
            record.seconds = time.perf_counter() - start

    with patch.object(solver, "step", new=timed_step), patch.object(solver._linear, "solve_pcg", new=timed_solve):
        yield record


def throughput(seconds, accepted_steps, dt, substeps):
    """Use aggregate completed work, including failed-attempt time in the denominator."""
    simulated = accepted_steps * dt
    return {
        "solver_seconds": seconds,
        "simulated_seconds": simulated,
        "completed_output_frames": accepted_steps // substeps,
        "solver_only_fps_at_frame_dt": accepted_steps // substeps / seconds if seconds > 0 else None,
        "solver_only_realtime_factor": simulated / seconds if seconds > 0 else None,
        "solver_wall_seconds_per_simulated_second": seconds / simulated if simulated > 0 else None,
    }


def frame_samples(rows, substeps, seconds_key):
    """Sum actual consecutive physical steps before calculating frame percentiles."""
    frames = []
    for start in range(0, len(rows) - substeps + 1, substeps):
        group = rows[start : start + substeps]
        if any(row.get("rollback", False) for row in group):
            break
        frames.append(
            {
                "time": group[-1]["time"],
                "phase": group[-1].get("phase", group[-1].get("stage")),
                "solver_seconds": sum(row[seconds_key] for row in group),
            }
        )
    return frames


class P2StageProfile(_StageProfile):
    """Report inclusive and exclusive synchronized time with exception-safe nesting."""

    def __init__(self, solver, variant):
        super().__init__(solver, variant)
        self.samples["solver_step"] = []
        self.samples["tet_assembly"] = []
        self.exclusive = {name: [] for name in self.samples}
        self.active = []
        self.sequence = []

    def _timed(self, name, function):
        def wrapped(*args, **kwargs):
            wp.synchronize_device(self.solver.model.device)
            frame = [time.perf_counter(), 0.0]
            self.active.append(frame)
            self.sequence.append(["enter", name])
            try:
                return function(*args, **kwargs)
            finally:
                wp.synchronize_device(self.solver.model.device)
                elapsed = (time.perf_counter() - frame[0]) * 1000
                self.active.pop()
                self.samples[name].append(elapsed)
                self.exclusive[name].append(elapsed - frame[1])
                if self.active:
                    self.active[-1][1] += elapsed
                self.sequence.append(["exit", name])

        return wrapped

    def _install(self):
        super()._install()
        self.stack.enter_context(patch.object(self.solver, "step", new=self._timed("solver_step", self.solver.step)))
        self.stack.enter_context(
            patch.object(
                solver_module,
                "assemble_tet_residual_tangent",
                new=self._timed("tet_assembly", solver_module.assemble_tet_residual_tangent),
            )
        )
        return self

    def summary(self):
        result = super().summary()
        for name, row in result.items():
            row.update(
                inclusive_total_ms=sum(self.samples[name]),
                exclusive_total_ms=sum(self.exclusive[name]),
                exclusive_percentiles_ms=_percentiles(self.exclusive[name]),
            )
        result["solver_step"]["unattributed_ms"] = sum(self.exclusive["solver_step"])
        result["solver_step"]["call_sequence"] = self.sequence
        return result


def snapshot_sources(output):
    """Save the actual working source, dependency inventory and dirty diff before running."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "source.json").exists():
        raise FileExistsError(f"Refuse to replace existing run provenance: {output}")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    hashes = {}
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(set(paths)):
            path = ROOT / relative
            if not path.is_file() or path.suffix not in (".py", ".json", ".toml", ".lock", ".yaml", ".md", ".rst"):
                continue
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr(relative, data)
    (output / "dirty.diff").write_bytes(git("diff", "HEAD", "--binary"))
    (output / "git-status.txt").write_bytes(git("status", "--short"))
    dependencies = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    payload = "\n".join(dependencies) + "\n"
    (output / "dependencies.txt").write_text(payload)
    record = {
        "schema": SCHEMA,
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "warp": wp.__version__,
        "thread_environment": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cpu_count": os.cpu_count(),
        "sources": hashes,
        "dependency_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (output / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def evidence_index(output):
    """Index every trace and binary artifact without self-referential hashes."""
    output = Path(output)
    rows = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "evidence-index.json" and not path.name.endswith(".sha256")
    }
    (output / "evidence-index.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def array_fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def export_candidate(solver, output, *, time_s, fixture):
    """Export a loaded solve input for offline algebra/replay, never a history checkpoint."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workspace = solver._linear
    candidate = solver._transaction.accepted
    original = solver._transaction._original
    arrays = {
        "R": candidate.residual.numpy(),
        "rhs_hat": solver._rhs_hat.numpy(),
        "D": workspace.diagonal.numpy(),
        "S": workspace.scale.numpy(),
        "dynamic_diagonal": workspace.dynamic_diagonal.numpy(),
        "aq_actor_dense": workspace.aq_actor_dense.numpy(),
        "candidate_x": candidate.state.particle_q.numpy(),
        "candidate_q": candidate.state.joint_q.numpy(),
        "particle_q_n": original.particle_q.numpy(),
        "particle_qd_n": original.particle_qd.numpy(),
        "frozen_particle_f": solver._transaction.inputs.particle_f.numpy(),
        "rest_x": solver.model.particle_q.numpy(),
        "gravity": solver.model.gravity.numpy(),
        "dynamic_particle_ids": solver._layout.dynamic_particle_ids.numpy(),
        "particle_to_dynamic": solver._layout.particle_to_dynamic.numpy(),
        "tet_indices": solver.model.tet_indices.numpy(),
    }
    for name, matrix in (("K", workspace.k_global_scalar_bsr), ("ax_internal", workspace.ax_internal_bsr3)):
        offsets = matrix.offsets.numpy()
        count = int(offsets[-1])
        arrays.update(
            {
                f"{name}_offsets": offsets,
                f"{name}_columns": matrix.columns.numpy()[:count],
                f"{name}_values": matrix.values.numpy()[:count],
            }
        )
    factors = workspace._factors
    count = int(factors.count.numpy()[0])
    for name in ("gq", "gx_columns", "gx_values", "weights", "kind", "candidate_tid"):
        arrays[f"factor_{name}"] = getattr(factors, name).numpy()[:count]
    arrays["static_face_pairs"] = solver.collision_pipeline._face_pairs.numpy()
    metadata = {
        "schema": SCHEMA,
        "scope": "offline matrix/RHS and tet input; not a physical history checkpoint",
        "time_s": time_s,
        "dt": candidate._dt,
        "lambda": workspace._lambda,
        "generation": asdict(workspace._generation),
        "fixture": fixture,
        "q_dof_count": workspace.layout.q_dof_count,
        "input_sha256": array_fingerprint(arrays),
        "operator": "Ahat = S K S + lambda I; rhs_hat = -S R; delta_z = S y",
    }
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return arrays
