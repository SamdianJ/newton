# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, instance-local observation of the existing production solver."""

import hashlib
import inspect
import platform
import subprocess
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def contact_pair_diagnostics(run, *, enabled=True):
    """Optionally omit host pair reports while retaining pooled force publication."""
    if enabled:
        yield
        return
    if not run._contact_pool:
        raise ValueError("Disabling pair diagnostics requires pooled or spatial contact")
    with patch.object(run._contact, "pair_diagnostics", new=lambda: []):
        yield


def capture_count(run):
    """Include pooled generation, reduction and BSR captures in tool accounting."""
    caches = [run._graphs]
    if run._contact_pool:
        caches.extend((run._contact._batch.graphs, run._contact.pool.graphs))
        caches.extend(owner.graphs for owner in run._linear._fixed_bsr)
    if run._contact_spatial:
        caches.extend((run._contact.query_graphs, run._contact._batch.reference_rows.graphs))
    return sum(sum(cache.captures.values()) for cache in caches)


def provenance():
    """Bind reports to imported source, including uncommitted tool changes."""
    import newton  # noqa: PLC0415

    if Path(newton.__file__).resolve().parent != ROOT / "newton":
        raise RuntimeError("Newton import does not belong to the tested worktree")
    paths = sorted((ROOT / "newton/_src/solvers/monolithic").glob("*.py"))
    paths += sorted((ROOT / "scripts/monolithic_reference").glob("*.py"))
    paths += sorted((ROOT / "newton/examples/softbody").glob("monolithic_*.py"))
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = {
        "root": str(ROOT),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_hashes": hashes,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "warp": wp.__version__,
        "warp_utils_sha256": hashlib.sha256(Path(inspect.getfile(wp.ScopedTimer)).read_bytes()).hexdigest(),
    }
    try:
        result["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,uuid,driver_version,utilization.gpu,memory.used", "--format=csv"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["gpu"] = None
    return result


class SolverProfiler:
    """Measure nested host scopes and stream intervals without per-scope waits.

    CUDA intervals include device idle time while the host submits work. They
    are not kernel busy time. Captured interiors are explicitly opaque.
    """

    def __init__(self, solver, *, activities=False):
        self.solver = solver
        self.device = solver.model.device
        self.activities = activities
        self.rows = []
        self.stack = []
        self.activity_rows = []
        self.step = 0
        self._patches = ExitStack()

    @contextmanager
    def scope(self, name):
        if self.device.is_capturing:
            yield
            return
        row = {
            "id": len(self.rows),
            "parent": self.stack[-1]["id"] if self.stack else None,
            "name": name,
            "step": self.step,
            "host_ms": 0.0,
            "capture_affected": False,
        }
        self.rows.append(row)
        self.stack.append(row)
        if self.device.is_cuda:
            stream = wp.get_stream(self.device)
            row["stream"] = stream.cuda_stream
            row["events"] = (wp.Event(self.device, enable_timing=True), wp.Event(self.device, enable_timing=True))
            stream.record_event(row["events"][0])
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            row["host_ms"] = (time.perf_counter_ns() - start) / 1e6
            if self.device.is_cuda:
                stream.record_event(row["events"][1])
            self.stack.pop()

    def _wrap(self, owner, method, name):
        original = getattr(owner, method)

        def observed(*args, **kwargs):
            if method == "step":
                self.step += 1
                first = len(self.rows)
                before = capture_count(self.solver)
                plan = getattr(self.solver._linear, "_masked_pcg_plan", None)
            with self.scope(name):
                result = original(*args, **kwargs)
            if method == "step":
                captured = before != capture_count(self.solver) or plan is not getattr(
                    self.solver._linear, "_masked_pcg_plan", None
                )
                for row in self.rows[first:]:
                    row["capture_affected"] = captured
                self.collect()
            return result

        self._patches.enter_context(patch.object(owner, method, observed))

    def __enter__(self):
        run = self.solver
        for method in (
            "step",
            "_prepare_step",
            "_evaluate",
            "_assemble",
            "_initial_reference",
            "_take_trial",
            "_trial_evaluation",
            "_physical_linear_error",
            "_validate_final",
            "_finish_candidate",
            "_restore",
        ):
            self._wrap(run, method, method.lstrip("_"))
        for method in (
            "begin_assembly",
            "finalize_assembly",
            "build_scaling",
            "set_regularization",
            "factor_actor_preconditioner",
            "solve_pcg",
        ):
            self._wrap(run._linear, method, "linear." + method)
        # On graph replay these Python methods do not run. Do not invent rows.
        for name, owner in (("articulation", run._objective), ("tet", run._tet), ("contact", run._contact)):
            self._wrap(owner, "evaluate", name + ".evaluate")
        self._wrap(run._contact, "assemble_tangent", "contact.assemble")
        self._wrap(run._contact, "begin_step", "contact.begin_step")
        if run._contact_pool:
            self._wrap(run._contact, "publish", "contact.publish")
            self._wrap(run._contact._batch, "_stage", "contact.generate")
            for method in ("begin", "append", "finish"):
                self._wrap(run._contact.pool, method, "contact.pool." + method)
            for name in ("energy", "residual", "force"):
                self._wrap(getattr(run._contact.pool, name), "reduce", "contact.tree." + name)
        if run._contact_spatial:
            self._wrap(run._contact, "refresh", "contact.refresh")
            self._wrap(run._contact._batch.reference_rows, "reduce", "contact.reference.tree")
        original = run._graphs.run

        def graph(name, parameters, buffers, execute):
            with self.scope("segment." + name):
                return original(name, parameters, buffers, execute)

        self._patches.enter_context(patch.object(run._graphs, "run", graph))
        if self.activities and self.device.is_cuda:
            wp.timing_begin(wp.TIMING_ALL)
        return self

    def collect(self):
        """Read unique events once at the completed step boundary."""
        pending = [row for row in self.rows if "events" in row]
        if pending:
            wp.synchronize_device(self.device)
            for row in pending:
                start, end = row.pop("events")
                row["gpu_interval_ms"] = wp.get_event_elapsed_time(start, end, synchronize=False)

    def __exit__(self, *exc):
        try:
            self.collect()
            if self.activities and self.device.is_cuda:
                self.activity_rows = [
                    {
                        "name": r.name,
                        "kind": r.filter,
                        "elapsed_ms": r.elapsed,
                        "device": str(r.device),
                        "module": "unattributed",
                    }
                    for r in wp.timing_end()
                ]
        finally:
            self._patches.close()

    def report(self):
        """Keep hierarchical self costs separate from inclusive timings."""
        self.collect()
        children = defaultdict(list)
        for row in self.rows:
            children[row["parent"]].append(row)
        groups = defaultdict(list)
        for row in self.rows:
            groups[row["name"]].append(row)
        modules = {}
        for name, rows in groups.items():
            data = {"calls": len(rows), "steady_calls": sum(not r["capture_affected"] for r in rows)}
            for metric in ("host_ms", "gpu_interval_ms"):
                values = [r[metric] for r in rows if metric in r]
                self_values = [
                    max(0.0, r[metric] - sum(c.get(metric, 0.0) for c in children[r["id"]]))
                    for r in rows
                    if metric in r
                ]
                data[metric] = {
                    "inclusive": sum(values),
                    "self": sum(self_values),
                    "steady_inclusive": sum(r.get(metric, 0) for r in rows if not r["capture_affected"]),
                    "p50": float(np.median(values)) if values else None,
                    "p95": float(np.percentile(values, 95)) if values else None,
                }
            modules[name] = data
        for metric in ("host_ms", "gpu_interval_ms"):
            total = sum(data[metric]["self"] for data in modules.values())
            for data in modules.values():
                data[metric]["self_fraction"] = data[metric]["self"] / total if total else None
        return {
            "schema": "monolithic-profile/1",
            "graph": self.solver.use_cuda_graph,
            "measurement": "instrumented, including marked capture; stream intervals include idle time, not GPU utilization",
            "opaque": ["captured objective interior", "PCG device loop"] if self.solver.use_cuda_graph else [],
            "capture_affected_calls": sum(r["capture_affected"] for r in self.rows),
            "modules": modules,
            "samples": self.rows,
            "cuda_activities": self.activity_rows,
            "activity_attribution": "unattributed; Warp activity API has no scope IDs",
        }
