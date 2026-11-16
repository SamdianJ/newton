# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Time the frozen floating-hand fixed-state window (P9d K4/K17); measurement only, no physics gate.

    python -m scripts.monolithic_reference.benchmark_floating_hand \\
        --fixed-state scripts/monolithic_reference/fixtures/floating_hand_fixed_grasp_box_v1.json \\
        --mount wrist --device cuda:0 --passes 3 --instrumented segments,syncs --output <new directory>

The recipe is replayed untimed to the window start, the state is checkpointed,
and the window runs once as the natural continuation (W0) and then ``--passes``
times from the checkpoint (W1..). Timed passes synchronize the device around
``solver.step`` only; control writes stay outside the timer. Instrumented passes
(segment events, host readback counts) are reported separately and never mixed
with the timed passes.
"""

import argparse
import gc
import hashlib
import inspect
import json
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.segment_graph import _SegmentGraphCache
from scripts.monolithic_reference.cloth_contact_run import publication, write_json
from scripts.monolithic_reference.floating_hand_scenes import MOUNTS, FloatingHandScene, load_recipe

SOURCE = Path(__file__).resolve().parents[2]
SOLVER_DIR = str(SOURCE / "newton/_src/solvers/monolithic")
WRAPPED_METHODS = ("_prepare_step", "_assemble", "_validate_final", "_finish_candidate")


def _git(*command):
    return subprocess.check_output(["git", *command], cwd=SOURCE, text=True).strip()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _gpu():
    """Clock, power and throttle state, and other compute processes (None if nvidia-smi is unavailable)."""
    fields = "pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,clocks_throttle_reasons.active,utilization.gpu"
    try:
        state = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader"], text=True, timeout=10
        ).strip()
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError) as failure:
        return {"error": str(failure)}
    return {"query": fields, "state": state, "compute_apps": [a for a in apps.splitlines() if a]}


def _graph_counters():
    totals = Counter()
    for cache in (o for o in gc.get_objects() if isinstance(o, _SegmentGraphCache)):
        for name in cache.entries:
            totals[f"{name}.captures"] += cache.captures[name]
            totals[f"{name}.evictions"] += cache.evictions[name]
    return dict(totals)


def _digest(state):
    h = hashlib.sha256()
    for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
        h.update(getattr(state, name).numpy().tobytes())
    return h.hexdigest()


def _percentiles(values):
    v = np.asarray(values, dtype=np.float64)
    if not len(v):
        return None
    return {
        "count": int(len(v)),
        "total_s": float(v.sum()),
        "median_ms": float(np.median(v) * 1e3),
        "p10_ms": float(np.quantile(v, 0.1) * 1e3),
        "p90_ms": float(np.quantile(v, 0.9) * 1e3),
        "p95_ms": float(np.quantile(v, 0.95) * 1e3),
        "max_ms": float(v.max() * 1e3),
    }


def _run_window(scene, window, device, *, timed=True):
    """Run the window from the current state; return per-step rows (time, counts, publication)."""
    solver, rows, original = scene.solver, [], scene.solver.step
    clock = {}

    def stepper(*args, **kwargs):
        wp.synchronize_device(device)
        start = time.perf_counter()
        result = original(*args, **kwargs)
        wp.synchronize_device(device)
        clock["solver"] = time.perf_counter() - start
        return result

    if timed:
        solver.step = stepper
    try:
        while scene.steps < window[1]:
            previous, step = solver.last_stats.step_generation, scene.steps
            outer = time.perf_counter()
            scene.step()
            outer = time.perf_counter() - outer
            stats = solver.last_stats
            forces = solver.get_contact_forces(scene.state) if not stats.rolled_back else None
            label = "HARD_FAILURE" if forces is None else publication(stats, forces, previous_generation=previous)
            rows.append(
                {
                    "step": step,
                    "solver_s": clock.get("solver"),
                    "with_harness_s": outer,
                    "newton": int(stats.nonlinear_iterations),
                    "pcg": int(stats.linear_iterations),
                    "line_search": int(stats.line_search_iterations),
                    "regularization_retries": int(stats.regularization_retries),
                    "publication": label,
                }
            )
            if label == "HARD_FAILURE":
                break
    finally:
        if timed:
            del solver.step
    return rows


def _summarize(rows):
    solver = [r["solver_s"] for r in rows if r["solver_s"] is not None]
    by_newton = defaultdict(list)
    for r in rows:
        if r["solver_s"] is not None:
            by_newton[r["newton"]].append(r["solver_s"])
    medians = {n: float(np.median(v) * 1e3) for n, v in sorted(by_newton.items())}
    newton = sum(r["newton"] for r in rows)
    return {
        "solver_step": _percentiles(solver),
        "with_harness": _percentiles([r["with_harness_s"] for r in rows]),
        "median_ms_by_newton": medians,
        "steps_by_newton": {n: len(v) for n, v in sorted(by_newton.items())},
        "marginal_newton_ms": medians[1] - medians[0] if 0 in medians and 1 in medians else None,
        "window_s_per_newton": float(sum(solver) / newton) if newton else None,
        "workload": {k: sum(r[k] for r in rows) for k in ("newton", "pcg", "line_search", "regularization_retries")},
        "publication": dict(Counter(r["publication"] for r in rows)),
        "per_step_counts": [[r["newton"], r["pcg"]] for r in rows],
    }


def _instrument_segments(scene, window, device):
    """One pass with CUDA events around every graph segment and selected solver phases."""
    records = defaultdict(lambda: [0.0, 0.0, 0])  # name -> host s, gpu ms, calls
    pending = []
    original_run = _SegmentGraphCache.run

    def timed(name, call):
        begin, end = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
        wp.record_event(begin)
        start = time.perf_counter()
        result = call()
        records[name][0] += time.perf_counter() - start
        wp.record_event(end)
        pending.append((name, begin, end))
        records[name][2] += 1
        return result

    def run(self, name, parameters, buffers, execute, *, warmup=True):
        return timed(f"segment:{name}", lambda: original_run(self, name, parameters, buffers, execute, warmup=warmup))

    solver = scene.solver
    originals = {m: getattr(solver, m) for m in WRAPPED_METHODS}
    for m, f in originals.items():
        setattr(solver, m, lambda *a, _f=f, _m=m, **k: timed(f"method:{_m}", lambda: _f(*a, **k)))
    pcg = solver._linear.solve_pcg
    solver._linear.solve_pcg = lambda *a, **k: timed("method:solve_pcg", lambda: pcg(*a, **k))
    _SegmentGraphCache.run = run
    try:
        start = time.perf_counter()
        rows = _run_window(scene, window, device, timed=False)
        wp.synchronize_device(device)
        elapsed = time.perf_counter() - start
        for name, begin, end in pending:
            records[name][1] += wp.get_event_elapsed_time(begin, end)
    finally:
        _SegmentGraphCache.run = original_run
        for m in originals:
            delattr(solver, m)
        del solver._linear.solve_pcg
    steps = max(1, len(rows))
    return {
        "window_s": elapsed,
        "steps": len(rows),
        "per_step": {
            name: {"host_ms": v[0] * 1e3 / steps, "gpu_ms": v[1] / steps, "calls": v[2] / steps}
            for name, v in sorted(records.items())
        },
        "note": "nested scopes overlap (methods contain segments); GPU time is the event interval on the solver stream",
    }


def _instrument_syncs(scene, window, device):
    """One pass counting host readbacks and syncs issued from solver code, by call site."""
    counts = Counter()
    targets = [
        (wp.array, "numpy"),
        (wp.array, "assign"),
        (wp, "synchronize_device"),
        (wp, "synchronize"),
        (wp, "copy"),
    ]
    saved = [(owner, name, getattr(owner, name)) for owner, name in targets]

    def site():
        for frame in inspect.stack()[2:]:
            if frame.filename.startswith(SOLVER_DIR):
                return f"{Path(frame.filename).name}:{frame.function}"
        return None

    def wrap(name, f):
        def counted(*a, **k):
            where = site()
            if where is not None:
                counts[(name, where)] += 1
            return f(*a, **k)

        return counted

    for owner, name, f in saved:
        setattr(owner, name, wrap(name, f))
    try:
        rows = _run_window(scene, window, device, timed=False)
    finally:
        for owner, name, f in saved:
            setattr(owner, name, f)
    steps = max(1, len(rows))
    return {
        "steps": len(rows),
        "per_step": {f"{name} @ {where}": n / steps for (name, where), n in counts.most_common()},
        "newton_mix": dict(Counter(r["newton"] for r in rows)),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed-state", type=Path, required=True)
    parser.add_argument("--mount", choices=MOUNTS, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--instrumented", default="none", help="comma list of segments,syncs or none")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    instrumented = set() if args.instrumented == "none" else set(args.instrumented.split(","))
    if instrumented - {"segments", "syncs"} or args.passes < 1:
        raise ValueError(f"--instrumented must list segments/syncs or be none, got {args.instrumented!r}; passes >= 1")
    if Path(newton.__file__).resolve() != SOURCE / "newton/__init__.py":
        raise RuntimeError(f"Newton import {newton.__file__} does not belong to {SOURCE}")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Tracked changes in the worktree; commit before measuring")
    fixed = json.loads(args.fixed_state.read_text())
    if fixed.get("status") != "FROZEN":
        raise ValueError(f"{args.fixed_state}: status must be FROZEN, got {fixed.get('status')!r}")
    recipe_path = SOURCE / fixed["recipe"]
    recipe = load_recipe(recipe_path)
    window = tuple(fixed["window_steps"])
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "p9d-floating-hand-benchmark/v1",
        "source_sha": _git("rev-parse", "HEAD"),
        "arguments": vars(args),
        "fixed_state_sha256": _sha(args.fixed_state),
        "recipe_sha256": _sha(recipe_path),
        "tool_sha256": _sha(__file__),
        "window_steps": window,
        "precision": fixed["precision"],
        "use_cuda_graph": fixed["use_cuda_graph"],
        "warp": wp.__version__,
        "timing": "wp.synchronize_device before and after solver.step; latency per step, not overlapped throughput",
        "passes": {},
    }
    try:
        scene = FloatingHandScene(
            recipe,
            device=args.device,
            precision=fixed["precision"],
            use_cuda_graph=fixed["use_cuda_graph"],
            mount=args.mount,
        )
    except NotImplementedError as missing:
        receipt["status"] = f"NOT_APPLICABLE: {missing}"
        write_json(args.output / "receipt.json", receipt)
        return 1
    device = scene.model.device
    replay = _run_window(scene, (0, window[0]), device, timed=False)
    receipt["replay"] = {"publication": dict(Counter(r["publication"] for r in replay)), "steps": len(replay)}
    if scene.steps != window[0]:
        receipt["status"] = f"FAILED: replay stopped at step {scene.steps}"
        write_json(args.output / "receipt.json", receipt)
        return 1
    wp.synchronize_device(device)
    receipt["fixed_state_digest"] = _digest(scene.state)
    with tempfile.TemporaryDirectory(prefix="p9d-bench-") as tmp:
        checkpoint = Path(tmp) / "window_start.npz"
        scene.solver.save_checkpoint(scene.state, checkpoint)
        plan = [("W0", "timed")] + [(f"W{i}", "timed") for i in range(1, args.passes + 1)]
        plan += [(f"I_{kind}", kind) for kind in ("segments", "syncs") if kind in instrumented]
        for name, kind in plan:
            if name != "W0":
                scene.solver.load_checkpoint(scene.state, checkpoint)
                scene.steps = window[0]
            entry = {"gpu_before": _gpu(), "graphs_before": _graph_counters()}
            if kind == "timed":
                entry.update(_summarize(_run_window(scene, window, device)))
            elif kind == "segments":
                entry.update(_instrument_segments(scene, window, device))
            else:
                entry.update(_instrument_syncs(scene, window, device))
            wp.synchronize_device(device)
            entry["end_digest"] = _digest(scene.state)
            entry["gpu_after"], entry["graphs_after"] = _gpu(), _graph_counters()
            receipt["passes"][name] = entry
    w0 = receipt["passes"]["W0"]
    receipt["repeats_reproduce_W0"] = {
        name: entry["per_step_counts"] == w0["per_step_counts"] and entry["end_digest"] == w0["end_digest"]
        for name, entry in receipt["passes"].items()
        if name.startswith("W") and name != "W0"
    }
    receipt["status"] = "COMPLETE"
    write_json(args.output / "receipt.json", receipt)
    print(
        json.dumps(
            {k: receipt["passes"][k]["solver_step"]["median_ms"] for k in receipt["passes"] if k.startswith("W")}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
