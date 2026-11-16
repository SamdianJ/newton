# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Per-step SHA-256 traces of feature-off scenes, for bitwise comparison of two checkouts (P9d set X2).

    python -m scripts.monolithic_reference.feature_off_trace trace --output a.json [--scenes ...] [--steps 100]
    python -m scripts.monolithic_reference.feature_off_trace compare a.json b.json

Each step hashes every Warp array of the state (including the solver's private
namespaces), the published contact forces and the Stats counters. Robot
windows and mesh scenes need CUDA; their CPU entries are NOT_APPLICABLE, not
PASS. Run the same command in both checkouts with the same device and precision.
"""

import argparse
import dataclasses
import enum
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

SOURCE = Path(__file__).resolve().parents[2]
FIXTURES = SOURCE / "scripts/monolithic_reference/fixtures"
# (scene, device, precision, use_cuda_graph)
CONFIGS = {
    "quick-windows": (("cuda:0", "fp64", True), ("cuda:0", "fp32", True)),
    "default": (("cpu", "fp64", False), ("cuda:0", "fp32", True)),
}


def _arrays(obj, prefix, seen):
    """Yield (name, numpy array) for every Warp array reachable through plain attributes."""
    if id(obj) in seen:
        return
    seen.add(id(obj))
    for name, value in sorted(vars(obj).items()):
        if isinstance(value, wp.array):
            yield f"{prefix}{name}", value.numpy()
        elif hasattr(value, "__dict__") and not callable(value) and type(value).__module__.startswith("newton"):
            yield from _arrays(value, f"{prefix}{name}.", seen)


def _digest(pairs):
    h = hashlib.sha256()
    for name, value in pairs:
        h.update(name.encode())
        h.update(str(value.dtype).encode() + str(value.shape).encode())
        h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def step_record(solver, state):
    stats = solver.last_stats
    record = {"state": _digest(_arrays(state, "", set()))}
    if not stats.rolled_back:
        forces = solver.get_contact_forces(state)
        record["forces"] = _digest(
            (name, getattr(forces, name)) for name in ("particle_force", "body_wrench", "joint_force", "cable_torque")
        )
    # Every scalar Stats field (repr is exact for floats), except wall-clock timing fields.
    record["stats"] = {
        field.name: repr(getattr(stats, field.name))
        for field in dataclasses.fields(stats)
        if "time" not in field.name
        and isinstance(getattr(stats, field.name), (int, float, str, bool, enum.Enum, type(None)))
    }
    record["stats"]["publication"] = str((stats.variational or {}).get("publication_status"))
    return record


# ---------------------------------------------------------------------------
# Scenes: each factory returns (solver, state, advance) where advance() runs one step.
# ---------------------------------------------------------------------------


def _quick(case_id):
    def build(device, precision, graph):
        from scripts.monolithic_reference.quick_scenes import WINDOWS  # noqa: PLC0415
        from scripts.monolithic_reference.retained_assets import ASSET_ROOT  # noqa: PLC0415
        from scripts.monolithic_reference.state_snapshots import load_snapshot  # noqa: PLC0415

        cases = json.loads((FIXTURES / "quick_cases_v1.json").read_text())["cases"]
        suffix = "-fp32" if precision == "fp32" else ""
        case = next(c for c in cases if c["id"] == case_id + suffix)
        window = WINDOWS[case["scene"]](precision, device)
        values, model_values = load_snapshot(ASSET_ROOT / "data" / case["snapshot"]["path"], case["snapshot"]["sha256"])
        window.start(values, model_values, case["snapshot"]["step"])
        return window.solver, window.state, window.advance

    return build


def _rigid64(device, precision, graph):
    from scripts.monolithic_reference.rigid_stack_scene import stack_scene, stack_solver  # noqa: PLC0415

    case = json.loads((FIXTURES / "rigid_stack_large_v1.json").read_text())
    pitch, (nx, ny) = case["pitch_m"], case["grid"]
    columns = tuple((pitch * (i - (nx - 1) / 2), pitch * (j - (ny - 1) / 2)) for i in range(nx) for j in range(ny))
    model = stack_scene(device, columns=columns, layers=case["layers"])
    solver = stack_solver(
        model,
        spatial=True,
        graph=graph,
        query_capacity=case["query_capacity"],
        support_capacity=case["support_capacity"],
        precision=precision,
    )
    state, control = solver.create_state(dt=case["dt"]), solver.create_control()
    return solver, state, lambda: solver.step(state, state, control, dt=case["dt"])


def _remote_tree(device, precision, graph):
    from scripts.monolithic_reference.run_dexterous_tasks import DT, TaskScene  # noqa: PLC0415

    scene = TaskScene("turn-knob", remote_tree=True, device=device, precision=precision)
    clock = {"t": 0.0}

    def advance():
        clock["t"] += DT
        scene.step(clock["t"])

    return scene.solver, scene.state, advance


def _tet(cells):
    def build(device, precision, graph):
        from scripts.monolithic_reference.tet_benchmarks import TetBenchmark  # noqa: PLC0415

        bench = TetBenchmark(
            "cantilever", cells=cells, device=device, pcg_mode="production", use_cuda_graph=graph, precision=precision
        )
        return bench.solver, bench.state, bench.step

    return build


def _cloth_e5(device, precision, graph):
    from scripts.monolithic_reference.run_cloth_stack import ClothStackScene  # noqa: PLC0415

    recipe = json.loads((FIXTURES / "cloth_stack_table_v1.json").read_text())
    config = json.loads((FIXTURES / "cloth_stack_point_cloud_v1.json").read_text())
    scene = ClothStackScene(
        recipe, device=device, precision=precision, use_cuda_graph=graph, method="point-cloud", method_config=config
    )
    return scene.solver, scene.state, scene.step


def _cable_cantilever(device, precision, graph):
    from scripts.monolithic_reference.cable_benchmarks import CableBenchmark  # noqa: PLC0415

    bench = CableBenchmark(
        "cantilever", device=device, pcg_mode="production", use_cuda_graph=graph, precision=precision
    )
    return bench.solver, bench.state, bench.step


def _cable_contact(device, precision, graph):
    """Cable against a driven sphere (stage material case): the trunk's cable-with-contact scene."""
    from scripts.monolithic_reference.material_scene import MaterialScene  # noqa: PLC0415

    scene = MaterialScene(
        "cable-bend-twist", device=device, pcg_mode="production", use_cuda_graph=graph, precision=precision
    )
    return scene.solver, scene.state, scene.step


SCENES = {
    "quick-sharpa-close": (_quick("sharpa-close"), "quick-windows"),
    "quick-sharpa-lift": (_quick("sharpa-lift"), "quick-windows"),
    "quick-franka-grasp": (_quick("franka-grasp"), "quick-windows"),
    "quick-franka-release": (_quick("franka-release"), "quick-windows"),
    "rigid-stack-64": (_rigid64, "default"),
    "remote-tree": (_remote_tree, "default"),
    "tet-cantilever-2": (_tet((20, 2, 2)), "default"),
    "tet-cantilever-4": (_tet((40, 4, 4)), "default"),
    "cloth-stack-e5": (_cloth_e5, "default"),
    "cable-cantilever": (_cable_cantilever, "default"),
    "cable-bend-twist": (_cable_contact, "default"),
}


def trace(scenes, steps, devices):
    report = {
        "source": str(SOURCE),
        "newton": newton.__file__,
        "warp": wp.__version__,
        "steps": steps,
        "scenes": {},
    }
    for name in scenes:
        factory, kind = SCENES[name]
        for device, precision, graph in CONFIGS[kind]:
            if device not in devices:
                continue
            key = f"{name}/{device}/{precision}"
            try:
                solver, state, advance = factory(device, precision, graph and device.startswith("cuda"))
                records = []
                for _ in range(steps):
                    try:
                        advance()
                        error = None
                    except Exception as failure:  # Keep the hash of the failing step too.
                        error = f"{type(failure).__name__}: {failure}"
                    record = step_record(solver, state)
                    record["error"] = error
                    records.append(record)
                report["scenes"][key] = {"status": "TRACED", "records": records}
            except Exception as failure:
                report["scenes"][key] = {"status": "NOT_RUN", "error": f"{type(failure).__name__}: {failure}"}
    return report


def compare(a, b):
    """Return per-scene IDENTICAL, DIFFERENT (first step and field) or MISSING."""
    result = {}
    for key in sorted(set(a["scenes"]) | set(b["scenes"])):
        x, y = a["scenes"].get(key), b["scenes"].get(key)
        if x is None or y is None or x["status"] != "TRACED" or y["status"] != "TRACED":
            result[key] = {"verdict": "MISSING", "a": x and x["status"], "b": y and y["status"]}
            continue
        if len(x["records"]) != len(y["records"]):
            result[key] = {"verdict": "DIFFERENT", "reason": "step counts differ"}
            continue
        if not x["records"]:
            # A trace without steps (trace --steps 0) compares nothing.
            result[key] = {"verdict": "MISSING", "reason": "no traced steps"}
            continue
        errors = sum(1 for r in x["records"] + y["records"] if r.get("error"))
        # A scene that fails in a checkout proves nothing about bit identity, even if both fail alike.
        verdict = {"verdict": "ERRORS" if errors else "IDENTICAL", "steps": len(x["records"]), "error_records": errors}
        if set(x["records"][0]["stats"]) != set(y["records"][0]["stats"]):
            # A Stats field added or removed between checkouts is not a numeric difference.
            verdict["stats_fields"] = {
                "only_a": sorted(set(x["records"][0]["stats"]) - set(y["records"][0]["stats"])),
                "only_b": sorted(set(y["records"][0]["stats"]) - set(x["records"][0]["stats"])),
            }
        shared = set(x["records"][0]["stats"]) & set(y["records"][0]["stats"])
        for i, (ra, rb) in enumerate(zip(x["records"], y["records"], strict=True)):
            p = dict(ra, stats={k: v for k, v in ra["stats"].items() if k in shared})
            q = dict(rb, stats={k: v for k, v in rb["stats"].items() if k in shared})
            fields = [f for f in ("state", "forces", "stats", "error") if p.get(f) != q.get(f)]
            if fields:
                verdict = {"verdict": "DIFFERENT", "first_step": i, "fields": fields}
                break
        result[key] = verdict
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--output", type=Path, required=True)
    t.add_argument("--scenes", nargs="+", choices=sorted(SCENES), default=sorted(SCENES))
    t.add_argument("--steps", type=int, default=100)
    t.add_argument("--devices", nargs="+", default=["cpu", "cuda:0"])
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    if args.command == "trace":
        if Path(newton.__file__).resolve() != SOURCE / "newton/__init__.py":
            raise RuntimeError(f"Newton import {newton.__file__} does not belong to {SOURCE}")
        report = trace(args.scenes, args.steps, args.devices)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report) + "\n")
        print({k: v["status"] for k, v in report["scenes"].items()})
        return 0
    result = compare(json.loads(args.a.read_text()), json.loads(args.b.read_text()))
    print(json.dumps(result, indent=2))
    return 0 if all(v["verdict"] == "IDENTICAL" for v in result.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
