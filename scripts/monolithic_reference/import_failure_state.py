# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Continue a saved monolithic failure state in the current solver; diagnostic only.

Older checkpoints carry configuration identities that ``load_checkpoint``
rejects by design. This tool copies only the coordinates and integration
history by field name, converts them to the solver precision and validates the
result; the state never passes through the checkpoint path. It first checks
every field shape and that the saved state belongs to the rebuilt scene: its
actor layout, a recorded ``model.npz`` or a recorded ``initial.npz`` must
match. Continuation receipts are diagnostics, not acceptance evidence.
"""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from scripts.monolithic_reference.retained_assets import ASSET_ROOT
from scripts.monolithic_reference.state_snapshots import (  # noqa: F401
    IDENTITY_FIELDS,
    INITIAL_COORDINATES,
    MODEL_ARRAYS,
    _sha,
    check_scene,
    import_state,
    read_source,
)

ROOT = Path(__file__).resolve().parents[2]
SHARPA_REFERENCE = ASSET_ROOT / "data/reference/sharpa-recorded-10mm--ArmijoOrResNorm--C-IE--production--graph1--fp64"
CASES = {
    # case: (default source, default recording step); P7 failures are runner state files.
    "sharpa-3444": (SHARPA_REFERENCE, 3443),
    "box-1736": (None, None),
    "helix-1440-ea10-2001": (None, None),
}


def _material_case(precision, device):
    from scripts.monolithic_reference.material_scene import MaterialScene  # noqa: PLC0415

    case = MaterialScene("precreased-box", device=device, precision=precision)
    return case, {"recipe": str(case.options["recipe"]), "recipe_sha256": _sha(case.options["recipe"])}


def _helix_case(precision, device):
    from scripts.monolithic_reference.cable_benchmarks import CableBenchmark  # noqa: PLC0415

    case = CableBenchmark("localized_helix", segments=1440, ea_multiplier=10, device=device, precision=precision)
    return case, {"recipe_sha256": _sha(Path(__file__).with_name("cable_recipes.json"))}


def _sharpa_case(precision, device):
    from newton.examples.softbody.monolithic_sharpa_recorded import setup, targets  # noqa: PLC0415
    from scripts.monolithic_reference.retained_assets import INPUTS  # noqa: PLC0415

    trajectory = INPUTS / "sharpa/recorded-control.json"
    with wp.ScopedDevice(device):
        scene = setup(SimpleNamespace(trajectory=trajectory, precision=precision))
    case = SimpleNamespace(solver=scene.solver, state=scene.state, steps=0)

    def step():
        q, v, _, _ = targets(scene.trajectory, scene.curve, scene.mapping, (case.steps + 1) * scene.trajectory.dt)
        scene.control.joint_target_q.assign(q)
        scene.control.joint_target_qd.assign(v)
        scene.solver.step(scene.state, scene.state, scene.control, dt=scene.trajectory.dt)
        if scene.solver.last_stats.converged:
            case.steps += 1

    case.step = step
    return case, {"trajectory_sha256": _sha(trajectory)}


BUILDERS = {"sharpa-3444": _sharpa_case, "box-1736": _material_case, "helix-1440-ea10-2001": _helix_case}


def continue_case(name, source, step, *, precision, device, steps):
    """Rebuild the scene, import the saved state and record each continued step."""
    case, scene = BUILDERS[name](precision, device)
    values, directory = read_source(source, step)
    initial = case.solver._states.copy(case.state)
    checks = check_scene(case.solver, initial, values, directory)
    report = import_state(case.solver, case.state, values)
    # Resume the loading program after the saved step; a state file counts its publications.
    case.steps = step if step is not None else int(values["history.force_generation"][0])
    rows = []
    for _ in range(steps):
        try:
            case.step()
        except (AssertionError, RuntimeError):
            pass  # The scene's strict step rejects nonconvergence; the stats below record it.
        stats = case.solver.last_stats
        rows.append(
            {
                "step": case.steps + int(not stats.converged),
                "status": stats.status.value,
                "converged": stats.converged,
                "rolled_back": stats.rolled_back,
                "failure_reason": stats.failure_reason,
                "nonlinear_iterations": stats.nonlinear_iterations,
                "linear_iterations": stats.linear_iterations,
            }
        )
        if stats.rolled_back:
            break
        if not stats.converged:
            case.steps += 1  # A published Stopped step still advances the loading program.
    return {
        "case": name,
        "qualification": "DIAGNOSTIC_ONLY",
        "precision": case.solver.precision.value,
        "device": str(case.solver.model.device),
        "source": str(source),
        "source_step": step,
        "source_schema_version": int(values["history.schema_version"][0]),
        "scene": scene,
        "scene_checks": checks,
        "import": report,
        "first_continued_step": rows[0]["step"] if rows else None,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=sorted(CASES))
    parser.add_argument("--source", type=Path, help="State file or recording directory")
    parser.add_argument("--step", type=int, help="Recording step to import")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5, help="Continued steps")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    default_source, default_step = CASES[args.case]
    source = args.source or default_source
    if source is None:
        parser.error(f"{args.case} requires --source")
    step = args.step if args.step is not None or args.source is not None else default_step
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = continue_case(args.case, source, step, precision=args.precision, device=args.device, steps=args.steps)
    receipt.update(
        commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        dirty=bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip()),
        environment={"python": platform.python_version(), "warp": wp.__version__, "numpy": np.__version__},
    )
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "import"}), flush=True)


if __name__ == "__main__":
    main()
