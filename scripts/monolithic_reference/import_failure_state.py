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
import hashlib
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from scripts.monolithic_reference.retained_assets import ASSET_ROOT

ROOT = Path(__file__).resolve().parents[2]
# Solver and layout identities belong to the new solver, never to the saved state.
IDENTITY_FIELDS = frozenset(
    "history." + name
    for name in ("schema_version", "config_digest", "precision", "layout_version", "layout_digest", "actor_layout")
)
MODEL_ARRAYS = (
    "particle_q",
    "particle_mass",
    "tet_indices",
    "tet_poses",
    "tri_indices",
    "edge_indices",
    "body_mass",
    "body_inertia",
    "body_com",
    "joint_type",
    "joint_parent",
    "joint_child",
    "joint_X_p",
    "joint_X_c",
    "joint_axis",
    "shape_type",
    "shape_body",
    "shape_scale",
    "shape_transform",
)
INITIAL_COORDINATES = ("particle_q", "body_q", "joint_q")
SHARPA_REFERENCE = ASSET_ROOT / "data/reference/sharpa-recorded-10mm--ArmijoOrResNorm--C-IE--production--graph1--fp64"
CASES = {
    # case: (default source, default recording step); P7 failures are runner state files.
    "sharpa-3444": (SHARPA_REFERENCE, 3443),
    "box-1736": (None, None),
    "helix-1440-ea10-2001": (None, None),
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_source(path, step=None):
    """Return one flat saved state and the directory holding its recording sidecars.

    ``path`` is a checkpoint-like ``.npz`` file or a recording directory; a recording
    needs ``step``, where step 0 is its ``initial.npz``.
    """
    path = Path(path)
    if path.is_file():
        if step is not None:
            raise ValueError("A state file holds exactly one state")
        with np.load(path, allow_pickle=False) as data:
            return {name: data[name] for name in data.files}, None
    if step is None or step < 0:
        raise ValueError("A recording requires a nonnegative step")
    if step == 0:
        return read_source(path / "initial.npz")[0], path
    index = path / "trajectory-index.json"
    if index.is_file():
        # Example recordings label every frame with its step.
        for chunk in json.loads(index.read_text())["chunks"]:
            with np.load(path / chunk["path"], allow_pickle=False) as data:
                hits = np.flatnonzero(data["frame_steps"] == step)
                if len(hits):
                    if _sha(path / chunk["path"]) != chunk["sha256"]:
                        raise ValueError("Recording chunk identity differs")
                    return {k: data[k][hits[0]] for k in data.files if not k.startswith("frame_")}, path
        raise ValueError(f"Recording lacks step {step}")
    # Native state chunks list consecutive published steps after the initial state.
    first = 1
    for chunk in json.loads((path / "receipt.json").read_text())["chunks"]:
        if first <= step < first + chunk["frames"]:
            if _sha(path / chunk["file"]) != chunk["sha256"]:
                raise ValueError("Recording chunk identity differs")
            with np.load(path / chunk["file"], allow_pickle=False) as data:
                return {k: data[k][step - first] for k in data.files}, path
        first += chunk["frames"]
    raise ValueError(f"Recording lacks step {step}")


def check_scene(solver, initial, source, directory):
    """Return the checks proving the saved state belongs to the rebuilt scene and solver layout."""
    checks = []
    layout = source.get("history.actor_layout")
    if layout is not None:
        if layout.tobytes() != solver._actors.layout_payload():
            raise ValueError("Saved actor layout differs from the rebuilt scene")
        checks.append("actor_layout")
    if directory is not None and (directory / "model.npz").is_file():
        with np.load(directory / "model.npz", allow_pickle=False) as saved:
            for name in MODEL_ARRAYS:
                actual = getattr(solver.model, name, None)
                if (name in saved.files) != isinstance(actual, wp.array) or (
                    name in saved.files and not np.array_equal(saved[name], actual.numpy())
                ):
                    raise ValueError(f"Recorded model differs from the rebuilt scene: {name}")
        checks.append("model_arrays")
    if directory is not None and (directory / "initial.npz").is_file():
        with np.load(directory / "initial.npz", allow_pickle=False) as saved:
            for name in INITIAL_COORDINATES:
                actual = getattr(initial, name).numpy()
                if saved[name].shape != actual.shape or not np.array_equal(saved[name].astype(actual.dtype), actual):
                    raise ValueError(f"Recorded initial state differs from the rebuilt scene: {name}")
        checks.append("initial_state")
    if not checks:
        raise ValueError("Saved state carries no layout, model or initial state to validate the scene")
    return checks


def _contact_free(name, source):
    """Whether an absent published contact force must be zero because no saved contact force is nonzero."""
    forces = [value for key, value in source.items() if key.startswith("history.contact_")]
    return name.startswith("history.contact_") and bool(forces) and not any(np.any(value) for value in forces)


def import_state(solver, state, source):
    """Write saved coordinates and history by name into a validated copy, then replace ``state``."""
    states = solver._states
    scratch = states.copy(state)
    history = getattr(scratch, states.namespace)
    targets = {name: value for name, value in vars(scratch).items() if isinstance(value, wp.array)}
    targets.update({"history." + name: getattr(history, name) for name in states.schema})
    unknown = source.keys() - targets.keys() - IDENTITY_FIELDS
    if unknown:
        raise ValueError(f"Saved state has unknown fields: {sorted(unknown)}")
    report = {"imported": [], "converted": [], "absent_empty": [], "zero_filled": []}
    for name, array in targets.items():
        if name in IDENTITY_FIELDS:
            continue
        expected = array.numpy()
        if name not in source:
            if not expected.size:
                report["absent_empty"].append(name)
            elif _contact_free(name, source):
                # Older schemas lack newer contact outputs; without any saved contact they are zero.
                array.zero_()
                report["zero_filled"].append(name)
            else:
                raise ValueError(f"Saved state lacks {name}")
            continue
        value = source[name]
        if value.shape != expected.shape or value.dtype.kind != expected.dtype.kind:
            raise ValueError(f"Saved field {name} has {value.dtype}{value.shape}, expected {expected.dtype}{expected.shape}")
        if value.dtype != expected.dtype:
            report["converted"].append(name)
        array.assign(value.astype(expected.dtype))
        report["imported"].append(name)
    states.validate(scratch)
    state.assign(scratch)
    return report


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
