# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark four certified B3 robot steps with an explicit lambda-zero policy."""

import argparse
import gc
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton.examples.softbody.monolithic_recording import serial, state_arrays
from scripts.monolithic_reference.compare_contact_dataflows import verify_contract
from scripts.monolithic_reference.profiling import capture_count, contact_pair_diagnostics, provenance
from scripts.monolithic_reference.replay_direction import recorded_frame, restore
from scripts.monolithic_reference.retained_assets import validate_inputs
from scripts.monolithic_reference.robot_contact_acceptance import FrozenRobotFactory
from scripts.monolithic_reference.run_precision import build, cases
from scripts.monolithic_reference.run_regression import reference_binding, select_cases
from scripts.monolithic_reference.workload_io import digest, load_arrays, save_arrays, write_json

STEPS = {"franka": (10, 3001), "sharpa": (1001, 4001)}


def numerical_output(run):
    """Read effective numerical output, excluding allocation padding and slot order."""
    c, p = run._contact, run._contact.pool
    n = int(p.records.count.numpy()[0])
    identities = p.records.identities.numpy()[:n]
    order = np.argsort(identities, kind="stable")
    output = {
        "contact.identities": identities[order],
        "contact.pairs": p.records.pairs.numpy()[:n][order],
        "contact.force": p.force.result.numpy(),
        "contact.residual": c.residual.numpy(),
        "contact.energy": c.energy.numpy(),
        "residual": run._complete_r.numpy(),
    }
    for name, matrix in (("global", run._linear.k_global_scalar_bsr), ("internal", run._linear.ax_internal_bsr3)):
        offsets = matrix.offsets.numpy()
        count = int(offsets[-1])
        output.update(
            {
                name + ".offsets": offsets,
                name + ".columns": matrix.columns.numpy()[:count],
                name + ".values": matrix.values.numpy()[:count],
            }
        )
    return output


def main():
    """Prepare authenticated local fixtures or measure their unchanged physical recipe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=STEPS, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--recording-root", type=Path, help="Required only when preparing fixtures")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--no-contact-diagnostics",
        action="store_true",
        help="Skip host pair reports during step timing; retain physical forces and validation",
    )
    args = parser.parse_args()
    if not 1 <= args.samples <= 10 or args.prepare != (args.recording_root is not None):
        parser.error("Use 1..10 samples and --recording-root only with --prepare")
    before, inputs = provenance(), validate_inputs()
    scene = "franka-cube" if args.scene == "franka" else "sharpa-recorded-10mm"
    spec = next(
        c
        for c in cases()
        if c["scene"] == scene
        and c["precision"] == "fp64"
        and c["law"] == "C-IE"
        and c["globalization"] == "ArmijoOrResNorm"
    )
    factory = FrozenRobotFactory(scene, dataflow="spatial")
    model, run, state, controls = build(spec, solver_factory=factory)
    retained = select_cases("nightly", [spec["reference_case"]])[0]
    _, control_path, _, _, _ = reference_binding(retained)
    np.testing.assert_array_equal(controls, np.load(control_path, allow_pickle=False))
    case = SimpleNamespace(model=model, solver=run, state=state, control=model.control())
    original = run._config
    complete = factory.recorder.audit.complete
    run._config = complete._config = replace(original, regularization_values=(0.0,))
    factory.recorder.audit.contract = verify_contract(complete, run)
    policy = {"constructor_digest": run._kinematics.digest.hex(), "effective_config": serial(asdict(run._config))}
    policy_hash = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    directory = args.fixtures / args.scene
    manifest_path = directory / "manifest.json"
    if args.prepare:
        directory.mkdir(parents=True, exist_ok=False)
        recording = args.recording_root / f"{args.scene}-lambda0"
        receipt = json.loads((recording / "receipt.json").read_text())
        root = Path(before["root"])
        for name, expected in receipt["source_hashes"].items():
            if digest(root / name) != expected:
                raise ValueError(f"Fixture preparation source differs: {name}")
        manifest = {
            "schema": "b3-contact-steps/1",
            "scene": args.scene,
            "policy": policy,
            "policy_sha256": policy_hash,
            "input_manifest_sha256": inputs,
            "source": before,
            "recording_receipt_sha256": digest(recording / "receipt.json"),
            "steps": [],
        }
        for step in STEPS[args.scene]:
            start, start_hash = recorded_frame(recording / "recording", step - 1)
            end, end_hash = recorded_frame(recording / "recording", step)
            entry = {"step": step, "start_chunk_sha256": start_hash, "end_chunk_sha256": end_hash}
            for name, arrays in (("start", start), ("end", end)):
                path = directory / f"{step}-{name}.npz"
                entry[name] = {"file": path.name, "schema": save_arrays(path, arrays), "sha256": digest(path)}
            manifest["steps"].append(entry)
    else:
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["schema"] != "b3-contact-steps/1"
            or manifest["scene"] != args.scene
            or [e["step"] for e in manifest["steps"]] != list(STEPS[args.scene])
        ):
            raise ValueError("Unsupported contact-step fixtures")
        if manifest["policy"] != policy or manifest["policy_sha256"] != policy_hash:
            raise ValueError("Effective physical configuration or lambda policy changed")
        if manifest["input_manifest_sha256"] != inputs:
            raise ValueError("Frozen inputs changed")
    rows = []
    for entry in manifest["steps"]:
        arrays = {}
        for name in ("start", "end"):
            item = entry[name]
            path = (directory / item["file"]).resolve()
            if not path.is_relative_to(directory.resolve()) or digest(path) != item["sha256"]:
                raise ValueError("Fixture payload path/hash differs")
            arrays[name] = load_arrays(path, item["schema"])
        step = entry["step"]
        expected_numerical = None
        if not args.prepare:
            item = entry["numerical"]
            path = directory / item["file"]
            if digest(path) != item["sha256"]:
                raise ValueError("Numerical fixture hash differs")
            expected_numerical = load_arrays(path, item["schema"])
        # Keep deferred scene/fixture setup garbage outside production-step timing.
        # Normal collection remains enabled throughout warmup and measurement.
        gc.collect()
        for repetition in range(-2, args.samples):
            restore(case, arrays["start"])
            case.control.joint_target_q.assign(controls[step, 0])
            case.control.joint_target_qd.assign(controls[step, 1])
            wp.synchronize_device(model.device)
            captures = capture_count(run)
            plan = getattr(run._linear, "_masked_pcg_plan", None)
            with contact_pair_diagnostics(run, enabled=not args.no_contact_diagnostics):
                begin = time.perf_counter()
                run.step(state, state, case.control, dt=spec["dt"])
                wp.synchronize_device(model.device)
                elapsed = (time.perf_counter() - begin) * 1000
            stats = serial(run.last_stats)
            actual = state_arrays(state)
            assert int(actual["history.force_generation"][0]) == stats["step_generation"]
            for name, value in actual.items():
                if name != "history.force_generation":
                    np.testing.assert_array_equal(value, arrays["end"][name], err_msg=name)
            assert not stats["rolled_back"]
            assert all(
                b["residual"] <= b["limit"]
                for p in stats["variational"]["linear_postchecks"]
                for b in p["blocks"].values()
            )
            pool, query = run._contact.evaluation_diagnostics(), run._contact.spatial_diagnostics()
            assert (
                not pool["tainted"] and not pool["dropped"] and not query["tainted"] and not query["capacity_dropped"]
            )
            numerical = numerical_output(run)
            if expected_numerical is None:
                expected_numerical = numerical
                path = directory / f"{step}-numerical.npz"
                entry["numerical"] = {"file": path.name, "schema": save_arrays(path, numerical), "sha256": digest(path)}
            for name, value in numerical.items():
                np.testing.assert_array_equal(value, expected_numerical[name], err_msg=name)
            row = {
                "step": step,
                "repetition": repetition,
                "wall_ms": elapsed,
                "new_captures": capture_count(run) - captures,
                "pcg_plan_changed": plan is not getattr(run._linear, "_masked_pcg_plan", None),
                "stats": stats,
                "pool": pool,
                "query": query,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("step", "repetition", "wall_ms", "new_captures")}), flush=True)
    after = provenance()
    assert before["source_hashes"] == after["source_hashes"] and inputs == validate_inputs()
    if args.prepare:
        write_json(manifest_path, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output,
        {
            "status": "LOCAL_EXACT_NOT_STAGE_QUALIFIED",
            "before": before,
            "after": after,
            "fixture_sha256": digest(manifest_path),
            "policy_sha256": policy_hash,
            "driver_sha256": digest(Path(__file__)),
            "steps": rows,
            "timing_scope": "step + device sync; restore/verification outside timing",
            "contact_pair_diagnostics": not args.no_contact_diagnostics,
        },
    )


if __name__ == "__main__":
    main()
