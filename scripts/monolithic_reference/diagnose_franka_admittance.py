# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture and replay one Franka admittance step with native history and controller state."""

import argparse
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.monolithic_franka import REPO, sha
from newton.examples.softbody.monolithic_paper_box import PaperBoxMetrics
from newton.examples.softbody.monolithic_recording import save, state_arrays
from scripts.monolithic_reference.run_franka_admittance import FrankaAdmittance
from scripts.monolithic_reference.workload_pcg import pcg_input


def observe_linear(solver, directory, direction_iteration=None):
    """Read each native PCG call without changing its inputs or acceptance policy."""
    workspace = solver._linear
    if workspace._material_operator is not None:
        raise ValueError("Sparse-only probe does not capture matrix-free material operators")
    solve = workspace.solve_pcg
    calls = 0
    captured = {}

    def observed(rhs, y, **kwargs):
        nonlocal calls
        arrays, metadata = pcg_input(workspace, rhs, y, **kwargs)
        if workspace._cloth_preconditioner is not None:
            cloth = workspace._cloth_preconditioner
            arrays.update(cloth_rows=cloth.rows.numpy(), cloth_factors=cloth.factors.numpy())
        result = solve(rhs, y, **kwargs)
        plan = workspace._masked_pcg_plan
        arrays.update(y_final=y.numpy(), true_residual=plan.true_r.numpy(), norms=plan.norms.numpy())
        if kwargs["generation"].nonlinear_iteration == direction_iteration and not captured:
            if not result.output_valid or workspace._lambda != 0:
                raise ValueError("Directional probe requires a qualified zero-regularization PCG output")
            arrays.update(
                tau=solver._criterion.tau.numpy(),
                residual=solver._complete_r.numpy(),
                direction=kwargs["delta"].numpy(),
            )
            if solver._rigid.count or solver._has_tet or solver._cables:
                raise ValueError("Directional decomposition currently covers articulation and cloth only")
            arrays.update(joint_rows=workspace.q_rows.numpy(), joint_tangent=solver._objective.tangent.numpy())
            triplets = workspace._global.values.numpy()
            for block in solver._cloths:
                name = f"cloth_{block.actor.source_index}"
                arrays[name + "_rows"] = block.triplet_rows
                arrays[name + "_columns"] = block.triplet_columns
                arrays[name + "_values"] = triplets[
                    block.triplet_offset : block.triplet_offset + len(block.triplet_rows)
                ]
            captured.update(
                arrays=arrays,
                call=calls,
                state=solver._states._copy_unchecked(solver._accepted),
                increments=tuple(wp.clone(a) for a in solver._steps[id(solver._accepted)]),
            )
        # MAX_ITERATIONS does not publish delta; only y/true_r describe that attempt.
        metadata.update(result=result, native_config=dataclasses.asdict(kwargs["config"]))
        np.savez_compressed(directory / f"linear-{calls:02d}.npz", **arrays)
        save(directory / f"linear-{calls:02d}.json", metadata)
        calls += 1
        return result

    workspace.solve_pcg = observed
    return captured


def probe_direction(solver, captured, directory, dt):
    """After saving the real result, probe one frozen base using native trial updates."""
    if not captured:
        raise ValueError("Requested direction iteration was not reached")
    a = captured["arrays"]
    n = len(a["scale"])
    if n > 1000:
        raise ValueError("The small dense comparison is restricted to at most 1000 DoFs")
    matrix = np.zeros((n, n))
    for i, (start, end) in enumerate(zip(a["offsets"][:-1], a["offsets"][1:], strict=True)):
        matrix[i, a["columns"][start:end]] = a["values"][start:end]
    scaled = a["scale"][:, None] * matrix * a["scale"][None, :]
    direct = a["scale"] * np.linalg.solve(scaled, a["rhs_hat"])
    components = {}
    for producer in solver._producers:
        if producer.name == "contact":
            continue
        part = np.zeros_like(matrix)
        if producer.name == "articulation":
            part[np.ix_(a["joint_rows"], a["joint_rows"])] = a["joint_tangent"]
        else:
            name = producer.name
            np.add.at(part, (a[name + "_rows"], a[name + "_columns"]), a[name + "_values"])
        components[producer.name] = part
    components["contact"] = matrix - sum(components.values())
    parts = {}
    for producer in solver._producers:
        evaluate = producer.evaluate

        def observed(owner, state, step_dt, residual, *, build_tangent, original=evaluate, name=producer.name):
            before = residual.numpy()
            result = original(owner, state, step_dt, residual, build_tangent=build_tangent)
            parts[name] = residual.numpy() - before
            return result

        producer.evaluate = observed
    solver._accepted.assign(captured["state"])
    for target, source in zip(solver._steps[id(solver._accepted)], captured["increments"], strict=True):
        wp.copy(target, source)
    solver._criterion.tau.assign(a["tau"])
    base_energy, base_residual, _ = solver._evaluate(solver._accepted, dt, build_tangent=False)
    np.testing.assert_array_equal(base_residual, a["residual"])
    base_merit = float(np.sqrt(np.mean((base_residual / a["tau"]) ** 2)))
    rows, vectors = [], {"base_residual": base_residual, "tau": a["tau"], "direct_direction": direct}
    for name, direction in (("pcg", a["direction"]), ("direct", direct)):
        solver._delta.assign(direction)
        vectors[name + "_Bv"] = matrix @ direction
        for alpha in (1.0, 0.0078125, 0.001, 0.0001, 0.00001, -0.00001, -0.0001, -0.001):
            solver._take_trial(alpha, dt)
            if not solver._domain(solver._trial):
                raise ValueError("Directional probe left the original physical domain")
            energy, residual, _ = solver._evaluate(solver._trial, dt, build_tangent=False)
            vectors[f"{name}_{alpha}_residual"] = residual
            for component, value in parts.items():
                vectors[f"{name}_{alpha}_{component}"] = value
            merit = float(np.sqrt(np.mean((residual / a["tau"]) ** 2)))
            rows.append(
                {
                    "direction": name,
                    "alpha": alpha,
                    "energy_change": energy - base_energy,
                    "merit": merit,
                    "merit_change": merit - base_merit,
                }
            )
    derivatives = []
    for name in ("pcg", "direct"):
        for h in (0.001, 0.0001, 0.00001):
            derivative = (vectors[f"{name}_{h}_residual"] - vectors[f"{name}_{-h}_residual"]) / (2 * h)
            predicted = vectors[name + "_Bv"]
            derivatives.append(
                {
                    "direction": name,
                    "h": h,
                    "relative_Bv_error": float(np.linalg.norm(derivative - predicted) / np.linalg.norm(predicted)),
                    "actual_merit_slope": float(np.mean(base_residual * derivative / a["tau"] ** 2) / base_merit),
                    "predicted_merit_slope": float(np.mean(base_residual * predicted / a["tau"] ** 2) / base_merit),
                }
            )
            direction = a["direction"] if name == "pcg" else direct
            derivatives[-1]["components"] = {
                component: {
                    "error_over_full_Bv_norm": float(
                        np.linalg.norm(
                            (vectors[f"{name}_{h}_{component}"] - vectors[f"{name}_{-h}_{component}"]) / (2 * h)
                            - block @ direction
                        )
                        / np.linalg.norm(predicted)
                    ),
                    "error_merit_slope": float(
                        np.mean(
                            base_residual
                            * (
                                (vectors[f"{name}_{h}_{component}"] - vectors[f"{name}_{-h}_{component}"]) / (2 * h)
                                - block @ direction
                            )
                            / a["tau"] ** 2
                        )
                        / base_merit
                    ),
                }
                for component, block in components.items()
            }
    save(
        directory / "direction-probe.json",
        {
            "call": captured["call"],
            "base_energy": base_energy,
            "base_merit": base_merit,
            "rows": rows,
            "derivatives": derivatives,
            "scope": "Post-publication local probe; fixed candidates/history/tau; no continuation",
        },
    )
    np.savez_compressed(directory / "direction-probe.npz", **vectors)


def save_start(scene, directory):
    """Save the published step boundary, applied controls and external virtual state."""
    scene.solver.save_checkpoint(scene.state, directory / "start.npz")
    np.savez_compressed(
        directory / "control.npz",
        **{name: value.numpy() for name, value in vars(scene.control).items() if isinstance(value, wp.array)},
    )
    save(
        directory / "controller.json",
        {
            "steps": scene.steps,
            "phase": scene.phase,
            "admittance": scene.admittance.snapshot(),
            "linear_max_iterations": scene.solver.linear_max_iterations,
        },
    )


def restore_start(scene, directory):
    """Restore physical history and feedback generation without resetting either controller."""
    scene.solver.load_checkpoint(scene.state, directory / "start.npz")
    with np.load(directory / "control.npz", allow_pickle=False) as controls:
        for name in controls.files:
            getattr(scene.control, name).assign(controls[name])
    saved = json.loads((directory / "controller.json").read_text())
    scene.steps, scene.phase, scene.failed = saved["steps"], saved["phase"], False
    scene.admittance.restore(saved["admittance"])


def observation(scene, stats, shell):
    """Use the original recorder while retaining the fields needed for prefix comparison."""
    forces = scene.solver.get_contact_forces(scene.state)
    measured = scene.factory.measure(scene.solver, scene.state)
    row = {
        "step": scene.steps,
        "publication": stats.variational["publication_status"],
        "newton_iterations": stats.nonlinear_iterations,
        "force_N": forces.joint_force[7:].tolist(),
        "finger_position_m": scene.state.joint_q.numpy()[7:].tolist(),
        "finger_velocity_m_s": scene.state.joint_qd.numpy()[7:].tolist(),
        "target_position_m": scene.control.joint_target_q.numpy()[7:].tolist(),
        "penetration_m": measured["penetration"],
        "field_penetration_m": measured["field_penetration"],
    }
    if shell is not None:
        row.update(shell.measure(scene.state.particle_q))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--step", type=int, default=1414)
    parser.add_argument("--checkpoint", type=Path, help="Prior capture directory; omit to reconstruct the prefix")
    parser.add_argument("--observe-linear", action="store_true", help="Save native PCG input/output for one replay")
    parser.add_argument("--linear-iterations", type=int, help="Diagnostic budget override, never a baseline PASS")
    parser.add_argument(
        "--window", type=int, default=1, help="Bounded budget-control window; stop at its first STOPPED"
    )
    parser.add_argument(
        "--direction-iteration", type=int, help="Probe a zero-regularization direction after single replay"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.step < 2:
        parser.error("Require a step after an existing published feedback sample")
    if (args.observe_linear or args.linear_iterations is not None) and not args.checkpoint:
        parser.error("Linear probes require a checkpoint replay")
    if args.linear_iterations is not None and args.linear_iterations < 1:
        parser.error("Linear iteration budget must be positive")
    if args.window < 1 or (args.window > 1 and (not args.checkpoint or args.linear_iterations is None)):
        parser.error("A longer window requires a checkpoint and explicit diagnostic budget")
    if args.direction_iteration is not None and (not args.observe_linear or args.window != 1):
        parser.error("Directional probes require --observe-linear and a single replay")
    command = json.loads((args.reference / "command.json").read_text())
    original = [json.loads(line) for line in (args.reference / "trace.jsonl").read_text().splitlines()]
    if args.step + args.window - 1 > len(original):
        parser.error("Reference does not contain the requested step")
    hashes = json.loads((args.reference / "source-hashes.json").read_text())
    if any(sha(REPO / name) != value for name, value in hashes.items()):
        raise ValueError("Reference source differs; do not silently migrate the case")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "tool-source.txt").write_bytes(Path(__file__).read_bytes())
    save(
        args.output / "identity.json",
        {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
            "newton_import": newton.__file__,
            "reference": str(args.reference.resolve()),
            "reference_summary_sha256": sha(args.reference / "summary.json"),
            "source_hashes_sha256": sha(args.reference / "source-hashes.json"),
            "tool_sha256": sha(Path(__file__)),
            "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
            "checkpoint_sha256": {
                name: sha(args.checkpoint / name) for name in ("start.npz", "control.npz", "controller.json")
            }
            if args.checkpoint
            else None,
            "observe_linear": args.observe_linear,
            "linear_iterations_override": args.linear_iterations,
            "window": args.window,
            "direction_iteration": args.direction_iteration,
            "step": args.step,
            "mode": "checkpoint_replay" if args.checkpoint else "prefix_capture",
        },
    )
    scene = FrankaAdmittance(
        command["asset_cache"],
        device=command["device"],
        precision=command["precision"],
        material=command["material"],
        paper_stiffness_scale=command.get("paper_stiffness_scale", 1.0),
        linear_max_iterations=command.get("linear_max_iterations", 200),
        newton_max_iterations=command.get("newton_max_iterations", 10),
        box_smoothing=command.get("box_smoothing", 0.0),
        alignment_policy=command.get("alignment_policy"),
        alignment_smoothing=command.get("alignment_smoothing"),
    )
    save(args.output / "configuration.json", scene.metadata)
    expected_config = json.loads((args.reference / "configuration.json").read_text())
    assert scene.metadata["config_digest"] == expected_config["config_digest"]
    shell = PaperBoxMetrics(scene.model) if command["material"] == "paper-box" else None
    if args.checkpoint:
        saved = json.loads((args.checkpoint / "controller.json").read_text())
        if saved.get(
            "linear_max_iterations", 200
        ) != scene.solver.linear_max_iterations and args.linear_iterations != saved.get("linear_max_iterations", 200):
            raise ValueError("A budget-control checkpoint requires its recorded explicit budget override")
        restore_start(scene, args.checkpoint)
        assert scene.steps == args.step - 1
    else:
        with np.load(args.reference / "initial.npz", allow_pickle=False) as initial:
            for name, value in state_arrays(scene.state).items():
                np.testing.assert_array_equal(value, initial[name], err_msg=name)
    if args.linear_iterations is not None:
        # Change only scheduling capacity after native checkpoint/config validation.
        # PCG history buffers are plan-owned and allocated from the new config.
        scene.solver.linear_max_iterations = args.linear_iterations
        workspace = scene.solver._linear
        workspace.capacities = dataclasses.replace(
            workspace.capacities,
            pcg_max_iterations=args.linear_iterations,
            pcg_true_residual_check_count=args.linear_iterations + 2,
        )
    if args.observe_linear:
        captured = observe_linear(scene.solver, args.output, args.direction_iteration)
    deviations = dict.fromkeys(("force_N", "finger_position_m", "finger_velocity_m_s", "target_position_m"), 0.0)
    frames = np.load(args.reference / "frames.npz", allow_pickle=False)
    latest = args.output / "latest-start"
    if args.window > 1:
        latest.mkdir()
    with (args.output / "prefix.jsonl").open("w") as stream:
        while scene.steps < args.step + args.window - 1:
            if scene.steps == args.step - 1:
                save_start(scene, args.output)
            if args.window > 1:
                save_start(scene, latest)
            stats = scene.step()
            row = observation(scene, stats, shell)
            expected = original[scene.steps - 1]
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if args.linear_iterations is None:
                assert row["publication"] == expected["publication"]
                assert row["newton_iterations"] == expected["newton_iterations"]
            for name in deviations:
                difference = float(np.max(np.abs(np.asarray(row[name]) - expected[name])))
                deviations[name] = max(deviations[name], difference)
                if args.linear_iterations is None:
                    np.testing.assert_allclose(
                        row[name], expected[name], rtol=0, atol=1e-8 if name == "force_N" else 1e-10
                    )
            if scene.steps % 10 == 0 and args.linear_iterations is None:
                index = np.flatnonzero(frames["steps"] == scene.steps)[0]
                np.testing.assert_allclose(
                    scene.state.particle_q.numpy(), frames["particle_q"][index], rtol=0, atol=1e-10
                )
            if scene.steps >= args.step - 5:
                save(args.output / f"stats-{scene.steps}.json", stats)
            if scene.steps % 250 == 0:
                print(json.dumps(row), flush=True)
            if args.window > 1 and row["publication"] != "CONVERGED":
                break
    frames.close()
    scene.solver.save_checkpoint(scene.state, args.output / "final.npz")
    comparison = {
        "matched": True if args.linear_iterations is None else None,
        "max_abs_deviations": deviations,
        "step": scene.steps,
    }
    save(args.output / "comparison.json", comparison)
    if args.direction_iteration is not None:
        if stats.converged or stats.nonlinear_iterations != args.direction_iteration:
            raise ValueError("Probe only the final unresolved iteration, whose candidate set remains current")
        probe_direction(scene.solver, captured, args.output, scene.metadata["dt"])
    print(json.dumps({"status": stats.status.value, **comparison}), flush=True)


if __name__ == "__main__":
    main()
