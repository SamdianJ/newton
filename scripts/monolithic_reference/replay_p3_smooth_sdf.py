# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay all frozen PR3 failures with complete history and recomputed start geometry."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

from scripts.monolithic_reference.build_p3_smooth_sdf import load_qualified_fields
from scripts.monolithic_reference.verify_p3_close import ARTIFACTS, DT, build, targets
from scripts.monolithic_reference.verify_p3_coupled import save, serial, state_arrays
from scripts.monolithic_reference.verify_p3_smooth_sdf import sha256


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for position in (18, 24):
        model, _, _, mapping, curve, terms, _ = build(position)
        field = load_qualified_fields(model, args.qualification, args.representation)
        for law in ("C-EE", "C-IE", "C-II"):
            for mode, graph in (("diagnostic", False), ("production", False), ("production", True)):
                options = {
                    "contact_model": law,
                    "globalization": "ResNorm",
                    "friction_velocity_scale": 0.01,
                    "contact_stiffness": 1e7,
                    "friction_coefficient": 0.5,
                    "normal_smoothing": 1e-4,
                    "material_model": "smith_log_stabilized",
                    "mass_mode": "consistent",
                    "tet_rest_density": wp.full(model.tet_count, 1000.0, dtype=float, device=model.device),
                    "joint_terms": terms,
                    "pcg_mode": mode,
                    "use_cuda_graph": graph,
                }
                baseline_solver = SolverMonolithic._create_coupled_variational(model, **options)
                baseline_digest = baseline_solver._kinematics.digest
                del baseline_solver
                gc.collect()
                solver = SolverMonolithic._create_coupled_variational(model, smooth_sdf=field, **options)
                roots = ["t5-radial-fix-v2"]
                if mode == "production" and graph:
                    roots.append("t5-v1")
                for root in roots:
                    source = ARTIFACTS / "p3/pr3" / root / f"close-{position}-{law}-{mode}-graph{graph}"
                    frozen_path = source / "failure-input.npz"
                    frozen = dict(np.load(frozen_path))
                    source_row = json.loads((source / "failure.json").read_text())["row"]
                    directory = output / f"{root}-{position}-{law}-{mode}-graph{graph}"
                    directory.mkdir()
                    state = solver._create_coupled_state(dt=DT)
                    for field_name, value in frozen.items():
                        name = field_name
                        owner = state
                        if name.startswith("history."):
                            owner, name = state._monolithic_coupled, name[8:]
                        getattr(owner, name).assign(value)
                    if not all(np.array_equal(value, state_arrays(state)[name]) for name, value in frozen.items()):
                        raise ValueError("Incomplete frozen state restoration")
                    if frozen["history.config_digest"].tobytes() != baseline_digest:
                        raise ValueError("Frozen state is not compatible with the unchanged physical configuration")
                    # Explicit diagnostic migration: retain every physical history array,
                    # bind only the new representation identity, invalidate old published forces.
                    state._monolithic_coupled.config_digest.assign(
                        np.frombuffer(solver._kinematics.digest, dtype=np.uint8)
                    )
                    state._monolithic_coupled.force_valid.zero_()
                    imported = state_arrays(state)
                    np.savez_compressed(directory / "imported.npz", **imported)
                    control = model.control()
                    _, qt, vt = targets(curve, mapping, source_row["time"], 23)
                    if not np.array_equal(qt, np.asarray(source_row["target_q"], np.float32)) or not np.array_equal(
                        vt, np.asarray(source_row["target_qd"], np.float32)
                    ):
                        raise ValueError("Frozen control mismatch")
                    control.joint_target_q.assign(qt)
                    control.joint_target_qd.assign(vt)
                    calls = []
                    solve = solver._linear.solve_pcg

                    def measured(*a, _solve=solve, _calls=calls, **kw):
                        result = _solve(*a, **kw)
                        _calls.append(serial(result))
                        return result

                    solver._linear.solve_pcg = measured
                    begin = time.perf_counter()
                    solver.step(state, state, control, dt=DT)
                    elapsed = time.perf_counter() - begin
                    solver._linear.solve_pcg = solve
                    final = state_arrays(state)
                    np.savez_compressed(directory / "final.npz", **final)
                    np.savez_compressed(directory / "accepted.npz", **state_arrays(solver._accepted))
                    np.savez_compressed(directory / "trial.npz", **state_arrays(solver._trial))
                    contact = solver._contact
                    np.savez_compressed(
                        directory / "geometry.npz",
                        start_y=contact.start.y.numpy(),
                        start_phi=contact.start.distance.numpy(),
                        start_g=contact.start.gradient.numpy(),
                        current_y=contact.current.y.numpy(),
                        current_phi=contact.current.distance.numpy(),
                        current_g=contact.current.gradient.numpy(),
                        current_normal=contact.current.soft_normal.numpy(),
                        start_uncertified=contact.start.uncertified.numpy(),
                        current_uncertified=contact.current.uncertified.numpy(),
                    )
                    stats = serial(solver.last_stats)
                    result = {
                        "position": position,
                        "law": law,
                        "mode": mode,
                        "graph": graph,
                        "source": str(source),
                        "source_sha256": sha256(frozen_path),
                        "representation": args.representation,
                        "field_digest": field.digest,
                        "time": source_row["time"],
                        "stats": stats,
                        "success": stats["converged"] and stats["status"] == "success",
                        "contact_flags": contact.flags.numpy(),
                        "start_flags": contact.start_flags.numpy(),
                        "rollback_preserved": all(np.array_equal(final[k], v) for k, v in imported.items()),
                        "migration": {
                            "old_digest": baseline_digest.hex(),
                            "new_digest": solver._kinematics.digest.hex(),
                            "changes": ["history.config_digest", "history.force_valid"],
                            "physical_history_exact": True,
                        },
                        "seconds": elapsed,
                        "actual_config": solver._config,
                        "pcg_calls": calls,
                        "command": sys.argv,
                    }
                    save(directory / "summary.json", result)
                    results.append(result)
                    save(output / "results.json", results)
                    print(
                        position,
                        law,
                        mode,
                        graph,
                        root,
                        result["success"],
                        stats["status"],
                        stats.get("reason"),
                        flush=True,
                    )
                del solver, state, control, solve, measured
                gc.collect()
        del model, field
        gc.collect()
    save(
        output / "summary.json",
        {
            "current_passed": sum(r["success"] for r in results if "t5-radial-fix-v2" in r["source"]),
            "current_total": 18,
            "radial_regression_passed": sum(r["success"] for r in results if "/t5-v1/" in r["source"]),
            "radial_regression_total": 6,
            "failed_rollback_preserved": all(r["rollback_preserved"] for r in results if not r["success"]),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", choices=("c1-hermite", "c2-bspline"), required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
