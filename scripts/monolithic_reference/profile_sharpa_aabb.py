# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay identical real-asset candidates and compare AABB/full-table physics."""

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline

import newton
from newton.examples.softbody.monolithic_sharpa_assets import load_hand_ball
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_fixture, sha256
from scripts.monolithic_reference.run_sharpa_assets import run as run_short


def records(contacts):
    tids = contacts.soft_contact_tids.numpy()
    keys = np.flatnonzero(tids >= 0)
    return keys, np.concatenate(
        [
            getattr(contacts, name).numpy()[tids[keys]]
            for name in ("soft_contact_barycentric", "soft_contact_normal", "soft_contact_body_pos")
        ],
        axis=1,
    )


def profile_candidates(args, ball):
    """Measure five warmed replays; all comparisons use the same candidate arrays."""
    parameters = load_fixture("newton/examples/softbody/sharpa_g1h.json")
    model, manifest = load_hand_ball(
        args.asset_dir, args.contact_dir, ball, device=args.device, parameters=parameters, position=(0.025, -0.03, 0.19)
    )
    trajectory = ClosureTrajectory(
        args.trajectory,
        manifest["joint_names"],
        model.joint_limit_lower.numpy(),
        model.joint_limit_upper.numpy(),
        model.joint_velocity_limit.numpy(),
    )
    pipelines = [
        MonolithicCollisionPipeline(model, soft_contact_gap=0.002, _enable_aabb=flag, _sdf_query_error=0.0004)
        for flag in (True, False)
    ]
    contacts = [p.contacts() for p in pipelines]
    state = model.state()
    rest = state.particle_q.numpy()
    candidates = []
    for t in (0.0, 0.5, 1.0, 1.5, 2.0, 4.0):
        for displacement in ((0, 0, 0), (0.004, -0.003, 0.002), (0.04, 0.02, 0), (1, 1, 1)):
            q, _ = trajectory.sample(t)
            state.joint_q.assign(q.astype(np.float32))
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            candidates.append((state.body_q.numpy(), (rest + displacement).astype(np.float32), t, displacement))
    timings, observations = [[], []], []
    for repeat in range(6):
        for index, (body_q, x, t, displacement) in enumerate(candidates):
            state.body_q.assign(body_q)
            state.particle_q.assign(x)
            actual = []
            for mode, (pipeline, buffer) in enumerate(zip(pipelines, contacts, strict=True)):
                wp.synchronize_device(model.device)
                start = time.perf_counter()
                pipeline.collide(state, buffer)
                wp.synchronize_device(model.device)
                elapsed = time.perf_counter() - start
                if repeat:
                    timings[mode].append(elapsed)
                actual.append(records(buffer))
            np.testing.assert_array_equal(actual[0][0], actual[1][0])
            np.testing.assert_array_equal(actual[0][1], actual[1][1])
            if repeat == 1:
                counts = pipelines[0]._query_counts.numpy().tolist()
                if displacement == (1, 1, 1):
                    assert counts[2:] == [0, 0], counts
                observations.append(
                    {
                        "candidate": index,
                        "source_time": t,
                        "offset": displacement,
                        "contacts": len(actual[0][0]),
                        "aabb_counts": counts,
                        "full_counts": pipelines[1]._query_counts.numpy().tolist(),
                    }
                )
    # Separate instrumented bounds-update timing; not added to the plain collision timings.
    update_times = []
    for _ in range(5):
        for body_q, x, _, _ in candidates:
            state.body_q.assign(body_q)
            state.particle_q.assign(x)
            wp.synchronize_device(model.device)
            start = time.perf_counter()
            pipelines[0]._bounds.update(state, 0.0022, pipelines[0]._status)
            wp.synchronize_device(model.device)
            update_times.append(time.perf_counter() - start)
    return {
        "passed": True,
        "manifest": manifest,
        "repeats": 5,
        "candidate_count": len(candidates),
        "ball_sha256": sha256(ball),
        "trajectory_sha256": sha256(args.trajectory),
        "static_pairs": pipelines[0].soft_contact_pair_count,
        "capacity": pipelines[0].soft_contact_max,
        "observations": observations,
        "collision_seconds_p50_p95": [np.percentile(t, [50, 95]).tolist() for t in timings],
        "bounds_update_seconds_p50_p95": np.percentile(update_times, [50, 95]).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", required=True)
    parser.add_argument("--contact-dir", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinements", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--candidates-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = []
    for refinement in args.refinements:
        ball = Path(__file__).with_name("fixtures") / "soft_ball" / f"ball_r{refinement}.npz"
        result = profile_candidates(args, ball)
        (args.output / f"candidates-r{refinement}.json").write_text(json.dumps(result, indent=2) + "\n")
        times = [[], []]
        if not args.candidates_only:
            for repeat in range(5):
                outputs = []
                for mode, disabled in enumerate((False, True)):
                    output = args.output / f"step-r{refinement}-{repeat}-{'off' if disabled else 'on'}"
                    gc.collect()
                    run_short(
                        SimpleNamespace(
                            asset_dir=args.asset_dir,
                            contact_dir=args.contact_dir,
                            ball=ball,
                            device=args.device,
                            position=[0.025, -0.03, 0.19],
                            steps=20,
                            disable_aabb=disabled,
                            output=output,
                        )
                    )
                    outputs.append(output)
                    trace = json.loads((output / "trace.json").read_text())
                    times[mode].extend(r["step_seconds"] for r in trace[1:])
                with np.load(outputs[0] / "state.npz") as a, np.load(outputs[1] / "state.npz") as b:
                    for field in a.files:
                        absolute = 1e-6
                        if field == "v":
                            absolute = (
                                2
                                * float(np.spacing(np.float32(max(np.max(np.abs(a["x"])), np.max(np.abs(b["x"]))))))
                                / 0.001
                            )
                        np.testing.assert_allclose(a[field], b[field], rtol=2e-5, atol=absolute)
                traces = [json.loads((out / "trace.json").read_text()) for out in outputs]
                for field in ("normal_force_n", "tangential_force_n", "deformation_rms_m"):
                    np.testing.assert_allclose(
                        [r[field] for r in traces[0]], [r[field] for r in traces[1]], rtol=2e-5, atol=1e-6
                    )
            result["step_seconds_p50_p95"] = [np.percentile(t, [50, 95]).tolist() for t in times]
        report.append({"refinement": refinement, **result})
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"refinement": refinement, "passed": True}), flush=True)


if __name__ == "__main__":
    main()
