# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independently audit observer snapshots after native execution has completed."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.p3_articulation_oracle import skew
from scripts.monolithic_reference.p5_scene_oracle import SceneOracle, unpack_symmetric
from scripts.monolithic_reference.run_p5_precision import sha


def error(actual, reference):
    """Measure unshifted maximum and RMS differences, retaining finite failures."""
    delta = np.asarray(actual, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "max": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
    }


def sparse_product(data, vector, values="values"):
    """Apply captured scalar BSR in FP64 without using the production matvec."""
    count = len(vector)
    offsets, columns = data["offsets"], data["columns"]
    result = np.zeros(count)
    for row in range(count):
        lo, hi = offsets[row : row + 2]
        result[row] = data[values].ravel()[lo:hi].astype(float) @ vector[columns[lo:hi]]
    return result


def goals(residual, reference, scale, nq, config):
    """Recompute the original thresholds and expose native scalar-rounding effects."""
    result = {}
    for name, part in [("global", slice(None)), ("q", slice(0, nq)), ("x", slice(nq, None))]:
        size = len(residual[part])
        merit = np.linalg.norm(scale[part] * residual[part]) / np.sqrt(size)
        ref = np.linalg.norm(scale[part] * reference[part]) / np.sqrt(size)
        goal = config["merit_absolute_" + name] + config["merit_relative_" + name] * ref
        result[name] = {
            "merit": float(merit),
            "reference": float(ref),
            "goal": float(goal),
            "ratio": float(merit / goal),
        }
    return result


def audit(directory):
    """Audit actual candidate/history pairs; keep rejected states and first failures."""

    def load(name):
        with np.load(directory / name, allow_pickle=False) as source:
            return dict(source)

    config = json.loads((directory / "physics.json").read_text())
    oracle = SceneOracle(load("model.npz"), load("physics.npz"), load("contact.npz"), load("sdf.npz"), config)
    records = json.loads((directory / "evaluations.json").read_text())
    receipt = json.loads((directory / "receipt.json").read_text())
    mode = receipt["case"]["globalization"]
    with gzip.open(directory / "steps.jsonl.gz", "rt") as stream:
        stats = {r["step"]: r for r in map(json.loads, stream)}
    cache, output, last_scale, native_reference, reference = {}, [], None, None, None
    saved = []
    for row in records:
        scale_rounding = None
        assert sha(directory / row["file"]) == row["sha256"]
        data = load(row["file"])
        data["dt"] = row["dt"]

        # Reuse only identical q/x in the same frozen step, never another trajectory.
        def evaluate(prefix, row=row, data=data):
            key = (row["step"], data[prefix + "joint_q"].tobytes(), data[prefix + "particle_q"].tobytes())
            if key not in cache:
                cache[key] = oracle.evaluate(data, prefix)
            return cache[key]

        result = evaluate("state.")
        if row["role"] == "original":
            reference = result["residual"]
            native_reference = data.get("native.residual")
            last_scale = None
        if "scale" in row:
            assert sha(directory / row["scale"]["file"]) == row["scale"]["sha256"]
            scale_data = load(row["scale"]["file"])
            last_scale = scale_data["scale"].astype(float)
            diagonal = np.zeros(len(last_scale))
            for i in range(len(diagonal)):
                lo, hi = scale_data["offsets"][i : i + 2]
                diagonal[i] = scale_data["diagonal"].ravel()[lo:hi][scale_data["columns"][lo:hi] == i][0]
            rebuilt_scale = 1 / np.sqrt(
                np.maximum(diagonal, config["config"]["epsilon_d"] * scale_data["dynamic"].astype(float))
            )
            scale_rounding = error(last_scale, rebuilt_scale)
        norms = data["native.composed_norms"][oracle.shapes].astype(float)
        bad = np.flatnonzero(abs(norms - 1) > 1e-10)
        item = dict(
            **row,
            energy=result["energy"],
            scale_rounding=scale_rounding,
            min_det_f=result["min_det_f"],
            penetration=result["penetration"],
            independent_geometry_valid=result["geometry_valid"],
            independent_local_domain=result["local_domain"],
            native_quaternion_norm_max_error=float(max(abs(norms - 1))),
            native_quaternion_bad_shapes=oracle.shapes[bad].tolist(),
            predicted_com_error=error(data["objective.predicted_t"], result["predicted_t"]),
            predicted_rotation_error=error(data["objective.predicted_r"], result["predicted_r"]),
        )
        if "native.residual" in data:
            item["residual_error"] = error(data["native.residual"], result["residual"])
            item["energy_error"] = abs(float(data["native.energy"]) - result["energy"])
            if last_scale is not None and reference is not None:
                item["independent_goals"] = goals(
                    result["residual"], reference, last_scale, oracle.k.n, config["config"]
                )
                if native_reference is not None:
                    item["native_residual_fp64_goals"] = goals(
                        data["native.residual"].astype(float),
                        native_reference.astype(float),
                        last_scale,
                        oracle.k.n,
                        config["config"],
                    )
        if row["role"] == "trial" and "native.residual" in data:
            base = evaluate("base.")
            effective = np.r_[
                (data["state.joint_q"].astype(float) - data["base.joint_q"].astype(float)),
                (data["state.particle_q"].astype(float) - data["base.particle_q"].astype(float))[oracle.ids].ravel(),
            ]
            slope = base["residual"] @ effective / row["alpha"]
            if last_scale is not None:
                bm = np.linalg.norm(last_scale * base["residual"]) / np.sqrt(len(effective))
                tm = np.linalg.norm(last_scale * result["residual"]) / np.sqrt(len(effective))
                # Fixed protocol constants, recomputed in FP64 without enlarging gates.
                direction = data["native.direction"].astype(float)
                eps = 128 * np.finfo(float).eps
                en = max(5.820766091346741e-11, eps * max(1, base["absolute"], result["absolute"]))
                rn = max(
                    2.0935893019213836e-9 * np.linalg.norm(last_scale) / np.sqrt(len(effective)), eps * max(1, bm, tm)
                )
                sn = max(
                    1.5498947892952669e-9 * np.linalg.norm(direction),
                    2.0935893019213836e-9 * abs(direction).sum(),
                    eps * max(1, abs(base["residual"] * direction).sum()),
                )
                a = bool(slope < -sn and result["energy"] <= base["energy"] + 1e-4 * row["alpha"] * slope + en)
                r = bool(tm <= bm + rn)
                item["independent_acceptance"] = {
                    "slope": float(slope),
                    "energy": a,
                    "residual": r,
                    "passed": bool(a if mode == "Armijo" else r if mode == "ResNorm" else a or r),
                    "energy_noise": en,
                    "residual_noise": rn,
                    "slope_noise": sn,
                }
        output.append(item)
        saved.append(result)
    # Native trial records are logged only for valid-domain evaluations.
    trial_disagreements = []
    for step, step_stats in stats.items():
        rows = [r for r in output if r["step"] == step and r["role"] == "trial"]
        native = step_stats["stats"]["variational"]["trials"]
        position = 0
        for row in rows:
            if row["error"] is not None:
                continue
            if not row["independent_local_domain"]:
                continue
            while position < len(native) and "alpha" not in native[position]:
                position += 1
            if position == len(native):
                row["native_trial_mapping"] = "UNRESOLVED"
                continue
            current = native[position]
            position += 1
            assert current["alpha"] == row["alpha"]
            row["native_acceptance"] = current
            if "independent_acceptance" in row:
                mismatch = bool(current["accepted"] and not row["independent_acceptance"]["passed"])
                row["accepted_but_independent_rejects"] = mismatch
                if mismatch:
                    trial_disagreements.append(row["file"])
    linear = []
    linear_positions = {}
    for row in json.loads((directory / "linear.json").read_text()):
        assert sha(directory / row["file"]) == row["sha256"]
        data = load(row["file"])
        direction = data["direction"].astype(float)
        ref = data["scale"].astype(float) * (sparse_product(data, direction) + data["residual"].astype(float))
        candidates = [
            (i, load(r["file"]))
            for i, r in enumerate(records)
            if r["step"] == row["step"] and r["error"] is None and r["role"] == "accepted"
        ]
        matches = [i for i, d in candidates if np.array_equal(d["native.residual"], data["residual"])]
        assert matches, "Physical posterior has no matching frozen candidate"
        independent = data["scale"].astype(float) * (sparse_product(data, direction) + saved[matches[0]]["residual"])
        position = linear_positions.get(row["step"], 0)
        postcheck = stats[row["step"]]["stats"]["variational"]["linear_postchecks"][position]
        linear_positions[row["step"]] = position + 1
        blocks = {}
        for name, part in [("global", slice(None)), ("q", slice(0, oracle.k.n)), ("x", slice(oracle.k.n, None))]:
            limit = postcheck["blocks"][name]["limit"]
            norm = float(np.linalg.norm(independent[part]))
            blocks[name] = {"independent_residual": norm, "frozen_native_limit": limit, "passed": norm <= limit}
        row["independent_residual_posterior"] = blocks
        linear.append(
            dict(
                **row,
                native_posterior_error=error(data["native_error"], ref),
                independent_norm=float(np.linalg.norm(ref)),
            )
        )
    publications = []
    receipt = json.loads((directory / "receipt.json").read_text())
    frames = []
    for chunk in receipt["chunks"]:
        assert sha(directory / chunk["file"]) == chunk["sha256"]
        block = load(chunk["file"])
        frames.extend({k: v[i] for k, v in block.items()} for i in range(chunk["frames"]))
    for step, state in enumerate(frames, 1):
        native = stats[step]
        if native["stats"]["rolled_back"]:
            publications.append(
                {"step": step, "publication": "HARD_FAILURE", "rollback_verified": receipt["rollback_verified"]}
            )
            continue
        matches = [
            i for i, r in enumerate(records) if r["step"] == step and r["role"] == "accepted" and r["error"] is None
        ]
        assert matches
        i = matches[-1]
        data, result = load(records[i]["file"]), saved[i]
        np.testing.assert_array_equal(data["state.joint_q"], state["joint_q"])
        np.testing.assert_array_equal(data["state.particle_q"], state["particle_q"])
        old_t, old_r, *_ = oracle.k.evaluate(data["original.joint_q"].astype(float))
        h = records[i]["dt"]
        rate = (result["rotation"] - old_r) @ result["rotation"].transpose(0, 2, 1) / h
        expected_omega = (
            np.stack(
                (rate[:, 2, 1] - rate[:, 1, 2], rate[:, 0, 2] - rate[:, 2, 0], rate[:, 1, 0] - rate[:, 0, 1]), axis=1
            )
            / 2
        )
        pub = {
            "step": step,
            "publication": native["publication"],
            "independent_goals": output[i].get("independent_goals"),
            "velocity_error": error(
                state["particle_qd"],
                (state["particle_q"].astype(float) - data["original.particle_q"].astype(float)) / h,
            ),
            "joint_velocity_error": error(
                state["joint_qd"], (state["joint_q"].astype(float) - data["original.joint_q"].astype(float)) / h
            ),
            "link_v_error": error(state["history.link_v_com"], (result["com"] - old_t) / h),
            "link_omega_error": error(state["history.link_omega"], expected_omega),
            "link_vsym_error": error(
                unpack_symmetric(state["history.link_vsym"]), (rate + rate.transpose(0, 2, 1)) / 2
            ),
            "particle_force_error": error(state["history.contact_particle_force"], result["particle_force"]),
            "body_wrench_error": error(state["history.contact_body_force"], result["body_wrench"]),
            "joint_force_error": error(state["history.contact_joint_force"], result["joint_force"]),
        }
        joint_v = np.zeros_like(state["history.joint_v"], dtype=float)
        joint_rate = np.zeros((len(joint_v), 3, 3))
        for j, kind in enumerate(oracle.k.joint_type):
            if kind == 3:
                continue
            coord = oracle.k.joint_q_start[j]
            axis = oracle.k.axes[coord]
            if kind == 0:
                joint_v[j] = axis * (float(state["joint_q"][coord]) - float(data["original.joint_q"][coord])) / h
            elif kind == 1:
                cross = skew(axis)
                old_angle = float(data["original.joint_q"][coord])
                angle = float(state["joint_q"][coord])
                old_rotation = np.eye(3) + np.sin(old_angle) * cross + (1 - np.cos(old_angle)) * cross @ cross
                rotation = np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross
                joint_rate[j] = (rotation - old_rotation) @ rotation.T / h
        joint_omega = (
            np.stack(
                (
                    joint_rate[:, 2, 1] - joint_rate[:, 1, 2],
                    joint_rate[:, 0, 2] - joint_rate[:, 2, 0],
                    joint_rate[:, 1, 0] - joint_rate[:, 0, 1],
                ),
                axis=1,
            )
            / 2
        )
        pub["joint_history_v_error"] = error(state["history.joint_v"], joint_v)
        pub["joint_history_omega_error"] = error(state["history.joint_omega"], joint_omega)
        pub["joint_history_vsym_error"] = error(
            unpack_symmetric(state["history.joint_vsym"]), (joint_rate + joint_rate.transpose(0, 2, 1)) / 2
        )
        pub["contact_impulse_error"] = error(
            state["history.contact_particle_force"].astype(float) * h, result["particle_force"] * h
        )
        pub["penetration"] = result["penetration"]
        pub["min_det_f"] = result["min_det_f"]
        pub["independent_masked_queries"] = int(result["mask"].sum())
        pub["native_goals"] = {
            name: {
                key: native["stats"].get(field.format(suffix=suffix))
                for key, field in [
                    ("merit", "merit{suffix}_final"),
                    ("reference", "merit{suffix}_reference"),
                    ("ratio", "convergence_ratio{suffix}"),
                ]
            }
            for name, suffix in [("global", ""), ("q", "_q"), ("x", "_x")]
        }
        # Scalar target rounding is reported separately from residual error.
        for name, suffix in [("global", ""), ("q", "_q"), ("x", "_x")]:
            ref = native["stats"].get("merit" + suffix + "_reference")
            if ref is not None:
                dtype = state["joint_q"].dtype.type
                native_goal = dtype(config["config"]["merit_absolute_" + name]) + dtype(
                    config["config"]["merit_relative_" + name]
                ) * dtype(ref)
                promoted_goal = config["config"]["merit_absolute_" + name] + config["config"][
                    "merit_relative_" + name
                ] * float(ref)
                pub["native_goals"][name]["native_goal"] = float(native_goal)
                pub["native_goals"][name]["target_rounding"] = float(native_goal) - promoted_goal
        pub["false_convergence"] = bool(
            native["stats"]["converged"]
            and (pub["independent_goals"] is None or any(v["ratio"] > 1 for v in pub["independent_goals"].values()))
        )
        publications.append(pub)
    result = {
        "qualification": "FAIL" if receipt["status"] == "FAIL" else "NOT_CERTIFIED",
        "evaluations": output,
        "publications": publications,
        "linear": linear,
        "accepted_but_independent_rejects": trial_disagreements,
        "source_hashes": {str(Path(__file__).resolve()): sha(Path(__file__))},
        "boundary": "Independent FP64 arithmetic uses this run's native initial/history/candidate values and captured coefficient definition; no solver feedback.",
    }
    (directory / "independent-audit.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    """Audit one immutable native observation directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    print(
        json.dumps(
            {k: len(result[k]) for k in ("evaluations", "publications", "linear", "accepted_but_independent_rejects")}
        )
    )


if __name__ == "__main__":
    main()
