# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare complete T4 trajectories without filling failed steps or shifting time."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .coupled_fixtures import DIRECTORY
from .verify_coupled import save

GATES = json.loads((DIRECTORY / "p3_pr2_v3.json").read_text())["same_discrete_oracle_max_errors"]
LIMITS = {
    "q_m": GATES["q_m"],
    "qd_m_s": GATES["qd_m_s"],
    "q_rad": GATES["q_rad"],
    "qd_rad_s": GATES["qd_rad_s"],
    "com": GATES["link_COM_m"],
    "rotation": GATES["link_rotation_rad"],
    "x": 5e-5,
}


def differences(a, b, case):
    """Return each time's worst coordinate/node/link; preserve translational units."""
    if a["time"].shape != b["time"].shape or not np.allclose(a["time"], b["time"], atol=1e-12, rtol=0):
        raise ValueError("Trajectory grids differ")
    translation = slice(0, 1) if case == "curved_release" else slice(None)
    result = {}
    for field, prefix in (("q", "q"), ("qd", "qd")):
        result[prefix + ("_m" if field == "q" else "_m_s")] = abs(
            a[field][:, translation] - b[field][:, translation]
        ).max(axis=1)
        if case == "curved_release":
            result[prefix + ("_rad" if field == "q" else "_rad_s")] = abs(a[field][:, 1] - b[field][:, 1])
    for field in ("com", "x"):
        result[field] = np.linalg.norm(a[field] - b[field], axis=-1).max(axis=1)
    relative = a["rotation"] @ b["rotation"].swapaxes(-1, -2)
    skew = np.stack(
        (
            relative[..., 2, 1] - relative[..., 1, 2],
            relative[..., 0, 2] - relative[..., 2, 0],
            relative[..., 1, 0] - relative[..., 0, 1],
        ),
        axis=-1,
    )
    angle = np.arctan2(np.linalg.norm(skew, axis=-1) / 2, (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2)
    result["rotation"] = angle.max(axis=1)
    return result


def errors(a, b, case):
    return {
        k: {"maximum": float(v.max()), "rms": float(np.sqrt(np.mean(v**2)))} for k, v in differences(a, b, case).items()
    }


def summarize(root):
    """Keep numerical, reference, refinement and physical results separate."""
    rows, hashes, arrays = {}, {}, {}
    for path in sorted(root.glob("*/summary.json")):
        row = json.loads(path.read_text())
        if row["case"].startswith("cube"):
            continue
        config = json.loads((path.parent / "configuration.json").read_text())
        key = (
            row["case"],
            row["law"],
            row["dt"],
            row["goal_scale"]
            if row["reference"]
            else (row["device"], row["mode"], row["graph"], config["globalization"]),
        )
        rows[key] = row
        for file in (path, path.parent / "trajectory.npz"):
            hashes[str(file.relative_to(root))] = hashlib.sha256(file.read_bytes()).hexdigest()
        if row["complete"]:
            with np.load(path.parent / "trajectory.npz") as values:
                arrays[key] = {k: values[k] for k in values.files}
            expected = np.arange(1, row["expected_steps"] + 1) * row["dt"]
            if arrays[key]["time"].shape != expected.shape or not np.allclose(
                arrays[key]["time"], expected, atol=1e-12, rtol=0
            ):
                raise ValueError(f"Missing endpoint: {key}")
    sensitivity, comparisons, refinement = [], [], []
    envelopes = {}
    for key, row in rows.items():
        case, law, dt, mode = key
        if row["reference"]:
            if mode == 0.1:
                strict = (case, law, dt, 0.01)
                sensitivity.append(
                    {
                        "case": case,
                        "law": law,
                        "dt": dt,
                        "complete": key in arrays and strict in arrays,
                        "errors": errors(arrays[key], arrays[strict], case)
                        if key in arrays and strict in arrays
                        else None,
                    }
                )
            continue
        reference = (case, law, dt, 0.1)
        record = {
            "case": case,
            "law": law,
            "dt": dt,
            "mode": mode,
            "physical": row["physical"],
            "signals": row.get("signals"),
            "complete": row["complete"],
            "same_discrete": "NOT_CERTIFIED_INCOMPLETE",
        }
        if key in arrays and reference in arrays:
            measured = errors(arrays[key], arrays[reference], case)
            record.update(
                errors=measured,
                gates={k: LIMITS[k] for k in measured},
                same_discrete="PASS" if all(v["maximum"] <= LIMITS[k] for k, v in measured.items()) else "FAIL",
            )
            strict = (case, law, dt, 0.01)
            extra = errors(arrays[reference], arrays[strict], case) if strict in arrays else None
            envelopes[key] = (
                {k: max(v["rms"] + extra[k]["rms"], 128 * np.finfo(float).eps) for k, v in measured.items()}
                if extra
                else None
            )
        comparisons.append(record)
    for case in ("normal_loading", "tangent_reversal", "curved_release"):
        for law in ("C-EE", "C-IE", "C-II"):
            reference_keys = [(case, law, dt, 0.1) for dt in (0.001, 0.0005, 0.00025)]
            for mode in (("cpu", "diagnostic", False, "ResNorm"), ("cuda:0", "production", True, "ResNorm")):
                keys = [(case, law, dt, mode) for dt in (0.001, 0.0005, 0.00025)]
                record = {"case": case, "law": law, "mode": mode, "status": "NOT_CERTIFIED_INCOMPLETE"}
                if all(k in arrays and envelopes.get(k) is not None for k in keys):
                    common = [
                        {k: v[factor - 1 :: factor] for k, v in arrays[key].items()}
                        for key, factor in zip(keys, (1, 2, 4), strict=True)
                    ]
                    coarse, half = errors(common[0], common[2], case), errors(common[1], common[2], case)
                    ref_common = [
                        {k: v[factor - 1 :: factor] for k, v in arrays[key].items()}
                        for key, factor in zip(reference_keys, (1, 2, 4), strict=True)
                    ]
                    ref_coarse = errors(ref_common[0], ref_common[2], case)
                    groups = {}
                    for field in coarse:
                        e1, e2 = coarse[field]["rms"], half[field]["rms"]
                        bound = max(envelopes[k][field] for k in keys)
                        sensitivity_bound = max(
                            next(
                                s["errors"][field]["rms"]
                                for s in sensitivity
                                if (s["case"], s["law"], s["dt"]) == k[:3]
                            )
                            for k in keys
                        )
                        ref_e1 = ref_coarse[field]["rms"]
                        ref_ok = sensitivity_bound <= max(ref_e1 / 200, 128 * np.finfo(float).eps)
                        groups[field] = {
                            "E1": e1,
                            "E2": e2,
                            "N": bound,
                            "ratio": e2 / e1 if e1 else None,
                            "reference_sensitivity": sensitivity_bound,
                            "reference_E1": ref_e1,
                            "reference_certified": ref_ok,
                            "signal_certified": e1 > 20 * bound,
                            "ratio_pass": e2 <= 0.8 * e1,
                            "passed": bool(ref_ok and e1 > 20 * bound and e2 <= 0.8 * e1),
                        }
                    dynamic = [k for k in groups if k.startswith("q")]
                    discrete = all(
                        next(
                            c["same_discrete"]
                            for c in comparisons
                            if (c["case"], c["law"], c["dt"], tuple(c["mode"])) == k
                        )
                        == "PASS"
                        for k in keys
                    )
                    record.update(
                        groups=groups,
                        same_discrete_pass=discrete,
                        status="PASS" if discrete and all(groups[k]["passed"] for k in dynamic) else "FAIL",
                        dynamic_groups=dynamic,
                    )
                refinement.append(record)
    result = {
        "reference_sensitivity": sensitivity,
        "same_discrete": comparisons,
        "refinement": refinement,
        "gates": LIMITS,
        "continuous_accuracy": "NOT_CERTIFIED; no new continuous target has been approved for these contact fixtures",
        "input_sha256": hashes,
    }
    save(root / "trajectory-comparisons.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.evidence_dir)
    print(
        json.dumps(
            {
                "comparisons": len(result["same_discrete"]),
                "refinement": [{k: v for k, v in r.items() if k != "groups"} for r in result["refinement"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
