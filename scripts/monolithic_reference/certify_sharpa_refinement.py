# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply unchanged same-discrete and refinement gates to saved Sharpa traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .articulation_fixtures import specification
from .summarize_articulation import angles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    root, hashes = args.evidence_dir, {}

    def read(path, *, lines=False):
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        if not lines:
            return json.loads(path.read_text())
        rows = [json.loads(line) for line in path.open()]
        return {k: np.asarray([row[k] for row in rows]) for k in ("time", "q", "qd", "com", "rotation")}

    gates = specification()["same_discrete_oracle_max_errors"]
    limits = {
        "q": gates["q_rad"],
        "qd": gates["qd_rad_s"],
        "com": gates["link_COM_m"],
        "rotation": gates["link_rotation_rad"],
    }
    comparisons, saved, noise = [], {}, {}
    for refinement, directory in (
        (1, "sharpa-diagnostic-v2"),
        (2, "sharpa-refinement-r2-v1"),
        (4, "sharpa-refinement-r4-v1"),
    ):
        reference_dir = root / f"sharpa-oracle-r{refinement}-v1"
        receipt = read(reference_dir / "summary.json")
        if not receipt["certified"] or receipt["steps"] != 4000 * refinement:
            raise RuntimeError("Incomplete or uncertified oracle")
        reference = read(reference_dir / "trace.jsonl", lines=True)
        for device in ("cpu", "cuda_0"):
            modes = ("Armijo", "ResNorm", "ArmijoOrResNorm") if refinement == 1 else ("ArmijoOrResNorm",)
            for mode in modes:
                path = root / directory / f"{device}-{mode}-r{refinement}"
                summary = read(path / "summary.json")
                values = read(path / "trace.jsonl", lines=True)
                np.testing.assert_allclose(values["time"], reference["time"], rtol=0, atol=1e-12)
                maxima, envelope = {}, {}
                for key in limits:
                    difference = (
                        angles(values[key], reference[key]) if key == "rotation" else values[key] - reference[key]
                    )
                    per_step = (
                        np.linalg.norm(difference, axis=-1).max(axis=1)
                        if key == "com"
                        else np.abs(difference).max(axis=1)
                    )
                    maxima[key] = float(per_step.max())
                    # Match the existing small-system envelope: RMS of each step's
                    # worst coordinate/link, before sampling the common output grid.
                    envelope[key] = max(float(np.sqrt(np.mean(per_step**2))), 128 * np.finfo(float).eps)
                comparisons.append(
                    {
                        "device": device,
                        "mode": mode,
                        "refinement": refinement,
                        "maximum_errors": maxima,
                        "gates": limits,
                        "same_discrete_pass": bool(
                            summary["all_steps_success"] and all(maxima[k] <= limits[k] for k in limits)
                        ),
                    }
                )
                if mode == "ArmijoOrResNorm":
                    saved[device, refinement] = {k: values[k][refinement - 1 :: refinement] for k in limits}
                    noise[device, refinement] = envelope
    refinements = []
    for device in ("cpu", "cuda_0"):
        a, b, c = [saved[device, r] for r in (1, 2, 4)]
        groups = {}
        for key in limits:
            coarse = angles(a[key], c[key]) if key == "rotation" else a[key] - c[key]
            half = angles(b[key], c[key]) if key == "rotation" else b[key] - c[key]
            e1, e2 = float(np.sqrt(np.mean(coarse**2))), float(np.sqrt(np.mean(half**2)))
            bound = max(noise[device, r][key] for r in (1, 2, 4))
            resolvable = e1 > 20 * bound
            groups[key] = {
                "coarse_rms": e1,
                "half_rms": e2,
                "noise_bound": bound,
                "signal_to_noise": e1 / bound,
                "ratio": e2 / e1 if e1 else None,
                "signal_certified": resolvable,
                "ratio_pass": e2 <= 0.8 * e1,
                "passed": bool(resolvable and e2 <= 0.8 * e1),
            }
        refinements.append(
            {
                "device": device,
                "groups": groups,
                "dynamic_q_qd_certified": all(groups[k]["passed"] for k in ("q", "qd")),
            }
        )
    report = {
        "schema": "p3-pr2-sharpa-refinement-certification/v1",
        "same_discrete": comparisons,
        "refinement": refinements,
        "input_sha256": hashes,
        "same_discrete_pass": all(row["same_discrete_pass"] for row in comparisons),
        "refinement_pass": all(row["dynamic_q_qd_certified"] for row in refinements),
        "continuous_accuracy": "NOT_CERTIFIED; prior small-system failures retained",
    }
    (root / "sharpa-refinement-certification-v1.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "input_sha256"}, indent=2))


if __name__ == "__main__":
    main()
