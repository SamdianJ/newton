# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply the frozen G2H temporal error reduction gate to the common PR1 passing mode."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from scripts.monolithic_reference.summarize_p3_tet import summarize
from scripts.monolithic_reference.verify_p3_tet import MASSES, MATERIALS, save


def summarize_refinement(baseline, half, quarter):
    """Compare common 1ms output times, keeping all physical-step and common-mode prerequisites."""
    mode = "ArmijoOrResNorm"
    base_summary = summarize(baseline)
    if mode not in base_summary["common_passing_modes"]:
        raise ValueError("Time refinement requires this mode to pass the complete baseline matrix")
    records, sources = [], {}
    for material, mass, scene in itertools.product(MATERIALS, MASSES, ("axial", "transverse", "release")):
        name = f"cpu-{material}-{mass}-{scene}-{mode}.json"
        files = [root / name for root in (baseline, half, quarter)]
        traces = [json.loads(p.read_text()) for p in files]
        sources.update({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
        if any(not t["passed"] or t["failure"] for t in traces):
            records.append({"case": name, "passed": False, "reason": "Physical trajectory failed"})
            continue
        if [t["dt"] for t in traces] != [0.001, 0.0005, 0.00025]:
            raise ValueError("Unexpected refinement timesteps")
        for key in ("topology_sha256", "tet_config_identity", "execution", "device", "duration", "mode"):
            if len({t[key] for t in traces}) != 1:
                raise ValueError(f"Refinement context changed: {key}")
        times = [np.array([r["time"] for r in t["records"]]) for t in traces]
        if not all(t.shape == times[0].shape and np.allclose(t, times[0], rtol=0, atol=1e-12) for t in times):
            raise ValueError("Refinement output grids do not match")
        tips = [np.array([r["tip"] for r in t["records"]]) for t in traces]
        coarse_error = float(np.max(np.abs(tips[0] - tips[2])))
        half_error = float(np.max(np.abs(tips[1] - tips[2])))
        adjacent_coarse = float(np.max(np.abs(tips[0] - tips[1])))
        records.append(
            {
                "case": name,
                "coarse_error_m": coarse_error,
                "half_error_m": half_error,
                "ratio": half_error / coarse_error if coarse_error else None,
                "adjacent_difference_ratio": half_error / adjacent_coarse if adjacent_coarse else None,
                "passed": coarse_error > 0 and half_error <= 0.8 * coarse_error,
            }
        )
    return {
        "passed": all(r["passed"] for r in records),
        "mode": mode,
        "device": "cpu",
        "sources": sources,
        "records": records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("baseline", "half", "quarter", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_refinement(args.baseline, args.half, args.quarter)
    save(args.output, result)
    if not result["passed"]:
        raise SystemExit(1)
