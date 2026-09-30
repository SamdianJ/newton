# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read retained P5 FP64 states as P3 frame blocks without rewriting their bytes."""

import gzip
import json
from collections import Counter

import numpy as np

from scripts.monolithic_reference.retained_assets import sha


def frames(directory, controls):
    """Join the separate initial state, native blocks, publications and frozen controls."""
    receipt = json.loads((directory / "receipt.json").read_text())
    case = receipt["case"]
    steps = case["steps"]
    if (
        receipt["status"] != "COMPLETE_NOT_AUDITED"
        or receipt["published_steps"] != steps
        or case["precision"] != "fp64"
        or case["dt"] != 0.001
        or receipt["counts"]["HARD_FAILURE"] != 0
        or sha(controls) != receipt["controls_sha256"]
    ):
        raise ValueError("Incomplete or mismatched native FP64 reference")
    samples = np.load(controls, allow_pickle=False)
    with np.load(directory / "initial.npz", allow_pickle=False) as data:
        initial = {name: data[name] for name in data.files}
    if (
        samples.dtype != np.float32
        or samples.shape != (steps + 1, 2, initial["joint_qd"].size)
        or not np.isfinite(samples).all()
        or np.any(initial["history.force_generation"] != 0)
        or np.any(initial["history.force_valid"] != 0)
    ):
        raise ValueError("Invalid native initial state or frozen control layout")

    def decorate(block, start, labels):
        n = len(labels)
        block.update(
            frame_steps=np.arange(start, start + n),
            frame_times=np.arange(start, start + n) * case["dt"],
            frame_publication=np.asarray(labels, dtype="U16"),
            frame_target_q=samples[start : start + n, 0],
            frame_target_qd=samples[start : start + n, 1],
        )
        return block

    pending = decorate({name: value[None] for name, value in initial.items()}, 0, ["INITIAL"])
    offset = 1
    counts = Counter({"CONVERGED": 0, "STOPPED": 0, "HARD_FAILURE": 0})
    seen = set()
    with gzip.open(directory / "steps.jsonl.gz", "rt") as log:
        for chunk in receipt["chunks"]:
            path = (directory / chunk["file"]).resolve()
            if not path.is_relative_to(directory.resolve()) or path in seen or sha(path) != chunk["sha256"]:
                raise ValueError("Native frame path/hash mismatch")
            seen.add(path)
            with np.load(path, allow_pickle=False) as data:
                block = {name: data[name] for name in data.files}
            n = chunk["frames"]
            if n <= 0 or offset + n > steps + 1 or block.keys() != initial.keys():
                raise ValueError("Invalid native frame count/schema")
            for name, value in block.items():
                if (
                    value.shape != (n, *initial[name].shape)
                    or value.dtype != initial[name].dtype
                    or not np.isfinite(value).all()
                    or not np.isfinite(initial[name]).all()
                ):
                    raise ValueError(f"Invalid native state field: {name}")
            if not np.array_equal(
                block["history.force_generation"].reshape(n), np.arange(offset, offset + n)
            ) or not np.all(block["history.force_valid"] == 1):
                raise ValueError("Missing/shifted native publication generation")
            labels = []
            for step in range(offset, offset + n):
                line = next(log, None)
                if line is None:
                    raise ValueError("Native publication log is incomplete")
                row = json.loads(line)
                if (
                    row["step"] != step
                    or row["publication"] not in ("CONVERGED", "STOPPED")
                    or row["stats"]["rolled_back"]
                    or not np.isfinite(row["stats"]["min_det_f"])
                    or row["stats"]["min_det_f"] < 0.2
                ):
                    raise ValueError("Invalid native publication log")
                counts[row["publication"]] += 1
                labels.append("STOPPED_ACCEPTED" if row["publication"] == "STOPPED" else "CONVERGED")
            block = decorate(block, offset, labels)
            pending = {name: np.concatenate((pending[name], value)) for name, value in block.items()}
            while len(pending["frame_steps"]) >= 100:
                yield {name: value[:100] for name, value in pending.items()}
                pending = {name: value[100:] for name, value in pending.items()}
            offset += n
        if offset != steps + 1 or next(log, None) is not None or dict(counts) != receipt["counts"]:
            raise ValueError("Native reference totals disagree")
    if len(pending["frame_steps"]):
        yield pending
