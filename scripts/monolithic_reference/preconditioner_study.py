# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen-matrix PR8E research; no production selection or automatic admission."""

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from scripts.monolithic_reference import benchmark_pcg as benchmark
from scripts.monolithic_reference.linear_evidence import array_fingerprint, evidence_index

CANDIDATES = ("actor_block", "local_block_4", "local_block_8", "mas_16", "mas_32")


@dataclass(frozen=True)
class Hierarchy:
    candidate: str
    block_size: int
    domains: tuple
    level_counts: tuple
    block_ids: np.ndarray
    local_ids: np.ndarray
    fingerprint: str


def build_hierarchy(arrays, candidate):
    """Build deterministic fixed-reference domains, splitting coarse aggregates by connectivity."""
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate: {candidate}")
    positions = np.asarray(arrays["rest_x"], dtype=np.float64)
    ids = np.asarray(arrays["dynamic_particle_ids"])
    tets = np.asarray(arrays["tet_indices"])
    if (
        positions.ndim != 2
        or positions.shape[1] != 3
        or not np.isfinite(positions).all()
        or ids.ndim != 1
        or ids.dtype.kind not in "iu"
        or len(np.unique(ids)) != len(ids)
        or np.any(ids < 0)
        or np.any(ids >= len(positions))
        or tets.ndim != 2
        or tets.shape[1] != 4
        or tets.dtype.kind not in "iu"
        or np.any(tets < 0)
        or np.any(tets >= len(positions))
    ):
        raise ValueError("Invalid reference geometry, dynamic IDs or tetrahedra")
    size = 1 if candidate == "actor_block" else int(candidate.rsplit("_", 1)[1])
    count = len(ids)
    original_to_dynamic = {int(original): i for i, original in enumerate(ids)}
    adjacency = [set() for _ in ids]
    for tet in tets:
        dynamic = [original_to_dynamic[int(i)] for i in tet if int(i) in original_to_dynamic]
        for i in dynamic:
            adjacency[i].update(j for j in dynamic if j != i)
    domains, levels = [], []
    if candidate.startswith("mas") and count:
        xyz = positions[ids]
        extent = np.ptp(xyz, axis=0)
        quantized = np.floor((xyz - xyz.min(axis=0)) / np.where(extent > 0, extent, 1) * (2**20 - 1)).astype(np.uint64)
        code = np.zeros(count, dtype=np.uint64)
        for bit in range(20):
            for axis in range(3):
                code |= ((quantized[:, axis] >> bit) & 1) << (3 * bit + axis)
        order = np.lexsort((ids, code))
        groups = [(int(i),) for i in order]
        for level in range(3):
            chunks = [groups[start : start + size] for start in range(0, len(groups), size)]
            domains.extend(tuple(chunk) for chunk in chunks)
            levels.append(len(chunks))
            if len(groups) <= size or level == 2:
                break
            membership = {i: g for g, group in enumerate(groups) for i in group}
            coarse_adj = [set() for _ in groups]
            for i, neighbors in enumerate(adjacency):
                coarse_adj[membership[i]].update(membership[j] for j in neighbors if membership[j] != membership[i])
            next_groups = []
            for start in range(0, len(groups), size):
                remaining = set(range(start, min(start + size, len(groups))))
                while remaining:
                    seed = min(remaining)
                    remaining.remove(seed)
                    queue, component = [seed], []
                    while queue:
                        g = queue.pop()
                        component.extend(groups[g])
                        neighbors = sorted(coarse_adj[g] & remaining)
                        remaining.difference_update(neighbors)
                        queue.extend(neighbors)
                    next_groups.append(tuple(sorted(component)))
            if len(next_groups) >= len(groups):
                break
            groups = next_groups
    elif count:
        remaining = set(range(count))
        while remaining:
            queue, group = deque([min(remaining)]), []
            queued = set(queue)
            while queue and len(group) < size:
                i = queue.popleft()
                if i not in remaining:
                    continue
                remaining.remove(i)
                group.append((i,))
                neighbors = sorted((adjacency[i] & remaining) - queued)
                queue.extend(neighbors)
                queued.update(neighbors)
            domains.append(tuple(group))
        levels.append(len(domains))
    block_ids = np.full((len(levels), count), -1, dtype=np.int32)
    local_ids = np.full_like(block_ids, -1)
    offset = 0
    for level, number in enumerate(levels):
        for block in range(offset, offset + number):
            for local, group in enumerate(domains[block]):
                block_ids[level, list(group)] = block
                local_ids[level, list(group)] = local
        offset += number
    fingerprint = array_fingerprint(
        {
            "block_ids": block_ids,
            "local_ids": local_ids,
            "size": np.array([size], dtype=np.int32),
            "ids": ids,
            "positions": positions,
            "tets": tets,
        }
    )
    block_ids.setflags(write=False)
    local_ids.setflags(write=False)
    return Hierarchy(candidate, size, tuple(domains), tuple(levels), block_ids, local_ids, fingerprint)


class BlockOracle:
    """Independent FP64 projected blocks and Cholesky solves on the full scaled operator."""

    def __init__(self, reference, hierarchy):
        import scipy.linalg as la
        import scipy.sparse as sp

        self.nq = reference.nq
        self.hierarchy = hierarchy
        self.q_factor = la.cho_factor(reference.matrix[: self.nq, : self.nq].toarray()) if self.nq else None
        soft = reference.matrix[self.nq :, self.nq :]
        if soft.shape[0] != 3 * hierarchy.block_ids.shape[1]:
            raise ValueError("Hierarchy does not match the soft matrix")
        self.restrictions, self.blocks, self.factors = [], [], []
        for block, groups in enumerate(hierarchy.domains):
            rows, columns, values = [], [], []
            coarse = block >= hierarchy.level_counts[0]
            for local, group in enumerate(groups):
                for particle in group:
                    for axis in range(3):
                        column = 3 * particle + axis
                        rows.append(3 * local + axis)
                        columns.append(column)
                        values.append(1.0 / reference.scale[self.nq + column] if coarse else 1.0)
            restriction = sp.csr_matrix((values, (rows, columns)), shape=(3 * len(groups), soft.shape[0]))
            matrix = (restriction @ soft @ restriction.T).toarray()
            if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-10):
                raise ValueError("Nonfinite or asymmetric projected block")
            self.restrictions.append(restriction)
            self.blocks.append(matrix)
            self.factors.append(la.cho_factor(matrix))

    def apply(self, residual):
        import scipy.linalg as la

        output = np.zeros_like(residual, dtype=np.float64)
        if self.nq:
            output[: self.nq] = la.cho_solve(self.q_factor, residual[: self.nq])
        for restriction, factor in zip(self.restrictions, self.factors, strict=True):
            output[self.nq :] += restriction.T @ la.cho_solve(factor, restriction @ residual[self.nq :])
        return output


def solve_reference_pcg(reference, apply):
    """Observe FP64 algebraic PCG with fresh residual acceptance; not native-policy timing."""
    y = np.zeros_like(reference.rhs)
    residual = reference.rhs.copy()
    z = apply(residual)
    direction = z.copy()
    rz = float(residual @ z)
    points = []
    for iteration in range(reference.config.maximum_iterations + 1):
        accuracy = reference.measure(y)
        points.append({"iterations": iteration, "accuracy": accuracy})
        if accuracy["independent_residual_pass"]:
            status = "SUCCESS"
            break
        if iteration == reference.config.maximum_iterations:
            status = "MAX_ITERATIONS"
            break
        ad = reference.matrix @ direction
        curvature = float(direction @ ad)
        if not np.isfinite([rz, curvature]).all() or rz <= 0 or curvature <= 0:
            status = "INVALID_SPD_PRODUCT"
            break
        alpha = rz / curvature
        y += alpha * direction
        residual -= alpha * ad
        z = apply(residual)
        new_rz = float(residual @ z)
        direction = z + (new_rz / rz) * direction
        rz = new_rz
    return {"status": status, "iterations": iteration, "accuracy": accuracy, "points": points, "y": y}


def synthetic_case(nodes=20):
    """Supply nonuniform scaling, nonzero q-x coupling and geometry for projection regressions."""
    import scipy.sparse as sp

    rng = np.random.default_rng(823)
    nq, n = 2, 2 + 3 * nodes
    factor = rng.normal(size=(n, n)) / np.sqrt(n)
    matrix = sp.csr_matrix((factor @ factor.T + 0.5 * np.eye(n)).astype(np.float32))
    scale = np.geomspace(0.4, 2.0, n).astype(np.float32)
    arrays = {
        "K_offsets": matrix.indptr,
        "K_columns": matrix.indices,
        "K_values": matrix.data,
        "S": scale,
        "rhs_hat": rng.normal(size=n).astype(np.float32),
        "rest_x": rng.normal(size=(nodes, 3)).astype(np.float32),
        "dynamic_particle_ids": np.arange(nodes, dtype=np.int32),
        "tet_indices": np.array([range(i, i + 4) for i in range(max(0, nodes - 3))], dtype=np.int32).reshape(-1, 4),
    }
    return benchmark.Case("projected_synthetic", arrays, {"q_dof_count": nq, "lambda": 0.03})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--linear-absolute-tolerance", type=float, default=1e-6)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    config = benchmark.pcg_config(linear_absolute_tolerance=args.linear_absolute_tolerance)
    results, vectors = [], {}
    cases = [benchmark.load_snapshot(path) for path in args.snapshot] if args.snapshot else [synthetic_case(40)]
    for case_index, case in enumerate(cases):
        reference = benchmark.Reference(case, config)
        for candidate in CANDIDATES:
            row = {
                "case": case.name,
                "candidate": candidate,
                "source": case.metadata,
                "input_sha256": array_fingerprint(case.arrays),
                "admission": "REVIEW_REQUIRED",
            }
            try:
                start = time.perf_counter()
                hierarchy = build_hierarchy(case.arrays, candidate)
                row["hierarchy_seconds"] = time.perf_counter() - start
                start = time.perf_counter()
                oracle = BlockOracle(reference, hierarchy)
                row["numeric_seconds"] = time.perf_counter() - start
                result = solve_reference_pcg(reference, oracle.apply)
                key = f"{case_index}-{candidate}"
                vectors[key] = result.pop("y")
                row.update(
                    result,
                    solution_key=key,
                    hierarchy_sha256=hierarchy.fingerprint,
                    level_counts=hierarchy.level_counts,
                    reference=reference.describe(),
                )
            except (ValueError, np.linalg.LinAlgError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
            results.append(row)
    np.savez_compressed(args.output / "solutions.npz", **vectors)
    (args.output / "results.json").write_text(
        json.dumps(benchmark._json_safe(results), indent=2, allow_nan=False) + "\n"
    )
    (args.output / "README.md").write_text(
        "# PR8E algebra study\n\nFP64 research PCG, not native-policy or GPU performance. "
        "See results.json for convergence and independently certified true residuals. "
        "No performance admission; scalar error limits remain unchanged.\n"
    )
    evidence_index(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
