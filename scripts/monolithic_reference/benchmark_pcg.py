# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: RUF001
"""Compare native PCG convergence, FP64 reference error and complete linear-solve costs.

Run as a standalone offline benchmark; never attach observers to timed solves.
Snapshot replay restores only the linear workspace, not a physical solver state.
"""

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
import warp.sparse as sparse

from newton._src.solvers.monolithic import linear, pcg_production
from newton._src.solvers.monolithic.solver_monolithic import _SolverMonolithicInternalConfig
from scripts.monolithic_reference.p2_measurement import (
    array_fingerprint,
    evidence_index,
    pcg_call_record,
    snapshot_sources,
)


def pcg_config(maximum_iterations=200, *, linear_absolute_tolerance=1e-6):
    """Use the solver's current tolerances and policies, without tuning per mode."""
    defaults = _SolverMonolithicInternalConfig()
    return linear.MonolithicPcgConfig(
        maximum_iterations,
        10,
        1e-4,
        defaults.residual_floor_global,
        defaults.residual_floor_q,
        defaults.residual_floor_x,
        0.0,
        1e-12,
        0.0,
        10,
        1e-3,
        linear_absolute_tolerance=linear_absolute_tolerance,
    )


@dataclass
class Case:
    name: str
    arrays: dict
    metadata: dict

    @property
    def nq(self):
        return self.metadata["q_dof_count"]

    @property
    def n(self):
        return len(self.arrays["rhs_hat"])

    def matrix(self):
        import scipy.sparse as sp

        a = self.arrays
        return sp.csr_matrix((a["K_values"], a["K_columns"], a["K_offsets"]), shape=(self.n, self.n))


def synthetic_cases():
    """Deterministic SPD systems with actor blocks, scaling, conditioning and empty blocks."""
    import scipy.sparse as sp

    rng = np.random.default_rng(20260913)
    cases = []
    for name, nq, nx, condition in (
        ("well_conditioned", 2, 10, 4),
        ("coupled_scaled", 2, 21, 1e3),
        ("ill_conditioned", 2, 21, 1e6),
        ("q_only", 22, 0, 1e2),
        ("x_only", 0, 32, 1e3),
        ("zero_rhs", 2, 10, 4),
    ):
        n = nq + 3 * nx
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        dense = ((q * np.geomspace(1, condition, n)) @ q.T).astype(np.float32)
        matrix = sp.csr_matrix(dense)
        internal = sp.bsr_matrix(dense[nq:, nq:], blocksize=(3, 3)) if nx else None
        arrays = {
            "K_offsets": matrix.indptr.astype(np.int32),
            "K_columns": matrix.indices.astype(np.int32),
            "K_values": matrix.data,
            "aq_actor_dense": dense[:nq, :nq].copy(),
            "ax_internal_offsets": internal.indptr.astype(np.int32) if nx else np.zeros(1, dtype=np.int32),
            "ax_internal_columns": internal.indices.astype(np.int32) if nx else np.zeros(0, dtype=np.int32),
            "ax_internal_values": internal.data if nx else np.zeros((0, 3, 3), dtype=np.float32),
            "S": (1 / np.sqrt(np.diag(dense))).astype(np.float32),
            "D": np.diag(dense).copy(),
            "dynamic_diagonal": np.diag(dense).copy(),
            "rhs_hat": np.zeros(n, dtype=np.float32) if name == "zero_rhs" else rng.normal(size=n).astype(np.float32),
            "factor_gq": np.zeros((0, nq), dtype=np.float32),
            "factor_gx_columns": np.zeros((0, 9), dtype=np.int32),
            "factor_gx_values": np.zeros((0, 9), dtype=np.float32),
            "factor_weights": np.zeros(0, dtype=np.float32),
            "factor_kind": np.zeros(0, dtype=np.int32),
            "factor_candidate_tid": np.zeros(0, dtype=np.int32),
        }
        cases.append(
            Case(
                name,
                arrays,
                {"q_dof_count": nq, "lambda": 0.0, "origin": "synthetic SPD", "target_raw_condition": condition},
            )
        )
    return cases


def load_snapshot(path):
    """Read and authenticate an existing export_candidate NPZ/JSON pair."""
    path = Path(path)
    try:
        metadata = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key].copy() for key in archive.files}
        if array_fingerprint(arrays) != metadata["input_sha256"]:
            raise ValueError("Snapshot fingerprint mismatch")
        case = Case(f"{path.parent.parent.name}-{path.stem}", arrays, {**metadata, "source": str(path.resolve())})
        if not 0 <= case.nq <= case.n or (case.n - case.nq) % 3:
            raise ValueError("Invalid q/x dimensions")
        matrix = case.matrix()
        if not matrix.has_canonical_format or not np.isfinite(matrix.data).all():
            raise ValueError("Expected finite canonical CSR storage")
        if np.any(arrays["S"] <= 0) or not np.isfinite(arrays["S"]).all() or not np.isfinite(arrays["rhs_hat"]).all():
            raise ValueError("Invalid scale or RHS")
        return case
    except (OSError, KeyError, TypeError) as error:
        raise ValueError(f"Cannot load linear snapshot {path}: {error}") from error


class Reference:
    """FP64 sparse direct solution of the actual quantized float32 inputs."""

    def __init__(self, case, config):
        import scipy.sparse as sp
        import scipy.sparse.linalg as sla

        self.config = config
        self.nq = case.nq
        self.scale = case.arrays["S"].astype(np.float64)
        self.rhs = case.arrays["rhs_hat"].astype(np.float64)
        scaling = sp.diags(self.scale)
        self.matrix = (
            scaling @ case.matrix().astype(np.float64) @ scaling
            + float(np.float32(case.metadata["lambda"])) * sp.eye(case.n)
        ).tocsr()
        self.norm1 = float(sla.norm(self.matrix, 1))
        factor = sla.splu(self.matrix.tocsc())
        self.y = factor.solve(self.rhs)
        residual = self.rhs - self.matrix @ self.y
        self.relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(self.rhs), np.finfo(float).tiny))
        self.backward_error = self._backward_error(self.y, residual)
        if not np.isfinite(self.y).all() or self.relative_residual > 1e-9 or self.backward_error > 1e-12:
            raise ValueError("Direct reference failed its residual certificate")
        inverse = sla.LinearOperator(
            self.matrix.shape,
            matvec=factor.solve,
            rmatvec=lambda v: factor.solve(v, trans="T"),
            matmat=factor.solve,
            dtype=np.float64,
        )
        self.condition_estimate_1 = float(self.norm1 * sla.onenormest(inverse))
        self.floors = np.array([config.residual_floor_global, config.residual_floor_q, config.residual_floor_x])
        self.denominators = np.maximum(self.block_norms(self.rhs), self.floors)
        self.residual_targets = np.maximum(
            config.linear_tolerance * self.denominators, config.linear_absolute_tolerance
        )
        self.energy = float(self.y @ (self.matrix @ self.y))

    def block_norms(self, vector):
        return np.array([np.linalg.norm(vector), np.linalg.norm(vector[: self.nq]), np.linalg.norm(vector[self.nq :])])

    def _backward_error(self, y, residual):
        return float(
            np.linalg.norm(residual, 1)
            / max(self.norm1 * np.linalg.norm(y, 1) + np.linalg.norm(self.rhs, 1), np.finfo(float).tiny)
        )

    def measure(self, y):
        y = np.asarray(y, dtype=np.float64)
        if not np.isfinite(y).all():
            return {"finite": False, "independent_residual_pass": False}
        error = y - self.y
        residual = self.rhs - self.matrix @ y
        ratios = self.block_norms(residual) / self.denominators
        acceptance_ratios = self.block_norms(residual) / self.residual_targets
        relative = self.block_norms(error) / np.maximum(self.block_norms(self.y), np.finfo(float).tiny)
        energy_error = float(error @ (self.matrix @ error))
        return {
            "finite": True,
            "true_ratios": ratios.tolist(),
            "max_true_ratio": float(max(ratios)),
            "acceptance_ratios": acceptance_ratios.tolist(),
            "max_acceptance_ratio": float(max(acceptance_ratios)),
            "independent_residual_pass": bool(max(acceptance_ratios) <= 1.0),
            "y_absolute_error": float(np.linalg.norm(error)),
            "delta_absolute_error": float(np.linalg.norm(self.scale * error)),
            "y_relative_error": float(relative[0]),
            "q_relative_error": float(relative[1]),
            "x_relative_error": float(relative[2]),
            "delta_relative_error": float(
                np.linalg.norm(self.scale * error) / max(np.linalg.norm(self.scale * self.y), np.finfo(float).tiny)
            ),
            "backward_error_1": self._backward_error(y, residual),
            "energy_relative_error": math.sqrt(max(0, energy_error) / max(self.energy, np.finfo(float).tiny)),
        }

    def describe(self):
        import scipy.sparse.linalg as sla

        return {
            "method": "SciPy SuperLU float64; original float32 K/S/RHS, no symmetrization",
            "relative_residual": self.relative_residual,
            "backward_error_1": self.backward_error,
            "condition_estimate_1": self.condition_estimate_1,
            "relative_asymmetry_1": float(sla.norm(self.matrix - self.matrix.T, 1) / self.norm1),
        }


def _bsr(arrays, prefix, rows, device, *, blocks=False):
    offsets = arrays[prefix + "_offsets"]
    row_indices = np.repeat(np.arange(rows, dtype=np.int32), np.diff(offsets))
    return sparse.bsr_from_triplets(
        rows,
        rows,
        wp.array(row_indices, dtype=int, device=device),
        wp.array(arrays[prefix + "_columns"], dtype=int, device=device),
        wp.array(arrays[prefix + "_values"], dtype=wp.mat33 if blocks else float, device=device),
    )


class Runner:
    """Offline binding of one shared input to an actual monolithic solve implementation."""

    def __init__(self, case, mode, device, config):
        self.mode, self.config = mode, config
        a = case.arrays
        nx = (case.n - case.nq) // 3
        capacity = linear.MonolithicLinearCapacities(
            0, 0, len(a["factor_weights"]), config.maximum_iterations, config.maximum_iterations + 2
        )
        self.workspace = w = linear.MonolithicLinearWorkspace(
            linear.MonolithicLinearLayout(case.nq, nx),
            capacity,
            wp.array(np.arange(nx, dtype=np.int32), device=device),
            device=device,
            pcg_mode="diagnostic" if mode == "diagnostic" else "production",
            use_cuda_graph=mode == "production_graph",
        )
        w.k_global_scalar_bsr = _bsr(a, "K", case.n, device)
        if nx:
            w.ax_internal_bsr3 = _bsr(a, "ax_internal", nx, device, blocks=True)
        w.aq_actor_dense.assign(a["aq_actor_dense"])
        for name, key in (("scale", "S"), ("diagonal", "D"), ("dynamic_diagonal", "dynamic_diagonal")):
            getattr(w, name).assign(a[key])
        w._matrix_diagonal.assign(case.matrix().diagonal())
        for name in ("gq", "gx_columns", "gx_values", "weights", "kind", "candidate_tid"):
            getattr(w._factors, name).assign(a["factor_" + name])
        w._factors.count.fill_(len(a["factor_weights"]))
        # This is a sealed offline algebra snapshot, not a resumed physical transaction.
        w._generation = linear.MonolithicLinearGeneration(1, 0, 1, 1)
        w._sealed = w._valid = w._scaling_valid = True
        self._require(w.set_regularization(case.metadata["lambda"], generation=w._generation))
        self.factor()
        self.rhs = wp.array(a["rhs_hat"], dtype=float, device=device)
        self.y = wp.zeros(case.n, dtype=float, device=device)
        self.delta = wp.zeros_like(self.y)

    @staticmethod
    def _require(status):
        if status != linear.MonolithicLinearStatus.SUCCESS:
            raise ValueError(f"Offline workspace setup failed: {status.name}")

    def refresh_generation(self):
        w = self.workspace
        w._generation = replace(w._generation, assembly_sequence=w._generation.assembly_sequence + 1)
        w._factor_valid = False
        w._warm_start_key = None

    def factor(self):
        self._require(
            self.workspace.factor_actor_preconditioner(generation=self.workspace._generation, pivot_tolerance=0)
        )

    def solve(self):
        w = self.workspace
        result = w.solve_pcg(
            self.rhs,
            self.y,
            generation=w._generation,
            warm_start=linear.MonolithicPcgWarmStart.ZERO,
            config=self.config,
            delta=self.delta,
        )
        if self.mode == "diagnostic" and result.status == linear.MonolithicLinearStatus.SUCCESS:
            self._require(w.recover_delta(self.y, self.delta, generation=w._generation))
        return result


def observe_convergence(runner, reference):
    """Read native iterates only in a separate, explicitly untimed solve."""
    if hasattr(runner, "observe"):
        return runner.observe(reference)
    runner.y.zero_()
    iterates = [runner.y.numpy().copy()]
    points = [{"iterations": 0, "slot": 0, "accuracy": reference.measure(iterates[0])}]
    launch = wp.launch
    last_iterations = 0

    def observed(kernel, *args, **kwargs):
        nonlocal last_iterations
        result = launch(kernel, *args, **kwargs)
        recursive = None
        if kernel is linear._update_monolithic_pcg_solution_residual:
            last_iterations += 1
            slot = last_iterations
            recursive = runner.workspace._r
        elif kernel is pcg_production._finish_update:
            plan = runner.workspace._masked_pcg_plan
            state = plan.state.numpy()[0]
            iterations = int(state["effective_iterations"])
            if iterations > last_iterations:
                last_iterations = iterations
                slot = kwargs["inputs"][-1]
                recursive = plan.r
        if recursive is not None:
            y = runner.y.numpy().astype(np.float64)
            iterates.append(y.copy())
            true_r = reference.rhs - reference.matrix @ y
            r = recursive.numpy().astype(np.float64)
            points.append(
                {
                    "iterations": last_iterations,
                    "slot": int(slot),
                    "accuracy": reference.measure(y),
                    "recursive_ratios": (reference.block_norms(r) / reference.denominators).tolist(),
                    "gap_ratios": (
                        reference.block_norms(r - true_r) / np.maximum(reference.block_norms(true_r), reference.floors)
                    ).tolist(),
                }
            )
        return result

    with patch.object(wp, "launch", side_effect=observed):
        result = runner.solve()
    return {
        "scope": "Untimed native non-graph solve; observers read but never modify solver buffers",
        "points": points,
        "iterate_y": np.stack(iterates),
        "result": pcg_call_record(result, 0.0, use_cuda_graph=False),
    }


def _timed_solve(runner, scope):
    device = runner.workspace.device
    wp.synchronize_device(device)
    start = time.perf_counter()
    factor_ms = 0.0
    if scope == "new_generation":
        runner.refresh_generation()
        runner.factor()
        factor_ms = (time.perf_counter() - start) * 1000
    solve_start = time.perf_counter()
    result = runner.solve()
    wp.synchronize_device(device)
    end = time.perf_counter()
    return {
        **(
            result
            if isinstance(result, dict)
            else pcg_call_record(result, end - solve_start, use_cuda_graph=runner.mode == "production_graph")
        ),
        "mode": runner.mode,
        "scope": scope,
        "factor_ms": factor_ms,
        "solve_ms": (end - solve_start) * 1000,
        "total_ms": (end - start) * 1000,
    }


def _gpu_sample(device):
    if not device.is_cuda:
        return {"device": "cpu"}
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pstate,clocks.sm,temperature.gpu,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {"unix_s": time.time(), "gpu_csv": query.stdout.strip(), "returncode": query.returncode}


def compare_case(case, device, *, repeats=10, warmups=3, config=None, include_external=False):
    """Interleave modes on identical fixed input, with separate reuse and refresh scopes."""
    device = wp.get_device(device)
    config = config or pcg_config()
    if repeats < 1 or warmups < 1:
        raise ValueError("repeats and warmups must be positive")
    reference = Reference(case, config)
    modes = ["diagnostic", "production"] + (["production_graph"] if device.is_cuda else [])
    if include_external:
        from scripts.monolithic_reference.pcg_backends import ExternalRunner  # noqa: PLC0415 - optional adapters

        modes += ["style3d", "kamino"] + (["style3d_graph", "kamino_graph"] if device.is_cuda else [])
    runners, setup, cold, bindings = {}, {}, [], {}
    vectors = {"reference_y": reference.y, "reference_delta": reference.scale * reference.y}

    def retain_solution(sample, runner, key):
        vectors[key] = runner.y.numpy().copy()
        sample.update(solution_key=key, accuracy=reference.measure(vectors[key]))

    for mode in modes:
        start = time.perf_counter()
        factory = ExternalRunner if mode.startswith(("style3d", "kamino")) else Runner
        runners[mode] = factory(case, mode, device, config)
        wp.synchronize_device(device)
        setup[mode] = (time.perf_counter() - start) * 1000
        w = runners[mode].workspace
        offsets = w.k_global_scalar_bsr.offsets.numpy()
        nnz = int(offsets[-1])  # BSR allocation can retain unused triplet capacity.
        bindings[mode] = array_fingerprint(
            {
                "offsets": offsets,
                "columns": w.k_global_scalar_bsr.columns[:nnz],
                "values": w.k_global_scalar_bsr.values[:nnz],
                "S": w.scale,
                "rhs": runners[mode].rhs,
                "q_cholesky": w._q_cholesky,
                "x_cholesky": w._x_cholesky,
            }
        )
        if len(set(bindings.values())) != 1:
            raise ValueError("Modes did not bind identical matrix, RHS, scale and actor factors")
        sample = _timed_solve(runners[mode], "first_solve")
        retain_solution(sample, runners[mode], f"{mode}-first-y")
        cold.append(sample)
    for _ in range(warmups):
        for runner in runners.values():
            runner.solve()
    wp.synchronize_device(device)
    samples, telemetry = [], []
    for scope in ("reuse", "new_generation"):
        for repeat in range(repeats):
            # Rotate and reverse each cycle so every mode occupies each position.
            offset = repeat % len(modes)
            order = modes[offset:] + modes[:offset]
            if repeat // len(modes) % 2:
                order = list(reversed(order))
            before = _gpu_sample(device)
            for mode in order:
                sample = _timed_solve(runners[mode], scope)
                sample["repeat"] = repeat
                retain_solution(sample, runners[mode], f"{mode}-{scope}-{repeat}-y")
                samples.append(sample)
            telemetry.append(
                {"scope": scope, "repeat": repeat, "order": order, "before": before, "after": _gpu_sample(device)}
            )
    convergence = {mode: observe_convergence(runners[mode], reference) for mode in modes if not mode.endswith("_graph")}
    for mode, curve in convergence.items():
        curve["iterate_key"] = f"{mode}-curve-y"
        vectors[curve["iterate_key"]] = curve.pop("iterate_y")
    counts = {}
    if include_external:
        from scripts.monolithic_reference.pcg_backends import software_counts  # noqa: PLC0415 - optional adapters

        counts = {mode: software_counts(runner) for mode, runner in runners.items()}
    return {
        "name": case.name,
        "n": case.n,
        "nq": case.nq,
        "nnz": case.matrix().nnz,
        "input_sha256": array_fingerprint(case.arrays),
        "metadata": case.metadata,
        "reference": reference.describe(),
        "modes": modes,
        "backend_metadata": {mode: getattr(runner, "metadata", {}) for mode, runner in runners.items()},
        "software_counts": counts,
        "setup_ms": setup,
        "bound_input_sha256": bindings,
        "first_solve": cold,
        "samples": samples,
        "telemetry": telemetry,
        "convergence": convergence,
        "vectors": vectors,
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_report(output, results, config):
    """Write compact numerical/timing tables and standalone convergence plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = [
        "# PCG 综合评估",
        "",
        "使用当前原生 diagnostic / production；CUDA 另测 production graph。参考解为相同 float32 输入定义的线性系统的 float64 直接解，不是非线性物理真值。",
        "",
        f"两种数值策略保留各自检查点、stagnation 和 restart 行为；K={config.maximum_iterations}、间隔 {config.true_residual_interval}、rtol={config.linear_tolerance:g}、atol={config.linear_absolute_tolerance:g}（默认待标定）。global/q/x 各自满足 ||r|| <= max(rtol * max(||rhs||, floor), atol)。原生 SUCCESS 与 float64 独立残差通过分开统计，不据此改写求解器状态。",
        "",
        "reuse：同 generation/绑定的预热重复求解。new_generation：相同矩阵更新 generation，计入 actor 因子重算、PCG 准备/捕获、结果读取及 delta 恢复；不包含 BSR 装配、Newton 或物理 step。初次/JIT/捕获成本另存在 JSON，不能用 reuse 替代整体集成效率。",
        "",
        "曲线是单独同步观察的原生非图求解，不计入耗时；各图模式的数值结果独立测量，沿用各自非图实现的迭代政策。GPU 频率/温度按交错轮次记录，未锁频；短基准不能代替稳定设备状态的完整轨迹验收。",
        "可选 Style3D/Kamino 使用共享 A/M 适配；Style3D FIXED_BUDGET 不声称原生收敛，迭代列表示调度更新数。Kamino SUCCESS 仅按全局递推残差判断。各模式统一以独立分块残差验收。外部 outer graph 由基准缓存，固定绑定时跨 generation 复用；并非完整外部物理 solver。",
        "",
        "| 系统 | 模式/范围 | 原生成功/独立通过/样本 | 迭代 p50 | 总耗时 p50/p95 ms | 解相对误差最大值 | 独立残差比最大值 | 图重建次数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    figures = []
    for result in results:
        for mode in result["modes"]:
            for scope in ("reuse", "new_generation"):
                rows = [r for r in result["samples"] if r["mode"] == mode and r["scope"] == scope]
                times = np.percentile([r["total_ms"] for r in rows], [50, 95])
                errors = [r["accuracy"].get("y_relative_error", math.inf) for r in rows]
                residuals = [r["accuracy"].get("max_true_ratio", math.inf) for r in rows]
                success = sum(r["accuracy"]["independent_residual_pass"] for r in rows)
                report.append(
                    f"| {result['name']} | {mode}/{scope} | {sum(r['status'] == 'SUCCESS' for r in rows)}/{success}/{len(rows)} | {np.median([r['iterations'] for r in rows]):.0f} | {times[0]:.3f}/{times[1]:.3f} | {max(errors):.3g} | {max(residuals):.3g} | {sum(r['graph_rebuilt'] for r in rows)} |"
                )
        fig, axes = plt.subplots(1, 3, figsize=(17 if len(result["modes"]) > 3 else 13, 4.2))
        for mode, curve in result["convergence"].items():
            x = [p["iterations"] for p in curve["points"]]
            for axis, metric in zip(axes[:2], ("max_acceptance_ratio", "y_relative_error"), strict=True):
                axis.semilogy(
                    x,
                    [max(p["accuracy"].get(metric, math.nan), 1e-16) for p in curve["points"]],
                    label=f"{mode}: {curve['result']['status']}",
                )
        axes[0].axhline(1.0, color="gray", linestyle="--", label="acceptance bound")
        axes[0].set_ylabel("Max global/q/x residual / acceptance bound")
        axes[1].set_ylabel("Relative solution error vs FP64")
        for axis in axes[:2]:
            axis.set_xlabel("PCG updates (Style3D: scheduled)")
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8)
        positions = np.arange(len(result["modes"]))
        for shift, scope in ((-0.18, "reuse"), (0.18, "new_generation")):
            values = [
                np.median([r["total_ms"] for r in result["samples"] if r["mode"] == mode and r["scope"] == scope])
                for mode in result["modes"]
            ]
            axes[2].bar(positions + shift, values, width=0.36, label=scope)
        axes[2].set_xticks(positions, [m.replace("production", "prod") for m in result["modes"]], rotation=40)
        axes[2].set_ylabel("Median complete linear call (ms)")
        axes[2].legend(fontsize=8)
        fig.suptitle(
            f"{result['name']} — n={result['n']}, estimated cond_1={result['reference']['condition_estimate_1']:.2g}"
        )
        fig.tight_layout()
        fig.savefig(output / f"{result['name']}.png", dpi=160)
        plt.close(fig)
        figures.extend(["", f"![{result['name']}]({result['name']}.png)", ""])
    report.extend(figures)
    (output / "REPORT.md").write_text("\n".join(report) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snapshot", type=Path, action="append", default=[])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--linear-absolute-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--include-external", action="store_true", help="Compare native Style3D/Kamino iterations with shared A/M"
    )
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmups < 1:
        parser.error("repeats and warmups must be positive")
    if args.output.exists():
        parser.error("Use a new output directory")
    cases = synthetic_cases() + [load_snapshot(path) for path in args.snapshot]
    if len({case.name for case in cases}) != len(cases):
        parser.error("Snapshot case names must be unique")
    args.output.mkdir(parents=True)
    wp.init()
    config = pcg_config(linear_absolute_tolerance=args.linear_absolute_tolerance)
    np.random.seed(20260913)  # noqa: NPY002 - SciPy onenormest uses NumPy's legacy global RNG.
    provenance = args.output / "provenance"
    provenance.mkdir()
    snapshot_sources(provenance)
    results = []
    for case in cases:
        print(f"COMPARE {case.name}: n={case.n}", flush=True)
        result = compare_case(
            case,
            args.device,
            repeats=args.repeats,
            warmups=args.warmups,
            config=config,
            include_external=args.include_external,
        )
        np.savez_compressed(args.output / f"{case.name}-input.npz", **case.arrays)
        result["vectors_file"] = f"{case.name}-solutions.npz"
        np.savez_compressed(args.output / result["vectors_file"], **result.pop("vectors"))
        (args.output / f"{case.name}.json").write_text(json.dumps(_json_safe(result), indent=2, allow_nan=False) + "\n")
        results.append(result)
    manifest = {
        "schema": "monolithic-pcg-benchmark/v2",
        "device": str(wp.get_device(args.device)),
        "device_name": wp.get_device(args.device).name,
        "config": asdict(config),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "include_external": args.include_external,
        "cases": [{"name": r["name"], "input_sha256": r["input_sha256"]} for r in results],
        "timing_scope": "Standalone linear solves; no physical stepping; convergence observers excluded from all timing",
        "performance_admission": "PENDING_MATCHED_DEVICE_STATE_REVIEW",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_report(args.output, results, config)
    evidence_index(args.output)
    print(f"Report: {args.output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
