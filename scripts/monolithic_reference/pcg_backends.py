# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Offline adapters: native Style3D/Kamino iterations with shared monolithic A and M.

Style3D's vec3 format cannot directly represent the coupled q/x actor matrix.
Only its matvec and preconditioner hooks are replaced; recurrence and reductions
remain native. Kamino exposes linear operator callbacks directly.
"""

import math
import time
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.kamino._src.linalg.conjugate import BatchedLinearOperator, CGSolver
from newton._src.solvers.monolithic import linear
from newton._src.solvers.style3d import linear_solver
from newton._src.solvers.style3d.linear_solver import PcgSolver
from scripts.monolithic_reference.benchmark_pcg import Reference, Runner


class SharedStyle3d(PcgSolver):
    """Keep native fixed-budget iterations; bind scalar A/M through zero-copy views."""

    def __init__(self, workspace, maximum_iterations):
        self.workspace = workspace
        n = workspace.layout.scalar_dof_count
        super().__init__((n + 2) // 3, workspace.device, maxIter=maximum_iterations)
        for array in (self.r, self.z, self.p, self.Ap):
            array.zero_()
        self.scalar_r, self.scalar_z, self.scalar_p, self.scalar_ap = (
            array.view(float).flatten()[:n] for array in (self.r, self.z, self.p, self.Ap)
        )

    def step2_update_z(self, inv_M):
        self.workspace.preconditioner.matvec(self.scalar_r, self.scalar_z, self.scalar_z, 1.0, 0.0)

    def step5_update_Ap(self, A_non_diag, A_diag, additional_Ap=None):
        self.workspace.operator.matvec(self.scalar_p, self.scalar_ap, self.scalar_ap, 1.0, 0.0)


class ExternalRunner(Runner):
    """Single-system adapter with explicit outer-graph reuse on stable bindings."""

    def __init__(self, case, mode, device, config):
        super().__init__(case, "diagnostic", device, config)
        self.mode = mode
        self.graph = None
        self.use_graph = mode.endswith("_graph")
        self.kind = mode.removesuffix("_graph")
        if self.kind == "style3d":
            self.solver = SharedStyle3d(self.workspace, config.maximum_iterations)
            self.solution = wp.zeros(self.solver.dim, dtype=wp.vec3, device=device)
            self.y = self.solution.view(float).flatten()[: case.n]
            self.b = wp.zeros_like(self.solution)
            self.rhs = self.b.view(float).flatten()[: case.n]
            self.rhs.assign(case.arrays["rhs_hat"])
            self.metadata = {
                "recurrence": "Native Style3D PcgSolver.solve/update/reduction kernels",
                "adapter": "Replace only A/M hooks with shared actor operators; vec3 padding stays zero",
                "stopping": "Fixed budget; no native convergence assertion or true-residual restart",
                "iteration_kind": "Scheduled updates, including zero-alpha updates",
            }
        elif self.kind == "kamino":
            dims = wp.array([case.n], dtype=int, device=device)
            offsets = wp.zeros(1, dtype=int, device=device)

            def operator(source):
                def gemv(x, y, active, alpha, beta):
                    source.matvec(x, y, y, alpha, beta)

                return BatchedLinearOperator(
                    gemv,
                    1,
                    case.n,
                    dims,
                    wp.get_device(device),
                    wp.float32,
                    vio=offsets,
                    total_vec_size=case.n,
                )

            reference = Reference(case, config)
            active_blocks = [0] + ([1] if case.nq else []) + ([2] if case.n > case.nq else [])
            # A global absolute bound sufficient for every nonempty block;
            # acceptance is still checked independently against true residuals.
            atol = float(config.linear_tolerance * min(reference.denominators[active_blocks]))
            self.solver = CGSolver(
                operator(self.workspace.operator),
                Mi=operator(self.workspace.preconditioner),
                world_active=wp.full(1, True, dtype=wp.bool, device=device),
                maxiter=wp.full(1, config.maximum_iterations, dtype=int, device=device),
                atol=atol,
                rtol=0.0,
                use_graph=self.use_graph,
                use_graph_conditionals=True,
                loop_granularity=1,
            )
            self.metadata = {
                "recurrence": "Native Kamino CGSolver, one active world",
                "adapter": "Public A/M callbacks bound to shared actor operators",
                "stopping": "Global recursive residual; no true-residual restart",
                "absolute_tolerance": atol,
                "iteration_kind": "Native cur_iter; graph conditional checks each cycle",
            }
        else:
            raise ValueError(f"Unknown external PCG mode: {mode}")
        self.metadata["graph_policy"] = "Benchmark-owned outer graph; reuse fixed addresses/dimensions/lambda/config"

    def _execute(self):
        if self.kind == "style3d":
            self.solver.solve(None, None, None, self.b, None, self.solution, self.config.maximum_iterations)
        else:
            self.y.zero_()
            self.solver.solve(self.rhs, self.y)
        # Include direction recovery for every external call, even when it fails.
        wp.launch(
            linear._scale_vector,
            self.y.size,
            [self.workspace.scale, self.y, self.delta],
            device=self.workspace.device,
        )

    def solve(self):
        rebuilt = False
        capture_seconds = 0.0
        if self.use_graph:
            if self.graph is None:
                self._execute()  # Compile all native kernels outside capture.
                wp.synchronize_device(self.workspace.device)
                start = time.perf_counter()
                with wp.ScopedCapture(device=self.workspace.device) as capture:
                    self._execute()
                self.graph = capture.graph
                wp.capture_launch(self.graph)
                capture_seconds = time.perf_counter() - start
                rebuilt = True
            else:
                wp.capture_launch(self.graph)
        else:
            self._execute()
        if self.kind == "style3d":
            status, iterations = "FIXED_BUDGET", self.config.maximum_iterations
        else:
            iterations = int(self.solver.cur_iter.numpy()[0])
            residual_sq = float(self.solver.dot_product[0].numpy()[0])
            tolerance_sq = float(self.solver.atol_sq.numpy()[0])
            status = "SUCCESS" if residual_sq <= tolerance_sq else "MAX_ITERATIONS"
            if not math.isfinite(residual_sq):
                status = "NONFINITE"
        return {
            "status": status,
            "iterations": iterations,
            "scheduled_slots": self.config.maximum_iterations if self.kind == "style3d" else None,
            "pcg_policy": self.kind,
            "true_residual_checks": 0,
            "residual_replacements": 0,
            "prepare_seconds": None,  # Included in solve_ms, not separately instrumented.
            "capture_seconds": capture_seconds,
            "graph_rebuilt": rebuilt,
            "use_cuda_graph": self.use_graph,
        }

    def observe(self, reference):
        """Collect native update vectors in a separate non-graph run."""
        if self.use_graph:
            raise ValueError("Convergence observers require a non-graph runner")
        self.y.zero_()
        iterates = [self.y.numpy().copy()]
        points = [{"iterations": 0, "accuracy": reference.measure(iterates[0])}]
        method = "step7_update_x_r" if self.kind == "style3d" else "do_iteration"
        original = getattr(self.solver, method)

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            y = self.y.numpy().copy()
            iterates.append(y)
            points.append({"iterations": len(points), "accuracy": reference.measure(y)})
            return result

        with patch.object(self.solver, method, side_effect=observed):
            result = self.solve()
        return {
            "scope": "Untimed native non-graph cycle observations; no solver buffer writes",
            "points": points,
            "iterate_y": np.stack(iterates),
            "result": result,
        }


def software_counts(runner):
    """Count Python launch/readback calls in a separate warmed solve, not GPU kernels."""
    counts = {"launch_calls": 0, "numpy_cuda_reads": 0, "graph_launch_calls": 0, "style3d_inner_calls": 0}

    def instrument(original, key, cuda_only=False):
        def call(*args, **kwargs):
            if not cuda_only or args[0].device.is_cuda:
                counts[key] += 1
            return original(*args, **kwargs)

        return call

    with ExitStack() as stack:
        for owner, name, key, cuda_only in (
            (wp, "launch", "launch_calls", False),
            (wp, "capture_launch", "graph_launch_calls", False),
            (wp.array, "numpy", "numpy_cuda_reads", True),
            (linear_solver, "array_inner", "style3d_inner_calls", False),
        ):
            stack.enter_context(patch.object(owner, name, instrument(getattr(owner, name), key, cuda_only)))
        runner.solve()
    return {"scope": "Untimed Python API counts; excludes native-internal CUDA operations", **counts}
