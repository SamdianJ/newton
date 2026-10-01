# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Fixed-topology, device-masked PCG for the monolithic linear workspace.

The synchronous adapter at the end of this file implements the production
dispatch from ``solve_pcg``; diagnostic retains its separate host policy.
``MaskedPcgPlan.enqueue_solve()`` is the capture-safe part: no host readback, allocation,
data-dependent Python branch, or Python early exit. CUDA graphs can stop between
checkpoint batches through device-side conditional nodes.

One enable mask covers the *entire coupled system*, not its q/x blocks.
All expensive kernels, including the CSR/BSR(1x1) matvec and block solves, honor
that mask. Scheduled true-residual kernels use RUNNING instead: a zero recursive
residual may suspend updates while awaiting the next static checkpoint.

Binding/storage and factor validity must be checked outside capture. The matrix
is the workspace's assembled scalar BSR, and the factors are its existing dense
q and per-particle 3x3 Cholesky blocks. This module does not assemble or factor.
"""

import enum
import hashlib
import math
import time
from dataclasses import dataclass

import numpy as np
import warp as wp

from .linear_types import (
    MonolithicLinearLayout,
    MonolithicLinearSolveResult,
    MonolithicLinearStatus,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from .pcg_common import group_data, group_policy
from .pcg_production_kernels import (
    _alpha_guard,
    _begin_slot,
    _checkpoint_decision,
    _clear_norm_work,
    _clear_products,
    _DeviceResult,
    _finalize_result,
    _finish_direction,
    _finish_norms,
    _finish_update,
    _gamma_and_beta_guard,
    _initialize_state,
    _initialize_vectors,
    _loop_condition,
    _make_products_kernel,
    _masked_direction,
    _masked_products,
    _masked_q_solve,
    _masked_scaled_csr_mv,
    _masked_update,
    _masked_x_solve,
    _Parameters,
    _publish_delta,
    _recover_candidate,
    _replace_recursive,
    _residual_and_maxima,
    _residual_norm_sums,
    _study_mask,
)

PCG_POLICY = "masked_smart_restart_v3"

wp.set_module_options({"enable_backward": False})

# Terminal codes intentionally match MonolithicLinearStatus in linear.py.


class ProductionStatus(enum.IntEnum):
    RUNNING = -1
    SUCCESS = 0
    NON_POSITIVE_CURVATURE = 13
    NON_POSITIVE_PRECONDITIONED_RESIDUAL = 15
    NONFINITE_ITERATION = 16
    MAX_ITERATIONS = 17


@dataclass(frozen=True)
class MaskedPcgConfig:
    maximum_iterations: int = 200
    true_residual_interval: int = 10
    linear_tolerance: float = 1.0e-4
    residual_floor_global: float = 1.0e-12
    residual_floor_q: float = 1.0e-12
    residual_floor_x: float = 1.0e-12
    residual_replacement_threshold: float = 0.1
    linear_absolute_tolerance: float = 1.0e-6

    def __post_init__(self):
        for name in ("maximum_iterations", "true_residual_interval"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value >= 2**31:
                raise ValueError(f"{name} must fit int32")
        positive = {"linear_tolerance", "residual_floor_global", "residual_floor_q", "residual_floor_x"}
        for name in positive | {
            "residual_replacement_threshold",
            "linear_absolute_tolerance",
        }:
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
                raise ValueError(f"{name} must be a real scalar")
            if not math.isfinite(value) or value < 0 or value > float(np.finfo(np.float32).max):
                raise ValueError(f"{name} must be nonnegative and finite in float32")
            if name in positive and value < np.finfo(np.float32).smallest_subnormal:
                raise ValueError(f"{name} must be positive and representable in float32")
            if name == "linear_absolute_tolerance" and 0 < value < np.finfo(np.float32).smallest_subnormal:
                raise ValueError(f"{name} must be representable in float32")

    @classmethod
    def from_monolithic(cls, config):
        return cls(**{name: getattr(config, name) for name in cls.__dataclass_fields__})


@dataclass(frozen=True)
class MaskedPcgResult:
    status: ProductionStatus
    enabled: bool
    pending_check: int
    iteration_slots: int
    iterations: int
    true_residual_checks: int
    residual_replacements: int
    last_true_slot: int
    convergence_slot: int
    failure_slot: int
    residual_matches_y: bool
    output_valid: bool
    ratios: tuple[float, float, float]
    current_drifts: tuple[float, float, float]


def _signature(arr):
    return (arr.ptr, arr.shape, arr.strides, str(arr.dtype), str(arr.device))


def _span(arr):
    start = arr.ptr or 0
    return start, start + arr.size * wp.types.type_size_in_bytes(arr.dtype)


def _overlap(a, b):
    if not a.size or not b.size:
        return False
    a0, a1 = _span(a)
    b0, b1 = _span(b)
    return a0 < b1 and b0 < a1


class MaskedPcgPlan:
    """Preallocated plan bound to one scalar-BSR matrix and actor factors.

    Array addresses, layout and config are immutable for this plan/graph.
    Contents may be updated in place between solves. Rebind (construct another
    plan) when a BSR rebuild reallocates storage. Calls on a plan must be
    serialized on one stream; it is not reentrant.
    """

    def __init__(
        self,
        *,
        offsets,
        columns,
        values,
        scale,
        q_cholesky,
        x_cholesky,
        rhs_hat,
        y,
        delta=None,
        config: MaskedPcgConfig | None = None,
        _study_soft=None,
        layout=None,
    ):
        self.config = MaskedPcgConfig() if config is None else config
        if not isinstance(self.config, MaskedPcgConfig):
            raise TypeError("config must be MaskedPcgConfig")
        self.device = wp.get_device(y.device)
        if self.device.is_capturing:
            raise RuntimeError("Construct/bind the plan outside graph capture")
        self._study_soft = _study_soft
        self._study_signature = None if _study_soft is None else _study_soft.signature()
        self._study_active = None if _study_soft is None else wp.zeros(1, dtype=int, device=self.device)
        self.n = y.size
        self.nq = q_cholesky.shape[0]
        self.nx = x_cholesky.size
        self.layout = layout or MonolithicLinearLayout(self.nq, self.nx)
        if (self.layout.q_dof_count, self.layout.dynamic_particle_count) != (self.nq, self.nx):
            raise ValueError("PCG factors do not match the actor layout")
        membership, aliases = group_data(self.layout)
        self.group_count = len(self.layout.groups)
        self.row_groups = wp.array(membership, dtype=int, device=self.device)
        self.compat_groups = wp.array(aliases, dtype=int, device=self.device)
        floors, _ = group_policy(self.layout, self.config)
        self.group_floors = wp.array(floors, dtype=float, device=self.device)
        if self.n <= 0 or self.n != self.nq + 3 * self.nx:
            raise ValueError("Expected n = q_dof_count + 3 * dynamic_particle_count > 0")
        arrays = (offsets, columns, values, scale, q_cholesky, x_cholesky, rhs_hat, y)
        for a in arrays:
            if not isinstance(a, wp.array) or a.device != self.device or not a.is_contiguous:
                raise ValueError("All bound arrays must be contiguous on the same device")
        expected = (
            (offsets, wp.int32, (self.n + 1,)),
            (columns, wp.int32, (columns.size,)),
            (values, wp.float32, (values.size,)),
            (scale, wp.float32, (self.n,)),
            (q_cholesky, wp.float32, (self.nq, self.nq)),
            (x_cholesky, wp.mat33, (self.nx,)),
            (rhs_hat, wp.float32, (self.n,)),
            (y, wp.float32, (self.n,)),
        )
        for a, dtype, shape in expected:
            if a.dtype != dtype or a.shape != shape:
                raise ValueError(f"Unexpected dtype/shape: expected {dtype}, {shape}")
        if columns.size != values.size:
            raise ValueError("columns and values must have equal storage capacities")
        self.offsets, self.columns, self.values = offsets, columns, values
        self.scale, self.q_cholesky, self.x_cholesky = scale, q_cholesky, x_cholesky
        self.rhs, self.y = rhs_hat, y
        self.delta = wp.zeros(self.n, dtype=float, device=self.device) if delta is None else delta
        if (
            not isinstance(self.delta, wp.array)
            or self.delta.device != self.device
            or self.delta.dtype != wp.float32
            or self.delta.shape != (self.n,)
            or not self.delta.is_contiguous
        ):
            raise ValueError("delta must be a contiguous same-device float32 vector")
        for out in (self.y, self.delta):
            for other in (*arrays[:-1], self.delta if out is self.y else self.y):
                if _overlap(out, other):
                    raise ValueError("Outputs must not overlap each other or bound input storage")
        self._binding_signature = tuple(map(_signature, (*arrays, self.delta)))
        for name in ("r", "z", "p", "ap", "true_r", "gap", "temporary", "delta_candidate"):
            setattr(self, name, wp.zeros(self.n, dtype=float, device=self.device))
        self.products = wp.zeros(1, dtype=float, device=self.device)
        self.scalars = wp.zeros(3, dtype=float, device=self.device)
        self.maxima = wp.zeros(3 * self.group_count, dtype=float, device=self.device)
        self.sums = wp.zeros(3 * self.group_count, dtype=float, device=self.device)
        self.norms = wp.zeros(3 * self.group_count, dtype=float, device=self.device)
        self.error = wp.zeros(1, dtype=int, device=self.device)
        self.state = wp.zeros(1, dtype=_DeviceResult, device=self.device)
        self.regularization = wp.zeros(1, dtype=float, device=self.device)
        self.zero_guess = wp.ones(1, dtype=int, device=self.device)
        self.condition = wp.zeros(1, dtype=int, device=self.device)
        self.residual_norm_limits = wp.full(self.group_count, float("inf"), dtype=float, device=self.device)
        self._conditional_graph_supported = self.device.is_cuda and wp.is_conditional_graph_supported()
        # Fuse single-tile dots; retain the parallel reduction for larger systems.
        self._products_kernel = (
            _make_products_kernel(1 << (self.n - 1).bit_length()) if self.device.is_cuda and self.n <= 2048 else None
        )
        self.graph = None
        self._workspace = None
        self._generation = None
        self._stream = self._stream_id()
        self._in_flight = False
        self._prepared_regularization = None
        self._storage = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        self._storage_signature = tuple(map(_signature, self._storage))
        self._schedule_signature = (
            self.config,
            self.n,
            self.nq,
            self.nx,
            self.layout,
            self._conditional_graph_supported,
        )
        self._layout_digest = self._layout_signature()
        c = self.config
        self.params = _Parameters()
        self.params.tolerance = float(c.linear_tolerance)
        self.params.absolute_tolerance = float(c.linear_absolute_tolerance)
        self.params.floors = self.group_floors
        self.params.aliases = self.compat_groups
        self.params.replacement_threshold = float(c.residual_replacement_threshold)
        self.params.residual_norm_limits = self.residual_norm_limits

    @classmethod
    def from_workspace(cls, workspace, rhs_hat, y, *, config, delta=None):
        matrix = workspace.k_global_scalar_bsr
        if not workspace._scaling_valid or not workspace._factor_valid:
            raise ValueError("Binding requires valid scaling and actor factors")
        plan = cls(
            offsets=matrix.offsets,
            columns=matrix.columns,
            values=matrix.values,
            scale=workspace.scale,
            q_cholesky=workspace._q_cholesky,
            x_cholesky=workspace._x_cholesky,
            rhs_hat=rhs_hat,
            y=y,
            delta=delta,
            config=config,
            _study_soft=workspace._study_soft,
            layout=workspace.layout,
        )
        plan._workspace = workspace
        plan._generation = workspace._generation
        return plan

    def _stream_id(self):
        return wp.get_stream(self.device).cuda_stream if self.device.is_cuda else None

    def _layout_signature(self):
        offsets = self.offsets.numpy()
        end = int(offsets[-1])
        if offsets[0] != 0 or np.any(np.diff(offsets.astype(np.int64)) < 0) or not 0 <= end <= self.values.size:
            raise ValueError("Invalid BSR offsets or capacity")
        columns = self.columns.numpy()[:end]
        if np.any(columns < 0) or np.any(columns >= self.n):
            raise ValueError("Invalid BSR column index")
        return hashlib.sha256(offsets.tobytes() + columns.tobytes()).digest()

    def _validate_storage(self):
        if self._study_soft is not None:
            self._study_soft.validate()
        arrays = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        if (
            self._stream_id() != self._stream
            or tuple(map(_signature, arrays)) != self._storage_signature
            or (self.config, self.n, self.nq, self.nx, self.layout, self._conditional_graph_supported)
            != self._schedule_signature
        ):
            raise ValueError("Stream, schedule or bound storage changed; construct a new PCG plan")

    def _validate_workspace(self, regularization):
        if self._workspace is not None:
            workspace = self._workspace
            if (
                not workspace._current(self._generation)
                or not workspace._scaling_valid
                or not workspace._factor_valid
                or regularization != workspace._lambda
                or not self.matches_workspace(workspace, self.rhs, self.y, self.config, self.delta)
            ):
                raise ValueError("Stale generation, scaling or regularization factors")

    def matches_workspace(self, workspace, rhs_hat, y, config, delta=None):
        """Check bindings; authenticate sparse layout once per numerical generation."""
        matrix = workspace.k_global_scalar_bsr
        arrays = (
            matrix.offsets,
            matrix.columns,
            matrix.values,
            workspace.scale,
            workspace._q_cholesky,
            workspace._x_cholesky,
            rhs_hat,
            y,
            self.delta if delta is None else delta,
        )
        bindings_match = (
            self._workspace is workspace
            and self._study_soft is workspace._study_soft
            and (self._study_soft is None or self._study_signature == self._study_soft.signature())
            and self._stream == self._stream_id()
            and self.layout == workspace.layout
            and config == self.config
            and tuple(map(_signature, arrays)) == self._binding_signature
        )
        if not bindings_match:
            return False
        return self._generation == workspace._generation or self._layout_signature() == self._layout_digest

    def set_solve_parameters(self, *, regularization: float, zero_initial_guess: bool = True):
        """Update device parameters before enqueue/replay; matrix/factors must match."""
        if self.device.is_capturing:
            raise RuntimeError("Set host-provided solve parameters outside capture")
        if (
            isinstance(regularization, (bool, np.bool_))
            or not math.isfinite(regularization)
            or regularization < 0
            or regularization > np.finfo(np.float32).max
        ):
            raise ValueError("regularization must be finite, nonnegative float32")
        if type(zero_initial_guess) is not bool:
            raise TypeError("zero_initial_guess must be bool")
        self._validate_storage()
        if self._in_flight:
            raise RuntimeError("Collect the previous result before reusing this plan")
        self._validate_workspace(regularization)
        if self._workspace is None and self._layout_signature() != self._layout_digest:
            raise ValueError("BSR layout changed; construct a new PCG plan")
        self._prepared_regularization = regularization
        self.regularization.fill_(float(regularization))
        self.zero_guess.fill_(int(zero_initial_guess))

    def _launch(self, kernel, dim, args):
        wp.launch(kernel, dim=dim, inputs=args, device=self.device)

    def _emit_matvec(self, x, *, checkpoint):
        self._launch(
            _masked_scaled_csr_mv,
            self.n,
            [
                self.offsets,
                self.columns,
                self.values,
                self.scale,
                self.regularization,
                x,
                self.ap,
                self.state,
                self.error,
                checkpoint,
            ],
        )

    def _emit_products(self, a, b):
        if self._products_kernel is not None:
            wp.launch_tiled(
                self._products_kernel,
                dim=1,
                inputs=[a, b, self.products, self.state],
                block_dim=128,
                device=self.device,
            )
        else:
            self._launch(_clear_products, 1, [self.products, self.state])
            self._launch(_masked_products, self.n, [a, b, self.products, self.state, self.error])

    def _emit_direction(self, slot):
        if self.nq:  # Static layout test, not a device-dependent branch.
            self._launch(
                _masked_q_solve,
                1,
                [self.q_cholesky, self.layout.q_scalar_offset, self.r, self.temporary, self.z, self.state, self.error],
            )
        if self._study_soft is not None:
            self._launch(_study_mask, 1, [self.state, self._study_active])
            self._study_soft.apply(self.r, self.z, active=self._study_active, error=self.error)
        elif self.nx:
            self._launch(
                _masked_x_solve,
                self.nx,
                [self.x_cholesky, self.layout.x_scalar_offset, self.r, self.z, self.state, self.error],
            )
        self._emit_products(self.r, self.z)
        self._launch(_gamma_and_beta_guard, 1, [self.products, self.scalars, self.state, self.error, self.params, slot])
        self._launch(_masked_direction, self.n, [self.z, self.p, self.scalars, self.state, self.error])
        self._launch(_finish_direction, 1, [self.state, self.error, slot])

    def _emit_checkpoint(self, slot, *, initial=False, final=False):
        self._emit_matvec(self.y, checkpoint=True)
        self._launch(_clear_norm_work, 3 * self.group_count, [self.maxima, self.sums, self.state, initial])
        self._launch(
            _residual_and_maxima,
            self.n,
            [
                self.rhs,
                self.ap,
                self.r,
                self.true_r,
                self.gap,
                self.maxima,
                self.state,
                self.error,
                self.row_groups,
                initial,
            ],
        )
        self._launch(
            _residual_norm_sums,
            self.n,
            [self.rhs, self.true_r, self.gap, self.maxima, self.sums, self.state, self.error, self.row_groups, initial],
        )
        self._launch(
            _finish_norms, 3 * self.group_count, [self.maxima, self.sums, self.norms, self.state, self.error, initial]
        )
        self._launch(_checkpoint_decision, 1, [self.norms, self.state, self.error, self.params, slot, initial, final])
        self._launch(_replace_recursive, self.n, [self.true_r, self.r, self.state])

    def enqueue_solve(self):
        """Emit bounded PCG with conditional checkpoint batches on supported CUDA graphs.

        Does not capture, launch a graph, read device data or return a host
        convergence result. The caller must prepare, warm kernels and serialize
        use of the bound buffers before entering this low-level boundary.
        Every replay resets all lifecycle state. Neither
        convergence nor error changes the host node sequence. CPU, non-captured
        execution and older CUDA drivers retain the fixed masked schedule.
        """
        nslots = self.config.maximum_iterations
        interval = self.config.true_residual_interval
        self._launch(_initialize_state, 1, [self.state, self.error, self.scalars, nslots])
        self._launch(_initialize_vectors, self.n, [self.y, self.r, self.zero_guess])
        self._emit_checkpoint(0, initial=True)
        self._emit_direction(0)

        def iteration(slot, *, checkpoint, final=False):
            self._launch(_begin_slot, 1, [self.state, self.error, slot])
            self._emit_matvec(self.p, checkpoint=False)
            self._emit_products(self.p, self.ap)
            self._launch(_alpha_guard, 1, [self.products, self.scalars, self.state, self.error, self.params, slot])
            self._launch(
                _masked_update, self.n, [self.y, self.r, self.p, self.ap, self.scalars, self.state, self.error]
            )
            self._launch(_finish_update, 1, [self.state, self.error, slot])
            if checkpoint:
                self._emit_checkpoint(slot, final=final)
            if not final:
                self._emit_direction(slot)

        if self.device.is_capturing and self._conditional_graph_supported:
            full_end = (nslots // interval) * interval

            def batch(count):
                for i in range(count):
                    iteration(-1, checkpoint=i == count - 1)

            def loop_body():
                batch(interval)
                self._launch(_loop_condition, 1, [self.state, self.condition, full_end])

            self._launch(_loop_condition, 1, [self.state, self.condition, full_end])
            if full_end:
                wp.capture_while(self.condition, loop_body)
            remainder = nslots - full_end
            if remainder:
                self._launch(_loop_condition, 1, [self.state, self.condition, nslots])
                wp.capture_if(self.condition, lambda: batch(remainder))
        else:
            for slot in range(1, nslots + 1):
                iteration(slot, checkpoint=slot % interval == 0 or slot == nslots, final=slot == nslots)
        self._launch(_recover_candidate, self.n, [self.scale, self.y, self.delta_candidate, self.state, self.error])
        self._launch(_finalize_result, 1, [self.state, self.error])
        self._launch(_publish_delta, self.n, [self.delta_candidate, self.delta, self.state])
        return self.state

    def capture(self):
        """Capture once after binding. Recreate the plan after storage reallocation."""
        if not self.device.is_cuda:
            raise RuntimeError("CUDA capture requires a CUDA device; enqueue_solve() also works on CPU")
        if self.device.is_capturing:
            raise RuntimeError("Call enqueue_solve(), not capture(), inside an external capture")
        self._validate_storage()
        self._validate_workspace(self._prepared_regularization)
        if self._in_flight or self._prepared_regularization is None:
            raise RuntimeError("Prepare an idle plan before capture")
        wp.load_module(module=_initialize_state.module, device=self.device)
        if self._products_kernel is not None:
            wp.load_module(module=self._products_kernel.module, device=self.device, block_dim=128)
        with wp.ScopedCapture(device=self.device) as capture:
            self.enqueue_solve()
        self.graph = capture.graph
        return self.graph

    def replay(self):
        if self.graph is None:
            raise RuntimeError("capture() must be called before replay()")
        self._validate_storage()
        if self._in_flight:
            raise RuntimeError("Collect the previous result before replaying")
        self._validate_workspace(self._prepared_regularization)
        self._in_flight = True
        wp.capture_launch(self.graph)

    def collect_result(self):
        """One packet readback at the PCG boundary. Never call during capture."""
        if self.device.is_capturing:
            raise RuntimeError("collect_result() is a host boundary, not capture-safe")
        item = self.state.numpy()[0]
        self._in_flight = False
        status = ProductionStatus(int(item["status"]))
        if status == ProductionStatus.RUNNING or (
            status == ProductionStatus.SUCCESS and not bool(item["output_valid"])
        ):
            raise RuntimeError("PCG has no terminal result")
        return MaskedPcgResult(
            status=status,
            enabled=bool(item["enabled"]),
            pending_check=int(item["pending_check"]),
            iteration_slots=int(item["slots"]),
            iterations=int(item["effective_iterations"]),
            true_residual_checks=int(item["true_residual_checks"]),
            residual_replacements=int(item["residual_replacements"]),
            last_true_slot=int(item["last_true_slot"]),
            convergence_slot=int(item["convergence_slot"]),
            failure_slot=int(item["failure_slot"]),
            residual_matches_y=bool(item["residual_current"]),
            output_valid=bool(item["output_valid"]),
            ratios=tuple(map(float, item["ratios"])),
            current_drifts=tuple(map(float, item["drifts"])),
        )


def solve_pcg_production(workspace, rhs_hat, y, *, generation, warm_start, config, delta=None, use_graph=False):
    """Synchronous compatibility adapter for existing solve_pcg callers.

    Host validation and the final packet read are deliberately outside the
    captured core. The outer Newton loop remains synchronous. To embed PCG in
    a larger graph, bind a MaskedPcgPlan outside it and call plan.enqueue_solve() instead.
    """

    def invalid(code):
        return MonolithicLinearSolveResult(
            status=code,
            generation=generation,
            iterations=0,
            true_residual_checks=0,
            residual_replacements=0,
            rho=math.nan,
            rho_q=math.nan,
            rho_x=math.nan,
            min_p_ap=math.nan,
            min_r_z=math.nan,
            pcg_policy=PCG_POLICY,
        )

    workspace._last_masked_pcg_result = None
    if not isinstance(config, MonolithicPcgConfig):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if workspace.device.is_capturing:
        raise RuntimeError("Use MaskedPcgPlan.enqueue_solve() inside capture, not the synchronous solve adapter")
    if not workspace._current(generation) or not workspace._scaling_valid:
        return invalid(MonolithicLinearStatus.STALE_GENERATION)
    if not workspace._factor_valid:
        return invalid(MonolithicLinearStatus.PRECONDITIONER_NOT_FACTORED)
    if warm_start not in (MonolithicPcgWarmStart.ZERO, MonolithicPcgWarmStart.SAME_GENERATION):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if not workspace._vector_valid(rhs_hat) or not workspace._vector_valid(y):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if warm_start == MonolithicPcgWarmStart.SAME_GENERATION and workspace._warm_start_key != workspace._solve_binding(
        y
    ):
        return invalid(MonolithicLinearStatus.STALE_GENERATION)
    prepare_start = time.perf_counter()
    try:
        prod_config = MaskedPcgConfig.from_monolithic(config)
        if prod_config.maximum_iterations > workspace.capacities.pcg_max_iterations:
            return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
        plan = getattr(workspace, "_masked_pcg_plan", None)
        if plan is not None and plan._in_flight:
            raise RuntimeError("Collect the previous result before reusing this plan")
        if plan is None or not plan.matches_workspace(workspace, rhs_hat, y, prod_config, delta):
            plan = MaskedPcgPlan.from_workspace(workspace, rhs_hat, y, config=prod_config, delta=delta)
            workspace._masked_pcg_plan = plan
        # Numeric validity was checked above; matching storage/layout permits
        # reuse of captured nodes. Bind this request without accepting stale callers.
        plan._generation = generation
        # Per-Newton residual caps change numerically without changing graph bindings.
        plan.residual_norm_limits.assign(np.asarray(group_policy(workspace.layout, config)[1], dtype=np.float32))
        plan.set_solve_parameters(
            regularization=workspace._lambda, zero_initial_guess=warm_start == MonolithicPcgWarmStart.ZERO
        )
    except (TypeError, ValueError, AttributeError):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    prepare_seconds = time.perf_counter() - prepare_start
    capture_seconds = 0.0
    graph_rebuilt = False
    workspace._warm_start_key = None
    if use_graph and workspace.device.is_cuda:
        if plan.graph is None:
            capture_start = time.perf_counter()
            plan.capture()
            # Warp lazily instantiates/uploads on first launch. Charge that
            # host cost to graph setup, including the first launch submission.
            plan.replay()
            capture_seconds = time.perf_counter() - capture_start
            graph_rebuilt = True
        else:
            plan.replay()
    else:
        plan._in_flight = True
        plan.enqueue_solve()
    result = plan.collect_result()
    workspace._last_masked_pcg_result = result
    if result.status == ProductionStatus.SUCCESS:
        workspace._warm_start_key = workspace._solve_binding(y)
    # Keep existing rho fields fresh; checkpoint_ratios retain the last valid check.
    ratios = result.ratios if result.residual_matches_y else (math.nan,) * 3
    return MonolithicLinearSolveResult(
        status=MonolithicLinearStatus(int(result.status)),
        generation=generation,
        iterations=result.iterations,
        true_residual_checks=result.true_residual_checks,
        residual_replacements=result.residual_replacements,
        rho=ratios[0],
        rho_q=ratios[1],
        rho_x=ratios[2],
        min_p_ap=math.nan,
        min_r_z=math.nan,
        warm_start=warm_start,
        initial_guess_norm=math.nan,
        recursive_true_residual_gap=(math.nan,) * 3,
        stagnation_window=None,
        pcg_policy=PCG_POLICY,
        scheduled_slots=result.iteration_slots,
        converged_at_slot=result.convergence_slot,
        last_checked_slot=result.last_true_slot,
        failure_slot=result.failure_slot,
        residual_current=result.residual_matches_y,
        output_valid=result.output_valid,
        checkpoint_ratios=result.ratios,
        prepare_seconds=prepare_seconds,
        capture_seconds=capture_seconds,
        graph_rebuilt=graph_rebuilt,
    )
