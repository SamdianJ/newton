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
import itertools
import math
import time
from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from .linear_types import (
    MonolithicLinearLayout,
    MonolithicLinearSolveResult,
    MonolithicLinearStatus,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from .pcg_common import group_data, group_policy, linear_rounding
from .pcg_production_kernels import (
    _NONFINITE,
    REDUCTION_CHUNK,
    _active_mask,
    _alpha_guard,
    _begin_slot,
    _checkpoint_decision,
    _chunk_norm_sums,
    _chunk_products,
    _clear_norm_work,
    _DeviceResult,
    _finalize_result,
    _finish_direction,
    _finish_norms,
    _finish_update,
    _gamma_and_beta_guard,
    _group_norm_sums,
    _initialize_state,
    _initialize_vectors,
    _loop_condition,
    _make_products_kernel,
    _masked_direction,
    _masked_q_solve,
    _masked_rigid_solve,
    _masked_scaled_abs_mv,
    _masked_scaled_csr_mv,
    _masked_update,
    _masked_x_solve,
    _Parameters,
    _phase_mask,
    _publish_delta,
    _recover_candidate,
    _replace_recursive,
    _residual_and_maxima,
    _sum_products,
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
    group_residual_floors: tuple[float, ...] | None = None

    def __post_init__(self):
        if self.group_residual_floors is not None:
            if not isinstance(self.group_residual_floors, tuple) or not self.group_residual_floors:
                raise ValueError("Group residual floors must be a nonempty tuple")
            for value in self.group_residual_floors:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, float, np.integer, np.floating))
                    or not np.isfinite(value)
                    or not np.finfo(np.float32).smallest_subnormal <= value <= np.finfo(np.float32).max
                ):
                    raise ValueError("Group residual floors must be positive finite float32 values")
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


def _plan_identity(config):
    """Return the configuration fields that the captured PCG schedule depends on."""
    return replace(config, linear_absolute_tolerance=0.0)


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
        layout=None,
        rigid_cholesky=None,
        rigid_rows=None,
        cable_preconditioner=None,
        cloth_preconditioner=None,
        material_operator=None,
    ):
        self.config = MaskedPcgConfig() if config is None else config
        if not isinstance(self.config, MaskedPcgConfig):
            raise TypeError("config must be MaskedPcgConfig")
        self.device = wp.get_device(y.device)
        if self.device.is_capturing:
            raise RuntimeError("Construct/bind the plan outside graph capture")
        self._cloth_preconditioner = cloth_preconditioner
        self._cloth_signature = None if cloth_preconditioner is None else cloth_preconditioner.signature()
        self._cloth_active = None if cloth_preconditioner is None else wp.zeros(1, dtype=int, device=self.device)
        self._cable_preconditioner = cable_preconditioner
        self._cable_signature = None if cable_preconditioner is None else cable_preconditioner.signature()
        self._cable_active = None if cable_preconditioner is None else wp.zeros(1, dtype=int, device=self.device)
        self._material_operator = material_operator
        self._material_signature = None if material_operator is None else material_operator.signature()
        self._material_active = None if material_operator is None else wp.zeros(1, dtype=int, device=self.device)
        self.n = y.size
        # Owned vectors use the solution's precision; stopping thresholds are FP32 and compared in FP64.
        self.dtype = y.dtype
        self.matrix_type = wp.mat33d if self.dtype == wp.float64 else wp.mat33
        self.nq = q_cholesky.shape[0]
        self.nx = x_cholesky.size
        self.layout = layout or MonolithicLinearLayout(self.nq, self.nx)
        count = len(self.layout.rigid_ranges)
        self.rigid_cholesky = (
            wp.zeros((0, 6, 6), dtype=self.dtype, device=self.device) if rigid_cholesky is None else rigid_cholesky
        )
        self.rigid_rows = wp.zeros((0, 6), dtype=int, device=self.device) if rigid_rows is None else rigid_rows
        if (self.layout.q_dof_count, self.layout.dynamic_particle_count) != (self.nq, self.nx):
            raise ValueError("PCG factors do not match the actor layout")
        membership, aliases = group_data(self.layout)
        self.group_count = len(self.layout.groups)
        self.row_groups = wp.array(membership, dtype=int, device=self.device)
        self.compat_groups = wp.array(aliases, dtype=int, device=self.device)
        floors, _ = group_policy(self.layout, self.config, require_limits=False)
        self.group_floors = wp.array(floors, dtype=float, device=self.device)
        if self.n <= 0 or self.n != self.layout.scalar_dof_count:
            raise ValueError("Expected n = q_dof_count + 3 * dynamic_particle_count > 0")
        arrays = (
            offsets,
            columns,
            values,
            scale,
            q_cholesky,
            x_cholesky,
            self.rigid_cholesky,
            self.rigid_rows,
            rhs_hat,
            y,
        )
        for a in arrays:
            if not isinstance(a, wp.array) or a.device != self.device or not a.is_contiguous:
                raise ValueError("All bound arrays must be contiguous on the same device")
        expected = (
            (offsets, wp.int32, (self.n + 1,)),
            (columns, wp.int32, (columns.size,)),
            (values, self.dtype, (values.size,)),
            (scale, self.dtype, (self.n,)),
            (q_cholesky, self.dtype, (self.nq, self.nq)),
            (x_cholesky, self.matrix_type, (self.nx,)),
            (self.rigid_cholesky, self.dtype, (count, 6, 6)),
            (self.rigid_rows, wp.int32, (count, 6)),
            (rhs_hat, self.dtype, (self.n,)),
            (y, self.dtype, (self.n,)),
        )
        for a, dtype, shape in expected:
            if a.dtype != dtype or a.shape != shape:
                raise ValueError(f"Unexpected dtype/shape: expected {dtype}, {shape}")
        if columns.size != values.size:
            raise ValueError("columns and values must have equal storage capacities")
        self.offsets, self.columns, self.values = offsets, columns, values
        self.scale, self.q_cholesky, self.x_cholesky = scale, q_cholesky, x_cholesky
        self.rhs, self.y = rhs_hat, y
        self.delta = wp.zeros(self.n, dtype=self.dtype, device=self.device) if delta is None else delta
        if (
            not isinstance(self.delta, wp.array)
            or self.delta.device != self.device
            or self.delta.dtype != self.dtype
            or self.delta.shape != (self.n,)
            or not self.delta.is_contiguous
        ):
            raise ValueError("delta must be a contiguous same-device vector in the solution precision")
        for out in (self.y, self.delta):
            for other in (*arrays[:-1], self.delta if out is self.y else self.y):
                if _overlap(out, other):
                    raise ValueError("Outputs must not overlap each other or bound input storage")
        self._binding_signature = tuple(map(_signature, (*arrays, self.delta)))
        for name in ("r", "z", "p", "ap", "magnitude", "true_r", "gap", "temporary", "delta_candidate"):
            setattr(self, name, wp.zeros(self.n, dtype=self.dtype, device=self.device))
        self.products = wp.zeros(1, dtype=self.dtype, device=self.device)
        # Products reduce fixed chunks of each world segment, so a world's chunk
        # tree does not depend on how many worlds precede or follow it.
        segments = self.layout.world_segments
        product_chunks, segment_chunks = [], [0]
        for start, end in itertools.pairwise(segments):
            product_chunks.extend((i, min(i + REDUCTION_CHUNK, end)) for i in range(start, end, REDUCTION_CHUNK))
            segment_chunks.append(len(product_chunks))
        product_chunks = np.asarray(product_chunks, dtype=np.int32).reshape(-1, 2)
        self.product_start, self.product_end = (
            wp.array(product_chunks[:, k], dtype=int, device=self.device) for k in range(2)
        )
        self.segment_chunks = wp.array(np.asarray(segment_chunks, dtype=np.int32), dtype=int, device=self.device)
        self.product_partials = wp.zeros(len(product_chunks), dtype=self.dtype, device=self.device)
        self.segment_products = wp.zeros(self.layout.world_count, dtype=self.dtype, device=self.device)
        # Residual norms reduce fixed chunks of each contiguous convergence group.
        chunks, group_chunks = [], [0]
        for group, (_, start, count) in enumerate(self.layout.groups[1:], 1):
            end = start + count
            chunks.extend((group, i, min(i + REDUCTION_CHUNK, end)) for i in range(start, end, REDUCTION_CHUNK))
            group_chunks.append(len(chunks))
        chunks = np.asarray(chunks, dtype=np.int32).reshape(-1, 3)
        self.chunk_group, self.chunk_start, self.chunk_end = (
            wp.array(chunks[:, k], dtype=int, device=self.device) for k in range(3)
        )
        self.group_chunks = wp.array(np.asarray(group_chunks, dtype=np.int32), dtype=int, device=self.device)
        self.norm_partials = wp.zeros((len(chunks), 8), dtype=self.dtype, device=self.device)
        self.scalars = wp.zeros(3, dtype=self.dtype, device=self.device)
        # Per group: true residual, residual gap, rhs and |A| |y| norms.
        self.maxima = wp.zeros(4 * self.group_count, dtype=self.dtype, device=self.device)
        self.sums = wp.zeros(4 * self.group_count, dtype=self.dtype, device=self.device)
        self.norms = wp.zeros(4 * self.group_count, dtype=self.dtype, device=self.device)
        self.error = wp.zeros(1, dtype=int, device=self.device)
        self.state = wp.zeros(1, dtype=_DeviceResult, device=self.device)
        self.regularization = wp.zeros(1, dtype=self.dtype, device=self.device)
        self.zero_guess = wp.ones(1, dtype=int, device=self.device)
        self.condition = wp.zeros(1, dtype=int, device=self.device)
        self.residual_norm_limits = wp.full(self.group_count, float("inf"), dtype=float, device=self.device)
        # The absolute tolerance follows each Newton iterate's limits, so it is a
        # bound value rather than part of the captured schedule.
        self.absolute_tolerance = wp.full(
            1, float(self.config.linear_absolute_tolerance), dtype=float, device=self.device
        )
        self._conditional_graph_supported = self.device.is_cuda and wp.is_conditional_graph_supported()
        # Fuse single-tile dots of one world; retain the chunk reduction otherwise.
        self._products_kernel = (
            _make_products_kernel(1 << (self.n - 1).bit_length(), self.dtype)
            if self.device.is_cuda and self.n <= 2048 and self.layout.world_count == 1
            else None
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
            _plan_identity(self.config),
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
        self.params.absolute_tolerance = self.absolute_tolerance
        self.params.floors = self.group_floors
        self.params.aliases = self.compat_groups
        self.params.replacement_threshold = float(c.residual_replacement_threshold)
        self.params.residual_norm_limits = self.residual_norm_limits
        self.params.rounding = linear_rounding(self.dtype)

    @classmethod
    def from_workspace(cls, workspace, rhs_hat, y, *, config, delta=None):
        matrix = workspace.k_global_scalar_bsr
        if not workspace._scaling_valid or not workspace._factor_is_current():
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
            layout=workspace.layout,
            rigid_cholesky=workspace._rigid_cholesky,
            rigid_rows=workspace.rigid_rows,
            cable_preconditioner=workspace._cable_preconditioner,
            cloth_preconditioner=workspace._cloth_preconditioner,
            material_operator=workspace._material_operator,
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
        if self._cloth_preconditioner is not None and self._cloth_signature != self._cloth_preconditioner.signature():
            raise ValueError("Cloth preconditioner storage changed")
        if self._cable_preconditioner is not None and self._cable_signature != self._cable_preconditioner.signature():
            raise ValueError("Cable factor storage changed; construct a new PCG plan")
        if self._material_operator is not None and self._material_signature != self._material_operator.signature():
            raise ValueError("Material operator storage changed; construct a new PCG plan")
        arrays = tuple(value for value in vars(self).values() if isinstance(value, wp.array))
        if (
            self._stream_id() != self._stream
            or tuple(map(_signature, arrays)) != self._storage_signature
            or (_plan_identity(self.config), self.n, self.nq, self.nx, self.layout, self._conditional_graph_supported)
            != self._schedule_signature
        ):
            raise ValueError("Stream, schedule or bound storage changed; construct a new PCG plan")

    def _validate_workspace(self, regularization):
        if self._workspace is not None:
            workspace = self._workspace
            if (
                not workspace._current(self._generation)
                or not workspace._scaling_valid
                or not workspace._factor_is_current()
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
            workspace._rigid_cholesky,
            workspace.rigid_rows,
            rhs_hat,
            y,
            self.delta if delta is None else delta,
        )
        bindings_match = (
            self._workspace is workspace
            and self._cloth_preconditioner is workspace._cloth_preconditioner
            and (self._cloth_preconditioner is None or self._cloth_signature == self._cloth_preconditioner.signature())
            and self._cable_preconditioner is workspace._cable_preconditioner
            and (self._cable_preconditioner is None or self._cable_signature == self._cable_preconditioner.signature())
            and self._material_operator is workspace._material_operator
            and (self._material_operator is None or self._material_signature == self._material_operator.signature())
            and self._stream == self._stream_id()
            and self.layout == workspace.layout
            and _plan_identity(config) == _plan_identity(self.config)
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
        if checkpoint:
            self._launch(
                _masked_scaled_abs_mv,
                self.n,
                [
                    self.offsets,
                    self.columns,
                    self.values,
                    self.scale,
                    self.regularization,
                    x,
                    self.magnitude,
                    self.state,
                ],
            )
        if self._material_operator is not None:
            # Element tangents applied to element-relative displacements, in the
            # same scaled units, after the assembled rows have been written.
            self._launch(_phase_mask, 1, [self.state, checkpoint, self._material_active])
            self._material_operator.add_product(
                x,
                self.scale,
                self.ap,
                magnitude=self.magnitude if checkpoint else None,
                active=self._material_active,
                error=self.error,
                code=_NONFINITE,
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
            partials = self.product_partials
            self._launch(
                _chunk_products,
                partials.size,
                [a, b, self.product_start, self.product_end, partials, self.state, self.error],
            )
            self._launch(
                _sum_products, 1, [partials, self.segment_chunks, self.segment_products, self.products, self.state]
            )

    def _emit_direction(self, slot):
        if self._cloth_preconditioner is not None:
            self._launch(_active_mask, 1, [self.state, self._cloth_active])
            self._cloth_preconditioner.apply(self.r, self.z, active=self._cloth_active)
        if self._cable_preconditioner is not None:
            self._launch(_active_mask, 1, [self.state, self._cable_active])
            self._cable_preconditioner.apply(self.r, self.z, active=self._cable_active)
        if len(self.layout.rigid_ranges):
            self._launch(
                _masked_rigid_solve,
                len(self.layout.rigid_ranges),
                [self.rigid_cholesky, self.rigid_rows, self.r, self.z, self.state, self.error],
            )
        if self.nq:  # Static layout test, not a device-dependent branch.
            self._launch(
                _masked_q_solve,
                1,
                [self.q_cholesky, self.layout.q_scalar_offset, self.r, self.temporary, self.z, self.state, self.error],
            )
        if self.nx:
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
        self._launch(_clear_norm_work, 4 * self.group_count, [self.maxima, self.sums, self.state, initial])
        self._launch(
            _residual_and_maxima,
            self.n,
            [
                self.rhs,
                self.ap,
                self.magnitude,
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
            _chunk_norm_sums,
            self.norm_partials.shape[0],
            [
                self.rhs,
                self.true_r,
                self.gap,
                self.magnitude,
                self.maxima,
                self.chunk_group,
                self.chunk_start,
                self.chunk_end,
                self.norm_partials,
                self.state,
                self.error,
                initial,
            ],
        )
        self._launch(
            _group_norm_sums,
            self.group_count,
            [self.norm_partials, self.group_chunks, self.sums, self.state, self.error, initial],
        )
        self._launch(
            _finish_norms, 4 * self.group_count, [self.maxima, self.sums, self.norms, self.state, self.error, initial]
        )
        self._launch(_checkpoint_decision, 1, [self.norms, self.state, self.error, self.params, slot, initial, final])
        self._launch(_replace_recursive, self.n, [self.true_r, self.r, self.state])

    def enqueue_solve(self, observer=None):
        """Emit bounded PCG with conditional checkpoint batches on supported CUDA graphs.

        Does not capture, launch a graph, read device data or return a host
        convergence result. The caller must prepare, warm kernels and serialize
        use of the bound buffers before entering this low-level boundary.
        Every replay resets all lifecycle state. Neither
        convergence nor error changes the host node sequence. CPU, non-captured
        execution and older CUDA drivers retain the fixed masked schedule.

        ``observer`` is a diagnostic host callback invoked after each scheduled
        checkpoint of an uncaptured solve; it may read device data but must not
        write any bound buffer, so the numerical path is unchanged.
        """
        if observer is not None and self.device.is_capturing:
            raise RuntimeError("A checkpoint observer requires an uncaptured solve")
        nslots = self.config.maximum_iterations
        interval = self.config.true_residual_interval
        self._launch(_initialize_state, 1, [self.state, self.error, self.scalars, nslots])
        self._launch(_initialize_vectors, self.n, [self.y, self.r, self.zero_guess])
        self._emit_checkpoint(0, initial=True)
        if observer is not None:
            observer()
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
                if observer is not None:
                    observer()
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


def solve_pcg_production(
    workspace, rhs_hat, y, *, generation, warm_start, config, delta=None, use_graph=False, observe=False
):
    """Synchronous compatibility adapter for existing solve_pcg callers.

    Host validation and the final packet read are deliberately outside the
    captured core. The outer Newton loop remains synchronous. To embed PCG in
    a larger graph, bind a MaskedPcgPlan outside it and call plan.enqueue_solve() instead.
    ``observe`` records every checkpoint's true-residual group norms without
    changing the solve; it requires an uncaptured solve.
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
            pcg_policy=PCG_POLICY,
        )

    workspace._last_masked_pcg_result = None
    if observe and use_graph:
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if not isinstance(config, MonolithicPcgConfig):
        return invalid(MonolithicLinearStatus.INVALID_ARGUMENT)
    if workspace.device.is_capturing:
        raise RuntimeError("Use MaskedPcgPlan.enqueue_solve() inside capture, not the synchronous solve adapter")
    if not workspace._current(generation) or not workspace._scaling_valid:
        return invalid(MonolithicLinearStatus.STALE_GENERATION)
    if not workspace._factor_is_current():
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
        # Per-Newton residual caps and the absolute tolerance change numerically
        # without changing graph bindings.
        plan.config = prod_config
        plan.absolute_tolerance.fill_(float(prod_config.linear_absolute_tolerance))
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
    history = []
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
        checks = 0

        def observer():
            nonlocal checks
            state = plan.state.numpy()[0]
            if int(state["true_residual_checks"]) > checks:
                checks = int(state["true_residual_checks"])
                norms = plan.norms.numpy()[: plan.group_count]
                history.append((int(state["last_true_slot"]), tuple(map(float, norms))))

        plan._in_flight = True
        plan.enqueue_solve(observer if observe else None)
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
        warm_start=warm_start,
        checkpoint_history=tuple(history),
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
