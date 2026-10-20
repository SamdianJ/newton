# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Workspace ownership and dispatch for monolithic linear solves."""

from functools import partial

import numpy as np
import warp as wp
import warp.sparse as sparse
from warp.optim.linear import LinearOperator

from . import bsr, preconditioner
from .bsr import (
    ASSEMBLY_STRUCTS,
    MonolithicLinearAssembly,
    _create_triplets,
    _monolithic_scaled_matvec,
    _pack_assembly_counts,
    _pack_assembly_result,
    _validate_aq_finite,
    _validate_bsr_finite,
    _validate_triplets,
)
from .bsr_reduction import (
    _fixed_bsr_publish_status,
    _FixedInternalBsrReduction,
    _FixedScalarBsrReduction,
    _ScalarBsrReduction,
)
from .cable_preconditioner import CablePreconditioner
from .cloth_preconditioner import ClothPreconditioner
from .linear_types import (
    MonolithicDenseLinearOracle,
    MonolithicLinearBuildResult,
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearSolveResult,
    MonolithicLinearStatus,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from .pcg_common import group_data
from .pcg_production import solve_pcg_production
from .preconditioner import _monolithic_preconditioner_matvec


class MonolithicLinearWorkspace:
    """Own one current assembly with fixed producer slots and guarded appends.

    Fixed-slot producers write the current assembly's reserved prefix. Contact
    producers reserve slots with guarded helpers. Raw array writes require the
    caller to enforce the current generation; only helper writes are token-guarded.
    """

    def __init__(
        self,
        layout: MonolithicLinearLayout,
        capacities: MonolithicLinearCapacities,
        particle_to_dynamic: wp.array[int],
        *,
        device: wp.DeviceLike,
        pcg_mode: str = "diagnostic",
        use_cuda_graph: bool = False,
        contact_support=None,
        cable_row_order=None,
        dtype=wp.float32,
    ):
        if dtype not in ASSEMBLY_STRUCTS:
            raise ValueError("The linear workspace stores float32 or float64 values")
        if any(k == "cable" for k, *_ in layout.material_ranges) and cable_row_order is None:
            raise ValueError("Material layouts require an explicit cable preconditioner ordering")
        if pcg_mode not in ("diagnostic", "production"):
            raise ValueError("pcg_mode must be diagnostic or production")
        if type(use_cuda_graph) is not bool or (
            use_cuda_graph and (pcg_mode != "production" or not wp.get_device(device).is_cuda)
        ):
            raise ValueError("use_cuda_graph requires CUDA production PCG and a bool")
        if contact_support is None and capacities.contact_factor_count * (layout.q_dof_count + 9) >= 2**31:
            raise ValueError("Contact scalar support exceeds int32 capacity")
        if capacities.global_scalar_triplet_count >= 2**30:
            raise ValueError("Scalar sort scratch exceeds int32 capacity")
        self.use_cuda_graph = use_cuda_graph
        self.dtype = dtype
        """Scalar type of every stored matrix, scaling, factor and PCG value."""
        self.matrix_type = wp.mat33d if dtype == wp.float64 else wp.mat33
        self._native = np.float64 if dtype == wp.float64 else np.float32
        structs = ASSEMBLY_STRUCTS[dtype]
        self._pcg_mode = pcg_mode
        self._pcg_active = False
        self._rhs_norms_cached = False
        self.layout = layout
        self._layout_identity = layout
        membership, aliases = group_data(layout)
        self.group_count = len(layout.groups)
        self.row_groups = wp.array(membership, dtype=int, device=device)
        self.compat_groups = wp.array(aliases, dtype=int, device=device)
        self.group_floors = wp.ones(self.group_count, dtype=float, device=device)
        self.group_limits = wp.full(self.group_count, float("inf"), dtype=float, device=device)
        self.capacities = capacities
        self.device = wp.get_device(device)
        self._cable_preconditioner = (
            CablePreconditioner(layout, cable_row_order, device=self.device, dtype=dtype)
            if any(k == "cable" for k, *_ in layout.material_ranges)
            else None
        )
        self._cloth_preconditioner = (
            ClothPreconditioner(layout, device=self.device, dtype=dtype)
            if any(k == "cloth" for k, *_ in layout.material_ranges)
            else None
        )
        if (
            particle_to_dynamic.device != self.device
            or particle_to_dynamic.dtype != wp.int32
            or particle_to_dynamic.ndim != 1
        ):
            raise ValueError("particle_to_dynamic must be an int array on the workspace device")
        mapping = particle_to_dynamic.numpy()
        if np.any(mapping < -1) or not np.array_equal(
            np.sort(mapping[mapping >= 0]), np.arange(layout.dynamic_particle_count)
        ):
            raise ValueError("particle_to_dynamic must map each dynamic particle exactly once")
        self.particle_to_dynamic = particle_to_dynamic
        self.aq_actor_dense = wp.zeros((layout.q_dof_count, layout.q_dof_count), dtype=dtype, device=device)
        self.k_global_scalar_bsr = sparse.bsr_zeros(
            layout.scalar_dof_count, layout.scalar_dof_count, dtype, device=device
        )
        self.ax_internal_bsr3 = sparse.bsr_zeros(
            layout.dynamic_particle_count, layout.dynamic_particle_count, self.matrix_type, device=device
        )
        self._active_token = wp.zeros(1, dtype=int, device=device)
        self._segment_graphs = None
        self._token = 0
        self._generation = None
        self._sealed = True
        self._valid = False
        self._fixed_patterns = None
        self._fixed_bsr = None
        self._global = _create_triplets(
            structs.scalar,
            capacities.global_scalar_triplet_count,
            layout.scalar_dof_count,
            dtype,
            device,
            self._active_token,
            0,
        )
        self._scalar_reduction = _ScalarBsrReduction(
            capacities.global_scalar_triplet_count, self.device, use_cuda_graph=use_cuda_graph, dtype=dtype
        )
        self._internal = _create_triplets(
            structs.mat33,
            capacities.internal_mat33_triplet_count,
            layout.dynamic_particle_count,
            self.matrix_type,
            device,
            self._active_token,
            0,
        )
        factors = structs.factors()
        if contact_support is None:
            # Standalone component producers declare their own factor support.
            width = layout.q_dof_count + 9
            offsets = np.arange(capacities.contact_factor_count + 1, dtype=np.int64) * width
            columns = np.zeros(int(offsets[-1]), dtype=np.int32)
            local_indices = np.zeros_like(columns)
        else:
            offsets, columns, local_indices = contact_support
            slots = capacities.contact_factor_count
            if slots < len(offsets) - 1:
                columns, local_indices = columns[: offsets[slots]], local_indices[: offsets[slots]]
                offsets = offsets[: slots + 1]
            elif slots > len(offsets) - 1:
                offsets = np.r_[offsets, np.full(slots - len(offsets) + 1, offsets[-1])]
        if offsets[-1] >= 2**31:
            raise ValueError("Contact scalar support exceeds int32 capacity")
        factors.offsets = wp.array(offsets.astype(np.int32), dtype=int, device=device)
        factors.columns = wp.array(columns, dtype=int, device=device)
        factors.local_indices = wp.array(local_indices, dtype=int, device=device)
        factors.values = wp.zeros(len(columns), dtype=dtype, device=device)
        factors.weights = wp.zeros(capacities.contact_factor_count, dtype=dtype, device=device)
        factors.kind = wp.zeros(capacities.contact_factor_count, dtype=int, device=device)
        factors.candidate_tid = wp.full(capacities.contact_factor_count, -1, dtype=int, device=device)
        factors.count = wp.zeros(1, dtype=int, device=device)
        factors.status = wp.zeros(1, dtype=int, device=device)
        factors.capacity = capacities.contact_factor_count
        factors.scalar_dof_count = layout.scalar_dof_count
        factors.active_token = self._active_token
        factors.token = 0
        self._factors = factors
        self._assembly_counts = wp.zeros(6, dtype=int, device=device)
        self._assembly_result = wp.zeros(3, dtype=int, device=device)
        n = layout.scalar_dof_count
        for name in (
            "dynamic_diagonal",
            "diagonal",
            "scale",
            "_matrix_diagonal",
            "_scaled_x",
            "_k_scaled",
            "_preconditioned",
        ):
            setattr(self, name, wp.zeros(n, dtype=dtype, device=device))
        self.factor_setup_count = 0
        self.factor_failure_count = 0
        self.factor_last_status = None
        self._status = wp.zeros(1, dtype=int, device=device)
        self.floor_count = wp.zeros(1, dtype=int, device=device)
        self._q_preconditioner = wp.zeros_like(self.aq_actor_dense)
        self._q_cholesky = wp.zeros_like(self.aq_actor_dense)
        self._x_preconditioner = wp.zeros(layout.dynamic_particle_count, dtype=self.matrix_type, device=device)
        self._x_cholesky = wp.zeros_like(self._x_preconditioner)
        rigid_rows = np.array([range(start, start + 6) for _, start in layout.rigid_ranges], dtype=np.int32).reshape(
            -1, 6
        )
        self.rigid_rows = wp.array(rigid_rows, dtype=int, device=device)
        self._rigid_preconditioner = wp.zeros((len(layout.rigid_ranges), 6, 6), dtype=dtype, device=device)
        self._rigid_cholesky = wp.zeros_like(self._rigid_preconditioner)
        self._scaling_valid = False
        self._factor_valid = False
        self._factor_binding = None
        self._lambda = 0.0
        self._warm_start_key = None
        self.operator = LinearOperator((n, n), dtype, self.device, partial(_monolithic_scaled_matvec, self))
        self.preconditioner = LinearOperator(
            (n, n), dtype, self.device, partial(_monolithic_preconditioner_matvec, self)
        )

    def _set_fixed_triplet_pattern(
        self,
        *,
        global_rows: np.ndarray,
        global_columns: np.ndarray,
        internal_rows: np.ndarray,
        internal_columns: np.ndarray,
    ) -> None:
        """Reserve immutable owner patterns once, before the first assembly."""
        if self._generation is not None or self._fixed_patterns is not None:
            raise ValueError("Fixed triplet patterns must be configured once before assembly")
        patterns = []
        for row_indices, column_indices, buffer in [
            (global_rows, global_columns, self._global),
            (internal_rows, internal_columns, self._internal),
        ]:
            rows, columns = np.asarray(row_indices), np.asarray(column_indices)
            if (
                rows.ndim != 1
                or columns.shape != rows.shape
                or not np.issubdtype(rows.dtype, np.integer)
                or not np.issubdtype(columns.dtype, np.integer)
                or rows.size > buffer.capacity
                or np.any(rows < 0)
                or np.any(rows >= buffer.row_count)
                or np.any(columns < 0)
                or np.any(columns >= buffer.column_count)
            ):
                raise ValueError("Fixed triplet pattern must contain in-range integer pairs within capacity")
            patterns.append((rows.astype(np.int32, copy=True), columns.astype(np.int32, copy=True)))
        self._fixed_patterns = tuple(
            tuple(wp.array(indices, dtype=int, device=self.device) for indices in pattern) for pattern in patterns
        )

    def _enable_fixed_bsr_storage(self):
        """Bind fixed CUDA builders once for a complete immutable owner pattern."""
        if self._generation is not None or self._fixed_bsr is not None or self._fixed_patterns is None:
            raise ValueError("Fixed BSR storage requires an unused workspace with owner patterns")
        rows, columns = self._fixed_patterns[1]
        if rows.size != self._internal.capacity:
            raise ValueError("Fixed BSR storage requires the complete internal owner capacity")
        # Release unused legacy scratch before allocating its fixed replacements.
        self._scalar_reduction = None
        scalar = _FixedScalarBsrReduction(
            self.layout.scalar_dof_count,
            self._global.capacity,
            self.device,
            use_cuda_graph=self.use_cuda_graph,
            dtype=self.dtype,
        )
        internal = _FixedInternalBsrReduction(
            self.layout.dynamic_particle_count,
            rows.numpy(),
            columns.numpy(),
            self.device,
            use_cuda_graph=self.use_cuda_graph,
            dtype=self.dtype,
        )
        self._fixed_bsr = scalar, internal
        self.k_global_scalar_bsr, self.ax_internal_bsr3 = scalar.matrix, internal.matrix

    def begin_assembly(self, generation: MonolithicLinearGeneration) -> MonolithicLinearAssembly:
        if self.layout != self._layout_identity:
            raise ValueError("Actor layout changed; rebuild the linear workspace")
        if self._generation is not None and (generation.step_index, generation.assembly_sequence) <= (
            self._generation.step_index,
            self._generation.assembly_sequence,
        ):
            raise ValueError("Assembly sequence must increase within each step")
        self._token += 1
        self._active_token.fill_(self._token)
        self._generation = generation
        self._sealed = False
        self._valid = False
        self.factor_setup_count = 0
        self.factor_failure_count = 0
        self.factor_last_status = None
        self._scaling_valid = False
        self._factor_valid = False
        self._warm_start_key = None
        # Copy only struct descriptors: old views retain their captured token.
        structs = ASSEMBLY_STRUCTS[self.dtype]
        for name, struct_type, fields in [
            (
                "_global",
                structs.scalar,
                (
                    "rows",
                    "columns",
                    "values",
                    "count",
                    "status",
                    "capacity",
                    "row_count",
                    "column_count",
                    "active_token",
                ),
            ),
            (
                "_internal",
                structs.mat33,
                (
                    "rows",
                    "columns",
                    "values",
                    "count",
                    "status",
                    "capacity",
                    "row_count",
                    "column_count",
                    "active_token",
                ),
            ),
            (
                "_factors",
                structs.factors,
                (
                    "offsets",
                    "columns",
                    "local_indices",
                    "values",
                    "weights",
                    "kind",
                    "candidate_tid",
                    "count",
                    "status",
                    "capacity",
                    "scalar_dof_count",
                    "active_token",
                ),
            ),
        ]:
            previous = getattr(self, name)
            current = struct_type()
            for field in fields:
                setattr(current, field, getattr(previous, field))
            current.token = self._token
            setattr(self, name, current)
        if self._segment_graphs is None:
            self._initialize_assembly_device()
        else:
            buffers = [
                self.aq_actor_dense,
                self._factors.offsets,
                self._factors.columns,
                self._factors.local_indices,
                self._factors.values,
                self._factors.weights,
                self._factors.count,
                self._factors.status,
            ]
            for buffer in (self._global, self._internal):
                buffers.extend((buffer.rows, buffer.columns, buffer.values, buffer.count, buffer.status))
            if self._fixed_patterns is not None:
                buffers.extend(array for pattern in self._fixed_patterns for array in pattern)
            self._segment_graphs.run(
                "assembly_initialize", (self.layout.identity,), buffers, self._initialize_assembly_device
            )
        return MonolithicLinearAssembly(generation, self.aq_actor_dense, self._internal, self._factors, self._global)

    def _initialize_assembly_device(self):
        self.aq_actor_dense.zero_()
        for buffer in (self._global, self._internal, self._factors):
            buffer.count.zero_()
            buffer.status.zero_()
        for index, buffer in enumerate((self._global, self._internal)):
            if self._fixed_patterns is not None:
                rows, columns = self._fixed_patterns[index]
                count = rows.size
                if count:
                    # Appended entries overwrite their values; only reserved slots need clearing.
                    buffer.values[:count].zero_()
                    wp.copy(buffer.rows, rows, count=count)
                    wp.copy(buffer.columns, columns, count=count)
                buffer.count.fill_(count)
        self._factors.values.zero_()
        self._factors.weights.zero_()

    def finalize_assembly(self, *, generation: MonolithicLinearGeneration) -> MonolithicLinearBuildResult:
        if generation != self._generation or self._sealed:
            return MonolithicLinearBuildResult(MonolithicLinearStatus.STALE_GENERATION, generation, 0, 0, 0, 0, 0)
        self._sealed = True
        self._active_token.zero_()
        wp.launch(
            _pack_assembly_counts,
            1,
            [self._global, self._internal, self._factors, self._assembly_counts],
            device=self.device,
        )
        packet = self._assembly_counts.numpy()
        counts = [int(value) for value in packet[:3]]
        nnz = (0, 0)
        status = MonolithicLinearStatus.SUCCESS
        for index, (buffer, overflow) in enumerate(
            [
                (self._global, MonolithicLinearStatus.GLOBAL_TRIPLET_OVERFLOW),
                (self._internal, MonolithicLinearStatus.INTERNAL_TRIPLET_OVERFLOW),
                (self._factors, MonolithicLinearStatus.CONTACT_FACTOR_OVERFLOW),
            ]
        ):
            code = int(packet[3 + index])
            if code:
                status = MonolithicLinearStatus(code)
                break
            minimum = self._fixed_patterns[index][0].size if self._fixed_patterns is not None and index < 2 else 0
            if counts[index] < minimum:
                status = MonolithicLinearStatus.INVALID_ARGUMENT
                break
            if counts[index] > buffer.capacity:
                status = overflow
                break
        if status == MonolithicLinearStatus.SUCCESS:
            # Fixed-slot kernels write raw arrays, bypassing append-time validation.
            for buffer, count in zip((self._global, self._internal), counts[:2], strict=True):
                if count:
                    wp.launch(
                        _validate_triplets,
                        count,
                        [buffer.rows, buffer.columns, buffer.values, buffer.row_count, self._global.status],
                        device=self.device,
                    )
            status = MonolithicLinearStatus(int(self._global.status.numpy()[0]))
        if status == MonolithicLinearStatus.SUCCESS:
            try:
                for (matrix, triplets), count in zip(
                    [(self.k_global_scalar_bsr, self._global), (self.ax_internal_bsr3, self._internal)],
                    counts[:2],
                    strict=True,
                ):
                    if self._fixed_bsr is not None:
                        reduction = self._fixed_bsr[0] if matrix is self.k_global_scalar_bsr else self._fixed_bsr[1]
                        if matrix is self.k_global_scalar_bsr:
                            reduction.build(triplets, layout_identity=self.layout.identity, active_count=count)
                        else:
                            reduction.build(triplets, layout_identity=self.layout.identity)
                        if matrix is self.ax_internal_bsr3:
                            wp.launch(
                                _fixed_bsr_publish_status, 1, [triplets.status, self._global.status], device=self.device
                            )
                        continue
                    if matrix is self.k_global_scalar_bsr:
                        self._scalar_reduction.build(
                            matrix,
                            triplets,
                            count,
                            layout_identity=self.layout.identity,
                            graphs=self._segment_graphs,
                        )
                        continue
                    arrays = (triplets.rows, triplets.columns, triplets.values)
                    # Warp sorts by array length; device count only masks the tail.
                    # Keep empty-capacity arrays intact: slicing them is unsupported.
                    if count < triplets.capacity:
                        arrays = tuple(array[:count] for array in arrays)
                    sparse.bsr_set_from_triplets(matrix, *arrays, count=triplets.count)
            except RuntimeError:
                status = MonolithicLinearStatus.BSR_BUILD_FAILURE
        if status == MonolithicLinearStatus.SUCCESS:
            # Finite local terms can overflow during owner or duplicate reductions.
            for matrix in (self.k_global_scalar_bsr, self.ax_internal_bsr3):
                if matrix.nrow:
                    wp.launch(
                        _validate_bsr_finite,
                        matrix.nrow,
                        [matrix.offsets, matrix.values, self._global.status],
                        device=self.device,
                    )
            wp.launch(
                _validate_aq_finite,
                self.aq_actor_dense.shape,
                [self.aq_actor_dense, self._global.status],
                device=self.device,
            )
            wp.launch(
                _pack_assembly_result,
                1,
                [
                    self._global.status,
                    self.k_global_scalar_bsr.offsets,
                    self.ax_internal_bsr3.offsets,
                    self._assembly_result,
                ],
                device=self.device,
            )
            result = self._assembly_result.numpy()
            status = MonolithicLinearStatus(int(result[0]))
            if status == MonolithicLinearStatus.SUCCESS:
                nnz = (int(result[1]), int(result[2]))
        self._valid = status == MonolithicLinearStatus.SUCCESS
        return MonolithicLinearBuildResult(
            status,
            generation,
            *counts,
            *nnz,
        )

    def _current(self, generation) -> bool:
        return self.layout == self._layout_identity and generation == self._generation and self._sealed and self._valid

    def _solve_binding(self, y):
        """Bind warm starts to layout, precision and actual operator/factor storage."""
        matrix = self.k_global_scalar_bsr
        arrays = (
            matrix.offsets,
            matrix.columns,
            matrix.values,
            self.scale,
            self._q_cholesky,
            self._x_cholesky,
            self._rigid_cholesky,
            self.rigid_rows,
            self.row_groups,
        )
        if self._cable_preconditioner is not None:
            arrays += self._cable_preconditioner.buffers()
        if self._cloth_preconditioner is not None:
            arrays += self._cloth_preconditioner.buffers()
        return (
            self._generation,
            y.ptr,
            self.layout,
            tuple((a.ptr, a.shape, a.strides, str(a.dtype), str(a.device)) for a in arrays),
        )

    def _factor_is_current(self):
        return self._factor_valid and self._factor_binding == self._factor_storage_binding()

    def _factor_storage_binding(self):
        arrays = (self.scale, self._q_cholesky, self._x_cholesky, self._rigid_cholesky, self.rigid_rows)
        if self._cable_preconditioner is not None:
            arrays += self._cable_preconditioner.buffers()
        if self._cloth_preconditioner is not None:
            arrays += self._cloth_preconditioner.buffers()
        return self._generation, self.layout, tuple((a.ptr, a.shape, a.strides, a.dtype, a.device) for a in arrays)

    def _vector_valid(self, vector) -> bool:
        return (
            isinstance(vector, wp.array)
            and vector.dtype == self.dtype
            and vector.device == self.device
            and vector.shape == (self.layout.scalar_dof_count,)
            and vector.is_contiguous
        )

    def _require_precision(self, *vectors) -> None:
        """Reject vectors stored in another precision before any kernel is specialized for them."""
        if any(vector.dtype != self.dtype for vector in vectors):
            raise ValueError(f"Linear operator vectors must be {self.dtype.__name__}")

    def _read_status(self) -> MonolithicLinearStatus:
        return MonolithicLinearStatus(int(self._status.numpy()[0]))

    def build_scaling(
        self, dynamic_diagonal: wp.array, *, epsilon_d: float, generation: MonolithicLinearGeneration
    ) -> MonolithicLinearStatus:
        return bsr.build_scaling(self, dynamic_diagonal, epsilon_d=epsilon_d, generation=generation)

    def _build_scaling_device(self, dynamic_diagonal, epsilon_d):
        return bsr._build_scaling_device(self, dynamic_diagonal, epsilon_d)

    def build_mixed_scaling(
        self,
        dynamic_diagonal: wp.array,
        coordinate_scale: wp.array,
        coordinate_mask: wp.array[wp.int32],
        *,
        epsilon_d: float,
        generation: MonolithicLinearGeneration,
    ) -> MonolithicLinearStatus:
        """Scale diagonal-policy rows from D and masked rows by fixed coordinate units."""
        return bsr.build_mixed_scaling(
            self, dynamic_diagonal, coordinate_scale, coordinate_mask, epsilon_d=epsilon_d, generation=generation
        )

    def build_coordinate_scaling(self, scale, *, generation):
        """Bind an explicit positive change of units for massless material angles."""
        if not self._current(generation):
            return MonolithicLinearStatus.STALE_GENERATION
        self._scaling_valid = False
        self._factor_valid = False
        self._warm_start_key = None
        values = np.asarray(scale, dtype=self._native)
        if values.shape != self.scale.shape or not np.isfinite(values).all() or np.any(values <= 0):
            return MonolithicLinearStatus.INVALID_ARGUMENT
        diagonal = 1.0 / values**2
        if not np.isfinite(diagonal).all() or np.any(diagonal <= 0):
            return MonolithicLinearStatus.INVALID_ARGUMENT
        self.scale.assign(values)
        self.diagonal.assign(diagonal)
        self.dynamic_diagonal.assign(diagonal)
        self._scaling_valid = True
        return MonolithicLinearStatus.SUCCESS

    def set_regularization(
        self, lambda_value: float, *, generation: MonolithicLinearGeneration
    ) -> MonolithicLinearStatus:
        if not self._current(generation) or not self._scaling_valid:
            return MonolithicLinearStatus.STALE_GENERATION
        if not np.isfinite(lambda_value) or lambda_value < 0 or lambda_value > np.finfo(self._native).max:
            return MonolithicLinearStatus.INVALID_ARGUMENT
        if lambda_value != self._lambda:
            self._factor_valid = False
        self._lambda = float(lambda_value)
        return MonolithicLinearStatus.SUCCESS

    def factor_actor_preconditioner(
        self, *, generation: MonolithicLinearGeneration, pivot_tolerance: float
    ) -> MonolithicLinearStatus:
        return preconditioner.factor_actor_preconditioner(self, generation=generation, pivot_tolerance=pivot_tolerance)

    def _factor_actor_device(self, pivot_tolerance):
        return preconditioner._factor_actor_device(self, pivot_tolerance)

    @property
    def pcg_mode(self) -> str:
        """Return the immutable execution mode selected at construction."""
        return self._pcg_mode

    def solve_pcg(
        self,
        rhs_hat: wp.array,
        y: wp.array,
        *,
        generation: MonolithicLinearGeneration,
        warm_start: MonolithicPcgWarmStart,
        config: MonolithicPcgConfig,
        delta: wp.array | None = None,
    ) -> MonolithicLinearSolveResult:
        """Solve Khat*y=rhs_hat using the current operator and actor preconditioner.

        SAME_GENERATION reuses only the last successful solution array for this
        unchanged K/D/S. Lambda may change, but requires refactorization first.
        Every solve recomputes its initial true residual, including warm starts.
        Both PCG modes run the same masked kernels; the diagnostic mode only
        records every checkpoint's true-residual group norms.
        """
        return solve_pcg_production(
            self,
            rhs_hat,
            y,
            generation=generation,
            warm_start=warm_start,
            config=config,
            delta=delta,
            use_graph=self.use_cuda_graph,
            observe=self.pcg_mode == "diagnostic",
        )

    def densify_for_test(self, *, generation: MonolithicLinearGeneration) -> MonolithicDenseLinearOracle:
        """Copy the finalized production BSR to a host dense debug oracle."""
        if generation != self._generation or not self._sealed or not self._valid:
            raise ValueError("Dense oracle requires the current valid sealed generation")
        matrix = self.k_global_scalar_bsr
        offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
        dense = np.zeros((matrix.nrow, matrix.ncol), dtype=np.float64)
        for row in range(matrix.nrow):
            start, end = offsets[row], offsets[row + 1]
            dense[row, columns[start:end]] = values[start:end]
        if not self._scaling_valid:
            return MonolithicDenseLinearOracle(dense)
        diagonal, scale = self.diagonal.numpy().astype(np.float64), self.scale.numpy().astype(np.float64)
        regularized = dense + self._lambda * np.diag(diagonal)
        # Match the production S*K*S + lambda*I operator, avoiding a second assembly.
        scaled = scale[:, None] * dense * scale[None, :] + self._lambda * np.eye(matrix.nrow)
        return MonolithicDenseLinearOracle(dense, self.dynamic_diagonal.numpy(), scale, regularized, scaled)
