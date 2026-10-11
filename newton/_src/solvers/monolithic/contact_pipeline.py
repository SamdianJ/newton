# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete-candidate pooled contact adapter for coupled assembly and publication."""

import warnings
from typing import Any

import numpy as np
import warp as wp

from . import state_checks
from .bsr import _MonolithicScalarTriplets
from .contact import ContactObjective
from .contact_batch import _ContactBatch
from .contact_pool import _CONTACT_WORKERS
from .contact_spatial import _ContactSpatial
from .segment_graph import _SegmentGraphCache


@wp.kernel
def _reserve_pool_triplets(
    records: Any, flags: wp.array[int], triplets: _MonolithicScalarTriplets, base: wp.array[int]
):
    base[0] = -1
    if triplets.active_token[0] != triplets.token:
        triplets.status[0] = 2
    elif records.flags[0] != 0 or flags[0] != 0:
        triplets.status[0] = 7
    elif records.triplet_count[0] < 0 or records.triplet_count[0] > records.triplet_values.shape[0]:
        triplets.status[0] = 1
    elif triplets.count[0] < 0 or triplets.count[0] > triplets.capacity - records.triplet_count[0]:
        triplets.status[0] = 4
    elif triplets.status[0] == 0:
        base[0] = triplets.count[0]
        triplets.count[0] += records.triplet_count[0]


@wp.kernel
def _copy_pool_triplets(records: Any, triplets: _MonolithicScalarTriplets, base: wp.array[int], workers: int):
    i = wp.tid()
    if base[0] < 0 or triplets.status[0] != 0:
        return
    while i < records.triplet_count[0]:
        triplets.rows[base[0] + i] = records.triplet_rows[i]
        triplets.columns[base[0] + i] = records.triplet_columns[i]
        triplets.values[base[0] + i] = records.triplet_values[i]
        if workers >= records.triplet_count[0] - i:
            break
        i += workers


@wp.kernel
def _pool_counts(records: Any, packet: wp.array[int]):
    packet[0] = records.count[0]
    packet[1] = records.used[0]
    packet[2] = records.requested[0]
    packet[3] = records.dropped[0]
    packet[4] = records.support_count[0]
    packet[5] = records.factor_count[0]
    packet[6] = records.triplet_count[0]


@wp.kernel
def _publish_pool_joint_force(rows: wp.array[int], residual: wp.array[Any], result: wp.array[Any]):
    i = wp.tid()
    result[i] = type(result[0])(0)
    if rows[i] >= 0:
        result[i] = -residual[rows[i]]


class _PooledContactObjective(ContactObjective):
    """Connect bounded native generation to the actual coupled solver, without spatial clipping.

    Construction uses only validated compact metadata. The solver initializes
    a full-support resource bound before allocating its linear workspace.
    Appending COO is deliberately outside overwrite-only graph capture.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, _metadata_only=True, **kwargs)
        self._evaluated = False
        self.evaluation_generation = 0
        self.candidate_generation = 0
        self._assembled_generation = -1
        self._last_assembly = None
        self._tainted = False
        self._warned = False

    def initialize_pool(self, factor_capacity, *, use_cuda_graph):
        actors = self.actor_samples.actors
        # CUDA geometry needs larger batches than a single row-reduction block.
        batch_capacity = 8192 if self.model.device.is_cuda else 256
        self._batch = _ContactBatch(
            self,
            batch_capacity=min(self.count, batch_capacity),
            capacity=None,
            contact_factor_capacity=factor_capacity,
            use_cuda_graph=use_cuda_graph,
        )
        self.pool = self._batch.pool
        self.triplet_capacity = self.pool.triplet_capacity
        self._append_base = wp.zeros(1, dtype=int, device=self.model.device)
        self._packet = wp.zeros(7, dtype=int, device=self.model.device)
        self._model_rows = wp.array(actors.maps["joint_dof_to_global"], dtype=int, device=self.model.device)
        self.dataflow_identity = np.frombuffer(b"complete-pool/fixed-tree/v1", dtype=np.uint8)
        self.resource_identity = np.asarray(
            [self.pool.capacity, self.pool.max_support, self._batch.batch_capacity, factor_capacity],
            dtype=np.int64,
        )

    @property
    def flags(self):
        return self._batch.flags

    @property
    def start_flags(self):
        return self._batch.start_flags

    @property
    def residual(self):
        return self.pool.residual.result

    @property
    def energy(self):
        return self.pool.energy.result[:2]

    @property
    def absolute_energy(self):
        return self.pool.energy.result[2:]

    def buffers(self):
        return (*self._batch.buffers(), self._append_base, self._packet, self._model_rows)

    def begin_step(self, *state, **options):
        self._evaluated = False
        self._assembled_generation = -1
        self._tainted = self._warned = False
        self.candidate_generation += 1
        self._batch.begin_step(*state, **options)

    def evaluate(self, *state, **options):
        self._evaluated = False
        self._assembled_generation = -1
        self._batch.evaluate(*state, **options)
        self.evaluation_generation += 1
        self._evaluated = True

    def evaluation_diagnostics(self):
        wp.launch(_pool_counts, 1, [self.pool.records, self._packet], device=self.model.device)
        counts = self._packet.numpy()
        self._tainted |= bool(counts[3])
        if counts[3] and not self._warned:
            warnings.warn(
                f"Contact capacity exceeded: discarded {int(counts[3])} complete contacts", RuntimeWarning, stacklevel=2
            )
            self._warned = True
        return dict(
            zip(
                ("admitted", "used", "requested", "dropped", "support_count", "factor_count", "triplet_count"),
                map(int, counts),
                strict=True,
            ),
            candidate_generation=self.candidate_generation,
            evaluation_generation=self.evaluation_generation,
            admission_generation=self.evaluation_generation,
            tainted=self._tainted,
        )

    def assemble_tangent(self, assembly):
        if not self._evaluated or assembly.generation.contact_generation != self.evaluation_generation:
            raise ValueError("Contact assembly requires the current evaluation generation")
        if self._last_assembly == assembly.generation:
            raise ValueError("Contact COO may be appended only once per assembly")
        triplets = assembly.global_scalar_triplets
        if triplets.row_count != self.layout.scalar_dof_count or triplets.values.device != self.model.device:
            raise ValueError("Contact assembly requires the matching linear layout")
        # Exactly one reservation and append per assembly invocation. Capture
        # warmup/replay may only derive from the complete immutable pool.
        wp.launch(
            _reserve_pool_triplets,
            1,
            [self.pool.records, self.flags, triplets, self._append_base],
            device=self.model.device,
        )
        if self.triplet_capacity:
            workers = min(self.triplet_capacity, _CONTACT_WORKERS)
            wp.launch(
                _copy_pool_triplets,
                workers,
                [self.pool.records, triplets, self._append_base, workers],
                device=self.model.device,
            )
        self._assembled_generation = self.evaluation_generation
        self._last_assembly = assembly.generation

    def publish(self, state):
        if not self._evaluated or self._assembled_generation != self.evaluation_generation:
            raise ValueError("Contact forces require the finally assembled evaluation")
        history = state._monolithic_coupled
        n = 3 * self.model.particle_count
        wp.copy(state_checks.scalar_view(history.contact_particle_force), self.pool.force.result[:n])
        wp.copy(state_checks.scalar_view(history.contact_body_force), self.pool.force.result[n:])
        if self._model_rows.size:
            wp.launch(
                _publish_pool_joint_force,
                self._model_rows.size,
                [self._model_rows, self.residual, history.contact_joint_force],
                device=self.model.device,
            )
        return self.pair_diagnostics()

    def pair_diagnostics(self):
        records = self.pool.records
        count = int(records.count.numpy()[0])
        identities = records.identities.numpy()[:count]
        values = records.pairs.numpy()[:count]
        samples = self._batch.all_samples
        sources, targets = samples.sources.numpy(), samples.targets.numpy()
        grouped = {pair: [] for pair in self.actor_samples.pair_kinds}
        for i in np.argsort(identities, kind="stable"):
            key = int(identities[i])
            if key < self.legacy_count:
                continue
            source, target = int(sources[key]), int(targets[key])
            label = (source, target) if source < 0 else tuple(sorted((source, target)))
            grouped.setdefault(label, []).append(values[i, ::-1] if source >= 0 and source != label[0] else values[i])
        result = []
        for (source, target), entries in grouped.items():
            tree = np.zeros((1 << max(0, len(entries) - 1).bit_length(), 2, 6), dtype=self.numpy_dtype)
            if entries:
                tree[: len(entries)] = entries
            step = 1
            while step < len(tree):
                tree[:: 2 * step] += tree[step :: 2 * step]
                step *= 2
            result.append(
                {
                    "source": source,
                    "target": target,
                    "kind": self.actor_samples.pair_kinds[source, target],
                    "world_wrench": tree[0].tolist(),
                    "bounds": {},
                }
            )
        return result


class _SpatialContactObjective(_PooledContactObjective):
    """Private B3 experiment: refresh at Newton boundaries, freeze all trials and final checks."""

    def __init__(self, *args, candidate_capacity, law_margin, **kwargs):
        if type(candidate_capacity) is not int or not 0 <= candidate_capacity < 2**31:
            raise ValueError("Invalid spatial candidate capacity")
        if isinstance(law_margin, (bool, np.bool_)) or not np.isfinite(law_margin) or law_margin < 0:
            raise ValueError("Invalid experimental law margin")
        self.candidate_capacity, self.law_margin = candidate_capacity, float(law_margin)
        self._spatial = None
        self._selection_ready = self._reference_mode = False
        super().__init__(*args, **kwargs)

    def initialize_pool(self, factor_capacity, *, use_cuda_graph):
        super().initialize_pool(factor_capacity, use_cuda_graph=use_cuda_graph)
        self._batch.initialize_reference()
        # Cold reference snapshots must not serialize uninitialized reduction
        # scratch as nonfinite live-in data. Later evaluations overwrite it.
        for owner in (
            self.pool._identity,
            self.pool.energy,
            self.pool.residual,
            self.pool.force,
            self._batch.reference_rows,
        ):
            for array in owner.buffers():
                array.zero_()
        self.query_graphs = _SegmentGraphCache(self.model.device, enabled=use_cuda_graph and self.model.device.is_cuda)
        self.dataflow_identity = np.frombuffer(b"spatial-pool/newton-frozen/v1", dtype=np.uint8)
        self.resource_identity = np.append(self.resource_identity, self.candidate_capacity)
        self.spatial_identity = np.array([self.law_margin], dtype=np.float64)

    @property
    def residual(self):
        return self._batch.reference_residual if self._reference_mode else super().residual

    @property
    def energy(self):
        return self._batch.reference_energy if self._reference_mode else super().energy

    @property
    def absolute_energy(self):
        return self._batch.reference_energy if self._reference_mode else super().absolute_energy

    def buffers(self):
        return super().buffers() + (() if self._spatial is None else self._spatial.buffers())

    def begin_step(self, *state, kinematics, q0, **options):
        self._selection_ready = False
        self._refreshes = self._query_dropped = 0
        self._query_warned = False
        if self._spatial is None:
            self._spatial = _ContactSpatial(
                self._batch,
                kinematics,
                q0,
                state[0],
                state[1],
                rigid_com=options.get("rigid_com"),
                capacity=self.candidate_capacity,
                law_margin=self.law_margin,
            )
        else:
            if kinematics is not self._spatial.kinematics:
                raise ValueError("Spatial contact kinematics changed")
            self._spatial.begin_step(q0, state[0], state[1], rigid_com=options.get("rigid_com"))
        self._evaluated = False
        self._assembled_generation = -1
        self._tainted = self._warned = False
        self._batch.begin_step(*state, **options)

    def refresh(self, q, x, *, rigid_com):
        selection = self._spatial
        if selection is None:
            raise ValueError("Freeze a physical step before querying candidates")
        selection.validate_storage()
        self.query_graphs.run(
            "contact_candidates",
            (),
            (q, x, rigid_com, *selection.buffers()),
            lambda: selection.update(q, x, rigid_com=rigid_com),
        )
        dropped = selection.check(warn=False)
        self._query_dropped += dropped
        if dropped and not self._query_warned:
            warnings.warn(
                f"Spatial capacity exceeded: discarded {dropped} complete candidates", RuntimeWarning, stacklevel=2
            )
            self._query_warned = True
        self.candidate_generation += 1
        self._refreshes += 1
        self._selection_ready = True
        self._evaluated = False
        self._assembled_generation = -1

    def evaluate(self, *state, **options):
        if self._reference_mode:
            self._batch.evaluate_reference(*state, **options)
            return
        if not self._selection_ready:
            raise ValueError("Refresh Newton candidates before spatial evaluation")
        # Out-of-envelope trials can miss contacts. Do not requery, reject or
        # repair them here; a later Newton refresh cannot guarantee recovery.
        super().evaluate(*state, selection=self._spatial, **options)

    def spatial_diagnostics(self):
        return {
            "candidate_generation": self.candidate_generation,
            "refreshes": self._refreshes,
            "candidates": int(self._spatial.count.numpy()[0]),
            "capacity_dropped": self._query_dropped,
            "tainted": bool(self._query_dropped),
            "scope": "clipped-system",
            "graphs": self.query_graphs.summary(),
            "reference_graphs": self._batch.reference_rows.graphs.summary(),
        }

    def evaluation_diagnostics(self):
        result = super().evaluation_diagnostics()
        result["tainted"] |= bool(self._query_dropped)
        return result
