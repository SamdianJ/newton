# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed-order segmented summaries of nonlinear rows in native precision."""

from typing import Any

import numpy as np
import warp as wp

wp.set_module_options({"fuse_fp": False})


@wp.kernel
def _chunks(
    terms: wp.array2d[Any],
    rows: wp.array[int],
    starts: wp.array[int],
    ends: wp.array[int],
    minimum: int,
    partials: wp.array2d[Any],
):
    chunk, metric = wp.tid()
    value = terms.dtype(0)
    if metric == minimum:
        value = terms.dtype(wp.inf)
    for j in range(starts[chunk], ends[chunk]):
        term = terms[metric, rows[j]]
        if metric == minimum:
            # Warp min may select the finite operand; a bad row must stay bad.
            if wp.isnan(value) or wp.isnan(term):
                value = terms.dtype(wp.nan)
            else:
                value = wp.min(value, term)
        else:
            value += term
    partials[chunk, metric] = value


@wp.kernel
def _groups(partials: wp.array2d[Any], offsets: wp.array[int], minimum: int, result: wp.array2d[Any]):
    group, metric = wp.tid()
    value = result.dtype(0)
    if metric == minimum:
        value = result.dtype(wp.inf)
    for j in range(offsets[group], offsets[group + 1]):
        term = partials[j, metric]
        if metric == minimum:
            if wp.isnan(value) or wp.isnan(term):
                value = result.dtype(wp.nan)
            else:
                value = wp.min(value, term)
        else:
            value += term
    result[group, metric] = value


class SegmentedReduction:
    """Reduce ordered row lists without allocating a group-by-row matrix.

    Linear groups contain a global row list and a disjoint partition; physical
    groups may use noncontiguous lists. Storage follows the number of supplied
    row indices, chunks and groups. Empty groups have sum zero and minimum inf.
    """

    CHUNK_SIZE = 128

    def __init__(self, size, groups, metrics, *, dtype, device):
        self.size, self.metrics = size, metrics
        rows, starts, ends, offsets = [], [], [], [0]
        for indices in groups:
            group = np.asarray(indices)
            if group.ndim != 1 or (
                group.size and (group.dtype.kind not in "iu" or np.any(group < 0) or np.any(group >= size))
            ):
                raise ValueError("Reduction group rows lie outside the layout")
            begin = len(rows)
            rows.extend(group.tolist())
            for start in range(begin, len(rows), self.CHUNK_SIZE):
                starts.append(start)
                ends.append(min(start + self.CHUNK_SIZE, len(rows)))
            offsets.append(len(starts))
        self.rows, self.starts, self.ends, self.offsets = (
            wp.array(np.asarray(values, dtype=np.int32), dtype=int, device=device)
            for values in (rows, starts, ends, offsets)
        )
        self.partials = wp.zeros((len(starts), metrics), dtype=dtype, device=device)
        self.result = wp.zeros((len(offsets) - 1, metrics), dtype=dtype, device=device)

    def buffers(self):
        return self.rows, self.starts, self.ends, self.offsets, self.partials, self.result

    def reduce(self, terms, *, minimum=-1):
        """Enqueue sums, optionally taking a minimum for one metric column."""
        if (
            terms.shape != (self.metrics, self.size)
            or terms.dtype != self.result.dtype
            or terms.device != self.result.device
        ):
            raise ValueError("Reduction terms do not match the bound layout")
        if not -1 <= minimum < self.metrics:
            raise ValueError("Invalid minimum metric")
        device = self.result.device
        wp.launch(
            _chunks,
            self.partials.shape,
            [terms, self.rows, self.starts, self.ends, minimum, self.partials],
            device=device,
        )
        wp.launch(_groups, self.result.shape, [self.partials, self.offsets, minimum, self.result], device=device)
        return self.result
