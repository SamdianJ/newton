# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Apply cable stretch and bend-twist tangents element by element.

A slender cable's stiff stretch blocks dwarf its soft bending modes: assembled
FP32 entries round by about eps EA/L, which exceeds the whole bending stiffness
of the rod in a general orientation and leaves the stored matrix indefinite.
Each element matrix annihilates rigid translation, so applying it to the
element's displacements relative to its first vertex computes the same product
from operands the size of the relative motion instead of the motion itself.

The cable element tangents therefore never enter the global triplets. This
operator supplies them wherever the full operator is needed: the PCG product,
its magnitude, the direction posterior, the criterion's ``|B| |u|`` and the
cable preconditioner's band. Every output row is gathered in a fixed order;
no floating atomic runs.
"""

from functools import cache
from typing import Any

import numpy as np
import warp as wp

from .material_batch import _DeviceRecords

wp.set_module_options({"enable_backward": False})


@cache
def _record_types(dtype):
    @wp.struct
    class Elements:
        rows: wp.array2d[int]
        matrix: wp.array3d[dtype]
        out: wp.array2d[dtype]
        magnitude: wp.array2d[dtype]
        positions: int

    @wp.struct
    class Gather:
        targets: wp.array[int]
        offsets: wp.array[int]
        kinds: wp.array[int]
        elements: wp.array[int]
        slots: wp.array[int]
        stretch: wp.array2d[dtype]
        bend: wp.array2d[dtype]
        stretch_size: wp.array2d[dtype]
        bend_size: wp.array2d[dtype]

    return Elements, Gather


@wp.func
def _value(x: wp.array[Any], scale: wp.array[Any], row: int):
    if row < 0:
        return x.dtype(0.0)
    return scale[row] * x[row]


@wp.func
def _relative(x: wp.array[Any], scale: wp.array[Any], rows: wp.array2d[int], element: int, entry: int, positions: int):
    """Entry of the element's displacement, positions taken relative to its first vertex."""
    value = _value(x, scale, rows[element, entry])
    if entry < positions:
        value = value - _value(x, scale, rows[element, entry % 3])
    return value


@wp.kernel
def _element_apply(
    batches: wp.array[Any],
    jobs: wp.array[wp.vec2i],
    x: wp.array[Any],
    scale: wp.array[Any],
    active: wp.array[int],
):
    job = jobs[wp.tid()]
    batch, index = job[0], job[1]
    if active[0] == 0:
        return
    data = batches[batch]
    rows, matrix, out, magnitude = data.rows, data.matrix, data.out, data.magnitude
    width = rows.shape[1]
    element = index
    for a in range(width):
        value = out.dtype(0.0)
        size = out.dtype(0.0)
        for b in range(width):
            entry = matrix[element, a, b]
            value = value + entry * _relative(x, scale, rows, element, b, data.positions)
            # The product uses relative motion, but |B| |x| still bounds the
            # rounding of the stored vector, as for the assembled rows.
            size = size + wp.abs(entry * _value(x, scale, rows[element, b]))
        out[element, a] = value
        magnitude[element, a] = size


@wp.kernel
def _element_abs(
    batches: wp.array[Any],
    jobs: wp.array[wp.vec2i],
    coordinates: wp.array[Any],
):
    job = jobs[wp.tid()]
    batch, index = job[0], job[1]
    data = batches[batch]
    rows, matrix, out = data.rows, data.matrix, data.magnitude
    width = rows.shape[1]
    element = index
    for a in range(width):
        size = out.dtype(0.0)
        for b in range(width):
            row = rows[element, b]
            if row >= 0:
                size = size + wp.abs(matrix[element, a, b]) * coordinates[row]
        out[element, a] = size


@wp.kernel
def _gather(
    batches: wp.array[Any],
    jobs: wp.array[wp.vec2i],
    magnitude: bool,
    scale: wp.array[Any],
    active: wp.array[int],
    out: wp.array[Any],
    error: wp.array[int],
    code: int,
):
    job = jobs[wp.tid()]
    batch, index = job[0], job[1]
    if active[0] == 0:
        return
    data = batches[batch]
    i = index
    row = data.targets[i]
    total = out.dtype(0.0)
    for k in range(data.offsets[i], data.offsets[i + 1]):
        if magnitude:
            if data.kinds[k] == 0:
                total = total + data.stretch_size[data.elements[k], data.slots[k]]
            else:
                total = total + data.bend_size[data.elements[k], data.slots[k]]
        else:
            if data.kinds[k] == 0:
                total = total + data.stretch[data.elements[k], data.slots[k]]
            else:
                total = total + data.bend[data.elements[k], data.slots[k]]
    value = out[row] + scale[row] * total
    out[row] = value
    if not wp.isfinite(value):
        wp.atomic_max(error, 0, code)


@wp.kernel
def _band(
    batches: wp.array[Any],
    jobs: wp.array[wp.vec2i],
    scale: wp.array[Any],
    band: wp.array[Any],
):
    job = jobs[wp.tid()]
    batch, index = job[0], job[1]
    data = batches[batch]
    slot = index
    total = band.dtype(0.0)
    for k in range(data.offsets[slot], data.offsets[slot + 1]):
        if data.kinds[k] == 0:
            entry = data.stretch[data.elements[k], data.first[k], data.second[k]]
        else:
            entry = data.bend[data.elements[k], data.first[k], data.second[k]]
        total = total + entry * scale[data.row_a[k]] * scale[data.row_b[k]]
    band[data.targets[slot]] = total


class _Cable:
    """Element-to-row maps and output storage of one cable."""

    def __init__(self, block, dtype, device):
        objective = block.objective
        nv = block.ids.size
        local = block.rows_host
        ne, ni = nv - 1, nv - 2
        self.stretch_matrix, self.bend_matrix = objective.stretch_b, objective.bend_twist_b
        self.stretch_rows_host = np.array([local[3 * e : 3 * e + 6] for e in range(ne)], dtype=np.int32).reshape(ne, 6)
        self.bend_rows_host = np.array(
            [np.r_[local[3 * k : 3 * k + 9], local[3 * nv + k : 3 * nv + k + 2]] for k in range(ni)], dtype=np.int32
        ).reshape(ni, 11)
        self.stretch_rows = wp.array(self.stretch_rows_host, dtype=int, device=device)
        self.bend_rows = wp.array(self.bend_rows_host, dtype=int, device=device)
        self.stretch_out = wp.zeros((ne, 6), dtype=dtype, device=device)
        self.bend_out = wp.zeros((ni, 11), dtype=dtype, device=device)
        self.stretch_size = wp.zeros_like(self.stretch_out)
        self.bend_size = wp.zeros_like(self.bend_out)
        # Each free row gathers its edges, then its stencils, in element order.
        targets, offsets, kinds, elements, slots = [], [0], [], [], []
        for local_row, row in enumerate(local):
            if row < 0:
                continue
            entries = []
            if local_row < 3 * nv:
                vertex, axis = divmod(local_row, 3)
                entries += [(0, e, 3 * (vertex - e) + axis) for e in range(max(0, vertex - 1), min(ne, vertex + 1))]
                entries += [(1, k, 3 * (vertex - k) + axis) for k in range(max(0, vertex - 2), min(ni, vertex + 1))]
            else:
                edge = local_row - 3 * nv
                entries += [(1, k, 9 + edge - k) for k in range(max(0, edge - 1), min(ni, edge + 1))]
            targets.append(row)
            for kind, element, slot in entries:
                kinds.append(kind)
                elements.append(element)
                slots.append(slot)
            offsets.append(len(kinds))
        self.targets_host = np.asarray(targets, dtype=np.int32)
        self.gather = [
            wp.array(np.asarray(v, dtype=np.int32), dtype=int, device=device)
            for v in (targets, offsets, kinds, elements, slots)
        ]

    def entries(self):
        """Yield ``(kind, element, a, b, row_a, row_b)`` for every free element entry, in element order."""
        for kind, rows in ((0, self.stretch_rows_host), (1, self.bend_rows_host)):
            for element, element_rows in enumerate(rows):
                for a, row_a in enumerate(element_rows):
                    for b, row_b in enumerate(element_rows):
                        if row_a >= 0 and row_b >= 0:
                            yield kind, element, a, b, int(row_a), int(row_b)


class CableOperator:
    """Supply every cable's element tangents to the global linear operator."""

    def __init__(self, blocks, size, *, dtype, device):
        self.device, self.dtype, self.size = wp.get_device(device), dtype, int(size)
        self.cables = [_Cable(block, dtype, self.device) for block in blocks]
        self.ones = wp.ones(self.size, dtype=dtype, device=self.device)
        self.always = wp.ones(1, dtype=int, device=self.device)
        self.error = wp.zeros(1, dtype=int, device=self.device)
        self.rows = np.unique(np.concatenate([c.targets_host for c in self.cables])) if self.cables else np.empty(0)
        elements, gathers = self._views()
        element_type, gather_type = _record_types(dtype)
        self._elements = _DeviceRecords(
            element_type, elements, [row[0].shape[0] for row in elements], device=self.device
        )
        self._gathers = _DeviceRecords(gather_type, gathers, [row[0].size for row in gathers], device=self.device)

    def _views(self):
        elements, gathers = [], []
        for cable in self.cables:
            elements.extend(
                (
                    (cable.stretch_rows, cable.stretch_matrix, cable.stretch_out, cable.stretch_size, 6),
                    (cable.bend_rows, cable.bend_matrix, cable.bend_out, cable.bend_size, 9),
                )
            )
            gathers.append((*cable.gather, cable.stretch_out, cable.bend_out, cable.stretch_size, cable.bend_size))
        return elements, gathers

    def _bind(self):
        elements, gathers = self._views()
        self._elements.bind(elements)
        self._gathers.bind(gathers)

    def buffers(self):
        """Return every owned or referenced array for plan and graph binding."""
        self._bind()
        return (self.ones, self.always, self.error, *self._elements.buffers(), *self._gathers.buffers())

    def signature(self):
        return tuple((a.ptr, a.shape, a.strides, a.dtype, a.device) for a in self.buffers())

    def _launch_elements(self, x, scale, active):
        if self.cables:
            wp.launch(
                _element_apply,
                self._elements.jobs.size,
                [self._elements.array, self._elements.jobs, x, scale, active],
                device=self.device,
            )

    def _launch_gather(self, scale, active, out, *, magnitude, error, code):
        if self.cables:
            wp.launch(
                _gather,
                self._gathers.jobs.size,
                [self._gathers.array, self._gathers.jobs, magnitude, scale, active, out, error, code],
                device=self.device,
            )

    def add_product(self, x, scale, out, *, magnitude=None, active=None, error=None, code=0):
        """Add ``S B S x`` of the cable elements to ``out`` and optionally ``S |B| |S x|`` to ``magnitude``.

        ``scale`` may be :attr:`ones` for a product in physical units. Rows of
        no cable are untouched. ``active`` gates every launch from the device.
        """
        self._bind()
        active = self.always if active is None else active
        error = self.error if error is None else error
        self._launch_elements(x, scale, active)
        self._launch_gather(scale, active, out, magnitude=False, error=error, code=code)
        if magnitude is not None:
            self._launch_gather(scale, active, magnitude, magnitude=True, error=error, code=code)

    def add_abs_coordinates(self, coordinates, out):
        """Add ``|B| |u|`` of the cable elements, the assembled operator's term in the rounding magnitude."""
        self._bind()
        if self.cables:
            wp.launch(
                _element_abs,
                self._elements.jobs.size,
                [self._elements.array, self._elements.jobs, coordinates],
                device=self.device,
            )
        self._launch_gather(self.ones, self.always, out, magnitude=True, error=self.error, code=0)

    def dense(self):
        """Return the cable element part as a dense matrix, for tests and diagnostics."""
        result = np.zeros((self.size, self.size))
        for cable in self.cables:
            values = (cable.stretch_matrix.numpy(), cable.bend_matrix.numpy())
            for kind, element, a, b, row_a, row_b in cable.entries():
                result[row_a, row_b] += values[kind][element, a, b]
        return result
