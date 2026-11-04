# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bind material kernel views without moving their owners' numerical arrays."""

import numpy as np
import warp as wp


class _DeviceRecords:
    """Keep immutable device descriptors and retain their referenced storage.

    A replacement allocation gets new descriptors, so a retained CUDA graph
    continues to own its old binding. Numerical updates need no host transfer.
    """

    def __init__(self, dtype, values, counts, *, device):
        self.dtype, self.device = dtype, wp.get_device(device)
        self._signature = self._layout = None
        jobs = [(actor, row) for actor, count in enumerate(counts) for row in range(count)]
        self.jobs = wp.array(np.asarray(jobs, dtype=np.int32).reshape(-1, 2), dtype=wp.vec2i, device=self.device)
        self.bind(values)

    def bind(self, values):
        values = tuple(tuple(row) for row in values)
        arrays = tuple(value for row in values for value in row if isinstance(value, wp.array))
        signature = tuple(
            tuple(
                (value.ptr, value.shape, value.strides, value.dtype, value.device)
                if isinstance(value, wp.array)
                else value
                for value in row
            )
            for row in values
        )
        if signature != self._signature:
            layout = tuple(
                tuple(
                    (value.shape, value.dtype, value.device) if isinstance(value, wp.array) else value for value in row
                )
                for row in values
            )
            if self._layout is not None and layout != self._layout:
                raise ValueError("Material batch storage changed its layout, dtype or device")
            if self.device.is_capturing:
                raise RuntimeError("Bind material batch storage before graph capture")
            records = []
            for row in values:
                record = self.dtype()
                for name, value in zip(self.dtype.vars, row, strict=True):
                    setattr(record, name, value)
                records.append(record)
            self.array = wp.array(records, dtype=self.dtype, device=self.device)
            self.references = arrays
            self._signature, self._layout = signature, layout
        return self.array

    def buffers(self):
        return (self.array, self.jobs, *self.references)
