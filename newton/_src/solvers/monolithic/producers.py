# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step contract shared by every monolithic physical producer."""

from typing import Any, Protocol

import numpy as np


class Producer(Protocol):
    """One physical term set owning a disjoint set of scalar rows.

    ``capturable`` producers perform no host reads in ``evaluate``/``assemble``.
    ``scale_policy`` is ``"diagonal"`` (dynamic diagonal) or ``"coordinate"``
    (fixed per-row unit scale). Producers without rows, such as contact, add
    coupling terms to rows owned by others.
    """

    name: str
    rows: np.ndarray
    capturable: bool
    scale_policy: str

    def begin_step(self, solver: Any, state_in: Any) -> None: ...

    def candidate(self, solver: Any, accepted: Any, trial: Any, direction: Any, alpha: float, dt: float) -> None: ...

    def evaluate(
        self, solver: Any, state: Any, dt: float, residual: Any, *, build_tangent: bool
    ) -> tuple[Any, Any]: ...

    def assemble(self, solver: Any, assembly: Any, dt: float) -> None: ...

    def scale(self, solver: Any, diagonal: Any, coordinate: np.ndarray) -> None: ...

    def displacement(self, solver: Any, accepted: Any, trial: Any, out: np.ndarray) -> None: ...

    def domain(self, solver: Any, accepted: Any, candidate: Any) -> bool: ...

    def publish(self, solver: Any, state: Any) -> None: ...


def validate_row_partition(producers, scalar_dof_count):
    """Require producer rows to cover every scalar DoF exactly once."""
    owner = np.full(scalar_dof_count, -1, dtype=np.int64)
    for index, producer in enumerate(producers):
        rows = np.asarray(producer.rows, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= scalar_dof_count):
            raise ValueError(f"Producer rows lie outside the layout: {producer.name}")
        if len(np.unique(rows)) != len(rows) or np.any(owner[rows] >= 0):
            raise ValueError(f"Producer rows overlap: {producer.name}")
        owner[rows] = index
    if np.any(owner < 0):
        raise ValueError("Producer rows leave scalar DoFs uncovered")
