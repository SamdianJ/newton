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


class _Producer:
    """Name, owned rows and declared policies of one physical term set."""

    capturable = True
    scale_policy = "diagonal"

    def __init__(self, name, rows):
        self.name, self.rows = name, np.asarray(rows, dtype=np.int32)


class ArticulationProducer(_Producer):
    """Articulation coordinates with joint, PD, limit and friction terms."""

    def __init__(self, solver):
        super().__init__("articulation", solver._layout.indices("articulation"))


class RigidProducer(_Producer):
    """Every FREE rigid actor: three COM and three rotation rows each."""

    def __init__(self, solver):
        super().__init__("rigid", solver._rigid.rows_numpy.ravel())


class TetProducer(_Producer):
    """Dynamic tet particles with inertia, elasticity and frozen external forces."""

    def __init__(self, solver):
        super().__init__("tet", solver._layout.indices("tet"))


class ClothProducer(_Producer):
    """One cloth object; its host-side evaluation is not graph-capturable yet."""

    capturable = False
    scale_policy = "coordinate"

    def __init__(self, solver, block):
        source = block.actor.source_index
        super().__init__(f"cloth_{source}", solver._layout.indices("cloth", source))
        self.block = block


class CableProducer(_Producer):
    """One cable object; theta rows use a fixed inverse-length scale."""

    capturable = False
    scale_policy = "coordinate"

    def __init__(self, solver, block):
        source = block.actor.source_index
        super().__init__(f"cable_{source}", solver._layout.indices("cable", source))
        self.block = block


class ContactProducer(_Producer):
    """Spatial contact; it adds coupling terms to rows owned by other producers."""

    scale_policy = "none"

    def __init__(self, solver):
        super().__init__("contact", np.empty(0, dtype=np.int32))


def build_producers(solver):
    """Order producers as articulation, rigid, tet, cloth, cable and contact."""
    layout = solver._layout
    producers = []
    if layout.q_dof_count:
        producers.append(ArticulationProducer(solver))
    if solver._rigid.count:
        producers.append(RigidProducer(solver))
    if layout.dynamic_particle_count and getattr(solver, "_tet", None) is not None:
        producers.append(TetProducer(solver))
    producers.extend(ClothProducer(solver, block) for block in getattr(solver, "_cloths", ()))
    producers.extend(CableProducer(solver, block) for block in getattr(solver, "_cables", ()))
    if solver._contact.count:
        producers.append(ContactProducer(solver))
    validate_row_partition([p for p in producers if len(p.rows)], layout.scalar_dof_count)
    return producers
