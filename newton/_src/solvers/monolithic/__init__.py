# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit articulated rigid/tet solver with P1Q3 contact.

.. experimental::

The solver advances supported single-articulation/tet models with three explicit
contact laws, scaled PCG and transactional state/force publication.
The collision pipeline can also detect tet-face contacts independently.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collision import MonolithicCollisionPipeline
    from .contact_forces import MonolithicContactForces
    from .smooth_sdf import SmoothSDFSet
    from .smooth_sdf_builder import build_smooth_sdf
    from .solver_monolithic import SolverMonolithic

__all__ = [
    "MonolithicCollisionPipeline",
    "MonolithicContactForces",
    "SmoothSDFSet",
    "SolverMonolithic",
    "build_smooth_sdf",
]

_LAZY_IMPORTS = {
    "MonolithicCollisionPipeline": ".collision",
    "MonolithicContactForces": ".contact_forces",
    "SolverMonolithic": ".solver_monolithic",
    "SmoothSDFSet": ".smooth_sdf",
    "build_smooth_sdf": ".smooth_sdf_builder",
}


def __getattr__(name: str):
    try:
        module_name = _LAZY_IMPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_IMPORTS))
