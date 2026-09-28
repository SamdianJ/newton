# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit articulated rigid/tet solver with P1Q3 contact.

.. experimental::

The solver advances supported single-articulation/tet models with normal-only
contact, scaled PCG and transactional state/force publication. Numerical defaults
have tiny-fixture validation; broader asset and trajectory calibration is pending.
The collision pipeline can also detect tet-face contacts independently.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .collision import MonolithicCollisionPipeline
    from .solver_monolithic import SolverMonolithic

__all__ = ["MonolithicCollisionPipeline", "SolverMonolithic"]

_LAZY_IMPORTS = {
    "MonolithicCollisionPipeline": ".collision",
    "SolverMonolithic": ".solver_monolithic",
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
