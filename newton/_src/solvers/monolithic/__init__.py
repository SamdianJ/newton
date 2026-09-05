# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental monolithic solver components and fixed P1Q3 collision.

.. experimental::

The collision pipeline can detect tet-face contacts independently. The solver
validates its model and allocates owned buffers at construction; physical
stepping and final contact force publication remain unimplemented.
"""

from .collision import MonolithicCollisionPipeline
from .solver_monolithic import SolverMonolithic

__all__ = ["MonolithicCollisionPipeline", "SolverMonolithic"]
