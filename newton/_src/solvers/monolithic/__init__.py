# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental implicit articulated rigid/tet solver with P1Q3 contact.

.. experimental::

The solver advances supported single-articulation/tet models with normal-only
contact, scaled PCG and transactional state/force publication. Numerical defaults
have tiny-fixture validation; broader asset and trajectory calibration is pending.
The collision pipeline can also detect tet-face contacts independently.
"""

from .collision import MonolithicCollisionPipeline
from .solver_monolithic import SolverMonolithic

__all__ = ["MonolithicCollisionPipeline", "SolverMonolithic"]
