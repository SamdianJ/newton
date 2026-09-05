.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.experimental.monolithic
======================================

Experimental implicit articulated rigid/tet solver with P1Q3 contact.

.. experimental::

The solver advances supported single-articulation/tet models with normal-only
contact, scaled PCG and transactional state/force publication. Numerical defaults
have tiny-fixture validation; broader asset and trajectory calibration is pending.
The collision pipeline can also detect tet-face contacts independently.

.. py:module:: newton.solvers.experimental.monolithic
.. currentmodule:: newton.solvers.experimental.monolithic

.. rubric:: Classes

.. autoclass:: MonolithicCollisionPipeline

.. autoclass:: SolverMonolithic

