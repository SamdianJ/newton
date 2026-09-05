.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.experimental.monolithic
======================================

Experimental monolithic solver components and fixed P1Q3 collision.

.. experimental::

The collision pipeline can detect tet-face contacts independently. The solver
validates its model and allocates owned buffers at construction; physical
stepping and final contact force publication remain unimplemented.

.. py:module:: newton.solvers.experimental.monolithic
.. currentmodule:: newton.solvers.experimental.monolithic

.. rubric:: Classes

.. autoclass:: MonolithicCollisionPipeline

.. autoclass:: SolverMonolithic

