.. SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
.. SPDX-License-Identifier: CC-BY-4.0

newton.solvers.experimental.monolithic
======================================

Experimental monolithic solver development scaffold.

.. experimental::

This scaffold provides internal layout, state-transaction, and sparse-assembly
contracts. Physical stepping is not implemented; constructing the solver raises
``NotImplementedError``. The collision pipeline will be added separately.

.. py:module:: newton.solvers.experimental.monolithic
.. currentmodule:: newton.solvers.experimental.monolithic

.. rubric:: Classes

.. autoclass:: SolverMonolithic

