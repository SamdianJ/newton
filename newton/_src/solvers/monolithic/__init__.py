# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental monolithic solver development scaffold.

.. experimental::

This scaffold provides internal layout, state-transaction, and sparse-assembly
contracts. Physical stepping is not implemented; constructing the solver raises
``NotImplementedError``. The collision pipeline will be added separately.
"""

from .solver_monolithic import SolverMonolithic

__all__ = ["SolverMonolithic"]
