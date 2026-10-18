# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate spatial contact experiment parameters and select the pooled audit route.

The solver evaluates contact through the spatial dataflow only. Offline audits may
still construct the complete-candidate pooled objective as an independent check of
spatial candidate selection; it is never a production route.
"""

from contextlib import contextmanager
from unittest.mock import patch

SOLVER_CONTACT = "newton._src.solvers.monolithic.solver_monolithic.ContactObjective"
MATERIAL_CONTACT = "newton._src.solvers.monolithic.material_runtime.ContactObjective"


def spatial_parameters(spatial):
    """Return validated ``{"candidate_capacity": n}`` from new or recorded parameters.

    Recordings made before P8a also carry ``law_margin``; it must be zero.
    """
    if type(spatial) is not dict or not {"candidate_capacity"} <= set(spatial) <= {"candidate_capacity", "law_margin"}:
        raise ValueError("Invalid spatial contact parameters")
    if spatial.get("law_margin", 0) != 0:
        raise ValueError("The spatial law margin is fixed at zero")
    capacity = spatial["candidate_capacity"]
    if type(capacity) is not int or not 0 <= capacity < 2**31:
        raise ValueError("Invalid spatial candidate capacity")
    return {"candidate_capacity": capacity}


@contextmanager
def pooled_audit_route():
    """Construct solvers with the complete-candidate pooled objective inside this block."""
    from newton._src.solvers.monolithic.contact_pipeline import _PooledContactObjective  # noqa: PLC0415

    with patch(SOLVER_CONTACT, _PooledContactObjective), patch(MATERIAL_CONTACT, _PooledContactObjective):
        yield
