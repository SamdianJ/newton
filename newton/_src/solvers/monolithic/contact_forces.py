# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent snapshots of published monolithic contact forces."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MonolithicContactForces:
    """Read-only native-precision force snapshots, independent of subsequent solver operations.

    Attributes:
        particle_force: World contact forces in N, shape ``(particle_count, 3)``.
        body_wrench: World force (N), then torque about COM (N m), shape ``(body_count, 6)``.
        joint_force: Generalized contact forces, shape ``(joint_dof_count,)``, in N or N m.
        force_valid: Whether the state has published forces; does not certify equilibrium.
        force_generation: Publication generation associated with these forces.
    """

    particle_force: np.ndarray
    body_wrench: np.ndarray
    joint_force: np.ndarray
    force_valid: bool
    force_generation: int

    def __post_init__(self):
        for name in ("particle_force", "body_wrench", "joint_force"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
