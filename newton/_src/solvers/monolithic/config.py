# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validated SolverMonolithic options and their checkpoint identity."""

import math
from dataclasses import dataclass

import numpy as np

from .solver_types import MassMode, MaterialModel, PCGMode, Precision

# The single supported contact law and trial acceptance (P8 production combination).
CONTACT_MODEL = "C-IE"
GLOBALIZATION = "ArmijoOrResNorm"
# Version 4: per-row physical convergence replaces the scaled-norm goals and the
# cable, cloth and joint residual tolerance options. Version 5: linear solves stop
# at the physical limits, and the absolute linear tolerance option is removed.
CONFIG_VERSION = 5


def _number(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, float, np.integer, np.floating))


def _text(value):
    return np.frombuffer(value.encode(), dtype=np.uint8)


@dataclass(frozen=True)
class MonolithicConfig:
    """Every non-object SolverMonolithic option, validated without a Model.

    Model-dependent checks (CUDA availability, material scenes, actor
    ownership) remain in the solver constructor.
    """

    friction_velocity_scale: float
    contact_stiffness: float = 1e7
    friction_coefficient: float = 0.5
    normal_smoothing: float = 1e-4
    material_model: MaterialModel = MaterialModel.SMITH_LOG_STABILIZED
    mass_mode: MassMode = MassMode.CONSISTENT
    min_det_f_guard: float = 0.2
    contact_factor_capacity: int | None = None
    contact_candidate_capacity: int | None = None
    publish_policy: str = "strict"
    alignment_policy: str = "guard"
    newton_max_iterations: int = 10
    line_search_max_iterations: int = 8
    linear_max_iterations: int = 200
    linear_tolerance: float = 1e-4
    pcg_mode: PCGMode = PCGMode.DIAGNOSTIC
    use_cuda_graph: bool = False
    precision: Precision = Precision.FP64

    def __post_init__(self):
        for name, kind in (
            ("material_model", MaterialModel),
            ("mass_mode", MassMode),
            ("pcg_mode", PCGMode),
            ("precision", Precision),
        ):
            object.__setattr__(self, name, kind(getattr(self, name)))
        for name, positive in (
            ("friction_velocity_scale", False),
            ("contact_stiffness", True),
            ("friction_coefficient", False),
            ("normal_smoothing", False),
        ):
            value = getattr(self, name)
            if not _number(value) or not math.isfinite(value) or value < 0 or (positive and value == 0):
                raise ValueError(f"Invalid {name}")
        if self.friction_coefficient > 0 and self.friction_velocity_scale == 0:
            raise ValueError("Friction requires a positive friction_velocity_scale")
        if not _number(self.min_det_f_guard) or not math.isfinite(self.min_det_f_guard) or self.min_det_f_guard <= 0:
            raise ValueError("Invalid detF guard")
        if self.contact_factor_capacity is not None and (
            type(self.contact_factor_capacity) is not int or self.contact_factor_capacity < 0
        ):
            raise ValueError("Contact factor capacity must be a nonnegative integer")
        if self.contact_candidate_capacity is not None and (
            type(self.contact_candidate_capacity) is not int or not 0 <= self.contact_candidate_capacity < 2**31
        ):
            raise ValueError("Invalid spatial candidate capacity")
        if self.publish_policy not in ("strict", "last_accepted", "current_iterate"):
            raise ValueError("Unknown coupled publication policy")
        if self.alignment_policy not in ("guard", "mask"):
            raise ValueError("Unknown contact alignment policy")
        for value in (self.newton_max_iterations, self.line_search_max_iterations, self.linear_max_iterations):
            if type(value) is not int or value <= 0:
                raise ValueError("Iteration budgets must be positive integers")
        if not _number(self.linear_tolerance) or not math.isfinite(self.linear_tolerance) or self.linear_tolerance < 0:
            raise ValueError("Invalid linear tolerance")
        if type(self.use_cuda_graph) is not bool or (self.use_cuda_graph and self.pcg_mode != PCGMode.PRODUCTION):
            raise ValueError("use_cuda_graph requires CUDA production PCG and a bool")

    def identity_fields(self):
        """Return the option part of checkpoint identity.

        Execution mode (PCG observation level and graph capture) is excluded.
        Capacity options are excluded because their resolved values, including
        the device-independent contact batch size, belong to the contact
        resource identity.
        """
        return {
            "config_version": np.array([CONFIG_VERSION], dtype=np.int32),
            "contact_model": _text(CONTACT_MODEL),
            "globalization": _text(GLOBALIZATION),
            "contact_options": np.array(
                [
                    self.contact_stiffness,
                    self.friction_coefficient,
                    self.friction_velocity_scale,
                    self.normal_smoothing,
                ],
                dtype=np.float64,
            ),
            "material_model": _text(self.material_model.value),
            "mass_mode": _text(self.mass_mode.value),
            "min_det_f_guard": np.array([self.min_det_f_guard], dtype=np.float64),
            "publish_policy": _text(self.publish_policy),
            "alignment_policy": _text(self.alignment_policy),
            "iteration_policy": np.array(
                [
                    self.newton_max_iterations,
                    self.line_search_max_iterations,
                    self.linear_max_iterations,
                    self.linear_tolerance,
                ],
                dtype=np.float64,
            ),
            "precision": _text(self.precision.value),
        }
