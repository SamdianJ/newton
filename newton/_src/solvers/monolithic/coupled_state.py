# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete private q/x continuation and final contact-force publication fields."""

import numpy as np
import warp as wp

from ...sim import Model
from .articulation_state import ArticulationState


class CoupledState(ArticulationState):
    """Extend PR2 ownership with authoritative FP64 particles and force generations."""

    namespace = "_monolithic_coupled"
    version = 2

    def __init__(self, kinematics):
        super().__init__(kinematics)
        n, b, q = self.model.particle_count, kinematics.bodies, kinematics.count
        self.schema.update(
            bound_x=((n,), wp.vec3d),
            bound_v=((n,), wp.vec3d),
            contact_particle_force=((n,), wp.vec3d),
            contact_body_force=((b,), wp.spatial_vectord),
            contact_joint_force=((q,), wp.float64),
            force_generation=((1,), wp.int64),
            force_valid=((1,), wp.int32),
        )
        self.fixed = self.model.particle_inv_mass.numpy() == 0
        self.fixed_positions = self.model.particle_q.numpy().astype(float)[self.fixed].copy()

    def create(self, *, dt, source=None):
        if source is not None:
            for name, value in vars(source).items():
                if isinstance(value, Model.AttributeNamespace) and name != self.namespace:
                    raise ValueError("Explicit coupled initialization cannot import legacy history")
        return super().create(dt=dt, source=source)

    def _copy_coordinates(self, state, source):
        super()._copy_coordinates(state, source)
        for name in ("particle_q", "particle_qd"):
            setattr(state, name, wp.array(getattr(source, name).numpy(), dtype=wp.vec3d, device=self.device))
        if getattr(source, "particle_f", None) is not None:
            wp.copy(state.particle_f, source.particle_f)

    def _particle_schema(self):
        n = self.model.particle_count
        return {"particle_q": ((n,), wp.vec3d), "particle_qd": ((n,), wp.vec3d), "particle_f": ((n,), wp.vec3)}

    def validate(self, state, *, bound=True):
        for name in self._particle_schema():
            if getattr(state, name, None) is None:
                raise ValueError(f"Missing coupled state array: {name}")
        history = super().validate(state, bound=bound)
        x, v = state.particle_q.numpy(), state.particle_qd.numpy()
        if not np.array_equal(x[self.fixed], self.fixed_positions) or np.any(v[self.fixed] != 0):
            raise ValueError("Fixed particles must retain their position and zero velocity")
        if bound:
            if not np.array_equal(x, history.bound_x.numpy()) or not np.array_equal(v, history.bound_v.numpy()):
                raise ValueError("Particle coordinates/rates changed; explicitly reset coupled history")
            valid, generation = int(history.force_valid.numpy()[0]), int(history.force_generation.numpy()[0])
            if valid not in (0, 1) or generation < 0 or (valid and generation == 0):
                raise ValueError("Invalid coupled force generation")
        return history

    def reset(self, state, *, dt):
        super().reset(state, dt=dt)
        history = getattr(state, self.namespace)
        for name in (
            "contact_particle_force",
            "contact_body_force",
            "contact_joint_force",
            "force_generation",
            "force_valid",
        ):
            getattr(history, name).zero_()

    def bind(self, state, *, dt):
        super().bind(state, dt=dt)
        history = getattr(state, self.namespace)
        wp.copy(history.bound_x, state.particle_q)
        wp.copy(history.bound_v, state.particle_qd)
