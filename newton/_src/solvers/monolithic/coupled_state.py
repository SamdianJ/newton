# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete private q/x continuation and final contact-force publication fields."""

import copy

import numpy as np
import warp as wp

from ...sim import Model
from .articulation_state import ArticulationState


class CoupledState(ArticulationState):
    """Extend PR2 ownership with authoritative native-precision particles and force generations."""

    namespace = "_monolithic_coupled"
    version = 3

    def __init__(self, kinematics):
        super().__init__(kinematics)
        n, b, q = self.model.particle_count, kinematics.bodies, kinematics.count
        self.schema.update(
            bound_x=((n,), self.vector),
            bound_v=((n,), self.vector),
            contact_particle_force=((n,), self.vector),
            contact_body_force=((b,), wp.spatial_vectord if self.dtype == wp.float64 else wp.spatial_vector),
            contact_joint_force=((q,), self.dtype),
            force_generation=((1,), wp.int64),
            force_valid=((1,), wp.int32),
            precision=((1,), wp.int32),
        )
        self.fixed = self.model.particle_inv_mass.numpy() == 0
        self.fixed_positions = self.model.particle_q.numpy().astype(kinematics.numpy_dtype)[self.fixed].copy()

    def create(self, *, dt, source=None):
        if source is not None:
            for name, value in vars(source).items():
                if isinstance(value, Model.AttributeNamespace) and name != self.namespace:
                    raise ValueError("Explicit coupled initialization cannot import legacy history")
        return super().create(dt=dt, source=source)

    def _copy_coordinates(self, state, source):
        super()._copy_coordinates(state, source)
        for name in ("particle_q", "particle_qd"):
            setattr(state, name, wp.array(getattr(source, name).numpy(), dtype=self.vector, device=self.device))
        if getattr(source, "particle_f", None) is not None:
            wp.copy(state.particle_f, source.particle_f)

    def _particle_schema(self):
        n = self.model.particle_count
        return {"particle_q": ((n,), self.vector), "particle_qd": ((n,), self.vector), "particle_f": ((n,), wp.vec3)}

    def validate(self, state, *, bound=True):
        for name in self._particle_schema():
            if getattr(state, name, None) is None:
                raise ValueError(f"Missing coupled state array: {name}")
        history = super().validate(state, bound=bound)
        x, v = state.particle_q.numpy(), state.particle_qd.numpy()
        if not np.array_equal(x[self.fixed], self.fixed_positions) or np.any(v[self.fixed] != 0):
            raise ValueError("Fixed particles must retain their position and zero velocity")
        if bound:
            if "precision" in self.schema and history.precision.numpy()[0] != self.precision_bits:
                raise ValueError("Incompatible state precision")
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

    @property
    def precision_bits(self):
        return 64 if self.dtype == wp.float64 else 32

    def load_checkpoint(self, state, path):
        """Validate schema-two FP64 history before adding the precision identity."""
        with np.load(path, allow_pickle=False) as checkpoint:
            legacy = "history.precision" not in checkpoint.files
        if not legacy:
            return super().load_checkpoint(state, path)
        if self.dtype != wp.float64:
            raise ValueError("Legacy checkpoints require FP64")
        scratch = self.copy(state)
        history = getattr(scratch, self.namespace)
        precision = history.precision
        del history.precision
        legacy_owner = copy.copy(self)
        legacy_owner.schema = {name: item for name, item in self.schema.items() if name != "precision"}
        legacy_owner.version = 2
        history.schema_version.fill_(2)
        # The old digest is checked by the old schema validator, never replaced.
        ArticulationState.load_checkpoint(legacy_owner, scratch, path)
        history.precision = precision
        history.precision.fill_(64)
        history.schema_version.fill_(self.version)
        self.validate(scratch)
        state.assign(scratch)

    def bind(self, state, *, dt):
        super().bind(state, dt=dt)
        history = getattr(state, self.namespace)
        if "precision" in self.schema:
            history.precision.fill_(self.precision_bits)
        wp.copy(history.bound_x, state.particle_q)
        wp.copy(history.bound_v, state.particle_qd)
