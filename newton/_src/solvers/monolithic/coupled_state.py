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
    version = 4

    def __init__(self, kinematics, actors, rigid=None):
        self.actors = actors
        self.rigid = rigid
        self.layout_metadata = {
            "layout_version": np.array([1], dtype=np.int32),
            "layout_digest": np.frombuffer(bytes.fromhex(actors.identity), dtype=np.uint8),
            "actor_layout": np.frombuffer(actors.layout_payload(), dtype=np.uint8),
        }
        super().__init__(kinematics)
        n, b, q = self.model.particle_count, self.model.body_count, self.model.joint_dof_count
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
        for name, value in self.layout_metadata.items():
            self.schema[name] = (value.shape, wp.int32 if value.dtype == np.int32 else wp.uint8)
        count = 0 if rigid is None else rigid.count
        self.schema.update(
            rigid_com=((count,), self.vector),
            rigid_rotation=((count,), wp.quatd if self.dtype == wp.float64 else wp.quat),
            rigid_v_com=((count,), self.vector),
            rigid_omega=((count,), self.vector),
            rigid_vsym=((count, 6), self.dtype),
            rigid_bound_com=((count,), self.vector),
            rigid_bound_rotation=((count,), wp.quatd if self.dtype == wp.float64 else wp.quat),
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
            if self.rigid is not None and self.rigid.count:
                self.rigid.validate(state)
            for name, expected in self.layout_metadata.items():
                if name in self.schema and not np.array_equal(getattr(history, name).numpy(), expected):
                    raise ValueError(f"Incompatible actor layout: {name}")
            if "precision" in self.schema and history.precision.numpy()[0] != self.precision_bits:
                raise ValueError("Incompatible state precision")
            if not np.array_equal(x, history.bound_x.numpy()) or not np.array_equal(v, history.bound_v.numpy()):
                raise ValueError("Particle coordinates/rates changed; explicitly reset coupled history")
            valid, generation = int(history.force_valid.numpy()[0]), int(history.force_generation.numpy()[0])
            if valid not in (0, 1) or generation < 0 or (valid and generation == 0):
                raise ValueError("Invalid coupled force generation")
        return history

    def reset(self, state, *, dt):
        if self.rigid is not None and self.rigid.count:
            scratch = self._copy_unchecked(state)
            self._reset(scratch, dt=dt)
            self.rigid.initialize(scratch, dt)
            self.bind(scratch, dt=dt)
            self.validate(scratch)
            state.assign(scratch)
        else:
            self._reset(state, dt=dt)

    def _reset(self, state, *, dt):
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
        """Validate the declared old schema before migrating canonical A+T history."""
        with np.load(path, allow_pickle=False) as checkpoint:
            if "history.schema_version" not in checkpoint.files:
                raise ValueError("Missing checkpoint schema version")
            version = checkpoint["history.schema_version"]
            if version.shape != (1,) or version.dtype != np.int32 or version[0] not in (2, 3, 4):
                raise ValueError("Unsupported checkpoint schema version")
            version = int(version[0])
        if version == 4:
            return super().load_checkpoint(state, path)
        if tuple(actor.kind for actor in self.actors.actors) != ("articulation", "tet"):
            raise ValueError("Legacy checkpoints require canonical A+T layout")
        if version == 2 and self.dtype != wp.float64:
            raise ValueError("Legacy schema 2 checkpoints require FP64")
        scratch = self.copy(state)
        history = getattr(scratch, self.namespace)
        legacy_owner = copy.copy(self)
        additions = set(self.layout_metadata) | {name for name in self.schema if name.startswith("rigid_")}
        if version == 2:
            additions.add("precision")
        saved = {name: getattr(history, name) for name in additions}
        for name in additions:
            delattr(history, name)
        legacy_owner.schema = {name: item for name, item in self.schema.items() if name not in additions}
        legacy_owner.version = version
        history.schema_version.fill_(version)
        # Original fields, precision and physical digest are checked in scratch.
        ArticulationState.load_checkpoint(legacy_owner, scratch, path)
        for name, value in saved.items():
            setattr(history, name, value)
        history.precision.fill_(self.precision_bits)
        history.schema_version.fill_(self.version)
        self.validate(scratch)
        state.assign(scratch)

    def bind(self, state, *, dt):
        super().bind(state, dt=dt)
        history = getattr(state, self.namespace)
        for name, value in self.layout_metadata.items():
            getattr(history, name).assign(value)
        if "precision" in self.schema:
            history.precision.fill_(self.precision_bits)
        wp.copy(history.bound_x, state.particle_q)
        wp.copy(history.bound_v, state.particle_qd)
        for name in ("com", "rotation"):
            wp.copy(getattr(history, "rigid_bound_" + name), getattr(history, "rigid_" + name))
