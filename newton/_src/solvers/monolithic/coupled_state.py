# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete private q/x continuation and final contact-force publication fields."""

import copy

import numpy as np
import warp as wp

from ...sim import Model
from . import state_checks
from .articulation_state import ArticulationState
from .material_inputs import _cable_geometry, _shell_adjacency


class CoupledState(ArticulationState):
    """Extend PR2 ownership with authoritative native-precision particles and force generations."""

    namespace = "_monolithic_coupled"
    version = 4

    def __init__(self, kinematics, actors, rigid=None, materials=None):
        self.actors = actors
        self.rigid = rigid
        self.materials = materials
        if materials is not None:
            self.version = 6
        self.layout_metadata = {
            "layout_version": np.array([2 if materials is not None else 1], dtype=np.int32),
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
        if materials is not None:
            edges = materials.theta_count
            twists = sum(len(obj["ref_twist"]) for obj in materials.cable)
            for name, shape, dtype in (
                ("contact_cable_torque", (edges,), self.dtype),
                ("cable_theta", (edges,), self.dtype),
                ("cable_ref_director", (edges,), self.vector),
                ("cable_ref_twist", (twists,), self.dtype),
                ("cable_bound_theta", (edges,), self.dtype),
                ("cable_bound_ref_director", (edges,), self.vector),
                ("cable_bound_ref_twist", (twists,), self.dtype),
            ):
                self.schema[name] = (shape, dtype)
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
        self._layout_expected = {
            name: wp.array(value, dtype=self.schema[name][1], device=self.device)
            for name, value in self.layout_metadata.items()
        }
        self._layout_status = wp.zeros(len(self.layout_metadata), dtype=int, device=self.device)
        self.fixed = self.model.particle_inv_mass.numpy() == 0
        if materials is not None:
            for obj in (*materials.cloth, *materials.cable):
                self.fixed[obj["particle_ids"]] = obj["fixed"]
        # Cable Dirichlet coordinates may move through step controls. Their
        # committed values are authenticated by bound_x/bound_v, not Model rest.
        if materials is not None:
            for obj in materials.cable:
                self.fixed[obj["particle_ids"]] = False
        self.fixed_positions = self.model.particle_q.numpy().astype(kinematics.numpy_dtype)[self.fixed].copy()
        self._fixed_ids = wp.array(np.flatnonzero(self.fixed), dtype=int, device=self.device)
        self._fixed_positions = wp.array(self.fixed_positions, dtype=self.vector, device=self.device)
        self._particle_status = wp.zeros(4, dtype=int, device=self.device)

    def create(self, *, dt, source=None):
        if source is not None:
            for name, value in vars(source).items():
                if isinstance(value, Model.AttributeNamespace) and name != self.namespace:
                    raise ValueError("Explicit coupled initialization cannot import legacy history")
        return super().create(dt=dt, source=source)

    def _initialize_history(self, state, source):
        if self.materials is None:
            return
        history = getattr(state, self.namespace)
        if source is not self.model:
            previous = getattr(source, self.namespace, None)
            if previous is None:
                if not self.materials.theta_count:
                    return
                raise ValueError("Explicit material initialization requires theta and compatible reference frames")
            identity = getattr(previous, "config_digest", None)
            if not isinstance(identity, wp.array) or not np.array_equal(
                identity.numpy(), np.frombuffer(self.kinematics.digest, dtype=np.uint8)
            ):
                raise ValueError("Explicit material initialization requires matching physics and layout")
            for name in self.schema:
                if name.startswith("cable_"):
                    array = getattr(previous, name, None)
                    shape, dtype = self.schema[name]
                    if not isinstance(array, wp.array) or (array.shape, array.dtype, array.device) != (
                        shape,
                        dtype,
                        self.device,
                    ):
                        raise ValueError(f"Invalid material initialization field: {name}")
                    wp.copy(getattr(history, name), array)
        else:
            for name, field in (("theta", "rest_theta"), ("ref_director", "rest_director"), ("ref_twist", "ref_twist")):
                values = [obj[field] for obj in self.materials.cable]
                if values:
                    getattr(history, "cable_" + name).assign(np.concatenate(values).astype(self.kinematics.numpy_dtype))

    def _copy_coordinates(self, state, source):
        # Model.state() leaves arrays of absent kinds as None; keep complete empty arrays.
        if not self.model.joint_count:
            state.joint_q = wp.zeros(0, dtype=self.dtype, device=self.device)
            state.joint_qd = wp.zeros(0, dtype=self.dtype, device=self.device)
        else:
            super()._copy_coordinates(state, source)
        if not self.model.body_count:
            for name, dtype in (
                ("body_q", wp.transform),
                ("body_qd", wp.spatial_vector),
                ("body_f", wp.spatial_vector),
            ):
                setattr(state, name, wp.zeros(0, dtype=dtype, device=self.device))
        if not self.model.particle_count:
            for name in ("particle_q", "particle_qd"):
                setattr(state, name, wp.zeros(0, dtype=self.vector, device=self.device))
            state.particle_f = wp.zeros(0, dtype=wp.vec3, device=self.device)
            return
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
        bits = self.precision_bits if "precision" in self.schema else 0
        precision = history.precision if bits else history.force_valid
        buffers = [
            state.particle_q,
            state.particle_qd,
            self._fixed_ids,
            self._fixed_positions,
            history.bound_x,
            history.bound_v,
            precision,
            history.force_valid,
            history.force_generation,
            self._particle_status,
        ]

        def validate_device():
            self._particle_status.zero_()
            wp.launch(
                state_checks.particle_check,
                max(1, self.model.particle_count),
                [*buffers[:7], bits, *buffers[7:9], bound, self._particle_status],
                device=self.device,
            )

        if self._graphs is None:
            validate_device()
        else:
            self._graphs.run("particle_state_validate", (bound, bits), buffers, validate_device)
        result = self._particle_status.numpy()
        for index, message in enumerate(
            (
                "Fixed particles must retain their position and zero velocity",
                "Incompatible state precision",
                "Particle coordinates/rates changed; explicitly reset coupled history",
                "Invalid coupled force generation",
            )
        ):
            if result[index]:
                raise ValueError(message)
        if bound:
            if self.rigid is not None and self.rigid.count:
                self.rigid.validate(state)
            self._layout_status.zero_()
            for index, (name, expected) in enumerate(self._layout_expected.items()):
                if name in self.schema:
                    wp.launch(
                        state_checks.equal,
                        expected.size,
                        [getattr(history, name), expected, self._layout_status, index],
                        device=self.device,
                    )
            for index, failed in enumerate(self._layout_status.numpy()):
                if failed:
                    raise ValueError(f"Incompatible actor layout: {tuple(self.layout_metadata)[index]}")
        if self.materials is not None:
            self._validate_material_history(state, history, bound=bound)
        return history

    def _validate_material_history(self, state, history, *, bound):
        positions = state.particle_q.numpy()
        for obj in self.materials.cloth:
            _, areas = _shell_adjacency(obj["triangles"], positions)
            if np.any(areas / obj["areas"] <= 1e-8):
                raise ValueError("Cloth state has collapsed triangles")
        theta = history.cable_theta.numpy()
        directors, twists = history.cable_ref_director.numpy(), history.cable_ref_twist.numpy()
        cursor = 0
        tolerance = 2e-6 if self.dtype == wp.float32 else 1e-6
        for obj in self.materials.cable:
            offset, count = int(obj["theta_offset"]), len(obj["rest_theta"])
            part = slice(offset, offset + count)
            lengths, _, _, principal = _cable_geometry(positions[obj["particle_ids"]], directors[part])
            if np.any(lengths / obj["rest_lengths"] <= 1e-8):
                raise ValueError("Cable state has collapsed edges")
            branch = twists[cursor : cursor + len(principal)]
            error = np.remainder(branch - principal + np.pi, 2 * np.pi) - np.pi
            if np.any(abs(error) > tolerance):
                raise ValueError("Cable reference twist is inconsistent with its directors")
            if bool(obj["gauge"]) and theta[offset] != obj["rest_theta"][0].astype(self.kinematics.numpy_dtype):
                raise ValueError("Cable state changed its permanent theta gauge")
            cursor += len(principal)
        if bound:
            for name in ("theta", "ref_director", "ref_twist"):
                if not np.array_equal(
                    getattr(history, "cable_" + name).numpy(), getattr(history, "cable_bound_" + name).numpy()
                ):
                    raise ValueError("Cable coordinates/frames changed; explicitly reset coupled history")

    def reset(self, state, *, dt):
        if self.materials is not None or (self.rigid is not None and self.rigid.count):
            scratch = self._copy_unchecked(state)
            self._reset(scratch, dt=dt)
            if self.rigid is not None and self.rigid.count:
                self.rigid.initialize(scratch, dt)
            self.bind(scratch, dt=dt)
            self.validate(scratch)
            state.assign(scratch)
        else:
            self._reset(state, dt=dt)

    def _reset(self, state, *, dt):
        super().reset(state, dt=dt)
        history = getattr(state, self.namespace)
        if self.materials is not None:
            history.contact_cable_torque.zero_()
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
            if version.shape != (1,) or version.dtype != np.int32 or version[0] not in (2, 3, 4, 5, 6):
                raise ValueError("Unsupported checkpoint schema version")
            version = int(version[0])
        if self.materials is not None and version != self.version:
            raise ValueError("Material checkpoints require complete theta and frame history")
        if version in (5, 6) and self.materials is None:
            raise ValueError("Material checkpoint cannot be loaded into an A/R/T solver")
        if version == self.version:
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
        if self.materials is not None:
            for name in ("theta", "ref_director", "ref_twist"):
                wp.copy(getattr(history, "cable_bound_" + name), getattr(history, "cable_" + name))
