# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Complete monolithic q+x solver with transactional state and force publication."""

import enum
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import warp as wp

from ...sim import JointType
from ..solver import SolverBase
from . import coupled_kernels as kernels
from .articulation_state import _predict, _recover_history
from .articulation_variational import ArticulationObjective, _TreeKinematics
from .collision import _validate_and_collect_boundary_faces
from .contact_forces import MonolithicContactForces
from .contact_variational import ContactObjective, ContactStatus
from .coupled_state import CoupledState
from .joint_parameters import JointParameters
from .linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
)
from .segment_graph import _SegmentGraphCache
from .solver_types import (
    JointTerms,
    MassMode,
    MaterialModel,
    PCGMode,
    Stats,
    Status,
    _SolverMonolithicInternalConfig,
    _StepFailure,
)
from .tet_assembly import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from .tet_precision import TetObjective
from .variational_iteration import iterate, measure


class SolverMonolithic(SolverBase):
    """Keep strict defaults; opt in to masked contact and stopped publication.

    Nonconservative energy acceptance is heuristic, not a variational guarantee.
    Stopped publication retains all physical/state gates except the residual goal.
    ``last_accepted`` requires an accepted Newton update; ``current_iterate``
    also permits the initialized predictor when no update was accepted. Neither
    policy publishes rejected trials. Force validity is not equilibrium.
    """

    class Precision(enum.Enum):
        """Construction-time precision of nonlinear state and decisions."""

        FP64 = "fp64"
        FP32 = "fp32"

    supports_collision_pipeline = False
    Status = Status
    Stats = Stats
    JointTerms = JointTerms
    PCGMode = PCGMode
    MaterialModel = MaterialModel
    MassMode = MassMode

    def __init__(
        self,
        model,
        *,
        contact_model,
        globalization,
        friction_velocity_scale,
        contact_stiffness=1e7,
        friction_coefficient=0.5,
        normal_smoothing=1e-4,
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=None,
        min_det_f_guard=0.2,
        contact_factor_capacity=None,
        publish_policy="strict",
        smooth_sdf=None,
        alignment_policy="guard",
        allow_nonconservative_energy=False,
        joint_terms=None,
        newton_max_iterations=10,
        line_search_max_iterations=8,
        linear_max_iterations=200,
        linear_tolerance=1e-4,
        linear_absolute_tolerance=1e-6,
        pcg_mode=PCGMode.DIAGNOSTIC,
        use_cuda_graph=False,
        precision=Precision.FP64,
    ):
        self._precision = self.Precision(precision)
        self._dtype = wp.float64 if self._precision == self.Precision.FP64 else wp.float32
        self._numpy_dtype = np.float64 if self._precision == self.Precision.FP64 else np.float32
        self._precision_identity = (self._precision, self._dtype, self._numpy_dtype)
        if smooth_sdf is not None:
            smooth_sdf = smooth_sdf.for_precision(self._dtype)
        if publish_policy not in ("strict", "last_accepted", "current_iterate"):
            raise ValueError("Unknown coupled publication policy")
        if contact_model not in ("C-EE", "C-IE", "C-II"):
            raise ValueError("An explicit supported contact law is required")
        if type(allow_nonconservative_energy) is not bool:
            raise ValueError("Nonconservative energy opt-in must be boolean")
        if (
            (contact_model != "C-EE" or alignment_policy == "mask")
            and globalization != "ResNorm"
            and not allow_nonconservative_energy
        ):
            raise ValueError("Nonconservative energy acceptance requires explicit opt-in")
        self._coupled_publish_policy = publish_policy
        self._allow_nonconservative_energy = allow_nonconservative_energy
        self._policy_identity = (globalization, publish_policy, alignment_policy, allow_nonconservative_energy)
        self._stopped_failure = None
        if not np.isfinite(min_det_f_guard) or min_det_f_guard <= 0:
            raise ValueError("Invalid detF guard")
        _validate_and_collect_boundary_faces(model)
        if not model.particle_count or not model.tet_count:
            raise ValueError("Coupled entry requires one tet body")
        if not np.all(model.particle_world.numpy() == model.body_world.numpy()[0]):
            raise ValueError("Articulation and tet must share one world")
        radius = model.particle_radius.numpy()
        if not np.isfinite(radius).all() or np.any(radius < 0) or not np.all(radius == radius[0]):
            raise ValueError("Coupled particles require one finite nonnegative radius")
        self._tet = TetObjective(
            model, material_model=material_model, mass_mode=mass_mode, density=tet_rest_density, dtype=self._dtype
        )
        self._contact = ContactObjective(
            model,
            self._tet.mapping.numpy(),
            q_dof_count=model.joint_dof_count,
            contact_model=contact_model,
            normal_stiffness=contact_stiffness,
            friction_coefficient=friction_coefficient,
            friction_velocity_scale=friction_velocity_scale,
            smoothing_width=normal_smoothing,
            particle_radius=float(radius[0]),
            smooth_sdf=smooth_sdf,
            alignment_policy=alignment_policy,
            dtype=self._dtype,
        )
        capacity = 3 * self._contact.count if contact_factor_capacity is None else contact_factor_capacity
        if type(capacity) is not int or capacity < 0:
            raise ValueError("Contact factor capacity must be a nonnegative integer")
        self._guard = min_det_f_guard
        self._physics_sources = dict(self._tet.physics.sources)
        for name in (
            "particle_radius",
            "particle_flags",
            "tet_activations",
            "tri_materials",
            "edge_bending_properties",
        ):
            if getattr(model, name) is not None:
                self._physics_sources[name] = getattr(model, name)
        self._physics_values = {name: value.numpy().copy() for name, value in self._physics_sources.items()}
        self._textures = []
        self._sdf_slots = tuple(model._texture_sdf_subgrid_start_slots)
        self._sdf_fields = self._freeze_sdf(model)
        self._coupled_config = {
            "contact_parameters": np.asarray(self._contact.parameters, dtype=float),
            "contact_model": np.frombuffer(contact_model.encode(), dtype=np.uint8),
            "globalization": np.frombuffer(globalization.encode(), dtype=np.uint8),
            "material": np.frombuffer(material_model.encode(), dtype=np.uint8),
            "mass": np.frombuffer(mass_mode.encode(), dtype=np.uint8),
            "guard_capacity": np.array([min_det_f_guard, capacity]),
            "physics_digest": np.frombuffer(bytes.fromhex(self._tet.physics.identity), dtype=np.uint8),
            "publication_policy": np.frombuffer(publish_policy.encode(), dtype=np.uint8),
            "alignment_policy": np.frombuffer(alignment_policy.encode(), dtype=np.uint8),
            "heuristic_energy": np.array([allow_nonconservative_energy], dtype=np.uint8),
        }
        if self._dtype == wp.float32:
            self._coupled_config["quaternion_norm_tolerance"] = np.array([self._contact.quaternion_norm_tolerance])
        if smooth_sdf is not None:
            self._coupled_config["smooth_sdf_digest"] = np.frombuffer(bytes.fromhex(smooth_sdf.digest), dtype=np.uint8)
        # Include effective numeric policy in checkpoint identity; device execution
        # mode is deliberately excluded so CPU/CUDA checkpoints remain portable.
        self._coupled_config["iteration_policy"] = np.array(
            [
                newton_max_iterations,
                line_search_max_iterations,
                linear_max_iterations,
                linear_tolerance,
                linear_absolute_tolerance,
            ]
        )
        self._scope_counts = tuple(getattr(model, name) for name in ("spring_count", "tri_count", "edge_count"))
        if globalization not in ("Armijo", "ResNorm", "ArmijoOrResNorm"):
            raise ValueError("Unknown articulation globalization/publication policy")
        for value in (newton_max_iterations, line_search_max_iterations, linear_max_iterations):
            if type(value) is not int or value <= 0:
                raise ValueError("Iteration budgets must be positive integers")
        for value in (linear_tolerance, linear_absolute_tolerance):
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError("Invalid linear tolerance")
        if joint_terms is None:
            joint_terms = self.JointTerms()
        if not isinstance(joint_terms, self.JointTerms):
            raise ValueError("joint_terms must be SolverMonolithic.JointTerms")
        options = {name: getattr(joint_terms, name) for name in joint_terms.__dataclass_fields__}
        self._joint_terms = JointParameters(model, **options)
        for name in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_friction"):
            if name not in self._joint_terms.allowed_properties and np.any(getattr(model, name).numpy() != 0):
                raise ValueError(f"Disabled joint term has nonzero parameters: {name}")
        parameters = self._joint_terms.params.numpy()
        self._kinematics = self._make_kinematics(
            model,
            extra_config={
                "joint_terms": parameters,
                "joint_switches": np.array(
                    [joint_terms.implicit_pd, joint_terms.limits, joint_terms.friction], dtype=np.uint8
                ),
            },
        )
        SolverBase.__init__(self, model)
        self._states = self._make_states()
        self._objective = ArticulationObjective(self._kinematics, parameters)
        self._config = _SolverMonolithicInternalConfig()
        # Sharpa refinement certifies these tighter goals within the existing budgets.
        self._config = replace(
            self._config,
            merit_absolute_global=self._config.merit_absolute_global * 0.1,
            merit_absolute_q=self._config.merit_absolute_q * 0.1,
            merit_relative_global=self._config.merit_relative_global * 0.1,
            merit_relative_q=self._config.merit_relative_q * 0.1,
        )
        self._globalization = globalization
        self.newton_max_iterations, self.line_search_max_iterations = newton_max_iterations, line_search_max_iterations
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance, self.linear_absolute_tolerance = linear_tolerance, linear_absolute_tolerance
        self._pcg_mode = self.PCGMode(pcg_mode)
        self._use_cuda_graph = use_cuda_graph
        device = model.device
        self._graphs = _SegmentGraphCache(device, enabled=use_cuda_graph)
        # Scratch states are allocated on first use, with the caller's legal initial dt.
        self._original = self._accepted = self._trial = None
        self._step_generation = self._assembly_sequence = 0
        self._last_stats = self.Stats(self.Status.NOT_RUN)
        self._old_joint_t = wp.empty_like(self._kinematics.joint_translations)
        self._old_joint_r = wp.empty_like(self._kinematics.joint_rotations)

        self._config = replace(
            self._config,
            merit_absolute_global=1e-7,
            merit_absolute_q=2.5e-8,
            merit_absolute_x=1e-7,
            merit_relative_global=1e-6,
            merit_relative_q=1e-5,
            merit_relative_x=1e-6,
            step_tolerance_global=1e-10,
            step_tolerance_x=1e-10,
        )
        n, nq, device = self._tet.ids.size, model.joint_dof_count, model.device
        pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), self._tet.mapping.numpy(), x_dof_start=nq)
        self._tet_assembly = create_tet_assembly_workspace(
            pattern, tet_count=model.tet_count, device=device, physics=self._tet.physics
        )
        self._linear = MonolithicLinearWorkspace(
            MonolithicLinearLayout(nq, n),
            MonolithicLinearCapacities(
                nq * nq + len(pattern.global_rows) + capacity * (nq + 9) ** 2,
                len(pattern.ax_rows),
                capacity,
                self.linear_max_iterations,
                self.linear_max_iterations + 2,
            ),
            self._tet.mapping,
            device=device,
            pcg_mode=self.pcg_mode.value,
            use_cuda_graph=self._use_cuda_graph,
        )
        self._linear._set_fixed_triplet_pattern(
            global_rows=np.r_[np.repeat(np.arange(nq, dtype=np.int32), nq), pattern.global_rows],
            global_columns=np.r_[np.tile(np.arange(nq, dtype=np.int32), nq), pattern.global_columns],
            internal_rows=pattern.ax_rows,
            internal_columns=pattern.ax_columns,
        )
        for name in ("_rhs", "_y", "_delta", "_diagonal"):
            setattr(self, name, wp.zeros(nq + 3 * n, dtype=float, device=device))
        self._complete_r = wp.zeros(nq + 3 * n, dtype=self._dtype, device=device)
        self._post_r, self._post_error = wp.zeros_like(self._complete_r), wp.zeros_like(self._complete_r)
        self._tet_r32 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._frozen_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        self._frozen_gravity = wp.zeros_like(model.gravity)
        self._out_backup = None
        self._dt = None

    def _freeze_sdf(self, model):
        """Bind actual texels and indirection, excluding process-specific GPU handles."""
        fields = {}
        table = model._texture_sdf_data
        if table is None or not table.size:
            return fields
        values = table.numpy()
        for name in values.dtype.names:
            if name not in ("coarse_texture", "subgrid_texture", "subgrid_start_slots"):
                fields["sdf." + name] = values[name].copy()
        for name in ("_texture_sdf_coarse_textures", "_texture_sdf_subgrid_textures"):
            textures = getattr(model, name)
            if len(textures) != table.size:
                raise ValueError("Mesh SDF requires owned texture assets")
            for i, texture in enumerate(textures):
                shape = (texture.depth, texture.height, texture.width)
                if texture.num_channels != 1:
                    shape += (texture.num_channels,)
                pixels = np.empty(shape, dtype=np.float32)
                texture.copy_to(pixels)
                self._textures.append((name, i, texture, pixels))
                fields[f"{name}.{i}"] = pixels
        for i, slots in enumerate(model._texture_sdf_subgrid_start_slots):
            fields[f"sdf.slots.{i}"] = slots.numpy().copy()
        return fields

    def _make_kinematics(self, model, *, extra_config):
        fields = dict(extra_config)
        fields.update(self._physics_values)
        fields.update(self._contact.values)
        fields.update(self._coupled_config)
        fields.update(self._sdf_fields)
        return _TreeKinematics(model, extra_config=fields, dtype=self._dtype)

    def _make_states(self):
        return CoupledState(self._kinematics)

    def create_state(self, *, dt, source=None):
        """Initialize full P3 history with positive dt, optionally from ordinary model state."""
        self._validate_components()
        self._kinematics.validate_identity()
        return self._states.create(dt=dt, source=source)

    def reset_state(self, state, *, dt):
        """Reinitialize history at current coordinates and clear published contact forces."""
        self._validate_components()
        self._kinematics.validate_identity()
        self._states.reset(state, dt=dt)

    def save_checkpoint(self, state, path):
        """Atomically save complete NPZ state, appending .npz as in NumPy's path API."""
        self._validate_components()
        self._kinematics.validate_identity()
        path = Path(path)
        if not str(path).endswith(".npz"):
            path = Path(str(path) + ".npz")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".npz", delete=False) as f:
                temporary = Path(f.name)
            self._states.save_checkpoint(state, temporary)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load_checkpoint(self, state, path):
        """Validate a checkpoint in scratch before replacing the target state."""
        self._validate_components()
        self._kinematics.validate_identity()
        self._states.load_checkpoint(state, path)

    def get_contact_forces(self, state):
        """Return an independent, read-only snapshot of this state's final contact forces."""
        self._validate_components()
        self._kinematics.validate_identity()
        history = self._states.validate(state)
        return MonolithicContactForces(
            history.contact_particle_force.numpy(),
            history.contact_body_force.numpy(),
            history.contact_joint_force.numpy(),
            bool(history.force_valid.numpy()[0]),
            int(history.force_generation.numpy()[0]),
        )

    @property
    def contacts(self):
        """Return None; monolithic contact geometry is evaluated internally."""
        return None

    @contacts.setter
    def contacts(self, value):
        if value is not None:
            raise ValueError("SolverMonolithic does not accept external Contacts")

    def update_contacts(self, contacts, state=None):
        """Reject external contact updates; use the independent collision pipeline separately."""
        raise ValueError("SolverMonolithic does not accept external Contacts")

    def set_collision_frequency(self, *, collision_frequency=None, collision_frequency_type=None):
        """Reject collision scheduling; no configuration is needed for internal contact."""
        if collision_frequency is not None or collision_frequency_type is not None:
            raise ValueError("SolverMonolithic does not support collision scheduling")

    def reset(self, state, world_mask=None, flags=None):
        """Require explicit dt when resetting the complete integration history."""
        raise NotImplementedError("Use reset_state(state, dt=...) for SolverMonolithic")

    def notify_model_changed(self, flags):
        """Require reconstruction after changes to the bound model."""
        raise NotImplementedError("Rebuild SolverMonolithic after model changes")

    @property
    def precision(self):
        """Return the immutable nonlinear precision; rebuild to change it."""
        return self._precision

    def _validate_components(self):
        if self._precision_identity != (self._precision, self._dtype, self._numpy_dtype):
            raise ValueError("Precision changed; rebuild solver")
        if self._policy_identity != (
            self._globalization,
            self._coupled_publish_policy,
            self._contact.alignment_policy,
            self._allow_nonconservative_energy,
        ):
            raise ValueError("Coupled policies changed; rebuild solver and rebind state")
        self._contact.validate_identity()
        if (
            tuple(getattr(self.model, name) for name in ("spring_count", "tri_count", "edge_count"))
            != self._scope_counts
        ):
            raise ValueError("Coupled model scope changed")
        for name, source in self._physics_sources.items():
            if getattr(self.model, name) is not source or not np.array_equal(
                source.numpy(), self._physics_values[name]
            ):
                raise ValueError(f"Coupled physics changed: {name}")
        for name, i, texture, pixels in self._textures:
            if getattr(self.model, name)[i] is not texture:
                raise ValueError("SDF texture replaced")
            actual = np.empty_like(pixels)
            texture.copy_to(actual)
            if not np.array_equal(actual, pixels):
                raise ValueError("SDF texels changed")
        if len(self.model._texture_sdf_subgrid_start_slots) != len(self._sdf_slots):
            raise ValueError("SDF indirection layout changed")
        for i, slots in enumerate(self.model._texture_sdf_subgrid_start_slots):
            if slots is not self._sdf_slots[i]:
                raise ValueError("SDF indirection replaced")
            if not np.array_equal(slots.numpy(), self._sdf_fields[f"sdf.slots.{i}"]):
                raise ValueError("SDF indirection changed")

    def step(self, state_in, state_out, control, contacts=None, dt=0.0):
        self._validate_components()
        self._stopped_failure = None
        if contacts is not None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Articulation step requires no contacts and finite positive dt")
        dt = self._kinematics.timestep(dt)
        self._kinematics.validate_identity()
        self._states.validate(state_in)
        self._states.validate(state_out)
        for name in ("joint_f", "joint_target_q", "joint_target_qd"):
            array = getattr(control, name, None)
            if (
                not isinstance(array, wp.array)
                or (array.shape, array.dtype, array.device)
                != ((self._kinematics.count,), wp.float32, self.model.device)
                or not np.isfinite(array.numpy()).all()
            ):
                raise ValueError(f"Invalid articulation control: {name}")
        self._joint_terms.snapshot_targets(control)
        force = state_in.body_f.numpy().astype(self._numpy_dtype)
        joint_force = control.joint_f.numpy().astype(self._numpy_dtype)
        gravity = self.model.gravity.numpy().astype(self._numpy_dtype)
        if (
            not np.isfinite(force).all()
            or not np.isfinite(joint_force).all()
            or not np.isfinite(gravity).all()
            or np.any(force[:, 3:] != 0)
        ):
            raise ValueError("Articulation requires finite forces and zero spatial torque")
        if self._original is None:
            self._original = self._states.copy(state_in)
            self._accepted = self._states.copy(state_in)
            self._trial = self._states.copy(state_in)
        for state in (self._original, self._accepted, self._trial):
            state.assign(state_in)
        self._prepare_step(state_in, state_out, dt)
        self._step_generation += 1
        self._metrics = {
            "matrix_assembly_count": 0,
            "regularization_retries": 0,
            "linear_iterations": 0,
            "linear_scheduled_slots": 0,
            "line_search_iterations": 0,
        }
        self._variational = {
            "globalization": self._globalization,
            "trials": [],
            "linear_postchecks": [],
            "configuration": self._kinematics.digest.hex(),
        }
        k, obj = self._kinematics, self._objective
        k.evaluate(state_in.joint_q)
        wp.copy(obj.tn, k.com)
        wp.copy(obj.rn, k.rotations)
        wp.copy(self._old_joint_t, k.joint_translations)
        wp.copy(self._old_joint_r, k.joint_rotations)
        wp.copy(obj.qn, state_in.joint_q)
        wp.copy(obj.vn, state_in.joint_qd)
        history = getattr(state_in, self._states.namespace)
        wp.launch(
            _predict,
            k.bodies,
            [
                obj.tn,
                obj.rn,
                history.link_v_com,
                history.link_omega,
                history.link_vsym,
                self._dtype(dt),
                obj.predicted_t,
                obj.predicted_r,
            ],
            device=self.model.device,
        )
        obj.force.assign(force[:, :3] + k.values["body_mass"][:, None] * gravity[k.values["body_world"]])
        obj.joint_force.assign(joint_force)
        obj.qt.assign(self._joint_terms.target_q.numpy().astype(self._numpy_dtype))
        obj.vt.assign(self._joint_terms.target_qd.numpy().astype(self._numpy_dtype))
        wp.launch(
            kernels._coordinate_predictor,
            k.count,
            [obj.qn, obj.vn, self._dtype(dt), self._accepted.joint_q, self._accepted.joint_qd],
            device=self.model.device,
        )
        k.evaluate(self._accepted.joint_q)
        if not self._domain(self._accepted):
            wp.copy(self._accepted.joint_q, obj.qn)
            self._accepted.joint_qd.zero_()
        failure = None
        status = self.Status.NONFINITE
        published = False
        try:
            status = self._iterate_articulation(dt)
            if self._can_publish(status):
                self._validate_final(dt, status)
                # A PD-stabilized B cannot certify the candidate's physical inertia.
                try:
                    k.validate_inertia()
                except ValueError as error:
                    raise _StepFailure(self.Status.NONFINITE, str(error)) from error
                final = getattr(self._accepted, self._states.namespace)
                wp.launch(
                    _recover_history,
                    k.bodies,
                    [
                        obj.tn,
                        obj.rn,
                        k.com,
                        k.rotations,
                        self._dtype(dt),
                        final.link_v_com,
                        final.link_omega,
                        final.link_vsym,
                    ],
                    device=self.model.device,
                )
                wp.launch(
                    _recover_history,
                    k.joints,
                    [
                        self._old_joint_t,
                        self._old_joint_r,
                        k.joint_translations,
                        k.joint_rotations,
                        self._dtype(dt),
                        final.joint_v,
                        final.joint_omega,
                        final.joint_vsym,
                    ],
                    device=self.model.device,
                )
                k.evaluate(self._accepted.joint_q, state=self._accepted)
                self._finish_candidate(self._accepted, dt)
                self._states.bind(self._accepted, dt=dt)
                self._states.validate(self._accepted)
                state_out.assign(self._accepted)
                self._metrics.update(self._joint_force_metrics(state_out))
                published = True
        except _StepFailure as error:
            status, failure = error.status, str(error)
        except Exception as error:
            status, failure = self.Status.NONFINITE, str(error)
            raise
        finally:
            if not published:
                self._restore(state_out)
            self._variational["graphs"] = self._graphs.summary()
            self._last_stats = self.Stats(
                status=status,
                failure_reason=failure,
                converged=status == self.Status.SUCCESS and published,
                rolled_back=not published,
                step_generation=self._step_generation,
                preconditioner_kind="actor_block",
                q_block_count=1,
                variational=MappingProxyType(self._variational.copy()),
                **self._metrics,
            )
        stats = self._last_stats
        stopped = not stats.converged and not stats.rolled_back
        detail = dict(stats.variational)
        detail.update(
            publication_status="STOPPED_ACCEPTED" if stopped else "CONVERGED" if stats.converged else "HARD_FAILURE",
            publish_policy=self._coupled_publish_policy,
            alignment_policy=self._contact.alignment_policy,
            energy_acceptance="variational"
            if self._conservative_energy()
            else "heuristic"
            if self._globalization != "ResNorm"
            else "unused",
            accepted_count=sum(bool(t.get("accepted")) for t in detail["trials"]),
        )
        detail["zero_accepted_publication"] = stopped and detail["accepted_count"] == 0
        if stopped:
            detail["stopped_candidate"] = {
                "original_status": stats.status.value,
                "original_failure": self._stopped_failure,
                "accepted_count": detail["accepted_count"],
                "final_ratios": {
                    name: self._metrics["convergence_ratio" + suffix]
                    for name, suffix in (("global", ""), ("q", "_q"), ("x", "_x"))
                },
            }
        self._last_stats = replace(
            stats,
            variational=MappingProxyType(detail),
            failure_reason=(self._stopped_failure or stats.status.value) if stopped else stats.failure_reason,
        )
        return state_out

    def _can_publish(self, status):
        return status == self.Status.SUCCESS or (
            self._coupled_publish_policy in ("last_accepted", "current_iterate")
            and status
            in (
                self.Status.NONLINEAR_MAX_ITERATIONS,
                self.Status.NONLINEAR_STAGNATION,
                self.Status.REGULARIZATION_EXHAUSTED,
            )
            and (
                self._coupled_publish_policy == "current_iterate"
                or any(t.get("accepted") for t in self._variational["trials"])
            )
        )

    def _iterate_articulation(self, dt):
        try:
            return iterate(self, dt)
        except _StepFailure as error:
            if not self._can_publish(error.status):
                raise
            self._stopped_failure = str(error)
            return error.status

    def _prepare_step(self, state_in, state_out, dt):
        self._dt = dt
        self._step_generation = max(
            self._step_generation,
            int(state_in._monolithic_coupled.force_generation.numpy()[0]),
            int(state_out._monolithic_coupled.force_generation.numpy()[0]),
        )
        if self._out_backup is None:
            self._out_backup = self._states.copy(state_out)
        else:
            self._out_backup.assign(state_out)
        wp.copy(self._frozen_force, state_in.particle_f)
        wp.copy(self._frozen_gravity, self.model.gravity)
        k = self._kinematics
        k.evaluate(state_in.joint_q)
        self._contact.begin_step(state_in.particle_q, k.poses, k.com, k.jt, k.jw)
        wp.launch(
            kernels.particle_predict,
            self.model.particle_count,
            [
                state_in.particle_q,
                state_in.particle_qd,
                self._tet.mapping,
                self._dtype(dt),
                self._accepted.particle_q,
                self._accepted.particle_qd,
            ],
            device=self.model.device,
        )

    def _evaluate(self, state, dt, *, build_tangent=True):
        k, contact, tet = self._kinematics, self._contact, self._tet
        buffers = [
            state.joint_q,
            state.particle_q,
            self._original.particle_q,
            self._original.particle_qd,
            self._frozen_force,
            self._frozen_gravity,
            self._complete_r,
        ]
        for owner in (self._objective, k, tet, contact, contact.start, contact.current):
            buffers.extend(value for value in vars(owner).values() if isinstance(value, wp.array))

        def evaluate():
            self._objective.evaluate(state.joint_q, dt, build_tangent=build_tangent)
            tet.evaluate(
                state.particle_q,
                self._original.particle_q,
                self._original.particle_qd,
                self._frozen_force,
                self._frozen_gravity,
                dt=dt,
                min_det_f_guard=self._guard,
            )
            contact.evaluate(state.particle_q, k.poses, k.com, k.jt, k.jw, dt=dt)
            wp.launch(
                kernels.combine,
                self._complete_r.size,
                [self._objective.residual, tet.residual, contact.residual, self._complete_r],
                device=self.model.device,
            )

        self._graphs.run("coupled_objective", (self.precision.value, dt, build_tangent), buffers, evaluate)
        self._current_state = state
        if tet.flags.numpy()[0]:
            status = self.Status.TET_INVERSION if tet.flags.numpy()[0] == 1 else self.Status.NONFINITE
            raise _StepFailure(status, "Invalid tet candidate")
        if contact.flags.numpy()[0]:
            raise _StepFailure(self.Status.NONFINITE, f"Invalid contact geometry: {contact.flags.numpy().tolist()}")
        # Graph replay executes device work, not evaluate()'s host-side marker.
        contact._evaluated = True
        residual = self._complete_r.numpy().copy()
        parts = np.r_[
            self._objective.link_energy.numpy(),
            self._objective.joint_energy.numpy(),
            tet.parts.numpy(),
            contact.energy.numpy().ravel(),
        ]
        if not np.isfinite(residual).all() or not np.isfinite(parts).all() or not np.isfinite(parts.sum()):
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite coupled E/r")
        self._metrics["min_det_f"] = float(tet.min_det_f.numpy()[0])
        return parts.sum(), residual, abs(parts).sum()

    def _initial_reference(self, dt):
        return self._evaluate(self._original, dt, build_tangent=False)[1]

    def _conservative_energy(self):
        return self._contact.conservative_energy

    def _trial_evaluation(self, dt):
        try:
            return self._evaluate(self._trial, dt, build_tangent=False)
        except _StepFailure as error:
            self._variational["trials"].append({"accepted": False, "domain_failure": str(error)})
            return None

    def _take_trial(self, alpha, dt):
        wp.launch(
            kernels._coordinate_trial,
            self._kinematics.count,
            [
                self._accepted.joint_q,
                self._delta,
                self._dtype(alpha),
                self._objective.qn,
                self._dtype(dt),
                self._trial.joint_q,
                self._trial.joint_qd,
            ],
            device=self.model.device,
        )
        wp.launch(
            kernels.particle_trial,
            self.model.particle_count,
            [
                self._accepted.particle_q,
                self._original.particle_q,
                self._tet.mapping,
                self._delta,
                self._kinematics.count,
                self._dtype(alpha),
                self._dtype(dt),
                self._trial.particle_q,
                self._trial.particle_qd,
            ],
            device=self.model.device,
        )

    def _effective_displacement(self):
        ids = self._tet.ids.numpy()
        return np.r_[
            self._trial.joint_q.numpy() - self._accepted.joint_q.numpy(),
            (self._trial.particle_q.numpy()[ids] - self._accepted.particle_q.numpy()[ids]).ravel(),
        ]

    def _assemble(self, iteration):
        k, linear, obj = self._kinematics, self._linear, self._objective
        self._assembly_sequence += 1
        self._generation = MonolithicLinearGeneration(
            self._step_generation,
            iteration,
            self._step_generation,
            self._assembly_sequence,
            history_epoch=self._step_generation,
        )
        assembly = linear.begin_assembly(self._generation)
        nq = k.count
        wp.launch(
            kernels._scatter,
            (nq, nq),
            [obj.tangent32, assembly.aq_actor_dense, assembly.global_scalar_triplets.values, self._diagonal],
            device=self.model.device,
        )
        scatter = TetScatterBuffers(
            assembly.ax_internal_triplets.values,
            assembly.global_scalar_triplets.values[nq * nq : nq * nq + 9 * assembly.ax_internal_triplets.capacity],
        )
        assemble_tet_residual_tangent(
            self.model,
            candidate_particle_q=self._current_state.particle_q,
            particle_q_n=self._original.particle_q,
            particle_qd_n=self._original.particle_qd,
            frozen_particle_f=self._frozen_force,
            dynamic_particle_ids=self._tet.ids,
            particle_to_dynamic=self._tet.mapping,
            residual_x=self._tet_r32,
            dt=float(self._dt),
            min_det_f_guard=self._guard,
            workspace=self._tet_assembly,
            scatter=scatter,
            frozen_gravity=self._frozen_gravity,
        )
        if self._tet_assembly.failure_flags.numpy()[0]:
            raise _StepFailure(self.Status.TET_PROJECTION_FAILURE, "Invalid tet tangent")
        self._contact.assemble_tangent(assembly)
        if self._contact.flags.numpy()[0]:
            status = (
                self.Status.CONTACT_OVERFLOW
                if self._contact.flags.numpy()[0] == ContactStatus.CAPACITY
                else self.Status.NONFINITE
            )
            raise _StepFailure(status, "Invalid contact tangent or capacity")
        self._require_linear(linear.finalize_assembly(generation=self._generation).status)
        wp.launch(
            kernels.dynamic_diagonal,
            self._diagonal.size,
            [
                k.jt,
                k.jw,
                k.rotations,
                k.mass,
                k.second,
                obj.armature,
                self._tet.ids,
                self._tet.physics.dynamic_mass_diagonal,
                self._dtype(self._dt),
                self._diagonal,
            ],
            device=self.model.device,
        )
        self._require_linear(
            linear.build_scaling(self._diagonal, epsilon_d=self._config.epsilon_d, generation=self._generation)
        )
        self._metrics["matrix_assembly_count"] += 1
        return None, linear.scale.numpy().astype(self._numpy_dtype)

    def _physical_linear_error(self, matrix, direction, residual, scale):
        self._post_r.assign(residual)
        b = self._linear.k_global_scalar_bsr
        buffers = [b.offsets, b.columns, b.values, self._delta, self._post_r, self._linear.scale, self._post_error]
        self._graphs.run(
            "coupled_posterior",
            (self.precision.value,),
            buffers,
            lambda: wp.launch(kernels.sparse_posterior, self._post_error.size, buffers, device=self.model.device),
        )
        return self._post_error.numpy().copy()

    def _validate_final(self, dt, status):
        energy, residual, _ = self._evaluate(self._accepted, dt)
        _, scale = self._assemble(self._metrics["nonlinear_iterations"])
        values = measure(self, residual, self._reference_residual, scale)
        self._variational.update(objective_total=energy, residual=residual.tolist())
        if not np.isfinite(energy) or not np.isfinite(residual).all() or not np.isfinite(scale).all():
            raise _StepFailure(self.Status.NONFINITE, "Final coupled candidate contains nonfinite E/r/scale")
        if not self._domain(self._accepted):
            raise _StepFailure(self.Status.NONFINITE, "Final coupled candidate left the local domain")
        if not self._can_publish(status) or (
            status == self.Status.SUCCESS and any(m > goal for m, _, goal, _ in values.values())
        ):
            raise _StepFailure(self.Status.NONLINEAR_STAGNATION, "Final coupled candidate failed strict physical gates")

    def _finish_candidate(self, state, dt):
        contact, k = self._contact, self._kinematics
        history = getattr(state, self._states.namespace)
        arrays = [history.contact_joint_force, history.contact_particle_force, history.contact_body_force]
        for array in arrays:
            array.zero_()
        if contact.count:
            wp.launch(
                kernels.contact_forces,
                contact.count,
                [
                    contact.current,
                    contact.local_residual,
                    contact.shapes,
                    contact.bodies,
                    contact.triangles,
                    state.particle_q,
                    k.com,
                    *arrays,
                ],
                device=self.model.device,
            )
        if any(not np.isfinite(array.numpy()).all() for array in arrays):
            raise _StepFailure(self.Status.NONFINITE, "Final contact force is nonfinite")
        history.force_generation.fill_(self._step_generation)
        history.force_valid.fill_(1)
        self._variational["contact_model"] = contact.contact_model
        self._variational["contact_force_generation"] = self._step_generation
        self._variational["contact_keys"] = contact.count

    def _restore(self, state):
        state.assign(self._out_backup)

    def _domain(self, state):
        k, obj = self._kinematics, self._objective
        delta = state.joint_q.numpy() - obj.qn.numpy()
        moving_types = k.values["joint_type"][k.dof_joint.numpy()]
        if np.any(abs(delta[moving_types == JointType.REVOLUTE]) >= math.pi / 2):
            return False
        # The caller has evaluated this candidate's private FK.
        return bool(np.all(np.einsum("bij,bij->b", k.rotations.numpy(), obj.rn.numpy()) > 1.0))

    def _joint_force_metrics(self, state):
        """Bind scalar force diagnostics to the successfully published state."""
        p = self._objective.params.numpy()
        q, v = state.joint_q.numpy(), state.joint_qd.numpy()
        raw = p[:, 0] * (q - self._objective.qt.numpy()) + p[:, 1] * (v - self._objective.vt.numpy())
        pd = -np.clip(raw, -p[:, 2], p[:, 2])
        limit, power = np.zeros_like(q), np.zeros_like(q)
        for sign, bound in ((1.0, p[:, 4]), (-1.0, p[:, 3])):
            d = sign * (q - bound)
            z = np.clip(d / p[:, 7], 0.0, 1.0)
            activation = z * z * (3.0 - 2.0 * z)
            damper = p[:, 6] * activation * np.maximum(sign * v, 0.0)
            limit -= sign * (p[:, 5] * np.maximum(d, 0.0) + damper)
            power -= damper * sign * v
        friction = -p[:, 8] * v / np.hypot(v, p[:, 9])
        return {
            "joint_force_generation": self._step_generation,
            "joint_pd_force": tuple(pd.tolist()),
            "joint_limit_force": tuple(limit.tolist()),
            "joint_friction_force": tuple(friction.tolist()),
            "joint_pd_saturated": tuple(((p[:, 0] + p[:, 1] > 0) & (abs(raw) >= p[:, 2])).tolist()),
            "joint_limit_damping_power": tuple(power.tolist()),
            "joint_friction_power": tuple((friction * v).tolist()),
        }

    @property
    def last_stats(self):
        return self._last_stats

    def _require_linear(self, status: MonolithicLinearStatus) -> None:
        if status == MonolithicLinearStatus.SUCCESS:
            return
        mapped = {
            "NON_POSITIVE_PIVOT": self.Status.PRECONDITIONER_FACTORIZATION_FAILED,
            "NON_POSITIVE_CURVATURE": self.Status.LINEAR_NON_POSITIVE_CURVATURE,
            "NEAR_ZERO_CURVATURE": self.Status.LINEAR_NON_POSITIVE_CURVATURE,
            "NON_POSITIVE_PRECONDITIONED_RESIDUAL": self.Status.LINEAR_PRECONDITIONER_NOT_SPD,
            "STAGNATION": self.Status.LINEAR_STAGNATION,
            "MAX_ITERATIONS": self.Status.LINEAR_MAX_ITERATIONS,
            "GLOBAL_TRIPLET_OVERFLOW": self.Status.CONTACT_OVERFLOW,
            "CONTACT_FACTOR_OVERFLOW": self.Status.CONTACT_OVERFLOW,
        }
        if status.name in ("STALE_GENERATION", "INVALID_ARGUMENT", "PRECONDITIONER_NOT_FACTORED"):
            raise ValueError(f"Invalid linear workspace contract: {status.name}")
        raise _StepFailure(mapped.get(status.name, self.Status.NONFINITE), status.name)

    @property
    def use_cuda_graph(self) -> bool:
        """Return the immutable experimental PCG and stable-segment capture option."""
        return self._use_cuda_graph

    @property
    def pcg_mode(self) -> PCGMode:
        """Return the immutable experimental PCG observation policy."""
        return self._pcg_mode
