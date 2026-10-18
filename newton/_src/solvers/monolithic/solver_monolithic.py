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

from ...sim import JointType, Model
from ..solver import SolverBase
from . import coupled_kernels as kernels
from . import iteration_device, material_runtime, state_checks
from .actors import Actor, ActorTable, _validate_and_collect_boundary_faces
from .articulation import ArticulationObjective, _TreeKinematics
from .articulation_state import _predict, _recover_history
from .contact import ContactStatus
from .contact_forces import MonolithicContactForces
from .contact_pipeline import _PooledContactObjective, _SpatialContactObjective
from .contact_pipeline import _SpatialContactObjective as ContactObjective
from .coupled_state import CoupledState
from .iteration import iterate, measure
from .joint_parameters import JointParameters
from .linear import MonolithicLinearWorkspace
from .linear_types import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
)
from .material_inputs import _MaterialInputs
from .rigid import RigidObjective, _scatter_residual, _scatter_tangent
from .segment_graph import _SegmentGraphCache
from .solver_types import (
    CableInput,
    ClothInput,
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

# The single supported contact law and trial acceptance (P8 production combination).
CONTACT_MODEL = "C-IE"
GLOBALIZATION = "ArmijoOrResNorm"


class SolverMonolithic(SolverBase):
    """Keep strict defaults; opt in to masked contact and stopped publication.

    Model topology, physics, geometry (including SDF storage), and solver
    policies must remain unchanged for this solver's lifetime. Rebuild after
    changes; runtime mutation of these inputs is unsupported and not checked.
    Control targets/forces, state external forces, gravity and dt are step inputs.
    Edit coordinates through reset_state or load_checkpoint to keep history valid.

    Spatial contact candidates refresh at Newton boundaries and remain frozen
    through trials, with zero additional law margin. Unbounded trials can miss
    contacts; the complete step-start residual remains the convergence reference.
    ``contact_candidate_capacity`` defaults to the model's static sample upper bound, and
    factor capacity defaults to three times that bound. Capacity is fixed for
    the solver lifetime; overflow discards complete contacts and reports taint.

    Contact uses the C-IE law and trials use ArmijoOrResNorm acceptance. C-IE
    energy is not a potential, so Armijo acceptance is heuristic, not a
    variational guarantee.

    ``cable_position_tolerance`` [N] and ``cable_theta_tolerance`` [N m]
    set per-cable original-unit L2 residual goals (defaults 1e-9 and 1e-11).
    Both are experimental numerical policies included in checkpoint identity.

    Material scenes optionally accept ``cloth_position_tolerance`` [N] and
    ``joint_residual_tolerance`` [N or N m per joint coordinate]. These cap each
    cloth/joint block's raw L2 residual in addition to the scaled relative goals.
    They are experimental numerical policies included in checkpoint identity.

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
    ClothInput = ClothInput
    CableInput = CableInput

    def __init__(
        self,
        model,
        *,
        friction_velocity_scale,
        contact_stiffness=1e7,
        friction_coefficient=0.5,
        normal_smoothing=1e-4,
        material_model="smith_log_stabilized",
        mass_mode="consistent",
        tet_rest_density=None,
        min_det_f_guard=0.2,
        contact_factor_capacity=None,
        contact_candidate_capacity=None,
        publish_policy="strict",
        smooth_sdf=None,
        alignment_policy="guard",
        joint_terms=None,
        newton_max_iterations=10,
        line_search_max_iterations=8,
        linear_max_iterations=200,
        linear_tolerance=1e-4,
        linear_absolute_tolerance=1e-6,
        pcg_mode=PCGMode.DIAGNOSTIC,
        use_cuda_graph=False,
        precision=Precision.FP64,
        cloth_inputs=(),
        cable_inputs=(),
        cable_position_tolerance=1e-9,
        cable_theta_tolerance=1e-11,
        joint_residual_tolerance=None,
        cloth_position_tolerance=None,
    ):
        cloth_inputs, cable_inputs = tuple(cloth_inputs), tuple(cable_inputs)
        self._materials = _MaterialInputs(model, cloth_inputs, cable_inputs) if cloth_inputs or cable_inputs else None
        self._actors = ActorTable.from_model(model, materials=self._materials)
        layout = MonolithicLinearLayout.from_actors(self._actors)
        self._layout = layout
        self._articulation_actor = next((actor for actor in self._actors.actors if actor.kind == "articulation"), None)
        self._precision = self.Precision(precision)
        self._dtype = wp.float64 if self._precision == self.Precision.FP64 else wp.float32
        self._numpy_dtype = np.float64 if self._precision == self.Precision.FP64 else np.float32
        residual_caps = {
            name: value
            for name, value in (
                ("joint_residual_tolerance", joint_residual_tolerance),
                ("cloth_position_tolerance", cloth_position_tolerance),
            )
            if value is not None
        }
        if residual_caps and self._materials is None:
            raise ValueError("Joint/cloth physical residual caps require a material scene")
        for name, value in (
            ("cable_position_tolerance", cable_position_tolerance),
            ("cable_theta_tolerance", cable_theta_tolerance),
            *residual_caps.items(),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"{name} must be a positive finite physical residual tolerance")
            with np.errstate(over="ignore", under="ignore"):
                representable = self._numpy_dtype(value)
            if not np.isfinite(representable) or representable <= 0:
                raise ValueError(f"{name} must be representable in the selected precision")
        self._rigid = RigidObjective(model, self._actors, dtype=self._dtype)
        if smooth_sdf is not None:
            smooth_sdf = smooth_sdf.for_precision(self._dtype)
        if publish_policy not in ("strict", "last_accepted", "current_iterate"):
            raise ValueError("Unknown coupled publication policy")
        self._coupled_publish_policy = publish_policy
        self._stopped_failure = None
        if not np.isfinite(min_det_f_guard) or min_det_f_guard <= 0:
            raise ValueError("Invalid detF guard")
        if self._materials is not None:
            self._use_cuda_graph = use_cuda_graph
            density = "none"
            if model.tet_count and mass_mode == self.MassMode.CONSISTENT:
                if not isinstance(tet_rest_density, wp.array) or (
                    tet_rest_density.shape,
                    tet_rest_density.dtype,
                    tet_rest_density.device,
                ) != ((model.tet_count,), wp.float32, model.device):
                    raise ValueError("consistent mass requires same-device float32 per-tet density")
                density = tet_rest_density.numpy().copy()
                if not np.isfinite(density).all() or np.any(density <= 0):
                    raise ValueError("tet density must be finite and positive")
            elif tet_rest_density is not None:
                raise ValueError("tet_rest_density requires a tet with consistent mass")
            if type(use_cuda_graph) is not bool or (
                use_cuda_graph and (self.PCGMode(pcg_mode) != self.PCGMode.PRODUCTION or not model.device.is_cuda)
            ):
                raise ValueError("use_cuda_graph requires CUDA production PCG and a bool")
            if contact_factor_capacity is not None and (
                type(contact_factor_capacity) is not int or contact_factor_capacity < 0
            ):
                raise ValueError("Contact factor capacity must be a nonnegative integer")
            self._material_smooth_sdf = smooth_sdf
            self._initialize_material_lifecycle(
                model,
                joint_terms=joint_terms,
                config={
                    "contact_model": CONTACT_MODEL,
                    "globalization": GLOBALIZATION,
                    "contact_stiffness": contact_stiffness,
                    "friction_coefficient": friction_coefficient,
                    "friction_velocity_scale": friction_velocity_scale,
                    "normal_smoothing": normal_smoothing,
                    "material_model": self.MaterialModel(material_model).value,
                    "mass_mode": self.MassMode(mass_mode).value,
                    "min_det_f_guard": min_det_f_guard,
                    "publish_policy": publish_policy,
                    "alignment_policy": alignment_policy,
                    "allow_nonconservative_energy": True,
                    "newton_max_iterations": newton_max_iterations,
                    "line_search_max_iterations": line_search_max_iterations,
                    "linear_max_iterations": linear_max_iterations,
                    "linear_tolerance": linear_tolerance,
                    "linear_absolute_tolerance": linear_absolute_tolerance,
                    "cable_position_tolerance": cable_position_tolerance,
                    "cable_theta_tolerance": cable_theta_tolerance,
                    **residual_caps,
                    "pcg_mode": self.PCGMode(pcg_mode).value,
                    "tet_rest_density": density,
                    "contact_factor_capacity": "automatic"
                    if contact_factor_capacity is None
                    else contact_factor_capacity,
                    "contact_candidate_capacity": "automatic"
                    if contact_candidate_capacity is None
                    else contact_candidate_capacity,
                    "smooth_sdf": "none" if smooth_sdf is None else smooth_sdf.digest,
                },
            )
            return
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
        dynamic_counts = np.count_nonzero(self._tet.mapping.numpy()[model.tet_indices.numpy()] >= 0, axis=1)
        base_triplets = (
            layout.q_dof_count**2
            + 36 * self._rigid.count
            + 9 * (layout.dynamic_particle_count + sum(int(k) ** 2 for k in dynamic_counts))
        )
        self._contact = ContactObjective(
            model,
            self._tet.mapping.numpy(),
            q_dof_count=self._articulation_actor.dof_count,
            layout=layout,
            articulation_actor=self._articulation_actor,
            actors=self._actors,
            base_triplet_count=base_triplets,
            normal_stiffness=contact_stiffness,
            friction_coefficient=friction_coefficient,
            friction_velocity_scale=friction_velocity_scale,
            smoothing_width=normal_smoothing,
            particle_radius=float(radius[0]),
            smooth_sdf=smooth_sdf,
            alignment_policy=alignment_policy,
            dtype=self._dtype,
            **({} if contact_candidate_capacity is None else {"candidate_capacity": contact_candidate_capacity}),
        )
        capacity = 3 * self._contact.count if contact_factor_capacity is None else contact_factor_capacity
        if type(capacity) is not int or capacity < 0:
            raise ValueError("Contact factor capacity must be a nonnegative integer")
        if not isinstance(self._contact, _PooledContactObjective):
            raise ValueError("Monolithic contact requires the pooled contact objective")
        self._contact_pool = True
        self._contact_spatial = isinstance(self._contact, _SpatialContactObjective)
        self._contact.initialize_pool(capacity, use_cuda_graph=use_cuda_graph)
        self._guard = min_det_f_guard
        physics_sources = dict(self._tet.physics.sources)
        for name in (
            "particle_radius",
            "particle_flags",
            "tet_activations",
            "tri_materials",
            "edge_bending_properties",
        ):
            if getattr(model, name) is not None:
                physics_sources[name] = getattr(model, name)
        self._physics_values = {name: value.numpy().copy() for name, value in physics_sources.items()}
        self._sdf_fields = self._freeze_sdf(model)
        self._coupled_config = {
            "contact_parameters": np.asarray(self._contact.parameters, dtype=float),
            "contact_model": np.frombuffer(CONTACT_MODEL.encode(), dtype=np.uint8),
            "globalization": np.frombuffer(GLOBALIZATION.encode(), dtype=np.uint8),
            "material": np.frombuffer(material_model.encode(), dtype=np.uint8),
            "mass": np.frombuffer(mass_mode.encode(), dtype=np.uint8),
            "guard_capacity": np.array([min_det_f_guard, capacity]),
            "physics_digest": np.frombuffer(bytes.fromhex(self._tet.physics.identity), dtype=np.uint8),
            "publication_policy": np.frombuffer(publish_policy.encode(), dtype=np.uint8),
            "alignment_policy": np.frombuffer(alignment_policy.encode(), dtype=np.uint8),
            "heuristic_energy": np.array([True], dtype=np.uint8),
        }
        self._coupled_config["contact_dataflow"] = self._contact.dataflow_identity
        self._coupled_config["contact_resources"] = self._contact.resource_identity
        if self._contact_spatial:
            self._coupled_config["contact_spatial"] = self._contact.spatial_identity
        if self._rigid.count:
            self._coupled_config.update(
                rigid_contact_pairs=self._rigid.values["shape_contact_pairs"],
                rigid_policy=np.array([1e-7, 1e-6, 2.5e-8, 1e-5, 1e-10, 9.100770221088748e-10, 1]),
            )
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
        self._coupled_config["direction_policy"] = np.array([1], dtype=np.int32)
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
        self._joint_terms = JointParameters(model, actor=self._articulation_actor, **options)
        for name in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_friction"):
            if name not in self._joint_terms.allowed_properties and np.any(
                getattr(model, name).numpy()[list(self._articulation_actor.dof_indices)] != 0
            ):
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
        self._joint_parameters_host = parameters.astype(self._numpy_dtype)
        self._dynamic_ids_host = self._tet.ids.numpy().copy()
        self._input_status = wp.zeros(4, dtype=int, device=model.device)
        self._config = _SolverMonolithicInternalConfig()
        # Sharpa refinement certifies these tighter goals within the existing budgets.
        self._config = replace(
            self._config,
            merit_absolute_global=self._config.merit_absolute_global * 0.1,
            merit_absolute_q=self._config.merit_absolute_q * 0.1,
            merit_relative_global=self._config.merit_relative_global * 0.1,
            merit_relative_q=self._config.merit_relative_q * 0.1,
        )
        self._globalization = GLOBALIZATION
        self.newton_max_iterations, self.line_search_max_iterations = newton_max_iterations, line_search_max_iterations
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance, self.linear_absolute_tolerance = linear_tolerance, linear_absolute_tolerance
        self._pcg_mode = self.PCGMode(pcg_mode)
        self._use_cuda_graph = use_cuda_graph
        device = model.device
        self._graphs = _SegmentGraphCache(device, enabled=use_cuda_graph)
        self._states._graphs = self._graphs
        self._force_status = wp.zeros(3, dtype=int, device=device)
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
        n, nq, device = layout.dynamic_particle_count, layout.q_dof_count, model.device
        pattern = build_tet_triplet_pattern(
            model.tet_indices.numpy(), self._tet.mapping.numpy(), x_dof_start=layout.x_scalar_offset
        )
        self._tet_assembly = create_tet_assembly_workspace(
            pattern, tet_count=model.tet_count, device=device, physics=self._tet.physics
        )
        self._linear = MonolithicLinearWorkspace(
            layout,
            MonolithicLinearCapacities(
                nq * nq + len(pattern.global_rows) + 36 * self._rigid.count + self._contact.triplet_capacity,
                len(pattern.ax_rows),
                0,
                self.linear_max_iterations,
                self.linear_max_iterations + 2,
            ),
            self._tet.mapping,
            device=device,
            pcg_mode=self.pcg_mode.value,
            use_cuda_graph=self._use_cuda_graph,
            contact_support=None,
        )
        rigid_rows = np.repeat(self._rigid.rows_numpy, 6, axis=1).ravel()
        rigid_columns = np.tile(self._rigid.rows_numpy, (1, 6)).ravel()
        self._rigid_triplet_offset = nq * nq + len(pattern.global_rows)
        self._linear._set_fixed_triplet_pattern(
            global_rows=np.r_[np.repeat(layout.indices("articulation"), nq), pattern.global_rows, rigid_rows],
            global_columns=np.r_[np.tile(layout.indices("articulation"), nq), pattern.global_columns, rigid_columns],
            internal_rows=pattern.ax_rows,
            internal_columns=pattern.ax_columns,
        )
        self._linear._enable_fixed_bsr_storage()
        for name in ("_rhs", "_y", "_delta", "_diagonal"):
            setattr(self, name, wp.zeros(layout.scalar_dof_count, dtype=float, device=device))
        self._complete_r = wp.zeros(layout.scalar_dof_count, dtype=self._dtype, device=device)
        self._post_r, self._post_error = wp.zeros_like(self._complete_r), wp.zeros_like(self._complete_r)
        self._post_original, self._post_regularization = (
            wp.zeros_like(self._complete_r),
            wp.zeros_like(self._complete_r),
        )
        self._post_lambda = wp.zeros(1, dtype=float, device=model.device)
        self._correction_rhs, self._correction_delta = wp.zeros_like(self._rhs), wp.zeros_like(self._delta)
        self._direction_base = wp.zeros_like(self._delta)
        self._direction_status = wp.zeros(1, dtype=int, device=model.device)
        self._tet_r32 = wp.zeros(n, dtype=wp.vec3, device=device)
        self._frozen_force = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)
        self._frozen_gravity = wp.zeros_like(model.gravity)
        self._out_backup = None
        self._dt = None
        # P6a reductions assume contiguous A -> T with no rigid blocks.
        self._device_iteration = (
            not self._rigid.count
            and self._layout.q_scalar_offset == 0
            and self._layout.x_scalar_offset == self._kinematics.count
        )
        iteration_device.initialize(self)

    def _initialize_material_lifecycle(self, model, *, joint_terms, config):
        """Allocate shared material layout and continuation before lazy operator setup."""
        self._material_config = config
        if config["alignment_policy"] not in ("guard", "mask"):
            raise ValueError("Unknown material solver policy")
        for name in ("newton_max_iterations", "line_search_max_iterations", "linear_max_iterations"):
            if type(config[name]) is not int or config[name] <= 0:
                raise ValueError("Iteration budgets must be positive integers")
        for name in ("linear_tolerance", "linear_absolute_tolerance", "friction_coefficient"):
            if isinstance(config[name], bool) or not np.isfinite(config[name]) or config[name] < 0:
                raise ValueError(f"Invalid {name}")
        for name in ("contact_stiffness", "friction_velocity_scale", "normal_smoothing"):
            if isinstance(config[name], bool) or not np.isfinite(config[name]) or config[name] <= 0:
                raise ValueError(f"Invalid {name}")
        joint_terms = self.JointTerms() if joint_terms is None else joint_terms
        if not isinstance(joint_terms, self.JointTerms):
            raise ValueError("joint_terms must be SolverMonolithic.JointTerms")
        # An empty kinematics view owns no Model bodies/joints and adds no actor.
        view = self._articulation_actor or Actor("articulation", 0)
        self._joint_terms = JointParameters(
            model, actor=view, **{name: getattr(joint_terms, name) for name in joint_terms.__dataclass_fields__}
        )
        for name in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_friction"):
            if name not in self._joint_terms.allowed_properties and np.any(
                getattr(model, name).numpy()[list(view.dof_indices)] != 0
            ):
                raise ValueError(f"Disabled joint term has nonzero parameters: {name}")
        fields = {
            "material_iteration_policy": np.array([2], dtype=np.int32),
            "material_digest": np.frombuffer(bytes.fromhex(self._materials.identity), dtype=np.uint8),
            "layout_digest": np.frombuffer(bytes.fromhex(self._actors.identity), dtype=np.uint8),
            "joint_terms": self._joint_terms.params.numpy(),
        }
        for name, value in config.items():
            fields[name] = (
                np.frombuffer(value.encode(), dtype=np.uint8) if isinstance(value, str) else np.asarray(value)
            )
        for name in (
            "tet_indices",
            "tet_poses",
            "tet_materials",
            "particle_mass",
            "particle_inv_mass",
            "particle_flags",
            "particle_q",
            "particle_radius",
            "shape_transform",
            "shape_type",
            "shape_scale",
            "shape_margin",
            "shape_body",
            "shape_contact_pairs",
            "_shape_sdf_index",
        ):
            value = getattr(model, name, None)
            if value is not None:
                fields[name] = value.numpy()
        self._contact = material_runtime.create_contact(self, model, config, self._material_smooth_sdf)
        fields.update(self._contact.actor_samples.fields)
        fields["material_contact.backend"] = np.frombuffer(type(self._contact).__name__.encode(), dtype=np.uint8)
        if isinstance(self._contact, _SpatialContactObjective):
            fields["material_contact.spatial"] = np.array([self._contact.candidate_capacity, 0.0])
        fields.update(self._freeze_sdf(model))
        self._kinematics = _TreeKinematics(model, actor=view, extra_config=fields, dtype=self._dtype)
        SolverBase.__init__(self, model)
        self._states = CoupledState(self._kinematics, self._actors, self._rigid, self._materials)
        self._last_stats = self.Stats(self.Status.NOT_RUN)
        self._pcg_mode = self.PCGMode(config["pcg_mode"])
        for name in (
            "newton_max_iterations",
            "line_search_max_iterations",
            "linear_max_iterations",
            "linear_tolerance",
            "linear_absolute_tolerance",
        ):
            setattr(self, name, config[name])

    def create_control(self):
        """Create experimental cable position [m], angle [rad] and torque [N m] controls.

        Position/angle targets apply only to their fixed vertices/edges; torque
        is supported only on physically free edges. The ordinary Model joint controls are retained.
        """
        control = self.model.control()
        if self._materials is not None:
            if not self.model.joint_count:
                for name in ("joint_f", "joint_target_q", "joint_target_qd"):
                    setattr(control, name, wp.zeros(0, dtype=float, device=self.model.device))
            namespace = Model.AttributeNamespace("_monolithic_cable")
            theta = [obj["rest_theta"] for obj in self._materials.cable]
            namespace.target_theta = wp.array(
                np.concatenate(theta) if theta else [], dtype=self._dtype, device=self.model.device
            )
            namespace.torque = wp.zeros(self._materials.theta_count, dtype=self._dtype, device=self.model.device)
            namespace.target_position = wp.array(
                self.model.particle_q.numpy(), dtype=self._states.vector, device=self.model.device
            )
            control._monolithic_cable = namespace
        return control

    def _snapshot_material_control(self, control):
        namespace = getattr(control, "_monolithic_cable", None)
        values = {}
        for name in ("target_theta", "torque"):
            array = getattr(namespace, name, None)
            if not isinstance(array, wp.array) or (array.shape, array.dtype, array.device) != (
                (self._materials.theta_count,),
                self._dtype,
                self.model.device,
            ):
                raise ValueError(f"Invalid cable control: {name}")
            values[name] = array.numpy().copy()
            if not np.isfinite(values[name]).all():
                raise ValueError(f"Nonfinite cable control: {name}")
        for obj in self._materials.cable:
            offset = int(obj["theta_offset"])
            part = slice(offset, offset + len(obj["rest_theta"]))
            if np.any(values["torque"][part][obj["fixed_theta"]] != 0):
                raise ValueError("External cable torques require physically free theta")
        array = getattr(namespace, "target_position", None)
        if (
            not isinstance(array, wp.array)
            or (array.shape, array.dtype, array.device)
            != (
                (self.model.particle_count,),
                self._states.vector,
                self.model.device,
            )
            or not np.isfinite(array.numpy()).all()
        ):
            raise ValueError("Invalid cable control: target_position")
        values["target_position"] = array.numpy().copy()
        return values

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
                fields[f"{name}.{i}"] = pixels
        for i, slots in enumerate(model._texture_sdf_subgrid_start_slots):
            fields[f"sdf.slots.{i}"] = slots.numpy().copy()
        return fields

    def _make_kinematics(self, model, *, extra_config):
        fields = dict(extra_config)
        fields.update(self._physics_values)
        fields.update(self._contact.values)
        if self._contact.actor_samples is not None:
            fields.update(self._contact.actor_samples.fields)
        fields.update(self._coupled_config)
        fields.update(self._sdf_fields)
        return _TreeKinematics(model, actor=self._articulation_actor, extra_config=fields, dtype=self._dtype)

    def _make_states(self):
        return CoupledState(self._kinematics, self._actors, self._rigid)

    def create_state(self, *, dt, source=None):
        """Initialize full integration history with positive dt, optionally from ordinary model state."""
        return self._states.create(dt=dt, source=source)

    def reset_state(self, state, *, dt):
        """Reinitialize history at current coordinates and clear published contact forces."""
        backup = self._states._copy_unchecked(state)
        try:
            self._states.reset(state, dt=dt)
        except Exception:
            state.assign(backup)
            raise

    def save_checkpoint(self, state, path):
        """Atomically save complete NPZ state, appending .npz as in NumPy's path API."""
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
        backup = self._states.copy(state)
        try:
            self._states.load_checkpoint(state, path)
        except Exception:
            state.assign(backup)
            raise

    def get_contact_forces(self, state):
        """Return an independent, read-only snapshot of this state's final contact forces."""
        history = self._states.validate(state)
        return MonolithicContactForces(
            history.contact_particle_force.numpy(),
            history.contact_body_force.numpy(),
            history.contact_joint_force.numpy(),
            bool(history.force_valid.numpy()[0]),
            int(history.force_generation.numpy()[0]),
            cable_torque=history.contact_cable_torque.numpy() if self._materials is not None else None,
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
        """Guard solver-owned storage without rereading immutable Model inputs."""
        if self._layout != self._linear.layout or self._layout.identity != self._actors.identity:
            raise ValueError("Actor layout changed; rebuild solver")
        for name, source in self._rigid._maps.items():
            if getattr(self._rigid, name) is not source:
                raise ValueError(f"Rigid mapping changed: {name}")

    def step(self, state_in, state_out, control, contacts=None, dt=0.0):
        if self._materials is not None:
            if not hasattr(self, "_cables"):
                material_runtime.initialize(self)
            inputs = self._snapshot_material_control(control)
            self._cable_target_x.assign(inputs["target_position"])
            self._cable_target_theta.assign(inputs["target_theta"])
            self._cable_torque.assign(inputs["torque"])
        self._validate_components()
        self._stopped_failure = None
        if contacts is not None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Articulation step requires no contacts and finite positive dt")
        dt = self._kinematics.timestep(dt)
        self._states.validate(state_in)
        if state_out is not state_in:
            self._states.validate(state_out)
        for name in ("joint_f", "joint_target_q", "joint_target_qd"):
            array = getattr(control, name, None)
            if not isinstance(array, wp.array) or (array.shape, array.dtype, array.device) != (
                (
                    self.model.joint_target_q.size
                    if name == "joint_target_q" and self.model.joint_count
                    else self.model.joint_dof_count,
                ),
                wp.float32,
                self.model.device,
            ):
                raise ValueError(f"Invalid articulation control: {name}")
        k, obj = self._kinematics, self._objective
        self._input_status.zero_()
        wp.launch(
            kernels.freeze_step_forces,
            max(self.model.joint_dof_count, control.joint_target_q.size, k.bodies, self.model.gravity.size),
            [
                state_in.body_f,
                self.model.gravity,
                k.mass,
                self.model.body_world,
                k.body_map,
                k.dof_map,
                control.joint_f,
                control.joint_target_q,
                control.joint_target_qd,
                obj.force,
                obj.joint_force,
                self._input_status,
            ],
            device=self.model.device,
        )
        input_status = self._input_status.numpy()
        for index, name in enumerate(("joint_f", "joint_target_q", "joint_target_qd")):
            if input_status[index]:
                raise ValueError(f"Invalid articulation control: {name}")
        self._joint_terms.snapshot_targets(control)
        if input_status[3]:
            raise ValueError("Articulation requires finite forces and zero spatial torque")
        self._rigid.prepare(state_in, control, dt)
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
        k.evaluate(k.gather_q(state_in))
        wp.copy(obj.tn, k.com)
        wp.copy(obj.rn, k.rotations)
        wp.copy(self._old_joint_t, k.joint_translations)
        wp.copy(self._old_joint_r, k.joint_rotations)
        wp.copy(obj.qn, k.gather_q(state_in))
        wp.copy(obj.vn, k.gather_qd(state_in))
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
        wp.utils.array_cast(self._joint_terms.target_q, obj.qt)
        wp.utils.array_cast(self._joint_terms.target_qd, obj.vt)
        wp.launch(
            kernels._coordinate_predictor,
            k.count,
            [obj.qn, obj.vn, self._dtype(dt), k.coord_map, k.dof_map, self._accepted.joint_q, self._accepted.joint_qd],
            device=self.model.device,
        )
        if self._rigid.count:
            wp.copy(self._accepted._monolithic_coupled.rigid_com, self._rigid.predicted_t)
        k.evaluate(k.gather_q(self._accepted))
        if not self._domain(self._accepted):
            if self._rigid.count:
                self._accepted.assign(self._original)
            else:
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
                k.evaluate(k.gather_q(self._accepted), state=self._accepted)
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
                self._variational.pop("actor_contact_pairs", None)
            if isinstance(self._variational.get("residual"), wp.array):
                self._variational["residual"] = self._variational["residual"].numpy().tolist()
            self._variational["graphs"] = self._graphs.summary()
            if self._contact_pool:
                self._variational["contact_graphs"] = {
                    "batch": self._contact._batch.graphs.summary(),
                    "pool": self._contact.pool.graphs.summary(),
                    "bsr": self._linear._fixed_bsr[0].graphs.summary(),
                    "internal": self._linear._fixed_bsr[1].graphs.summary(),
                }
            if self._contact_spatial:
                self._variational["contact_spatial"] = self._contact.spatial_diagnostics()
            self._last_stats = self.Stats(
                status=status,
                failure_reason=failure,
                converged=status == self.Status.SUCCESS and published,
                rolled_back=not published,
                step_generation=self._step_generation,
                preconditioner_kind="actor_block",
                q_block_count=int(k.count > 0),
                variational=MappingProxyType(self._variational.copy()),
                **self._metrics,
            )
        stats = self._last_stats
        stopped = not stats.converged and not stats.rolled_back
        detail = dict(stats.variational)
        detail.update(
            publication_status="STOPPED_ACCEPTED" if stopped else "CONVERGED" if stats.converged else "HARD_FAILURE",
            publish_policy=self._coupled_publish_policy,
            alignment_policy=self._material_config["alignment_policy"]
            if self._materials is not None
            else self._contact.alignment_policy,
            energy_acceptance="variational" if self._conservative_energy() else "heuristic",
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
                    if "convergence_ratio" + suffix in self._metrics
                },
            }
            if self._materials is not None:
                detail["stopped_candidate"]["final_ratios"].update(
                    {
                        name: values["convergence_ratio_" + name]
                        for name, values in detail.get("material_blocks", {}).items()
                    }
                )
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
        if self._materials is not None:
            wp.launch(
                kernels.particle_predict,
                self.model.particle_count,
                [
                    state_in.particle_q,
                    state_in.particle_qd,
                    self._particle_mapping,
                    self._dtype(dt),
                    self._accepted.particle_q,
                    self._accepted.particle_qd,
                ],
                device=self.model.device,
            )
            for cloth in self._cloths:
                cloth.begin_step(self._original, self._frozen_force)
                cloth.candidate(self._original, self._original, self._accepted, self._delta, 0.0, dt, predict=True)
                if not cloth.path_valid(self._original, self._accepted):
                    # An inertial guess is optional; retain the valid input chart.
                    cloth.candidate(self._original, self._original, self._accepted, self._delta, 0.0, dt)
            for cable in self._cables:
                cable.begin_step(self._original, self._frozen_force)
                cable.candidate(
                    self._original,
                    self._original,
                    self._accepted,
                    self._delta,
                    self._cable_target_x,
                    self._cable_target_theta,
                    0.0,
                    dt,
                    predict=True,
                )
            if self._contact_pool:
                k = self._kinematics
                k.evaluate(k.gather_q(state_in))
                h = state_in._monolithic_coupled
                self._contact._batch.material_surface.update(state_in, begin=True)
                self._contact.begin_step(
                    state_in.particle_q,
                    k.poses,
                    k.com,
                    k.jt,
                    k.jw,
                    rigid_com=h.rigid_com,
                    rigid_rotation=h.rigid_rotation,
                    **({"kinematics": k, "q0": k.gather_q(state_in)} if self._contact_spatial else {}),
                )
            return
        k = self._kinematics
        k.evaluate(k.gather_q(state_in))
        history = state_in._monolithic_coupled
        self._contact.begin_step(
            state_in.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            rigid_com=history.rigid_com,
            rigid_rotation=history.rigid_rotation,
            **({"kinematics": k, "q0": k.gather_q(state_in)} if self._contact_spatial else {}),
        )
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

    def _evaluate(self, state, dt, *, build_tangent=True, device_result=False):
        if self._materials is not None:
            return material_runtime.evaluate(self, state, dt, build_tangent=build_tangent)
        k, contact, tet = self._kinematics, self._contact, self._tet
        local_q = k.gather_q(state)
        buffers = [
            state.joint_q,
            state.particle_q,
            self._original.particle_q,
            self._original.particle_qd,
            self._frozen_force,
            self._frozen_gravity,
            self._complete_r,
        ]
        if device_result:
            buffers.extend((self._energy_terms, self._energy_sums, self._evaluation_status, self._evaluation_packet))
        else:
            buffers.append(self._host_evaluation_packet)
        rigid_history = state._monolithic_coupled
        buffers.extend([rigid_history.rigid_com, rigid_history.rigid_rotation])
        for owner in (self._objective, k, tet, self._rigid):
            buffers.extend(value for value in vars(owner).values() if isinstance(value, wp.array))

        def evaluate_owners():
            self._objective.evaluate(local_q, dt, build_tangent=build_tangent)
            self._rigid.evaluate(state, dt, build_tangent)
            tet.evaluate(
                state.particle_q,
                self._original.particle_q,
                self._original.particle_qd,
                self._frozen_force,
                self._frozen_gravity,
                dt=dt,
                min_det_f_guard=self._guard,
            )

        def evaluate_contact():
            contact.evaluate(
                state.particle_q,
                k.poses,
                k.com,
                k.jt,
                k.jw,
                dt=dt,
                rigid_com=rigid_history.rigid_com,
                rigid_rotation=rigid_history.rigid_rotation,
            )

        def evaluate_result():
            wp.launch(
                kernels.combine,
                self._complete_r.size,
                [
                    self._objective.residual,
                    tet.residual,
                    contact.residual,
                    self._layout.q_scalar_offset,
                    self._layout.x_scalar_offset,
                    self._complete_r,
                ],
                device=self.model.device,
            )
            wp.launch(
                _scatter_residual,
                (self._rigid.count, 6),
                [self._rigid.rows, self._rigid.residual, self._complete_r],
                device=self.model.device,
            )
            if device_result:
                iteration_device.evaluate(self)
            else:
                wp.launch(
                    iteration_device.host_evaluation_packet,
                    self._host_evaluation_packet.size,
                    [
                        self._rigid.energy,
                        self._objective.link_energy,
                        self._objective.joint_energy,
                        tet.parts,
                        contact.energy.flatten(),
                        contact.absolute_energy,
                        tet.flags,
                        contact.flags,
                        tet.min_det_f,
                        self._contact_pool,
                        self._host_evaluation_packet,
                    ],
                    device=self.model.device,
                )

        reference_mode = self._contact_spatial and contact._reference_mode
        key = (self._layout.identity, self.precision.value, dt, build_tangent, device_result, reference_mode)
        self._graphs.run("coupled_objective_owners", key, buffers, evaluate_owners)
        # Pool begin/append have side effects; they must run once, never
        # inside the owner's warmup/capture/replay overwrite chain.
        evaluate_contact()
        result_buffers = (
            self._objective.residual,
            tet.residual,
            contact.residual,
            self._complete_r,
            self._rigid.rows,
            self._rigid.residual,
            self._rigid.energy,
            self._objective.link_energy,
            self._objective.joint_energy,
            tet.parts,
            contact.energy,
            contact.absolute_energy,
            tet.flags,
            contact.flags,
            tet.min_det_f,
            self._energy_terms,
            self._energy_sums,
            self._evaluation_status,
            self._evaluation_packet,
            self._host_evaluation_packet,
        )
        self._graphs.run("coupled_objective_result", key, result_buffers, evaluate_result)
        if not reference_mode:
            self._variational["contact_pool"] = contact.evaluation_diagnostics()
        self._current_state = state
        if device_result:
            packet = self._evaluation_packet.numpy()
            if packet[0]:
                status = self.Status.TET_INVERSION if packet[0] == 1 else self.Status.NONFINITE
                raise _StepFailure(status, "Invalid tet candidate")
            if packet[1]:
                raise _StepFailure(self.Status.NONFINITE, f"Invalid contact geometry: {[int(packet[1])]}")
            contact._evaluated = True
            if packet[2] or not np.isfinite(packet[3]):
                raise _StepFailure(self.Status.NONFINITE, "Nonfinite coupled E/r")
            self._metrics["min_det_f"] = float(packet[5])
            return packet[3], self._complete_r, packet[4]
        packet = self._host_evaluation_packet.numpy()
        if packet[0]:
            status = self.Status.TET_INVERSION if packet[0] == 1 else self.Status.NONFINITE
            raise _StepFailure(status, "Invalid tet candidate")
        if packet[1]:
            # The offending contact identity is int32 and may not fit FP32 exactly.
            raise _StepFailure(self.Status.NONFINITE, f"Invalid contact geometry: {contact.flags.numpy().tolist()}")
        # Graph replay executes device work, not evaluate()'s host-side marker.
        contact._evaluated = True
        residual = self._complete_r.numpy().copy()
        parts = packet[5:]
        if not np.isfinite(residual).all() or not np.isfinite(parts).all() or not np.isfinite(parts.sum()):
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite coupled E/r")
        self._metrics["min_det_f"] = float(packet[2])
        absolute = abs(parts).sum()
        if self._contact_pool:
            absolute = abs(parts[: -contact.energy.size]).sum() + packet[3:5].sum()
        return parts.sum(), residual, absolute

    def _initial_reference(self, dt, *, device_result=False):
        if self._contact_spatial:
            self._contact._reference_mode = True
        try:
            residual = self._evaluate(self._original, dt, build_tangent=False, device_result=device_result)[1]
        finally:
            if self._contact_spatial:
                self._contact._reference_mode = self._contact._evaluated = False
        if device_result:
            wp.copy(self._reference_device, residual)
            return self._reference_device
        return residual

    def _refresh_candidates(self):
        """Refresh before the current E/r/B/baseline, including the budget's final check."""
        state, k = self._accepted, self._kinematics
        try:
            self._contact.refresh(k.gather_q(state), state.particle_q, rigid_com=state._monolithic_coupled.rigid_com)
        except ValueError as error:
            raise _StepFailure(self.Status.NONFINITE, str(error)) from error

    def _conservative_energy(self):
        if self._materials is not None and not self._contact_pool:
            return not self._rigid.has_torque
        return self._contact.conservative_energy and not self._rigid.has_torque

    def _trial_evaluation(self, dt, *, device_result=False):
        try:
            return self._evaluate(self._trial, dt, build_tangent=False, device_result=device_result)
        except _StepFailure as error:
            self._variational["trials"].append({"accepted": False, "domain_failure": str(error)})
            return None

    def _take_trial(self, alpha, dt):
        direction = self._native_delta if self._materials is not None else self._delta
        wp.launch(
            kernels._coordinate_trial,
            self._kinematics.count,
            [
                self._accepted.joint_q,
                direction,
                self._layout.q_scalar_offset,
                self._dtype(alpha),
                self._objective.qn,
                self._dtype(dt),
                self._kinematics.coord_map,
                self._kinematics.dof_map,
                self._trial.joint_q,
                self._trial.joint_qd,
            ],
            device=self.model.device,
        )
        self._rigid.trial(self._accepted, self._trial, direction, alpha)
        wp.launch(
            kernels.particle_trial,
            self.model.particle_count,
            [
                self._accepted.particle_q,
                self._original.particle_q,
                self._particle_mapping if self._materials is not None else self._tet.mapping,
                direction,
                self._layout.x_scalar_offset,
                self._dtype(alpha),
                self._dtype(dt),
                self._trial.particle_q,
                self._trial.particle_qd,
            ],
            device=self.model.device,
        )

        if self._materials is not None:
            for cloth in self._cloths:
                cloth.candidate(self._accepted, self._original, self._trial, direction, alpha, dt)
            for cable in self._cables:
                cable.candidate(
                    self._accepted,
                    self._original,
                    self._trial,
                    direction,
                    self._cable_target_x,
                    self._cable_target_theta,
                    alpha,
                    dt,
                )

    def _effective_displacement(self, *, device_result=False):
        if device_result:
            wp.launch(
                iteration_device.effective_displacement,
                self._effective_device.size,
                [
                    self._trial.joint_q,
                    self._accepted.joint_q,
                    self._trial.particle_q,
                    self._accepted.particle_q,
                    self._tet.ids,
                    self._effective_device,
                ],
                device=self.model.device,
            )
            return self._effective_device
        ids = self._dynamic_ids_host
        displacement = np.empty(self._layout.scalar_dof_count, dtype=self._numpy_dtype)
        coords = [] if self._articulation_actor is None else list(self._articulation_actor.coord_indices)
        displacement[self._layout.indices("articulation")] = (
            self._trial.joint_q.numpy()[coords] - self._accepted.joint_q.numpy()[coords]
        )
        displacement[self._layout.indices("tet")] = (
            self._trial.particle_q.numpy()[ids] - self._accepted.particle_q.numpy()[ids]
        ).ravel()
        if self._materials is not None:
            ax, bx = self._accepted.particle_q.numpy(), self._trial.particle_q.numpy()
            at, bt = (
                self._accepted._monolithic_coupled.cable_theta.numpy(),
                self._trial._monolithic_coupled.cable_theta.numpy(),
            )
            for cloth in self._cloths:
                local = (bx[cloth.actor.particle_indices, :] - ax[cloth.actor.particle_indices, :]).ravel()
                free = cloth.rows_host >= 0
                displacement[cloth.rows_host[free]] = local[free]
            for cable in self._cables:
                local = np.r_[
                    (bx[cable.actor.particle_indices, :] - ax[cable.actor.particle_indices, :]).ravel(),
                    bt[list(cable.actor.theta_indices)] - at[list(cable.actor.theta_indices)],
                ]
                free = cable.rows_host >= 0
                displacement[cable.rows_host[free]] = local[free]
        if self._rigid.count:
            a, b = self._accepted._monolithic_coupled, self._trial._monolithic_coupled
            displacement[self._rigid.rows_numpy[:, :3]] = b.rigid_com.numpy() - a.rigid_com.numpy()
            # Each trial is exactly the left Exp(alpha*delta) retraction within the local domain.
            qa, qb = a.rigid_rotation.numpy(), b.rigid_rotation.numpy()
            vector = qb[:, 3:] * -qa[:, :3] + qa[:, 3:] * qb[:, :3] + np.cross(qb[:, :3], -qa[:, :3])
            scalar = np.sum(qb * qa, axis=1)
            sign = np.where(scalar < 0, -1, 1)
            vector *= sign[:, None]
            length = np.linalg.norm(vector, axis=1)
            factor = np.divide(
                2 * np.arctan2(length, abs(scalar)), length, out=np.full_like(length, 2), where=length > 0
            )
            displacement[self._rigid.rows_numpy[:, 3:]] = vector * factor[:, None]
        return displacement

    def _assemble(self, iteration, *, device_result=False):
        if self._materials is not None:
            return material_runtime.assemble(self, iteration)
        k, linear, obj = self._kinematics, self._linear, self._objective
        self._assembly_sequence += 1
        self._generation = MonolithicLinearGeneration(
            self._step_generation,
            iteration,
            self._contact.evaluation_generation if self._contact_pool else self._step_generation,
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
        wp.launch(
            _scatter_tangent,
            (self._rigid.count, 6, 6),
            [
                self._rigid.tangent,
                self._rigid.rows,
                self._rigid.diagonal,
                self._rigid_triplet_offset,
                assembly.global_scalar_triplets.values,
                self._diagonal,
            ],
            device=self.model.device,
        )
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
                self._layout.q_scalar_offset,
                self._layout.x_scalar_offset,
                self._diagonal,
            ],
            device=self.model.device,
        )
        self._require_linear(
            linear.build_scaling(self._diagonal, epsilon_d=self._config.epsilon_d, generation=self._generation)
        )
        self._metrics["matrix_assembly_count"] += 1
        return None, linear.scale if device_result else linear.scale.numpy().astype(self._numpy_dtype)

    def _physical_linear_error(self, matrix, direction, residual, scale, *, device_result=False):
        if isinstance(residual, wp.array):
            wp.copy(self._post_r, residual)
        else:
            self._post_r.assign(residual)
        b = self._linear.k_global_scalar_bsr
        self._post_lambda.fill_(self._linear._lambda)
        buffers = [
            b.offsets,
            b.columns,
            b.values,
            self._native_delta if self._materials is not None else self._delta,
            self._post_r,
            self._linear.scale,
            self._post_lambda,
            self._post_original,
            self._post_regularization,
            self._post_error,
        ]
        self._graphs.run(
            "coupled_posterior",
            (
                self._layout.identity,
                self.precision.value,
            ),
            buffers,
            lambda: wp.launch(kernels.sparse_posterior, self._post_error.size, buffers, device=self.model.device),
        )
        return self._post_error if device_result else self._post_error.numpy().copy()

    def _prepare_direction_correction(self):
        wp.launch(
            kernels.direction_correction_rhs,
            self._post_error.size,
            [self._post_error, self._correction_rhs],
            device=self.model.device,
        )

    def _combine_direction_correction(self):
        # A saved base makes warmup/capture/replay overwrite-only, rather than
        # adding the same correction twice on the first graph invocation.
        wp.copy(self._direction_base, self._delta)
        buffers = [self._direction_base, self._correction_delta, self._delta, self._direction_status]

        def execute():
            self._direction_status.zero_()
            wp.launch(kernels.combine_direction, self._delta.size, buffers, device=self.model.device)

        self._graphs.run("coupled_direction_correction", (), buffers, execute)
        status = self._direction_status.numpy()[0]
        if status == 2:
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite corrected direction")
        return status == 1

    def _validate_final(self, dt, status):
        device = self._device_iteration
        energy, residual, _ = self._evaluate(self._accepted, dt, device_result=device)
        if device:
            wp.copy(self._accepted_r, residual)
            residual = self._accepted_r
        _, scale = self._assemble(self._metrics["nonlinear_iterations"], device_result=device)
        values = measure(self, residual, self._reference_residual, scale)
        if device:
            wp.copy(self._stats_residual, residual)
        self._variational.update(objective_total=energy, residual=self._stats_residual if device else residual.tolist())
        # Objective/scaling boundaries already check every residual/scale entry on device.
        if not np.isfinite(energy):
            raise _StepFailure(self.Status.NONFINITE, "Final coupled candidate contains nonfinite E/r/scale")
        if not self._domain(self._accepted):
            raise _StepFailure(self.Status.NONFINITE, "Final coupled candidate left the local domain")
        if not self._can_publish(status) or (
            status == self.Status.SUCCESS and any(m > goal for m, _, goal, _ in values.values())
        ):
            raise _StepFailure(self.Status.NONLINEAR_STAGNATION, "Final coupled candidate failed strict physical gates")

    def _finish_candidate(self, state, dt):
        self._rigid.recover(state, dt)
        if self._materials is not None:
            self._variational["cloth_reactions"] = tuple(c.raw.numpy().tolist() for c in self._cloths)
            self._variational["cable_reactions"] = tuple(c.raw.numpy().tolist() for c in self._cables)
            state._monolithic_coupled.contact_cable_torque.zero_()
        contact = self._contact
        history = getattr(state, self._states.namespace)
        arrays = [history.contact_joint_force, history.contact_particle_force, history.contact_body_force]
        for array in arrays:
            array.zero_()
        if self._contact_pool:
            self._variational["actor_contact_pairs"] = contact.publish(state)
        for pair in self._variational.get("actor_contact_pairs", ()):
            pair["force_generation"] = self._step_generation
        self._force_status.zero_()
        if self._materials is not None:
            arrays.append(history.contact_cable_torque)
        for index, array in enumerate(arrays):
            values = state_checks.scalar_view(array)
            if values.size:
                wp.launch(
                    state_checks.finite,
                    values.size,
                    [values, self._force_status, min(index, 2)],
                    device=self.model.device,
                )
        if self._force_status.numpy().any():
            raise _StepFailure(self.Status.NONFINITE, "Final contact force is nonfinite")
        history.force_generation.fill_(self._step_generation)
        history.force_valid.fill_(1)
        self._variational["contact_model"] = contact.contact_model
        self._variational["contact_force_generation"] = self._step_generation
        self._variational["contact_keys"] = contact.count

    def _restore(self, state):
        state.assign(self._out_backup)

    def _domain(self, state):
        if self._materials is not None and not all(
            c.path_valid(self._accepted, state) for c in self._cloths + self._cables
        ):
            return False
        return iteration_device.domain(self, state, int(JointType.REVOLUTE)) and self._rigid.domain(state)

    def _joint_force_metrics(self, state):
        """Bind scalar force diagnostics to the successfully published state."""
        p = self._joint_parameters_host
        q, v = self._kinematics.gather_q(state).numpy(), self._kinematics.gather_qd(state).numpy()
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
