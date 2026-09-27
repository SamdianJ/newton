# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private PR3 q/x path, using the shared PR2 iteration and bounded sparse linear system."""

import inspect
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import warp as wp

from . import coupled_kernels as kernels
from .articulation_solver import _ArticulationSolver, _scatter
from .articulation_variational import _TreeKinematics
from .collision import _validate_and_collect_boundary_faces
from .contact_variational import ContactObjective, ContactStatus
from .coupled_state import CoupledState
from .linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearWorkspace,
)
from .solver_monolithic import _StepFailure
from .tet import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from .tet_precision import TetObjective64
from .variational_iteration import measure


class _CoupledSolver(_ArticulationSolver):
    """Keep strict defaults; opt in to masked contact and last-accepted publication.

    Nonconservative energy acceptance is heuristic, not a variational guarantee.
    Stopped publication retains all physical/state gates except the residual goal.
    Force validity describes state/generation consistency, not equilibrium.
    """

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
        **options,
    ):
        if publish_policy not in ("strict", "last_accepted"):
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
        self._tet = TetObjective64(model, material_model=material_model, mass_mode=mass_mode, density=tet_rest_density)
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
        if smooth_sdf is not None:
            self._coupled_config["smooth_sdf_digest"] = np.frombuffer(bytes.fromhex(smooth_sdf.digest), dtype=np.uint8)
        # Include effective numeric policy in checkpoint identity; device execution
        # mode is deliberately excluded so CPU/CUDA checkpoints remain portable.
        defaults = inspect.signature(_ArticulationSolver.__init__).parameters
        names = (
            "newton_max_iterations",
            "line_search_max_iterations",
            "linear_max_iterations",
            "linear_tolerance",
            "linear_absolute_tolerance",
        )
        self._coupled_config["iteration_policy"] = np.array(
            [options.get(name, defaults[name].default) for name in names]
        )
        self._scope_counts = tuple(getattr(model, name) for name in ("spring_count", "tri_count", "edge_count"))
        super().__init__(model, globalization=globalization, publish_policy="strict", **options)
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
        self._complete_r = wp.zeros(nq + 3 * n, dtype=wp.float64, device=device)
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
        return _TreeKinematics(model, extra_config=fields)

    def _make_states(self):
        return CoupledState(self._kinematics)

    def _create_coupled_state(self, *, dt, source=None):
        return self._states.create(dt=dt, source=source)

    def _reset_coupled_state(self, state, *, dt):
        self._validate_components()
        self._states.reset(state, dt=dt)

    def _validate_components(self):
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
        result = super().step(state_in, state_out, control, contacts, dt)
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
        return result

    def _can_publish(self, status):
        return status == self.Status.SUCCESS or (
            self._coupled_publish_policy == "last_accepted"
            and status
            in (
                self.Status.NONLINEAR_MAX_ITERATIONS,
                self.Status.NONLINEAR_STAGNATION,
                self.Status.REGULARIZATION_EXHAUSTED,
            )
            and any(t.get("accepted") for t in self._variational["trials"])
        )

    def _iterate_articulation(self, dt):
        try:
            return super()._iterate_articulation(dt)
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
                wp.float64(dt),
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

        self._graphs.run("coupled_objective", (dt, build_tangent), buffers, evaluate)
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
        return float(parts.sum()), residual, float(abs(parts).sum())

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
        super()._take_trial(alpha, dt)
        wp.launch(
            kernels.particle_trial,
            self.model.particle_count,
            [
                self._accepted.particle_q,
                self._original.particle_q,
                self._tet.mapping,
                self._delta,
                self._kinematics.count,
                wp.float64(alpha),
                wp.float64(dt),
                self._trial.particle_q,
                self._trial.particle_qd,
            ],
            device=self.model.device,
        )

    def _effective_displacement(self):
        ids = self._tet.ids.numpy()
        return np.r_[
            super()._effective_displacement(),
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
            _scatter,
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
            dt=self._dt,
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
                wp.float64(self._dt),
                self._diagonal,
            ],
            device=self.model.device,
        )
        self._require_linear(
            linear.build_scaling(self._diagonal, epsilon_d=self._config.epsilon_d, generation=self._generation)
        )
        self._metrics["matrix_assembly_count"] += 1
        return None, linear.scale.numpy().astype(float)

    def _physical_linear_error(self, matrix, direction, residual, scale):
        self._post_r.assign(residual)
        b = self._linear.k_global_scalar_bsr
        buffers = [b.offsets, b.columns, b.values, self._delta, self._post_r, self._linear.scale, self._post_error]
        self._graphs.run(
            "coupled_posterior",
            (),
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
