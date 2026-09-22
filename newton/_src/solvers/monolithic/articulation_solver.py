# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private variational articulation entry using the existing BSR/PCG workspace."""

import math
from dataclasses import replace
from types import MappingProxyType

import numpy as np
import warp as wp

from ...sim import JointType
from ..solver import SolverBase
from .articulation import MonolithicJointTermsWorkspace
from .articulation_state import ArticulationState, _predict, _recover_history
from .articulation_variational import ArticulationKinematics, ArticulationObjective
from .linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearWorkspace,
    MonolithicPcgConfig,
    MonolithicPcgWarmStart,
)
from .segment_graph import _SegmentGraphCache
from .solver_monolithic import SolverMonolithic, _SolverMonolithicInternalConfig, _StepFailure
from .tet_variational import acceptance


@wp.kernel
def _scatter(matrix: wp.array2d[float], actor: wp.array2d[float], values: wp.array[float], diagonal: wp.array[float]):
    i, j = wp.tid()
    actor[i, j] = matrix[i, j]
    values[i * matrix.shape[0] + j] = matrix[i, j]
    if i == j:
        diagonal[i] = matrix[i, i]


@wp.kernel
def _coordinate_trial(
    q: wp.array[wp.float64],
    direction: wp.array[float],
    alpha: wp.float64,
    qn: wp.array[wp.float64],
    h: wp.float64,
    out: wp.array[wp.float64],
    vout: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = q[i] + alpha * wp.float64(direction[i])
    vout[i] = (out[i] - qn[i]) / h


@wp.kernel
def _coordinate_predictor(
    q: wp.array[wp.float64],
    v: wp.array[wp.float64],
    h: wp.float64,
    out: wp.array[wp.float64],
    vout: wp.array[wp.float64],
):
    i = wp.tid()
    out[i] = q[i] + h * v[i]
    vout[i] = (out[i] - q[i]) / h


class _ArticulationSolver(SolverMonolithic):
    """Bounded PR2 path; old coupled, diagnostic and tet paths are untouched."""

    def __init__(
        self,
        model,
        *,
        globalization,
        joint_terms=None,
        publish_policy="strict",
        newton_max_iterations=10,
        line_search_max_iterations=8,
        linear_max_iterations=200,
        linear_tolerance=1e-4,
        linear_absolute_tolerance=1e-6,
        pcg_mode=SolverMonolithic.PCGMode.DIAGNOSTIC,
        use_cuda_graph=False,
    ):
        if globalization not in ("Armijo", "ResNorm", "ArmijoOrResNorm") or publish_policy not in ("strict", "legacy"):
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
        self._joint_terms = MonolithicJointTermsWorkspace(model, **options)
        for name in ("joint_target_ke", "joint_target_kd", "joint_limit_ke", "joint_limit_kd", "joint_friction"):
            if name not in self._joint_terms.allowed_properties and np.any(getattr(model, name).numpy() != 0):
                raise ValueError(f"Disabled joint term has nonzero parameters: {name}")
        parameters = self._joint_terms.params.numpy()
        self._kinematics = ArticulationKinematics(
            model,
            extra_config={
                "joint_terms": parameters,
                "joint_switches": np.array(
                    [joint_terms.implicit_pd, joint_terms.limits, joint_terms.friction], dtype=np.uint8
                ),
            },
        )
        SolverBase.__init__(self, model)
        self._states = ArticulationState(self._kinematics)
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
        self._globalization, self._publish_policy = globalization, publish_policy
        self.newton_max_iterations, self.line_search_max_iterations = newton_max_iterations, line_search_max_iterations
        self.linear_max_iterations = linear_max_iterations
        self.linear_tolerance, self.linear_absolute_tolerance = linear_tolerance, linear_absolute_tolerance
        self._pcg_mode = self.PCGMode(pcg_mode)
        self._use_cuda_graph = use_cuda_graph
        self._use_optimized_articulation_mass_matrix = False
        self._tet_only, self._tet_physics = False, None
        self.collision_pipeline = None
        n, device = model.joint_dof_count, model.device
        self._linear = MonolithicLinearWorkspace(
            MonolithicLinearLayout(n, 0),
            MonolithicLinearCapacities(n * n, 0, 0, linear_max_iterations, linear_max_iterations + 2),
            wp.empty(0, dtype=int, device=device),
            device=device,
            pcg_mode=self.pcg_mode.value,
            use_cuda_graph=use_cuda_graph,
        )
        self._linear._set_fixed_triplet_pattern(
            global_rows=np.repeat(np.arange(n, dtype=np.int32), n),
            global_columns=np.tile(np.arange(n, dtype=np.int32), n),
            internal_rows=np.empty(0, dtype=np.int32),
            internal_columns=np.empty(0, dtype=np.int32),
        )
        self._graphs = _SegmentGraphCache(device, enabled=use_cuda_graph)
        for name in ("_rhs", "_y", "_delta", "_diagonal"):
            setattr(self, name, wp.zeros(n, dtype=float, device=device))
        # Scratch states are allocated on first use, with the caller's legal initial dt.
        self._original = self._accepted = self._trial = None
        self._step_generation = self._assembly_sequence = 0
        self._last_stats = self.Stats(self.Status.NOT_RUN)
        self._old_joint_t = wp.empty_like(self._kinematics.joint_translations)
        self._old_joint_r = wp.empty_like(self._kinematics.joint_rotations)

    @property
    def last_stats(self):
        return self._last_stats

    def _create_articulation_state(self, *, dt, source=None):
        return self._states.create(dt=dt, source=source)

    def _reset_articulation_state(self, state, *, dt):
        self._states.reset(state, dt=dt)

    def _evaluate(self, state, dt, *, build_tangent=True):
        obj = self._objective
        buffers = [state.joint_q] + [v for v in vars(obj).values() if isinstance(v, wp.array)]
        buffers += [v for v in vars(self._kinematics).values() if isinstance(v, wp.array)]
        self._graphs.run(
            "articulation_objective",
            (dt, build_tangent),
            buffers,
            lambda: obj.evaluate(state.joint_q, dt, build_tangent=build_tangent),
        )
        # CPU numpy() can alias device scratch, which the next trial overwrites.
        residual = obj.residual.numpy().copy()
        parts = np.concatenate((obj.link_energy.numpy(), obj.joint_energy.numpy()))
        if not np.isfinite(residual).all() or not np.isfinite(parts).all():
            raise _StepFailure(self.Status.NONFINITE, "Nonfinite articulation objective")
        return float(parts.sum()), residual, float(abs(parts).sum())

    def _domain(self, state):
        k, obj = self._kinematics, self._objective
        delta = state.joint_q.numpy() - obj.qn.numpy()
        moving_types = k.values["joint_type"][k.dof_joint.numpy()]
        if np.any(abs(delta[moving_types == JointType.REVOLUTE]) >= math.pi / 2):
            return False
        # The caller has evaluated this candidate's private FK.
        return bool(np.all(np.einsum("bij,bij->b", k.rotations.numpy(), obj.rn.numpy()) > 1.0))

    def _assemble(self, iteration):
        obj, linear = self._objective, self._linear
        b = obj.tangent32.numpy()
        if not np.isfinite(b).all() or not np.array_equal(b, b.T) or np.any(np.diag(b) <= 0):
            raise _StepFailure(self.Status.NONFINITE, "Invalid FP32 articulation tangent")
        self._assembly_sequence += 1
        self._generation = MonolithicLinearGeneration(
            self._step_generation,
            iteration,
            0,
            self._assembly_sequence,
            config_generation=0,
            history_epoch=self._step_generation,
        )
        assembly = linear.begin_assembly(self._generation)
        wp.launch(
            _scatter,
            b.shape,
            [obj.tangent32, assembly.aq_actor_dense, assembly.global_scalar_triplets.values, self._diagonal],
            device=self.model.device,
        )
        result = linear.finalize_assembly(generation=self._generation)
        self._require_linear(result.status)
        self._require_linear(
            linear.build_scaling(self._diagonal, epsilon_d=self._config.epsilon_d, generation=self._generation)
        )
        self._metrics["matrix_assembly_count"] += 1
        return b.astype(float), linear.scale.numpy().astype(float)

    def _iterate_articulation(self, dt):
        config = self._config
        reference = None
        small = False
        for iteration in range(self.newton_max_iterations + 1):
            e, residual, absolute = self._evaluate(self._accepted, dt)
            b, scale = self._assemble(iteration)
            if reference is None:
                reference = residual.copy()
            merit = float(np.linalg.norm(scale * residual) / math.sqrt(len(residual)))
            ref = float(np.linalg.norm(scale * reference) / math.sqrt(len(reference)))
            goals = (
                config.merit_absolute_global + config.merit_relative_global * ref,
                config.merit_absolute_q + config.merit_relative_q * ref,
            )
            self._metrics.update(
                nonlinear_iterations=iteration,
                merit_final=merit,
                merit_q_final=merit,
                merit_reference=ref,
                merit_q_reference=ref,
                convergence_ratio=merit / goals[0],
                convergence_ratio_q=merit / goals[1],
                raw_residual_q_norm=float(np.linalg.norm(residual)),
            )
            if iteration == 0:
                self._metrics.update(merit_initial=merit, merit_q_initial=merit)
            self._variational.update(objective_total=e, residual=residual.tolist())
            if merit <= min(goals):
                return self.Status.SUCCESS
            if small:
                raise _StepFailure(
                    self.Status.NONLINEAR_STAGNATION, "Small effective step with unresolved physical residual"
                )
            if iteration == self.newton_max_iterations:
                return self.Status.NONLINEAR_MAX_ITERATIONS
            rhs64 = -scale * residual
            self._rhs.assign(rhs64.astype(np.float32))
            caps = tuple(0.5 * math.sqrt(len(residual)) * goal for goal in goals)
            pcg = MonolithicPcgConfig(
                maximum_iterations=self.linear_max_iterations,
                true_residual_interval=10,
                linear_tolerance=self.linear_tolerance,
                linear_absolute_tolerance=self.linear_absolute_tolerance,
                residual_norm_limits=(*caps, 1.0),
                residual_floor_global=config.residual_floor_global,
                residual_floor_q=config.residual_floor_q,
                residual_floor_x=config.residual_floor_x,
                curvature_absolute_tolerance=0.0,
                curvature_relative_tolerance=1e-12,
                preconditioner_positive_tolerance=0.0,
                stagnation_window=10,
                stagnation_minimum_reduction=1e-3,
            )
            accepted = False
            for retry, regularization in enumerate(config.regularization_values):
                self._metrics["regularization_retries"] += int(retry > 0)
                self._metrics["lambda_value"] = regularization
                try:
                    self._require_linear(self._linear.set_regularization(regularization, generation=self._generation))
                    self._require_linear(
                        self._linear.factor_actor_preconditioner(generation=self._generation, pivot_tolerance=0.0)
                    )
                    result = self._linear.solve_pcg(
                        self._rhs,
                        self._y,
                        generation=self._generation,
                        warm_start=MonolithicPcgWarmStart.ZERO,
                        config=pcg,
                        delta=self._delta,
                    )
                    self._metrics["linear_iterations"] += result.iterations
                    self._metrics["linear_scheduled_slots"] += result.scheduled_slots
                    self._require_linear(result.status)
                    if self.pcg_mode == self.PCGMode.DIAGNOSTIC:
                        self._require_linear(
                            self._linear.recover_delta(self._y, self._delta, generation=self._generation)
                        )
                    elif not result.output_valid:
                        raise _StepFailure(self.Status.NONFINITE, "Invalid PCG output")
                except _StepFailure:
                    continue
                direction = self._delta.numpy().astype(float)
                post = float(np.linalg.norm(scale * (b @ direction + residual)))
                limit = min(
                    *caps,
                    max(
                        self.linear_absolute_tolerance,
                        self.linear_tolerance * max(np.linalg.norm(rhs64), config.residual_floor_global),
                    ),
                    max(
                        self.linear_absolute_tolerance,
                        self.linear_tolerance * max(np.linalg.norm(rhs64), config.residual_floor_q),
                    ),
                )
                self._variational["linear_postchecks"].append(
                    {"residual": post, "limit": limit, "regularization": regularization}
                )
                if not np.isfinite(post) or post > limit:
                    continue
                for search in range(self.line_search_max_iterations):
                    alpha = 0.5**search
                    self._metrics["line_search_iterations"] += 1
                    wp.launch(
                        _coordinate_trial,
                        len(residual),
                        [
                            self._accepted.joint_q,
                            self._delta,
                            wp.float64(alpha),
                            self._objective.qn,
                            wp.float64(dt),
                            self._trial.joint_q,
                            self._trial.joint_qd,
                        ],
                        device=self.model.device,
                    )
                    te, tr, ta = self._evaluate(self._trial, dt, build_tangent=False)
                    if not self._domain(self._trial):
                        continue
                    effective = self._trial.joint_q.numpy() - self._accepted.joint_q.numpy()
                    slope = float(residual @ effective) / alpha
                    trial_merit = float(np.linalg.norm(scale * tr) / math.sqrt(len(tr)))
                    # CPU/CUDA common envelope frozen in fixtures/p3_pr2_noise_v1.json.
                    eps = 128 * np.finfo(float).eps
                    en = max(5.820766091346741e-11, eps * max(1.0, absolute, ta))
                    rn = max(
                        2.0935893019213836e-9 * float(np.linalg.norm(scale)) / math.sqrt(len(scale)),
                        eps * max(1.0, merit, trial_merit),
                    )
                    sn = max(
                        1.5498947892952669e-9 * float(np.linalg.norm(direction)),
                        2.0935893019213836e-9 * float(abs(direction).sum()),
                        eps * max(1.0, float(abs(residual * direction).sum())),
                    )
                    passes, a, r, improved = acceptance(
                        self._globalization,
                        e,
                        te,
                        slope,
                        alpha,
                        merit,
                        trial_merit,
                        energy_noise=en,
                        residual_noise=rn,
                        slope_noise=sn,
                    )
                    self._variational["trials"].append(
                        {
                            "iteration": iteration,
                            "retry": retry,
                            "alpha": alpha,
                            "energy": te,
                            "merit": trial_merit,
                            "slope": slope,
                            "Armijo": a,
                            "ResNorm": r,
                            "accepted": passes,
                            "improved": improved,
                        }
                    )
                    if passes:
                        scaled_step = float(np.linalg.norm(effective / scale) / math.sqrt(len(effective)))
                        small = scaled_step <= min(config.step_tolerance_global, config.step_tolerance_q)
                        self._accepted, self._trial = self._trial, self._accepted
                        self._metrics.update(accepted_alpha=alpha, scaled_step=scaled_step, scaled_step_q=scaled_step)
                        accepted = True
                        break
                if accepted:
                    break
            if not accepted:
                raise _StepFailure(
                    self.Status.REGULARIZATION_EXHAUSTED, "No direction/trial passed unchanged physical gates"
                )
        raise AssertionError("Unreachable")

    def _joint_force_metrics(self, state):
        """Bind scalar force diagnostics to the successfully published state."""
        p = self._objective.params.numpy()
        q, v = state.joint_q.numpy(), state.joint_qd.numpy()
        raw = p[:, 0] * (q - self._objective.qt.numpy()) + p[:, 1] * (v - self._objective.vt.numpy())
        pd = -np.clip(raw, -p[:, 2], p[:, 2])
        limit, power = np.zeros(len(q)), np.zeros(len(q))
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

    def step(self, state_in, state_out, control, contacts=None, dt=0.0):
        if contacts is not None or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Articulation step requires no contacts and finite positive dt")
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
        force = state_in.body_f.numpy().astype(float)
        joint_force = control.joint_f.numpy().astype(float)
        gravity = self.model.gravity.numpy().astype(float)
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
        history = state_in._monolithic_articulation
        wp.launch(
            _predict,
            k.bodies,
            [
                obj.tn,
                obj.rn,
                history.link_v_com,
                history.link_omega,
                history.link_vsym,
                wp.float64(dt),
                obj.predicted_t,
                obj.predicted_r,
            ],
            device=self.model.device,
        )
        obj.force.assign(force[:, :3] + k.values["body_mass"][:, None] * gravity[k.values["body_world"]])
        obj.joint_force.assign(joint_force)
        obj.qt.assign(self._joint_terms.target_q.numpy().astype(float))
        obj.vt.assign(self._joint_terms.target_qd.numpy().astype(float))
        wp.launch(
            _coordinate_predictor,
            k.count,
            [obj.qn, obj.vn, wp.float64(dt), self._accepted.joint_q, self._accepted.joint_qd],
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
            if status == self.Status.SUCCESS or (
                self._publish_policy == "legacy" and status == self.Status.NONLINEAR_MAX_ITERATIONS
            ):
                # Re-evaluate the actual stored candidate against the original frozen stage.
                energy, residual, _ = self._evaluate(self._accepted, dt, build_tangent=False)
                scale = self._linear.scale.numpy().astype(float)
                merit = float(np.linalg.norm(scale * residual) / math.sqrt(len(residual)))
                reference = self._metrics["merit_reference"]
                goals = (
                    self._config.merit_absolute_global + self._config.merit_relative_global * reference,
                    self._config.merit_absolute_q + self._config.merit_relative_q * reference,
                )
                self._metrics.update(
                    merit_final=merit,
                    merit_q_final=merit,
                    convergence_ratio=merit / goals[0],
                    convergence_ratio_q=merit / goals[1],
                    raw_residual_q_norm=float(np.linalg.norm(residual)),
                )
                self._variational.update(objective_total=energy, residual=residual.tolist())
                if status == self.Status.SUCCESS and merit > min(goals):
                    raise _StepFailure(
                        self.Status.NONLINEAR_STAGNATION, "Final articulation residual failed publication gates"
                    )
                if not self._domain(self._accepted):
                    raise _StepFailure(self.Status.NONFINITE, "Final articulation candidate left the local domain")
                # A PD-stabilized B cannot certify the candidate's physical inertia.
                try:
                    k.validate_inertia()
                except ValueError as error:
                    raise _StepFailure(self.Status.NONFINITE, str(error)) from error
                final = self._accepted._monolithic_articulation
                wp.launch(
                    _recover_history,
                    k.bodies,
                    [
                        obj.tn,
                        obj.rn,
                        k.com,
                        k.rotations,
                        wp.float64(dt),
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
                        wp.float64(dt),
                        final.joint_v,
                        final.joint_omega,
                        final.joint_vsym,
                    ],
                    device=self.model.device,
                )
                k.evaluate(self._accepted.joint_q, state=self._accepted)
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
        return state_out

    def _restore(self, state):
        for name, value in vars(self._original).items():
            if isinstance(value, wp.array):
                wp.copy(getattr(state, name), value)
        for name in self._states.schema:
            wp.copy(
                getattr(state._monolithic_articulation, name), getattr(self._original._monolithic_articulation, name)
            )
