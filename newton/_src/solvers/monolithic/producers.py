# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step contract shared by every monolithic physical producer."""

from typing import Any, Protocol

import numpy as np
import warp as wp

from ...sim import JointType
from . import coupled_kernels as kernels
from . import iteration_device
from .articulation_state import _predict, _recover_step
from .rigid import _scatter_residual, _scatter_tangent
from .solver_types import _StepFailure
from .state_checks import scalar_view
from .tet_assembly import TetScatterBuffers, assemble_tet_residual_tangent


class Producer(Protocol):
    """One physical term set owning a disjoint set of scalar rows.

    ``capturable`` producers perform no host reads in ``evaluate``/``assemble``.
    ``scale_policy`` is ``"diagonal"`` (dynamic diagonal) or ``"coordinate"``
    (fixed per-row unit scale). Producers without rows, such as contact, add
    coupling terms to rows owned by others.
    """

    name: str
    rows: np.ndarray
    capturable: bool
    scale_policy: str

    def begin_step(self, solver: Any, state_in: Any) -> None: ...

    def candidate(self, solver: Any, accepted: Any, trial: Any, direction: Any, alpha: float, dt: float) -> None: ...

    def evaluate(
        self, solver: Any, state: Any, dt: float, residual: Any, *, build_tangent: bool
    ) -> tuple[Any, Any]: ...

    def magnitude(self, solver: Any, out: Any) -> None: ...

    def coordinates(self, solver: Any, state: Any, out: Any, *, absolute: bool = False) -> None: ...

    def assemble(self, solver: Any, assembly: Any, dt: float) -> None: ...

    def scale(self, solver: Any, diagonal: Any, coordinate: np.ndarray) -> None: ...

    def displacement(self, solver: Any, accepted: Any, trial: Any, out: np.ndarray) -> None: ...

    def domain(self, solver: Any, accepted: Any, candidate: Any) -> bool: ...

    def publish(self, solver: Any, state: Any) -> None: ...


def validate_row_partition(producers, scalar_dof_count):
    """Require producer rows to cover every scalar DoF exactly once."""
    owner = np.full(scalar_dof_count, -1, dtype=np.int64)
    for index, producer in enumerate(producers):
        rows = np.asarray(producer.rows, dtype=np.int64)
        if np.any(rows < 0) or np.any(rows >= scalar_dof_count):
            raise ValueError(f"Producer rows lie outside the layout: {producer.name}")
        if len(np.unique(rows)) != len(rows) or np.any(owner[rows] >= 0):
            raise ValueError(f"Producer rows overlap: {producer.name}")
        owner[rows] = index
    if np.any(owner < 0):
        raise ValueError("Producer rows leave scalar DoFs uncovered")


@wp.kernel
def _scatter_values(values: wp.array[Any], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = values[i]


@wp.kernel
def _add_values(source: wp.array[Any], target: wp.array[Any]):
    i = wp.tid()
    target[i] += source[i]


@wp.kernel
def _add_offset(values: wp.array[Any], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] += values[i]


@wp.kernel
def _add_absolute(values: wp.array[Any], output: wp.array[Any]):
    i = wp.tid()
    output[i] += wp.abs(values[i])


@wp.kernel
def _add_rows(values: wp.array[Any], rows: wp.array[int], output: wp.array[Any]):
    i = wp.tid()
    if rows[i] >= 0:
        output[rows[i]] += values[i]


@wp.kernel
def _write_absolute(values: wp.array[Any], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = wp.abs(values[i])


@wp.kernel
def _particle_coordinates(x: wp.array[Any], ids: wp.array[int], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = type(output[0])(wp.abs(x[ids[i / 3]][i % 3]))


@wp.kernel
def _rigid_coordinates(
    com: wp.array[Any], rotation: wp.array[Any], absolute: bool, rows: wp.array2d[int], output: wp.array[Any]
):
    i, j = wp.tid()
    # Absolute rotations are orthonormal matrices with entries of unit magnitude.
    value = type(output[0])(1.0)
    if j < 3:
        value = type(output[0])(wp.abs(com[i][j]))
    elif not absolute:
        # A rotation step q q_n^-1 near identity turns by about twice its vector part.
        value = type(output[0])(2.0) * type(output[0])(wp.abs(rotation[i][j - 3]))
    output[rows[i, j]] = value


@wp.kernel
def _material_coordinates(
    x: wp.array[Any], ids: wp.array[int], theta: wp.array[Any], rows: wp.array[int], output: wp.array[Any]
):
    r = wp.tid()
    if rows[r] >= 0:
        nv = ids.shape[0]
        if r < 3 * nv:
            output[rows[r]] = type(output[0])(wp.abs(x[ids[r / 3]][r % 3]))
        else:
            output[rows[r]] = wp.abs(theta[r - 3 * nv])


@wp.kernel
def _raise_coordinates(
    count: wp.array[int],
    supports: wp.array[int],
    columns: wp.array[int],
    values: wp.array[Any],
    output: wp.array[Any],
):
    i = wp.tid()
    # Each contact has three factors over its supports.
    if i < 3 * supports[count[0]]:
        column = columns[i]
        wp.atomic_max(output, column, values[column])


class _Producer:
    """Name, owned rows, declared policies and default no-op step methods."""

    capturable = True
    scale_policy = "diagonal"
    # Step-start predictions read state_in; articulation and rigid prediction run last.
    begin_rank = 0

    def __init__(self, name, rows):
        self.name, self.rows = name, np.asarray(rows, dtype=np.int32)

    def begin_step(self, solver, state_in):
        """Freeze step-start inputs and write this producer's predictor into the accepted state."""

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        """Write this producer's trial coordinates from the accepted state and direction."""

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        """Add this producer's residual rows; return its host energy and absolute energy."""
        return 0.0, 0.0

    def magnitude(self, solver, out):
        """Add the absolute contributions summed into this producer's last residual rows."""

    def coordinates(self, solver, state, out, *, absolute=False):
        """Write the magnitude of each owned row's coordinate as this producer's terms read it.

        Terms formed from the step increment write its magnitude; with
        ``absolute``, write the magnitude of the absolute coordinate instead,
        for terms that still read absolute positions.
        """

    def assemble(self, solver, assembly, dt):
        """Write this producer's tangent into its fixed triplet slots."""

    def scale(self, solver, diagonal, coordinate):
        """Write coordinate-policy unit scales for this producer's rows."""

    def displacement(self, solver, accepted, trial, out):
        """Write the effective trial displacement of this producer's rows."""

    def domain(self, solver, accepted, candidate):
        """Return whether the candidate lies in this producer's local chart."""
        return True

    def publish(self, solver, state):
        """Publish this producer's final history or forces into the accepted state."""


class ArticulationProducer(_Producer):
    """Articulation coordinates with joint, PD, limit and friction terms."""

    begin_rank = 1

    def __init__(self, solver):
        super().__init__("articulation", solver._layout.indices("articulation"))

    def begin_step(self, solver, state_in):
        k, obj, device = solver._kinematics, solver._objective, solver.model.device
        obj.begin(k.gather_q(state_in))
        wp.copy(obj.vn, k.gather_qd(state_in))
        history = getattr(state_in, solver._states.namespace)
        dt = solver._dtype(solver._dt)
        wp.launch(
            _predict,
            k.bodies,
            [
                obj.tn,
                obj.rn,
                history.link_v_com,
                history.link_omega,
                history.link_vsym,
                dt,
                obj.predicted_t,
                obj.predicted_r,
                obj.step_t,
                obj.step_r,
            ],
            device=device,
        )
        wp.utils.array_cast(solver._joint_terms.target_q, obj.qt)
        wp.utils.array_cast(solver._joint_terms.target_qd, obj.vt)
        accepted = solver._accepted
        wp.launch(
            kernels._coordinate_predictor,
            k.count,
            [
                obj.qn,
                obj.vn,
                dt,
                k.coord_map,
                k.dof_map,
                accepted.joint_q,
                accepted.joint_qd,
                solver._joint_step(accepted),
            ],
            device=device,
        )

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        k = solver._kinematics
        wp.launch(
            kernels._coordinate_trial,
            k.count,
            [
                solver._joint_step(accepted),
                direction,
                solver._layout.q_scalar_offset,
                solver._dtype(alpha),
                solver._objective.qn,
                solver._dtype(dt),
                k.coord_map,
                k.dof_map,
                trial.joint_q,
                trial.joint_qd,
                solver._joint_step(trial),
            ],
            device=solver.model.device,
        )

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        k, obj = solver._kinematics, solver._objective
        step = solver._joint_step(state)
        obj.evaluate(k.gather_q(state), dt, build_tangent=build_tangent, step=step)
        wp.launch(
            _scatter_values,
            k.count,
            [obj.residual, solver._layout.q_scalar_offset, residual],
            device=solver.model.device,
        )
        parts = np.r_[obj.link_energy.numpy(), obj.joint_energy.numpy()]
        return parts.sum(), abs(parts).sum()

    def magnitude(self, solver, out):
        obj = solver._objective
        wp.launch(
            _add_offset,
            obj.absolute.size,
            [obj.absolute, solver._layout.q_scalar_offset, out],
            device=solver.model.device,
        )

    def coordinates(self, solver, state, out, *, absolute=False):
        k = solver._kinematics
        q = k.gather_q(state) if absolute else solver._joint_step(state)
        wp.launch(_write_absolute, k.count, [q, solver._layout.q_scalar_offset, out], device=solver.model.device)

    def assemble(self, solver, assembly, dt):
        nq = solver._kinematics.count
        wp.launch(
            kernels._scatter,
            (nq, nq),
            [
                solver._objective.tangent,
                assembly.aq_actor_dense,
                assembly.global_scalar_triplets.values,
                solver._diagonal,
            ],
            device=solver.model.device,
        )

    def displacement(self, solver, accepted, trial, out):
        out[self.rows] = solver._joint_step(trial).numpy() - solver._joint_step(accepted).numpy()

    def domain(self, solver, accepted, candidate):
        return iteration_device.domain(solver, candidate, int(JointType.REVOLUTE))

    def publish(self, solver, state):
        k, obj, device = solver._kinematics, solver._objective, solver.model.device
        final = getattr(state, solver._states.namespace)
        dt = solver._dtype(solver._dt)
        q = k.gather_q(state)
        k.evaluate_step(q, solver._joint_step(state), obj.rn, obj.axes_n)
        wp.launch(
            _recover_step,
            k.bodies,
            [k.com_steps, k.rotation_steps, dt, final.link_v_com, final.link_omega, final.link_vsym],
            device=device,
        )
        wp.launch(
            _recover_step,
            k.joints,
            [
                k.joint_translation_steps,
                k.joint_rotation_steps,
                dt,
                final.joint_v,
                final.joint_omega,
                final.joint_vsym,
            ],
            device=device,
        )
        k.evaluate(q, state=state)


class RigidProducer(_Producer):
    """Every FREE rigid actor: three COM and three rotation rows each."""

    begin_rank = 2

    def __init__(self, solver):
        super().__init__("rigid", solver._rigid.rows_numpy.ravel())

    def begin_step(self, solver, state_in):
        # The predictor moves the COM by h v and keeps the step-start rotation.
        wp.copy(solver._accepted._monolithic_coupled.rigid_com, solver._rigid.predicted_t)
        wp.copy(solver._rigid_steps(solver._accepted)[0], solver._rigid.step_t)

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        solver._rigid.trial(solver._rigid_steps(accepted), trial, solver._rigid_steps(trial), direction, alpha)

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        rigid = solver._rigid
        rigid.evaluate(state, solver._rigid_steps(state), dt, build_tangent)
        wp.launch(
            _scatter_residual, (rigid.count, 6), [rigid.rows, rigid.residual, residual], device=solver.model.device
        )
        energy = rigid.energy.numpy()
        return energy.sum(), abs(energy).sum()

    def magnitude(self, solver, out):
        rigid = solver._rigid
        if rigid.count:
            wp.launch(
                _scatter_residual, (rigid.count, 6), [rigid.rows, rigid.absolute, out], device=solver.model.device
            )

    def coordinates(self, solver, state, out, *, absolute=False):
        rigid = solver._rigid
        if rigid.count:
            if absolute:
                inputs = [state._monolithic_coupled.rigid_com, state._monolithic_coupled.rigid_rotation, absolute]
            else:
                inputs = [*solver._rigid_steps(state), absolute]
            wp.launch(_rigid_coordinates, (rigid.count, 6), [*inputs, rigid.rows, out], device=solver.model.device)

    def assemble(self, solver, assembly, dt):
        rigid = solver._rigid
        wp.launch(
            _scatter_tangent,
            (rigid.count, 6, 6),
            [
                rigid.tangent,
                rigid.rows,
                rigid.diagonal,
                solver._rigid_triplet_offset,
                assembly.global_scalar_triplets.values,
                solver._diagonal,
            ],
            device=solver.model.device,
        )

    def displacement(self, solver, accepted, trial, out):
        rows = solver._rigid.rows_numpy
        (ta, qa), (tb, qb) = ((t.numpy(), q.numpy()) for t, q in map(solver._rigid_steps, (accepted, trial)))
        out[rows[:, :3]] = tb - ta
        # Each trial is exactly the left Exp(alpha*delta) retraction within the local domain.
        vector = qb[:, 3:] * -qa[:, :3] + qa[:, 3:] * qb[:, :3] + np.cross(qb[:, :3], -qa[:, :3])
        scalar = np.sum(qb * qa, axis=1)
        vector *= np.where(scalar < 0, -1, 1)[:, None]
        length = np.linalg.norm(vector, axis=1)
        factor = np.divide(2 * np.arctan2(length, abs(scalar)), length, out=np.full_like(length, 2), where=length > 0)
        out[rows[:, 3:]] = vector * factor[:, None]

    def domain(self, solver, accepted, candidate):
        return solver._rigid.domain(candidate)

    def publish(self, solver, state):
        solver._rigid.recover(state, solver._rigid_steps(state), solver._dt)


class TetProducer(_Producer):
    """Dynamic tet particles with inertia, elasticity and frozen external forces."""

    def __init__(self, solver):
        super().__init__("tet", solver._layout.indices("tet"))

    def begin_step(self, solver, state_in):
        accepted = solver._accepted
        wp.launch(
            kernels.particle_predict,
            solver.model.particle_count,
            [
                state_in.particle_q,
                state_in.particle_qd,
                solver._tet.mapping,
                solver._dtype(solver._dt),
                accepted.particle_q,
                accepted.particle_qd,
                solver._displacement(accepted),
            ],
            device=solver.model.device,
        )

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        wp.launch(
            kernels.particle_trial,
            solver.model.particle_count,
            [
                solver._displacement(accepted),
                solver._original.particle_q,
                solver._tet.mapping,
                direction,
                solver._layout.x_scalar_offset,
                solver._dtype(alpha),
                solver._dtype(dt),
                trial.particle_q,
                trial.particle_qd,
                solver._displacement(trial),
            ],
            device=solver.model.device,
        )

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        tet = solver._tet
        tet.evaluate(
            state.particle_q,
            solver._original.particle_q,
            solver._original.particle_qd,
            solver._frozen_force,
            solver._frozen_gravity,
            dt=dt,
            min_det_f_guard=solver._guard,
            displacement=solver._displacement(state),
        )
        if tet.flags.numpy()[0]:
            raise _StepFailure(solver.Status.TET_INVERSION, "Invalid tet candidate")
        wp.launch(
            _scatter_values,
            3 * tet.ids.size,
            [scalar_view(tet.residual), solver._layout.x_scalar_offset, residual],
            device=solver.model.device,
        )
        parts = tet.parts.numpy()
        solver._metrics["min_det_f"] = float(tet.min_det_f.numpy()[0])
        return parts.sum(), abs(parts).sum()

    def magnitude(self, solver, out):
        absolute = scalar_view(solver._tet.absolute)
        wp.launch(
            _add_offset, absolute.size, [absolute, solver._layout.x_scalar_offset, out], device=solver.model.device
        )

    def coordinates(self, solver, state, out, *, absolute=False):
        # Tet terms read the exact step displacement, so its rounding, not that
        # of the absolute position, bounds how closely a row can be resolved.
        ids = solver._tet.ids
        x = state.particle_q if absolute else solver._displacement(state)
        wp.launch(
            _particle_coordinates,
            3 * ids.size,
            [x, ids, solver._layout.x_scalar_offset, out],
            device=solver.model.device,
        )

    def assemble(self, solver, assembly, dt):
        nq = solver._kinematics.count
        scatter = TetScatterBuffers(
            assembly.ax_internal_triplets.values,
            assembly.global_scalar_triplets.values[nq * nq : nq * nq + 9 * assembly.ax_internal_triplets.capacity],
        )
        assemble_tet_residual_tangent(
            solver.model,
            candidate_particle_q=solver._current_state.particle_q,
            particle_q_n=solver._original.particle_q,
            particle_qd_n=solver._original.particle_qd,
            frozen_particle_f=solver._frozen_force,
            dynamic_particle_ids=solver._tet.ids,
            particle_to_dynamic=solver._tet.mapping,
            residual_x=solver._tet_assembly_residual,
            dt=float(dt),
            min_det_f_guard=solver._guard,
            workspace=solver._tet_assembly,
            scatter=scatter,
            frozen_gravity=solver._frozen_gravity,
        )
        if solver._tet_assembly.failure_flags.numpy()[0]:
            raise _StepFailure(solver.Status.TET_PROJECTION_FAILURE, "Invalid tet tangent")

    def displacement(self, solver, accepted, trial, out):
        ids = solver._dynamic_ids_host
        trial_step, accepted_step = solver._displacement(trial).numpy(), solver._displacement(accepted).numpy()
        out[self.rows] = (trial_step[ids] - accepted_step[ids]).ravel()


class _MaterialProducer(_Producer):
    """One cloth or cable object; host-side evaluation is not graph-capturable yet."""

    capturable = False
    scale_policy = "coordinate"

    def __init__(self, solver, kind, block):
        source = block.actor.source_index
        super().__init__(f"{kind}_{source}", solver._layout.indices(kind, source))
        self.block = block

    def gravity(self, solver):
        return wp.vec3(solver._frozen_gravity.numpy()[int(self.block.reference["world"])])

    def assemble(self, solver, assembly, dt):
        self.block.assemble(assembly, dt)

    def magnitude(self, solver, out):
        block = self.block
        wp.launch(_add_rows, block.absolute.size, [block.absolute, block.rows, out], device=solver.model.device)

    def theta(self, solver, state, *, absolute=False):
        """Return this object's angle increments or angles; cloth has none."""
        return wp.zeros(0, dtype=state.particle_q.dtype._wp_scalar_type_, device=state.particle_q.device)

    def coordinates(self, solver, state, out, *, absolute=False):
        # Shell and rod terms read the exact step increments, so their rounding,
        # not that of the absolute coordinates, bounds how closely a row can be
        # resolved.
        block = self.block
        x = state.particle_q if absolute else solver._displacement(state)
        wp.launch(
            _material_coordinates,
            block.rows.size,
            [x, block.ids, self.theta(solver, state, absolute=absolute), block.rows, out],
            device=solver.model.device,
        )

    def displacement(self, solver, accepted, trial, out):
        local = self.local(solver, trial) - self.local(solver, accepted)
        free = self.block.rows_host >= 0
        out[self.block.rows_host[free]] = local[free]

    def local(self, solver, state):
        """Return this object's step increments in local row order."""
        return solver._displacement(state).numpy()[list(self.block.actor.particle_indices), :].ravel()

    def domain(self, solver, accepted, candidate):
        return self.block.path_valid(solver._original, solver._displacement(accepted), solver._displacement(candidate))


class ClothProducer(_MaterialProducer):
    """One cloth object."""

    def __init__(self, solver, block):
        super().__init__(solver, "cloth", block)

    def begin_step(self, solver, state_in):
        cloth, original, accepted = self.block, solver._original, solver._accepted
        start, step = solver._displacement(original), solver._displacement(accepted)
        cloth.begin_step(original, solver._frozen_force)
        cloth.candidate(start, original, accepted, step, solver._delta, 0.0, solver._dt, predict=True)
        if not cloth.path_valid(original, start, step):
            # An inertial guess is optional; retain the valid input chart.
            cloth.candidate(start, original, accepted, step, solver._delta, 0.0, solver._dt)

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        self.block.candidate(
            solver._displacement(accepted), solver._original, trial, solver._displacement(trial), direction, alpha, dt
        )

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        try:
            return self.block.evaluate(
                solver._displacement(state), self.gravity(solver), dt, residual, build_tangent=build_tangent
            )
        except ValueError as error:
            raise _StepFailure(solver.Status.NONFINITE, str(error)) from error


class CableProducer(_MaterialProducer):
    """One cable object; theta rows use a fixed inverse-length scale."""

    def __init__(self, solver, block):
        super().__init__(solver, "cable", block)

    def begin_step(self, solver, state_in):
        self.block.begin_step(solver._original, solver._frozen_force)
        self.block.candidate(
            self.steps(solver, solver._original),
            solver._original,
            solver._accepted,
            self.steps(solver, solver._accepted),
            solver._delta,
            solver._cable_target_x,
            solver._cable_target_theta,
            0.0,
            solver._dt,
            predict=True,
        )

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        self.block.candidate(
            self.steps(solver, accepted),
            solver._original,
            trial,
            self.steps(solver, trial),
            direction,
            solver._cable_target_x,
            solver._cable_target_theta,
            alpha,
            dt,
        )

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        try:
            return self.block.evaluate(
                state,
                solver._displacement(state),
                solver._theta_step(state),
                solver._original,
                solver._cable_torque,
                self.gravity(solver),
                dt,
                residual,
            )
        except ValueError as error:
            raise _StepFailure(solver.Status.NONFINITE, str(error)) from error

    def scale(self, solver, diagonal, coordinate):
        rows = self.block.rows_host[3 * self.block.ids.size :]
        coordinate[rows[rows >= 0]] = 1 / float(self.block.reference["length_scale"])

    @staticmethod
    def steps(solver, state):
        return solver._displacement(state), solver._theta_step(state)

    def local(self, solver, state):
        theta = solver._theta_step(state).numpy()[list(self.block.actor.theta_indices)]
        return np.r_[super().local(solver, state), theta]

    def theta(self, solver, state, *, absolute=False):
        start = self.block.theta_offset
        theta = state._monolithic_coupled.cable_theta if absolute else solver._theta_step(state)
        return theta[start : start + self.block.objective.lengths.size]


class ContactProducer(_Producer):
    """Spatial contact; it adds coupling terms to rows owned by other producers."""

    scale_policy = "none"

    def __init__(self, solver):
        super().__init__("contact", np.empty(0, dtype=np.int32))

    def begin_step(self, solver, state_in):
        contact, k = solver._contact, solver._kinematics
        q = k.gather_q(state_in)
        k.evaluate(q)
        history = state_in._monolithic_coupled
        surface = contact._batch.material_surface
        if surface is not None:
            surface.update(state_in, solver._displacement(solver._original), begin=True)
        contact.begin_step(
            state_in.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            rigid_com=history.rigid_com,
            rigid_rotation=history.rigid_rotation,
            **({"kinematics": k, "q0": q} if solver._contact_spatial else {}),
        )

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        contact, k = solver._contact, solver._kinematics
        reference_mode = solver._contact_spatial and contact._reference_mode
        history = state._monolithic_coupled
        surface = contact._batch.material_surface
        if surface is not None:
            try:
                surface.update(state, solver._displacement(state))
            except ValueError as error:
                raise _StepFailure(solver.Status.NONFINITE, str(error)) from error
        contact.evaluate(
            state.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            dt=dt,
            rigid_com=history.rigid_com,
            rigid_rotation=history.rigid_rotation,
            steps=solver._contact_steps(state),
        )
        diagnostics = {} if reference_mode else contact.evaluation_diagnostics()
        if not reference_mode:
            solver._variational["contact_pool"] = diagnostics
        if (
            contact.flags.numpy()[0]
            or diagnostics.get("dropped", 0)
            or (solver._contact_spatial and contact._query_dropped)
        ):
            raise _StepFailure(solver.Status.NONFINITE, "Invalid or incomplete contact evaluation")
        wp.launch(_add_values, residual.size, [contact.residual, residual], device=solver.model.device)
        # Material blocks report their raw residual including the contact reaction.
        forces = contact.pool.force.result.numpy()
        model = solver.model
        for block in solver._cloths + solver._cables:
            raw = block.raw.numpy()
            if reference_mode:
                rows = block.rows_host
                free = rows >= 0
                raw[free] += contact.residual.numpy()[rows[free]]
                block.raw.assign(raw)
                continue
            ids = block.ids.numpy()
            raw[: 3 * len(ids)] -= forces[: 3 * model.particle_count].reshape(-1, 3)[ids].ravel()
            if block in solver._cables:
                offset = 3 * model.particle_count + 6 * model.body_count + int(block.reference["theta_offset"])
                raw[3 * len(ids) :] -= forces[offset : offset + len(ids) - 1]
            block.raw.assign(raw)
        return contact.energy.numpy().sum(), contact.absolute_energy.numpy().sum()

    def magnitude(self, solver, out):
        residual = solver._contact.residual
        wp.launch(_add_absolute, residual.size, [residual, out], device=solver.model.device)

    def coordinates(self, solver, state, out, *, absolute=False):
        # Contact gaps are formed from absolute positions, so every coordinate a
        # contact factor reads is resolved only to the rounding of its absolute
        # value. Raising (not adding) keeps the result independent of order.
        if absolute:
            return
        values = solver._absolute_coordinates
        values.zero_()
        for producer in solver._producers:
            if producer is not self:
                producer.coordinates(solver, state, values, absolute=True)
        records = solver._contact.pool.records
        wp.launch(
            _raise_coordinates,
            records.factor_columns.size,
            [records.count, records.support_offsets, records.factor_columns, values, out],
            device=solver.model.device,
        )

    def assemble(self, solver, assembly, dt):
        solver._contact.assemble_tangent(assembly)

    def publish(self, solver, state):
        solver._variational["actor_contact_pairs"] = solver._contact.publish(state)


def build_producers(solver):
    """Order producers as articulation, rigid, tet, cloth, cable and contact."""
    layout = solver._layout
    producers = []
    if layout.q_dof_count:
        producers.append(ArticulationProducer(solver))
    if solver._rigid.count:
        producers.append(RigidProducer(solver))
    if solver._has_tet:
        producers.append(TetProducer(solver))
    producers.extend(ClothProducer(solver, block) for block in solver._cloths)
    producers.extend(CableProducer(solver, block) for block in solver._cables)
    # A contact-free scene keeps its empty contact producer, as it keeps an empty pool.
    producers.append(ContactProducer(solver))
    validate_row_partition([p for p in producers if len(p.rows)], layout.scalar_dof_count)
    return producers
