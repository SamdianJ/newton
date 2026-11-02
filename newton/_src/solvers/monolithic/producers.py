# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Step contract shared by every monolithic physical producer."""

from typing import Any, Protocol

import numpy as np
import warp as wp

from ...sim import JointType
from . import coupled_kernels as kernels
from .articulation_state import _predict, _recover_step
from .rigid import _matrices, _scatter_residual, _scatter_tangent
from .solver_types import _StepFailure
from .state_checks import scalar_view
from .tet_assembly import TetScatterBuffers, assemble_tet_residual_tangent


class Producer(Protocol):
    """One physical term set owning a disjoint set of scalar rows.

    Owner ``evaluate`` methods submit device work; interactions also manage
    once-per-evaluation host generations and capacity summaries.
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

    def evaluate(self, solver: Any, state: Any, dt: float, residual: Any, *, build_tangent: bool) -> None: ...

    def magnitude(self, solver: Any, out: Any) -> None: ...

    def coordinates(self, solver: Any, state: Any, out: Any) -> None: ...

    def assemble(self, solver: Any, assembly: Any, dt: float) -> None: ...

    def scale(self, solver: Any, diagonal: Any, coordinate: np.ndarray) -> None: ...

    def difference(self, solver: Any, accepted: Any, trial: Any, out: Any) -> None: ...

    def domain(self, solver: Any, accepted: Any, candidate: Any, flags: Any) -> None: ...

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
def _scatter_rows(values: wp.array[Any], rows: wp.array[int], output: wp.array[Any]):
    i = wp.tid()
    output[rows[i]] = values[i]


@wp.kernel
def _add_values(source: wp.array[Any], target: wp.array[Any]):
    i = wp.tid()
    target[i] = target[i] + source[i]


@wp.kernel
def _add_offset(values: wp.array[Any], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = output[start + i] + values[i]


@wp.kernel
def _add_absolute(values: wp.array[Any], output: wp.array[Any]):
    i = wp.tid()
    output[i] = output[i] + wp.abs(values[i])


@wp.kernel
def _add_rows(values: wp.array[Any], rows: wp.array[int], output: wp.array[Any]):
    i = wp.tid()
    if rows[i] >= 0:
        output[rows[i]] = output[rows[i]] + values[i]


@wp.kernel
def _write_absolute(values: wp.array[Any], rows: wp.array[int], output: wp.array[Any]):
    i = wp.tid()
    output[rows[i]] = wp.abs(values[i])


@wp.kernel
def _joint_difference(a: wp.array[Any], b: wp.array[Any], rows: wp.array[int], out: wp.array[Any]):
    i = wp.tid()
    out[rows[i]] = b[i] - a[i]


@wp.kernel
def _particle_difference(a: wp.array[Any], b: wp.array[Any], ids: wp.array[int], start: int, out: wp.array[Any]):
    i = wp.tid()
    out[start + i] = b[ids[i // 3]][i % 3] - a[ids[i // 3]][i % 3]


@wp.kernel
def _material_difference(
    a: wp.array[Any],
    b: wp.array[Any],
    ids: wp.array[int],
    theta_a: wp.array[Any],
    theta_b: wp.array[Any],
    rows: wp.array[int],
    out: wp.array[Any],
):
    i = wp.tid()
    if rows[i] >= 0:
        value = out.dtype(0)
        if i < 3 * ids.shape[0]:
            value = b[ids[i // 3]][i % 3] - a[ids[i // 3]][i % 3]
        else:
            j = i - 3 * ids.shape[0]
            value = theta_b[j] - theta_a[j]
        out[rows[i]] = value


@wp.kernel
def _rigid_difference(
    ta: wp.array[Any],
    qa: wp.array[Any],
    tb: wp.array[Any],
    qb: wp.array[Any],
    rows: wp.array2d[int],
    out: wp.array[Any],
):
    i = wp.tid()
    # Left relative quaternion logarithm, using the short representative.
    relative = qb[i] * wp.quat_inverse(qa[i])
    vector = wp.vector(relative[0], relative[1], relative[2])
    scalar = relative[3]
    if scalar < out.dtype(0):
        vector = -vector
    length = wp.length(vector)
    factor = out.dtype(2)
    if length > out.dtype(0):
        factor = out.dtype(2) * wp.atan2(length, wp.abs(scalar)) / length
    for axis in range(3):
        out[rows[i, axis]] = tb[i][axis] - ta[i][axis]
        out[rows[i, axis + 3]] = factor * vector[axis]


@wp.kernel
def _articulation_domain(
    q: wp.array[Any],
    qn: wp.array[Any],
    dof_joint: wp.array[int],
    types: wp.array[int],
    rotations: wp.array[Any],
    rn: wp.array[Any],
    revolute: int,
    half_pi: Any,
    flags: wp.array[int],
):
    i = wp.tid()
    if i < q.shape[0] and types[dof_joint[i]] == revolute:
        if wp.abs(q[i] - qn[i]) >= half_pi:
            wp.atomic_max(flags, 0, 1)
    if i < rotations.shape[0]:
        value = type(q[0])(0.0)
        for a in range(3):
            for b in range(3):
                value += rotations[i][a, b] * rn[i][a, b]
        if not (value > type(q[0])(1.0)):
            wp.atomic_max(flags, 0, 1)


@wp.kernel
def _rotation_domain(rotations: wp.array[Any], original: wp.array[Any], flags: wp.array[int]):
    i = wp.tid()
    value = type(rotations[0][0, 0])(0)
    for a in range(3):
        for b in range(3):
            value += rotations[i][a, b] * original[i][a, b]
    if not (value > type(value)(1)):
        wp.atomic_max(flags, 0, 1)


@wp.kernel
def _contact_reaction(
    ids: wp.array[int],
    rows: wp.array[int],
    residual: wp.array[Any],
    forces: wp.array[Any],
    theta_offset: int,
    reference: bool,
    raw: wp.array[Any],
):
    i = wp.tid()
    if reference:
        if rows[i] >= 0:
            raw[i] += residual[rows[i]]
    elif i < 3 * ids.shape[0]:
        raw[i] -= forces[3 * ids[i // 3] + i % 3]
    else:
        raw[i] -= forces[theta_offset + i - 3 * ids.shape[0]]


@wp.kernel
def _particle_coordinates(x: wp.array[Any], ids: wp.array[int], start: int, output: wp.array[Any]):
    i = wp.tid()
    output[start + i] = type(output[0])(wp.abs(x[ids[i / 3]][i % 3]))


@wp.kernel
def _rigid_coordinates(com: wp.array[Any], rotation: wp.array[Any], rows: wp.array2d[int], output: wp.array[Any]):
    i, j = wp.tid()
    # A rotation step q q_n^-1 near identity turns by about twice its vector part.
    value = type(output[0])(2.0) * type(output[0])(wp.abs(rotation[i][j % 3]))
    if j < 3:
        value = type(output[0])(wp.abs(com[i][j]))
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
    rounding: wp.array[Any],
    output: wp.array[Any],
):
    contact = wp.tid()
    if contact >= count[0]:
        return
    begin, end = supports[contact], supports[contact + 1]
    # Every factor of a contact shares its support columns; read the first.
    for j in range(end - begin):
        wp.atomic_max(output, columns[3 * begin + j], rounding[contact])


class _Producer:
    """Name, owned rows, declared policies and default no-op step methods."""

    capturable = True
    interaction = False
    scale_policy = "diagonal"
    # Step-start predictions read state_in; articulation and rigid prediction run last.
    begin_rank = 0

    def __init__(self, name, rows):
        self.name, self.rows = name, np.asarray(rows, dtype=np.int32)
        self.energies, self.checks, self.finite_values, self.owners = (), (), (), ()

    def begin_step(self, solver, state_in):
        """Freeze step-start inputs and write this producer's predictor into the accepted state."""

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        """Write this producer's trial coordinates from the accepted state and direction."""

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        """Submit residual and registered energy outputs into bound device buffers."""
        raise NotImplementedError(self.name)

    def magnitude(self, solver, out):
        """Add the absolute contributions summed into this producer's last residual rows."""

    def coordinates(self, solver, state, out):
        """Write the magnitude of each owned row's coordinate as this producer's terms read it.

        Terms are formed from the exact step increments, so each row's
        coordinate is its increment.
        """

    def assemble(self, solver, assembly, dt):
        """Write this producer's tangent into its fixed triplet slots."""

    def scale(self, solver, diagonal, coordinate):
        """Write coordinate-policy unit scales for this producer's rows."""

    def difference(self, solver, accepted, trial, out):
        """Write the effective trial displacement of this producer's rows."""

    def domain(self, solver, accepted, candidate, flags):
        """Mark invalid candidate or path geometry in the shared device flags."""

    def publish(self, solver, state):
        """Publish this producer's final history or forces into the accepted state."""


class ArticulationProducer(_Producer):
    """Articulation coordinates with joint, PD, limit and friction terms."""

    begin_rank = 1

    def __init__(self, solver):
        super().__init__("articulation", solver._layout.articulation_indices())
        obj = solver._objective
        self.energies = ((obj, "link_energy", None), (obj, "joint_energy", None))
        self.owners = (obj, solver._kinematics)

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
                solver._linear.q_rows,
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
            _scatter_rows,
            k.count,
            [obj.residual, solver._linear.q_rows, residual],
            device=solver.model.device,
        )

    def magnitude(self, solver, out):
        obj = solver._objective
        wp.launch(
            _add_rows,
            obj.absolute.size,
            [obj.absolute, solver._linear.q_rows, out],
            device=solver.model.device,
        )

    def coordinates(self, solver, state, out):
        k = solver._kinematics
        q = solver._joint_step(state)
        wp.launch(_write_absolute, k.count, [q, solver._linear.q_rows, out], device=solver.model.device)

    def assemble(self, solver, assembly, dt):
        wp.launch(
            kernels._scatter,
            solver._objective.tangent.size,
            [
                solver._objective.tangent.flatten(),
                assembly.aq_actor_dense.flatten(),
                assembly.global_scalar_triplets.values,
                solver._linear.q_pair_rows,
                solver._linear.q_pair_columns,
                solver._diagonal,
            ],
            device=solver.model.device,
        )

    def difference(self, solver, accepted, trial, out):
        wp.launch(
            _joint_difference,
            len(self.rows),
            [solver._joint_step(accepted), solver._joint_step(trial), solver._linear.q_rows, out],
            device=solver.model.device,
        )

    def domain(self, solver, accepted, candidate, flags):
        k, obj = solver._kinematics, solver._objective
        wp.launch(
            _articulation_domain,
            max(k.count, k.bodies),
            [
                k.gather_q(candidate),
                obj.qn,
                k.dof_joint,
                k.types,
                k.rotations,
                obj.rn,
                int(JointType.REVOLUTE),
                solver._dtype(np.pi / 2),
                flags,
            ],
            device=solver.model.device,
        )

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
        self.energies = ((solver._rigid, "energy", None),)
        self.owners = (solver._rigid,)

    def begin_step(self, solver, state_in):
        solver._rigid.predict_pose(state_in, solver._accepted, solver._rigid_steps(solver._accepted), solver._dt)

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        solver._rigid.trial(solver._rigid_steps(accepted), trial, solver._rigid_steps(trial), direction, alpha)

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        rigid = solver._rigid
        rigid.evaluate(state, solver._rigid_steps(state), dt, build_tangent)
        wp.launch(
            _scatter_residual, (rigid.count, 6), [rigid.rows, rigid.residual, residual], device=solver.model.device
        )

    def magnitude(self, solver, out):
        rigid = solver._rigid
        if rigid.count:
            wp.launch(
                _scatter_residual, (rigid.count, 6), [rigid.rows, rigid.absolute, out], device=solver.model.device
            )

    def coordinates(self, solver, state, out):
        rigid = solver._rigid
        if rigid.count:
            steps = solver._rigid_steps(state)
            wp.launch(_rigid_coordinates, (rigid.count, 6), [*steps, rigid.rows, out], device=solver.model.device)

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

    def difference(self, solver, accepted, trial, out):
        wp.launch(
            _rigid_difference,
            solver._rigid.count,
            [*solver._rigid_steps(accepted), *solver._rigid_steps(trial), solver._rigid.rows, out],
            device=solver.model.device,
        )

    def domain(self, solver, accepted, candidate, flags):
        rigid = solver._rigid
        wp.launch(
            _matrices,
            rigid.count,
            [candidate._monolithic_coupled.rigid_rotation, rigid.rotations],
            device=solver.model.device,
        )
        wp.launch(_rotation_domain, rigid.count, [rigid.rotations, rigid.rn, flags], device=solver.model.device)

    def publish(self, solver, state):
        solver._rigid.recover(state, solver._rigid_steps(state), solver._dt)


class TetProducer(_Producer):
    """Dynamic tet particles with inertia, elasticity and frozen external forces."""

    def __init__(self, solver):
        super().__init__("tet", solver._layout.indices("tet"))
        self.energies = ((solver._tet, "parts", None),)
        self.checks = ((solver._tet, "flags", {1: solver.Status.TET_INVERSION}, "Invalid tet candidate"),)
        self.owners = (solver._tet, solver._tet.physics)

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
        wp.launch(
            _scatter_values,
            3 * tet.ids.size,
            [scalar_view(tet.residual), solver._layout.x_scalar_offset, residual],
            device=solver.model.device,
        )

    def magnitude(self, solver, out):
        absolute = scalar_view(solver._tet.absolute)
        wp.launch(
            _add_offset, absolute.size, [absolute, solver._layout.x_scalar_offset, out], device=solver.model.device
        )

    def coordinates(self, solver, state, out):
        # Tet terms read the exact step displacement, so its rounding, not that
        # of the absolute position, bounds how closely a row can be resolved.
        ids = solver._tet.ids
        x = solver._displacement(state)
        wp.launch(
            _particle_coordinates,
            3 * ids.size,
            [x, ids, solver._layout.x_scalar_offset, out],
            device=solver.model.device,
        )

    def assemble(self, solver, assembly, dt):
        offset = solver._objective.tangent.size
        scatter = TetScatterBuffers(
            assembly.ax_internal_triplets.values,
            assembly.global_scalar_triplets.values[offset : offset + 9 * assembly.ax_internal_triplets.capacity],
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

    def difference(self, solver, accepted, trial, out):
        wp.launch(
            _particle_difference,
            len(self.rows),
            [
                solver._displacement(accepted),
                solver._displacement(trial),
                solver._tet.ids,
                solver._layout.x_scalar_offset,
                out,
            ],
            device=solver.model.device,
        )


class _MaterialProducer(_Producer):
    """One cloth or cable object expanded into a typed device execution plan."""

    capturable = True
    scale_policy = "coordinate"

    def __init__(self, solver, kind, block):
        source = block.actor.source_index
        super().__init__(f"{kind}_{source}", solver._layout.indices(kind, source))
        self.block = block
        self.checks = ((block.objective, "flags", {}, f"Invalid {kind} geometry or projection"),)
        self.finite_values = ((block, "raw"),)
        self.owners = (block, block.objective)

    def assemble(self, solver, assembly, dt):
        self.block.assemble(assembly, dt)

    def magnitude(self, solver, out):
        block = self.block
        wp.launch(_add_rows, block.absolute.size, [block.absolute, block.rows, out], device=solver.model.device)

    def theta(self, solver, state):
        """Bind existing angle storage; cloth kernels have no angle rows to read."""
        return solver._theta_step(state)

    def coordinates(self, solver, state, out):
        # Shell and rod terms read the exact step increments, so their rounding,
        # not that of the absolute coordinates, bounds how closely a row can be
        # resolved.
        block = self.block
        x = solver._displacement(state)
        wp.launch(
            _material_coordinates,
            block.rows.size,
            [x, block.ids, self.theta(solver, state), block.rows, out],
            device=solver.model.device,
        )

    def difference(self, solver, accepted, trial, out):
        block = self.block
        wp.launch(
            _material_difference,
            block.rows.size,
            [
                solver._displacement(accepted),
                solver._displacement(trial),
                block.ids,
                self.theta(solver, accepted),
                self.theta(solver, trial),
                block.rows,
                out,
            ],
            device=solver.model.device,
        )

    def domain(self, solver, accepted, candidate, flags):
        self.block.check_path(solver._original, solver._displacement(accepted), solver._displacement(candidate), flags)


class ClothProducer(_MaterialProducer):
    """One cloth object."""

    def __init__(self, solver, block):
        super().__init__(solver, "cloth", block)
        self.energies = ((block.objective, "energy", None), (block, "external_energy", None))

    def begin_step(self, solver, state_in):
        cloth, original, accepted = self.block, solver._original, solver._accepted
        start, step = solver._displacement(original), solver._displacement(accepted)
        cloth.begin_step(original, solver._frozen_force)
        cloth.candidate(start, original, accepted, step, solver._delta, 0.0, solver._dt, predict=True)
        cloth.path_status.zero_()
        cloth.check_path(original, start, step, cloth.path_status)
        # An inertial guess is optional; retain the valid input chart on device.
        cloth.candidate(start, original, accepted, step, solver._delta, 0.0, solver._dt, fallback=True)

    def candidate(self, solver, accepted, trial, direction, alpha, dt):
        self.block.candidate(
            solver._displacement(accepted), solver._original, trial, solver._displacement(trial), direction, alpha, dt
        )

    def evaluate(self, solver, state, dt, residual, *, build_tangent):
        self.block.evaluate(
            solver._displacement(state), solver._frozen_gravity, dt, residual, build_tangent=build_tangent
        )


class CableProducer(_MaterialProducer):
    """One cable object; theta rows use a fixed inverse-length scale."""

    def __init__(self, solver, block):
        super().__init__(solver, "cable", block)
        obj = block.objective
        self.energies = (
            (obj, "stretch_energy", None),
            (obj, "bend_twist_energy", None),
            (block, "external_energy", None),
        )

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
        self.block.evaluate(
            state,
            solver._displacement(state),
            solver._theta_step(state),
            solver._original,
            solver._cable_torque,
            solver._frozen_gravity,
            dt,
            residual,
            build_tangent=build_tangent,
        )

    def scale(self, solver, diagonal, coordinate):
        rows = self.block.rows_host[3 * self.block.ids.size :]
        coordinate[rows[rows >= 0]] = 1 / float(self.block.reference["length_scale"])

    @staticmethod
    def steps(solver, state):
        return solver._displacement(state), solver._theta_step(state)

    def theta(self, solver, state):
        start = self.block.theta_offset
        theta = solver._theta_step(state)
        return theta[start : start + self.block.objective.lengths.size]


class ContactProducer(_Producer):
    """Spatial contact; it adds coupling terms to rows owned by other producers."""

    scale_policy = "none"
    interaction = True

    def __init__(self, solver):
        super().__init__("contact", np.empty(0, dtype=np.int32))
        contact = solver._contact
        self.energies = ((contact, "energy", "absolute_energy"),)
        self.checks = ((contact, "flags", {}, "Invalid contact geometry"),)
        surface = contact._batch.material_surface
        if surface is not None:
            self.checks += ((surface, "flags", {}, "Invalid cable contact frame"),)

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
                surface.update(state, solver._displacement(state), check=False)
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
        if diagnostics.get("tainted", False):
            raise _StepFailure(solver.Status.CONTACT_OVERFLOW, "Incomplete contact evaluation")

        def accumulate():
            wp.launch(_add_values, residual.size, [contact.residual, residual], device=solver.model.device)
            # Include fixed-point reactions as well as free-row contact residuals.
            for block in solver._cloths + solver._cables:
                theta_offset = 3 * solver.model.particle_count + 6 * solver.model.body_count
                theta_offset += int(block.reference.get("theta_offset", 0))
                wp.launch(
                    _contact_reaction,
                    block.raw.size,
                    [
                        block.ids,
                        block.rows,
                        contact.residual,
                        contact.pool.force.result,
                        theta_offset,
                        reference_mode,
                        block.raw,
                    ],
                    device=solver.model.device,
                )

        buffers = [contact.residual, contact.pool.force.result, residual]
        buffers.extend(
            array for block in solver._cloths + solver._cables for array in (block.ids, block.rows, block.raw)
        )
        # This adds to the freshly overwritten owner residuals exactly once.
        solver._graphs.run("problem_interaction_rows", (reference_mode,), buffers, accumulate, warmup=False)

    def magnitude(self, solver, out):
        residual = (
            solver._contact.pool.magnitude
            if solver._contact._alignment_policy == "frozen"
            else solver._contact.residual
        )
        wp.launch(_add_absolute, residual.size, [residual, out], device=solver.model.device)

    def coordinates(self, solver, state, out):
        # A contact's gap reads its local point, the step-start point plus the
        # slip. Raise each coordinate it reads to the size whose rounding moves
        # the gap as much as rounding that point. Raising (not adding) keeps
        # the result independent of order.
        records = solver._contact.pool.records
        wp.launch(
            _raise_coordinates,
            records.rounding.size,
            [records.count, records.support_offsets, records.factor_columns, records.rounding, out],
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
