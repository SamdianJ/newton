# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
import warnings
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.articulation import (
    MonolithicArticulationWorkspace,
    eval_articulation_passive_candidate,
    scatter_articulation_actor_tangent,
)
from newton._src.solvers.monolithic.linear import (
    MonolithicLinearCapacities,
    MonolithicLinearGeneration,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
    MonolithicLinearWorkspace,
)
from newton._src.solvers.monolithic.solver_monolithic import (
    SolverMonolithic,
    _build_layout,
    _Candidate,
    _FrozenStepInputs,
    _StepTransaction,
)
from newton._src.solvers.monolithic.tet import (
    TetScatterBuffers,
    assemble_tet_residual_tangent,
    build_tet_triplet_pattern,
    create_tet_assembly_workspace,
)
from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _fixture(device):
    return build_tiny_cpu_fixture(device=device)


def test_layout_and_candidate(test, device):
    """Verify q/x ordering, BE recovery and trials rebuilt from accepted coordinates."""
    fixture = _fixture(device)
    model, state = fixture.model, fixture.state
    state.joint_qd.assign(np.array([0.2, -0.1], dtype=np.float32))
    state.particle_qd.assign(np.full((4, 3), 0.03, dtype=np.float32))
    layout = _build_layout(model)
    test.assertEqual((layout.q_dof_count, layout.dynamic_particle_count, layout.scalar_dof_count), (2, 4, 14))
    np.testing.assert_array_equal(layout.q_dof_to_joint_q.numpy(), [0, 1])
    np.testing.assert_array_equal(layout.q_dof_to_joint_qd.numpy(), [0, 1])
    np.testing.assert_array_equal(layout.particle_to_dynamic.numpy(), [0, 1, 2, 3])
    q0, x0 = state.joint_q.numpy(), state.particle_q.numpy()
    vq0, vx0 = state.joint_qd.numpy(), state.particle_qd.numpy()
    accepted, trial = _Candidate(model, layout), _Candidate(model, layout)
    accepted.load_predictor(state, 0.1)
    expected = np.concatenate((q0 + 0.1 * vq0, (x0 + 0.1 * vx0).ravel()))
    np.testing.assert_allclose(accepted.z.numpy(), expected, atol=1e-7)
    test.assertEqual(accepted.generation, 0)
    np.testing.assert_allclose(accepted.qdd.numpy(), 0.0, atol=1e-5)
    delta = wp.array(np.linspace(-0.001, 0.001, 14, dtype=np.float32), device=device)
    for alpha in (1.0, 0.5, 0.25):
        trial.form_trial(accepted, delta, alpha, 0.1)
        z = expected + alpha * delta.numpy()
        np.testing.assert_allclose(trial.z.numpy(), z, atol=1e-7)
        np.testing.assert_allclose(trial.state.joint_qd.numpy(), (z[:2] - q0) / 0.1, atol=1e-6)
        np.testing.assert_allclose(trial.qdd.numpy(), (z[:2] - q0 - 0.1 * vq0) / 0.01, atol=1e-5)
        np.testing.assert_allclose(trial.state.particle_qd.numpy(), (z[2:].reshape(4, 3) - x0) / 0.1, atol=1e-6)
        np.testing.assert_allclose(
            trial.particle_acceleration.numpy(), (z[2:].reshape(4, 3) - x0 - 0.1 * vx0) / 0.01, atol=1e-5
        )
    test.assertEqual(trial.generation, 3)
    reference = model.state()
    newton.eval_fk(model, trial.state.joint_q, trial.state.joint_qd, reference)
    np.testing.assert_allclose(trial.state.body_q.numpy(), reference.body_q.numpy(), atol=1e-6)
    np.testing.assert_allclose(trial.state.body_qd.numpy(), reference.body_qd.numpy(), atol=1e-6)
    np.testing.assert_array_equal(state.joint_q.numpy(), q0)
    np.testing.assert_array_equal(state.particle_q.numpy(), x0)
    test.assertNotEqual(accepted.state.particle_q.ptr, state.particle_q.ptr)
    test.assertNotEqual(trial.state.particle_q.ptr, accepted.state.particle_q.ptr)


def test_frozen_inputs(test, device):
    """Snapshot all applied loads independently of later control and state edits."""
    fixture = _fixture(device)
    fixture.control.joint_f.fill_(2.0)
    fixture.state.body_f.fill_(wp.spatial_vector(3.0))
    fixture.state.particle_f.fill_(wp.vec3(4.0))
    frozen = _FrozenStepInputs(fixture.model)
    frozen.snapshot(fixture.state, fixture.control)
    fixture.control.joint_f.zero_()
    fixture.state.body_f.zero_()
    fixture.state.particle_f.zero_()
    np.testing.assert_array_equal(frozen.joint_f.numpy(), 2.0)
    np.testing.assert_array_equal(frozen.body_f.numpy(), 3.0)
    np.testing.assert_array_equal(frozen.particle_f.numpy(), 4.0)


def test_terminal_transaction(test, device):
    """Verify rollback, safe soft stop, generation and one terminal warning."""
    for inplace in (False, True):
        for status in SolverMonolithic.Status:
            if status == SolverMonolithic.Status.NOT_RUN:
                continue
            fixture = _fixture(device)
            state_in = fixture.state
            state_out = state_in if inplace else fixture.state_next
            original = state_in.particle_q.numpy().copy()
            transaction = _StepTransaction(fixture.model, _build_layout(fixture.model))
            transaction.begin(state_in, state_out, fixture.control, 0.1)
            with test.assertRaises(RuntimeError):
                transaction.accept_trial()
            test.assertEqual(transaction.step_generation, 1)
            delta = wp.full(14, 0.001, dtype=float, device=device)
            transaction.trial.form_trial(transaction.accepted, delta, 1.0, 0.1)
            transaction.accept_trial()
            with test.assertRaises(RuntimeError):
                transaction.accept_trial()
            test.assertEqual(transaction.accepted.generation, 1)
            expected = transaction.accepted.state.particle_q.numpy().copy()
            kwargs = {
                "accepted_safe": True,
                "failure_reason": None if status == SolverMonolithic.Status.SUCCESS else "synthetic terminal",
                "rho": 0.125,
                "nonlinear_iterations": 2,
                "linear_iterations": 3,
            }
            if status == SolverMonolithic.Status.SUCCESS:
                with test.assertNoLogs("newton._src.solvers.monolithic.solver_monolithic", level="WARNING"):
                    stats = transaction.finish_state(status, **kwargs)
            else:
                with test.assertLogs("newton._src.solvers.monolithic.solver_monolithic", level="WARNING") as logs:
                    stats = transaction.finish_state(status, **kwargs)
                test.assertEqual(len(logs.output), 1)
                for field in (
                    status.value,
                    "synthetic terminal",
                    "converged=",
                    "rolled_back=",
                    "step_generation=1",
                    "nonlinear_iterations=2",
                    "linear_iterations=3",
                    "rho=0.125",
                ):
                    test.assertIn(field, logs.output[0])
            rollback = status not in (SolverMonolithic.Status.SUCCESS, SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS)
            np.testing.assert_array_equal(state_out.particle_q.numpy(), original if rollback else expected)
            if not inplace:
                np.testing.assert_array_equal(state_in.particle_q.numpy(), original)
            test.assertEqual(stats.rolled_back, rollback)
            test.assertEqual(stats.converged, status == SolverMonolithic.Status.SUCCESS)
            test.assertIs(transaction.last_stats, stats)
            with test.assertRaises(FrozenInstanceError):
                stats.rolled_back = False
            with test.assertRaises(TypeError):
                stats.timings["step"] = 1.0
            with test.assertRaises(RuntimeError):
                transaction.finish_state(status, **kwargs)
            transaction.begin(state_out, state_out, None, 0.1)
            test.assertEqual(transaction.step_generation, 2)
            test.assertEqual(transaction.accepted.generation, 0)


def test_contract_rejections(test, device):
    """Reject invalid dt and moving fixed nodes before candidate mutation."""
    fixture = _fixture(device)
    model, state = fixture.model, fixture.state
    inv_mass = model.particle_inv_mass.numpy().copy()
    inv_mass[0] = 0.0
    model.particle_inv_mass.assign(inv_mass)
    layout = _build_layout(model)
    np.testing.assert_array_equal(layout.particle_to_dynamic.numpy(), [-1, 0, 1, 2])
    candidate = _Candidate(model, layout)
    before = candidate.state.particle_q.numpy().copy()
    for dt in (0.0, -1.0, float("nan"), float("inf")):
        with test.assertRaises(ValueError):
            candidate.load_predictor(state, dt)
        np.testing.assert_array_equal(candidate.state.particle_q.numpy(), before)
    v = state.particle_qd.numpy()
    v[0, 0] = np.nextafter(np.float32(0.0), np.float32(1.0))
    state.particle_qd.assign(v)
    with test.assertRaisesRegex(ValueError, "Dirichlet"):
        candidate.load_predictor(state, 0.1)
    np.testing.assert_array_equal(candidate.state.particle_q.numpy(), before)
    state.particle_qd.zero_()
    candidate.load_predictor(state, 0.1)
    trial = _Candidate(model, layout)
    trial.form_trial(candidate, wp.full(11, 0.01, dtype=float, device=device), 1.0, 0.1)
    np.testing.assert_array_equal(trial.state.particle_q.numpy()[0], state.particle_q.numpy()[0])
    np.testing.assert_array_equal(trial.state.particle_qd.numpy()[0], [0.0, 0.0, 0.0])
    transaction = _StepTransaction(model, layout)
    transaction.begin(state, fixture.state_next, None, 0.1)
    with test.assertRaisesRegex(ValueError, "safe"):
        transaction.finish_state(SolverMonolithic.Status.NONLINEAR_MAX_ITERATIONS, accepted_safe=False)


def test_scope_world(test, device):
    """Allow one global or local label and reject mixed participating worlds."""
    fixture = _fixture(device)
    model = fixture.model
    for label in (-1, 0, 2):
        model.world_count = max(0, label + 1)
        model.body_world.fill_(label)
        model.shape_world.fill_(label)
        model.particle_world.fill_(label)
        _build_layout(model)
    model.particle_world.fill_(-1)
    with test.assertRaisesRegex(ValueError, "world"):
        _build_layout(model)


def test_fixed_joint_and_non_target(test, device):
    """Exclude fixed joints and preserve fixed and unrelated particle state."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    child = builder.add_link(mass=1.0, inertia=wp.mat33(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    fixed = builder.add_joint_fixed(-1, root)
    moving = builder.add_joint_revolute(root, child, axis=newton.Axis.Z)
    builder.add_articulation([fixed, moving])
    builder.add_particle(pos=(1.0, 2.0, 3.0), vel=(4.0, 5.0, 6.0), mass=1.0)
    for p in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        builder.add_particle(pos=p, vel=(0, 0, 0), mass=1.0)
    builder.add_tetrahedron(1, 2, 3, 4)
    model = builder.finalize(device=device)
    inv_mass = model.particle_inv_mass.numpy().copy()
    inv_mass[1] = 0.0
    model.particle_inv_mass.assign(inv_mass)
    worlds = model.particle_world.numpy()
    worlds[0] = 5
    model.particle_world.assign(worlds)
    layout = _build_layout(model)
    test.assertEqual(layout.scalar_dof_count, 10)
    np.testing.assert_array_equal(layout.q_dof_to_joint_q.numpy(), [0])
    np.testing.assert_array_equal(layout.q_dof_to_joint_qd.numpy(), [0])
    np.testing.assert_array_equal(layout.particle_to_dynamic.numpy(), [-1, -1, 0, 1, 2])
    np.testing.assert_array_equal(layout.body_to_link_index.numpy(), [0, 1])
    np.testing.assert_array_equal(layout.body_to_articulation.numpy(), [0, 0])
    state = model.state()
    accepted, trial = _Candidate(model, layout), _Candidate(model, layout)
    accepted.load_predictor(state, 0.1)
    trial.form_trial(accepted, wp.full(10, 0.01, dtype=float, device=device), 1.0, 0.1)
    trial.commit(state)
    np.testing.assert_array_equal(state.particle_q.numpy()[:2], [(1, 2, 3), (0, 0, 0)])
    np.testing.assert_array_equal(state.particle_qd.numpy()[:2], [(4, 5, 6), (0, 0, 0)])
    np.testing.assert_array_equal(trial.particle_acceleration.numpy()[:2], 0.0)


def test_scope_topology(test, device):
    """Reject unsupported joints, invalid tets and invalid mass mappings."""
    fixture = _fixture(device)
    model = fixture.model
    types = model.joint_type.numpy().copy()
    model.joint_type.fill_(int(newton.JointType.FREE))
    with test.assertRaisesRegex(ValueError, "FIXED"):
        _build_layout(model)
    model.joint_type.assign(types)
    parents = model.joint_parent.numpy().copy()
    model.joint_parent.fill_(-1)
    with test.assertRaisesRegex(ValueError, "tree"):
        _build_layout(model)
    model.joint_parent.assign(parents)
    tets = model.tet_indices.numpy().copy()
    model.tet_indices.assign([[0, 0, 2, 3]])
    with test.assertRaisesRegex(ValueError, "topology"):
        _build_layout(model)
    model.tet_indices.assign(tets)
    inv_mass = model.particle_inv_mass.numpy().copy()
    for invalid in (-1.0, float("nan"), float("inf")):
        bad = inv_mass.copy()
        bad[0] = invalid
        model.particle_inv_mass.assign(bad)
        with test.assertRaisesRegex(ValueError, "inverse masses"):
            _build_layout(model)
    model.particle_inv_mass.assign(inv_mass)
    layout = _build_layout(model)
    candidate = _Candidate(model, layout)
    candidate.load_predictor(fixture.state, 0.1)
    trial = _Candidate(model, layout)
    for delta, alpha, dt in (
        (wp.zeros(13, device=device), 1.0, 0.1),
        (wp.zeros(14, device=device), 0.0, 0.1),
        (wp.zeros(14, device=device), 1.0, 0.2),
    ):
        with test.assertRaises(ValueError):
            trial.form_trial(candidate, delta, alpha, dt)
    bad_state = model.state()
    bad_state.particle_q = wp.zeros(3, dtype=wp.vec3, device=device)
    transaction = _StepTransaction(model, layout)
    with test.assertRaisesRegex(ValueError, "particle_q"):
        transaction.begin(fixture.state, bad_state, fixture.control, 0.1)
    test.assertEqual(transaction.step_generation, 0)


def test_state_extension_rejection(test, device):
    """Reject optional and custom input/output arrays before any transaction mutation."""
    for extension in ("body_qdd", "mujoco:qfrc_actuator", "custom"):
        for role in ("input", "output"):
            fixture = _fixture(device)
            model = fixture.model
            transaction = _StepTransaction(model, _build_layout(model))
            if extension == "custom":
                extended = model.state()
                extended.custom = wp.ones(1, device=device)
            else:
                model.request_state_attributes(extension)
                extended = model.state()
            state_in = extended if role == "input" else fixture.state
            state_out = extended if role == "output" else fixture.state_next
            state_in.body_f.fill_(wp.spatial_vector(9.0))
            state_out.particle_q.fill_(wp.vec3(42.0))
            before = {
                name: value.numpy().copy() for name, value in vars(state_out).items() if isinstance(value, wp.array)
            }
            with test.assertRaisesRegex(ValueError, "Unsupported.*state"):
                transaction.begin(state_in, state_out, fixture.control, 0.1)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(state_out, name).numpy(), value)
            np.testing.assert_array_equal(transaction.inputs.body_f.numpy(), 0.0)
            test.assertEqual(transaction.step_generation, 0)
            test.assertEqual(transaction.last_stats.status, SolverMonolithic.Status.NOT_RUN)


def test_terminal_destination_validation(test, device):
    """Keep all output arrays unchanged if its schema changes after begin."""
    for status in (SolverMonolithic.Status.SUCCESS, SolverMonolithic.Status.NONFINITE):
        for extension in ("body_qdd", "mujoco", "particle_qd"):
            fixture = _fixture(device)
            transaction = _StepTransaction(fixture.model, _build_layout(fixture.model))
            out = fixture.state_next
            transaction.begin(fixture.state, out, fixture.control, 0.1)
            out.particle_q.fill_(wp.vec3(42.0))
            if extension == "body_qdd":
                out.body_qdd = wp.zeros(fixture.model.body_count, dtype=wp.spatial_vector, device=device)
            elif extension == "mujoco":
                out.mujoco = newton.Model.AttributeNamespace("mujoco")
                out.mujoco.qfrc_actuator = wp.zeros(fixture.model.joint_dof_count, device=device)
            else:
                out.particle_qd = wp.zeros(1, dtype=wp.vec3, device=device)
            before = {name: value.numpy().copy() for name, value in vars(out).items() if isinstance(value, wp.array)}
            with test.assertRaises(ValueError):
                transaction.finish_state(status, accepted_safe=True)
            for name, value in before.items():
                np.testing.assert_array_equal(getattr(out, name).numpy(), value)
            test.assertEqual(transaction.last_stats.status, SolverMonolithic.Status.NOT_RUN)


def test_pipeline_ownership(test, device):
    """Allocate separate final/trial buffers and reject external contacts before mutation."""
    fixture = _fixture(device)
    pipeline = MonolithicCollisionPipeline(fixture.model)
    solver = SolverMonolithic(fixture.model, collision_pipeline=pipeline, contact_stiffness=1.0)
    test.assertIs(solver.collision_pipeline, pipeline)
    test.assertIsNot(solver.contacts, solver._trial_contacts)
    pipeline.validate_contacts(solver.contacts)
    pipeline.validate_contacts(solver._trial_contacts)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NOT_RUN)
    for slot in solver.CollisionSlot:
        test.assertEqual(solver.collision_frequency_type[slot], solver.CollisionFrequencyType.NONE)
    fixture.state_next.particle_q.fill_(wp.vec3(42.0))
    before = fixture.state_next.particle_q.numpy().copy()
    generation = solver.contacts.contact_generation.numpy().copy()
    for external in (solver.contacts, solver._trial_contacts, pipeline.contacts()):
        with test.assertRaisesRegex(ValueError, "must be None"):
            solver.step(fixture.state, fixture.state_next, fixture.control, external, 0.01)
    with test.assertRaisesRegex(ValueError, "No valid final"):
        solver.update_contacts(solver.contacts, fixture.state_next)
    np.testing.assert_array_equal(fixture.state_next.particle_q.numpy(), before)
    np.testing.assert_array_equal(solver.contacts.contact_generation.numpy(), generation)
    test.assertEqual(solver.last_stats.status, SolverMonolithic.Status.NOT_RUN)


def test_constructor_validation(test, device):
    """Reject invalid pipelines and solver parameters before allocating owned contacts."""
    fixture = _fixture(device)
    pipeline = MonolithicCollisionPipeline(fixture.model)
    other = _fixture(device)
    for invalid in (None, newton.CollisionPipeline(fixture.model), MonolithicCollisionPipeline(other.model)):
        with test.assertRaises(ValueError):
            SolverMonolithic(fixture.model, collision_pipeline=invalid, contact_stiffness=1.0)
    for name in ("contact_stiffness", "linear_tolerance"):
        for value in (0.0, -1.0, float("nan"), float("inf"), 1.0e-50, 1.0e100):
            kwargs = {"contact_stiffness": 1.0, name: value}
            with (
                warnings.catch_warnings(),
                patch.object(pipeline, "contacts", side_effect=AssertionError("premature allocation")),
            ):
                warnings.simplefilter("error", RuntimeWarning)
                with test.assertRaises(ValueError):
                    SolverMonolithic(fixture.model, collision_pipeline=pipeline, **kwargs)
    for name in ("newton_max_iterations", "line_search_max_iterations", "linear_max_iterations"):
        for value in (0, -1, 1.5, True):
            with patch.object(pipeline, "contacts", side_effect=AssertionError("premature allocation")):
                with test.assertRaises(ValueError):
                    SolverMonolithic(fixture.model, collision_pipeline=pipeline, contact_stiffness=1.0, **{name: value})
    for name in ("joint_armature", "tet_materials"):
        values = getattr(fixture.model, name).numpy().copy()
        invalid = values.copy()
        if name == "tet_materials":
            invalid[:, 2] = 1.0
        else:
            invalid[:] = 1.0
        getattr(fixture.model, name).assign(invalid)
        with patch.object(pipeline, "contacts", side_effect=AssertionError("premature allocation")):
            with test.assertRaises(ValueError):
                SolverMonolithic(fixture.model, collision_pipeline=pipeline, contact_stiffness=1.0)
        getattr(fixture.model, name).assign(values)


def test_physical_owner_assembly(test, device):
    """Assemble real articulation/tet producers into one positive-definite 14-DoF BSR."""
    fixture = _fixture(device)
    model = fixture.model
    layout = _build_layout(model)
    nq = layout.q_dof_count
    pattern = build_tet_triplet_pattern(model.tet_indices.numpy(), layout.particle_to_dynamic.numpy(), x_dof_start=nq)
    global_rows = np.concatenate((np.repeat(np.arange(nq), nq), pattern.global_rows)).astype(np.int32)
    global_columns = np.concatenate((np.tile(np.arange(nq), nq), pattern.global_columns)).astype(np.int32)
    linear = MonolithicLinearWorkspace(
        MonolithicLinearLayout(nq, layout.dynamic_particle_count),
        MonolithicLinearCapacities(len(global_rows), len(pattern.ax_rows), 0, 0, 0),
        layout.particle_to_dynamic,
        device=device,
    )
    linear._set_fixed_triplet_pattern(
        global_rows=global_rows,
        global_columns=global_columns,
        internal_rows=pattern.ax_rows,
        internal_columns=pattern.ax_columns,
    )
    articulation = MonolithicArticulationWorkspace(model)
    tet = create_tet_assembly_workspace(pattern, tet_count=model.tet_count, device=device)
    residual = wp.zeros(layout.dynamic_particle_count, dtype=wp.vec3, device=device)
    matrices = []
    for sequence, stretch in enumerate((1.0, 1.1, 1.0)):
        state = model.state()
        state.assign(fixture.state)
        state.particle_q.assign(fixture.state.particle_q.numpy() * np.array([stretch, 1.0, 1.0]))
        state.joint_q.assign(fixture.state.joint_q.numpy() + np.array([stretch - 1.0, 0.0]))
        eval_articulation_passive_candidate(model, state, articulation)
        generation = MonolithicLinearGeneration(1, sequence, 0, sequence)
        assembly = linear.begin_assembly(generation)
        triplets = assembly.global_scalar_triplets
        wp.launch(
            scatter_articulation_actor_tangent,
            (nq, nq),
            [
                0,
                nq,
                0,
                0,
                100.0,
                articulation.M,
                assembly.aq_actor_dense,
                triplets.rows,
                triplets.columns,
                triplets.values,
            ],
            device=device,
        )
        assemble_tet_residual_tangent(
            model,
            candidate_particle_q=state.particle_q,
            particle_q_n=fixture.state.particle_q,
            particle_qd_n=fixture.state.particle_qd,
            frozen_particle_f=fixture.state.particle_f,
            dynamic_particle_ids=layout.dynamic_particle_ids,
            particle_to_dynamic=layout.particle_to_dynamic,
            residual_x=residual,
            dt=0.1,
            min_det_f_guard=0.2,
            workspace=tet,
            scatter=TetScatterBuffers(assembly.ax_internal_triplets.values, triplets.values[nq * nq :]),
        )
        test.assertEqual(int(tet.failure_flags.numpy()[0]), 0)
        test.assertGreaterEqual(float(tet.min_det_f.numpy()[0]), 0.2)
        result = linear.finalize_assembly(generation=generation)
        test.assertEqual(result.status, MonolithicLinearStatus.SUCCESS)
        test.assertEqual(result.global_triplet_count, len(global_rows))
        matrix = linear.densify_for_test(generation=generation).raw_matrix
        owner = np.zeros((14, 14))
        owner[:nq, :nq] = assembly.aq_actor_dense.numpy()
        blocks = assembly.ax_internal_triplets.values.numpy()
        for row, column, block in zip(pattern.ax_rows, pattern.ax_columns, blocks, strict=True):
            owner[nq + 3 * row : nq + 3 * row + 3, nq + 3 * column : nq + 3 * column + 3] += block
        np.testing.assert_allclose(matrix, owner, rtol=5e-5, atol=1e-6)
        np.testing.assert_array_equal(matrix[:nq, nq:], 0.0)
        np.testing.assert_array_equal(matrix[nq:, :nq], 0.0)
        np.testing.assert_allclose(matrix, matrix.T, rtol=5e-5, atol=1e-6)
        test.assertGreater(np.linalg.eigvalsh(matrix)[0], 0.0)
        test.assertTrue(np.isfinite(residual.numpy()).all())
        matrices.append(matrix)
    np.testing.assert_allclose(matrices[0], matrices[2], rtol=5e-5, atol=1e-6)
    test.assertGreater(np.linalg.norm(matrices[0] - matrices[1]), 0.0)


class TestSolverMonolithic(unittest.TestCase):
    pass


for test_function in (
    test_layout_and_candidate,
    test_frozen_inputs,
    test_terminal_transaction,
    test_contract_rejections,
    test_scope_world,
    test_fixed_joint_and_non_target,
    test_scope_topology,
    test_state_extension_rejection,
    test_terminal_destination_validation,
    test_pipeline_ownership,
    test_constructor_validation,
    test_physical_owner_assembly,
):
    add_function_test(TestSolverMonolithic, test_function.__name__, test_function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main(verbosity=2)
