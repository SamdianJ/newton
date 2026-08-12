# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for VBD cloth bending plasticity."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _hinge_angle(positions: np.ndarray, edge: np.ndarray) -> float:
    opposite0, opposite1, vertex0, vertex1 = (int(index) for index in edge)
    normal0 = np.cross(positions[vertex0] - positions[opposite0], positions[vertex1] - positions[opposite0])
    normal1 = np.cross(positions[vertex1] - positions[opposite1], positions[vertex0] - positions[opposite1])
    edge_direction = positions[vertex1] - positions[vertex0]
    normal0 /= np.linalg.norm(normal0)
    normal1 /= np.linalg.norm(normal1)
    edge_direction /= np.linalg.norm(edge_direction)
    return float(
        np.arctan2(
            np.dot(np.cross(normal0, normal1), edge_direction),
            np.clip(np.dot(normal0, normal1), -1.0, 1.0),
        )
    )


def _build_hinge_model(
    device, yield_angle: float, hardening: float, plastic_enabled: bool = True, edge_stiffness: float = 0.0
):
    vertices = [
        wp.vec3(-0.5, -0.5, 0.0),
        wp.vec3(0.5, -0.5, 0.0),
        wp.vec3(0.5, 0.5, 0.0),
        wp.vec3(-0.5, 0.5, 0.0),
    ]
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0, 0.0, 0.0))
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=[0, 1, 2, 0, 2, 3],
        density=1.0,
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        edge_ke=edge_stiffness,
        edge_kd=0.0,
        bending_plasticity=newton.ClothPlasticity(
            yield_angle=yield_angle,
            hardening_modulus=hardening,
        ),
    )

    hinge_edge = None
    for edge_index, edge in enumerate(builder.edge_indices):
        if {int(edge[2]), int(edge[3])} == {0, 2}:
            hinge_edge = edge_index
            break
    if hinge_edge is None:
        raise RuntimeError("Could not find the shared hinge edge")

    for edge_index in range(len(builder.edge_plastic_mask)):
        builder.edge_plastic_mask[edge_index] = int(plastic_enabled and edge_index == hinge_edge)

    builder.color(include_bending=True)
    return builder.finalize(device=device), hinge_edge


def _run_hinge_step(device, yield_angle: float, hardening: float, plastic_enabled: bool = True):
    model, hinge_edge = _build_hinge_model(device, yield_angle, hardening, plastic_enabled)
    state_in = model.state()
    state_out = model.state()
    positions = model.particle_q.numpy()
    positions[3, 2] = 0.75
    state_in.particle_q.assign(wp.array(positions, dtype=wp.vec3, device=device))

    initial_rest_angle = float(state_in.edge_rest_angle.numpy()[hinge_edge])
    initial_yield_angle = float(state_in.edge_plastic_yield_angle.numpy()[hinge_edge])
    current_angle = _hinge_angle(positions, model.edge_indices.numpy()[hinge_edge])

    solver = newton.solvers.SolverVBD(model, iterations=1)
    state_in.clear_forces()
    solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
    return {
        "hinge_edge": hinge_edge,
        "initial_rest_angle": initial_rest_angle,
        "initial_yield_angle": initial_yield_angle,
        "current_angle": current_angle,
        "rest_angle": float(state_out.edge_rest_angle.numpy()[hinge_edge]),
        "yield_angle": float(state_out.edge_plastic_yield_angle.numpy()[hinge_edge]),
    }


def test_cloth_plasticity_parameter_validation(test, device):
    """Reject invalid plastic parameters and per-edge array lengths."""
    with test.assertRaisesRegex(ValueError, "yield_angle must contain only finite, nonnegative values"):
        newton.ClothPlasticity(yield_angle=-0.1)
    with test.assertRaisesRegex(ValueError, "hardening_modulus must contain only finite, nonnegative values"):
        newton.ClothPlasticity(yield_angle=0.1, hardening_modulus=np.inf)

    builder = newton.ModelBuilder()
    with test.assertRaisesRegex(ValueError, "ClothPlasticity.yield_angle must be scalar or contain"):
        builder.add_cloth_mesh(
            pos=wp.vec3(0.0, 0.0, 0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0, 0.0, 0.0),
            vertices=[wp.vec3(0.0, 0.0, 0.0), wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0)],
            indices=[0, 1, 2],
            density=1.0,
            bending_plasticity=newton.ClothPlasticity(yield_angle=[0.1, 0.2]),
        )


def test_cloth_plasticity_builder_initializes_model_and_state(test, device):
    """Initialize per-edge plastic parameters in the model and clone evolving values into state."""
    yield_angles = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
    hardening = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    mask = np.array([0, 1, 0, 1, 1], dtype=np.int32)
    builder = newton.ModelBuilder()
    builder.add_cloth_mesh(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=[
            wp.vec3(-0.5, -0.5, 0.0),
            wp.vec3(0.5, -0.5, 0.0),
            wp.vec3(0.5, 0.5, 0.0),
            wp.vec3(-0.5, 0.5, 0.0),
        ],
        indices=[0, 1, 2, 0, 2, 3],
        density=1.0,
        bending_plasticity=newton.ClothPlasticity(
            yield_angle=yield_angles,
            hardening_modulus=hardening,
            mask=mask,
        ),
    )
    model = builder.finalize(device=device)
    state = model.state()

    test.assertEqual(model.edge_count, 5)
    np.testing.assert_array_equal(model.edge_plastic_mask.numpy(), mask)
    np.testing.assert_allclose(model.edge_plastic_yield_angle.numpy(), yield_angles)
    np.testing.assert_allclose(model.edge_plastic_hardening.numpy(), hardening)
    np.testing.assert_allclose(state.edge_rest_angle.numpy(), model.edge_rest_angle.numpy())
    np.testing.assert_allclose(state.edge_plastic_yield_angle.numpy(), yield_angles)

    state_yield_angles = state.edge_plastic_yield_angle.numpy()
    state_yield_angles[0] = 1.5
    state.edge_plastic_yield_angle.assign(wp.array(state_yield_angles, dtype=float, device=device))
    test.assertAlmostEqual(float(model.edge_plastic_yield_angle.numpy()[0]), float(yield_angles[0]), places=6)

    replicated_builder = newton.ModelBuilder()
    replicated_builder.add_world(builder)
    replicated_builder.add_world(builder)
    replicated_model = replicated_builder.finalize(device=device)
    np.testing.assert_array_equal(replicated_model.edge_plastic_mask.numpy(), np.tile(mask, 2))
    np.testing.assert_allclose(replicated_model.edge_plastic_yield_angle.numpy(), np.tile(yield_angles, 2))
    np.testing.assert_allclose(replicated_model.edge_plastic_hardening.numpy(), np.tile(hardening, 2))

    grid_builder = newton.ModelBuilder()
    grid_builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=1.0,
        cell_y=1.0,
        mass=1.0,
        bending_plasticity=newton.ClothPlasticity(yield_angle=0.2, hardening_modulus=0.3),
    )
    np.testing.assert_array_equal(
        grid_builder.edge_plastic_mask, np.ones(len(grid_builder.edge_indices), dtype=np.int32)
    )
    np.testing.assert_allclose(grid_builder.edge_plastic_yield_angle, 0.2)
    np.testing.assert_allclose(grid_builder.edge_plastic_hardening, 0.3)

    plain_builder = newton.ModelBuilder()
    plain_builder.add_cloth_grid(
        pos=wp.vec3(0.0, 0.0, 0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=1,
        dim_y=1,
        cell_x=1.0,
        cell_y=1.0,
        mass=1.0,
    )
    plain_model = plain_builder.finalize(device=device)
    plain_state = plain_model.state()
    test.assertIsNone(plain_model.edge_plastic_mask)
    test.assertIsNone(plain_model.edge_plastic_yield_angle)
    test.assertIsNone(plain_model.edge_plastic_hardening)
    test.assertIsNone(plain_state.edge_rest_angle)
    test.assertIsNone(plain_state.edge_plastic_yield_angle)


def test_cloth_plasticity_flow_and_hardening(test, device):
    """Update the rest angle and yield angle according to the plastic flow rule."""
    hardening = 0.25
    result = _run_hinge_step(device, yield_angle=0.1, hardening=hardening)
    delta = _wrap_angle(result["current_angle"] - result["initial_rest_angle"])
    trial_excess = abs(delta) - result["initial_yield_angle"]
    plastic_increment = trial_excess / (1.0 + hardening)
    test.assertGreater(plastic_increment, 0.0)

    expected_rest_angle = _wrap_angle(result["initial_rest_angle"] + np.sign(delta) * plastic_increment)
    expected_yield_angle = result["initial_yield_angle"] + hardening * plastic_increment
    test.assertAlmostEqual(result["rest_angle"], expected_rest_angle, places=5)
    test.assertAlmostEqual(result["yield_angle"], expected_yield_angle, places=5)


def test_cloth_plasticity_below_yield_is_noop(test, device):
    """Keep plastic state unchanged when the hinge remains below yield."""
    result = _run_hinge_step(device, yield_angle=3.0, hardening=0.5)
    test.assertAlmostEqual(result["rest_angle"], result["initial_rest_angle"], places=6)
    test.assertAlmostEqual(result["yield_angle"], result["initial_yield_angle"], places=6)


def test_cloth_plasticity_mask_disables_flow(test, device):
    """Keep plastic state unchanged when the hinge plastic mask is disabled."""
    result = _run_hinge_step(device, yield_angle=0.1, hardening=0.5, plastic_enabled=False)
    test.assertAlmostEqual(result["rest_angle"], result["initial_rest_angle"], places=6)
    test.assertAlmostEqual(result["yield_angle"], result["initial_yield_angle"], places=6)


def test_cloth_plasticity_accumulates_across_steps(test, device):
    """Accumulate plastic flow across steps while carrying rest angle and hardening state forward."""
    hardening = 0.25
    initial_yield_angle = 0.1
    model, hinge_edge = _build_hinge_model(device, initial_yield_angle, hardening)
    state_in = model.state()
    state_out = model.state()
    solver = newton.solvers.SolverVBD(model, iterations=1)
    edge = model.edge_indices.numpy()[hinge_edge]
    rest_positions = model.particle_q.numpy()

    def run_step(height: float):
        positions = rest_positions.copy()
        positions[3, 2] = height
        state_in.particle_q.assign(wp.array(positions, dtype=wp.vec3, device=device))
        state_in.particle_qd.zero_()
        state_in.clear_forces()
        angle = _hinge_angle(positions, edge)
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        return (
            angle,
            float(state_out.edge_rest_angle.numpy()[hinge_edge]),
            float(state_out.edge_plastic_yield_angle.numpy()[hinge_edge]),
        )

    initial_rest_angle = float(state_in.edge_rest_angle.numpy()[hinge_edge])
    angle1, rest_angle1, yield_angle1 = run_step(0.75)
    state_in, state_out = state_out, state_in
    angle2, rest_angle2, yield_angle2 = run_step(0.75)
    state_in, state_out = state_out, state_in
    angle3, rest_angle3, yield_angle3 = run_step(3.0)

    delta1 = _wrap_angle(angle1 - initial_rest_angle)
    trial_excess1 = abs(delta1) - initial_yield_angle
    plastic_increment1 = trial_excess1 / (1.0 + hardening)
    expected_rest1 = _wrap_angle(initial_rest_angle + np.sign(delta1) * plastic_increment1)
    expected_yield1 = initial_yield_angle + hardening * plastic_increment1
    test.assertGreater(plastic_increment1, 0.0)
    test.assertAlmostEqual(rest_angle1, expected_rest1, places=5)
    test.assertAlmostEqual(yield_angle1, expected_yield1, places=5)

    delta2 = _wrap_angle(angle2 - rest_angle1)
    test.assertLessEqual(abs(delta2), yield_angle1 + 1.0e-6)
    test.assertAlmostEqual(rest_angle2, rest_angle1, places=5)
    test.assertAlmostEqual(yield_angle2, yield_angle1, places=5)

    delta3 = _wrap_angle(angle3 - rest_angle2)
    trial_excess3 = abs(delta3) - yield_angle2
    plastic_increment3 = trial_excess3 / (1.0 + hardening)
    expected_rest3 = _wrap_angle(rest_angle2 + np.sign(delta3) * plastic_increment3)
    expected_yield3 = yield_angle2 + hardening * plastic_increment3
    test.assertGreater(plastic_increment3, 0.0)
    test.assertAlmostEqual(rest_angle3, expected_rest3, places=5)
    test.assertAlmostEqual(yield_angle3, expected_yield3, places=5)


def test_cloth_plasticity_wraps_angle_difference(test, device):
    """Treat dihedral angles on opposite sides of the pi branch as a small elastic difference."""
    yield_angle = 0.1
    authored_rest_angle = 3.1
    model, hinge_edge = _build_hinge_model(device, yield_angle, hardening=0.0, edge_stiffness=100.0)
    positions = model.particle_q.numpy()
    axis_start = positions[0].copy()
    axis = positions[2] - axis_start
    axis /= np.linalg.norm(axis)
    offset = positions[3] - axis_start
    rotation = 3.1
    rotated_offset = (
        offset * np.cos(rotation)
        + np.cross(axis, offset) * np.sin(rotation)
        + axis * np.dot(axis, offset) * (1.0 - np.cos(rotation))
    )
    positions[3] = axis_start + rotated_offset
    current_angle = _hinge_angle(positions, model.edge_indices.numpy()[hinge_edge])
    raw_delta = current_angle - authored_rest_angle
    wrapped_delta = _wrap_angle(raw_delta)
    test.assertGreater(abs(raw_delta), np.pi)
    test.assertLess(abs(wrapped_delta), yield_angle)

    def solve_with_rest_angle(rest_angle):
        state_in = model.state()
        state_out = model.state()
        state_in.particle_q.assign(wp.array(positions, dtype=wp.vec3, device=device))
        rest_angles = state_in.edge_rest_angle.numpy()
        rest_angles[hinge_edge] = rest_angle
        state_in.edge_rest_angle.assign(wp.array(rest_angles, dtype=float, device=device))
        solver = newton.solvers.SolverVBD(model, iterations=1)
        state_in.clear_forces()
        solver.step(state_in, state_out, model.control(), None, 1.0 / 60.0)
        return state_out

    state_wrapped = solve_with_rest_angle(authored_rest_angle)
    state_equivalent = solve_with_rest_angle(authored_rest_angle - 2.0 * np.pi)
    np.testing.assert_allclose(state_wrapped.particle_q.numpy(), state_equivalent.particle_q.numpy(), atol=1.0e-5)
    test.assertAlmostEqual(float(state_wrapped.edge_rest_angle.numpy()[hinge_edge]), authored_rest_angle, places=5)
    test.assertAlmostEqual(float(state_wrapped.edge_plastic_yield_angle.numpy()[hinge_edge]), yield_angle, places=5)


def test_cloth_plasticity_reset_restores_initial_state(test, device):
    """Restore selected plastic rest and yield angles from the model during reset."""
    model, hinge_edge = _build_hinge_model(device, yield_angle=0.1, hardening=0.25)
    state = model.state()
    solver = newton.solvers.SolverVBD(model, iterations=1)

    rest_angles = state.edge_rest_angle.numpy()
    yield_angles = state.edge_plastic_yield_angle.numpy()
    rest_angles[hinge_edge] += 0.5
    yield_angles[hinge_edge] += 0.25
    state.edge_rest_angle.assign(rest_angles)
    state.edge_plastic_yield_angle.assign(yield_angles)

    world_mask_values = np.zeros(model.world_count + 1, dtype=np.bool_)
    world_mask = wp.array(world_mask_values, dtype=wp.bool, device=device)
    solver.reset(state, world_mask=world_mask, flags=0)

    test.assertAlmostEqual(float(state.edge_rest_angle.numpy()[hinge_edge]), float(rest_angles[hinge_edge]), places=6)
    test.assertAlmostEqual(
        float(state.edge_plastic_yield_angle.numpy()[hinge_edge]), float(yield_angles[hinge_edge]), places=6
    )

    world_mask_values[-1] = True
    world_mask.assign(world_mask_values)
    solver.reset(state, world_mask=world_mask, flags=0)

    np.testing.assert_allclose(state.edge_rest_angle.numpy(), model.edge_rest_angle.numpy())
    np.testing.assert_allclose(state.edge_plastic_yield_angle.numpy(), model.edge_plastic_yield_angle.numpy())


devices = get_test_devices(mode="basic")


class TestClothPlasticity(unittest.TestCase):
    pass


add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_parameter_validation",
    test_cloth_plasticity_parameter_validation,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_builder_initializes_model_and_state",
    test_cloth_plasticity_builder_initializes_model_and_state,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_flow_and_hardening",
    test_cloth_plasticity_flow_and_hardening,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_below_yield_is_noop",
    test_cloth_plasticity_below_yield_is_noop,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_mask_disables_flow",
    test_cloth_plasticity_mask_disables_flow,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_accumulates_across_steps",
    test_cloth_plasticity_accumulates_across_steps,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_wraps_angle_difference",
    test_cloth_plasticity_wraps_angle_difference,
    devices=devices,
)
add_function_test(
    TestClothPlasticity,
    "test_cloth_plasticity_reset_restores_initial_state",
    test_cloth_plasticity_reset_restores_initial_state,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
