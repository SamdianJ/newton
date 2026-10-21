# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check native articulation components without introducing a q-only solver."""

import unittest
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.articulation import ArticulationObjective, _TreeKinematics
from newton._src.solvers.monolithic.articulation_state import ArticulationState, _predict, _recover_step
from newton._src.solvers.monolithic.joint_parameters import JointParameters
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference.articulation_fixtures import build_case, specification, target
from scripts.monolithic_reference.articulation_oracle import Kinematics, joint_terms, skew


def component(case, device, dtype=None):
    """Freeze the same component inputs for both native dtypes and baseline capture."""
    model, terms = build_case(case, device)
    k = _TreeKinematics(model, **({} if dtype is None else {"dtype": dtype}))
    owner = ArticulationState(k)
    state = owner.create(dt=0.001)
    history = getattr(state, owner.namespace)
    obj = ArticulationObjective(k, JointParameters(model, **asdict(terms)).params.numpy())
    obj.begin(state.joint_q)
    wp.copy(obj.vn, state.joint_qd)
    dt_type = wp.float64 if dtype is None else dtype
    wp.launch(
        _predict,
        k.bodies,
        [
            obj.tn,
            obj.rn,
            history.link_v_com,
            history.link_omega,
            history.link_vsym,
            dt_type(0.001),
            obj.predicted_t,
            obj.predicted_r,
            obj.step_t,
            obj.step_r,
        ],
        device=device,
    )
    qt, vt = target(case, 0.001)
    obj.qt.assign(qt.astype(np.float32))
    obj.vt.assign(vt.astype(np.float32))
    obj.force.assign(model.body_mass.numpy()[:, None] * model.gravity.numpy()[-1])
    q = wp.array(state.joint_q.numpy() + 0.02, dtype=state.joint_q.dtype, device=device)
    obj.evaluate(q, 0.001)
    return k, owner, state, obj, q


def test_components(test, device):
    """Compare seven fixtures in both precisions against independent transform and force math."""
    for case in specification()["cases"]:
        for dtype in (wp.float64, wp.float32):
            with test.subTest(case=case["id"], dtype=dtype):
                k, owner, state, obj, q = component(case, device, dtype)
                host = np.float64 if dtype == wp.float64 else np.float32
                test.assertEqual(q.dtype, dtype)
                for a in (k.com, k.rotations, k.jt, k.jw, obj.predicted_r, obj.residual, obj.tangent):
                    test.assertEqual(a.numpy().dtype, host)
                history = getattr(state, owner.namespace)
                for name in (
                    "joint_v",
                    "joint_omega",
                    "joint_vsym",
                    "link_v_com",
                    "link_omega",
                    "link_vsym",
                    "last_dt",
                ):
                    test.assertEqual(getattr(history, name).numpy().dtype, host)
                oracle = Kinematics(k.model)
                t, r, jt, jr, _, _ = oracle.evaluate(q.numpy().astype(float))
                tol = 2e-12 if dtype == wp.float64 else 2e-6
                for actual, expected in ((k.com.numpy(), t), (k.rotations.numpy(), r), (k.jt.numpy(), jt)):
                    np.testing.assert_allclose(actual, expected, atol=tol, rtol=tol)
                angular = jr @ r.transpose(0, 2, 1)[:, None]
                expected_jw = np.stack((angular[:, :, 2, 1], angular[:, :, 0, 2], angular[:, :, 1, 0]), axis=2)
                np.testing.assert_allclose(k.jw.numpy(), expected_jw, atol=tol, rtol=tol)
                h = float(host(0.001))
                je, jres, jb = joint_terms(
                    *[a.numpy().astype(float) for a in (q, obj.qn, obj.vn)],
                    h,
                    *[
                        a.numpy().astype(float)
                        for a in (obj.params, obj.armature, obj.damping, obj.qt, obj.vt, obj.joint_force)
                    ],
                )
                le, lr, _ = oracle.inertia(
                    q.numpy().astype(float),
                    obj.predicted_t.numpy().astype(float),
                    obj.predicted_r.numpy().astype(float),
                    h,
                    obj.force.numpy().astype(float),
                )
                le += np.sum(obj.force.numpy().astype(float) * obj.tn.numpy().astype(float))
                atol, rtol = (2e-8, 1e-9) if dtype == wp.float64 else (0.02, 3e-4)
                for actual, expected in (
                    (obj.joint_energy.numpy(), je),
                    (obj.joint_r.numpy(), jres),
                    (obj.joint_b.numpy(), jb),
                    (obj.residual.numpy(), lr + jres),
                    (obj.link_energy.numpy().sum(), le),
                ):
                    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
                np.testing.assert_allclose(obj.tangent.numpy(), obj.tangent.numpy().T, atol=atol, rtol=rtol)
                test.assertGreater(np.linalg.eigvalsh(obj.tangent.numpy()).min(), 0)
                # Check host positivity arithmetic is selected too, not silently promoted.
                eigen = np.linalg.eigvalsh
                with patch("numpy.linalg.eigvalsh", wraps=eigen) as checked:
                    k.validate_inertia()
                    test.assertEqual(checked.call_args.args[0].dtype, host)
                wp.launch(
                    _recover_step,
                    k.bodies,
                    [
                        k.com_steps,
                        k.rotation_steps,
                        dtype(h),
                        history.link_v_com,
                        history.link_omega,
                        history.link_vsym,
                    ],
                    device=device,
                )
                a = (
                    (k.rotations.numpy().astype(float) - obj.rn.numpy().astype(float))
                    @ k.rotations.numpy().astype(float).transpose(0, 2, 1)
                    / h
                )
                expected_sym = np.column_stack(
                    (
                        a[:, 0, 0],
                        a[:, 1, 1],
                        a[:, 2, 2],
                        (a[:, 0, 1] + a[:, 1, 0]) / 2,
                        (a[:, 0, 2] + a[:, 2, 0]) / 2,
                        (a[:, 1, 2] + a[:, 2, 1]) / 2,
                    )
                )
                np.testing.assert_allclose(history.link_vsym.numpy(), expected_sym, atol=tol * 1000, rtol=tol)
                copied = owner.copy(state)
                for name in owner.schema:
                    np.testing.assert_array_equal(
                        getattr(getattr(copied, owner.namespace), name).numpy(), getattr(history, name).numpy()
                    )
                owner.reset(state, dt=0.001)
                w = history.link_omega.numpy().astype(float)
                expected = np.array([h * skew(v) @ skew(v) / (1 + np.sqrt(1 - h * h * np.dot(v, v))) for v in w])
                np.testing.assert_allclose(
                    history.link_vsym.numpy()[:, :3], np.diagonal(expected, axis1=1, axis2=2), atol=tol, rtol=tol
                )


def test_rejection(test, device):
    """Reject mixed precision, unsupported dtypes and unrepresentable dt before state writes."""
    case = specification()["cases"][0]
    k, owner, state, obj, _ = component(case, device, wp.float32)
    with test.assertRaises(ValueError):
        k.evaluate(wp.array(state.joint_q.numpy(), dtype=wp.float64, device=device))
    other = ArticulationState(_TreeKinematics(k.model))
    with test.assertRaises(ValueError):
        other.validate(state)
    with test.assertRaises(ValueError):
        _TreeKinematics(k.model, dtype=wp.float16)
    before = {n: getattr(getattr(state, owner.namespace), n).numpy() for n in owner.schema}
    for dt in (1e-50, 1e40, -1.0, float("nan")):
        with test.assertRaises(ValueError):
            owner.reset(state, dt=dt)
        for n, value in before.items():
            np.testing.assert_array_equal(getattr(getattr(state, owner.namespace), n).numpy(), value)
        saved = obj.residual.numpy()
        with test.assertRaises(ValueError):
            obj.evaluate(state.joint_q, dt)
        np.testing.assert_array_equal(obj.residual.numpy(), saved)


def test_graphs(test, device):
    """Keep both precision graphs alive and replay them against eager evaluation."""
    setups = [component(specification()["cases"][-1], device, dtype) for dtype in (wp.float64, wp.float32)]

    def launch(k, owner, state, obj, q):
        history = getattr(state, owner.namespace)
        obj.evaluate(q, 0.001)
        wp.launch(
            _recover_step,
            k.bodies,
            [
                k.com_steps,
                k.rotation_steps,
                k.dtype(0.001),
                history.link_v_com,
                history.link_omega,
                history.link_vsym,
            ],
            device=device,
        )
        return obj.residual, obj.tangent, history.link_v_com, history.link_omega, history.link_vsym

    for setup in setups:
        launch(*setup)
    graphs = []
    for setup in setups:
        with wp.ScopedCapture(device=device) as capture:
            launch(*setup)
        graphs.append(capture.graph)
    for increment in (0.01, -0.03):
        for graph, setup in zip(graphs, setups, strict=True):
            _k, owner, state, obj, q = setup
            history = getattr(state, owner.namespace)
            q.assign(q.numpy() + increment)
            wp.capture_launch(graph)
            values = [
                a.numpy()
                for a in (obj.residual, obj.tangent, history.link_v_com, history.link_omega, history.link_vsym)
            ]
            for value, actual in zip(values, launch(*setup), strict=True):
                np.testing.assert_array_equal(actual.numpy(), value)


class TestArticulationPrecision(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestArticulationPrecision, "test_components", test_components, devices=[_device])
    add_function_test(TestArticulationPrecision, "test_rejection", test_rejection, devices=[_device])
    if _device.is_cuda:
        add_function_test(TestArticulationPrecision, "test_graphs", test_graphs, devices=[_device])


if __name__ == "__main__":
    unittest.main()
