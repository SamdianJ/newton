# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise shared joint inputs without the historical joint evaluation workspace."""

import subprocess
import sys
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.monolithic.joint_parameters import JointParameters, _validate_joint_coordinate_layout
from newton.tests.monolithic_test_utils import build_tiny_cpu_fixture
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_snapshot_and_rejection(test, device):
    """Freeze targets and reject duplicate forcing, nonfinite controls and stale storage."""
    fixture = build_tiny_cpu_fixture(device=device)
    model, control = fixture.model, fixture.control
    model.joint_target_ke.fill_(4)
    model.joint_target_kd.fill_(0.3)
    model.joint_effort_limit.fill_(2)
    model.joint_velocity_limit.fill_(5)
    model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION_VELOCITY))
    terms = JointParameters(model, implicit_pd=True)
    test.assertEqual(terms.allowed_properties, frozenset(("joint_target_ke", "joint_target_kd")))
    control.joint_target_q.fill_(0.125)
    control.joint_target_qd.fill_(0.25)
    terms.snapshot_targets(control)
    control.joint_target_q.fill_(0.5)
    np.testing.assert_array_equal(terms.target_q.numpy(), np.full(2, 0.125, dtype=np.float32))
    for name, value, message in (
        ("joint_f", 1.0, "Duplicate"),
        ("joint_target_q", float("nan"), "Nonfinite"),
    ):
        array = getattr(control, name)
        old = array.numpy().copy()
        array.fill_(value)
        with test.assertRaisesRegex(ValueError, message):
            terms.snapshot_targets(control)
        np.testing.assert_array_equal(terms.target_q.numpy(), np.full(2, 0.125, dtype=np.float32))
        array.assign(old)
    model.joint_target_ke = wp.clone(model.joint_target_ke)
    with test.assertRaisesRegex(ValueError, "Stale"):
        terms.snapshot_targets(control)


def test_configuration_and_layout(test, device):
    """Reject invalid switches, parameter ranges and scalar coordinate layouts."""
    model = build_tiny_cpu_fixture(device=device).model
    np.testing.assert_array_equal(_validate_joint_coordinate_layout(model), [0, 1])
    for options in (
        {"implicit_pd": 1},
        {"limits": True},
        {"limits": True, "limit_width": (0.1, 0)},
        {"friction": True, "friction_velocity_scale": (0.1,)},
        {"friction_velocity_scale": (0.1, 0.1)},
    ):
        with test.assertRaises(ValueError):
            JointParameters(model, **options)
    model.joint_friction.fill_(-1)
    with test.assertRaisesRegex(ValueError, "nonnegative"):
        JointParameters(model, friction=True, friction_velocity_scale=(0.1, 0.1))
    starts = model.joint_q_start.numpy().copy()
    model.joint_q_start.fill_(0)
    with test.assertRaisesRegex(ValueError, "prefix layout"):
        JointParameters(model)
    model.joint_q_start.assign(starts)
    model.joint_dof_dim.zero_()
    with test.assertRaisesRegex(ValueError, "joint_dof_dim"):
        _validate_joint_coordinate_layout(model)


class TestJointParameters(unittest.TestCase):
    def test_component_import_boundary(self):
        """Load shared parameters and FP64 kinematics without the historical articulation."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton._src.solvers.monolithic.joint_parameters import JointParameters; "
                "assert 'newton._src.solvers.monolithic.articulation' not in sys.modules; "
                "from newton._src.solvers.monolithic.articulation import ArticulationKinematics; "
                "assert 'newton._src.solvers.monolithic.solver_monolithic' not in sys.modules",
            ],
            check=True,
        )


for _device in get_test_devices():
    for _test in (test_snapshot_and_rejection, test_configuration_and_layout):
        add_function_test(TestJointParameters, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
