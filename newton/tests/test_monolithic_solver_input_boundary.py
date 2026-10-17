# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check supported contact arguments, state initialization and lazy component imports."""

import subprocess
import sys
import unittest

import newton
from newton.tests.test_monolithic_solver_coupled import scene, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_unsupported_contact_inputs(test, device):
    """Reject spring parameters and laws outside the supported contact-law API."""
    model = scene(device)
    with test.assertRaisesRegex(TypeError, "tangential_stiffness"):
        solver(model, tangential_stiffness=1e6)
    for law in ("spring", "Huber", "EE", "IE", "II"):
        with test.assertRaisesRegex(ValueError, "supported contact law"):
            solver(model, law)


def test_initialization_history_boundary(test, device):
    """Allow plain initialization but reject history owned by another solver."""
    model = scene(device)
    run = solver(model)
    source = model.state()
    state = run.create_state(dt=0.001, source=source)
    test.assertEqual(state._monolithic_coupled.force_valid.numpy()[0], 0)
    source.foreign_solver = newton.Model.AttributeNamespace("foreign_solver")
    with test.assertRaisesRegex(ValueError, "cannot import legacy history"):
        run.create_state(dt=0.001, source=source)
    test.assertEqual(state._monolithic_coupled.force_valid.numpy()[0], 0)


class TestContactBoundary(unittest.TestCase):
    def test_component_imports(self):
        """Resolve contact and history without loading the solver implementation."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton._src.solvers.monolithic.contact import ContactObjective; "
                "from newton._src.solvers.monolithic.coupled_state import CoupledState; "
                "assert not any('newton._src.solvers.monolithic.' + name in sys.modules "
                "for name in ('solver_monolithic',))",
            ],
            check=True,
        )


add_function_test(
    TestContactBoundary, "test_unsupported_contact_inputs", test_unsupported_contact_inputs, devices=get_test_devices()
)
add_function_test(
    TestContactBoundary,
    "test_initialization_history_boundary",
    test_initialization_history_boundary,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
