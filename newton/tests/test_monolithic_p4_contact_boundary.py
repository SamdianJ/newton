# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Keep P3 contact independent of the historical spring and xi contract."""

import subprocess
import sys
import unittest

import warp as wp

import newton
from newton.tests.test_monolithic_p3_coupled import scene, solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


def test_legacy_inputs(test, device):
    """Reject spring parameters and laws instead of translating them to P3."""
    model = scene(device)
    with test.assertRaisesRegex(TypeError, "tangential_stiffness"):
        solver(model, tangential_stiffness=1e6)
    for law in ("spring", "Huber", "EE", "IE", "II"):
        with test.assertRaisesRegex(ValueError, "supported contact law"):
            solver(model, law)


def test_legacy_history(test, device):
    """Allow plain initialization but reject importing a spring-history namespace."""
    model = scene(device)
    run = solver(model)
    source = model.state()
    state = run._create_coupled_state(dt=0.001, source=source)
    test.assertFalse(any("xi" in name for name in run._states.schema))
    test.assertEqual(state._monolithic_coupled.force_valid.numpy()[0], 0)
    source._monolithic = newton.Model.AttributeNamespace("_monolithic")
    source._monolithic.xi_local = wp.zeros(1, dtype=wp.vec3, device=device)
    with test.assertRaisesRegex(ValueError, "cannot import legacy history"):
        run._create_coupled_state(dt=0.001, source=source)
    test.assertEqual(state._monolithic_coupled.force_valid.numpy()[0], 0)


class TestContactBoundary(unittest.TestCase):
    def test_component_imports(self):
        """Resolve contact, history and collision without loading old contact code."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from newton._src.solvers.monolithic.contact_variational import ContactObjective; "
                "from newton._src.solvers.monolithic.coupled_state import CoupledState; "
                "from newton.solvers.experimental.monolithic import MonolithicCollisionPipeline; "
                "assert not any('newton._src.solvers.monolithic.' + name in sys.modules "
                "for name in ('contact', 'articulation', 'solver_monolithic', 'articulation_solver', 'coupled_solver'))",
            ],
            check=True,
        )


add_function_test(TestContactBoundary, "test_legacy_inputs", test_legacy_inputs, devices=get_test_devices())
add_function_test(TestContactBoundary, "test_legacy_history", test_legacy_history, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
