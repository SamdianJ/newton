# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the P1 input boundary without enabling pending physics."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from newton._src.solvers.monolithic import solver_monolithic as impl
from newton._src.solvers.monolithic.contact import _MonolithicHistoryIdentity
from newton._src.solvers.monolithic.linear import (
    MonolithicContactFactorKind,
    MonolithicLinearCapacities,
    MonolithicLinearLayout,
    MonolithicLinearStatus,
)
from newton.solvers.experimental.monolithic import SolverMonolithic
from newton.tests.test_solver_monolithic_contact import _contact_scene, _current
from newton.tests.test_solver_monolithic_sharpa import make_solver
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestP1Contract(unittest.TestCase):
    def test_pending_modes(self):
        """Reject pending physics and malformed values before model allocation."""
        cases = (
            {"material_model": "smith_log_stabilized"},
            {"material_model": "unknown"},
            {"mass_mode": "consistent"},
            {"mass_mode": "unknown"},
            {"tet_rest_density": object()},
            {"normal_smoothing_width": 0.001},
            {"normal_smoothing_width": -1},
            {"friction_coefficient": 0.5, "tangential_stiffness": 1e6},
            {"friction_coefficient": float("nan")},
            {"friction_coefficient": True},
            {"tangential_stiffness": -1},
        )
        for options in cases:
            with self.subTest(options=options), patch.object(impl, "_build_layout") as allocate:
                with self.assertRaises(ValueError):
                    SolverMonolithic(None, collision_pipeline=None, contact_stiffness=1, **options)
                with self.assertRaises(ValueError):
                    SolverMonolithic._create_joint_diagnostic(
                        None, joint_terms=SolverMonolithic.JointTerms(), **options
                    )
                allocate.assert_not_called()

    def test_capacity_overflow(self):
        """Reject noninteger and int32-overflow capacities before device allocation."""
        for count in (True, -1, 1.5, 2**31):
            with self.subTest(count=count), self.assertRaises(ValueError):
                MonolithicLinearCapacities(count, 0, 0, 1, 2)

    def test_p1_capacity_bound(self):
        """Reserve three scalar factors per P1Q3 slot and reject overflow arithmetically."""
        layout = MonolithicLinearLayout(22, 100)
        normal = MonolithicLinearCapacities.for_p1(
            layout, tet_count=200, static_pair_count=10, friction=False, pcg_max_iterations=200
        )
        friction = MonolithicLinearCapacities.for_p1(
            layout, tet_count=200, static_pair_count=10, friction=True, pcg_max_iterations=200
        )
        self.assertEqual((normal.contact_factor_count, friction.contact_factor_count), (30, 90))
        self.assertEqual(friction.internal_mat33_triplet_count, 3300)
        self.assertEqual(friction.global_scalar_triplet_count, 116674)
        with self.assertRaises(ValueError):
            MonolithicLinearCapacities.for_p1(
                layout, tet_count=200, static_pair_count=np.int64(2**62), friction=True, pcg_max_iterations=200
            )

    def test_history_identity(self):
        """Bind reserved history identity to both hashes and reject invalid epochs."""
        identity = _MonolithicHistoryIdentity("a" * 64, "b" * 64, 0, 0)
        for field, value in (("topology_sha256", "c" * 64), ("config_sha256", "d" * 64), ("history_epoch", 1)):
            self.assertNotEqual(identity, replace(identity, **{field: value}))
        for field, value in (("config_sha256", "unknown"), ("history_epoch", -1), ("config_generation", True)):
            with self.assertRaises(ValueError):
                replace(identity, **{field: value})


def test_generation_identity(test, device):
    """Reject refactoring an assembly under a different config or history epoch."""
    model, solver = make_solver(device)
    state, control = model.state(), model.control()
    control.joint_target_q.fill_(0.01)
    solver.step(state, state, control, None, 0.001)
    generation = solver._generation
    test.assertEqual(generation.history_epoch, 0)
    for field in ("config_generation", "history_epoch"):
        changed = replace(generation, **{field: getattr(generation, field) + 1})
        test.assertEqual(
            solver._linear.factor_actor_preconditioner(generation=changed, pivot_tolerance=0),
            MonolithicLinearStatus.STALE_GENERATION,
        )


def test_factor_keys(test, device):
    """Keep factor keys stable when atomically appended contact records are reordered."""
    scene = _contact_scene(device)
    contacts = scene[-1]
    _, assembly, residual = _current(scene)
    count = int(contacts.soft_contact_count.numpy()[0])
    factors = assembly.contact_factors
    active = int(factors.count.numpy()[0])
    test.assertGreater(active, 0)
    keys = np.sort(factors.candidate_tid.numpy()[:active])
    np.testing.assert_array_equal(factors.kind.numpy()[:active], int(MonolithicContactFactorKind.NORMAL))
    original = residual.numpy().copy()
    for name in ("indices", "barycentric", "shape", "body_pos", "body_vel", "normal", "particle"):
        array = getattr(contacts, "soft_contact_" + name)
        values = array.numpy()
        values[:count] = values[:count][::-1]
        array.assign(values)
    tids = contacts.soft_contact_tids.numpy()
    tids[tids >= 0] = count - 1 - tids[tids >= 0]
    contacts.soft_contact_tids.assign(tids)
    _, assembly, residual = _current(scene, sequence=1)
    factors = assembly.contact_factors
    test.assertEqual(int(factors.status.numpy()[0]), 0)
    test.assertEqual(int(factors.count.numpy()[0]), active)
    np.testing.assert_array_equal(np.sort(factors.candidate_tid.numpy()[:active]), keys)
    np.testing.assert_allclose(residual.numpy(), original, rtol=1e-6, atol=1e-7)


for device in get_test_devices():
    add_function_test(TestP1Contract, "test_generation_identity", test_generation_identity, devices=[device])
    add_function_test(TestP1Contract, "test_factor_keys", test_factor_keys, devices=[device])


if __name__ == "__main__":
    unittest.main()
