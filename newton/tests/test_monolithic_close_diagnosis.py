# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The diagnostic viewer must not advance into deferred grasp stages."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.monolithic_reference.view_contact_continuity import Example


class TestCloseDiagnosis(unittest.TestCase):
    def example(self, step_count):
        example = Example.__new__(Example)
        example.case = SimpleNamespace(step_count=step_count, failure=None, fixture={"dt": 0.001}, save=Mock())
        example.close_steps = 2500
        example.stopped = False
        example.output = "unused"
        example.sim_time = step_count * 0.001

        def advance():
            example.case.step_count += 1

        example.case.step = Mock(side_effect=advance)
        return example

    def test_stop_exactly_at_close_boundary(self):
        example = self.example(2497)
        example.step()
        self.assertEqual(example.case.step_count, 2500)
        self.assertEqual(example.case.step.call_count, 3)
        self.assertEqual(example.sim_time, 2.5)
        example.case.save.assert_called_once_with(example.output)
        example.step()
        self.assertEqual(example.case.step.call_count, 3)

    def test_failure_retains_rolled_back_time(self):
        example = self.example(1435)

        def failure():
            example.case.failure = "REGULARIZATION_EXHAUSTED"

        example.case.step.side_effect = failure
        example.step()
        self.assertTrue(example.stopped)
        self.assertEqual(example.case.step_count, 1435)
        self.assertEqual(example.sim_time, 1.435)
        example.step()
        example.case.step.assert_called_once()
        example.case.save.assert_called_once()


if __name__ == "__main__":
    unittest.main()
