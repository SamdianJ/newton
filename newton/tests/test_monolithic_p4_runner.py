# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Guard exact profile expansion and qualification labels."""

import json
import subprocess
import sys
import unittest

from scripts.monolithic_reference.run_p4 import aggregate, select_cases


class TestP4Runner(unittest.TestCase):
    def test_profiles(self):
        """Expand ten full and two dev cases without granting filtered qualification."""
        full, dev = select_cases("full"), select_cases("dev")
        self.assertEqual(len(full), 10)
        self.assertEqual(dev, [full[0], full[-1]])
        self.assertEqual(select_cases("full", [full[2]["id"]]), [full[2]])
        self.assertEqual(aggregate([{"status": "PASS"}], True), "PARTIAL")
        self.assertEqual(aggregate([{"status": "PASS"}], False), "PASS")
        self.assertEqual(aggregate([{"status": "FAIL"}], True), "FAIL")
        with self.assertRaises(ValueError):
            select_cases("dev", [full[2]["id"]])

    def test_list(self):
        """List the frozen matrix without initializing a GPU or loading scenes."""
        result = subprocess.run(
            [sys.executable, "-m", "scripts.monolithic_reference.run_p4", "--profile", "full", "--list"],
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertEqual(json.loads(result.stdout), select_cases("full"))

    def test_scene_imports(self):
        """Load supported scenes without importing historical diagnostic drivers."""
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import newton.examples.softbody.monolithic_franka; "
                "import newton.examples.softbody.monolithic_sharpa_recorded; import sys; "
                "assert not any(n.startswith('scripts.monolithic_reference.') for n in sys.modules)",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
