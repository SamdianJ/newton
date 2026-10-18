# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record pooled actor routes without weakening their input or physical guards."""

import copy
import gzip
import json
import tempfile
import unittest
from functools import partial
from pathlib import Path
from unittest.mock import patch

from newton._src.solvers.monolithic.contact_pipeline import _SpatialContactObjective
from newton.tests.test_monolithic_actor_coupling import test_active_recording
from newton.tests.unittest_utils import add_function_test, get_test_devices
from scripts.monolithic_reference import run_actor_coupling as coupling
from scripts.monolithic_reference.check_recording import check_actor, compare_actor
from scripts.monolithic_reference.workload_scene import build_scene


def test_actor_dataflow_recording(test, device):
    """Audit a bounded recording, checkpoint continuation and creation order for each route."""
    graphs = (False, True) if device.is_cuda else (False,)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        references = {}
        for algorithm in ("pooled", "spatial"):
            for graph in graphs:
                options = {"device": device, "graph": graph, "contact_dataflow": algorithm}
                if algorithm == "spatial":
                    options["spatial"] = {"candidate_capacity": 4096}
                _, run, state, _ = build_scene("actor-coupling", **options)
                folder = root / f"{algorithm}-{graph}"
                summary = coupling.record_run(run, state, folder, end_step=3, checkpoint_step=2)
                test.assertEqual(summary["status"], "PASS_PREFIX", summary)
                test.assertEqual(check_actor(folder, require_complete=False)["frames"], 4)
                metadata = json.loads((folder / "configuration.json").read_text())
                test.assertEqual(metadata["schema"], "actor-coupling-recording/v2")
                test.assertEqual(metadata["algorithm"]["contact_dataflow"], algorithm)
                if algorithm == "pooled":
                    references[graph] = folder
                else:
                    with test.assertRaisesRegex(ValueError, "explicit dataflow comparison"):
                        compare_actor(references[graph], folder, creation_order=True, require_complete=False)

                _, restored, target, _ = build_scene("actor-coupling", **options)
                coupling.restore_midpoint(restored, target, folder / "midpoint.npz", 2)
                resumed = folder.with_name(folder.name + "-restored")
                summary = coupling.record_run(restored, target, resumed, start_step=2, end_step=3)
                test.assertEqual(summary["status"], "PASS_PREFIX", summary)
                test.assertEqual(
                    compare_actor(folder, resumed, require_complete=False)["status"], "PASS_FROZEN_ENVELOPE"
                )
                _, reordered, other, _ = build_scene("actor-coupling-before", **options)
                reordered_folder = folder.with_name(folder.name + "-reordered")
                summary = coupling.record_run(reordered, other, reordered_folder, before=True, end_step=3)
                test.assertEqual(summary["status"], "PASS_PREFIX", summary)
                test.assertEqual(
                    compare_actor(folder, reordered_folder, creation_order=True, require_complete=False)["status"],
                    "PASS_FROZEN_ENVELOPE",
                )
                for change in (
                    lambda m: m["algorithm"].update(contact_factor_capacity=1),
                    lambda m: m["algorithm"].update(contact_dataflow="legacy"),
                    lambda m: m["sampling"]["fields"]["actor_contact.weights"].__setitem__(0, 0.0),
                ):
                    bad = copy.deepcopy(metadata)
                    change(bad)
                    (folder / "configuration.json").write_text(json.dumps(bad))
                    with test.assertRaises(ValueError):
                        check_actor(folder, require_complete=False)
                (folder / "configuration.json").write_text(json.dumps(metadata))

                with gzip.open(folder / "steps.jsonl.gz", "rt") as stream:
                    row = json.loads(stream.readline())
                corruptions = [
                    ("contact_pool", "tainted", True),
                    ("contact_pool", "dropped", 1),
                ]
                if algorithm == "spatial":
                    corruptions.append(("contact_spatial", "capacity_dropped", 1))
                for name, field, value in corruptions:
                    bad = copy.deepcopy(row)
                    bad["stats"]["variational"][name][field] = value
                    with test.assertRaisesRegex(ValueError, "capacity"):
                        coupling.validate_row(bad, coupling.load_case())
                if algorithm == "spatial":
                    _, overflow, empty, _ = build_scene(
                        "actor-coupling", **{**options, "spatial": {"candidate_capacity": 0}}
                    )
                    summary = coupling.record_run(
                        overflow, empty, folder.with_name(folder.name + "-overflow"), end_step=1
                    )
                    test.assertEqual(summary["status"], "FAIL", summary)
                    test.assertIn("capacity truncation", summary["error"])


def test_spatial_actor_active_recording(test, device):
    """Reuse the existing active force-chain and layout/continuation assertions on B3."""
    with patch(
        "newton._src.solvers.monolithic.solver_monolithic.ContactObjective",
        partial(_SpatialContactObjective, candidate_capacity=4096),
    ):
        test_active_recording(test, device)


class TestActorDataflow(unittest.TestCase):
    pass


for device in get_test_devices():
    add_function_test(
        TestActorDataflow, "test_actor_dataflow_recording", test_actor_dataflow_recording, devices=[device]
    )
    add_function_test(
        TestActorDataflow, "test_spatial_actor_active_recording", test_spatial_actor_active_recording, devices=[device]
    )


if __name__ == "__main__":
    unittest.main()
