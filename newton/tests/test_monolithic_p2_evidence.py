# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check reproducible P2 diagnostic evidence indexing."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.monolithic_reference.analyze_p2_0 import evidence_index


class TestP2EvidenceIndex(unittest.TestCase):
    def test_index_excludes_its_outputs(self):
        """Keep the index stable when its own digest is rewritten."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "compact.json").write_text("{}\n")
            expected = evidence_index(root)
            (root / "evidence-index.json").write_text("{}\n")
            digest = root / "evidence-index.sha256"
            digest.write_text("old\n")
            self.assertEqual(evidence_index(root), expected)
            digest.write_text("new\n")
            self.assertEqual(evidence_index(root), expected)

    def test_index_covers_raw_evidence(self):
        """Bind raw traces, snapshots, analysis scripts and execution logs."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ("tet/trace.jsonl", "common_snapshots.npz", "compare_temporal.py", "logs/run.log")
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            indexed = {Path(row["path"]).relative_to(root).as_posix(): row for row in evidence_index(root)}
            self.assertEqual(set(indexed), set(names))
            for name in names:
                self.assertEqual(indexed[name]["sha256"], hashlib.sha256(name.encode()).hexdigest())
                self.assertEqual(indexed[name]["bytes"], len(name.encode()))


if __name__ == "__main__":
    unittest.main()
