# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared production robot case list and source-identity helpers for the runners."""

import hashlib
import sys
from pathlib import Path

LAWS = ("C-IE",)
GLOBALIZATIONS = ("ArmijoOrResNorm",)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases():
    """List the production robot cases without importing Newton or initializing CUDA."""
    return [
        {"id": f"franka-cube--{g}--{law}--production--graph1", "law": law, "globalization": g, "steps": 5250}
        for g in GLOBALIZATIONS
        for law in LAWS
    ] + [
        {
            "id": "sharpa-recorded-10mm--ArmijoOrResNorm--C-IE--production--graph1",
            "law": "C-IE",
            "globalization": "ArmijoOrResNorm",
            "steps": 8667,
            "reuse_only": True,
        }
    ]


def module_origins(root, source_hashes):
    """Check real imported paths instead of trusting the process working directory."""
    result = {}
    for name, module in tuple(sys.modules.items()):
        if not (name == "newton" or name.startswith(("newton.", "scripts.monolithic_reference."))):
            continue
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = Path(path).resolve()
        if path.suffix != ".py" or not path.is_relative_to(root):
            raise ValueError(f"Unexpected module origin: {name}: {path}")
        relative = path.relative_to(root).as_posix()
        digest = sha(path)
        if source_hashes.get(relative) != digest:
            raise ValueError(f"Imported source is not frozen: {name}")
        result[name] = {"path": str(path), "sha256": digest}
    return result
