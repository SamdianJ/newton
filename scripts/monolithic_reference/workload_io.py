# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Versioned numerical bundles; never deserialize Python objects or handles."""

import hashlib
import json
from pathlib import Path

import numpy as np

from newton.examples.softbody.monolithic_recording import serial


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(serial(value), indent=2, allow_nan=False) + "\n")


def save_arrays(path, arrays):
    """Write numeric data and return exact schema descriptors."""
    if any(np.asarray(a).dtype.hasobject for a in arrays.values()):
        raise ValueError("Object arrays are not supported")
    np.savez_compressed(path, **arrays)
    return {name: {"shape": list(a.shape), "dtype": a.dtype.str} for name, a in arrays.items()}


def load_arrays(path, schema):
    """Reject missing/extra fields and dtype/shape changes before device allocation."""
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(schema):
            raise ValueError("Array schema keys differ")
        values = {name: archive[name] for name in schema}
    for name, value in values.items():
        if (
            value.dtype.hasobject
            or list(value.shape) != schema[name]["shape"]
            or value.dtype.str != schema[name]["dtype"]
        ):
            raise ValueError(f"Array schema mismatch: {name}")
    return values


def complete_bundle(root, manifest):
    """Publish COMPLETE only after all referenced payloads have been written."""
    manifest["files"] = {
        str(p.relative_to(root)): digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest["schema"] = "monolithic-workload/1"
    manifest["status"] = "COMPLETE"
    temporary = root / "manifest.tmp"
    write_json(temporary, manifest)
    temporary.replace(root / "manifest.json")


def open_bundle(root):
    """Authenticate every file and forbid paths outside the bundle."""
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "monolithic-workload/1" or manifest.get("status") != "COMPLETE":
        raise ValueError("Incomplete or unsupported workload")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path) != expected:
            raise ValueError(f"Missing, escaped or modified payload: {name}")
    return manifest
