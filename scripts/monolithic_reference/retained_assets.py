# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Resolve retained files without rewriting frozen manifests or physical identities."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "asset/monolithic"
INPUTS = ASSET_ROOT / "data/inputs"
FROZEN = INPUTS / "frozen-inputs-rebuilt.json"
ORIGINAL_FROZEN = INPUTS / "frozen-inputs-original.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_input(path):
    """Translate recorded absolute locations using the explicit relocation map."""
    path = Path(path)
    moves = json.loads((ASSET_ROOT / "relocations.json").read_text())
    for move in sorted(moves, key=lambda row: len(row["old_path"]), reverse=True):
        old = Path(move["old_path"])
        if path.is_relative_to(old):
            return ASSET_ROOT / move["asset_path"] / path.relative_to(old)
    # Repository fixtures move with the checkout, independently of the workspace location.
    marker = "/worktrees/Monolithic-refactor/"
    if marker in path.as_posix():
        return ROOT / path.as_posix().split(marker, 1)[1]
    return path


def validate_inputs(input_manifest=FROZEN):
    """Check original content hashes after relocation; return the unchanged manifest hash."""
    for name, digest in json.loads(input_manifest.read_text())["input_hashes"].items():
        path = resolve_input(name)
        if sha(path) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    return sha(input_manifest)


def retained_hashes():
    """Read the tracked content inventory rather than trusting hashes computed on demand."""
    return dict(line.split("  ", 1)[::-1] for line in (ASSET_ROOT / "files.sha256").read_text().splitlines())


def verify_retained(path, hashes):
    """Reject missing, unlisted or modified retained input/reference files."""
    relative = str(path.relative_to(ASSET_ROOT))
    if relative not in hashes or sha(path) != hashes[relative]:
        raise ValueError(f"Retained file identity changed: {path}")
