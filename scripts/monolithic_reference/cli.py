# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect offline inputs; no physics adapter is available in PR-0."""

import argparse
import json
from pathlib import Path

from .manifest import compute_manifest_sha256, load_manifest, require_reference_worktree, validate_manifest


def main() -> None:
    """Validate a manifest or verify an exact clean reference worktree."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--require-frozen", action="store_true")
    inspect = commands.add_parser("inspect-worktree")
    inspect.add_argument("worktree", type=Path)
    inspect.add_argument("--expected-sha", required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            validate_manifest(manifest, require_frozen=args.require_frozen)
            print(json.dumps({"status": manifest.data["status"], "manifest_sha256": compute_manifest_sha256(manifest)}))
        else:
            identity = require_reference_worktree(args.worktree, expected_sha=args.expected_sha)
            print(json.dumps({"worktree": str(identity.path), "sha": identity.sha, "clean": True}))
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
