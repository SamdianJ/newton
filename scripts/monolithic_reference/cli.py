# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run isolated DRAFT reference adapters and inspect their explicit mapping limits."""

import argparse
import json
from pathlib import Path

from .manifest import (
    MappingEntry,
    compare_runs,
    compute_manifest_sha256,
    load_manifest,
    require_reference_worktree,
    run_adapter,
    validate_manifest,
)


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
    run = commands.add_parser("run")
    run.add_argument("implementation", choices=("newton", "superdex"))
    run.add_argument("manifest", type=Path)
    run.add_argument("output_dir", type=Path)
    run.add_argument("--worktree", type=Path, required=True)
    run.add_argument("--python", type=Path, required=True, dest="python_executable")
    run.add_argument("--device", default="cpu")
    run.add_argument("--build-type", required=True)
    run.add_argument("--timeout-s", type=int, default=300)
    compare = commands.add_parser("compare")
    compare.add_argument("manifest", type=Path)
    compare.add_argument("newton_records", type=Path)
    compare.add_argument("superdex_records", type=Path)
    compare.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            validate_manifest(manifest, require_frozen=args.require_frozen)
            print(json.dumps({"status": manifest.data["status"], "manifest_sha256": compute_manifest_sha256(manifest)}))
        elif args.command == "run":
            run_adapter(
                args.implementation,
                args.manifest,
                args.output_dir,
                worktree=args.worktree,
                device=args.device,
                build_type=args.build_type,
                timeout_s=args.timeout_s,
                python_executable=args.python_executable,
            )
        elif args.command == "compare":
            manifest = load_manifest(args.manifest)
            entries = [MappingEntry(**entry) for entry in manifest.data["mappings"]]
            # Bind the supplied mappings to these records, not another manifest's hash.
            expected = compute_manifest_sha256(manifest)
            for path in (args.newton_records, args.superdex_records):
                first = json.loads(path.read_text().splitlines()[0])
                if first["manifest_sha256"] != expected:
                    raise ValueError("Comparison mappings belong to another manifest")
            parser.exit(compare_runs(args.newton_records, args.superdex_records, entries, args.output))
        else:
            identity = require_reference_worktree(args.worktree, expected_sha=args.expected_sha)
            print(json.dumps({"worktree": str(identity.path), "sha": identity.sha, "clean": True}))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
