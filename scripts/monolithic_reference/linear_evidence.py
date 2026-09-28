# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Source and matrix evidence shared by retained linear benchmarks."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "monolithic-p2-measurement/v3"


def pcg_call_record(result, seconds, *, use_cuda_graph):
    """Record actual work and fixed scheduling separately, including failed retries."""
    return {
        "iterations": result.iterations,
        "status": result.status.name,
        "seconds": seconds,
        "rho": result.rho,
        "rho_q": result.rho_q,
        "rho_x": result.rho_x,
        "pcg_policy": result.pcg_policy,
        "scheduled_slots": result.scheduled_slots,
        "converged_at_slot": result.converged_at_slot,
        "last_checked_slot": result.last_checked_slot,
        "failure_slot": result.failure_slot,
        "residual_current": result.residual_current,
        "output_valid": result.output_valid,
        "true_residual_checks": result.true_residual_checks,
        "residual_replacements": result.residual_replacements,
        "checkpoint_ratios": result.checkpoint_ratios,
        "use_cuda_graph": use_cuda_graph,
        "prepare_seconds": result.prepare_seconds,
        "capture_seconds": result.capture_seconds,
        "graph_rebuilt": result.graph_rebuilt,
    }


def snapshot_sources(output):
    """Save the actual working source, dependency inventory and dirty diff before running."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "source.json").exists():
        raise FileExistsError(f"Refuse to replace existing run provenance: {output}")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    hashes = {}
    with zipfile.ZipFile(output / "source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(set(paths)):
            path = ROOT / relative
            if not path.is_file() or path.suffix not in (".py", ".json", ".toml", ".lock", ".yaml", ".md", ".rst"):
                continue
            data = path.read_bytes()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            archive.writestr(relative, data)
    (output / "dirty.diff").write_bytes(git("diff", "HEAD", "--binary"))
    (output / "git-status.txt").write_bytes(git("status", "--short"))
    dependencies = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib.metadata.distributions())
    payload = "\n".join(dependencies) + "\n"
    (output / "dependencies.txt").write_text(payload)
    record = {
        "schema": SCHEMA,
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "platform": platform.platform(),
        "warp": wp.__version__,
        "thread_environment": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "cpu_count": os.cpu_count(),
        "sources": hashes,
        "dependency_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (output / "source.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def evidence_index(output):
    """Index every trace and binary artifact without self-referential hashes."""
    output = Path(output)
    rows = {
        str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "evidence-index.json" and not path.name.endswith(".sha256")
    }
    (output / "evidence-index.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def array_fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
        digest.update(name.encode())
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _rss_bytes():
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            name, value, _unit = line.split()
            values[name.removesuffix(":").lower() + "_bytes"] = int(value) * 1024
    return values
