# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one frozen P3 Franka case in an isolated process (private P4 T0 tooling).

Invoke this file by absolute path with the baseline checkout's Python environment.
The solver and scene helpers must come from --source-root; only the two new tool
files come from the development checkout. This is not the T3 dev/full runner.
"""

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

BASELINE = "39f80c48da2022187823f8218131a249807597a0"
LAWS = ("C-IE", "C-EE", "C-II")
GLOBALIZATIONS = ("ArmijoOrResNorm", "Armijo", "ResNorm")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cases():
    """Expand the temporary matrix without importing Newton or initializing CUDA."""
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


def validate_source(root, manifest):
    """Reject a moved baseline or changed source/input before constructing a scene."""
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if head != BASELINE or manifest["baseline_commit"] != BASELINE:
        raise ValueError("Baseline commit mismatch")
    for relative, expected in manifest["source_hashes"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"Baseline source changed: {relative}")
    for path, expected in manifest["input_hashes"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")


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


def run(args):
    """Run the unchanged baseline step/recording methods with explicit scene options."""
    root = args.source_root.resolve()
    manifest = json.loads(args.manifest.read_text())
    selected = next((c for c in cases() if c["id"] == args.case), None)
    if selected is None or selected.get("reuse_only"):
        raise ValueError("Choose one Franka case; Sharpa is verified and reused separately")
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "case": selected,
        "baseline_commit": BASELINE,
        "argv": sys.argv,
        "cwd": str(Path.cwd()),
        "manifest_sha256": sha(args.manifest),
        "tool_hashes": {p.name: sha(p) for p in (Path(__file__), Path(__file__).with_name("p3_baseline_franka.py"))},
        "status": "RUNNING",
        "baseline_only": True,
    }
    begin = time.perf_counter()
    example = None
    try:
        validate_source(root, manifest)
        if any(name == "newton" or name.startswith("newton.") for name in sys.modules):
            raise ValueError("Baseline must start before importing Newton")
        sys.path.insert(0, str(root))
        import numpy as np  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        from newton.viewer import ViewerNull  # noqa: PLC0415

        # Load just the independent tool by file path. Its dependencies resolve to
        # the fixed checkout; no solver factory, method, or option is monkeypatched.
        path = Path(__file__).with_name("p3_baseline_franka.py")
        spec = importlib.util.spec_from_file_location("p4_baseline_franka_tool", path)
        adapter = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = adapter
        spec.loader.exec_module(adapter)
        receipt["modules_before"] = module_origins(root, manifest["source_hashes"])
        receipt["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "warp": wp.__version__}
        receipt["device"] = str(wp.get_device(args.device))
        if not wp.get_device(args.device).is_cuda:
            raise ValueError("Baseline requires CUDA; no CPU fallback")
        args.law, args.globalization = selected["law"], selected["globalization"]
        output = args.output
        args.output = output / "recording"
        try:
            example = adapter.BaselineFrankaExample(ViewerNull(), args)
        finally:
            args.output = output
        solver = example.case.solver
        if (solver._contact.contact_model, solver._globalization, solver.pcg_mode.value, solver._use_cuda_graph) != (
            args.law,
            args.globalization,
            "production",
            True,
        ):
            raise ValueError("Effective configuration differs from requested case")
        receipt["configuration_digest"] = solver._kinematics.digest.hex()
        receipt["build_seconds"] = time.perf_counter() - begin
        # Freeze the actual source node arithmetic and coefficient identities.
        from scripts.monolithic_reference.verify_p3_smooth_sdf import _nodes  # noqa: PLC0415

        smooth = solver._contact.smooth_sdf
        table = example.case.model._texture_sdf_data.numpy()
        builds = []
        for index, field in enumerate(smooth.fields):
            dims = (
                np.array(example.case.model._texture_sdf_subgrid_start_slots[index].shape)
                * table[index]["subgrid_size"]
                + 1
            )
            nodes = wp.empty(int(np.prod(dims)), dtype=wp.float64, device=args.device)
            wp.launch(
                _nodes,
                nodes.size,
                [example.case.model._texture_sdf_data, index, wp.vec3i(dims), nodes],
                device=args.device,
            )
            values = nodes.numpy()
            coefficients = field.data.coefficients.numpy()
            builds.append(
                {
                    "index": index,
                    "dims": dims.tolist(),
                    "node_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                    "coefficient_sha256": hashlib.sha256(coefficients.tobytes()).hexdigest(),
                    "identity": field.identity,
                }
            )
        (output / "smooth-build.json").write_text(
            json.dumps({"digest": smooth.digest, "fields": builds}, indent=2) + "\n"
        )
        while not example.closed:
            example.step()
        example.test_final()
        receipt["status"] = "PASS"
    except Exception:
        receipt["status"] = "FAIL"
        receipt["exception"] = traceback.format_exc()
    finally:
        if example is not None:
            example.close()
            receipt["summary"] = example.summary
            receipt["segment_graphs"] = example.case.solver._graphs.summary()
        try:
            receipt["modules_after"] = module_origins(root, manifest["source_hashes"])
        except Exception:
            receipt["status"] = "FAIL"
            receipt["origin_error"] = traceback.format_exc()
        receipt["elapsed_seconds"] = time.perf_counter() - begin
        # Example summaries contain NumPy arrays; use the original serializer if loaded.
        serializer = getattr(sys.modules.get("scripts.monolithic_reference.verify_p3_coupled"), "serial", lambda x: x)
        (args.output / "receipt.json").write_text(json.dumps(serializer(receipt), indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({"case": args.case, "status": receipt["status"], "elapsed_seconds": receipt["elapsed_seconds"]}),
        flush=True,
    )
    return int(receipt["status"] != "PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--asset-cache", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.list:
        print(json.dumps(cases(), indent=2))
        return 0
    if any(getattr(args, name) is None for name in ("source_root", "manifest", "case", "output", "asset_cache")):
        parser.error("Running requires --source-root, --manifest, --case, --output and --asset-cache")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
