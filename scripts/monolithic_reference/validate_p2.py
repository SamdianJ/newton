# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the small P2 development gate or a resumable serial overnight matrix."""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEV_MODULES = (
    "test_monolithic_owned_mass_matrix",
    "test_solver_monolithic_linear",
    "test_monolithic_pcg_modes",
    "test_monolithic_execution_options",
    "test_monolithic_p2_8bc_runner",
    "test_monolithic_validation_workflow",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def build_jobs(suite, repo, baseline):
    modules = (
        DEV_MODULES
        if suite == "dev"
        else tuple(p.stem for p in sorted((repo / "newton/tests").glob("test_*monolithic*.py")))
    )
    jobs = [
        {
            "name": "regression",
            "kind": "tests",
            "cwd": str(repo),
            "arguments": ["-m", "unittest", "-v", *("newton.tests." + m for m in modules)],
        }
    ]
    if suite == "dev":
        return jobs

    def add(kind, mesh, variant, repeat):
        name = f"{kind}-r{mesh}-{variant}-rep{repeat:02d}"
        old = variant == "baseline"
        arguments = [
            "-m",
            "scripts.monolithic_reference.run_p2_8a" if old else "scripts.monolithic_reference.run_p2_8bc",
            "--kind",
            kind,
            "--refinement",
            str(mesh),
            "--device",
            "cuda:0",
        ]
        if not old:
            arguments += [
                "--mass-matrix",
                "owned" if variant in ("owned", "combined") else "reference",
                "--pcg-mode",
                "production" if variant in ("production", "combined") else "diagnostic",
            ]
        jobs.append({"name": name, "kind": kind, "cwd": str(baseline if old else repo), "arguments": arguments})

    for repeat in range(1, 6):
        for kind, mesh, variants in (
            ("sharpa", 3, ("baseline", "diagnostic", "owned", "production")),
            ("sharpa", 2, ("diagnostic", "owned")),
            ("tet", 5, ("baseline", "production")),
        ):
            for variant in variants if repeat % 2 else reversed(variants):
                add(kind, mesh, variant, repeat)
    for mesh in (3, 4, 6):
        for variant in ("baseline", "production"):
            add("tet", mesh, variant, 1)
    add("tet", 5, "diagnostic", 1)
    add("sharpa", 2, "production", 1)
    for mesh in (3, 2):
        add("sharpa", mesh, "combined", 1)
    return jobs


def source_signature(repo, *, clean):
    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=repo)

    status = git("status", "--porcelain").decode()
    if clean and status:
        raise ValueError(f"Night validation requires a clean frozen checkout: {repo}")
    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")
    return {
        "repo": str(repo),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "status": status,
        "diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "untracked": {name: digest(repo / name) for name in untracked if name and (repo / name).is_file()},
    }


def validate_resume(state, signature):
    if state["signature"] != signature:
        raise ValueError("Source, dependencies or plan changed; use a new output directory")
    for result in state["completed"].values():
        if digest(Path(result["log"])) != result["log_sha256"]:
            raise ValueError("A completed job log changed")
        if result["output"]:
            output = Path(result["output"])
            index = output / "evidence-index.json"
            if digest(index) != result["index_sha256"]:
                raise ValueError("A completed evidence index changed")
            for name, expected in json.loads(index.read_text()).items():
                if digest(output / name) != expected:
                    raise ValueError(f"Completed evidence changed: {output / name}")


def archive_attempt(root, name, output):
    parent = root / "attempts" / name
    parent.mkdir(parents=True, exist_ok=True)
    archive = parent / f"attempt-{len(list(parent.iterdir())) + 1:03d}"
    output.rename(archive)
    return archive


def pcg_budget(rows):
    iterations = [call["iterations"] for row in rows for call in row["pcg_calls"]]
    p95 = float(np.percentile(iterations, 95)) if iterations else 0.0
    maximum = max(iterations, default=0)
    return {
        "actual_calls": len(iterations),
        "p95": p95,
        "maximum": maximum,
        "p95_limit": 64,
        "hard_cap": 200,
        "p95_passed": p95 <= 64,
        "hard_cap_passed": maximum <= 200,
    }


def summarize_run(output, kind):
    compact = json.loads((output / "compact.json").read_text())
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    dt = compact["physical_dt_s"] if kind == "sharpa" else compact["dt"]
    duration = 4.5 if kind == "sharpa" else 36.0
    normal = sum(row["status"] == "SUCCESS" for row in rows) / max(1, len(rows))
    ratios = [
        row[key]
        for row in rows
        if row["status"] == "SUCCESS"
        for key in ("residual_ratio", "q_residual_ratio", "x_residual_ratio")
    ]
    det = [row["min_det_f"] for row in rows if row["min_det_f"] is not None]
    with np.load(output / "final_state.npz") as final:
        finite = all(np.isfinite(final[key]).all() for key in ("q", "qd", "x", "v"))
    checks = {
        "full_schedule": compact["complete"]
        and len(rows) == round(duration / dt)
        and all(abs(row["time"] - (i + 1) * dt) < 1e-8 for i, row in enumerate(rows)),
        "no_rollback": not any(row["rollback"] for row in rows),
        "normal_fraction": normal >= 0.99,
        "finite_final_state": bool(finite),
        "normal_residuals": bool(ratios) and all(np.isfinite(value) and value <= 1 for value in ratios),
        "det_f": bool(det) and all(np.isfinite(value) and value >= (0.1 if kind == "sharpa" else 0.2) for value in det),
    }
    if kind == "sharpa":
        checks["sharpa_stability"] = compact["quality"]["stability_passed"]
        checks["anchor_unchanged"] = all(row["anchor_unchanged"] for row in rows)
    budget = pcg_budget(rows)
    return {
        "screen_checks": checks,
        "screen_passed": all(checks.values()),
        "pcg": budget,
        "solver_seconds": compact["work"]["solver_seconds"],
        "setup_seconds": compact["setup_seconds"],
        "performance_admission": "REVIEW_REQUIRED: compare quality, repeated costs, tails, memory and actual clocks",
        "limits": "This screen does not close timestep accuracy, fixed-node trajectory comparison, near-static or G6/G7.",
    }


def run_jobs(root, jobs, state, *, clean):
    env = {
        **os.environ,
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
    }
    progress = root / "progress.json"
    for job in jobs:
        name = job["name"]
        if name in state["completed"]:
            print(f"SKIP {name}: verified completed evidence", flush=True)
            continue
        actual = source_signature(Path(job["cwd"]), clean=clean)
        if actual != state["signature"]["sources"][job["cwd"]]:
            raise ValueError("Frozen source changed during validation")
        output = None if job["kind"] == "tests" else root / "runs" / name
        if output and output.exists():
            archived = archive_attempt(root, name, output)
            state.setdefault("archived_attempts", []).append(str(archived))
        attempt = len(list((root / "logs").glob(name + ".*.log"))) + 1
        log_path = root / "logs" / f"{name}.{attempt:03d}.log"
        command = [sys.executable, *job["arguments"]]
        if output:
            command += ["--output", str(output)]
        state["status"] = "RUNNING"
        state["active"] = {"name": name, "command": command, "start_unix_s": time.time(), "log": str(log_path)}
        write_json(progress, state)
        print(f"START {name} ({len(state['completed']) + 1}/{len(jobs)}) — {log_path}", flush=True)
        with log_path.open("x") as log:
            child = subprocess.Popen(
                command, cwd=job["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            state["active"]["pid"] = child.pid
            write_json(progress, state)
            try:
                code = child.wait()
            except KeyboardInterrupt:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                state["status"] = "INTERRUPTED"
                write_json(progress, state)
                raise
        result = {
            **state.pop("active"),
            "returncode": code,
            "end_unix_s": time.time(),
            "log_sha256": digest(log_path),
            "output": str(output) if output else None,
        }
        if code:
            state["status"] = "FAILED"
            state.setdefault("failures", []).append(result)
            write_json(progress, state)
            return 1
        if output:
            result["index_sha256"] = digest(output / "evidence-index.json")
            result["screen"] = summarize_run(output, job["kind"])
        state["completed"][name] = result
        write_json(progress, state)
        print(f"DONE {name}: {result['end_unix_s'] - result['start_unix_s']:.1f}s", flush=True)
    open_gates = [
        name
        for name, result in state["completed"].items()
        if "screen" in result
        and (
            not result["screen"]["screen_passed"]
            or not result["screen"]["pcg"]["p95_passed"]
            or not result["screen"]["pcg"]["hard_cap_passed"]
        )
    ]
    state["status"] = "COMPLETED_WITH_OPEN_GATES" if open_gates else "COMPLETED"
    state["open_gates"] = open_gates
    state["performance_admission"] = "NOT_AUTOMATIC; 8A budgets remain draft and telemetry requires review"
    write_json(progress, state)
    report = [
        f"# {state['status']}",
        "",
        f"Completed jobs: {len(state['completed'])}; runs with open gates: {len(open_gates)}.",
        "",
        "| Job | Solver seconds | PCG call p95 | Numerical screen | PCG budget |",
        "|---|---:|---:|---|---|",
    ]
    for name, result in state["completed"].items():
        if "screen" not in result:
            report.append(f"| {name} | — | — | Tests passed | — |")
            continue
        screen = result["screen"]
        pcg = screen["pcg"]
        report.append(
            f"| {name} | {screen['solver_seconds']:.3f} | {pcg['p95']:.1f} | "
            f"{'PASS' if screen['screen_passed'] else 'FAIL'} | "
            f"{'PASS' if pcg['p95_passed'] and pcg['hard_cap_passed'] else 'FAIL'} |"
        )
    report += [
        "",
        "Performance admission requires review of matched clocks, quality, repeated costs, tails and memory.",
        "These screens do not close timestep accuracy, fixed-node trajectory comparison, near-static or G6/G7.",
        "See plan.json, progress.json and the per-job logs/raw evidence for provenance and details.",
    ]
    (root / "summary.md").write_text("\n".join(report) + "\n")
    print(f"{state['status']}: {len(state['completed'])} jobs; {len(open_gates)} runs with open gates", flush=True)
    return 2 if open_gates else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("dev", "night"), default="dev")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-repo", type=Path, default=REPO.parent / "monolithic-p28-baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--list", action="store_true", help="Print the plan without importing Warp or running jobs.")
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    baseline = args.baseline_repo.resolve()
    jobs = build_jobs(args.suite, REPO, baseline)
    if args.list:
        print(json.dumps(jobs, indent=2))
        return 0
    if args.resume and args.output is None:
        parser.error("--resume requires --output")
    root = (args.output or Path("/tmp") / f"monolithic-p2-{args.suite}-{datetime.now():%Y%m%d-%H%M%S}").resolve()
    if root.is_relative_to(REPO):
        parser.error("Keep evidence outside the source checkout")
    if args.suite == "night" and baseline == REPO:
        parser.error("Use a separate, reviewed reference checkout for the baseline")
    root.mkdir(parents=True, exist_ok=True)
    with (root / "validation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sources = {cwd: source_signature(Path(cwd), clean=args.suite == "night") for cwd in {j["cwd"] for j in jobs}}
        dependencies = subprocess.check_output(["uv", "pip", "freeze", "--python", sys.executable])
        signature = {
            "sources": sources,
            "jobs": jobs,
            "driver_sha256": digest(Path(__file__)),
            "dependencies_sha256": hashlib.sha256(dependencies).hexdigest(),
        }
        progress = root / "progress.json"
        if args.resume:
            state = json.loads(progress.read_text())
            validate_resume(state, signature)
        else:
            if progress.exists():
                parser.error("Output already contains a run; use --resume or a fresh --output")
            state = {"signature": signature, "completed": {}, "status": "PLANNED", "suite": args.suite}
            (root / "dependencies.txt").write_bytes(dependencies)
            write_json(root / "plan.json", {"signature": signature, "jobs": jobs})
            write_json(progress, state)
        (root / "logs").mkdir(exist_ok=True)
        # The probe exits before measurements, so it cannot occupy GPU memory or pollute telemetry.
        with (root / "device-check.log").open("a") as log:
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import warp as wp; wp.init(); "
                    "assert wp.is_cuda_available(), 'CUDA validation requires an available GPU'",
                ],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if probe.returncode:
            raise RuntimeError(f"CUDA preflight failed; see {root / 'device-check.log'}")
        return run_jobs(root, jobs, state, clean=args.suite == "night")


if __name__ == "__main__":
    sys.exit(main())
