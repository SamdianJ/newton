# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the two-sided FP32 precision check set against FP64 on the same inputs.

Every case is run in both precisions. The FP64 run is the reference response,
not ground truth for physical acceptance. Each FP32 case reports both sides:

- coverage: every step converges (no stall, no rollback, no Stopped step);
- resolution: every loaded step takes at least one Newton update, the rounding
  allowance on the loaded rows stays below the load, and the response matches
  FP64 within the case tolerance plus a few FP32 coordinate ulps.

Results are diagnostics for the development stage. The physical stage gates
remain those of the scene runners.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton
from newton._src.solvers.monolithic.tet_precision import TetObjective

from .cable_benchmarks import CableBenchmark
from .material_recipes import shell_coefficients
from .precision_checks import SolverTrace, load_margin

CASES_PATH = Path(__file__).with_name("precision_cases.json")
REPOSITORY = Path(__file__).resolve().parents[2]
FP32_EPS = float(np.finfo(np.float32).eps)


def _ulp32(value):
    return float(np.spacing(np.float32(abs(value))))


def _particle_rows(solver, particle):
    start = solver._actors.maps["particle_to_global"][particle]
    if start < 0:
        raise ValueError(f"Particle {particle} has no free rows")
    return np.arange(start, start + 3)


def _run(solver, rows, advance, steps, observe):
    """Advance ``steps`` steps under a trace; return per-step observations and traces."""
    records = []
    with SolverTrace(solver, rows=rows) as trace:
        for step in range(1, steps + 1):
            t = trace.step(lambda step=step: advance(step))
            stats = solver.last_stats
            records.append(
                {
                    "step": step,
                    "status": stats.status.name,
                    "converged": bool(stats.converged),
                    "updates": t.updates,
                    "regularized": t.regularized,
                    "pcg_iterations": int(sum(a["pcg_iterations"] for a in t.attempts)),
                    "rejected_directions": int(sum(not a["accepted"] for a in t.attempts)),
                    "response": float(observe()),
                    "allowance": t.iterates[0]["allowance"] if t.iterates and "allowance" in t.iterates[0] else None,
                }
            )
            if not stats.converged:
                break
    return records


def _judge(records32, records64, load, rows, tolerance, coordinate_scale, representation_ulps):
    """Both sides of one FP32 case against its FP64 reference."""
    complete = len(records32) == len(records64) and all(r["converged"] for r in records32)
    responses32 = np.array([r["response"] for r in records32])
    responses64 = np.array([r["response"] for r in records64])[: len(responses32)]
    peak = float(np.max(np.abs(responses64), initial=0.0))
    limit = tolerance * peak + representation_ulps * _ulp32(coordinate_scale)
    error = float(np.max(np.abs(responses32 - responses64), initial=0.0))
    allowance = np.zeros_like(load)
    if records32 and records32[0]["allowance"] is not None:
        allowance[rows] = records32[0]["allowance"]
    margin = load_margin(load, allowance, rows)
    # A step whose load is already balanced needs no update; FP64 tells which steps had work.
    # An FP32 run that stopped early is shorter; coverage already fails it.
    pairs = zip(records32, records64, strict=False)
    resolution = {
        "updates": bool(all(a["updates"] >= 1 for a, b in pairs if b["updates"] >= 1)),
        "allowance_below_load": bool(margin < 1),
        "response": bool(complete and error <= limit),
    }
    return {
        "coverage": {"converged": bool(complete)},
        "resolution": resolution,
        "PASS": bool(complete and all(resolution.values())),
        "response_error": error,
        "response_limit": limit,
        "response_peak_fp64": peak,
        "allowance_over_load": margin,
    }


def _load_vector(rows, values):
    vector = np.zeros(int(np.max(rows)) + 1)
    vector[rows] = values
    return vector


def cable_cantilever(spec, *, device, representation_ulps):
    def run(angle, segments, load):
        runs = {}
        for precision in ("fp64", "fp32"):
            case = CableBenchmark("cantilever", segments=segments, angle=angle, precision=precision, device=device)
            direction = case.rotation[:, case.axis]
            force = np.zeros((segments + 1, 3), dtype=np.float32)
            force[segments] = load["force"] * direction
            rows = _particle_rows(case.solver, segments)

            def advance(_step, case=case, force=force):
                case.state.particle_f.assign(force)
                case.solver.step(case.state, case.state, case.control, dt=case.dt)

            def observe(case=case, direction=direction):
                return (case.state.particle_q.numpy()[segments].astype(float) - case.rest[segments]) @ direction

            runs[precision] = _run(case.solver, rows, advance, load["steps"], observe)
        vector = _load_vector(rows, load["force"] * direction)
        scale = float(np.abs(case.rest[segments]).max())
        judgement = _judge(
            runs["fp32"], runs["fp64"], vector, rows, spec["response_tolerance"], scale, representation_ulps
        )
        return runs | {"judgement": judgement}

    for angle in spec["angles"]:
        for segments in spec["segments"]:
            for name, load in spec["loads"].items():
                yield (
                    f"cable_cantilever/{angle:g}deg/n{segments}/{name}",
                    {"angle": angle, "segments": segments, "load": name} | load,
                    lambda angle=angle, segments=segments, load=load: run(angle, segments, load),
                )


def cable_twist(spec, *, device, representation_ulps):
    def run(angle, segments):
        runs = {}
        edge = segments - 1
        for precision in ("fp64", "fp32"):
            case = CableBenchmark("cantilever", segments=segments, angle=angle, precision=precision, device=device)
            rows = np.array([case.solver._actors.maps["theta_to_global"][edge]])
            torque = np.zeros(segments)
            torque[edge] = spec["torque"]

            def advance(_step, case=case, torque=torque):
                case.state.particle_f.zero_()
                case.control._monolithic_cable.torque.assign(torque)
                case.solver.step(case.state, case.state, case.control, dt=case.dt)

            def observe(case=case):
                return float(case.state._monolithic_coupled.cable_theta.numpy()[edge])

            runs[precision] = _run(case.solver, rows, advance, spec["steps"], observe)
        # Twist angles are not coordinates; one ulp of a unit angle bounds their representation.
        judgement = _judge(
            runs["fp32"],
            runs["fp64"],
            _load_vector(rows, spec["torque"]),
            rows,
            spec["response_tolerance"],
            1.0,
            representation_ulps,
        )
        return runs | {"judgement": judgement}

    for angle in spec["angles"]:
        for segments in spec["segments"]:
            yield (
                f"cable_twist/{angle:g}deg/n{segments}",
                {"angle": angle, "segments": segments, "torque": spec["torque"], "steps": spec["steps"]},
                lambda angle=angle, segments=segments: run(angle, segments),
            )


def _shell(tilt, cells, precision, device):
    common = json.loads(Path(__file__).with_name("cloth_recipes.json").read_text())["common"]
    patch = json.loads(Path(__file__).with_name("cloth_recipes.json").read_text())["patch"]
    width = common["width"]
    uv = np.array([(i, j) for i in range(cells + 1) for j in range(cells + 1)]) / cells * width - width / 2
    c, s = np.cos(np.radians(tilt)), np.sin(np.radians(tilt))
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) if tilt else np.eye(3)
    positions = (np.c_[uv, np.zeros(len(uv))] @ rotation.T).astype(np.float32).astype(float)
    triangles = np.array(
        [
            face
            for i in range(cells)
            for j in range(cells)
            for face in (
                (i * (cells + 1) + j, (i + 1) * (cells + 1) + j, (i + 1) * (cells + 1) + j + 1),
                (i * (cells + 1) + j, (i + 1) * (cells + 1) + j + 1, i * (cells + 1) + j + 1),
            )
        ]
    )
    fixed = np.any(np.isclose(abs(uv), width / 2), axis=1)
    material = shell_coefficients(
        {
            "young_modulus_Pa": patch["young_modulus"],
            "poisson_ratio": common["poisson_ratio"],
            "thickness_m": patch["thickness"],
        }
    )
    faces = positions[triangles]
    area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
    mass = np.zeros(len(positions))
    np.add.at(mass, triangles.ravel(), np.repeat(area * patch["surface_density"] / 3, 3))
    builder = newton.ModelBuilder(gravity=(0, 0, 0))
    for i, x in enumerate(positions):
        builder.add_particle(wp.vec3(x), wp.vec3(0), 0 if fixed[i] else mass[i])
    for face in triangles:
        builder.add_triangle(*face, tri_ke=material[1], tri_ka=material[0], tri_kd=0, tri_drag=0, tri_lift=0)
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.1,
        cloth_inputs=[
            SolverMonolithic.ClothInput(
                triangle_ids=tuple(range(len(triangles))),
                bending=material[2:],
                thickness=patch["thickness"],
                surface_density=patch["surface_density"],
            )
        ],
        precision=SolverMonolithic.Precision(precision),
        newton_max_iterations=60,
        linear_max_iterations=300,
        linear_tolerance=1e-4,
    )
    center = (cells // 2) * (cells + 1) + cells // 2
    return solver, model, rotation[:, 2], center, common["dt"]


def shell_patch(spec, *, device, representation_ulps):
    def run(tilt, force):
        runs = {}
        for precision in ("fp64", "fp32"):
            solver, model, normal, center, dt = _shell(tilt, spec["cells"], precision, device)
            state, control = solver.create_state(dt=dt), solver.create_control()
            rest = state.particle_q.numpy().astype(float)
            forces = np.zeros((model.particle_count, 3), dtype=np.float32)
            forces[center] = force * normal
            rows = _particle_rows(solver, center)

            def advance(_step, solver=solver, state=state, control=control, forces=forces, dt=dt):
                state.particle_f.assign(forces)
                solver.step(state, state, control, dt=dt)

            def observe(state=state, rest=rest, center=center, normal=normal):
                return (state.particle_q.numpy()[center].astype(float) - rest[center]) @ normal

            runs[precision] = _run(solver, rows, advance, spec["steps"], observe)
        # The patch centre sits at the origin; scale its ulps by the patch width.
        judgement = _judge(
            runs["fp32"],
            runs["fp64"],
            _load_vector(rows, force * normal),
            rows,
            spec["response_tolerance"],
            float(np.abs(rest).max()),
            representation_ulps,
        )
        return runs | {"judgement": judgement}

    for tilt in spec["tilts"]:
        for name, force in spec["loads"].items():
            yield (
                f"shell_patch/{tilt:g}deg/{name}",
                {"tilt": tilt, "cells": spec["cells"], "force": force, "steps": spec["steps"]},
                lambda tilt=tilt, force=force: run(tilt, force),
            )


def _moving_tet(vx, g, precision, device):
    builder = newton.ModelBuilder(gravity=(0, 0, -g))
    root = builder.add_link(mass=1.0, inertia=wp.mat33(0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01))
    builder.add_articulation([builder.add_joint_revolute(-1, root, axis=(0, 1, 0))])
    builder.add_soft_mesh(
        pos=(0.0, 0.0, 1.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(vx, 0, 0),
        vertices=[(0, 0, 0), (0.04, 0, 0), (0, 0.03, 0), (0, 0, 0.02)],
        indices=[0, 1, 2, 3],
        density=1000,
        k_mu=10000 / 2.6,
        k_lambda=10000 * 0.3 / (1.3 * 0.4),
        k_damp=0,
        particle_radius=0.001,
        tri_ke=0,
        tri_ka=0,
        tri_kd=0,
        tri_drag=0,
        tri_lift=0,
        edge_ke=0,
        edge_kd=0,
    )
    model = builder.finalize(device=device)
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        tet_rest_density=wp.full(model.tet_count, 1000, dtype=float, device=model.device),
        joint_terms=SolverMonolithic.JointTerms(limits=True, limit_width=(0.01,)),
        precision=SolverMonolithic.Precision(precision),
    )
    return model, solver


def moving_tet(spec, *, device, representation_ulps):
    dt = spec["dt"]

    def run(vx, g):
        runs = {}
        for precision in ("fp64", "fp32"):
            model, solver = _moving_tet(vx, g, precision, device)
            state, control = solver.create_state(dt=dt), model.control()
            rows = np.concatenate([_particle_rows(solver, p) for p in range(model.particle_count)])

            def advance(_step, solver=solver, state=state, control=control):
                solver.step(state, state, control, dt=dt)

            def observe(state=state):
                return float(state.particle_qd.numpy()[:, 2].astype(float).mean())

            runs[precision] = _run(solver, rows, advance, 1, observe)
        # The load is gravity on every particle's z row.
        mass = model.particle_mass.numpy().astype(float)
        z_rows = rows[2::3]
        judgement = _judge(
            [r | {"allowance": r["allowance"][2::3]} for r in runs["fp32"]],
            runs["fp64"],
            _load_vector(z_rows, -g * mass),
            z_rows,
            spec["response_tolerance"],
            0.0,
            representation_ulps,
        )
        # FP64 is no reference here: a fast body can stop at its predictor in either
        # precision. Galilean invariance fixes the expected gravity response instead.
        expected = -g * dt
        galilean = {
            precision: float(abs(runs[precision][0]["response"] - expected) / abs(expected))
            for precision in ("fp64", "fp32")
        }
        judgement["galilean_error"] = galilean
        judgement["resolution"]["updates"] = all(runs[p][0]["updates"] >= 1 for p in ("fp64", "fp32"))
        judgement["resolution"]["response"] = all(e <= spec["response_tolerance"] for e in galilean.values())
        judgement["PASS"] = bool(judgement["coverage"]["converged"] and all(judgement["resolution"].values()))
        return runs | {"judgement": judgement}

    for g in spec["gravity"]:
        for vx in spec["velocities"]:
            yield (
                f"moving_tet/g{g:g}/vx{vx:g}",
                {"vx": vx, "gravity": g, "dt": dt},
                lambda vx=vx, g=g: run(vx, g),
            )


def _rotation(degrees):
    """Rotation by ``degrees`` about (1, 1, 1): it turns every element and strains none."""
    axis = np.ones(3) / np.sqrt(3.0)
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = np.radians(degrees)
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * k @ k


def _rotations(spec):
    """The unrotated case first, under its original name, then each rotated variant."""
    return (0.0, *spec.get("rotations", ()))


def _rotated(name, degrees):
    return f"{name}/rot{degrees:g}" if degrees else name


def _tet_grid(nu, stretch, device, rotation=0.0):
    """Return a 2x1x1 tet grid stretched isochorically and turned by ``rotation`` degrees, and an evaluator."""
    vectors = {wp.float32: wp.vec3, wp.float64: wp.vec3d}
    young = 1.0e6
    mu, lam = young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_soft_grid(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        vel=wp.vec3(0.0),
        dim_x=2,
        dim_y=1,
        dim_z=1,
        cell_x=0.1,
        cell_y=0.1,
        cell_z=0.1,
        density=1000.0,
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
        add_surface_mesh_edges=False,
        particle_radius=0.001,
    )
    model = builder.finalize(device=device)
    rest = model.particle_q.numpy().astype(float)
    lateral = 1 / np.sqrt(stretch)
    xn = (rest @ np.diag([stretch, lateral, lateral]) @ _rotation(rotation).T).astype(np.float32)
    objectives = {
        dtype: TetObjective(model, material_model="kim_stable_no_log", mass_mode="lumped", dtype=dtype)
        for dtype in (wp.float32, wp.float64)
    }
    zeros = wp.zeros(model.particle_count, dtype=wp.vec3, device=device)

    def evaluate(dtype, step):
        native = np.float32 if dtype == wp.float32 else np.float64

        def array(a):
            return wp.array(np.asarray(a, dtype=native), dtype=vectors[dtype], device=device)

        o = objectives[dtype]
        x = (xn + step).astype(np.float32)
        o.evaluate(
            array(x), array(xn), array(np.zeros_like(xn)), zeros, model.gravity, dt=1e-2, displacement=array(step)
        )
        return o.residual.numpy().astype(float).ravel(), o.absolute.numpy().astype(float).ravel()

    return xn, evaluate, lam / mu


def tet_noise(spec, *, device, representation_ulps):
    """Element-level coverage: FP32 residual jitter against the rounding allowance."""
    del representation_ulps

    def run(nu, stretch, rotation):
        xn, evaluate, ratio_moduli = _tet_grid(nu, stretch, device, rotation)
        rng = np.random.default_rng(0)
        step0 = (spec["scale"] * rng.standard_normal(xn.shape)).astype(np.float32)
        errors, magnitudes = [], []
        for _ in range(spec["samples"]):
            step = (step0 + spec["scale"] * 1e-3 * rng.standard_normal(xn.shape)).astype(np.float32)
            r32, a32 = evaluate(wp.float32, step)
            r64, _ = evaluate(wp.float64, step.astype(np.float64))
            errors.append(r32 - r64)
            magnitudes.append(a32)
        errors, magnitudes = np.array(errors), np.array(magnitudes)
        jitter = errors.max(axis=0) - errors.min(axis=0)
        allowance = spec["noise_multiple"] * FP32_EPS * magnitudes.max(axis=0)
        ratio = float(np.max(jitter / np.maximum(allowance, np.finfo(float).tiny)))
        judgement = {
            "coverage": {"noise_below_allowance": ratio <= 1},
            "noise_over_allowance": ratio,
            "lambda_over_mu": ratio_moduli,
            "PASS": bool(ratio <= 1),
        }
        return {"judgement": judgement}

    for rotation in _rotations(spec):
        for nu in spec["poisson"]:
            for stretch in spec["stretch"]:
                yield (
                    _rotated(f"tet_noise/nu{nu:g}/stretch{stretch:g}", rotation),
                    {"poisson": nu, "stretch": stretch, "rotation_deg": rotation},
                    lambda nu=nu, stretch=stretch, rotation=rotation: run(nu, stretch, rotation),
                )


def tet_resolution(spec, *, device, representation_ulps):
    """Element-level resolution: the force of a small isochoric shear step must exceed the allowance.

    The step shears the grid by ``strain`` (x displacement proportional to y),
    which changes no volume to first order: the case most exposed to the
    volumetric rounding of nearly incompressible materials.
    """
    del representation_ulps

    def run(nu, stretch, strain, rotation):
        xn, evaluate, ratio_moduli = _tet_grid(nu, stretch, device, rotation)
        shear = np.zeros(xn.shape)
        shear[:, 0] = strain * xn[:, 1].astype(float)
        step = shear.astype(np.float32)
        base64, _ = evaluate(wp.float64, np.zeros(xn.shape))
        moved64, _ = evaluate(wp.float64, step.astype(np.float64))
        _, magnitude32 = evaluate(wp.float32, step)
        response = np.abs(moved64 - base64)
        load = float(response.max())
        allowance = spec["noise_multiple"] * FP32_EPS * magnitude32
        loaded = response > 0.1 * load
        ratio = float(np.max(allowance[loaded] / response[loaded]))
        judgement = {
            "resolution": {"allowance_below_load": ratio < 1},
            "allowance_over_load": ratio,
            "load": load,
            "lambda_over_mu": ratio_moduli,
            "PASS": bool(ratio < 1),
        }
        return {"judgement": judgement}

    for rotation in _rotations(spec):
        for nu in spec["poisson"]:
            for stretch in spec["stretch"]:
                for strain in spec["strains"]:
                    yield (
                        _rotated(f"tet_resolution/nu{nu:g}/stretch{stretch:g}/strain{strain:g}", rotation),
                        {"poisson": nu, "stretch": stretch, "strain": strain, "rotation_deg": rotation},
                        lambda nu=nu, stretch=stretch, strain=strain, rotation=rotation: run(
                            nu, stretch, strain, rotation
                        ),
                    )


def _shell_element_patch(nu, stretch, device, rotation=0.0):
    """Return a 2x2 shell patch, area-preservingly stretched and turned by ``rotation`` degrees, and an evaluator."""
    from newton._src.solvers.monolithic.cloth import ClothObjective  # noqa: PLC0415

    cells, width = 2, 0.2
    uv = np.array([(i, j) for i in range(cells + 1) for j in range(cells + 1)]) / cells * width - width / 2
    rest = np.c_[uv, np.zeros(len(uv))].astype(np.float32).astype(float)
    triangles = np.array(
        [
            face
            for i in range(cells)
            for j in range(cells)
            for face in (
                (i * (cells + 1) + j, (i + 1) * (cells + 1) + j, (i + 1) * (cells + 1) + j + 1),
                (i * (cells + 1) + j, (i + 1) * (cells + 1) + j + 1, i * (cells + 1) + j + 1),
            )
        ]
    )
    edges = {}
    for face, triangle in enumerate(triangles):
        for opposite in range(3):
            a, b = triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3]
            edges.setdefault(tuple(sorted((a, b))), []).append((face, opposite))
    stencils = []
    for face, triangle in enumerate(triangles):
        ids = list(triangle)
        for opposite in range(3):
            edge = tuple(sorted((triangle[(opposite + 1) % 3], triangle[(opposite + 2) % 3])))
            neighbors = [entry for entry in edges[edge] if entry[0] != face]
            ids.append(int(triangles[neighbors[0][0], neighbors[0][1]]) if neighbors else -1)
        stencils.append(ids)
    young, thickness = 2.0e5, 2.0e-4
    mu = young * thickness / (2 * (1 + nu))
    lam = young * thickness * nu / (1 - nu**2)
    materials = np.tile([lam, mu, lam * thickness**2 / 12, 2 * mu * thickness**2 / 12], (len(triangles), 1))
    xn = ((rest * np.array([stretch, 1 / stretch, 1.0])) @ _rotation(rotation).T).astype(np.float32).astype(float)
    objectives = {
        dtype: ClothObjective(rest, stencils, materials, dtype=dtype, device=device)
        for dtype in (wp.float32, wp.float64)
    }

    def evaluate(dtype, step):
        native = np.float32 if dtype == wp.float32 else np.float64
        o = objectives[dtype]
        _, residual, _ = o.evaluate_host(xn.astype(native), step=np.asarray(step, dtype=native))
        return residual.astype(float).ravel(), o.rounding.numpy().astype(float).ravel()

    return xn, evaluate, lam / mu


def shell_noise(spec, *, device, representation_ulps):
    """Element-level coverage of the shell rounding magnitude, membrane included."""
    del representation_ulps

    def run(nu, stretch, rotation):
        xn, evaluate, ratio_moduli = _shell_element_patch(nu, stretch, device, rotation)
        rng = np.random.default_rng(0)
        step0 = (spec["scale"] * rng.standard_normal(xn.shape)).astype(np.float32)
        errors, magnitudes = [], []
        for _ in range(spec["samples"]):
            step = (step0 + spec["scale"] * 1e-3 * rng.standard_normal(xn.shape)).astype(np.float32)
            r32, a32 = evaluate(wp.float32, step)
            r64, _ = evaluate(wp.float64, step.astype(np.float64))
            errors.append(r32 - r64)
            magnitudes.append(a32)
        errors, magnitudes = np.array(errors), np.array(magnitudes)
        jitter = errors.max(axis=0) - errors.min(axis=0)
        allowance = spec["noise_multiple"] * FP32_EPS * magnitudes.max(axis=0)
        ratio = float(np.max(jitter / np.maximum(allowance, np.finfo(float).tiny)))
        judgement = {
            "coverage": {"noise_below_allowance": ratio <= 1},
            "noise_over_allowance": ratio,
            "lambda_over_mu": ratio_moduli,
            "PASS": bool(ratio <= 1),
        }
        return {"judgement": judgement}

    for rotation in _rotations(spec):
        for nu in spec["poisson"]:
            for stretch in spec["stretch"]:
                yield (
                    _rotated(f"shell_noise/nu{nu:g}/stretch{stretch:g}", rotation),
                    {"poisson": nu, "stretch": stretch, "rotation_deg": rotation},
                    lambda nu=nu, stretch=stretch, rotation=rotation: run(nu, stretch, rotation),
                )


def shell_resolution(spec, *, device, representation_ulps):
    """Element-level resolution: an in-plane shear step's force must exceed the shell allowance."""
    del representation_ulps

    def run(nu, stretch, strain, rotation):
        xn, evaluate, ratio_moduli = _shell_element_patch(nu, stretch, device, rotation)
        shear = np.zeros(xn.shape)
        shear[:, 0] = strain * xn[:, 1]
        step = shear.astype(np.float32)
        base64, _ = evaluate(wp.float64, np.zeros(xn.shape))
        moved64, _ = evaluate(wp.float64, step.astype(np.float64))
        _, magnitude32 = evaluate(wp.float32, step)
        response = np.abs(moved64 - base64)
        load = float(response.max())
        allowance = spec["noise_multiple"] * FP32_EPS * magnitude32
        loaded = response > 0.1 * load
        ratio = float(np.max(allowance[loaded] / response[loaded]))
        judgement = {
            "resolution": {"allowance_below_load": ratio < 1},
            "allowance_over_load": ratio,
            "load": load,
            "lambda_over_mu": ratio_moduli,
            "PASS": bool(ratio < 1),
        }
        return {"judgement": judgement}

    for rotation in _rotations(spec):
        for nu in spec["poisson"]:
            for stretch in spec["stretch"]:
                for strain in spec["strains"]:
                    yield (
                        _rotated(f"shell_resolution/nu{nu:g}/stretch{stretch:g}/strain{strain:g}", rotation),
                        {"poisson": nu, "stretch": stretch, "strain": strain, "rotation_deg": rotation},
                        lambda nu=nu, stretch=stretch, strain=strain, rotation=rotation: run(
                            nu, stretch, strain, rotation
                        ),
                    )


FAMILIES = {
    "cable_cantilever": cable_cantilever,
    "cable_twist": cable_twist,
    "shell_patch": shell_patch,
    "moving_tet": moving_tet,
    "tet_noise": tet_noise,
    "tet_resolution": tet_resolution,
    "shell_noise": shell_noise,
    "shell_resolution": shell_resolution,
}


def iter_cases(spec=None, *, families=None, device="cpu"):
    """Yield every case of the check set as ``(name, parameters, run)``, ``run`` deferred."""
    spec = json.loads(CASES_PATH.read_text()) if spec is None else spec
    for family in families or FAMILIES:
        yield from FAMILIES[family](spec[family], device=device, representation_ulps=spec["representation_ulps"])


def _identity():
    """Hash the solver package, every loaded reference module and the recipes read."""
    root = Path(__file__).resolve().parent
    loaded = {
        Path(m.__file__).resolve()
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and Path(m.__file__).resolve().parent == root
    }
    sources = sorted(
        {*(REPOSITORY / "newton/_src/solvers/monolithic").glob("*.py"), *loaded}
        | {CASES_PATH, root / "cable_recipes.json", root / "cloth_recipes.json"}
    )
    hashes = {str(p.relative_to(REPOSITORY)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}

    def git(*args):
        return subprocess.run(["git", *args], cwd=REPOSITORY, capture_output=True, text=True, check=False).stdout

    if Path(newton.__file__).resolve() != REPOSITORY / "newton/__init__.py":
        raise RuntimeError("Precision checks imported Newton outside the tested checkout")
    return {
        "commit": git("rev-parse", "HEAD").strip(),
        "clean": git("status", "--porcelain").strip() == "",
        "newton": newton.__file__,
        "warp": wp.__version__,
        "source_hashes": hashes,
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", action="append", choices=sorted(FAMILIES), help="Run only these families")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    wp.config.log_level = wp.LOG_WARNING
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    spec = json.loads(CASES_PATH.read_text())
    identity = _identity()
    (args.output / "receipt.json").write_text(json.dumps(identity | {"device": args.device}, indent=2) + "\n")
    selected = args.family or list(FAMILIES)
    results, start = [], time.perf_counter()
    for name, parameters, run in iter_cases(spec, families=selected, device=args.device):
        # One failing case never stops the collection of the others.
        try:
            result = {"name": name, "parameters": parameters} | run()
            verdict = "PASS" if result["judgement"]["PASS"] else "FAIL"
        except Exception as error:
            result = {"name": name, "parameters": parameters, "error": f"{type(error).__name__}: {error}"}
            result["judgement"] = {"PASS": False}
            verdict = "ERROR"
        result["verdict"] = verdict
        results.append(result)
        print(f"{verdict} {name}", flush=True)
    summary = {
        "cases_version": spec["version"],
        "cases_status": spec["status"],
        "families": selected,
        "filtered": sorted(selected) != sorted(FAMILIES),
        "seconds": time.perf_counter() - start,
        "passed": sum(r["verdict"] == "PASS" for r in results),
        "failed": sum(r["verdict"] == "FAIL" for r in results),
        "errors": sum(r["verdict"] == "ERROR" for r in results),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, default=_jsonable) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
