# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Allowlisted scene recipes shared by profiling and computational replay."""

from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import numpy as np
import warp as wp

from scripts.monolithic_reference.coupled_fixtures import make_solver, make_tiny
from scripts.monolithic_reference.retained_assets import validate_inputs
from scripts.monolithic_reference.run_precision import build, cases


def build_scene(name, *, device="cuda:0", graph=True, contact_dataflow="legacy", spatial=None):
    """Select the contact algorithm at the existing offline constructor seam."""
    if contact_dataflow not in ("legacy", "pooled", "spatial"):
        raise ValueError("Unsupported contact dataflow")
    if (contact_dataflow == "spatial") != (spatial is not None):
        raise ValueError("Spatial experiments require explicit capacity and law margin")
    context = nullcontext()
    if contact_dataflow == "pooled":
        from newton._src.solvers.monolithic.contact_pipeline import _PooledContactObjective  # noqa: PLC0415

        context = patch("newton._src.solvers.monolithic.solver_monolithic.ContactObjective", _PooledContactObjective)
    if contact_dataflow == "spatial":
        from newton._src.solvers.monolithic.contact_pipeline import _SpatialContactObjective  # noqa: PLC0415

        if type(spatial) is not dict or set(spatial) != {"candidate_capacity", "law_margin"}:
            raise ValueError("Invalid spatial experiment parameters")
        context = patch(
            "newton._src.solvers.monolithic.solver_monolithic.ContactObjective",
            partial(_SpatialContactObjective, **spatial),
        )
    with context:
        result = _build_scene(name, device=device, graph=graph)
    if contact_dataflow == "spatial":
        # Allocate fixed query topology before snapshot reconstruction. Actual
        # steps overwrite these initial references from their physical input.
        _, run, state, _ = result
        k, h = run._kinematics, state._monolithic_coupled
        q = k.gather_q(state)
        k.evaluate(q)
        run._contact.begin_step(
            state.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            kinematics=k,
            q0=q,
            rigid_com=h.rigid_com,
            rigid_rotation=h.rigid_rotation,
        )
    return result


def _build_scene(name, *, device, graph):
    """Reconstruct a supported scene without advancing its physical clock."""
    if name in ("actor-coupling", "actor-coupling-before", "actor-coupling-permuted"):
        from scripts.monolithic_reference.run_actor_coupling import (  # noqa: PLC0415
            build_model,
            load_case,
            targets,
        )
        from scripts.monolithic_reference.run_actor_coupling import (  # noqa: PLC0415
            make_solver as make_actor_solver,
        )

        model = build_model(device=device, before=name.endswith("-before"))
        if name.endswith("-permuted"):
            from newton._src.solvers.monolithic.actors import ActorTable  # noqa: PLC0415

            table = ActorTable.from_model(model, actor_order=(3, 2, 0, 1))
            # Use the existing internal layout seam only in this offline recipe.
            with patch.object(ActorTable, "from_model", return_value=table):
                run = make_actor_solver(model, mode="production", graph=graph)
        else:
            run = make_actor_solver(model, mode="production", graph=graph)
        state = run.create_state(dt=0.001)
        case = load_case()
        samples = []
        arm = run._articulation_actor
        for step in range(case["steps"] + 1):
            q, qd, _ = targets(case, step)
            position = np.zeros(model.joint_coord_count, dtype=np.float32)
            velocity = np.zeros(model.joint_dof_count, dtype=np.float32)
            position[arm.coord_indices[0]] = q
            velocity[arm.dof_indices[0]] = qd
            samples.append((position, velocity))
        return model, run, state, samples
    if name == "tiny-contact":
        model = make_tiny("normal_loading", device)
        model.joint_q.fill_(0.005)
        run = make_solver(model, "normal_loading", "C-IE", mode="production", graph=graph)
        state = run.create_state(dt=0.001)
        return model, run, state, None
    names = {"franka-default": "franka-cube", "sharpa-default": "sharpa-recorded-10mm"}
    if name not in names or not graph or not wp.get_device(device).is_cuda:
        raise ValueError("Supported robot recipes require CUDA production graph")
    validate_inputs()
    case = next(
        c
        for c in cases()
        if c["scene"] == names[name]
        and c["precision"] == "fp64"
        and c["law"] == "C-IE"
        and c["globalization"] == "ArmijoOrResNorm"
    )
    with wp.ScopedDevice(device):
        return build(case)


def set_control(control, samples, step):
    """Apply the original endpoint-sampled robot controls."""
    if samples is not None:
        control.joint_target_q.assign(samples[step][0])
        control.joint_target_qd.assign(samples[step][1])


def compare_arrays(actual, expected, *, rtol=2e-5, atol=2e-6):
    """Use existing small q+x tolerances; require exact discrete metadata."""
    if actual.keys() != expected.keys():
        raise ValueError("Snapshot array keys differ")
    errors = {}
    for name, value in expected.items():
        got = actual[name]
        if got.shape != value.shape or got.dtype != value.dtype:
            raise ValueError(f"Snapshot shape/dtype differs: {name}")
        if value.dtype.kind in "fc":
            if not np.isfinite(got).all() or not np.isfinite(value).all():
                raise ValueError(f"Nonfinite snapshot: {name}")
            np.testing.assert_allclose(got, value, rtol=rtol, atol=atol, err_msg=name)
            errors[name] = float(np.max(np.abs(got - value), initial=0))
        else:
            np.testing.assert_array_equal(got, value, err_msg=name)
    return errors
