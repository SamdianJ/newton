# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Point-cloud branch adapter for the shared cloth-contact experiments."""

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic


def create_solver(model, *, cloth_inputs, solver_kwargs, method_config):
    """Construct the unified solver with this branch's point-cloud configuration."""
    return SolverMonolithic(model, cloth_inputs=cloth_inputs, cloth_contact=method_config, **solver_kwargs)


def diagnostics(solver, state):
    """Export publication-bound cloth forces and separate inter/self loads [N]."""
    result = solver.get_cloth_contact_diagnostics(state)
    if result.get("enabled"):
        result["particle_force_N"] = result["particle_force_N"].tolist()
        for field in ("candidates", "active", "compressive_load_N"):
            result[field] = result["inter"][field] + result["self"][field]
        result["unavailable"] = {"continuous_path": "DCD has no continuous triangle path test"}
    return result


def evaluate_static(model, cloth_inputs, config, positions, *, precision, device, solver_kwargs=None):
    """Evaluate normal forces [N] without stepping or changing fixed-point state."""
    if wp.get_device(device) != model.device:
        raise ValueError("Static contact model and requested device differ")
    options = dict(solver_kwargs or {})
    options.update(precision=SolverMonolithic.Precision(precision), friction_coefficient=0.0, use_cuda_graph=False)
    options.setdefault("friction_velocity_scale", 0.1)
    solver = create_solver(model, cloth_inputs=cloth_inputs, solver_kwargs=options, method_config=config)
    contact = solver._cloth_contact
    points = np.asarray(positions)
    if points.shape != (model.particle_count, 3) or not np.isfinite(points).all():
        raise ValueError("Static contact positions must be finite particle coordinates")
    contact.start.assign(points)
    contact.refresh(contact.start)
    zero = wp.zeros(model.particle_count, dtype=contact.vector, device=device)
    contact.evaluate(zero, 1.0)
    if int(contact.pool.candidates.overflow.numpy()[0]) or int(contact.pool.records.dropped.numpy()[0]):
        raise ValueError("Static cloth contact capacity exceeded")
    if int(contact.flags.numpy()[0]) or int(contact.pool.records.flags.numpy()[0]):
        raise ValueError("Invalid static cloth contact geometry")
    summary = contact.summary.numpy()
    return {
        "method": "point-cloud",
        "normal_energy_J": float(contact.energy.numpy()[0]),
        "particle_force_N": contact.pool.force.result.numpy().reshape(-1, 3).copy(),
        "inter": {"candidates": int(summary[0]), "active": int(summary[1]), "compressive_load_N": float(summary[2])},
        "self": {"candidates": int(summary[3]), "active": int(summary[4]), "compressive_load_N": float(summary[5])},
        "configuration": dict(contact.config),
    }
