# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen Close force balance, defined energy terms, and physical increment audit."""

import json
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.contact import evaluate_monolithic_p1q3_contacts
from newton._src.solvers.monolithic.tet import _prepare_evaluation


def smith_energy(x, indices, poses, materials):
    points = x[indices]
    f = np.swapaxes(points[:, 1:] - points[:, :1], 1, 2) @ poses
    ic = np.sum(f * f, axis=(1, 2))
    d = np.linalg.det(f) - 1
    mu, lam = materials[:, 0], materials[:, 1]
    return float(
        np.sum(
            (0.5 * (4 * mu / 3) * (ic - 3 - np.log((ic + 1) / 4)) + 0.5 * (lam + 5 * mu / 6) * d * d - mu * d)
            / (6 * np.linalg.det(poses))
        )
    )


def transform_points(poses, points):
    xyz, rotation, w = poses[..., :3], poses[..., 3:6], poses[..., 6:7]
    return xyz + points + 2 * np.cross(rotation, np.cross(rotation, points) + w * points)


def probe_close_balance(case, output):
    solver, model = case.solver, case.model
    accepted, trial = solver._transaction.accepted, solver._transaction.trial
    dt, nq = case.fixture["dt"], solver._layout.q_dof_count
    z, delta = accepted.z.numpy().astype(float), solver._delta.numpy().astype(float)
    scale = solver._linear.scale.numpy().astype(float)
    physics = solver._tet_physics
    if physics.mass_mode != "consistent" or physics.material_model != "smith_log_stabilized":
        raise ValueError("This Close audit requires Smith with consistent mass")
    indices = model.tet_indices.numpy()
    poses, materials = physics.poses.numpy().astype(float), physics.materials.numpy().astype(float)
    dynamic = solver._layout.dynamic_particle_ids.numpy()
    x_n = solver._transaction._original.particle_q.numpy().astype(float)
    v_n = solver._transaction._original.particle_qd.numpy().astype(float)
    row_mass = physics.row_mass.numpy().astype(float)
    mass = np.zeros((model.particle_count, model.particle_count))
    coefficients = physics.coefficients.numpy().astype(float)
    for a in range(4):
        for b in range(4):
            np.add.at(mass, (indices[:, a], indices[:, b]), coefficients * (2 if a == b else 1))
    worlds = model.particle_world.numpy()
    gravity = model.gravity.numpy().astype(float)[worlds]
    frozen_force = solver._transaction.inputs.particle_f.numpy().astype(float)
    x_external = -row_mass[:, None] * gravity - frozen_force
    n = len(z)
    direction = wp.zeros_like(solver._delta)
    history = {
        name: getattr(solver._contact.committed, name).numpy().copy() for name in ("valid", "xi_local", "normal_local")
    }
    original_state = solver._transaction._original.particle_q.numpy().copy()
    original_launch = wp.launch

    def normal_only_launch(kernel, *args, **kwargs):
        if kernel is evaluate_monolithic_p1q3_contacts:
            inputs = kwargs.get("inputs", args[1] if len(args) > 1 else None)
            context = inputs[16]
            mu = context.mu
            context.mu = 0.0
            try:
                return original_launch(kernel, *args, **kwargs)
            finally:
                context.mu = mu
        return original_launch(kernel, *args, **kwargs)

    def vector(q=None, x=None):
        result = np.zeros(n)
        if q is not None:
            result[:nq] = q
        if x is not None:
            result[nq:] = x.reshape(-1)
        return result

    def evaluate(offset):
        direction.assign(np.asarray(offset, dtype=np.float32))
        trial.form_trial(accepted, direction, 1.0, dt)
        value = solver._evaluate_trial(trial, dt)
        residual = trial.residual.numpy().astype(float)
        x = trial.state.particle_q.numpy().astype(float)
        tet_residual = solver._particle_residual.numpy().astype(float)
        art = solver._articulation
        parts = {
            "q_inertia": vector(q=art.M.numpy()[0].astype(float) @ trial.qdd.numpy().astype(float)),
            "q_coriolis": vector(q=art.C.numpy().astype(float)),
            "q_gravity": vector(q=art.g.numpy().astype(float)),
            "q_external": vector(
                q=-solver._transaction.inputs.joint_f.numpy().astype(float)
                - art.generalized_body_force.numpy().astype(float)
            ),
            "contact": solver._contact._residual[1].numpy().astype(float),
        }
        if solver._joint_terms is not None:
            for i, name in enumerate(("q_pd", "q_limit", "q_friction")):
                parts[name] = vector(q=solver._joint_terms.residual.numpy()[:, i].astype(float))
        history_samples = solver._contact._samples[1].numpy().astype(float)
        contacts = solver._trial_contacts
        count = int(contacts.soft_contact_count.numpy()[0])
        ids = contacts.soft_contact_indices[:count].numpy()
        bary = contacts.soft_contact_barycentric[:count].numpy().astype(float)
        shapes = contacts.soft_contact_shape[:count].numpy()
        bodies = model.shape_body.numpy()[shapes]
        body_q = trial.state.body_q.numpy().astype(float)
        xs = np.einsum("ij,ijk->ik", bary, x[ids])
        xr = transform_points(body_q[bodies], contacts.soft_contact_body_pos[:count].numpy().astype(float))
        normals = contacts.soft_contact_normal[:count].numpy().astype(float)
        gap = (
            np.sum(normals * (xs - xr), axis=1) - solver.collision_pipeline.r_soft - model.shape_margin.numpy()[shapes]
        )
        rest = model.particle_q.numpy().astype(float)[ids]
        area = 0.5 * np.linalg.norm(np.cross(rest[:, 1] - rest[:, 0], rest[:, 2] - rest[:, 0]), axis=1)
        weight = solver._contact.contact_stiffness * area / 3
        u, eps = -gap, solver._contact.normal_smoothing_width
        p = np.maximum(u, 0)
        if eps > 0:
            inside = np.abs(u) < eps
            t = (u[inside] + eps) / (2 * eps)
            p[inside] = eps * (2 * t**3 - t**4)
        inertia_displacement = x - x_n - dt * v_n
        energies = {
            "smith": smith_energy(x, indices, poses, materials),
            "x_incremental_inertia": float(0.5 * np.sum(inertia_displacement * (mass @ inertia_displacement)) / dt**2),
            "x_external": float(np.sum(x_external * (x - x_n))),
            "normal_gap_potential": float(np.sum(0.5 * weight * p * p)),
            # Local Huber energy equals retained elastic energy + radial return
            # dissipation. Varying cap/frame/history prevents treating its sum
            # as a potential whose full gradient is the implemented force.
            "contact_huber_local": float(np.sum(history_samples[:, 6] + history_samples[:, 8])),
        }
        body_com = transform_points(body_q, model.body_com.numpy().astype(float))
        body_gravity = model.gravity.numpy().astype(float)[model.body_world.numpy()]
        energies["q_gravity"] = float(-np.sum(model.body_mass.numpy()[:, None] * body_gravity * body_com))
        if solver._joint_terms is not None:
            terms = solver._joint_terms
            params = terms.params.numpy().astype(float)
            q, velocity = trial.state.joint_q.numpy().astype(float), trial.state.joint_qd.numpy().astype(float)
            target = terms.target_q.numpy().astype(float)[terms.target_map.numpy()]
            v_target = terms.target_qd.numpy().astype(float)
            kp, kd, effort = params[:, 0], params[:, 1], params[:, 2]
            stiffness = kp + kd / dt
            active = stiffness > 0
            raw = kp * (q - target) + kd * (velocity - v_target)
            absolute = np.abs(raw[active])
            cap = effort[active]
            energies["q_pd_incremental"] = float(
                np.sum(np.where(absolute < cap, 0.5 * absolute**2, cap * (absolute - 0.5 * cap)) / stiffness[active])
            )
            energies["q_limit_elastic_only"] = float(
                np.sum(
                    0.5 * params[:, 5] * (np.maximum(q - params[:, 4], 0) ** 2 + np.maximum(params[:, 3] - q, 0) ** 2)
                )
            )
            f, eps = params[:, 8], params[:, 9]
            energies["q_friction_incremental"] = float(np.sum(f * dt * (np.sqrt(velocity**2 + eps**2) - eps)))
        # Isolate normal residual without changing any solver configuration or
        # committed history, then restore the full residual before continuing.
        with patch.object(wp, "launch", normal_only_launch):
            solver._evaluate_trial(trial, dt)
        normal_residual = solver._contact._residual[1].numpy().astype(float)
        parts["contact_normal"] = normal_residual
        parts["contact_friction"] = parts.pop("contact") - normal_residual
        solver._evaluate_trial(trial, dt)
        # Execute the same inertia kernels in scratch, excluding elasticity;
        # reevaluation below restores every scratch field before returning.
        _prepare_evaluation(model, solver._tet_arguments(trial, dt))
        inertial_residual = solver._particle_residual.numpy().astype(float)
        parts["x_external"] = vector(x=x_external[dynamic])
        parts["x_inertia"] = vector(x=inertial_residual - x_external[dynamic])
        parts["smith"] = vector(x=tet_residual - inertial_residual)
        reconstructed = sum(parts.values())
        dz = trial.z.numpy().astype(float) - z
        row = {
            "merit": value.merit,
            "merit_q": value.merit_q,
            "merit_x": value.merit_x,
            "min_det_f": value.min_det_f,
            "reconstruction_scaled_rms": float(np.linalg.norm(scale * (reconstructed - residual)) / np.sqrt(n)),
            "energies_J": energies,
            "component_scaled_l2": {key: float(np.linalg.norm(scale * value)) for key, value in parts.items()},
            "actual_q_translation_max_m": float(abs(dz[0])),
            "actual_q_rotation_max_rad": float(np.max(np.abs(dz[1:nq]))),
            "actual_x_max_m": float(np.max(np.linalg.norm(dz[nq:].reshape(-1, 3), axis=1))),
            "actual_x_velocity_change_max_m_s": float(np.max(np.linalg.norm(dz[nq:].reshape(-1, 3), axis=1)) / dt),
        }
        solver._evaluate_trial(trial, dt)
        return residual, parts, row, dz

    residual, parts, center, _ = evaluate(np.zeros(n))
    repeats = [evaluate(np.zeros(n))[2] for _ in range(3)]
    balance = {}
    for name, section in (("q", slice(0, nq)), ("x", slice(nq, n))):
        norms = {key: float(np.linalg.norm((scale * part)[section])) for key, part in parts.items()}
        total = float(np.linalg.norm((scale * residual)[section]))
        balance[name] = {
            "scaled_component_l2": norms,
            "scaled_residual_l2": total,
            "remaining_fraction_of_component_sum": total / sum(norms.values()),
        }
    rows = []
    arrays = {"z": z, "delta": delta, "scale": scale, "residual": residual, **parts}
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, -0.125):
        r, terms, row, dz = evaluate(alpha * delta)
        row.update(
            alpha=alpha,
            passes_original_line_search=bool(
                alpha > 0 and row["merit"] <= (1 - 1e-4 * alpha) * center["merit"] + solver._config.merit_noise
            ),
            delta_energy_J={key: row["energies_J"][key] - value for key, value in center["energies_J"].items()},
            residual_work_J={key: float(0.5 * (parts[key] + value) @ dz) for key, value in terms.items()},
        )
        rows.append(row)
        arrays[f"residual_{alpha}"] = r
        arrays[f"dz_{alpha}"] = dz
        for key, value in terms.items():
            arrays[f"component_{key}_{alpha}"] = value
    x_residual = residual[nq:].reshape(-1, 3)
    net = x_residual.sum(axis=0)
    cube_mass = float(row_mass.sum())
    result = {
        "schema": "close-balance-v3-components",
        "step": case.step_count + 1,
        "scope": "fixed terminal trial audit; no physical step accepted; energies are separate terms, not a certified total potential",
        "balance": balance,
        "center": center,
        "center_repeats": repeats,
        "trials": rows,
        "initial_residual_work_J": {key: float(value @ delta) for key, value in parts.items()},
        "cube_net_residual_N": net.tolist(),
        "cube_net_residual_fraction_weight": float(np.linalg.norm(net) / (cube_mass * np.linalg.norm(gravity[0]))),
        "cube_COM_inertial_velocity_correction_m_s": (-dt * net / cube_mass).tolist(),
    }
    oracle_path = output / "close-direction-oracle.npz"
    if oracle_path.exists():
        with np.load(oracle_path) as oracle:
            for key in ("correction_1.0", "correction_0.5"):
                correction = oracle[key]
                result[key] = {
                    "translation_m": float(abs(correction[0])),
                    "rotation_max_rad": float(np.max(np.abs(correction[1:nq]))),
                    "x_max_m": float(np.max(np.linalg.norm(correction[nq:].reshape(-1, 3), axis=1))),
                }
    evaluate(np.zeros(n))
    np.testing.assert_array_equal(accepted.z.numpy(), z.astype(np.float32))
    np.testing.assert_array_equal(solver._transaction._original.particle_q.numpy(), original_state)
    for name, before in history.items():
        np.testing.assert_array_equal(getattr(solver._contact.committed, name).numpy(), before)
    result["accepted_and_history_unchanged"] = True
    np.savez_compressed(output / "close-balance.npz", **arrays)
    (output / "close-balance.json").write_text(json.dumps(result, indent=2) + "\n")
    print("Close balance", balance, flush=True)
