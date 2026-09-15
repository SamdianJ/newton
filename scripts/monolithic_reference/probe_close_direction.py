# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Terminal-only residual decomposition and independent finite-difference direction oracle."""

import json
from unittest.mock import patch

import numpy as np
import warp as wp
from threadpoolctl import threadpool_limits

from newton._src.solvers.monolithic.contact import evaluate_monolithic_p1q3_contacts
from newton._src.solvers.monolithic.linear import MonolithicContactFactorKind


def probe_close_direction(case, output):
    from scipy.linalg import lstsq

    solver = case.solver
    accepted, trial = solver._transaction.accepted, solver._transaction.trial
    dt = case.fixture["dt"]
    scale = solver._linear.scale.numpy().astype(float)
    z = accepted.z.numpy().astype(float)
    delta = solver._delta.numpy().astype(float)
    nq = solver._layout.q_dof_count
    direction = wp.zeros_like(solver._delta)
    history = {
        name: getattr(solver._contact.committed, name).numpy().copy() for name in ("valid", "xi_local", "normal_local")
    }
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

    def evaluate(displacement, normal_only=False):
        direction.assign(np.asarray(displacement, dtype=np.float32))
        trial.form_trial(accepted, direction, 1.0, dt)
        if normal_only:
            with patch.object(wp, "launch", normal_only_launch):
                value = solver._evaluate_trial(trial, dt)
        else:
            value = solver._evaluate_trial(trial, dt)
        return (
            trial.residual.numpy().astype(float),
            solver._contact._residual[1].numpy().astype(float),
            trial.z.numpy().astype(float),
            value,
        )

    center, _, _, base = evaluate(np.zeros_like(z))
    rhat = scale * center
    arrays = {"z": z, "scale": scale, "residual": center, "pcg_delta": delta}
    contacts = solver._trial_contacts
    contact_count = int(contacts.soft_contact_count.numpy()[0])
    record_keys = solver._contact._record_candidate_tid[:contact_count].numpy()
    normals = contacts.soft_contact_normal[:contact_count].numpy().astype(float)
    samples = solver._contact._samples[1].numpy().astype(float)
    loads = samples[record_keys, 0]
    tangent_force = solver._contact._forces[1][:contact_count].numpy().astype(float) + loads[:, None] * normals
    ids = contacts.soft_contact_indices[:contact_count].numpy()
    bary = contacts.soft_contact_barycentric[:contact_count].numpy().astype(float)
    bodies = case.model.shape_body.numpy()[contacts.soft_contact_shape[:contact_count].numpy()]
    positions = trial.state.particle_q.numpy().astype(float)
    poses = trial.state.body_q.numpy().astype(float)
    local_com = case.model.body_com.numpy().astype(float)
    particle_map = solver._linear.particle_to_dynamic.numpy()
    articulation = solver._articulation
    spatial = articulation.scratch.J.numpy().astype(float)
    actor = articulation.body_to_articulation.numpy()
    link = articulation.body_to_link_index.numpy()
    # Adjoint contraction of the frozen force-to-residual map at each sample.
    weighted = scale * rhat
    adjoints = np.zeros_like(tangent_force)
    for record, body in enumerate(bodies):
        point = bary[record] @ positions[ids[record]]
        rotation, w = poses[body, 3:6], poses[body, 6]
        com = (
            poses[body, :3]
            + local_com[body]
            + 2 * np.cross(rotation, np.cross(rotation, local_com[body]) + w * local_com[body])
        )
        block = spatial[actor[body], 6 * link[body] : 6 * link[body] + 6, :nq]
        point_jacobian = block[:3] + np.cross(block[3:].T, point - com).T
        adjoints[record] = -point_jacobian @ weighted[:nq]
        for corner, particle in enumerate(ids[record]):
            dynamic = particle_map[particle]
            if dynamic >= 0:
                adjoints[record] += bary[record, corner] * weighted[nq + 3 * dynamic : nq + 3 * dynamic + 3]
    sliding = (samples[record_keys, 3] == 2) & (loads > 0)
    load_coefficients = np.zeros(contact_count)
    load_coefficients[sliding] = np.sum(adjoints[sliding] * tangent_force[sliding], axis=1) / loads[sliding]
    factors = solver._linear._factors
    count = int(factors.count.numpy()[0])
    normal_rows = factors.kind[:count].numpy() == int(MonolithicContactFactorKind.NORMAL)
    keys = factors.candidate_tid[:count].numpy()[normal_rows]
    g = np.zeros((len(keys), len(z)))
    g[:, :nq] = factors.gq[:count].numpy()[normal_rows]
    columns = factors.gx_columns[:count].numpy()[normal_rows]
    values = factors.gx_values[:count].numpy()[normal_rows]
    for row in range(len(keys)):
        valid = columns[row] >= 0
        np.add.at(g[row], nq + columns[row, valid], values[row, valid])
    weights = factors.weights[:count].numpy()[normal_rows].astype(float)
    arrays.update(normal_G=g, normal_keys=keys, normal_weights=weights)
    decomposition = []
    for h in (0.25, 0.125, 0.0625):
        plus, cp, _, _ = evaluate(h * delta)
        samples_plus = solver._contact._samples[1].numpy()[:, 0].astype(float)
        normal_plus = samples_plus[keys]
        minus, cm, _, _ = evaluate(-h * delta)
        samples_minus = solver._contact._samples[1].numpy()[:, 0].astype(float)
        normal_minus = samples_minus[keys]
        _, np_, _, _ = evaluate(h * delta, True)
        _, nm, _, _ = evaluate(-h * delta, True)
        pieces = {
            "total": (plus - minus) / (2 * h),
            "normal": (np_ - nm) / (2 * h),
            "tangent": (cp - cm - np_ + nm) / (2 * h),
            "noncontact": (plus - minus - cp + cm) / (2 * h),
        }
        pieces["normal_load_at_fixed_G"] = -g.T @ ((normal_plus - normal_minus) / (2 * h))
        pieces["normal_frame_and_mapping"] = pieces["normal"] - pieces["normal_load_at_fixed_G"]
        pieces["normal_assembled"] = g.T @ (weights * (g @ delta))
        tangent_load = float(
            load_coefficients @ ((samples_plus[record_keys] - samples_minus[record_keys]) / (2 * h)) / len(z)
        )
        decomposition.append(
            {
                "h": h,
                "normal_active_keys_unchanged": bool(
                    np.array_equal(np.flatnonzero(samples_plus > 0), np.sort(keys))
                    and np.array_equal(np.flatnonzero(samples_minus > 0), np.sort(keys))
                ),
                **{key: float(rhat @ (scale * v) / len(z)) for key, v in pieces.items()},
                "tangent_load_at_fixed_frame": tangent_load,
                "tangent_remaining": float(rhat @ (scale * pieces["tangent"]) / len(z)) - tangent_load,
                "normal_frame_and_mapping_q": float(
                    rhat[:nq] @ (scale * pieces["normal_frame_and_mapping"])[:nq] / len(z)
                ),
                "normal_frame_and_mapping_x": float(
                    rhat[nq:] @ (scale * pieces["normal_frame_and_mapping"])[nq:] / len(z)
                ),
            }
        )
    print("Close terminal decomposition:", decomposition, flush=True)
    (output / "close-decomposition.json").write_text(json.dumps(decomposition, indent=2) + "\n")

    # Independent research oracle only: no replacement of the production PCG matrix,
    # no accepted trial, and no altered contact law in direction validation.
    sweeps = []
    for multiplier in (1.0, 0.5):
        jacobian = np.empty((len(z), len(z)))
        for column in range(len(z)):
            offset = np.zeros_like(z)
            offset[column] = multiplier * (2e-4 if column < nq else 2e-6)
            rp, _, zp, _ = evaluate(offset)
            rm, _, zm, _ = evaluate(-offset)
            width = zp[column] - zm[column]
            if width <= 0:
                raise ValueError("FD increment is not representable")
            jacobian[:, column] = scale * (rp - rm) / width * scale[column]
            if column % 200 == 0:
                print("Close FD oracle", multiplier, column, len(z), flush=True)
        with threadpool_limits(limits=1):
            y, _, rank, _singular = lstsq(jacobian, -rhat, cond=1e-10)
            eigenvalues = np.linalg.eigvalsh(0.5 * (jacobian + jacobian.T))
        correction = scale * y
        rows = []
        for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
            _, _, _, result = evaluate(alpha * correction)
            rows.append(
                {
                    "alpha": alpha,
                    "merit": result.merit,
                    "merit_q": result.merit_q,
                    "merit_x": result.merit_x,
                    "min_det_f": result.min_det_f,
                    "passes_original_line_search": bool(
                        result.merit <= (1 - 1e-4 * alpha) * base.merit + solver._config.merit_noise
                    ),
                }
            )
        arrays[f"jacobian_{multiplier}"] = jacobian
        arrays[f"correction_{multiplier}"] = correction
        sweeps.append(
            {
                "fd_multiplier": multiplier,
                "rank": int(rank),
                "symmetric_min_eigenvalue": float(eigenvalues[0]),
                "symmetric_negative_eigenvalues": int(np.count_nonzero(eigenvalues < 0)),
                "relative_asymmetry": float(np.linalg.norm(jacobian - jacobian.T) / np.linalg.norm(jacobian)),
                "linear_relative_residual": float(np.linalg.norm(jacobian @ y + rhat) / np.linalg.norm(rhat)),
                "sweep": rows,
            }
        )
    for name, before in history.items():
        np.testing.assert_array_equal(getattr(solver._contact.committed, name).numpy(), before)
    np.testing.assert_array_equal(accepted.z.numpy(), z.astype(np.float32))
    evaluate(np.zeros_like(z))
    np.savez_compressed(output / "close-direction-oracle.npz", **arrays)
    (output / "close-direction-oracle.json").write_text(
        json.dumps(
            {
                "step": case.step_count + 1,
                "scope": "terminal FD oracle, no accepted step; GPU performance not measured",
                "base_merit": base.merit,
                "decomposition": decomposition,
                "directions": sweeps,
                "committed_history_and_accepted_coordinates_unchanged": True,
            },
            indent=2,
        )
        + "\n"
    )
