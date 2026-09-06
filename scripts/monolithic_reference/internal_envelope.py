# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent homogeneous-tet static oracle and pinned release envelope.

This oracle applies only to the declared centered, anchored single tetrahedron
against a horizontal plane. It uses float64 Kim stress and rest-area penalty,
without reading the production trajectory or contact records to set tolerances.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import newton

DIRECTORY = Path(__file__).with_name("fixtures")
FROZEN_FIXTURE = DIRECTORY / "normal_loading_frozen_v3.json"
ENVELOPE = DIRECTORY / "internal_normal_envelope_v2.json"
ENVELOPE_SHA256 = "0ebffba07e670e1b39a9cd579543df460770d4c4d4fc10e51fdcd313988616c6"


def physics_sha256(fixture):
    """Bind all simulated parameters, control, acceptance and upstream evidence."""
    keys = (
        "dt_s",
        "substeps",
        "gravity_m_s2",
        "rigid",
        "soft",
        "contact",
        "drive",
        "acceptance",
        "calibration_status",
        "calibration_provenance",
        "support_status",
        "support_provenance",
    )
    encoded = json.dumps({key: fixture[key] for key in keys}, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def static_curve(fixture, *, count=401):
    """Solve Pxx=Pyy=0 and vertical equilibrium along homogeneous compression."""
    soft = fixture["soft"]
    rest = np.asarray(soft["rest_positions_m"], dtype=np.float64)
    height = rest[3, 2] - rest[0, 2]
    area = np.linalg.norm(np.cross(rest[1] - rest[0], rest[2] - rest[0])) / 2
    mu, lame = soft["mu_pa"], soft["lambda_pa"] + soft["mu_pa"]
    result = []
    for c in np.linspace(1, 0.70, count):
        a2 = ((lame + mu) * c - mu) / (lame * c * c)
        force = area / 3 * mu * (a2 / c - c)
        penetration = force / (area * fixture["contact"]["stiffness_n_m3"])
        result.append(
            {
                "closure_m": float(height * (1 - c) + penetration),
                "force_n": float(force),
                "lateral_stretch": float(np.sqrt(a2)),
                "axial_stretch": float(c),
                "penetration_m": float(penetration),
            }
        )
    return result


def impact_budget(fixture, curve):
    """Linearize the initial normal impact with actual BE/frozen-PD timing.

    This is a declared two-mode calibration model, not a rigorous bound for
    arbitrary nonlinear assets. Modal absolute sums bound its initial phase.
    """
    rest = np.asarray(fixture["soft"]["rest_positions_m"])
    area = np.linalg.norm(np.cross(rest[1] - rest[0], rest[2] - rest[0])) / 2
    volume = abs(np.linalg.det((rest[1:] - rest[0]).T)) / 6
    masses = np.asarray([fixture["rigid"]["mass_kg"], 0.75 * volume * fixture["soft"]["density_kg_m3"]])
    mass = np.diag(masses)
    contact = area * fixture["contact"]["stiffness_n_m3"]
    slope = (curve[1]["force_n"] - curve[0]["force_n"]) / (curve[1]["closure_m"] - curve[0]["closure_m"])
    soft = 1 / (1 / slope - 1 / contact)
    stiffness = np.asarray([[contact, -contact], [-contact, contact + soft]])
    drive = fixture["drive"]
    dt = fixture["dt_s"]
    control_stiffness = np.diag([drive["kp_n_m"], 0])
    damping = np.diag([drive["kd_n_s_m"], 0])
    vx = -dt * np.linalg.solve(mass + dt * dt * stiffness, stiffness + control_stiffness)
    vv = np.linalg.solve(mass + dt * dt * stiffness, mass - dt * damping)
    transition = np.block([[np.eye(2) + dt * vx, dt * vv], [vx, vv]])
    eigenvalues, eigenvectors = np.linalg.eig(transition)
    vmax = 1.875 * (drive["loading_target_m"] - drive["free_target_m"]) / (drive["loading_end_s"] - drive["free_end_s"])
    # First force-producing record may be up to one timestep past activation.
    # Add independent initial position and velocity absolute modal sums.
    initial = np.diag([vmax * dt, 0, vmax, 0])
    amplitudes = np.abs(np.asarray([contact, -contact, 0, 0]) @ eigenvectors) * np.sum(
        np.abs(np.linalg.solve(eigenvectors, initial)), axis=1
    )
    radii = np.abs(eigenvalues)
    if not np.all(radii < 1):
        raise ValueError("Declared BE/PD impact linearization is unstable")
    return {
        "method": "two-DOF initial normal linearization; contact/soft BE and explicit frozen PD; absolute modal phase sum",
        "scope": "initial small-strain base-face impact of this exact fixture; not a nonlinear theorem",
        "anchor": "n=0 at first force-producing record in the sustained onset streak",
        "dirichlet_mass_rule": "three dynamic vertices each rho*V0/4; fixed apex removed",
        "masses_kg": masses.tolist(),
        "contact_stiffness_n_m": float(contact),
        "soft_static_tangent_n_m": float(soft),
        "transition": transition.tolist(),
        "initial_velocity_bound_m_s": vmax,
        "initial_penetration_bound_m": vmax * dt,
        "modal_amplitudes_n": amplitudes.tolist(),
        "modal_decay": radii.tolist(),
    }


def make_envelope(fixture):
    """Declare static bounds and a priori spatial/inertial/sampling budgets."""
    curve = static_curve(fixture)
    x = np.asarray([p["closure_m"] for p in curve])
    f = np.asarray([p["force_n"] for p in curve])
    rigid, soft, drive = (fixture[key] for key in ("rigid", "soft", "drive"))
    clearance = (
        soft["rest_positions_m"][0][2]
        - rigid["initial_plane_z_m"]
        - soft["particle_radius_m"]
        - fixture["contact"]["shape_margin_m"]
    )
    target = drive["loading_target_m"] - clearance
    end = float(np.interp(target, x + f / drive["kp_n_m"], x))
    end_force = float(np.interp(end, x, f))
    duration = drive["loading_end_s"] - drive["free_end_s"]
    amplitude = drive["loading_target_m"] - drive["free_target_m"]
    vmax = 1.875 * amplitude / duration
    amax = (10 / np.sqrt(3)) * amplitude / duration**2
    rest = np.asarray(soft["rest_positions_m"])
    mass = soft["density_kg_m3"] * abs(np.linalg.det((rest[1:] - rest[0]).T)) / 6
    shift = 2 * fixture["dt_s"] * vmax
    slope = float(np.max(np.diff(f) / np.diff(x)))
    inertia = (rigid["mass_kg"] + mass) * amax
    absolute = slope * shift + inertia + fixture["acceptance"]["force_floor_n"]
    # The existing uniform-plane C4 gate supplies the spatial budget. The
    # inertial budget bounds imposed quintic acceleration at the total mass.
    relative = 0.05
    low, high = fixture["acceptance"]["closure_secant_interval_m"]
    secant = float((np.interp(high, x, f) - np.interp(low, x, f)) / (high - low))
    work_x = np.append(x[x < end], end)
    work = float(np.trapezoid(np.interp(work_x, x, f), work_x))

    def bounds(value, budget):
        return [max(0.0, value - budget), value + budget]

    onset_relative = (
        soft["rest_positions_m"][0][2]
        - soft["particle_radius_m"]
        - fixture["contact"]["shape_margin_m"]
        - soft["rest_positions_m"][3][2]
        + rigid["reference_point_body_m"][2]
    )
    return {
        "schema_version": "internal_normal_envelope/v2",
        "impact_budget": impact_budget(fixture, curve),
        "status": "FROZEN",
        "physics_sha256": physics_sha256(fixture),
        "calibration_provenance": fixture["calibration_provenance"],
        "support_provenance": fixture["support_provenance"],
        "method": "float64 homogeneous Kim Pxx=Pyy=0; -A0*Pzz/3=A0*kn*penetration; q_target=q+force/kp",
        "budget_policy": {
            "relative_spatial": relative,
            "closure_sampling_m": shift,
            "quintic_max_velocity_m_s": vmax,
            "quintic_max_acceleration_m_s2": float(amax),
            "total_mass_kg": float(mass + rigid["mass_kg"]),
            "inertial_force_n": float(inertia),
            "maximum_static_slope_n_m": slope,
            "force_absolute_n": float(absolute),
            "provenance": "C4 plane 5 percent plus 2 dt*vmax axis uncertainty plus total_mass*amax; no dynamic trajectory fit",
        },
        "metrics": {
            "contact_onset_relative_m": [onset_relative - shift, onset_relative + shift],
            "secant_stiffness_n_m": bounds(secant, relative * secant + 2 * absolute / (high - low)),
            "peak_force_n": bounds(end_force, relative * end_force + absolute),
            "settled_force_n": bounds(end_force, relative * end_force + absolute),
            "curve_work_j": bounds(work, relative * work + absolute * end + shift * end_force),
        },
        "static_endpoint_closure_m": end,
        "static_curve": curve,
        "curve_interval_m": [0.0, end - shift],
    }


def load_envelope(fixture):
    """Reject a forged envelope or inherited freeze on different physics."""
    data = ENVELOPE.read_bytes()
    if hashlib.sha256(data).hexdigest() != ENVELOPE_SHA256:
        raise ValueError("Internal envelope content hash does not match its release pin")
    envelope = json.loads(data)
    if (
        fixture.get("internal_envelope_sha256") != ENVELOPE_SHA256
        or physics_sha256(fixture) != envelope["physics_sha256"]
    ):
        raise ValueError("Frozen internal envelope does not match fixture physics/acceptance/provenance")
    return envelope


def assess_envelope(records, summary, fixture):
    """Check every scalar and the whole loading curve, independently of exit labels."""
    if fixture.get("status") != "FROZEN":
        return {"pass": False, "gates": {"frozen_internal_envelope": False}, "observations": {}}
    envelope = load_envelope(fixture)
    observations = {name: summary.get(name) for name in envelope["metrics"]}
    gates = {
        name: isinstance(value, (float, int))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and envelope["metrics"][name][0] <= value <= envelope["metrics"][name][1]
        for name, value in observations.items()
    }
    origin = summary.get("contact_onset_step")
    curve = envelope["static_curve"]
    x = np.asarray([p["closure_m"] for p in curve])
    f = np.asarray([p["force_n"] for p in curve])
    budget = envelope["budget_policy"]
    samples = records[origin:] if type(origin) is int and 0 <= origin < len(records) else []
    valid = bool(samples)
    reached = 0.0
    for sample_index, record in enumerate(samples):
        closure = record.get("actual_closure_m")
        force = record.get("normal_compressive_force_n")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (closure, force)):
            valid = False
            continue
        reached = max(reached, closure)
        expected = float(np.interp(closure, x, f))
        impact = envelope["impact_budget"]
        transient = sum(
            amplitude * decay**sample_index
            for amplitude, decay in zip(impact["modal_amplitudes_n"], impact["modal_decay"], strict=True)
        )
        valid &= (
            0 <= closure <= x[-1]
            and abs(force - expected) <= budget["relative_spatial"] * expected + budget["force_absolute_n"] + transient
        )
    gates["force_displacement_curve"] = bool(valid and reached >= envelope["curve_interval_m"][1])
    return {"pass": all(gates.values()), "gates": gates, "observations": observations}


def verify_static_cpu(fixture):
    """Check prescribed analytic equilibria through actual Newton CPU stepping."""
    from .normal_loading import build_scene  # noqa: PLC0415 - avoid the runner/oracle import cycle

    records = []
    rest = np.asarray(fixture["soft"]["rest_positions_m"], dtype=float)
    clearance = (
        rest[0, 2]
        - fixture["rigid"]["initial_plane_z_m"]
        - fixture["soft"]["particle_radius_m"]
        - fixture["contact"]["shape_margin_m"]
    )
    for point in static_curve(fixture, count=13):
        model, state, control, solver = build_scene(fixture, device="cpu")
        positions = rest.copy()
        positions[:3, :2] *= point["lateral_stretch"]
        positions[:3, 2] = rest[3, 2] + point["axial_stretch"] * (rest[:3, 2] - rest[3, 2])
        state.particle_q.assign(positions.astype(np.float32))
        state.joint_q.assign(np.asarray([clearance + point["closure_m"]], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        before = state.particle_q.numpy().copy()
        control.joint_f.assign(np.asarray([point["force_n"]], dtype=np.float32))
        solver.step(state, state, control, None, fixture["dt_s"])
        count = int(solver.contacts.soft_contact_count.numpy()[0])
        force = -float(solver.contacts.force.numpy()[:count, 2].sum())
        drift = float(np.max(np.abs(state.particle_q.numpy() - before)))
        record = {
            **point,
            "actual_force_n": force,
            "force_error_n": abs(force - point["force_n"]),
            "position_drift_m": drift,
            "active_samples": solver.last_stats.active_sample_count,
            "converged": solver.last_stats.converged,
            "rolled_back": solver.last_stats.rolled_back,
        }
        record["pass"] = bool(
            record["converged"]
            and not record["rolled_back"]
            and drift < 1e-6
            and record["force_error_n"] < 1e-4
            and record["active_samples"] in (0, 3)
        )
        records.append(record)
    return {
        "method": "13 independent prescribed analytical static states; actual Newton CPU step with matching frozen force",
        "records": records,
        "pass": all(r["pass"] for r in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-cpu", action="store_true")
    args = parser.parse_args()
    from .normal_loading import load_fixture  # noqa: PLC0415 - CLI-only runner import

    evidence = verify_static_cpu(load_fixture()) if args.verify_cpu else make_envelope(load_fixture())
    with args.output.open("x") as stream:
        json.dump(evidence, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
