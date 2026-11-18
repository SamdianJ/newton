# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""P9d PR3 small integration: the floating root's 6D pose drive, measured offline in FP64.

    python -m scripts.monolithic_reference.floating_drive_integration --device cuda:0 --precision fp32 \\
        --output result.json

Scenes:

- ``allegro-hold``: the floating Allegro hand (FREE root, fingers held at their start pose by their
  joint PD, self-collision off, no ground) hangs in gravity from its root drive (the frozen E1a
  root gains) for 200 steps at a fixed root target. Reported: root COM and rotation tracking errors
  against the static deflections M g / k_t and |tau_g| / k_r, where tau_g is gravity's torque about the
  root COM.
- ``chain-120``: the E2 capsule chain (undamped revolute springs) in zero gravity, its root driven back
  from 120 degrees (1000 steps). Reported: the root's rotation error every 100 steps.

Every step must publish; Newton and PCG iteration counts are recorded (R2).
"""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import warp as wp

import newton
from scripts.monolithic_reference import floating_hand_scenes as fhs
from scripts.monolithic_reference.floating_hand_criteria import quaternion_matrix

FIXTURES = Path(__file__).resolve().parent / "fixtures"
G = 9.81


def _drive(builder, joint, linear_ke, linear_kd, angular_ke, angular_kd, force, torque):
    start = builder.joint_qd_start[joint]
    for k in range(6):
        linear = k < 3
        builder.joint_target_ke[start + k] = linear_ke if linear else angular_ke
        builder.joint_target_kd[start + k] = linear_kd if linear else angular_kd
        builder.joint_effort_limit[start + k] = force if linear else torque
        builder.joint_target_mode[start + k] = int(newton.JointTargetMode.POSITION_VELOCITY)


def allegro_hold():
    recipe = json.loads((FIXTURES / "floating_hand_e1a_capsule_v1.json").read_text())
    pd = recipe["root_pd"]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -G))
    builder.default_shape_cfg.margin = 0.0
    palm = wp.transform((0.0, 0.0, 0.3), wp.quat_rpy(0.0, np.pi / 2, 0.0))
    layout = fhs.add_allegro(builder, mount="floating", palm_pose=palm, self_collisions=False)
    _drive(
        builder,
        layout["root_joint"],
        pd["linear_ke"],
        pd["linear_kd"],
        pd["angular_ke"],
        pd["angular_kd"],
        pd["force_limit_N"],
        pd["torque_limit_Nm"],
    )
    return builder, layout["root_joint"], pd, None, 200, recipe["joint_terms"].get("limit_width")


def chain_120():
    recipe = json.loads((FIXTURES / "floating_chain_e2_v1.json").read_text())
    spec = dict(recipe["chain"])
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    fhs.add_chain(builder, spec)
    # About critical damping for the chain (mass about 0.06 kg, inertia about 1e-4 kg m^2).
    pd = {
        "linear_ke": 10.0,
        "linear_kd": 1.5,
        "angular_ke": 0.05,
        "angular_kd": 6e-3,
        "force_limit_N": 100.0,
        "torque_limit_Nm": 10.0,
    }
    _drive(
        builder,
        0,
        *(pd[n] for n in ("linear_ke", "linear_kd", "angular_ke", "angular_kd", "force_limit_N", "torque_limit_Nm")),
    )
    return builder, 0, pd, np.radians(120.0) * np.array([0.0, 0.6, 0.8]), 1000, None


def _rotation(vector):
    angle = np.linalg.norm(vector)
    if angle == 0:
        return np.eye(3)
    k = np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]]) / angle
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def _quaternion(matrix):
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(matrix).as_quat()
    return q if q[3] >= 0 else -q


def _widths(model, width):
    """Per-DoF limit widths over the articulation DoFs (a floating root's six included)."""
    if width is None:
        return None
    return np.full(fhs.articulation_dof_count(model), width)


def run(name, builder, root, pd, turn, steps, device, precision, limit_width=None):
    from newton.solvers.experimental.monolithic import SolverMonolithic  # noqa: PLC0415

    model = builder.finalize(device=device)
    cuda = wp.get_device(device).is_cuda
    solver = SolverMonolithic(
        model,
        friction_velocity_scale=0.01,
        contact_stiffness=1e9,
        friction_coefficient=0.5,
        publish_policy="current_iterate",
        joint_terms=SolverMonolithic.JointTerms(
            implicit_pd=True, limits=limit_width is not None, limit_width=_widths(model, limit_width)
        ),
        precision=precision,
        pcg_mode="production",
        use_cuda_graph=cuda,
    )
    h = 0.001
    state, control = solver.create_state(dt=h), solver.create_control()
    history = state._monolithic_coupled
    start_com = history.floating_com.numpy()[0].astype(np.float64)
    start_rotation = quaternion_matrix(history.floating_rotation.numpy()[0].astype(np.float64))
    # Targets: every coordinate at its start (fingers, hinges); the root possibly turned about its COM.
    targets = control.joint_target_q.numpy()
    q = model.joint_q.numpy().astype(np.float64)
    target_start = model.joint_target_q_start.numpy()
    for joint in range(model.joint_count):
        width = int(model.joint_q_start.numpy()[joint + 1] - model.joint_q_start.numpy()[joint])
        coordinate = int(model.joint_q_start.numpy()[joint])
        targets[target_start[joint] : target_start[joint] + width] = q[coordinate : coordinate + width]
    target_rotation = start_rotation
    if turn is not None:
        parent = quaternion_matrix(model.joint_X_p.numpy()[root][3:].astype(np.float64))
        joint_rotation = parent.T @ _rotation(turn) @ parent @ quaternion_matrix(q[3:7])
        offset = target_start[root]
        targets[offset + 3 : offset + 7] = _quaternion(joint_rotation)
        target_rotation = _rotation(turn) @ start_rotation
        # Turning the joint frame about its own origin moves the COM target; read it from the drive.
    control.joint_target_q.assign(targets)
    counts = {"CONVERGED": 0, "STOPPED_ACCEPTED": 0, "HARD_FAILURE": 0}
    newton_iterations = linear_iterations = 0
    rotation_errors, com_errors = [], []
    for _ in range(steps):
        solver.step(state, state, control, dt=h)
        stats = solver.last_stats
        label = "HARD_FAILURE" if stats.rolled_back else stats.variational["publication_status"]
        counts[label] += 1
        if stats.rolled_back:
            break
        newton_iterations += stats.nonlinear_iterations
        linear_iterations += stats.linear_iterations
        rotation = quaternion_matrix(history.floating_rotation.numpy()[0].astype(np.float64))
        relative = rotation @ target_rotation.T
        rotation_errors.append(float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
        # The drive's own start error is the COM error at the step start (c_n - c_t).
        com_errors.append(float(np.linalg.norm(solver._floating.error_t.numpy()[0])))
    result = {
        "counts": counts,
        "newton_iterations": newton_iterations,
        "linear_iterations": linear_iterations,
        "root_pd": pd,
        "rotation_error_rad_every_100": rotation_errors[::100] + rotation_errors[-1:],
        "final_rotation_error_rad": rotation_errors[-1] if rotation_errors else None,
        "final_com_error_m": com_errors[-1] if com_errors else None,
        "start_com": start_com.tolist(),
    }
    if turn is None:
        # Static deflections: gravity's force on the root drive and its torque about the root COM.
        solver._kinematics.evaluate_state(state)
        mass = model.body_mass.numpy().astype(np.float64)
        com = solver._kinematics.com.numpy().astype(np.float64)
        root_com = history.floating_com.numpy()[0].astype(np.float64)
        torque = sum(m * np.cross(c - root_com, [0.0, 0.0, -G]) for m, c in zip(mass, com, strict=True))
        result.update(
            expected_com_deflection_m=float(mass.sum() * G / pd["linear_ke"]),
            expected_rotation_deflection_rad=float(np.linalg.norm(torque) / pd["angular_ke"]),
            total_mass_kg=float(mass.sum()),
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--precision", choices=("fp64", "fp32"), default="fp64")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[2]
    result = {
        "source_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip(),
        "device": args.device,
        "precision": args.precision,
        "scenes": {},
    }
    for name, factory in (("allegro-hold", allegro_hold), ("chain-120", chain_120)):
        builder, root, pd, turn, steps, width = factory()
        result["scenes"][name] = run(name, builder, root, pd, turn, steps, args.device, args.precision, width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {n: v[n] for n in v if n != "root_pd"} for k, v in result["scenes"].items()}))


if __name__ == "__main__":
    main()
