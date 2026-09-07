# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit the adapted closure with independent float64 URDF forward kinematics."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.examples.softbody.sharpa_close import ClosureTrajectory, load_hand


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def origin(element):
    out = np.eye(4)
    if element is not None:
        out[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        out[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    return out


def fk(root, coordinates):
    children = {j.find("child").get("link") for j in root.findall("joint")}
    poses = {link.get("name"): np.eye(4) for link in root.findall("link") if link.get("name") not in children}
    pending = list(root.findall("joint"))
    while pending:
        ready = [j for j in pending if j.find("parent").get("link") in poses]
        if not ready:
            raise ValueError("Invalid URDF tree")
        for j in ready:
            relative = origin(j.find("origin"))
            if j.get("type") == "revolute":
                relative[:3, :3] = relative[:3, :3] @ rotation(
                    np.fromstring(j.find("axis").get("xyz"), sep=" "), coordinates.get(j.get("name"), 0.0)
                )
            poses[j.find("child").get("link")] = poses[j.find("parent").get("link")] @ relative
            pending.remove(j)
    return poses


def main():
    import matplotlib
    import trimesh

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: PLC0415 - optional plotting dependency

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model, manifest = load_hand(args.asset_dir, device="cpu", parameters=json.loads(Path(args.fixture).read_text()))
    trajectory = ClosureTrajectory(
        args.trajectory,
        manifest["joint_names"],
        model.joint_limit_lower.numpy(),
        model.joint_limit_upper.numpy(),
        model.joint_velocity_limit.numpy(),
    )
    root = ET.parse(args.asset_dir / "left_sharpa_wave.urdf").getroot()
    state = model.state()
    samples = []
    # Both directions exercise every axis, including the two zero-motion thumb channels.
    for i, name in enumerate(trajectory.names):
        for sign in (-1, 1):
            q = np.zeros(22)
            q[i] = sign * 0.05
            samples.append((f"{name}:{sign:+}", q))
    samples += [(label, trajectory.sample(t)[0]) for label, t in (("open", 0), ("middle", 1), ("closed", 4))]
    evidence = []
    for label, q in samples:
        poses = fk(root, dict(zip(trajectory.names, q, strict=True)))
        state.joint_q.assign(q.astype(np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        actual = state.body_q.numpy()
        errors = []
        for i, name in enumerate(model.body_label):
            expected = poses[name.rsplit("/", 1)[-1]]
            matrix = np.asarray(wp.quat_to_matrix(wp.quat(*actual[i, 3:]))).reshape(3, 3)
            errors.append(
                max(np.max(np.abs(expected[:3, 3] - actual[i, :3])), np.max(np.abs(expected[:3, :3] - matrix)))
            )
        tips = {name: pose[:3, 3].tolist() for name, pose in poses.items() if "fingertip" in name}
        evidence.append({"sample": label, "max_fk_error": float(max(errors)), "tip_positions": tips})
        if label in ("open", "middle", "closed"):
            fig = plt.figure(figsize=(9, 7))
            ax = fig.add_subplot(projection="3d")
            for link in root.findall("link"):
                for visual in link.findall("visual"):
                    mesh = visual.find("geometry/mesh")
                    filename = args.asset_dir / mesh.get("filename").split("package://left_sharpa_wave/")[1]
                    geometry = trimesh.load(filename, force="mesh")
                    transform = poses[link.get("name")] @ origin(visual.find("origin"))
                    points = np.asarray(geometry.vertices) @ transform[:3, :3].T + transform[:3, 3]
                    faces = points[np.asarray(geometry.faces)]
                    ax.add_collection3d(
                        Poly3DCollection(
                            faces,
                            facecolors=np.clip(
                                (
                                    0.5
                                    + 0.5
                                    * np.abs(
                                        np.asarray(geometry.face_normals)
                                        @ transform[:3, :3].T
                                        @ np.array([0.3, -0.6, 0.74])
                                    )
                                )[:, None]
                                * np.array([0.45, 0.68, 0.88])[None, :],
                                0,
                                1,
                            ),
                            edgecolor="none",
                            alpha=1,
                        )
                    )
            ax.set(
                xlim=(-0.10, 0.13),
                ylim=(-0.13, 0.10),
                zlim=(-0.01, 0.22),
                xlabel="x [m]",
                ylabel="y [m]",
                zlabel="z [m]",
                title=f"Adapted Sharpa: {label}",
            )
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=23, azim=-55)
            fig.savefig(args.output / f"{label}.png", dpi=120)
            if label == "closed":
                ax.set_axis_off()
                ax.set_title("")
                fig.set_size_inches(3.2, 3.2)
                fig.savefig(args.output / "thumbnail.jpg", dpi=100)
            plt.close(fig)
    report = {
        "mapping": trajectory.mapping,
        "asset": manifest,
        "samples": evidence,
        "fk_passed": all(x["max_fk_error"] < 2e-6 for x in evidence),
        "interpretation": "URDF-adapted closure; axis equivalence to source CAD is not asserted",
    }
    (args.output / "mapping-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["fk_passed"]


if __name__ == "__main__":
    main()
