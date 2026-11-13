# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drop independent square cloth layers onto a fixed sphere or table, in SI units."""

import argparse
import json

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic, build_smooth_sdf

import newton

from .cloth_contact_run import add_arguments, create_solver, execute, publication
from .material_recipes import shell_coefficients
from .run_cloth_contact import _grid


def _sphere_mesh(radius, num_latitudes, num_longitudes):
    """Build a sphere without collapsed polar faces rejected by material sampling."""
    if any(type(n) is not int or n < 3 for n in (num_latitudes, num_longitudes)):
        raise ValueError("Sphere mesh subdivisions must be integers of at least three")
    mesh = newton.Mesh.create_sphere(
        radius, num_latitudes=num_latitudes, num_longitudes=num_longitudes, compute_inertia=False
    )
    faces = mesh.indices.reshape(-1, 3)
    points = mesh.vertices[faces].astype(float)
    area2 = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
    keep = area2 > np.finfo(float).eps * radius**2
    return newton.Mesh(mesh.vertices, faces[keep].ravel(), compute_inertia=False)


class ClothStackScene:
    """Advance freely falling cloth without attachments or artificial settling forces."""

    def __init__(
        self, recipe, *, device="cpu", precision="fp64", use_cuda_graph=False, method="off", method_config=None
    ):
        if recipe.get("units") != "SI":
            raise ValueError("Require an SI cloth-stack recipe")
        self.dt = float(recipe["dt_s"])
        duration = float(recipe["duration_s"])
        if not np.isfinite([self.dt, duration]).all() or min(self.dt, duration) <= 0:
            raise ValueError("Require finite positive timestep and duration")
        if not np.isclose(duration / self.dt, round(duration / self.dt)):
            raise ValueError("Require an integral planned step count")
        self.total_steps = round(duration / self.dt)
        self.steps, self.phase = 0, "free-drop"
        cloth, material = recipe["cloth"], recipe["material"]
        if ("sphere" in recipe) == ("table" in recipe):
            raise ValueError("Require exactly one fixed sphere or table")
        support = "sphere" if "sphere" in recipe else "table"
        shape = recipe[support]
        mesh_spec = shape.get("mesh") if support == "sphere" else None
        layers, cells = cloth["layers"], cloth["cells_per_side"]
        if type(layers) is not int or layers < 1 or len(cloth["rotation_z_deg"]) != layers:
            raise ValueError("Require a positive layer count and one rotation per layer")
        center = np.asarray(shape["center_m"], dtype=float)
        if support == "sphere":
            top_offset = float(shape["radius_m"])
        else:
            half = np.asarray(shape["half_extents_m"], dtype=float)
            if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                raise ValueError("Require finite positive table half extents")
            top_offset = half[2]
        gap, clearance = float(cloth["layer_gap_m"]), float(cloth[f"{support}_clearance_m"])
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Require a finite support center")
        if not np.isfinite([top_offset, gap, clearance]).all() or min(top_offset, gap, clearance) <= 0:
            raise ValueError("Require positive support size and initially separated layers")
        if gap <= material["thickness_m"] or clearance <= material["thickness_m"] / 2:
            raise ValueError("Initial layers must not overlap one another or the support")
        sheet, faces = _grid(cells, cells, cloth["width_m"], cloth["width_m"])
        sheet[:, 0] -= cloth["width_m"] / 2
        positions = []
        for layer, angle in enumerate(np.deg2rad(cloth["rotation_z_deg"])):
            if not np.isfinite(angle):
                raise ValueError("Require finite layer rotations")
            rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            positions.append(sheet @ rotation.T + center + [0, 0, top_offset + clearance + layer * gap])
        rest = np.vstack(positions).astype(np.float32).astype(float)
        self.triangles = np.vstack([faces + layer * len(sheet) for layer in range(layers)])
        self.owners = np.repeat(np.arange(layers), len(faces))
        self.thicknesses = np.full(len(self.triangles), material["thickness_m"])
        face_x = rest[self.triangles]
        areas = np.linalg.norm(np.cross(face_x[:, 1] - face_x[:, 0], face_x[:, 2] - face_x[:, 0]), axis=1) / 2
        mass = np.zeros(len(rest))
        np.add.at(mass, self.triangles.ravel(), np.repeat(areas * material["surface_density_kg_m2"] / 3, 3))
        lam, mu, alpha, beta = shell_coefficients(material)
        builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
        if mesh_spec is not None:
            if not wp.get_device(device).is_cuda:
                raise ValueError("Sphere mesh contact requires CUDA SDF textures")
            mesh = _sphere_mesh(top_offset, mesh_spec["num_latitudes"], mesh_spec["num_longitudes"])
            sdf = mesh_spec["sdf"]
            mesh.build_sdf(
                device=device,
                max_resolution=sdf["max_resolution"],
                margin=sdf["margin_m"],
                narrow_band_range=tuple(sdf["narrow_band_m"]),
                texture_format="float32",
                sign_method="winding",
            )
            builder.add_shape_mesh(-1, xform=wp.transform(center, wp.quat_identity()), mesh=mesh)
        elif support == "sphere":
            builder.add_shape_sphere(-1, xform=wp.transform(center, wp.quat_identity()), radius=top_offset)
        else:
            builder.add_shape_box(
                -1, xform=wp.transform(center, wp.quat_identity()), hx=half[0], hy=half[1], hz=half[2]
            )
        for point, value in zip(rest, mass, strict=True):
            builder.add_particle(wp.vec3(point), wp.vec3(0), float(value), radius=material["thickness_m"] / 2)
        for face in self.triangles:
            builder.add_triangle(*face, tri_ke=mu, tri_ka=lam, tri_kd=0, tri_drag=0, tri_lift=0)
        self.model = builder.finalize(device=device)
        inputs = [
            SolverMonolithic.ClothInput(
                triangle_ids=tuple(np.flatnonzero(self.owners == layer).tolist()),
                bending=(alpha, beta),
                thickness=material["thickness_m"],
                surface_density=material["surface_density_kg_m2"],
            )
            for layer in range(layers)
        ]
        options = dict(
            recipe["solver"],
            precision=SolverMonolithic.Precision(precision),
            pcg_mode=SolverMonolithic.PCGMode.PRODUCTION,
            use_cuda_graph=use_cuda_graph,
        )
        if mesh_spec is not None:
            options["smooth_sdf"] = build_smooth_sdf(self.model)
        self.solver = create_solver(self.model, inputs, options, method, method_config)
        self.state = self.solver.create_state(dt=self.dt)
        self.control = self.solver.create_control()
        self.metadata = {
            "kind": f"cloth-stack-{support}",
            "recipe": recipe,
            "method": method,
            "method_config": method_config,
            "total_mass_kg": float(mass.sum()),
            "layer_mass_kg": [float(mass[i * len(sheet) : (i + 1) * len(sheet)].sum()) for i in range(layers)],
            "vertices_per_layer": len(sheet),
            "triangles_per_layer": len(faces),
            "gravity_m_s2": [0, 0, -9.81],
            support: f"fixed analytic collision {support}; no additional ground plane",
            "cloth_attachment": "none; all layers released simultaneously at rest",
        }
        if mesh_spec is not None:
            self.metadata["sphere"] = "fixed triangle mesh with C2 SDF collision; no additional ground plane"
            self.metadata["support_mesh"] = {"vertices": len(mesh.vertices), "triangles": len(mesh.indices) // 3}
            self.metadata["smooth_sdf"] = options["smooth_sdf"].build_metadata

    def step(self):
        """Advance the physical clock only after valid solver publication and feedback."""
        previous = self.solver.last_stats.step_generation
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        stats = self.solver.last_stats
        self.last_forces = self.solver.get_contact_forces(self.state)
        if publication(stats, self.last_forces, previous_generation=previous) == "HARD_FAILURE":
            raise RuntimeError(f"{stats.status}: {stats.failure_reason}")
        self.steps += 1
        return stats


def main():
    """Record the shared falling-stack scene with an explicit method sidecar."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text())
    config = json.loads(args.method_config.read_text()) if args.method_config else None
    return execute(
        lambda: ClothStackScene(
            recipe,
            device=args.device,
            precision=args.precision,
            use_cuda_graph=args.use_cuda_graph,
            method=args.method,
            method_config=config,
        ),
        args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
