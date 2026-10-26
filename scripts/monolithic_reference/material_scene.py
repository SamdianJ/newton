# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the frozen coupled material/actuator acceptance scenes."""

import numpy as np
import warp as wp
from newton.solvers.experimental.monolithic import SolverMonolithic

import newton

from .material_recipes import MATERIAL_RECIPE_PATH, build_case, shell_coefficients


def load_fraction(step, common):
    """Evaluate the frozen smooth loading protocol at the step start."""
    for first, last, phase in common["phases"]:
        if first <= step < last:
            s = (step - first) / (last - first)
            smooth = s * s * (3 - 2 * s)
            return {"rest": 0.0, "load": smooth, "hold": 1.0, "unload": 1 - smooth, "recover": 0.0}[phase]
    raise ValueError("Step outside the recipe")


class MaterialScene:
    """Keep scene construction and controls identical across device and restart paths."""

    def __init__(
        self,
        name,
        *,
        recipe=MATERIAL_RECIPE_PATH,
        device="cpu",
        pcg_mode="diagnostic",
        use_cuda_graph=False,
        precision="fp64",
    ):
        self.name = name
        self.options = {
            "recipe": recipe,
            "device": device,
            "pcg_mode": pcg_mode,
            "use_cuda_graph": use_cuda_graph,
            "precision": precision,
        }
        self.case = case = build_case(name, recipe=recipe)
        self.common, self.recipe = c, r = case["common"], case["recipe"]
        self.dt, self.steps = c["dt"], 0
        self.fixed = np.zeros(len(case["positions"]), dtype=bool)
        self.fixed[case["fixed"]] = True
        self.triangles = case["triangles"]
        points = case["positions"].astype(np.float32).astype(float)
        builder = newton.ModelBuilder(gravity=c["gravity"])
        pose = wp.transform(wp.vec3(r["driver_center_rest_m"]), wp.quat_identity())
        body = builder.add_link(
            mass=c["driver_mass"], inertia=wp.diag(wp.vec3(c["driver_inertia_diagonal"])), xform=pose
        )
        joint = builder.add_joint_prismatic(
            -1,
            body,
            parent_xform=pose,
            axis=wp.vec3(r["driver_axis"]),
            target_ke=c["drive_stiffness"],
            target_kd=c["drive_damping"],
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            limit_ke=0,
            limit_kd=0,
            damping=0,
            friction=0,
        )
        builder.add_articulation([joint])
        cfg = builder.ShapeConfig(density=0.0, margin=0.0)
        if r["driver_shape"] == "sphere":
            builder.add_shape_sphere(body, radius=r["driver_radius_m"], cfg=cfg)
        else:
            builder.add_shape_plane(
                body=body,
                width=0,
                length=0,
                xform=wp.transform(wp.vec3(0), wp.quat_from_axis_angle(wp.vec3(1, 0, 0), np.pi)),
                cfg=cfg,
            )
        self.mass = np.zeros(len(points))
        cloth, cables = [], []
        if r["material"] == "cable":
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            self.mass[:-1] += lengths * r["linear_density_kg_m"] / 2
            self.mass[1:] += lengths * r["linear_density_kg_m"] / 2
            fixed_theta = np.zeros(len(lengths), dtype=bool)
            fixed_theta[r["fixed_theta_edges"]] = True
            cables.append(
                SolverMonolithic.CableInput(
                    particle_ids=tuple(range(len(points))),
                    EA=r["EA_N"],
                    bending=r["bending_Nm2"],
                    GJ=r["GJ_Nm2"],
                    radius=r["radius_m"],
                    linear_density=r["linear_density_kg_m"],
                    rest_theta=np.zeros(len(lengths)),
                    rest_director=np.tile([0.0, 1.0, 0.0], (len(lengths), 1)),
                    fixed_theta=fixed_theta,
                )
            )
        else:
            faces = points[self.triangles]
            self.rest_area = np.linalg.norm(np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0]), axis=1) / 2
            np.add.at(self.mass, self.triangles.ravel(), np.repeat(self.rest_area * r["surface_density_kg_m2"] / 3, 3))
            material = shell_coefficients(r)
            cloth.append(
                SolverMonolithic.ClothInput(
                    triangle_ids=tuple(range(len(self.triangles))),
                    bending=material[2:],
                    thickness=r["thickness_m"],
                    surface_density=r["surface_density_kg_m2"],
                )
            )
        for i, point in enumerate(points):
            builder.add_particle(wp.vec3(point), wp.vec3(0), 0 if self.fixed[i] else self.mass[i])
        for face in self.triangles:
            builder.add_triangle(*face, tri_ke=material[1], tri_ka=material[0], tri_kd=0, tri_drag=0, tri_lift=0)
        self.model = builder.finalize(device=device)
        self.rest = self.model.particle_q.numpy().astype(float)
        retired = {"joint_residual_tolerance", "cloth_position_tolerance", "linear_absolute_tolerance"} & c.keys()
        if retired:
            raise ValueError(f"Recipe sets retired solver options {sorted(retired)}; use material_acceptance_v8.json")
        if "precision" in c:
            raise ValueError("Recipe fixes the precision; the runner selects it")
        self.solver = SolverMonolithic(
            self.model,
            cloth_inputs=cloth,
            cable_inputs=cables,
            mass_mode="lumped",
            joint_terms=SolverMonolithic.JointTerms(implicit_pd=True),
            precision=SolverMonolithic.Precision(precision),
            pcg_mode=SolverMonolithic.PCGMode(pcg_mode),
            use_cuda_graph=use_cuda_graph,
            contact_stiffness=r.get("contact_stiffness", c["contact_stiffness"]),
            **{
                key: c[key]
                for key in (
                    "publish_policy",
                    "friction_coefficient",
                    "friction_velocity_scale",
                    "normal_smoothing",
                    "newton_max_iterations",
                    "line_search_max_iterations",
                    "linear_max_iterations",
                    "linear_tolerance",
                )
            },
        )
        self.state = self.solver.create_state(dt=self.dt)
        self.control = self.solver.create_control()

    def step(self):
        """Advance one fully coupled step and require genuine convergence."""
        scale = load_fraction(self.steps, self.common)
        self.control.joint_target_q.fill_(self.recipe["driver_travel_m"] * scale)
        self.control.joint_target_qd.zero_()
        if self.recipe["material"] == "cable":
            theta = np.zeros(len(self.rest) - 1)
            theta[-1] = self.recipe["end_theta_load_rad"] * scale
            self.control._monolithic_cable.target_theta.assign(theta)
        self.solver.step(self.state, self.state, self.control, dt=self.dt)
        stats = self.solver.last_stats
        if not stats.converged or stats.rolled_back:
            raise AssertionError(f"Step {self.steps + 1}: {stats.status}; {stats.failure_reason}")
        self.steps += 1
