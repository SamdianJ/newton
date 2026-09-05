# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Isolated public-API adapters for the DRAFT no-contact articulated/tet fixture."""

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Executed by a separate interpreter; the selected implementation comes from the guarded worktree.
if __package__:
    from .manifest import _validate_body_and_mesh_inputs, compute_manifest_sha256, load_manifest, validate_step_record
else:
    from manifest import _validate_body_and_mesh_inputs, compute_manifest_sha256, load_manifest, validate_step_record


def world_nodes(local, quaternion, translation):
    """Map native local nodes to world coordinates using an xyzw quaternion."""

    local = np.asarray(local, dtype=np.float64).reshape(-1, 3)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector, scalar = quaternion[:3], quaternion[3]
    return local + 2 * np.cross(vector, np.cross(vector, local) + scalar * local) + translation


def validate_adapter_scope(physics):
    """Reject inputs outside the measured no-contact subset before allocating a scene."""
    _validate_body_and_mesh_inputs(physics)
    if physics.get("shapes") != [] or physics.get("contact") is not None or physics.get("drive") is not None:
        raise ValueError("Adapter supports only explicit shapes=[] and contact/drive=null; normal loading is UNMAPPED")
    if any(physics["fixed_nodes"]) or any(physics["surface_materials"]) or any(physics["edge_materials"]):
        raise ValueError("Adapter requires unconstrained nodes and zero surface/edge terms")
    materials = physics["tet_materials_pa_pa_pas"]
    if not materials or any(row != materials[0] for row in materials) or materials[0][2] != 0:
        raise ValueError("Adapter requires homogeneous zero-damping tet materials")
    if any(link["xform"] != [0, 0, 0, 0, 0, 0, 1] for link in physics["links"]):
        raise ValueError("Adapter requires identity builder link transforms")
    if physics["material_law"] != "stable_neo_hookean":
        raise ValueError("Unsupported requested material law")
    if physics["integrator"] != "backward_euler" or physics["mass_lumping"] != "rest_volume_equal_nodes":
        raise ValueError("Unsupported integrator or mass lumping")
    if physics["dt_s"] is None or physics["dt_s"] <= 0 or physics["substeps"] is None or physics["substeps"] <= 0:
        raise ValueError("Adapter requires positive dt and substeps")
    if len(physics["links"]) != len(physics["joints"]):
        raise ValueError("Adapter requires one scalar joint per link")
    for i, joint in enumerate(physics["joints"]):
        if joint["type"] not in ("REVOLUTE", "PRISMATIC") or joint["child"] != i or joint["parent"] >= i:
            raise ValueError("Adapter requires parent-first revolute/prismatic links")
        if joint["child_xform"] != [0, 0, 0, 0, 0, 0, 1]:
            raise ValueError("Nonidentity child joint transforms are outside this adapter's measured subset")


def _base_record(manifest, args, parameters, version):
    schema = json.loads(Path(__file__).with_name("step_record.schema.json").read_text())
    record = dict.fromkeys(schema["required"])
    record.update(
        implementation=args.implementation,
        manifest_sha256=compute_manifest_sha256(manifest),
        **manifest.data["repositories"],
        device=args.device,
        hardware=platform.platform() + " " + platform.processor(),
        build_type=args.build_type,
        warp_or_build_version=version,
        thread_count=1,
        actual_parameters_json=json.dumps(parameters, sort_keys=True, allow_nan=False),
        timing_s={"assembly": None, "collision": None, "linear_solve": None, "total": None},
    )
    return record


def _write_record(output, record):
    reasons = {}
    for key, value in record.items():
        if value is None:
            reasons[key] = (
                "Not independently exposed/measured by this adapter; no value is inferred from another field."
            )
    for key, value in record["timing_s"].items():
        if value is None:
            reasons[f"timing_s.{key}"] = (
                "Public runtime does not expose a comparable separately measured stage duration."
            )
    record["unavailable_fields"] = reasons
    validate_step_record(record)
    output.write(json.dumps(record, allow_nan=False) + "\n")


def _newton(manifest, args, output):
    import warp as wp  # noqa: PLC0415 - Only the selected offline runtime is installed.
    from newton.solvers.experimental.monolithic import (  # noqa: PLC0415
        MonolithicCollisionPipeline,
        SolverMonolithic,
    )

    import newton  # noqa: PLC0415 - Resolve the selected guarded worktree first.

    physics = manifest.data["physics"]
    builder = newton.ModelBuilder(gravity=physics["gravity_m_s2"], up_axis=newton.Axis.Z)
    for link in physics["links"]:
        builder.add_link(
            mass=link["mass_kg"], com=link["com_m"], inertia=wp.mat33(*np.asarray(link["inertia_kg_m2"]).ravel())
        )
    joints = []
    for joint in physics["joints"]:
        add = builder.add_joint_revolute if joint["type"] == "REVOLUTE" else builder.add_joint_prismatic
        transform = joint["parent_xform"]
        joints.append(
            add(
                joint["parent"],
                joint["child"],
                axis=joint["axis"],
                parent_xform=wp.transform(transform[:3], transform[3:]),
                armature=0.0,
                damping=0.0,
                friction=0.0,
                limit_ke=0.0,
                limit_kd=0.0,
                target_ke=0.0,
                target_kd=0.0,
                actuator_mode=newton.JointTargetMode.NONE,
            )
        )
    builder.add_articulation(joints)
    builder.joint_q[:] = physics["initial"]["q"]
    builder.joint_qd[:] = physics["initial"]["qd"]
    mu, lam, _ = physics["tet_materials_pa_pa_pas"][0]
    builder.add_soft_mesh(
        pos=(0, 0, 0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=(0, 0, 0),
        vertices=physics["rest_positions_m"],
        indices=np.asarray(physics["tet_indices"]).ravel().tolist(),
        density=physics["density_kg_m3"],
        k_mu=mu,
        k_lambda=lam,
        k_damp=0.0,
        particle_radius=physics["particle_radius_m"],
        tri_ke=0.0,
        tri_ka=0.0,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
        edge_ke=0.0,
        edge_kd=0.0,
    )
    model = builder.finalize(device=args.device)
    np.testing.assert_allclose(model.particle_mass.numpy(), physics["node_masses_kg"], rtol=1e-6, atol=0)
    state, next_state = model.state(), model.state()
    state.particle_q.assign(np.asarray(physics["initial"]["x_m"], np.float32))
    state.particle_qd.assign(np.asarray(physics["initial"]["v_m_s"], np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    solver = SolverMonolithic(model, collision_pipeline=MonolithicCollisionPipeline(model), contact_stiffness=1e5)
    control = model.control()
    parameters = {
        "scope": "DRAFT no-contact art+tet",
        "dt_s": physics["dt_s"],
        "material_law": "Kim no-log stable Neo-Hookean",
        "stored_tet_materials": model.tet_materials.numpy().tolist(),
        "node_masses_kg": model.particle_mass.numpy().tolist(),
        "gravity_m_s2": model.gravity.numpy().tolist(),
        "solver": {name: getattr(solver, name) for name in ("iterations", "linear_tolerance") if hasattr(solver, name)},
        "internal_config": repr(solver._config),
    }
    base = _base_record(manifest, args, parameters, wp.__version__)
    base["hardware"] = str(wp.get_device(args.device).name)
    rest = np.asarray(physics["rest_positions_m"])
    indices = np.asarray(physics["tet_indices"])
    inverse_rest = np.linalg.inv(np.transpose(rest[indices[:, 1:]] - rest[indices[:, :1]], (0, 2, 1)))
    for step in range(1, physics["substeps"] + 1):
        wp.synchronize_device(model.device)
        start = time.perf_counter()
        solver.step(state, next_state, control, None, physics["dt_s"])
        wp.synchronize_device(model.device)
        elapsed = time.perf_counter() - start
        state, next_state = next_state, state
        nodes = state.particle_q.numpy()
        deformation = np.transpose(nodes[indices[:, 1:]] - nodes[indices[:, :1]], (0, 2, 1)) @ inverse_rest
        stats = solver.last_stats
        record = base.copy()
        record.update(
            step=step,
            time_s=step * physics["dt_s"],
            joint_q=state.joint_q.numpy().tolist(),
            link_xform=state.body_q.numpy().tolist(),
            node_positions_m=nodes.tolist(),
            soft_com_m=np.average(nodes, axis=0, weights=model.particle_mass.numpy()).tolist(),
            min_det_f=float(np.linalg.det(deformation).min()),
            penetration_m=float(stats.max_penetration),
            nonlinear_iterations=stats.nonlinear_iterations,
            linear_iterations=stats.linear_iterations,
            convergence_status=stats.status.name,
            converged=stats.converged,
            rolled_back=stats.rolled_back,
            committed_unconverged=not stats.converged and not stats.rolled_back,
            timing_s={"assembly": None, "collision": None, "linear_solve": None, "total": elapsed},
        )
        _write_record(output, record)


def _superdex(manifest, args, output):
    import superdex.physics as p  # noqa: PLC0415 - SuperDex remains an isolated offline dependency.

    if args.device != "cpu" or p.uses_double_precision():
        raise ValueError("Measured reference adapter requires CPU float32 build")
    physics = manifest.data["physics"]
    p.initialize(num_worker_threads=0)
    scene = p.create_scene("monolithic_reference_draft")
    try:
        scene.set_gravity(physics["gravity_m_s2"])
        params = scene.get_solver_params()
        params.integration_method = p.IntegrationMethod.BACKWARD_EULER
        scene.set_solver_params(params)
        shape = p.create_tet_mesh_shape(
            np.asarray(physics["rest_positions_m"], np.float32).ravel(),
            np.asarray(physics["tet_indices"], np.uint32).ravel(),
        )
        joints, links = [], []
        for i, (joint, link) in enumerate(zip(physics["joints"], physics["links"], strict=True)):
            transform = joint["parent_xform"]
            joints.append(
                p.ArticulatedJointParams(
                    type=getattr(p.ArticulatedJointType, joint["type"]),
                    axis=joint["axis"],
                    parent_link_from_joint=p.TransformRT(rotation=transform[3:], translation=transform[:3]),
                    inertia=0.0,
                    limit_stiffness=0.0,
                    limit_damping=0.0,
                )
            )
            inertia = np.asarray(link["inertia_kg_m2"])
            links.append(
                p.ArticulatedLinkParams(
                    name=f"link{i}",
                    parent_link=joint["parent"],
                    shape=shape,
                    collider_type=p.ColliderType.NONE,
                    mass=link["mass_kg"],
                    center_of_mass=link["com_m"],
                    moment_of_inertia=inertia[np.triu_indices(3)].tolist(),
                    has_gravity=True,
                )
            )
        art = scene.create_articulated_actor(
            name="art", joints=joints, links=links, joint_velocities=np.asarray(physics["initial"]["qd"], np.float32)
        )
        art.set_articulated_pose_from_joints(np.asarray(physics["initial"]["q"], np.float32))
        mu, lam, _ = physics["tet_materials_pa_pa_pas"][0]
        young, poisson = mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))
        material = p.SoftMaterialParams(
            type=p.SoftMaterialType.NEO_HOOKEAN,
            density=physics["density_kg_m3"],
            neo_hookean=p.NeoHookeanMaterialParams(youngs_modulus=young, poisson_ratio=poisson),
            mass_damping_coefficient=0.0,
            stiffness_damping_coefficient=0.0,
        )
        soft = scene.create_soft_actor(
            name="tet", shape=shape, material=material, boundary_element_type=p.ActorBoundaryElementType.P1Q3
        )
        scene.enable_actor_contact_symmetric(art.get_handle(), soft.get_handle(), False, p.IncludeNestedActors.YES)
        scene.enable_actor_contact_symmetric(art.get_handle(), art.get_handle(), False, p.IncludeNestedActors.YES)
        soft.register_query(p.QueryType.CONTACT_POINTS)
        soft.register_query(p.QueryType.TOTAL_CONTACT_FORCE)
        soft.register_query(p.QueryType.NODE_POSITIONS)
        soft.register_query(p.QueryType.ELEMENTS_DEFORMATION_GRADIENT)
        soft.set_node_positions_local(np.asarray(physics["initial"]["x_m"], np.float32).ravel())
        soft.set_node_velocities_local(np.asarray(physics["initial"]["v_m_s"], np.float32).ravel())
        link_actors = [scene.get_actor(handle) for handle in art.get_nested_link_actors()]
        measured_mass = [actor.get_mass() for actor in link_actors]
        measured_com = [np.asarray(actor.get_rigid_center_of_mass_local()).tolist() for actor in link_actors]
        np.testing.assert_allclose(measured_com, [link["com_m"] for link in physics["links"]], rtol=1e-6, atol=0)
        measured_inertia = [np.asarray(actor.get_rigid_moment_of_inertia_local()).tolist() for actor in link_actors]
        for inertia, link in zip(measured_inertia, physics["links"], strict=True):
            np.testing.assert_allclose(
                inertia, np.asarray(link["inertia_kg_m2"])[np.triu_indices(3)], rtol=1e-6, atol=0
            )
        initial_q, initial_qd = np.empty(len(joints), np.float32), np.empty(len(joints), np.float32)
        art.get_articulated_pose(initial_q)
        art.get_articulated_joint_velocities(initial_qd)
        np.testing.assert_allclose(initial_q, physics["initial"]["q"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(initial_qd, physics["initial"]["qd"], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(measured_mass, [link["mass_kg"] for link in physics["links"]], rtol=1e-6, atol=0)
        np.testing.assert_allclose(soft.get_mass(), sum(physics["node_masses_kg"]), rtol=1e-6, atol=0)
        effective = soft.get_soft_material_params()
        nonlinear = scene.get_solver_params().non_linear_solver
        linear = scene.get_solver_params().linear_solver
        friction = art.get_articulated_joint_friction_params()
        if any(value.coulomb or value.viscous or value.stiction_extra for value in friction):
            raise ValueError("No-drive adapter unexpectedly has native joint friction")
        if any(art.get_articulated_joint_inertia_params()):
            raise ValueError("No-drive adapter unexpectedly has native joint armature")
        native = Path(sys.modules[p.initialize.__module__].__file__)
        parameters = {
            "scope": "DRAFT no-contact art+tet",
            "integration_method": str(params.integration_method),
            "dt_s": physics["dt_s"],
            "num_worker_threads": 0,
            "precision": "float32",
            "material_law": "Smith log-stabilized Neo-Hookean (public NEO_HOOKEAN)",
            "youngs_modulus": effective.neo_hookean.youngs_modulus,
            "poisson_ratio": effective.neo_hookean.poisson_ratio,
            "psd_strategy": str(effective.neo_hookean.psd_strategy),
            "soft_mass_kg": soft.get_mass(),
            "link_masses_kg": measured_mass,
            "link_com_m": measured_com,
            "link_inertia_kg_m2": measured_inertia,
            "initial_q": initial_q.tolist(),
            "initial_qd": initial_qd.tolist(),
            "inertia_carrier": "tet shape with ColliderType.NONE; explicit mass/COM/inertia",
            "nonlinear": {name: str(getattr(nonlinear, name)) for name in dir(nonlinear) if not name.startswith("_")},
            "linear": {name: str(getattr(linear, name)) for name in dir(linear) if not name.startswith("_")},
            "joint_friction": [
                {
                    name: float(getattr(value, name))
                    for name in ("coulomb", "viscous", "stiction_extra", "falloff_vel", "stribeck_vel")
                }
                for value in friction
            ],
            "native_module": str(native),
            "native_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "native_smith_mu_hat_pa": 4 * mu / 3,
            "native_smith_lambda_hat_pa": lam + 5 * mu / 6,
            "newton_kim_lambda_tilde_pa": mu + lam,
        }
        base = _base_record(manifest, args, parameters, "pinned SuperDex public API fp32")
        for step in range(1, physics["substeps"] + 1):
            scene.step(physics["dt_s"])
            q = np.empty(len(joints), np.float32)
            art.get_articulated_pose(q)
            root = soft.get_root_transform()
            nodes = world_nodes(
                soft.get_node_positions_local(), np.asarray(root.rotation), np.asarray(root.translation)
            )
            deformation = np.asarray(soft.get_elements_deformation_gradient()).reshape(-1, 3, 3)
            stats = scene.get_solver_stats()
            if len(soft.get_contact_points_world()) != 0 or np.any(np.asarray(soft.get_contact_force_world()) != 0):
                raise ValueError("No-contact adapter unexpectedly produced contact")
            record = base.copy()
            record.update(
                step=step,
                time_s=step * physics["dt_s"],
                joint_q=q.tolist(),
                link_xform=[
                    [
                        *np.asarray(actor.get_root_transform().translation).tolist(),
                        *np.asarray(actor.get_root_transform().rotation).tolist(),
                    ]
                    for actor in link_actors
                ],
                node_positions_m=nodes.tolist(),
                soft_com_m=np.average(nodes, axis=0, weights=physics["node_masses_kg"]).tolist(),
                min_det_f=float(np.linalg.det(deformation).min()),
                penetration_m=0.0,
                normal_physical_force_n=np.asarray(soft.get_contact_force_world()).tolist(),
                tangential_physical_force_n=[0.0, 0.0, 0.0],
                nonlinear_iterations=stats.max_non_linear_iters,
                convergence_status=str(stats.convergence_status),
                converged=stats.convergence_status == p.ConvergenceStatus.CONVERGED,
                timing_s={
                    "assembly": None,
                    "collision": None,
                    "linear_solve": None,
                    "total": scene.get_performance_stats().total_step_duration_sec,
                },
            )
            _write_record(output, record)
    finally:
        p.destroy_scene(scene)
        p.shutdown()


def main():
    """Run only the selected runtime, leaving diagnostics on stdout/stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--implementation", choices=("newton", "superdex"), required=True)
    for name in ("manifest", "worktree", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--build-type", required=True)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    validate_adapter_scope(manifest.data["physics"])
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.worktree))
    with args.output.open("w") as output:
        (_newton if args.implementation == "newton" else _superdex)(manifest, args, output)


if __name__ == "__main__":
    main()
