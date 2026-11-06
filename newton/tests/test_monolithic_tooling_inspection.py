# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read-only state capture: schema, values, binding roles, non-perturbation and guards."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton.solvers
from newton._src.solvers.monolithic import inspection
from newton._src.solvers.monolithic.solver_types import _StepFailure
from newton.examples.softbody.monolithic_recording import serial
from newton.tests.test_monolithic_material_state import scene
from newton.tests.test_monolithic_solver_actor_contacts import CASE, contact_scene, contact_solver, contact_state
from newton.tests.test_monolithic_solver_coupled import snapshot
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Frozen schema version 1: name -> (binding, unit, frame). A change here needs a new schema version.
FIELDS_V1 = {
    "joint_q": ("state", "joint", "joint"),
    "joint_qd": ("state", "joint_rate", "joint"),
    "particle_q": ("state", "m", "world"),
    "particle_qd": ("state", "m/s", "world"),
    "rigid_com": ("state", "m", "world"),
    "rigid_rotation": ("state", "quaternion", "world"),
    "rigid_v_com": ("state", "m/s", "world"),
    "rigid_omega": ("state", "rad/s", "world"),
    "cable_theta": ("state", "rad", "material"),
    "contact_particle_force": ("state", "N", "world"),
    "contact_body_force": ("state", "N|N m", "world"),
    "contact_joint_force": ("state", "joint_force", "joint"),
    "contact_cable_torque": ("state", "N m", "material"),
    "force_generation": ("state", "1", "none"),
    "force_valid": ("state", "1", "none"),
    "residual": ("last_evaluation", "row_unit", "row"),
    "row_unit": ("static", "1", "row"),
    "row_producer": ("static", "1", "row"),
    "row_block": ("static", "1", "row"),
    "row_group": ("static", "1", "row"),
    "energy_terms": ("last_evaluation", "J", "none"),
    "energy_term_producer": ("static", "1", "none"),
    "producer_energy": ("last_evaluation", "J", "none"),
    "tet_min_det_f": ("last_evaluation", "1", "none"),
    "tet_det_f": ("derived", "1", "element"),
    "contact_identity": ("last_evaluation", "1", "none"),
    "contact_source": ("last_evaluation", "1", "none"),
    "contact_target": ("last_evaluation", "1", "none"),
    "contact_wrench": ("last_evaluation", "N|N m", "world"),
    "pair_source": ("last_evaluation", "1", "none"),
    "pair_target": ("last_evaluation", "1", "none"),
    "pair_wrench": ("last_evaluation", "N|N m", "world"),
}
STEP_BOUND = {name for name, (binding, _, _) in FIELDS_V1.items() if binding in ("last_evaluation", "derived")}


def chain(device, precision="fp64", *, graph=False, mode="production"):
    model = contact_scene(device, "chain")
    run = contact_solver(model, mode=mode, graph=graph, precision=precision)
    return run, contact_state(run, "chain"), model.control()


def step_chain(run, state, control, step):
    case = json.loads(CASE.read_text())
    targets = control.joint_target_q.numpy()
    targets[list(run._articulation_actor.target_indices)] = (step + 1) * case["target_increment_per_step"]
    control.joint_target_q.assign(targets)
    run.step(state, state, control, dt=case["dt"])


def precisions():
    return ("fp32", "fp64")


def test_schema_stable(test, device):
    """Freeze schema version 1 and round-trip it through disk."""
    shapes = {}
    for precision in precisions():
        with test.subTest(precision=precision):
            run, state, control = chain(device, precision)
            step_chain(run, state, control, 0)
            result = inspection.capture(run, state)
            schema, arrays = result["schema"], result["arrays"]
            test.assertEqual(schema["schema_version"], 1)
            test.assertEqual(inspection.SCHEMA_VERSION, 1)
            names = set(schema["fields"]) | set(schema["absent"])
            test.assertEqual(names, set(FIELDS_V1))
            test.assertEqual(set(schema["fields"]), set(arrays))
            for name, spec in schema["fields"].items():
                test.assertEqual((spec["binding"], spec["unit"], spec["frame"]), FIELDS_V1[name], name)
                for key in ("dtype", "shape", "meaning", "source", "valid_when"):
                    test.assertIn(key, spec, name)
                test.assertEqual(spec["dtype"], str(arrays[name].dtype), name)
                test.assertEqual(tuple(spec["shape"]), arrays[name].shape, name)
            for key in ("precision", "device", "configuration", "layout", "step_generation", "role"):
                test.assertIn(key, schema)
            test.assertTrue(schema["host_synchronizing"])
            test.assertIsInstance(schema["reads"], list)
            test.assertEqual(schema["precision"], precision)
            floating = np.float32 if precision == "fp32" else np.float64
            test.assertEqual(arrays["particle_q"].dtype, floating)
            test.assertEqual(arrays["residual"].dtype, floating)
            shapes[precision] = {n: (s["shape"], s["binding"]) for n, s in schema["fields"].items()}
            with tempfile.TemporaryDirectory() as folder:
                target = Path(folder) / "capture"
                inspection.save_capture(result, target, metadata={"sha": "test"})
                loaded = inspection.load_capture(target)
                test.assertEqual(set(loaded["arrays"]), set(arrays))
                for name, value in arrays.items():
                    np.testing.assert_array_equal(loaded["arrays"][name], value, err_msg=name)
                    test.assertEqual(loaded["arrays"][name].dtype, value.dtype, name)

                def reject(value):
                    raise ValueError(value)

                text = (target / "schema.json").read_text()
                test.assertEqual(json.loads(text, parse_constant=reject)["metadata"], {"sha": "test"})
                with test.assertRaises(FileExistsError):
                    inspection.save_capture(result, target)
    test.assertEqual(shapes["fp32"], shapes["fp64"])


def test_state_fields(test, device):
    """State-bound fields equal the state and the public force snapshot, and never alias."""
    run, state, control = chain(device)
    step_chain(run, state, control, 0)
    result = inspection.capture(run, state)
    arrays = result["arrays"]
    h = state._monolithic_coupled
    for name, source in (
        ("joint_q", state.joint_q),
        ("joint_qd", state.joint_qd),
        ("particle_q", state.particle_q),
        ("particle_qd", state.particle_qd),
        ("rigid_com", h.rigid_com),
        ("rigid_rotation", h.rigid_rotation),
        ("rigid_v_com", h.rigid_v_com),
        ("rigid_omega", h.rigid_omega),
        ("contact_particle_force", h.contact_particle_force),
        ("contact_body_force", h.contact_body_force),
        ("contact_joint_force", h.contact_joint_force),
        ("force_generation", h.force_generation),
        ("force_valid", h.force_valid),
    ):
        np.testing.assert_array_equal(arrays[name], source.numpy(), err_msg=name)
    forces = run.get_contact_forces(state)
    np.testing.assert_array_equal(arrays["contact_particle_force"], forces.particle_force)
    np.testing.assert_array_equal(arrays["contact_body_force"], forces.body_wrench)
    np.testing.assert_array_equal(arrays["contact_joint_force"], forces.joint_force)
    kept = {name: value.copy() for name, value in arrays.items()}
    step_chain(run, state, control, 1)
    for name, value in kept.items():
        np.testing.assert_array_equal(arrays[name], value, err_msg=f"{name} changed by a later step")


def test_solver_fields_match_stats(test, device):
    """Evaluation-bound fields agree with Stats on a published step."""
    for precision in precisions():
        with test.subTest(precision=precision):
            run, state, control = chain(device, precision)
            for step in range(2):
                step_chain(run, state, control, step)
            stats = run.last_stats
            test.assertTrue(stats.converged, stats)
            result = inspection.capture(run, state)
            schema, arrays = result["schema"], result["arrays"]
            test.assertEqual(schema["role"], "published")
            test.assertEqual(schema["step_generation"], stats.step_generation)
            np.testing.assert_array_equal(arrays["residual"], stats.variational["residual"])
            rows = arrays["residual"].size
            producers = arrays["row_producer"]
            test.assertEqual(producers.shape, (rows,))
            test.assertTrue((producers >= 0).all())
            for index, producer in enumerate(run._problem.producers):
                test.assertEqual(sorted(np.flatnonzero(producers == index)), sorted(producer.rows.tolist()))
            test.assertEqual(schema["aggregate_block"], run._problem.blocks[0][0])
            test.assertTrue((arrays["row_block"] >= 0).all())
            for index, (name, part) in enumerate(run._problem.blocks[1:]):
                test.assertEqual(schema["blocks"][index], name)
                test.assertTrue((arrays["row_block"][part] == index).all())
            np.testing.assert_array_equal(arrays["row_unit"], run._criterion.rows.units)
            np.testing.assert_array_equal(arrays["row_group"], run._criterion.rows.groups)
            names = [p.name for p in run._problem.producers]
            test.assertEqual(schema["producers"], names)
            total = stats.variational["objective_total"]
            values = arrays["energy_terms"][0].astype(np.float64)
            eps = float(np.finfo(arrays["energy_terms"].dtype).eps)
            bound = values.size * eps * float(np.sum(np.abs(values)))
            test.assertLessEqual(abs(float(np.sum(arrays["producer_energy"])) - float(total)), bound)
            for index in range(len(names)):
                expected = math.fsum(values[arrays["energy_term_producer"] == index])
                test.assertEqual(arrays["producer_energy"][index], expected)
            test.assertEqual(arrays["tet_min_det_f"].item(), stats.min_det_f)
            test.assertEqual(arrays["tet_det_f"].min(), stats.min_det_f)
            test.assertEqual(arrays["tet_det_f"].shape, (run.model.tet_count,))
            test.assertEqual(result["records"]["trials"], json.loads(json.dumps(serial(stats.variational["trials"]))))
            pairs = stats.variational["actor_contact_pairs"]
            test.assertEqual(arrays["pair_wrench"].shape, (len(pairs), 2, 6))
            for index, pair in enumerate(pairs):
                test.assertEqual(arrays["pair_source"][index], pair["source"])
                test.assertEqual(arrays["pair_target"][index], pair["target"])
                np.testing.assert_array_equal(arrays["pair_wrench"][index], np.asarray(pair["world_wrench"]))
            test.assertGreater(arrays["contact_wrench"].shape[0], 0)
            test.assertEqual(arrays["contact_wrench"].shape[1:], (2, 6))


def test_binding_roles(test, device):
    """The capture states which evaluation the solver-held fields describe."""
    run, state, control = chain(device)
    result = inspection.capture(run, state)
    test.assertEqual(result["schema"]["role"], "none")
    test.assertTrue(STEP_BOUND <= set(result["schema"]["absent"]))
    with test.assertRaisesRegex(ValueError, "residual"):
        inspection.capture(run, state, fields=["residual"])
    old = run._states.copy(state)
    step_chain(run, state, control, 0)
    test.assertEqual(inspection.capture(run, state)["schema"]["role"], "published")
    stale = inspection.capture(run, old)
    test.assertNotEqual(stale["schema"]["role"], "published")
    test.assertFalse(stale["schema"]["describes_state"])

    _, stopped = scene(device=device, publish_policy="current_iterate", pcg_mode="production", tet=True)
    stopped_state, stopped_control = stopped.create_state(dt=0.001), stopped.create_control()
    iterate = stopped._iterate_articulation

    def stop(dt, iterate=iterate, run=stopped):
        iterate(dt)
        return run.Status.NONLINEAR_MAX_ITERATIONS

    with patch.object(stopped, "_iterate_articulation", stop):
        stopped.step(stopped_state, stopped_state, stopped_control, dt=0.001)
    schema = inspection.capture(stopped, stopped_state)["schema"]
    test.assertEqual(schema["role"], "published")
    test.assertEqual(schema["publication_status"], "STOPPED_ACCEPTED")

    _, failed = scene(device=device, pcg_mode="production", tet=True)
    failed_state, failed_control = failed.create_state(dt=0.001), failed.create_control()
    failed.step(failed_state, failed_state, failed_control, dt=0.001)
    finish = failed._finish_candidate

    def fail_after_finish(candidate, dt, finish=finish, run=failed):
        finish(candidate, dt)
        raise _StepFailure(run.Status.NONFINITE, "injected publication failure")

    with patch.object(failed, "_finish_candidate", side_effect=fail_after_finish):
        failed.step(failed_state, failed_state, failed_control, dt=0.001)
    test.assertTrue(failed.last_stats.rolled_back)
    schema = inspection.capture(failed, failed_state)["schema"]
    test.assertNotEqual(schema["role"], "published")
    test.assertIn(schema["role"], ("accepted_iterate", "trial", "original"))
    test.assertFalse(schema["describes_state"])


def without_wall_clock(value):
    """Drop wall-clock timing fields (``*_seconds``); they never repeat between runs."""
    if isinstance(value, dict):
        return {k: without_wall_clock(v) for k, v in value.items() if not str(k).endswith("_seconds")}
    if isinstance(value, list):
        return [without_wall_clock(v) for v in value]
    return value


def frames(run, state, control, steps, *, hook=None):
    result = []
    for step in range(steps):
        step_chain(run, state, control, step)
        if hook is not None:
            hook(run, state, control, step)
        result.append({"snapshot": snapshot(state), "stats": without_wall_clock(serial(run.last_stats))})
    return result


def same(left, right):
    for a, b in zip(left, right, strict=True):
        if a["stats"] != b["stats"] or a["snapshot"].keys() != b["snapshot"].keys():
            return False
        for name, value in a["snapshot"].items():
            if not np.array_equal(value, b["snapshot"][name]):
                return False
    return True


def test_capture_does_not_perturb(test, device):
    """Capturing after every step leaves every later step, Stats and graph cache bit for bit unchanged."""
    graphs = (False, True) if device.is_cuda else (False,)
    for graph in graphs:
        with test.subTest(graph=graph):

            def captured(run, state, control, step):
                first = inspection.capture(run, state)
                if step == 2:
                    second = inspection.capture(run, state)
                    for name, value in first["arrays"].items():
                        np.testing.assert_array_equal(second["arrays"][name], value, err_msg=name)

            def nudge(run, state, control, step):
                # A tiny external input change; editing the state itself is rejected by the solver.
                if step == 2:
                    forces = state.particle_f.numpy()
                    forces[0, 0] += 1e-6
                    state.particle_f.assign(forces)

            control_a = frames(*chain(device, graph=graph), 6)
            control_b = frames(*chain(device, graph=graph), 6)
            if not same(control_a, control_b):
                test.skipTest("runs without capture are not bitwise repeatable here; inconclusive")
            test.assertTrue(same(control_a, frames(*chain(device, graph=graph), 6, hook=captured)))
            # The comparator must see a tiny input change, or the equality above proves nothing.
            test.assertFalse(same(control_a, frames(*chain(device, graph=graph), 6, hook=nudge)))


def test_guards(test, device):
    """Reject foreign states, unknown fields, graph capture and existing output, and stay off the step path."""
    run, state, control = chain(device)
    step_chain(run, state, control, 0)
    with test.assertRaisesRegex(ValueError, "no_such_field"):
        inspection.capture(run, state, fields=["no_such_field"])
    _, other = scene(device=device, pcg_mode="production", tet=True)
    with test.assertRaises(ValueError):
        inspection.capture(run, other.create_state(dt=0.001))
    if device.is_cuda:
        with wp.ScopedCapture(device=device, force_module_load=False):
            with test.assertRaisesRegex(RuntimeError, "graph capture"):
                inspection.capture(run, state)
    public = newton.solvers.experimental.monolithic.__all__
    test.assertNotIn("inspection", public)
    test.assertNotIn("capture", public)
    with patch.object(inspection, "capture", side_effect=AssertionError("capture on the step path")):
        step_chain(run, state, control, 1)
    test.assertTrue(run.last_stats.converged)


class TestToolingInspection(unittest.TestCase):
    pass


for function in (
    test_schema_stable,
    test_state_fields,
    test_solver_fields_match_stats,
    test_binding_roles,
    test_capture_does_not_perturb,
    test_guards,
):
    add_function_test(TestToolingInspection, function.__name__, function, devices=get_test_devices())


if __name__ == "__main__":
    unittest.main()
