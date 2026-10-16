# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Round-trip material reference data through authored USD files and native solvers."""

import importlib.util
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

import newton
from newton._src.solvers.monolithic.material_inputs import _MaterialInputs
from newton.solvers.experimental.monolithic import SolverMonolithic, export_materials_usd, import_materials_usd
from newton.tests.test_monolithic_material_state import scene
from scripts.monolithic_reference.material_scene import MaterialScene


def inputs(solver):
    return {
        kind + "_inputs": tuple(
            cls(
                **{
                    field.name: item[field.name].item() if item[field.name].ndim == 0 else item[field.name]
                    for field in fields(cls)
                }
            )
            for item in objects
        )
        for kind, cls, objects in (
            ("cloth", SolverMonolithic.ClothInput, solver._materials.cloth),
            ("cable", SolverMonolithic.CableInput, solver._materials.cable),
        )
    }


@unittest.skipUnless(importlib.util.find_spec("pxr"), "USD Python bindings are unavailable")
class TestMaterialUSD(unittest.TestCase):
    def test_import_preserves_world_gravity(self):
        """Append materials to the active world without changing its gravity."""
        model, solver = scene(cloth=False, cables=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "world.usda"
            export_materials_usd(path, model, **inputs(solver))
            for world in (-1, 0):
                with self.subTest(world=world):
                    builder = newton.ModelBuilder(gravity=(0, 0, -9.81))
                    if world == 0:
                        builder.begin_world(gravity=(1, 2, 3))
                    expected = np.array(builder.world_gravity[world] if world >= 0 else builder.gravity)
                    import_materials_usd(builder, path)
                    actual = builder.world_gravity[world] if world >= 0 else builder.gravity
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(builder.particle_world, [world] * model.particle_count)

    def test_round_trip(self):
        """Preserve anisotropy, unwrapped twist, natural folds and fixed boundaries."""
        box = MaterialScene("precreased-box")
        for model, solver in (scene(), (box.model, box.solver)):
            with self.subTest(particles=model.particle_count), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "rest.usda"
                export_materials_usd(path, model, **inputs(solver))
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                imported = import_materials_usd(builder, path)
                actual_model = builder.finalize(device="cpu")
                material = _MaterialInputs(actual_model, **imported)
                np.testing.assert_array_equal(actual_model.particle_q.numpy(), model.particle_q.numpy())
                for expected, actual in zip(
                    solver._materials.cloth + solver._materials.cable, material.cloth + material.cable, strict=True
                ):
                    self.assertEqual(expected.keys(), actual.keys())
                    for key in expected:
                        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
                export_materials_usd(path, actual_model, **imported)
                second = newton.ModelBuilder(gravity=(0, 0, 0))
                second_inputs = import_materials_usd(second, path)
                run = SolverMonolithic(
                    second.finalize(device="cpu"),
                    **second_inputs,
                    contact_model="C-IE",
                    friction_velocity_scale=0.1,
                    globalization="ResNorm",
                )
                state = run.create_state(dt=0.001)
                run.step(state, state, run.create_control(), dt=0.001)
                self.assertTrue(run.last_stats.converged)
                np.testing.assert_allclose(state.particle_q.numpy(), model.particle_q.numpy(), atol=1e-10)

    def test_units_transform_and_builder_offsets(self):
        """Bake world transform and stage units once, preserving SI coefficients and ID offsets."""
        from pxr import Gf, Usd, UsdGeom

        model, solver = scene(cloth=False, cables=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "units.usda"
            export_materials_usd(path, model, **inputs(solver))
            stage = Usd.Stage.Open(str(path))
            UsdGeom.SetStageMetersPerUnit(stage, 0.01)
            prim = stage.GetPrimAtPath("/material_0000")
            points = UsdGeom.PointBased(prim).GetPointsAttr()
            points.Set([Gf.Vec3f(*(50 * np.array(p)).tolist()) for p in points.Get()])
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(100, 0, 0))
            xform.AddRotateZOp().Set(90)
            xform.AddScaleOp().Set(Gf.Vec3f(2))
            stage.GetRootLayer().Save()
            builder = newton.ModelBuilder(gravity=(0, 0, 0))
            builder.add_particle((0, 0, 0), (0, 0, 0), 0)
            imported = import_materials_usd(builder, path)
            actual = builder.finalize(device="cpu")
            expected = model.particle_q.numpy()[:, [1, 0, 2]] * [-1, 1, 1] + [1, 0, 0]
            np.testing.assert_allclose(actual.particle_q.numpy()[1:], expected, atol=1e-6)
            item = imported["cable_inputs"][0]
            self.assertEqual(item.particle_ids, (1, 2, 3))
            self.assertEqual(item.radius, 0.01)
            self.assertEqual(item.EA, 10)
            np.testing.assert_allclose(item.rest_director, [[-1, 0, 0]] * 2, atol=1e-12)
            _MaterialInputs(actual, **imported)

    def test_rejection_is_atomic(self):
        """Reject unsupported authoring before changing the caller's builder."""
        from pxr import Gf, Sdf, Usd, UsdGeom

        model, solver = scene(cloth=False, cables=1)
        for invalid in ("missing", "tensor", "closed", "scale", "animated", "version_type"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "invalid.usda"
                export_materials_usd(path, model, **inputs(solver))
                stage = Usd.Stage.Open(str(path))
                prim = stage.GetPrimAtPath("/material_0000")
                if invalid == "missing":
                    prim.RemoveProperty("newton:monolithic:restDirector")
                elif invalid == "tensor":
                    prim.GetAttribute("newton:monolithic:bending").Set(Gf.Vec4d(1, 2, 2, 1))
                elif invalid == "closed":
                    UsdGeom.BasisCurves(prim).GetWrapAttr().Set(UsdGeom.Tokens.periodic)
                elif invalid == "scale":
                    UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(2, 1, 1))
                elif invalid == "animated":
                    prim.CreateAttribute("newton:monolithic:EA", Sdf.ValueTypeNames.Double).Set(20.0, 1.0)
                else:
                    prim.RemoveProperty("newton:monolithic:version")
                    prim.CreateAttribute("newton:monolithic:version", Sdf.ValueTypeNames.Bool).Set(True)
                stage.GetRootLayer().Save()
                builder = newton.ModelBuilder()
                builder.add_particle((0, 0, 0), (0, 0, 0), 1)
                with self.assertRaises(ValueError):
                    import_materials_usd(builder, path)
                self.assertEqual(builder.particle_count, 1)
                self.assertEqual(builder.tri_count, 0)


if __name__ == "__main__":
    unittest.main()
