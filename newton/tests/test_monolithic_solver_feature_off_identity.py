# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Structural identity of feature-off scenes (P9d bit-identity set X1).

With no floating tree, no CAPSULE/CONVEX_MESH contact shape and self-collision off, every
scene must build the same solver structure as the reference: kinematics digest, layout
payload, state schema, producer order, contact samples and capacities, pair kinds. The
expected hashes were captured on the reference checkout named below by running
``structural_identity`` there; recapture only when the trunk itself legitimately changes
them (P9d plan, set X1), and say so in the PR record.
"""

import hashlib
import json
import unittest

import numpy as np

REFERENCE_SHA = "7f733beb"  # P9d PR0 close; solver tree identical to trunk d50ead9b


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def structural_identity(solver):
    """Hashes of everything that fixes the solver's structure, independent of stepping."""
    states = solver._states
    samples = getattr(solver._contact, "actor_samples", None)
    sample_arrays = {}
    if samples is not None:
        for name in ("points", "normals", "weights", "offsets", "targets", "source_shapes", "faces", "slots"):
            value = getattr(samples, name, None)
            if value is not None:
                sample_arrays[name] = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    return {
        "digest": bytes(solver._kinematics.digest).hex()
        if isinstance(solver._kinematics.digest, (bytes, bytearray))
        else str(solver._kinematics.digest),
        "layout": hashlib.sha256(solver._actors.layout_payload()).hexdigest(),
        "schema_version": int(states.version),
        "schema": _hash(sorted((name, str(spec)) for name, spec in states.schema.items())),
        "producers": [type(p).__name__ for p in solver._producers],
        "contact_count": int(solver._contact.count),
        "contact_triplet_capacity": int(solver._contact.triplet_capacity),
        "samples": _hash(sample_arrays),
        "pair_kinds": sorted({str(k) for k in (samples.pair_kinds.values() if samples is not None else ())}),
    }


def _fixtures():
    """Feature-off scenes that build on CPU, each returning its solver."""
    import json as _json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    fixtures = Path(__file__).resolve().parents[2] / "scripts/monolithic_reference/fixtures"

    def rigid_stack():
        from scripts.monolithic_reference.rigid_stack_scene import stack_scene, stack_solver  # noqa: PLC0415

        return stack_solver(stack_scene("cpu"), spatial=True)

    def remote_tree():
        from scripts.monolithic_reference.run_dexterous_tasks import TaskScene  # noqa: PLC0415

        return TaskScene("turn-knob", remote_tree=True, device="cpu", precision="fp64").solver

    def tet():
        from scripts.monolithic_reference.tet_benchmarks import TetBenchmark  # noqa: PLC0415

        return TetBenchmark("cantilever", cells=(20, 2, 2), device="cpu").solver

    def cloth_point_cloud():
        from scripts.monolithic_reference.run_cloth_stack import ClothStackScene  # noqa: PLC0415

        recipe = _json.loads((fixtures / "cloth_stack_table_v1.json").read_text())
        config = _json.loads((fixtures / "cloth_stack_point_cloud_v1.json").read_text())
        return ClothStackScene(recipe, method="point-cloud", method_config=config).solver

    def cable():
        from scripts.monolithic_reference.cable_benchmarks import CableBenchmark  # noqa: PLC0415

        return CableBenchmark("cantilever").solver

    def cable_sphere():
        from scripts.monolithic_reference.material_scene import MaterialScene  # noqa: PLC0415

        return MaterialScene("cable-bend-twist", device="cpu").solver

    def allegro_wrist():
        from scripts.monolithic_reference.floating_hand_scenes import FloatingHandScene, load_recipe  # noqa: PLC0415

        recipe = load_recipe(fixtures / "floating_hand_wrist_box_baseline_v1.json")
        return FloatingHandScene(recipe, device="cpu", precision="fp64").solver

    return {
        "rigid-stack": rigid_stack,
        "remote-tree": remote_tree,
        "tet-cantilever": tet,
        "cloth-point-cloud": cloth_point_cloud,
        "cable-cantilever": cable,
        "cable-sphere": cable_sphere,
        "allegro-wrist-box": allegro_wrist,
    }


# Captured on REFERENCE_SHA with capture(); see the module docstring.
EXPECTED = {
    "allegro-wrist-box": {
        "contact_count": 49632,
        "contact_triplet_capacity": 30921984,
        "digest": "d2b1c2e0136d2c30d87d338921a87cb5ce58b7972f51ed5364172d72cc780508",
        "layout": "28b1748985459c0c0a9b861daa98b71700c493d7c07da1028a1215e1f485ee3d",
        "pair_kinds": ["A-R", "A-static", "rigid-static"],
        "producers": ["ArticulationProducer", "RigidProducer", "ContactProducer"],
        "samples": "2d5e9e4bd633e8a2d94086dee729a7d0c182cd658fdb3e3fe416554fe66c08a3",
        "schema": "d4f73ed9b57dc20b5e778fac54783ccb0d985ff938ecada76d148abe39fd03c4",
        "schema_version": 4,
    },
    "cable-cantilever": {
        "contact_count": 0,
        "contact_triplet_capacity": 0,
        "digest": "1b8e7dcbd27a7473c23dba1119a7ccb1db6361652a01c4282e4bbc43c228f50c",
        "layout": "d796eaee93ef4784d823ce7775e0793e62a11343837168d0d8112119519f3ad0",
        "pair_kinds": [],
        "producers": ["CableProducer", "ContactProducer"],
        "samples": "bc23f8bc662a1bca2c08475f3c3b88373bb21597a6b1907801d41790fd99b50a",
        "schema": "df8b43f727ca0f097a9aee3675e2b81f79d9666e04f0c3a6f3b3fe9771866e50",
        "schema_version": 6,
    },
    "cable-sphere": {
        "contact_count": 2816,
        "contact_triplet_capacity": 482304,
        "digest": "e62eb5ef94567f372624a62e95eb6e6e8691eac004cdfca1be9da42fccc29578",
        "layout": "5392121c3d8ec193645b0101767e192237c64e2919a6229b4719dfa7d8186d42",
        "pair_kinds": ["cable-shape"],
        "producers": ["ArticulationProducer", "CableProducer", "ContactProducer"],
        "samples": "c793126b369b21656aa39366242a2b8911593cf90f2d92d63d0dc3a008fb1bbd",
        "schema": "14242b99bc1d38b7bf07c9e246b005394ef4fa0a3c23b8f1592445b254a0106d",
        "schema_version": 6,
    },
    "cloth-point-cloud": {
        "contact_count": 2400,
        "contact_triplet_capacity": 583200,
        "digest": "4e3eafca1e518c490ce168cb729288c854101b3a1b68d1ca54c7a136e5d57ced",
        "layout": "8cf9bf46c88173de45e325da9fb7b79643dc8199d4d2d08518dac5caa6cdc371",
        "pair_kinds": ["cloth-shape"],
        "producers": ["ClothProducer", "ClothProducer", "ContactProducer", "_ClothContactProducer"],
        "samples": "f5d750021b1274d28401836867a68a555b063a15d982fe24ebe19033beb1257a",
        "schema": "54df6d0364e9bd07d313b15a48225d35744fd33221405cdaa0643faba633cf3b",
        "schema_version": 8,
    },
    "remote-tree": {
        "contact_count": 36000,
        "contact_triplet_capacity": 2513880,
        "digest": "5cebec613293351f2b7727ee81a60166b2db7692b7c7ea0107f2a33d5b7b3f9c",
        "layout": "4b049bf7cbb022a9696a329e74297844def7828da1249ba37a92061c841aa3c9",
        "pair_kinds": ["A-A", "A-static"],
        "producers": ["ArticulationProducer", "ContactProducer"],
        "samples": "276eab1d79e221d4b37ea2ae84c24ca754b07db39cf062666558a1de3cf218dd",
        "schema": "ee5218f5e7d8343f67d2baea25f1a29f87d1e8fd59b56a8397bfd5cd4d3bc034",
        "schema_version": 7,
    },
    "rigid-stack": {
        "contact_count": 1296,
        "contact_triplet_capacity": 513216,
        "digest": "6f7ec373bce55e78ab1017856022526738f02aa523dfc17fe5d0a93d4ab7c284",
        "layout": "dca475dbbad611b47c1fc0866e5361e28bdcea06d76ec385c9fdd127ca199bd8",
        "pair_kinds": ["R-R", "rigid-static"],
        "producers": ["RigidProducer", "ContactProducer"],
        "samples": "90c6f161cf6cd1ac1937bdd44df356dbb412e501e8abd83586a8638f9a06fbd3",
        "schema": "7a57e4bed64b9cc064b342dd5d2dd04b1f0a0d694fb778e36dcc7cef79812096",
        "schema_version": 4,
    },
    "tet-cantilever": {
        "contact_count": 0,
        "contact_triplet_capacity": 0,
        "digest": "d21d4ce5d716c11a70d6aa527ab8482cea15c4f31c234151d36f3e93ac23af9c",
        "layout": "4577e883c9904d0732b9fe323eea4a8adc1d8bb9d34d2e98a41e51e542940dd3",
        "pair_kinds": [],
        "producers": ["TetProducer", "ContactProducer"],
        "samples": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        "schema": "a3b79108d19b5015af7f8badd7f4194e4da621d71211b8fc770b049caed40c96",
        "schema_version": 4,
    },
}


class TestFeatureOffIdentity(unittest.TestCase):
    def test_structure_matches_reference(self):
        import os  # noqa: PLC0415

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        for name, build in _fixtures().items():
            with self.subTest(scene=name):
                if name not in EXPECTED:
                    self.fail(f"No reference identity captured for {name}")
                self.assertEqual(structural_identity(build()), EXPECTED[name])


def capture():
    """Print the identities as a Python literal (run on the reference checkout)."""
    print(
        json.dumps(
            {name: structural_identity(build()) for name, build in _fixtures().items()}, indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    unittest.main()
