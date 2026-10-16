# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private frozen-reference admission and offline robot contact recording seams."""

import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict
from functools import partial
from unittest.mock import patch

import numpy as np

import newton
from newton._src.solvers.monolithic import solver_monolithic
from newton._src.solvers.monolithic.articulation import _configuration_digest, _TreeKinematics
from newton._src.solvers.monolithic.contact import ContactObjective
from newton._src.solvers.monolithic.contact_pipeline import _PooledContactObjective, _SpatialContactObjective
from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit
from scripts.monolithic_reference.coupled_fixtures import NORMAL


def check_robot_reports(directory, run, expected_steps):
    """Keep Stopped labels and require full residual gates for every claimed convergence."""
    import gzip  # noqa: PLC0415
    import json  # noqa: PLC0415

    from newton._src.solvers.monolithic.iteration import _block_policy, blocks  # noqa: PLC0415
    from scripts.monolithic_reference.check_recording import sha  # noqa: PLC0415

    counts = {"full_converged": 0, "stopped": 0}
    count = 0
    with gzip.open(directory / "steps.jsonl.gz", "rt") as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            report = row["full_system"]
            if (
                row["step"] != count
                or report["step"] != count
                or report["qualification"] != "FINAL_RESIDUAL_ONLY"
                or report["full_count"] != run._contact.count
                or report["blocks"].keys() != dict(blocks(run)).keys()
                or not np.isfinite(report["residual_difference_l2"])
                or report["residual_difference_l2"] < 0
                or not 0 <= report["candidate_count"] <= min(run._contact.count, run._contact.candidate_capacity)
                or row["contact_pool"]["tainted"]
                or row["contact_pool"]["dropped"]
                or row["contact_spatial"]["tainted"]
                or row["contact_spatial"]["capacity_dropped"]
            ):
                raise ValueError("Invalid or tainted robot contact report")
            all_full = True
            for name, part in blocks(run):
                value = report["blocks"][name]
                goal = (
                    _block_policy(run._config, "merit_absolute", name)
                    + _block_policy(run._config, "merit_relative", name) * value["reference_merit"]
                )
                if (
                    value["size"] != part.stop - part.start
                    or value["goal"] != goal
                    or any(
                        not np.isfinite(value[n]) or value[n] < 0
                        for n in ("clipped_merit", "full_merit", "reference_merit")
                    )
                    or value["full_pass"] is not bool(value["full_merit"] <= goal)
                    or value["clipped_pass"] is not bool(value["clipped_merit"] <= goal)
                ):
                    raise ValueError("Robot contact report changed an original physical gate")
                all_full &= value["full_pass"]
                if row["publication"] == "CONVERGED" and not (value["full_pass"] and value["clipped_pass"]):
                    raise ValueError("Clipped convergence failed the complete system gate")
            if report["status"] != ("PASS_FULL_RESIDUAL" if all_full else "FAIL_FULL_RESIDUAL"):
                raise ValueError("Robot contact report status differs")
            if row["publication"] not in ("CONVERGED", "STOPPED_ACCEPTED"):
                raise ValueError("Robot contact report is not a published state")
            counts["full_converged" if row["publication"] == "CONVERGED" else "stopped"] += 1
    if count != expected_steps:
        raise ValueError("Robot contact report clock differs")
    return {
        "status": "PASS_REPORT",
        "steps": count,
        **counts,
        "steps_sha256": sha(directory / "steps.jsonl.gz"),
        "qualification": "FINAL_RESIDUAL_ONLY; STOPPED_REMAINS_STOPPED",
    }


def certify_configuration(run, fields, reference_digest):
    """Reproduce the actual hash, then project only reviewed policy/resource fields."""
    if _configuration_digest(fields) != run._kinematics.digest or run._states.precision_bits != 64:
        raise ValueError("Actual constructor configuration digest differs")
    projected = {name: np.asarray(value).copy() for name, value in fields.items()}
    direction = projected.pop("direction_policy")
    if direction.dtype != np.int32 or not np.array_equal(direction, np.array([1], np.int32)):
        raise ValueError("Unreviewed Q1 direction policy")
    removed = ["direction_policy=int32[1]"]
    c = run._contact
    algorithm = {"contact_dataflow", "contact_resources", "contact_spatial"}
    if run._contact_spatial:
        if c.law_margin != 0:
            raise ValueError("Frozen experimental recipe requires zero law margin")
        expected = {
            "contact_dataflow": np.frombuffer(b"spatial-pool/newton-frozen/v1", np.uint8),
            "contact_resources": np.array(
                [
                    c.pool.capacity,
                    c.pool.max_support,
                    c._batch.batch_capacity,
                    run._coupled_config["guard_capacity"][1],
                    c.candidate_capacity,
                ],
                np.int64,
            ),
            "contact_spatial": np.array([c.law_margin], np.float64),
        }
        for name, value in expected.items():
            actual = projected.pop(name)
            if actual.dtype != value.dtype or not np.array_equal(actual, value):
                raise ValueError("Contact algorithm/resource identity differs")
        guard = projected["guard_capacity"]
        if guard.shape != (2,) or guard[1] != c.resource_identity[3]:
            raise ValueError("Contact factor budget differs")
        # The frozen constructor used the full static factor budget. Preserve its physical guard.
        guard[1] = 3 * c.count
        removed += [*sorted(algorithm), "guard_capacity[1]: full static factor budget"]
    elif run._contact_pool or projected.keys() & algorithm:
        raise ValueError("Frozen robot comparison supports legacy Q1 or explicit B3")
    if _configuration_digest(projected).hex() != reference_digest:
        raise ValueError("Projected configuration differs from frozen reference")
    return _ConfigurationCertificate(run, fields, reference_digest, removed)


class _ConfigurationCertificate:
    def __init__(self, run, fields, reference, changes):
        self.run = run
        self.fields = {name: np.asarray(value).copy() for name, value in fields.items()}
        self.actual, self.reference = run._kinematics.digest.hex(), reference
        self.changes = changes

    def validate(self):
        if self.run._kinematics.digest.hex() != self.actual:
            raise ValueError("Certified actual configuration digest changed")
        certify_configuration(self.run, self.fields, self.reference)
        return self.reference, self.actual

    def receipt(self):
        self.validate()
        return {
            "adapter": "frozen-robot-q1-b3/v1",
            "reference_digest": self.reference,
            "actual_digest": self.actual,
            "authenticated_changes": self.changes,
        }


class FrozenRobotFactory:
    """Select a bounded experiment at construction without changing the solver API/default."""

    def __init__(self, scene, *, dataflow="legacy"):
        if scene not in ("franka-cube", "sharpa-recorded-10mm") or dataflow not in ("legacy", "spatial", "default"):
            raise ValueError("Unsupported frozen robot recipe")
        self.scene, self.dataflow = scene, dataflow
        self.fields, self.run, self.recorder = None, None, None

    def __call__(self, model, **options):
        original, captured = _TreeKinematics.__init__, []

        def capture(k, *args, **kwargs):
            original(k, *args, **kwargs)
            fields = {**k._source_values, **kwargs.get("extra_config", {})}
            if _configuration_digest(fields) != k.digest:
                raise ValueError("Captured constructor fields differ")
            captured.append({name: np.asarray(value).copy() for name, value in fields.items()})

        if self.run is not None or "contact_factor_capacity" in options:
            raise ValueError("Robot recipe cannot be reused or override resources")
        if self.dataflow != "legacy":
            # Sharpa retains the complete static resource bounds; calibration may reduce them later.
            # Derive the exact supported enumeration from compact metadata, not a shape-count guess.
            with patch.object(solver_monolithic, "ContactObjective", _PooledContactObjective):
                complete = solver_monolithic.SolverMonolithic(model, contact_factor_capacity=0, **options)
            context, resources = nullcontext(), {}
            if self.dataflow == "spatial":
                query = 8192 if self.scene == "franka-cube" else complete._contact.count
                factory = partial(_SpatialContactObjective, candidate_capacity=query, law_margin=0.0)
                resources["contact_factor_capacity"] = (
                    24576 if self.scene == "franka-cube" else 3 * complete._contact.count
                )
                context = patch.object(solver_monolithic, "ContactObjective", factory)
            with context, patch.object(_TreeKinematics, "__init__", capture):
                run = solver_monolithic.SolverMonolithic(model, **resources, **options)
            self.recorder = RobotContactRecorder(run, complete)
        else:
            with (
                patch.object(solver_monolithic, "ContactObjective", ContactObjective),
                patch.object(_TreeKinematics, "__init__", capture),
            ):
                run = solver_monolithic.SolverMonolithic(model, **options)
        if len(captured) != 1:
            raise ValueError("Expected one robot constructor configuration")
        self.run, self.fields = run, captured[0]
        return run

    def certify(self, reference_digest):
        return certify_configuration(self.run, self.fields, reference_digest)

    def measure(self, run, state):
        if run is not self.run or self.recorder is None:
            raise ValueError("Recording belongs to a different robot recipe")
        return self.recorder.measure(state)

    def recipe(self):
        c = self.run._contact
        # The constructor digest does not authenticate the internal lambda schedule.
        policy = {
            "constructor_digest": self.run._kinematics.digest.hex(),
            "effective_internal_config": asdict(self.run._config),
        }
        return {
            "scene": self.scene,
            "dataflow": "spatial" if self.run._contact_spatial else "legacy",
            "precision": "fp64",
            "production_default": self.dataflow == "default",
            "resources": c.resource_identity.tolist() if self.run._contact_pool else None,
            "law_margin": c.law_margin if self.run._contact_spatial else None,
            "zero_overflow_required": True,
            "additional_approximation_budget": None,
            "numerical_policy": policy,
            "numerical_policy_sha256": hashlib.sha256(
                json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }


class RobotContactRecorder:
    """Read committed forces and stream full diagnostics on independent offline scratch."""

    def __init__(self, run, complete):
        self.audit = ContactSystemAudit(run, complete)

    def measure(self, state):
        run, complete = self.audit.run, self.audit.complete
        report = self.audit.evaluate()
        k, b = complete._kinematics, complete._contact._batch
        h = self.audit.state._monolithic_coupled
        args = b._state(self.audit.state.particle_q, k.poses, k.com, k.jt, k.jw, h.rigid_com, h.rigid_rotation)
        penetration, masked_penetration, slip_max, work, load = 0.0, 0.0, 0.0, 0.0, 0.0
        keys, omitted_sum = [], np.zeros(4)
        for mode in b._chunks():
            b._stage(args, mode, run._dt, begin=False)
            if b._flags.numpy()[0]:
                raise ValueError("Invalid full robot diagnostic geometry")
            n = int(b.pool.candidates.count.numpy()[0])
            ids = b.pool.candidates.identities.numpy()[:n]
            gap = b.current.distance.numpy()[:n] - b.samples.gaps.numpy()[:n]
            penetration = max(penetration, float(np.maximum(-gap, 0).max(initial=0)))
            normal = b.normal.numpy()[:n]
            slide = b.current.y.numpy()[:n] - b._start_chunk.y.numpy()[:n]
            slip = slide - np.einsum("ni,ni->n", normal, slide)[:, None] * normal
            friction = b.weights.numpy()[:n, 1, None] * slip
            gradient = b.current.gradient.numpy()[:n]
            local = b.local_r.numpy()[:n]
            load += float(
                (
                    -np.einsum("ni,ni->n", local - friction, gradient) / np.maximum(np.sum(gradient**2, axis=1), 1e-300)
                ).sum()
            )
            work += float(np.sum(friction * slide))
            slip_max = max(slip_max, float(np.linalg.norm(slip, axis=1).max(initial=0)))
            omitted = b.omitted.numpy()[:n]
            active = np.any(omitted[:, 2:] > 0, axis=1)
            keys.extend(ids[active].tolist())
            omitted_sum += omitted.sum(axis=0)
            masked_penetration = max(masked_penetration, float(np.maximum(-gap[active], 0).max(initial=0)))
        c, physical = run._contact, state._monolithic_coupled
        records = c.pool.records
        n = int(records.count.numpy()[0])
        ids, pairs = records.identities.numpy()[:n], records.pairs.numpy()[:n]
        sources, targets = c._batch.all_samples.sources.numpy(), c._batch.all_samples.targets.numpy()
        forces = np.zeros((2, 3))
        types = run.model.shape_type.numpy()
        for slot in np.argsort(ids, kind="stable"):
            key = int(ids[slot])
            if sources[key] != -1:
                raise ValueError("Robot recorder requires canonical tet-to-shape samples")
            forces[int(types[targets[key]] != int(newton.GeoType.PLANE))] += pairs[slot, 0, :3]
        pforce, bforce = physical.contact_particle_force.numpy(), physical.contact_body_force.numpy()
        return {
            "penetration": penetration,
            "contact_load": load,
            "friction_work": work,
            "slip": slip_max,
            "ground_force": forces[0],
            "robot_force": forces[1],
            "full_system": report,
            "contact_pool": c.evaluation_diagnostics(),
            "contact_spatial": c.spatial_diagnostics(),
            "mask": {
                "masked_active_count": len(keys),
                "masked_active_keys": keys,
                "omitted_normal_energy": omitted_sum[0],
                "omitted_friction_energy": omitted_sum[1],
                "omitted_normal_load_sum": omitted_sum[2],
                "omitted_friction_load_sum": omitted_sum[3],
                "max_masked_penetration": masked_penetration,
                "max_all_penetration": penetration,
                "switch_history": "NOT_RECORDED_BY_BOUNDED_STREAM",
            },
            "force_balance": float(
                np.linalg.norm(pforce.sum(axis=0) + bforce[:, :3].sum(axis=0))
                / max(
                    np.linalg.norm(pforce, axis=1).sum() + np.linalg.norm(bforce[:, :3], axis=1).sum(),
                    NORMAL["acceptance"]["force_floor_n"],
                )
            ),
            "body_contact_force": bforce,
            "q": state.joint_q.numpy().copy(),
            "qd": state.joint_qd.numpy().copy(),
            "x": state.particle_q.numpy().copy(),
            "v": state.particle_qd.numpy().copy(),
        }
