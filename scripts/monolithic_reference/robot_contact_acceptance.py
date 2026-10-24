# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private frozen-reference admission and offline robot contact recording seams."""

import hashlib
import json
from dataclasses import asdict
from unittest.mock import patch

import numpy as np

import newton
from newton._src.solvers.monolithic import solver_monolithic
from newton._src.solvers.monolithic.articulation import _configuration_digest, _TreeKinematics
from scripts.monolithic_reference.contact_routes import pooled_audit_route
from scripts.monolithic_reference.contact_system_audit import ContactSystemAudit
from scripts.monolithic_reference.coupled_fixtures import NORMAL


def check_robot_reports(directory, run, expected_steps):
    """Keep Stopped labels and require full residual gates for every claimed convergence."""
    import gzip  # noqa: PLC0415
    import json  # noqa: PLC0415

    from newton._src.solvers.monolithic.iteration import blocks  # noqa: PLC0415
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
                if (
                    value["size"] != part.stop - part.start
                    or any(not np.isfinite(value[n]) or value[n] < 0 for n in ("clipped_ratio", "full_ratio"))
                    or value["full_pass"] is not bool(value["full_ratio"] <= 1)
                    or value["clipped_pass"] is not bool(value["clipped_ratio"] <= 1)
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


def certify_configuration(run, fields, reference_digest, *, precision_bits=64):
    """Bind the actual constructor identity and report whether it equals the frozen reference.

    Robot runs qualify on physical gates (D11), so a frozen-reference trajectory
    comparison is diagnostic. The actual digest must reproduce from the
    recorded fields; a differing reference only leaves the certificate
    unauthenticated.
    """
    if _configuration_digest(fields) != run._kinematics.digest or run._states.precision_bits != precision_bits:
        raise ValueError("Actual constructor configuration digest differs")
    authenticated = run._kinematics.digest.hex() == reference_digest
    reason = None if authenticated else "Configuration identity differs from the frozen reference"
    return _ConfigurationCertificate(run, fields, reference_digest, authenticated, reason)


class _ConfigurationCertificate:
    def __init__(self, run, fields, reference, authenticated, reason):
        self.run = run
        self.fields = {name: np.asarray(value).copy() for name, value in fields.items()}
        self.actual, self.reference = run._kinematics.digest.hex(), reference
        self.authenticated, self.reason = authenticated, reason

    def validate(self):
        if self.run._kinematics.digest.hex() != self.actual or _configuration_digest(self.fields).hex() != self.actual:
            raise ValueError("Certified actual configuration digest changed")
        return self.reference, self.actual

    def receipt(self):
        self.validate()
        return {
            "adapter": "frozen-robot-identity/v2",
            "reference_digest": self.reference,
            "actual_digest": self.actual,
            "authenticated": self.authenticated,
            "reason": self.reason,
        }


class FrozenRobotFactory:
    """Select a bounded experiment at construction without changing the solver API/default."""

    def __init__(self, scene, *, dataflow="spatial", precision="fp64"):
        if scene not in ("franka-cube", "sharpa-recorded-10mm") or dataflow not in ("spatial", "default"):
            raise ValueError("Unsupported frozen robot recipe")
        self.scene, self.dataflow = scene, dataflow
        self.precision = solver_monolithic.SolverMonolithic.Precision(precision)
        self.fields, self.run, self.recorder = None, None, None

    def __call__(self, model, **options):
        original, captured = _TreeKinematics.__init__, []

        def capture(k, *args, **kwargs):
            original(k, *args, **kwargs)
            fields = {**k._source_values, **kwargs.get("extra_config", {})}
            if self.precision == solver_monolithic.SolverMonolithic.Precision.FP32:
                # FP32 kinematics add their native width to the digested fields.
                fields["native_precision"] = np.array([32], dtype=np.int32)
            if _configuration_digest(fields) != k.digest:
                raise ValueError("Captured constructor fields differ")
            captured.append({name: np.asarray(value).copy() for name, value in fields.items()})

        if self.run is not None or "contact_factor_capacity" in options:
            raise ValueError("Robot recipe cannot be reused or override resources")
        if solver_monolithic.SolverMonolithic.Precision(options.get("precision", "fp64")) != self.precision:
            raise ValueError("Robot scene precision differs from its recipe")
        # Sharpa retains the complete static resource bounds; calibration may reduce them later.
        # Derive the exact supported enumeration from compact metadata, not a shape-count guess.
        with pooled_audit_route():
            complete = solver_monolithic.SolverMonolithic(model, contact_factor_capacity=0, **options)
        resources = {}
        if self.dataflow == "spatial":
            # The reduced Franka bounds were calibrated in FP64. FP32 candidate envelopes are wider
            # (Franka needs about 21000 of 23328 candidates), so FP32 keeps the complete bounds.
            calibrated = (
                self.scene == "franka-cube" and self.precision == solver_monolithic.SolverMonolithic.Precision.FP64
            )
            resources["contact_candidate_capacity"] = 8192 if calibrated else complete._contact.count
            resources["contact_factor_capacity"] = 24576 if calibrated else 3 * complete._contact.count
        with patch.object(_TreeKinematics, "__init__", capture):
            run = solver_monolithic.SolverMonolithic(model, **resources, **options)
        self.recorder = RobotContactRecorder(run, complete)
        if len(captured) != 1:
            raise ValueError("Expected one robot constructor configuration")
        self.run, self.fields = run, captured[0]
        return run

    def certify(self, reference_digest):
        bits = 64 if self.precision == solver_monolithic.SolverMonolithic.Precision.FP64 else 32
        return certify_configuration(self.run, self.fields, reference_digest, precision_bits=bits)

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
            "dataflow": "spatial",
            "precision": self.precision.value,
            "production_default": self.dataflow == "default",
            "resources": c.resource_identity.tolist(),
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
        args = b._state(
            self.audit.state.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            h.rigid_com,
            h.rigid_rotation,
            run._contact_steps(run._accepted),
        )
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
            slide = b.current.slip.numpy()[:n]
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
