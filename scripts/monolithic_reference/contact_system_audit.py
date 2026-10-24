# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit the complete native residual outside the spatial solver's step."""

import json

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.coupled_kernels import combine
from newton._src.solvers.monolithic.iteration import blocks
from newton._src.solvers.monolithic.rigid import _scatter_residual
from scripts.monolithic_reference.contact_audit import verify_contract


def check_system_report(directory, run, summary):
    """Authenticate the saved report and its original block goals and complete clock."""
    from scripts.monolithic_reference.check_recording import sha  # noqa: PLC0415

    report = summary["full_system"]
    if (
        not run._contact_spatial
        or report.keys() != {"file", "sha256", "contract", "checked_steps", "failed_steps"}
        or report["file"] != "full-system.jsonl"
        or sha(directory / report["file"]) != report["sha256"]
        or report["checked_steps"] != summary["checked_steps"]
        or report["failed_steps"] != 0
        or report["contract"]["comparison"] != "pooled/spatial"
        or report["contract"]["config_digests"][1] != run._kinematics.digest.hex()
    ):
        raise ValueError("Complete system report identity differs")
    rows = [json.loads(line) for line in (directory / report["file"]).read_text().splitlines()]
    if [row["step"] for row in rows] != list(range(summary["start_step"] + 1, summary["end_step"] + 1)):
        raise ValueError("Complete system report clock differs")
    for row in rows:
        if (
            row["status"] != "PASS_FULL_RESIDUAL"
            or row["qualification"] != "FINAL_RESIDUAL_ONLY"
            or row["full_count"] != run._contact.count
            or type(row["candidate_count"]) is not int
            or not 0 <= row["candidate_count"] <= min(run._contact.count, run._contact.candidate_capacity)
            or not np.isfinite(row["residual_difference_l2"])
            or row["residual_difference_l2"] < 0
            or row["blocks"].keys() != dict(blocks(run)).keys()
        ):
            raise ValueError("Invalid complete system report")
        for name, part in blocks(run):
            value = row["blocks"][name]
            if (
                value.keys() != {"clipped_ratio", "full_ratio", "size", "clipped_pass", "full_pass"}
                or value["size"] != part.stop - part.start
                or any(not np.isfinite(value[n]) or value[n] < 0 for n in ("clipped_ratio", "full_ratio"))
                or value["clipped_pass"] is not True
                or value["full_pass"] is not True
                or max(value["clipped_ratio"], value["full_ratio"]) > 1
            ):
                raise ValueError("Complete system report changed a physical gate")
    return report


class ContactSystemAudit:
    """Replace only contact in the final objective; never feed audit results into Newton."""

    def __init__(self, run, complete):
        self.contract = verify_contract(complete, run)
        self.run, self.complete = run, complete
        complete._contact._batch.initialize_reference()
        self.state = complete.create_state(dt=0.001)
        self.residual = wp.zeros_like(run._complete_r)

    def _contact(self, source, *, begin):
        audit, state = self.complete, self.state
        state.joint_q.assign(source.joint_q)
        state.particle_q.assign(source.particle_q)
        history, original = state._monolithic_coupled, source._monolithic_coupled
        history.rigid_com.assign(original.rigid_com)
        history.rigid_rotation.assign(original.rigid_rotation)
        k = audit._kinematics
        k.evaluate(k.gather_q(state))
        method = audit._contact.begin_step if begin else audit._contact._batch.evaluate_reference
        method(
            state.particle_q,
            k.poses,
            k.com,
            k.jt,
            k.jw,
            rigid_com=history.rigid_com,
            rigid_rotation=history.rigid_rotation,
            # Slip uses the run's exact step increments, as in production; increments
            # derived from rounded absolute poses differ visibly in FP32.
            **({} if begin else {"dt": self.run._dt, "steps": self.run._contact_steps(source)}),
        )

    def evaluate(self):
        """Report every original convergence block for the published final geometry."""
        run, c = self.run, self.run._contact
        if (
            run.last_stats.rolled_back
            or run._current_state is not run._accepted
            or run._step_generation != run.last_stats.step_generation
            or not c._evaluated
            or c._assembled_generation != c.evaluation_generation
            or c.evaluation_diagnostics()["tainted"]
            or c.spatial_diagnostics()["tainted"]
        ):
            raise ValueError("Complete system audit requires an untainted final spatial evaluation")
        self._contact(run._original, begin=True)
        self._contact(run._accepted, begin=False)
        batch = self.complete._contact._batch
        if batch.flags.numpy()[0] or batch.start_flags.numpy()[0]:
            raise ValueError("Invalid complete native contact residual")

        def total(contact):
            # Preserve the production native addition order, including rigid rows.
            wp.launch(
                combine,
                self.residual.size,
                [
                    run._objective.residual,
                    run._tet.residual,
                    contact,
                    run._layout.q_scalar_offset,
                    run._layout.x_scalar_offset,
                    self.residual,
                ],
                device=run.model.device,
            )
            wp.launch(
                _scatter_residual,
                (run._rigid.count, 6),
                [run._rigid.rows, run._rigid.residual, self.residual],
                device=run.model.device,
            )
            return self.residual.numpy().copy()

        clipped = total(c.residual)
        np.testing.assert_array_equal(clipped, run._complete_r.numpy())
        full = total(batch.reference_residual)
        # The published state's frozen row tolerances judge both contact sets.
        tau = run._criterion.tau.numpy()
        if any(not np.isfinite(v).all() for v in (full, clipped, tau)) or np.any(tau <= 0):
            raise ValueError("Nonfinite complete system audit")
        report = {}
        for name, part in blocks(run):
            ratios = [float(np.max(np.abs(r[part]) / tau[part])) for r in (clipped, full)]
            report[name] = {
                "clipped_ratio": ratios[0],
                "full_ratio": ratios[1],
                "size": part.stop - part.start,
                "clipped_pass": ratios[0] <= 1,
                "full_pass": ratios[1] <= 1,
            }
        passed = all(block["full_pass"] for block in report.values())
        return {
            "status": "PASS_FULL_RESIDUAL" if passed else "FAIL_FULL_RESIDUAL",
            "step": run._step_generation,
            "full_count": c.count,
            "candidate_count": c.spatial_diagnostics()["candidates"],
            "residual_difference_l2": float(np.linalg.norm(full - clipped)),
            "blocks": report,
            "qualification": "FINAL_RESIDUAL_ONLY",
        }
