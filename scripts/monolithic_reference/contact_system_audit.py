# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Audit the complete native residual outside the spatial solver's step."""

import json
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.monolithic.coupled_kernels import combine
from newton._src.solvers.monolithic.iteration import _block_policy, blocks, measure
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
            goal = (
                _block_policy(run._config, "merit_absolute", name)
                + _block_policy(run._config, "merit_relative", name) * value["reference_merit"]
            )
            if (
                value.keys()
                != {"clipped_merit", "full_merit", "reference_merit", "goal", "size", "clipped_pass", "full_pass"}
                or value["size"] != part.stop - part.start
                or value["goal"] != goal
                or any(
                    not np.isfinite(value[n]) or value[n] < 0
                    for n in ("clipped_merit", "full_merit", "reference_merit")
                )
                or value["clipped_pass"] is not True
                or value["full_pass"] is not True
                or max(value["clipped_merit"], value["full_merit"]) > goal
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
            **({} if begin else {"dt": self.run._dt}),
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
        reference = run._reference_residual
        if isinstance(reference, wp.array):
            reference = reference.numpy()
        scale = run._linear.scale.numpy().astype(run._numpy_dtype)
        if any(not np.isfinite(v).all() for v in (full, clipped, reference, scale)) or np.any(scale <= 0):
            raise ValueError("Nonfinite complete system audit")
        # measure writes diagnostics; a report-only view keeps production stats intact.
        view = SimpleNamespace(
            _numpy_dtype=run._numpy_dtype,
            _linear=run._linear,
            _config=run._config,
            _metrics={},
            _variational={},
        )
        values = [measure(view, residual, reference, scale) for residual in (clipped, full)]
        blocks = {
            name: {
                "clipped_merit": values[0][name][0],
                "full_merit": value[0],
                "reference_merit": value[1],
                "goal": value[2],
                "size": value[3],
                "clipped_pass": bool(values[0][name][0] <= value[2]),
                "full_pass": bool(value[0] <= value[2]),
            }
            for name, value in values[1].items()
        }
        passed = all(block["full_pass"] for block in blocks.values())
        return {
            "status": "PASS_FULL_RESIDUAL" if passed else "FAIL_FULL_RESIDUAL",
            "step": run._step_generation,
            "full_count": c.count,
            "candidate_count": c.spatial_diagnostics()["candidates"],
            "residual_difference_l2": float(np.linalg.norm(full - clipped)),
            "blocks": blocks,
            "qualification": "FINAL_RESIDUAL_ONLY",
        }
