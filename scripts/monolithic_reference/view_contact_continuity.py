# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Inspect the contact-continuity experiment live; retain the rolled-back failure state."""

import functools
import json
from unittest.mock import patch

import newton.examples
from newton._src.solvers.monolithic.collision import MonolithicCollisionPipeline
from newton.examples.softbody import monolithic_sharpa_grasp as grasp
from newton.examples.softbody.example_monolithic_sharpa_soft_cube import Example as CubeExample


class Example(CubeExample):
    @staticmethod
    def create_parser():
        parser = CubeExample.create_parser()
        parser.add_argument("--variant", choices=("baseline", "sdf", "history", "combined"), default="combined")
        parser.set_defaults(position_offset_mm=(10, 10, -5), viewer="gl")
        return parser

    def __init__(self, viewer, args):
        args.output.mkdir(parents=True, exist_ok=False)
        options = {
            "_cubic_sdf": args.variant in ("sdf", "combined"),
            "_smooth_history": args.variant in ("history", "combined"),
        }
        with patch.object(
            grasp, "MonolithicCollisionPipeline", functools.partial(MonolithicCollisionPipeline, **options)
        ):
            super().__init__(viewer, args)
        (args.output / "view_configuration.json").write_text(
            json.dumps(
                {"options": options, "calibration_status": "REVALIDATION_REQUIRED", "performance_sample": False},
                indent=2,
            )
            + "\n"
        )
        self.stopped = False
        self.close_steps = round(self.case.fixture["stage_ends"][1] / self.case.fixture["dt"])

    def step(self):
        if self.stopped:
            return
        for _ in range(10):
            if self.case.step_count >= self.close_steps:
                break
            self.case.step()
            self.sim_time = self.case.step_count * self.case.fixture["dt"]
            if self.case.failure:
                break
        if self.case.failure or self.case.step_count >= self.close_steps:
            self.case.save(self.output)
            self.stopped = True
            print(
                f"Stopped at {self.sim_time:.3f} s: {self.case.failure or 'Close complete; grasp stages deferred'}",
                flush=True,
            )
            print("State saved; the viewer remains open for inspection.", flush=True)


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.case.save(args.output)
