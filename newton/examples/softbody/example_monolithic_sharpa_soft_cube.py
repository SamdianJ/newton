# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental PR8H all-dynamic cube grasp; requires its own frozen contact envelope."""

from pathlib import Path

import newton.examples
from newton.examples.softbody.example_monolithic_sharpa_soft_ball import Example as BallExample


class Example(BallExample):
    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        for name in ("asset-dir", "contact-dir", "trajectory", "calibration"):
            parser.add_argument(f"--{name}", required=True)
        parser.add_argument("--cube", default="scripts/monolithic_reference/fixtures/soft_cube_40mm/cube_r2.npz")
        parser.add_argument("--output", type=Path, default=Path("output/monolithic-sharpa-soft-cube"))
        parser.add_argument("--position-offset-mm", type=float, nargs=3, default=(0, 0, 0))
        parser.add_argument("--pcg-mode", choices=("diagnostic", "production"), default="production")
        parser.add_argument("--friction-off", action="store_true")
        parser.add_argument("--disable-aabb", action="store_true")
        parser.set_defaults(experiment="grasp", num_frames=900, device="cuda:0")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)
