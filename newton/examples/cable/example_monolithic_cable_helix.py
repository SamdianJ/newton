# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental monolithic localized helical buckling under end rotation/shortening.

The naturally straight cable is isotropic for comparison with the Kirchhoff
localized solution. Twist and positions are solved in the same Newton step.
"""

import newton.examples
from newton.examples.cable.example_monolithic_cable_cantilever import Example as CantileverExample


class Example(CantileverExample):
    kind = "localized_helix"


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
