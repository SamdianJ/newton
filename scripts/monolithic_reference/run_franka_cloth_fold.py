# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the shared Franka cloth recipe through the common evidence recorder."""

import argparse
import json

from .cloth_contact_run import add_arguments, execute
from .cloth_franka_scene import FrankaClothScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    method_config = json.loads(args.method_config.read_text()) if args.method_config else None
    execute(
        lambda: FrankaClothScene(
            args.recipe,
            device=args.device,
            precision=args.precision,
            use_cuda_graph=args.use_cuda_graph,
            method=args.method,
            method_config=method_config,
        ),
        args,
    )


if __name__ == "__main__":
    main()
