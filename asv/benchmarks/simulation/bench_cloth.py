# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import warp as wp
from asv_runner.benchmarks.mark import SkipNotImplemented, skip_benchmark_if

wp.config.log_level = wp.LOG_WARNING

import newton.examples
from newton.examples.cloth.example_cloth_franka import Example as ExampleClothManipulation
from newton.examples.cloth.example_cloth_twist import Example as ExampleClothTwist
from newton.viewer import ViewerNull


class FastExampleClothManipulation:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        self.num_frames = 30
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothManipulation(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, args=None)

        wp.synchronize_device()


class FastExampleClothTwist:
    repeat = 5
    number = 1

    def setup(self):
        self.num_frames = 100
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = ExampleClothTwist(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        newton.examples.run(self.example, None)

        wp.synchronize_device()


class FastExampleClothPlasticFold:
    timeout = 300
    repeat = 3
    number = 1

    def setup(self):
        if not hasattr(newton, "ClothPlasticity"):
            raise SkipNotImplemented

        from newton.examples.cloth.example_cloth_plastic_fold import Example  # noqa: PLC0415

        self.num_frames = 60
        if hasattr(newton.examples, "default_args"):
            args = newton.examples.default_args()
        else:
            args = None
        self.example = Example(ViewerNull(num_frames=self.num_frames), args)

    @skip_benchmark_if(wp.get_cuda_device_count() == 0)
    def time_simulate(self):
        for _ in range(self.num_frames):
            self.example.simulate()

        wp.synchronize_device()


if __name__ == "__main__":
    import argparse

    from newton.utils import run_benchmark

    benchmark_list = {
        "FastExampleClothManipulation": FastExampleClothManipulation,
        "FastExampleClothPlasticFold": FastExampleClothPlasticFold,
        "FastExampleClothTwist": FastExampleClothTwist,
    }

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-b",
        "--bench",
        default=None,
        action="append",
        choices=benchmark_list.keys(),
        help="Run a specific benchmark; may be repeated to run multiple (e.g., --bench A --bench B).",
    )
    args = parser.parse_known_args()[0]

    if args.bench is None:
        benchmarks = benchmark_list.keys()
    else:
        benchmarks = args.bench

    for key in benchmarks:
        benchmark = benchmark_list[key]
        run_benchmark(benchmark)
