# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read changing graph parameters while preserving standalone scalar kernels."""

from typing import Any

import warp as wp


@wp.func
def scalar_value(value: wp.float32):
    return value


@wp.func
def scalar_value(value: wp.float64):
    return value


@wp.func
def scalar_value(value: wp.array[Any]):
    return value[0]
