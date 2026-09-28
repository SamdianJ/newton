# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the supported recorded Sharpa scene through its migrated entry point."""

from newton.examples.softbody.monolithic_sharpa_recorded import build_recorded, main, run, targets

__all__ = ["build_recorded", "main", "run", "targets"]

if __name__ == "__main__":
    raise SystemExit(main())
